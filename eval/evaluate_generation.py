from pathlib import Path
from copy import deepcopy
import json
from PIL import Image
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from transformers import SamModel, SamProcessor
from ultralytics import YOLO
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler


from src.pipeline.bas_pipeline import ControlNetX, StableDiffusionControlNetPipeline, attach_loaders_mixin
from eval.metrics import ClipSimilarity, compute_yolo_metrics

from src.utils.scene import DiffusionScene
from src.utils.segmentation import get_containing_box
from eval.layouts_utils import gen_jsons_path, get_relations

DEVICE = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
DTYPE = torch.float16

##### Build-A-Scene
print("==> Loading Build-A-Scene models")
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
controlnet = ControlNetX.from_pretrained("lllyasviel/control_v11f1p_sd15_depth").to(DEVICE, DTYPE)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    controlnet=controlnet,
    requires_safety_checker=False,
    safety_checker=None,
    torch_dtype=DTYPE,
    scheduler=scheduler,
).to(DEVICE)
pipe.controlnet = attach_loaders_mixin(pipe.controlnet)
pipe.controlnet.load_attn_procs("shariqfarooq/loose-control-3dbox")

##### SAM
print("==> Loading SAM models")
sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(DEVICE)
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

#### For metrics
clip_sim = ClipSimilarity()
yolo_model = YOLO("yolov8x.pt")


NUM_INFER_STEPS = 20
SAVE_IMGS = False

# Create directories to save generations
output_path = Path("workspace/build-a-scene/generation")
metrics_path = output_path / "metrics"
if not metrics_path.exists():
    metrics_path.mkdir(parents=True)

images_path = output_path / "images"
if SAVE_IMGS:
    if not images_path.exists():
        images_path.mkdir(parents=True)


### Processing loop
for json_fname in sorted(gen_jsons_path.glob("*.json"), reverse=False):
    file = open(json_fname, "r")
    layout_dict = json.load(file)

    fname = json_fname.stem
    if SAVE_IMGS and not images_path.exists():
        imgs_path = images_path / fname
        imgs_path.mkdir()

    print(f"Processing layout {json_fname}")
    # Build the scene
    scene_size = layout_dict["scene_size"]
    print("==> Scene size: ", scene_size)
    scene = DiffusionScene(scene_size=scene_size)
    camera_angle = layout_dict["camera_angle"]
    print("==> Camera Angle: ", camera_angle)
    scene.move_camera(rotation_angle=camera_angle, rotation_axis=[1, 0, 0], translation=[0, 0, 0])
    scene.build_floor(scale_x=2, scale_y=4, floor_offset=-scene_size)
    scene.set_pipe(pipe, "bas", NUM_INFER_STEPS, DEVICE, DTYPE)

    # Render empty scene
    prompt_empty = layout_dict["scene"]
    depth_empty = scene.render()
    depth_cond_empty = [Image.fromarray(d) for d in [depth_empty]]

    out_imgs = []
    seeds = sorted(layout_dict["seeds"])
    for seed in seeds:

        out_empty, latents_list_empty = scene.generate(prompts=[prompt_empty], depth_cond=depth_cond_empty, seed=seed)
        if SAVE_IMGS:
            out_empty[-1].save(f"{imgs_path}/{fname}_{seed}_0.png")

        # Box 1
        b1_id = "box_1"
        b1_size = layout_dict[b1_id]["size"]
        b1_origin = layout_dict[b1_id]["origin"]
        b1_prompt = layout_dict[b1_id]["prompt"]
        scene.add_box(id=b1_id, size=b1_size, origin=b1_origin, prompt=b1_prompt)
        b1_mask, latent_mask_b1, p_image_b1 = scene.get_box_masks(box_id=b1_id)

        depth_b1 = scene.render()
        depth_cond_b1 = [Image.fromarray(d) for d in [depth_empty, depth_b1]]
        prompts = [prompt_empty, b1_prompt]

        latent_blending_ratio = 0.9
        end_latent_blending = int(NUM_INFER_STEPS * latent_blending_ratio)
        out_b1, latents_list_b1 = scene.generate(
            prompts=prompts,
            depth_cond=depth_cond_b1,
            seed=seed,
            attn_mask=b1_mask,
            latent_mask=latent_mask_b1,
            bg_latents_list=latents_list_empty,
            blend_latents_timesteps=range(end_latent_blending),
        )
        b1_editor = deepcopy(scene.editor)
        if SAVE_IMGS:
            out_b1[-1].save(f"{imgs_path}/{json_fname.stem}_{seed}_1.png")

        # Box 2
        b2_id = "box_2"
        b2_size = layout_dict[b2_id]["size"]
        b2_origin = layout_dict[b2_id]["origin"]
        b2_prompt = layout_dict[b2_id]["prompt"]
        b2_relation = layout_dict[b2_id]["relation"]
        scene.add_box(id=b2_id, size=b2_size, origin=b2_origin, prompt=b2_prompt)
        b2_mask, latent_mask_b2, p_image_b2 = scene.get_box_masks(box_id=b2_id)

        depth_b2 = scene.render()
        depth_cond_b2 = [Image.fromarray(d) for d in [depth_b1, depth_b2]]
        prompts = [b1_prompt, b2_prompt]
        out_b2, latents_list_b2 = scene.generate(
            prompts=prompts,
            depth_cond=depth_cond_b2,
            seed=seed,
            attn_mask=b2_mask,
            latent_mask=latent_mask_b2,
            bg_latents_list=latents_list_b1,
            blend_latents_timesteps=range(end_latent_blending),
        )

        b2_editor = deepcopy(scene.editor)
        if SAVE_IMGS:
            out_b2[-1].save(f"{imgs_path}/{fname}_{seed}_2.png")

        out_imgs.append(out_b2[-1])

    b1_bb = get_containing_box(b1_mask)
    b2_bb = get_containing_box(b2_mask)

    ##### Compute metrics
    b1_pos, b2_pos = get_relations(b2_relation)
    prompt_empty = layout_dict["scene"] + f" with {b1_prompt} on the {b1_pos} and {b2_prompt} on the {b2_pos} "

    metrics = {}
    ## Clip metrics
    clip = clip_sim.t2i_similarity(prompt_empty, out_imgs)
    metrics["fname"] = json_fname.stem
    metrics["clip"] = {
        "prompt": prompt_empty,
        "per_seed": {s: c.item() for s, c in zip(seeds, clip)},
        "clip_avg": clip.mean().item(),
        "clip_max": clip.max().item(),
    }
    ## Yolo metrics
    masks = {"b1": b1_mask, "b2": b2_mask}
    metrics["yolo_metrics"] = compute_yolo_metrics(
        yolo_model(out_imgs), seeds, masks, {"b1": b1_prompt, "b2": b2_prompt}, list(masks.keys())
    )

    file = open(f"{metrics_path}/{fname}.json", "w")
    json.dump(metrics, file, indent=2)
    file.close()
