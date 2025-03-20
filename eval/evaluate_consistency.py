from pathlib import Path
from copy import deepcopy
import json
from PIL import Image
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from transformers import SamModel, SamProcessor
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler


from src.pipeline.bas_pipeline import ControlNetX, StableDiffusionControlNetPipeline, attach_loaders_mixin
from eval.metrics import ClipSimilarity, compute_fg_cons_metrics, compute_bg_cons_metrics
from src.utils.scene import DiffusionScene
from src.utils.segmentation import get_containing_box, scale_object_in_image
from eval.layouts_utils import cons_jsons_path


DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
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

NUM_INFER_STEPS = 20
SAVE_IMGS = False

# Create directories to save generations
output_path = Path("workspace/build-a-scene/consistency")
metrics_path = output_path / "metrics"
if not metrics_path.exists():
    metrics_path.mkdir(parents=True)

images_path = output_path / "images"
if SAVE_IMGS:
    if not images_path.exists():
        images_path.mkdir(parents=True)

### Processing loop
for idx, json_fname in enumerate(sorted(cons_jsons_path.glob("*.json"), reverse=False)):

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
    scene.build_floor(scale_x=2, scale_y=2, floor_offset=-scene_size)
    extra_pos_prompts = ""  # Optionally add extra positive prompts
    neg_prompts = ["blurry, text, caption, lowquality, lowresolution, low-res, grainy, ugly"]
    scene.set_pipe(pipe, "bas", NUM_INFER_STEPS, DEVICE, DTYPE, extra_pos_prompts, neg_prompts)

    # Render empty scene
    prompt_empty = layout_dict["scene"]
    depth_empty = scene.render()
    depth_cond_empty = [Image.fromarray(d) for d in [depth_empty]]

    b1_imgs = []
    b1m_imgs = []
    valid_seeds = []
    seeds = sorted(layout_dict["seeds"])
    # seeds = [layout_dict["seeds"][0]]
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
            mask_dsa_v=True,
            blend_sattn=True,
        )
        b1_editor = deepcopy(scene.editor)
        if SAVE_IMGS:
            out_b1[-1].save(f"{imgs_path}/{json_fname.stem}_{seed}_1.png")

        #### Move the box
        action = layout_dict[b1_id]["action"]
        action_scale = layout_dict[b1_id]["action_scale"]
        ref_box = scene.box(b1_id)
        ref_box.reset()
        action_fn = getattr(ref_box, action)
        action_fn(action_scale)

        depth_b1m = scene.render()
        b1m_mask, latent_mask_b1m, p_image_b1m = scene.get_box_masks(box_id=b1_id)
        p_image_b1m = p_image_b1m - scene.box(b1_id).shift[None, None]

        ### Segment the object before moving
        ref_image = out_b1[-1]
        seg_out = scene.segment_object(sam_processor, sam_model, b1_editor, ref_image, b1_prompt)

        # Scale the object
        try:
            warped_img, scaled_object_b1 = scale_object_in_image(
                ref_image, object_mask=seg_out["sam_mask"], target_image=out_empty[0], curr_p_image=p_image_b1, new_p_image=p_image_b1m
            )
        except:
            continue

        if warped_img is None:  # Scaling object failed
            continue

        valid_seeds.append(seed)

        # Invert the warped image
        lat, interm_lats = pipe.invert_depth(
            Image.fromarray(warped_img),
            depth=Image.fromarray(depth_b1m),
            prompt=b1_prompt,
            num_inference_steps=NUM_INFER_STEPS,
            device=DEVICE,
        )

        ### Depth condition
        depth_cond = [Image.fromarray(d) for d in [depth_empty, depth_b1m]]
        new_mask, new_latent_mask = scene.get_attn_latent_masks((scaled_object_b1 > 0)[..., 0])

        out_b1m, latents_list_b1m = scene.generate(
            prompts=prompts,
            depth_cond=depth_cond,
            seed=seed,
            attn_mask=new_mask,
            latent_mask=new_latent_mask,
            bg_latents_list=latents_list_empty,
            blend_latents_timesteps=range(end_latent_blending),
            fg_latents_list=interm_lats,
            mask_dsa_v=True,
            blend_sattn=True,
        )

        if SAVE_IMGS:
            out_b1m[-1].save(f"{imgs_path}/{json_fname.stem}_{seed}_2.png")

        b1_imgs.append(out_b1[-1])
        b1m_imgs.append(out_b1m[-1])

        # break

    ##### Compute metrics
    metrics = {}
    metrics["prompt_empty"] = prompt_empty
    metrics["prompt_b1"] = b1_prompt

    b1_bb = get_containing_box(b1_mask)
    b1m_bb = get_containing_box(b1m_mask)

    # Foreground metrics
    if b1_imgs:
        metrics["fg"] = compute_fg_cons_metrics(b1_imgs, b1m_imgs, b1_bb, b1m_bb, valid_seeds, clip_sim)
        metrics["bg"] = compute_bg_cons_metrics(b1_imgs, b1m_imgs, b1_mask, b1m_mask, valid_seeds, clip_sim)
    else:
        continue

    ## Background metrics

    file = open(f"{metrics_path}/{fname}.json", "w")
    json.dump(metrics, file, indent=2)
    file.close()
