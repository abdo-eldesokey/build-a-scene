import random
import json
from typing import Iterator, List, Optional, Tuple, Union

import cv2
import trimesh
import numpy as np
import torch
import torch.nn.functional as F
from PIL.Image import Image

from src.utils.render import render_mesh, get_camera_pose
from src.utils.io import create_parent
from src.utils.segmentation import get_ca_object_mask, get_containing_box, find_largest_blob
from src.attention.dynamic_sattn import DynamicSelfAttention, MutualSelfAttentionControl
from src.attention.masactrl_utils import regiter_attention_editor_diffusers


# Set seed for reproducibility
def seed_everything(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class Box(trimesh.Trimesh):
    """
    A class for creating a 3D box mesh with a given size and origin.
    This is the main structuring element of the scene
    """

    def __init__(self, size=[1, 1, 1], origin=[0, 0, 0], prompt="", seed=None):
        mesh = self.create_mesh(size, origin)
        self.__dict__.update(mesh.__dict__)

        self.size = size  # Width, depth, height
        self.origin = origin
        self.prompt = prompt
        self.seed = seed

        self.shift = np.zeros_like(self.origin)  # Stores any applied translations

    @staticmethod
    def create_mesh(size, origin):
        mesh = trimesh.creation.box(extents=size)
        mesh.vertices += origin
        return mesh

    @classmethod
    def from_dict(cls, args):
        return cls(**args)

    @staticmethod
    def normalize_mesh(mesh: trimesh.Trimesh):
        mesh.vertices = mesh.vertices - mesh.vertices.mean(0)
        scale = np.linalg.norm(mesh.vertices, axis=1, ord=2).max()
        mesh.vertices = mesh.vertices / scale

    @staticmethod
    def translate(mesh: trimesh.Trimesh, T: Union[np.ndarray, List]):
        mesh.shift += T
        mesh.apply_translation(T)

    @staticmethod
    def rotate(mesh, angle, axis):
        which_axis = np.zeros(3)
        which_axis[axis] = 1
        R = trimesh.transformations.rotation_matrix(np.radians(angle), which_axis, point=mesh.origin)  # Rotate 45 degrees around the x-axis
        mesh.apply_transform(R)

    def move_left(self, shift):
        T = [-shift, 0, 0]
        self.translate(self, T)

    def move_right(self, shift):
        T = [shift, 0, 0]
        self.translate(self, T)

    def move_up(self, shift):
        T = [0, 0, shift]
        self.translate(self, T)

    def move_down(self, shift):
        T = [0, 0, -shift]
        self.translate(self, T)

    def zoom_in(self, shift):
        T = [0, -shift, 0]
        self.translate(self, T)

    def zoom_out(self, shift):
        T = [0, shift, 0]
        self.translate(self, T)

    def rotate_left(self, angle):
        self.rotate(self, -angle, 2)

    def rotate_right(self, angle):
        self.rotate(self, angle, 2)

    def rotate_up(self, angle):
        self.rotate(self, -angle, 0)

    def rotate_down(self, angle):
        self.rotate(self, angle, 0)

    def show(self):
        self.show()

    def reset(self):
        mesh = self.create_mesh(self.size, self.origin)
        self.__dict__.update(mesh.__dict__)
        self.shift = np.zeros_like(self.origin)  # Stores any applied translations

    def set_seed(self, seed):
        self.seed = seed


class DiffusionScene:
    def __init__(self, scene_size) -> None:
        self.scene_size = scene_size

        # Initialize the camera to be looking at an empty cubic scene
        self.camera_pose = get_camera_pose(self.scene_size)

        self.floor = None
        self.floor_offset = None  # An offset for shifting the floor up and down
        self.boxes = {}

        self.scene_dict = None

    @classmethod
    def from_json(cls, json_path):
        scene_dict = json.load(open(json_path, "r"))
        scene_size = scene_dict["scene_size"]
        rotation_angle = scene_dict["floor"]["camera_rotation"]
        scale_x, scale_y, offset = (
            scene_dict["floor"]["scale_x"],
            scene_dict["floor"]["scale_y"],
            scene_dict["floor"]["offset"],
        )
        scene = cls(scene_size=scene_size)
        scene.move_camera(rotation_angle=rotation_angle, rotation_axis=[1, 0, 0], translation=[0, 0, 0])
        scene.build_floor(scale_x=scale_x, scale_y=scale_y, floor_offset=offset)
        scene.scene_dict = scene_dict
        scene.pos_prompts = scene_dict.get("pos_prompts", "4k, high-res, realistic, ")
        scene.neg_prompts = scene_dict.get("neg_prompts", ["blurry, text, caption, lowquality, lowresolution, low-res, grainy, ugly"])
        return scene

    def add_box_from_json(self, box_id):
        if self.scene_dict is not None:
            box = self.scene_dict["boxes"][box_id]
            self.add_box(id=box_id, size=box["size"], origin=box["origin"], prompt=box["prompt"])
            seed = box["seed"]
            return seed
        else:
            raise RuntimeError("No JSON was loaded to the scene!")

    def move_camera(self, rotation_angle, rotation_axis, translation=[0, 0, 0]):
        self.camer_rotation_angle = rotation_angle
        rotation_matrix = trimesh.transformations.rotation_matrix(np.radians(self.camer_rotation_angle), rotation_axis)
        move_camera = np.eye(4)
        move_camera[:3, 3] = translation
        move_camera[:3, :3] = rotation_matrix[:3, :3]
        self.camera_pose = move_camera @ self.camera_pose

    def reset_camera(self):
        self.camera_pose = get_camera_pose(self.scene_size)

    def build_floor(self, scale_x, scale_y, floor_offset=0.0):
        self.floor_scale_x = scale_x
        self.floor_scale_y = scale_y
        self.floor_offset = floor_offset

        floor_vertices = np.array(
            [
                [-self.scene_size * scale_x, -self.scene_size * scale_y, self.floor_offset],
                [-self.scene_size * scale_x, self.scene_size * scale_y, self.floor_offset],
                [self.scene_size * scale_x, self.scene_size * scale_y, self.floor_offset],
                [self.scene_size * scale_x, -self.scene_size * scale_y, self.floor_offset],
            ],
            dtype=float,
        )
        floor_faces = np.array([[0, 1, 2], [0, 2, 3]])
        self.floor = trimesh.Trimesh(vertices=floor_vertices, faces=floor_faces)

    def get_floor(self) -> Optional[trimesh.Trimesh]:
        if self.floor:
            return self.floor
        else:
            raise RuntimeError("Floor is not created yet!")

    def add_box(self, id, size, origin=[0, 0, 0], prompt=""):
        # If the origin is not providied, set it to be exactly above the floor
        w, d, h = size
        if origin[-1] == 0:
            origin[-1] = self.floor_offset + h / 2
        self.boxes[id] = Box(size=size, origin=origin, prompt=prompt)

    def remove_box(self, id):
        del self.boxes[id]
        print(f"Deleted box '{id}'!")

    def box(self, id) -> Box:
        if id in self.boxes:
            return self.boxes[id]
        else:
            raise RuntimeError(f"Box with id: `{id}` does not exit!")

    def get_boxes(self, ids) -> List[trimesh.Trimesh]:
        return [self.box(id) for id in ids]

    def get_all_boxes(self) -> List[trimesh.Trimesh]:
        return self.get_boxes(self.boxes.keys())

    # Get attention and latent masks
    def get_attn_latent_masks(self, mask: np.ndarray, dilate_attn_mask=False, dilate_latent_mask=False):
        attn_mask = mask.copy().astype(np.uint8)
        if dilate_attn_mask:
            kernel = np.ones((5, 5), np.uint8)
            attn_mask = cv2.dilate(attn_mask.astype(np.uint8), kernel, iterations=2)
        attn_mask = torch.from_numpy(attn_mask).to(self.device, self.dtype)

        # Process latents mask
        latents_mask = mask.copy().astype(np.uint8)
        if dilate_latent_mask:
            kernel = np.ones((5, 5), np.uint8)
            latents_mask = cv2.erode(latents_mask, kernel, iterations=2)  # Latent mask is the inverse of attn_mask, so we use erode
        latents_mask = torch.from_numpy(latents_mask).to(self.device, self.dtype)
        latents_mask = F.interpolate((1 - latents_mask).to(self.device, self.dtype).unsqueeze(0).unsqueeze(0), (64, 64), mode="bilinear")
        return (
            attn_mask,
            latents_mask,
        )

    def get_box_masks(self, box_id, dilate_attn_mask=False, dilate_latent_mask=False):
        box_depth, p_image = render_mesh(self.box(box_id), self.camera_pose)
        mask = box_depth != 0
        attn_mask, latent_mask = self.get_attn_latent_masks(mask, dilate_attn_mask=dilate_attn_mask, dilate_latent_mask=dilate_latent_mask)
        return attn_mask, latent_mask, p_image

    def render(self):
        combined_mesh = trimesh.util.concatenate(self.get_all_boxes(), self.get_floor())
        depth, _ = render_mesh(combined_mesh, self.camera_pose)
        return depth

    def set_pipe(
        self,
        pipe,
        method,
        num_infer_steps,
        device,
        dtype,
        pos_prompts="4k, high-res, realistic, ",
        neg_prompts=["blurry, text, caption, lowquality, lowresolution, low-res, grainy, ugly"],
    ):
        self.pipe = pipe
        self.method = method
        self.device = device
        self.dtype = dtype
        self.generator = torch.Generator(device=device)
        self.num_infer_steps = num_infer_steps
        self.pos_prompts = pos_prompts
        self.neg_prompt = neg_prompts
        print(f"==> Using '{self.method}' pipeline! ")

    def generate(
        self,
        depth_cond: List[Image],
        prompts: List[str],
        seed: int,
        attn_mask: Optional[torch.Tensor] = None,
        latent_mask: Optional[torch.Tensor] = None,
        blend_latents_timesteps: range = range(0),
        bg_latents_list: Optional[List[torch.Tensor]] = None,
        fg_latents_list: Optional[List[torch.Tensor]] = None,
        mask_dsa_k: bool = True,
        mask_dsa_v: bool = False,
        use_adain: bool = False,
        blend_sattn: bool = False,
    ) -> Tuple[List[Image], List[torch.Tensor]]:
        """Build-a-scene image generation function

        Args:
            depth_cond (List[Image]): A list of depth condition images
            prompts (List[str]): A list of prompts
            seed (int): generator seed
            attn_mask (Optional[torch.Tensor], optional): Mask for Dynnamic Self-Attention (DSA). $\\mathcal{M}_\\text{FG}$ in the paper." Defaults to None.
            latent_mask (Optional[torch.Tensor], optional): Mask for Latent Blending. $\\mathcal{M}_\\text{BG}$ in the paper. Defaults to None.
            blend_latents_timesteps (range, optional): _description_. Defaults to range(0).
            bg_latents_list (Optional[List[torch.Tensor]], optional): List of latents for all timesteps for the previous generation stage. Defaults to None.
            fg_latents_list (Optional[List[torch.Tensor]], optional): List of latents for all timesteps for the inverted warped image. Defaults to None.
            mask_dsa_k (bool, optional): Either to mask the keys in DSA. Defaults to True.
            mask_dsa_v (bool, optional): Either to mask the values in DSA. Defaults to False.
            use_adain (bool, optional): Either to use AdaIN after latent blending. Defaults to False.
            blend_sattn (bool, optional): Blend features after self-attention for better consistency. Defaults to False.
        Raises:
            RuntimeError: _description_

        Returns:
            Tuple[List[Image], List[torch.Tensor]]: Return the generated images and the latents list for this generation stage
        """

        # Store prompt and seed of empty scene
        if len(self.boxes) == 0:
            self.empty_prompt = prompts[0]
            self.empty_seed = seed

        seed_everything(seed)
        self.generator.manual_seed(seed)

        ### Extended Masked Attention
        STEP = 10  # 4
        LAYER = 0  # 10
        if self.method == "lc":
            self.editor = MutualSelfAttentionControl(STEP, LAYER)  # Baseline
        elif self.method == "bas":
            total_steps = int(self.num_infer_steps)
            self.editor = DynamicSelfAttention(
                STEP, LAYER, mask=attn_mask, total_steps=total_steps, mask_dsa_k=mask_dsa_k, mask_dsa_v=mask_dsa_v, blend_sattn=blend_sattn
            )
        else:
            raise RuntimeError("Undefined method ", self.method)

        regiter_attention_editor_diffusers(self.pipe, self.editor)

        ### Depth condition
        batch_size = len(depth_cond)

        ### Prompts
        negative_prompt = self.neg_prompt * batch_size

        ### Latents
        start_code = torch.randn([1, 4, 64, 64], generator=self.generator, device=self.device).to(self.dtype)
        latents = start_code.repeat(batch_size, 1, 1, 1)

        if isinstance(prompts, List):
            prompts = [self.pos_prompts + p for p in prompts]

        out, latents_list = self.pipe(
            prompts,
            depth_cond,
            height=512,
            width=512,
            guidance_scale=7.5,
            negative_prompt=negative_prompt,
            num_inference_steps=self.num_infer_steps,
            latents=latents,
            controlnet_conditioning_scale=1.0,
            return_latents=True,
            latents_mask=latent_mask,
            blend_latents_timesteps=blend_latents_timesteps,
            bg_latents_list=bg_latents_list,
            fg_latents_list=fg_latents_list,
            use_adain=use_adain,
        )

        return out.images, latents_list

    def segment_object(self, sam_processor, sam_model, editor, image, prompt, token_idx=None):
        # Get a coarse segmentation using from Cross-Attention maps
        object_cn_mask = get_ca_object_mask(self.pos_prompts + prompt, self.pipe, editor, threshold=75, token_idx=token_idx)

        object_cn_mask = find_largest_blob(object_cn_mask)

        # Get a bounding from the segmentation map to use as input for SAM
        sam_box = get_containing_box(object_cn_mask, padding=(10, 10))

        inputs = sam_processor(image, input_boxes=[[[sam_box]]], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = sam_model(**inputs)

        masks = sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
        )
        object_sam_mask = masks[0][0, -1].cpu().numpy()

        output = {"cn_mask": object_cn_mask, "sam_box": sam_box, "sam_mask": object_sam_mask}
        return output

    def save_scene_dict(self, json_path):

        create_parent(json_path)

        scene_dict = {
            "empty_prompt": self.empty_prompt,
            "empty_seed": self.empty_seed,
            "scene_size": self.scene_size,
            "pos_prompts": self.pos_prompts,
            "neg_prompts": self.neg_prompt,
            "floor": {
                "scale_x": self.floor_scale_x,
                "scale_y": self.floor_scale_y,
                "offset": self.floor_offset,
                "camera_rotation": self.camer_rotation_angle,
            },
            "boxes": {k: {"size": v.size, "origin": v.origin, "seed": v.seed, "prompt": v.prompt} for k, v in self.boxes.items()},
        }
        with open(json_path, "w") as f:
            json.dump(scene_dict, f, indent=2)


def get_box_renders(
    scene: DiffusionScene, box_id, seed, dilate_attn_mask=False, dilate_latent_mask=False
) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor, str, np.ndarray]:
    # Render the new scene
    depth_b = scene.render()
    attn_mask_b, latent_mask_b, p_image_b = scene.get_box_masks(
        box_id=box_id, dilate_attn_mask=dilate_attn_mask, dilate_latent_mask=dilate_latent_mask
    )
    scene.box(box_id).set_seed(seed)
    prompt_b = scene.box(box_id).prompt
    # imshowp(depth_b)
    return depth_b, attn_mask_b, latent_mask_b, prompt_b, p_image_b
