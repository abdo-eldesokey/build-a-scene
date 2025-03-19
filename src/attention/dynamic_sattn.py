import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange

from src.attention.masactrl_utils import AttentionBase


# Copied from MasaCtrl
class MutualSelfAttentionControl(AttentionBase):
    MODEL_TYPE = {"SD": 16, "SDXL": 70}

    def __init__(self, start_step=4, start_layer=10, layer_idx=None, step_idx=None, total_steps=50, model_type="SD"):
        """
        Mutual self-attention control for Stable-Diffusion model
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            model_type: the model type, SD or SDXL
        """
        super().__init__()
        self.total_steps = total_steps
        self.total_layers = self.MODEL_TYPE.get(model_type, 16)
        self.start_step = start_step
        self.start_layer = start_layer
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, self.total_layers))
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, total_steps))
        # print("MasaCtrl at denoising steps: ", self.step_idx)
        # print("MasaCtrl at U-Net layers: ", self.layer_idx)

    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Performing attention for a batch of queries, keys, and values
        """
        b = q.shape[0] // num_heads
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        attn = sim.softmax(-1)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "h (b n) d -> b n (h d)", b=b)
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        qu, qc = q.chunk(2)
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)
        attnu, attnc = attn.chunk(2)

        out_u = self.attn_batch(qu, ku[:num_heads], vu[:num_heads], sim[:num_heads], attnu, is_cross, place_in_unet, num_heads, **kwargs)
        out_c = self.attn_batch(qc, kc[:num_heads], vc[:num_heads], sim[:num_heads], attnc, is_cross, place_in_unet, num_heads, **kwargs)

        out = torch.cat([out_u, out_c], dim=0)

        return out


class DynamicSelfAttention(MutualSelfAttentionControl):
    def __init__(
        self,
        start_step=4,
        start_layer=10,
        layer_idx=None,
        step_idx=None,
        total_steps=50,
        model_type="SD",
        mask=None,
        mask_dsa_k: bool = False,
        mask_dsa_v: bool = False,
        blend_sattn: bool = False,
    ):
        """
        Dynamic Self-Attention Control
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            model_type: the model type, SD or SDXL
            mask_dsa_k (bool, optional): Either to mask the keys in DSA. Defaults to True.
            mask_dsa_v (bool, optional): Either to mask the values in DSA. Defaults to False.
            use_adain (bool, optional): Either to use AdaIN after latent blending. Defaults to False.
            blend_sattn (bool, optional): Blend features after self-attention for better consistency. Defaults to False.
        """
        super().__init__(start_step, start_layer, layer_idx, step_idx, total_steps, model_type)
        self.mask = mask
        self.mask_dsa_k = mask_dsa_k
        self.mask_dsa_v = mask_dsa_v
        self.blend_sattn = blend_sattn

        # Cross attention
        self.cross_attns = []

    def aggregate_cross_attn_map(self, idx):
        """Aggregate cross attention maps to be use later in segmenting objects"""
        attn_map = torch.stack(self.cross_attns, dim=1).mean(1)  # (B, N, dim)
        B = attn_map.shape[0]
        res = int(np.sqrt(attn_map.shape[-2]))
        attn_map = attn_map.reshape(-1, res, res, attn_map.shape[-1])
        image = attn_map[..., idx]
        if isinstance(idx, list):
            image = image.sum(-1)
        image_min = image.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
        image_max = image.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        image = (image - image_min) / (image_max - image_min)
        return image

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        # Store cross-attention
        if is_cross:
            if attn.shape[1] == 16 * 16:
                self.cross_attns.append(attn.reshape(-1, num_heads, *attn.shape[-2:]).mean(1))

        if q.shape[0] // num_heads <= 2 or is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        qu_s, qu_t, qc_s, qc_t = q.chunk(4)
        ku_s, ku_t, kc_s, kc_t = k.chunk(4)
        vu_s, vu_t, vc_s, vc_t = v.chunk(4)
        attnu_s, attnu_t, attnc_s, attnc_t = attn.chunk(4)

        # source image branch
        out_u_s = AttentionBase.forward(self, qu_s, ku_s, vu_s, sim, attnu_s, is_cross, place_in_unet, num_heads, **kwargs)
        out_c_s = AttentionBase.forward(self, qc_s, kc_s, vc_s, sim, attnc_s, is_cross, place_in_unet, num_heads, **kwargs)

        H = W = int(np.sqrt(q.shape[1]))
        if self.mask is not None:
            mask = F.interpolate(self.mask.unsqueeze(0).unsqueeze(0), (H, W), mode="nearest")
        else:
            mask = torch.ones((1, 1, H, W))

        mask = mask.reshape(-1, 1)  # (hw, 1)

        # DSA Starts Here
        kk_u = torch.cat([ku_s, ku_t * mask]) if self.mask_dsa_k else torch.cat([ku_s, ku_t])
        vv_u = torch.cat([vu_s, vu_t * mask]) if self.mask_dsa_v else torch.cat([vu_s, vu_t])
        out_u_t = self.attn_batch(
            qu_t,
            kk_u,
            vv_u,
            None,
            None,
            is_cross,
            place_in_unet,
            num_heads,
            **kwargs,
        )

        kk_c = torch.cat([kc_s, kc_t * mask]) if self.mask_dsa_k else torch.cat([kc_s, kc_t])
        vv_c = torch.cat([vc_s, vc_t * mask]) if self.mask_dsa_v else torch.cat([vc_s, vc_t])

        out_c_t = self.attn_batch(
            qc_t,
            kk_c,
            vv_c,
            None,
            None,
            is_cross,
            place_in_unet,
            num_heads,
            **kwargs,
        )

        out = torch.cat([out_u_s, out_u_t, out_c_s, out_c_t], dim=0)

        return out
