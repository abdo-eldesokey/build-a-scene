from typing import Optional, Union
from IPython.core.display_functions import display

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch


def to_HWC(img):
    if isinstance(img, np.ndarray) and img.shape[0] in [1, 3]:
        return img.transpose((1, 2, 0))
    elif isinstance(img, torch.Tensor) and img.shape[0] in [1, 3]:
        return img.permute((1, 2, 0)).detach().cpu().numpy()
    elif isinstance(img, torch.Tensor) and len(img.shape) == 2:
        return img.detach().cpu().numpy()
    elif isinstance(img, Image.Image):
        return np.asarray(img)
    else:
        return img


def imshow(
    img: Union[torch.Tensor, np.ndarray, Image.Image],
    title: Optional[str] = None,
    colorbar=False,
    cmap: Optional[str] = None,
    alpha: float = 1.0,
    show: float = False,
):
    # assert len(img.shape) < 4, "Please provide a max of 3D input"
    img = to_HWC(img)
    plt.imshow(img, cmap=cmap, alpha=alpha)
    plt.axis("off")
    if colorbar:
        plt.colorbar()
    if title is not None:
        plt.title(title)
    if show:
        plt.show()


def imshowp(img):
    if isinstance(img, np.ndarray):
        display(Image.fromarray(img.astype(np.uint8)))
    else:
        display(img)


def visualize_box_renders(depth, attn_mask, latent_mask):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(depth, cmap="gray")
    plt.axis("off")
    plt.title("Depth Map")
    plt.subplot(1, 3, 2)
    plt.imshow(attn_mask.detach().cpu())
    plt.axis("off")
    plt.title("Attention Mask")
    plt.subplot(1, 3, 3)
    plt.imshow(latent_mask[0, 0].detach().cpu())
    plt.axis("off")
    plt.title("Latent Blending Mask")
    plt.show()
