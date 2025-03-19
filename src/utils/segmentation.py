from typing import Tuple
import torch
import numpy as np
import cv2
from PIL.Image import Image
from src.utils.render import find_correspondence_bw_images


def get_containing_box(mask, padding=[0, 0]):
    """
    Get the bounding box of a mask
    """
    if isinstance(mask, torch.Tensor):
        xx, yy = torch.where(mask != 0)
    elif isinstance(mask, np.ndarray):
        xx, yy = np.where(mask != 0)
    else:
        raise ValueError("mask should be either a torch.Tensor or a np.ndarray")

    x1 = xx.min().item()
    x2 = xx.max().item()
    y1 = yy.min().item()
    y2 = yy.max().item()
    pad_y = padding[0]
    pad_x = padding[1]
    return np.asarray([y1 - pad_y, x1 - pad_x, y2 + pad_y, x2 + pad_x])


def get_ca_object_mask(prompt, pipe, editor, threshold=50, dilate=True, token_idx=None):
    """
    Get the cross-attention mask of a specific word in the prompt (usually the object)
    """
    # We assume that the object is in the last word of the prompt
    if token_idx is None:  # Take the last word
        token_idx = len(prompt.split(" ")) - 1
    obj_token_idx = get_word_inds(prompt, token_idx, pipe.tokenizer)  ### Get prompt tokens
    new_object_mask = editor.aggregate_cross_attn_map(obj_token_idx)
    mask = (new_object_mask[-1, ..., 0].cpu().numpy() * 255).astype(np.uint8)
    mask = cv2.resize(mask, (512, 512))
    ret3, mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
    if dilate:
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel)
    return mask


def get_word_inds(text: str, word_place: int, tokenizer):
    """
    Get the indices of a word in the prompt after tokenization
    """
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


def find_largest_blob(binary_image):
    """
    Find the largest blob in a binary image
    """
    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a mask for the largest contour
    mask = np.zeros_like(binary_image)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # Extract the largest blob
    largest_blob = cv2.bitwise_and(binary_image, mask)

    return largest_blob


def scale_object_in_image(
    source_image: Image,
    object_mask: np.ndarray,
    target_image,
    curr_p_image,
    new_p_image,
    scale_factor=0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    # Extract object from the original image
    x1, y1, x2, y2, _, _ = find_correspondence_bw_images(curr_p_image, new_p_image, thresh=0.05)

    # Find indices of coordinates on the object mask
    idx = [i for i, _ in enumerate(x1) if object_mask[x1[i], y1[i]]]

    # Filter the coordinates
    x1_, y1_, x2_, y2_ = x1[idx], y1[idx], x2[idx], y2[idx]

    # Extract object from the original image
    source_img_np = np.array(source_image)
    object_image = np.ones_like(source_img_np) * np.nan
    object_image[object_mask] = source_img_np[object_mask]

    b1_xmin, b1_ymin, b1_xmax, b1_ymax = x1_.min(), y1_.min(), x1_.max(), y1_.max()
    b2_xmin, b2_ymin, b2_xmax, b2_ymax = x2_.min(), y2_.min(), x2_.max(), y2_.max()

    scale_x = abs(b2_xmax - b2_xmin) / abs(b1_xmax - b1_xmin)
    scale_y = abs(b2_ymax - b2_ymin) / abs(b1_ymax - b1_ymin)
    if scale_factor != 0:
        pad_x = scale_x * scale_factor
        scale_x -= pad_x
        pad_y = scale_y * scale_factor
        scale_y -= pad_y

    # # Extract the object from the image using the bounding box
    mask_bb = get_containing_box(object_mask)
    object_roi = object_image[mask_bb[1] : mask_bb[3], mask_bb[0] : mask_bb[2]]

    # # Enlarge the object ROI
    enlarged_object_roi = cv2.resize(object_roi, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

    # # Replace the enlarged object ROI back into the original image
    enlarged_image = np.ones_like(object_image) * np.nan
    try:
        new_H = enlarged_object_roi.shape[0]
        new_W = enlarged_object_roi.shape[1]
        new_xmin = int(b2_xmin + new_H * (scale_factor))
        new_xmax = int(b2_xmin + new_H * (1 + scale_factor))
        new_ymin = int(b2_ymin + new_W * (scale_factor))
        new_ymax = int(b2_ymin + new_W * (1 + scale_factor))
        enlarged_image[new_xmin:new_xmax, new_ymin:new_ymax] = enlarged_object_roi
    except:
        raise RuntimeError("Error in warping the object")

    new_mask = ~np.isnan(enlarged_image)
    # new_mask = enlarged_image != -1
    new_img = np.array(target_image).copy()
    new_img[new_mask] = enlarged_image[new_mask]
    return new_img, enlarged_image
