import torch
import clip
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity

from eval.layouts_utils import YOLO8_LABELS
from src.utils.segmentation import get_containing_box


class ClipSimilarity:
    def __init__(self):
        self.device = self._get_device()
        self.model, self.preprocess = self._initialize_model("ViT-B/32", self.device)
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=1)

    def _get_device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _initialize_model(self, model_name="ViT-B/32", device="cpu"):
        model, preprocess = clip.load(model_name, device=device)
        return model, preprocess

    def _embed_images(self, images):
        if isinstance(images, Image.Image):
            images = [images]
        preprocessed_image = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        image_embeddings = self.model.encode_image(preprocessed_image)
        return image_embeddings

    def _embed_text(self, text):
        text_token = clip.tokenize([text]).to(self.device)
        text_embed = self.model.encode_text(text_token)
        return text_embed

    def temporal_similarity(self, images):
        image_embeds_1 = self._embed_images(images)
        image_embeds_2 = torch.roll(image_embeds_1.clone(), -1, 0)
        # print(image_embeds_1.shape, image_embeds_2.shape)
        raw_similarity = self.cosine_similarity(image_embeds_1, image_embeds_2)
        # print(raw_similarity.shape)
        return raw_similarity.mean()

    @torch.no_grad()
    def t2i_similarity(self, text, images):
        text_embeds = self._embed_text(text)
        image_embeds = self._embed_images(images)
        raw_similarity = self.cosine_similarity(text_embeds, image_embeds)
        return raw_similarity

    @torch.no_grad()
    def i2i_similarity(self, image_1, image_2):
        image_1_embeds = self._embed_images(image_1)
        image_2_embeds = self._embed_images(image_2)
        raw_similarity = self.cosine_similarity(image_1_embeds, image_2_embeds)
        return raw_similarity


def calculate_iou(bbox1, bbox2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Args:
    - bbox1: Tuple (x_min, y_min, x_max, y_max) defining the first bounding box.
    - bbox2: Tuple (x_min, y_min, x_max, y_max) defining the second bounding box.

    Returns:
    - IoU: Intersection over Union (IoU) score.
    """
    # Extract coordinates of the bounding boxes
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2

    # Calculate the intersection area
    x_overlap = max(0, min(x2, x4) - max(x1, x3))
    y_overlap = max(0, min(y2, y4) - max(y1, y3))
    intersection_area = x_overlap * y_overlap

    # Calculate the area of each bounding box
    area_bbox1 = (x2 - x1) * (y2 - y1)
    area_bbox2 = (x4 - x3) * (y4 - y3)

    # Calculate the union area
    union_area = area_bbox1 + area_bbox2 - intersection_area

    # Calculate the Intersection over Union (IoU)
    iou = intersection_area / union_area if union_area > 0 else 0

    return iou


def compute_yolo_metrics(results, seeds, masks, prompts, boxes_ids):
    out_dict = {}

    classes = results[0].names
    moa = []
    miou = []
    out_dict["per_seed"] = {}
    for seed, result in zip(seeds, results):
        output = {}
        boxes = result.boxes
        bbs = boxes.xyxy
        cls_idx = boxes.cls
        cls_conf = boxes.conf
        valid_boxes = [idx for idx in range(cls_idx.shape[0]) if cls_conf[idx] > 0.7]
        best_cls_idx = [cls_idx[idx].int().item() for idx in valid_boxes]
        best_cls_name = [classes[idx] for idx in best_cls_idx]
        best_cls_bb = [bbs[idx].cpu().numpy() for idx in valid_boxes]

        mean_oa = 0
        mean_iou = []
        num_correct = 0
        for bid in boxes_ids:
            box_prompt = prompts[bid]
            box_mask = masks[bid]
            box_label = YOLO8_LABELS[box_prompt]

            output[f"{bid}_gt_label"] = box_label
            output[f"{bid}_gt_cls"] = box_prompt

            box_bb = get_containing_box(box_mask)

            box_iou = [calculate_iou(box_bb, pred_bb) for pred_bb in best_cls_bb]
            if not box_iou:
                continue
            box_match = np.array(box_iou).argmax()
            box_best_miou = box_iou[box_match]
            if box_best_miou < 0.1:
                continue
            box_pred_label = best_cls_idx[box_match]
            box_pred_cls = best_cls_name[box_match]
            output[f"{bid}_pred_label"] = box_pred_label
            output[f"{bid}_pred_cls"] = box_pred_cls
            output[f"{bid}_gt_bb"] = box_bb.tolist()
            output[f"{bid}_pred_bb"] = best_cls_bb[box_match].astype(int).tolist()
            output[f"{bid}_iou"] = box_best_miou

            mean_oa += output[f"{bid}_pred_label"] == output[f"{bid}_gt_label"]
            mean_iou.append(output[f"{bid}_iou"])
            num_correct += 1

        output["mean_oa"] = mean_oa / 2
        output["mean_iou"] = np.array(mean_iou).mean().item() if mean_iou else 0

        out_dict["per_seed"][seed] = output
        moa.append(output["mean_oa"])
        miou.append(output["mean_iou"])
    miou = np.array(miou)
    out_dict["mean_oa"] = np.array(moa).mean()
    out_dict["max_oa"] = np.array(moa).max()
    out_dict["mean_iou"] = miou[miou != 0].mean()

    return out_dict


def calculate_psnr(img1, img2):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
    - img1: First image (numpy array).
    - img2: Second image (numpy array).

    Returns:
    - PSNR: Peak Signal-to-Noise Ratio (PSNR) score.
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def compute_fg_cons_metrics(b1_imgs, b1m_imgs, b1_bb, b1m_bb, seeds, clip_sim):
    metrics = {}
    b1_cropped = []
    for img in b1_imgs:
        b1_cropped.append(img.crop(b1_bb))

    b1m_cropped = []
    for idx, img in enumerate(b1m_imgs):
        b1m_cropped.append(img.crop(b1m_bb).resize(b1_cropped[idx].size))

    ssim = []
    for idx, img in enumerate(b1_cropped):
        ssim.append(structural_similarity(np.array(img.convert("L")), np.array(b1m_cropped[idx].convert("L")), full=True)[0])

    psnr = []
    for idx, img in enumerate(b1_cropped):
        psnr.append(calculate_psnr(np.array(img), np.array(b1m_cropped[idx])))

    clip = clip_sim.i2i_similarity(b1_cropped, b1m_cropped)
    metrics["per_seed"] = {s: {"clip": clip[idx].item(), "ssim": ssim[idx], "psnr": psnr[idx]} for idx, s in enumerate(seeds)}
    metrics["mean_clip"] = clip.mean().item()
    metrics["max_clip"] = clip.max().item()
    metrics["mean_psnr"] = np.array(psnr).mean().item()
    metrics["max_psnr"] = np.array(psnr).max().item()
    metrics["mean_ssim"] = np.array(ssim).mean().item()
    metrics["max_ssim"] = np.array(ssim).max().item()

    return metrics


def compute_bg_cons_metrics(b1_imgs, b1m_imgs, b1_mask, b1m_mask, seeds, clip_sim):
    b1_mask = b1_mask.cpu().bool().numpy()
    b1m_mask = b1m_mask.cpu().bool().numpy()

    b1_bg_imgs = []
    for img in b1_imgs:
        b1_bg = np.array(img)
        b1_bg[b1_mask] = 0
        b1_bg[b1m_mask] = 0
        b1_bg_imgs.append(Image.fromarray(b1_bg))

    b1m_bg_imgs = []
    for img in b1m_imgs:
        b1m_bg = np.array(img)
        b1m_bg[b1_mask] = 0
        b1m_bg[b1m_mask] = 0
        b1m_bg_imgs.append(Image.fromarray(b1m_bg))

    metrics = {}
    ssim = []
    for idx, img in enumerate(b1_bg_imgs):
        ssim.append(structural_similarity(np.array(img.convert("L")), np.array(b1m_bg_imgs[idx].convert("L")), full=True)[0])

    psnr = []
    for idx, img in enumerate(b1_bg_imgs):
        psnr.append(calculate_psnr(np.array(img), np.array(b1m_bg_imgs[idx])))

    clip = clip_sim.i2i_similarity(b1_bg_imgs, b1m_bg_imgs)

    metrics["per_seed"] = {s: {"clip": clip[idx].item(), "ssim": ssim[idx], "psnr": psnr[idx]} for idx, s in enumerate(seeds)}
    metrics["mean_clip"] = clip.mean().item()
    metrics["max_clip"] = clip.max().item()
    metrics["mean_psnr"] = np.array(psnr).mean().item()
    metrics["max_psnr"] = np.array(psnr).max().item()
    metrics["mean_ssim"] = np.array(ssim).mean().item()
    metrics["max_ssim"] = np.array(ssim).max().item()

    return metrics
