from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import FastSAM

LOG = logging.getLogger("fast_sam")


class FastSAMSegmenter:
    """
    High-speed, box-prompted neural instance segmenter for RGB-D hospital perception.
    Runs FastSAM-s in FP16 on CUDA (taking ~5-15ms per frame) to produce crisp pixel masks
    for all 2D detections from YOLO and Grounding DINO.
    """

    def __init__(
        self,
        weights_path: str | Path = "FastSAM-s.pt",
        device: str | None = None,
        half: bool = True,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.half = half and (self.device == "cuda")
        self.weights_path = str(weights_path)
        LOG.info("[FastSAM] Loading weights from %s on device=%s (half=%s)", self.weights_path, self.device, self.half)
        self.model = FastSAM(self.weights_path)

    def segment_boxes(
        self,
        rgb_img: np.ndarray,
        bboxes_xyxy: list[list[float]] | list[tuple[float, float, float, float]],
        erosion_pixels: int = 2,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Segment all bounding boxes in a single parallel batched pass.
        Returns a list of tuples: (raw_mask, eroded_mask) where each mask is a uint8 binary image (0 or 1)
        of shape (H, W).
        """
        h_img, w_img = rgb_img.shape[:2]
        if not bboxes_xyxy:
            return []

        # Convert to float list of boxes
        boxes_list = [[float(v) for v in b] for b in bboxes_xyxy]

        results = self.model(
            rgb_img,
            bboxes=boxes_list,
            device=self.device,
            half=self.half,
            verbose=False,
        )

        masks_data = None
        if results and results[0].masks is not None:
            masks_tensor = results[0].masks.data
            if isinstance(masks_tensor, torch.Tensor):
                # Resize masks to original image shape if needed
                m_np = masks_tensor.detach().cpu().numpy()
                if m_np.shape[1:] != (h_img, w_img):
                    resized = []
                    for m in m_np:
                        resized.append(cv2.resize(m.astype(np.float32), (w_img, h_img), interpolation=cv2.INTER_NEAREST))
                    masks_data = np.stack(resized, axis=0) if resized else None
                else:
                    masks_data = m_np

        output_masks: list[tuple[np.ndarray, np.ndarray]] = []
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) if erosion_pixels > 0 else None

        for box in boxes_list:
            bx1, by1, bx2, by2 = (int(round(v)) for v in box)
            bx1, by1 = max(0, bx1), max(0, by1)
            bx2, by2 = min(w_img, bx2), min(h_img, by2)

            best_mask = None
            if masks_data is not None and len(masks_data) > 0:
                best_iou = 0.0
                box_area = max(1.0, float((bx2 - bx1) * (by2 - by1)))
                for m in masks_data:
                    m_uint8 = (m > 0.5).astype(np.uint8)
                    ys, xs = np.where(m_uint8 > 0)
                    if len(xs) < 10:
                        continue
                    mx1, my1, mx2, my2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
                    ix1, iy1 = max(bx1, mx1), max(by1, my1)
                    ix2, iy2 = min(bx2, mx2), min(by2, my2)
                    if ix2 > ix1 and iy2 > iy1:
                        inter = (ix2 - ix1) * (iy2 - iy1)
                        union = box_area + (mx2 - mx1) * (my2 - my1) - inter
                        iou = inter / max(1.0, union)
                        if iou > best_iou:
                            best_iou = iou
                            best_mask = m_uint8

            # Fallback if no neural mask matched this box: inner foreground box mask
            if best_mask is None:
                best_mask = np.zeros((h_img, w_img), dtype=np.uint8)
                best_mask[by1:by2, bx1:bx2] = 1

            # Strictly clip mask to box boundaries to prevent bleeding outside YOLO detection
            clipped_mask = np.zeros((h_img, w_img), dtype=np.uint8)
            clipped_mask[by1:by2, bx1:bx2] = best_mask[by1:by2, bx1:bx2]

            # 2-pixel erosion to strip flying boundary edges
            if kernel is not None and int(np.sum(clipped_mask)) > 40:
                eroded = cv2.erode(clipped_mask, kernel, iterations=erosion_pixels)
                if int(np.sum(eroded)) >= 15:
                    eroded_mask = eroded
                else:
                    eroded_mask = clipped_mask
            else:
                eroded_mask = clipped_mask

            output_masks.append((clipped_mask, eroded_mask))

        return output_masks


def overlay_mask_on_frame(
    img: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.40,
    draw_contour: bool = True,
) -> np.ndarray:
    """
    Renders a translucent mask overlay and crisp outer contour on the image in-place or returned.
    """
    if mask is None or not np.any(mask):
        return img

    m_bool = (mask > 0)
    # Fast alpha blend on masked pixels only
    img[m_bool] = (
        (1.0 - alpha) * img[m_bool].astype(np.float32) + alpha * np.array(color, dtype=np.float32)
    ).astype(np.uint8)

    if draw_contour:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(img, contours, -1, color, 2, cv2.LINE_AA)

    return img

