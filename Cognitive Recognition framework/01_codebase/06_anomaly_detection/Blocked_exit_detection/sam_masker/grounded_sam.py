"""
grounded_sam.py  —  Grounded-SAM door mask refiner
────────────────────────────────────────────────────
Pairs Grounding DINO bounding boxes with SAM (Segment Anything Model)
to produce pixel-tight polygon masks around detected doors/exit regions.

Usage pattern inside infer_hospitalguard_temporal.py
─────────────────────────────────────────────────────
  # Lazy-init once:
  refiner = GroundedSAMRefiner(sam_ckpt_path, device="cuda")

  # On every DINO frame (same cadence as active_dino):
  door_boxes = active_dino.get("door", [])
  if door_boxes:
      refined = refiner.refine_boxes(bgr, door_boxes)
      # refined is a list of (x1, y1, x2, y2, conf, mask_polygon_pts)

Design choices
──────────────
  • SAM runs on the same DINO cadence (every 15 frames), not every frame.
  • No depth / semantic_mapping dependency — pure OpenCV + segment-anything.
  • Falls back to original DINO box if SAM mask fails quality checks.
  • mask_to_tight_bbox() snaps the DINO box to the SAM mask's tight bounding
    rectangle, giving a much cleaner box that excludes glass/background.
  • Polygon contour is stored on each detection for optional overlay rendering.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

# ── SAM import guard ────────────────────────────────────────────────────────
try:
    from segment_anything import sam_model_registry, SamPredictor
    _SAM_AVAILABLE = True
except ImportError:
    _SAM_AVAILABLE = False


# ── Default SAM checkpoint path (can be overridden) ─────────────────────────
# Download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
_DEFAULT_SAM_CKPT = Path(__file__).parent / "sam_vit_h_4b8939.pth"
_SAM_MODEL_TYPE   = "vit_h"


def _mask_to_tight_bbox(
    mask: np.ndarray,
    orig_box: Tuple[float, float, float, float],
    min_area_frac: float = 0.05,
    max_area_frac: float = 0.97,
    min_inside_frac: float = 0.70,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Convert a binary SAM mask to a tight bounding box.
    Returns None if the mask fails quality gates (too small, overflows box, etc.).
    """
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None

    ox1, oy1, ox2, oy2 = [float(v) for v in orig_box]
    orig_area = max(1.0, (ox2 - ox1) * (oy2 - oy1))
    mask_area = int(len(xs))

    # Gate: mask must cover at least min_area_frac of original box
    if mask_area / orig_area < min_area_frac:
        return None

    # Gate: mask must not overflow more than max_area_frac of original box
    if mask_area / orig_area > max_area_frac:
        return None

    mx1, my1 = int(xs.min()), int(ys.min())
    mx2, my2 = int(xs.max()), int(ys.max())

    # Gate: tight box must be mostly inside original DINO box
    ix1 = max(mx1, int(ox1))
    iy1 = max(my1, int(oy1))
    ix2 = min(mx2, int(ox2))
    iy2 = min(my2, int(oy2))
    if ix2 <= ix1 or iy2 <= iy1:
        return None

    intersection = (ix2 - ix1) * (iy2 - iy1)
    tight_area   = max(1, (mx2 - mx1) * (my2 - my1))
    if intersection / tight_area < min_inside_frac:
        return None

    return mx1, my1, mx2, my2


def _mask_to_polygon(mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Convert binary SAM mask to the largest contour polygon (Nx1x2 int32).
    Returns None if no valid contour found.
    """
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 100:
        return None
    return largest


class GroundedSAMRefiner:
    """
    Refines DINO door bounding boxes into pixel-tight boxes using SAM masks.

    Parameters
    ──────────
    ckpt_path : Path to SAM weights file.  If None, tries _DEFAULT_SAM_CKPT.
    model_type: SAM model variant ("vit_h", "vit_l", "vit_b").
    device    : "cuda" or "cpu".
    """

    def __init__(
        self,
        ckpt_path: Optional[str | Path] = None,
        model_type: str = _SAM_MODEL_TYPE,
        device: str = "cuda",
    ):
        if not _SAM_AVAILABLE:
            raise ImportError(
                "segment-anything is not installed. "
                "Run: pip install git+https://github.com/facebookresearch/segment-anything.git"
            )

        ckpt = Path(ckpt_path) if ckpt_path else _DEFAULT_SAM_CKPT
        if not ckpt.exists():
            raise FileNotFoundError(
                f"SAM checkpoint not found: {ckpt}\n"
                "Download vit_h: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
            )

        self._device = device
        sam = sam_model_registry[model_type](checkpoint=str(ckpt))
        sam.to(device=device)
        sam.eval()
        self._predictor = SamPredictor(sam)
        self._last_embedding_id: Optional[int] = None
        print(f"  [SAM] {model_type} loaded from {ckpt.name} on {device.upper()}")

    def _set_image(self, bgr: np.ndarray, frame_id: Optional[int] = None) -> None:
        """Set SAM image embedding (cached by frame_id to avoid re-encoding)."""
        if frame_id is not None and frame_id == self._last_embedding_id:
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._predictor.set_image(rgb)
        self._last_embedding_id = frame_id

    def refine_boxes(
        self,
        bgr: np.ndarray,
        door_dets: List[Tuple],
        frame_id: Optional[int] = None,
    ) -> List[Tuple]:
        """
        Refine a list of DINO door detections with SAM masks.

        Each input detection is a tuple: (x1, y1, x2, y2, conf, *extra).
        Each output detection is: (x1, y1, x2, y2, conf, polygon_or_None).
        Falls back to original DINO box on any SAM failure.

        Parameters
        ──────────
        bgr      : Current video frame (BGR numpy array).
        door_dets: List of (x1, y1, x2, y2, conf) from DINO/YOLO.
        frame_id : Optional frame index for embedding cache.
        """
        if not door_dets:
            return []

        H, W = bgr.shape[:2]
        refined: List[Tuple] = []

        try:
            self._set_image(bgr, frame_id)
        except Exception as e:
            print(f"  [SAM] set_image failed: {e} — keeping DINO boxes")
            return [(d[0], d[1], d[2], d[3], d[4], None) for d in door_dets]

        for det in door_dets:
            x1, y1, x2, y2, conf = det[0], det[1], det[2], det[3], det[4]
            box_np = np.array([
                float(np.clip(x1, 0, W - 1)),
                float(np.clip(y1, 0, H - 1)),
                float(np.clip(x2, 0, W - 1)),
                float(np.clip(y2, 0, H - 1)),
            ])

            try:
                with torch.no_grad():
                    masks, scores, _ = self._predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=box_np[None, :],
                        multimask_output=True,
                    )

                # Pick highest-scoring mask
                best_idx  = int(scores.argmax())
                best_mask = masks[best_idx].astype(bool)
                best_score = float(scores[best_idx])

                polygon   = _mask_to_polygon(best_mask)
                tight_box = _mask_to_tight_bbox(best_mask, (x1, y1, x2, y2))

                if tight_box is not None:
                    tx1, ty1, tx2, ty2 = tight_box
                    # Blend SAM score with original DINO confidence
                    blended_conf = float(conf) * 0.6 + best_score * 0.4
                    refined.append((float(tx1), float(ty1), float(tx2), float(ty2),
                                    blended_conf, polygon))
                    print(f"  [SAM] door refined: [{int(x1)},{int(y1)},{int(x2)},{int(y2)}]"
                          f" → [{tx1},{ty1},{tx2},{ty2}]  score={best_score:.2f}")
                else:
                    # SAM mask failed quality gate — keep original box
                    refined.append((x1, y1, x2, y2, conf, polygon))
                    print(f"  [SAM] door mask quality gate failed — kept DINO box")

            except Exception as e:
                print(f"  [SAM] predict failed for door box: {e} — keeping DINO box")
                refined.append((x1, y1, x2, y2, conf, None))

        gc.collect()
        if self._device == "cuda":
            torch.cuda.empty_cache()

        return refined

    def draw_masks(
        self,
        scene: np.ndarray,
        sam_dets: List[Tuple],
        color_bgr: Tuple[int, int, int] = (255, 0, 0),
        alpha: float = 0.25,
    ) -> np.ndarray:
        """
        Overlay SAM polygon masks on a scene frame.
        sam_dets entries must be (x1, y1, x2, y2, conf, polygon_or_None).
        """
        for det in sam_dets:
            polygon = det[5] if len(det) > 5 else None
            if polygon is None:
                continue
            # Semi-transparent fill
            overlay = scene.copy()
            cv2.fillPoly(overlay, [polygon], color_bgr)
            cv2.addWeighted(overlay, alpha, scene, 1 - alpha, 0, scene)
            # Solid contour outline
            cv2.polylines(scene, [polygon], isClosed=True, color=color_bgr, thickness=2)
        return scene
