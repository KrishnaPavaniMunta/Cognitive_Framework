from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from semantic_mapping.frame_loader import FrameData
from semantic_mapping.detection.depth_enrichment import Detection2DWithDepth
from semantic_mapping.segmentation.sam_masker import SAMMasker, DetectionsRefinements
from semantic_mapping.segmentation.mask_refinement import build_geometry_mask_from_sam

@dataclass(frozen=True)
class Detection2DWithMask:
    label: str
    score: float
    box: tuple[float, float, float, float]
    imgsz: int | None
    depth: Any
    mask: np.ndarray | None
    mask_info: dict[str, Any]
    geom_mask: np.ndarray | None
    geom_mask_info: dict[str, Any]


def _det_to_legacy_dict(det: Detection2DWithDepth) -> dict[str, Any]:
    """Adapter verso le utility legacy-like di SAM/refinement."""

    depth_dict = None
    if det.depth is not None:
        depth_dict = {
            "n": det.depth.n,
            "median": det.depth.median,
            "p10": det.depth.p10,
            "p90": det.depth.p90,
            "spread_p90_p10": det.depth.spread_p90_p10,
        }

    return {
        "label": det.label,
        "score": det.score,
        "box": det.box,
        "imgsz": det.imgsz,
        "depth": depth_dict,
    }


def segment_detections(
    frame_data: FrameData,
    detections: list[Detection2DWithDepth],
    masker: SAMMasker,
    *,
    use_box_exclusion: bool = True,
) -> list[Detection2DWithMask]:
    """
    Legacy-like SAM segmentation.

    Follows the original pipeline:
    1. convert detections to legacy dict format
    2. precompute bbox-based exclusion masks
    3. run SAM per detection
    4. subtract exclusion mask for background-like objects
    """

    H, W = frame_data.depth.shape

    legacy_dets = [
        _det_to_legacy_dict(det)
        for det in detections
    ]

    if use_box_exclusion:
        box_exclusions = DetectionsRefinements.build_box_exclusion_masks(
            detections_depth=legacy_dets,
            H=H,
            W=W,
            contain_thr=0.85,
            front_margin_m=0.20,
            only_background_like=True,
            background_like_labels={"wall", "floor", "door", "window", "ceiling"},
        )
    else:
        box_exclusions = [
            np.zeros((H, W), dtype=bool)
            for _ in detections
        ]

    outputs: list[Detection2DWithMask] = []

    for idx, det in enumerate(detections):
        mask, info = masker.segment(
            image_rgb=frame_data.image_rgb,
            box_xyxy=det.box,
            label=det.label,
            depth_map=frame_data.depth,
        )

        excl_info = None

        if mask is not None:
            mask, excl_info = DetectionsRefinements.apply_exclusion_to_mask(
                mask=mask,
                exclusion_mask=box_exclusions[idx],
                min_remaining_pixels=50,
            )

            info = {
                **(info or {}),
                "exclusion": excl_info,
            }
        geom_mask, geom_mask_info = build_geometry_mask_from_sam(
            mask,
            frame_data.depth,
            min_pixels=50,
            q_ref=15.0,
            tol_far_m=0.35,
        )
        outputs.append(
            Detection2DWithMask(
                label=det.label,
                score=det.score,
                box=det.box,
                imgsz=det.imgsz,
                depth=det.depth,
                mask=mask,
                mask_info=info or {"reason": "sam_none"},
                geom_mask=geom_mask,
                geom_mask_info=geom_mask_info,
            )
        )

    return outputs