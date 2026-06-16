from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def trim_far_pixels_from_sam_mask(
    sam_mask: np.ndarray | None,
    depth_map: np.ndarray,
    *,
    min_pixels: int = 50,
    q_ref: float = 15.0,
    tol_far_m: float = 0.35,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """
    Remove only pixels that are clearly farther than the frontal part
    of the SAM mask.

    Legacy logic:
    - compute z_ref as a low percentile inside the SAM mask
    - keep pixels with depth <= z_ref + tol_far_m
    - if the operation removes too much, fall back to the original mask

    This is useful when SAM leaks from the object to a farther wall/background.
    """

    if sam_mask is None:
        return None, {"reason": "no_sam"}

    mask = sam_mask.astype(bool)

    n_sam = int(np.count_nonzero(mask))
    if n_sam < int(min_pixels):
        return mask, {
            "reason": "sam_too_small",
            "sam_px": n_sam,
        }

    vals = depth_map[mask]
    vals = vals[np.isfinite(vals) & (vals > 0)]

    if vals.size < int(min_pixels):
        return mask, {
            "reason": "too_few_valid_depth",
            "sam_px": n_sam,
            "n_valid": int(vals.size),
        }

    z_ref = float(np.percentile(vals, float(q_ref)))
    z_thr = z_ref + float(tol_far_m)

    keep = (
        mask
        & np.isfinite(depth_map)
        & (depth_map > 0)
        & (depth_map <= z_thr)
    )

    n_keep = int(np.count_nonzero(keep))

    if n_keep < int(min_pixels):
        return mask, {
            "reason": "trim_too_aggressive_fallback_sam",
            "sam_px": n_sam,
            "keep_px": n_keep,
            "z_ref": z_ref,
            "z_thr": z_thr,
        }

    return keep, {
        "reason": "ok",
        "sam_px": n_sam,
        "keep_px": n_keep,
        "z_ref": z_ref,
        "z_thr": z_thr,
        "trimmed_px": n_sam - n_keep,
    }


def keep_largest_component(
    mask: np.ndarray | None,
    *,
    min_pixels: int = 50,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """
    Keep only the largest connected component.

    Legacy logic:
    useful to remove residual blobs after SAM/depth trimming.
    """

    if mask is None:
        return None, {"reason": "no_mask"}

    mask_bool = mask.astype(bool)

    n_pixels = int(np.count_nonzero(mask_bool))
    if n_pixels < int(min_pixels):
        return mask_bool, {
            "reason": "too_small",
            "pixels": n_pixels,
        }

    mask_u8 = (mask_bool.astype(np.uint8) * 255)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8,
        connectivity=8,
    )

    if n_labels <= 1:
        return mask_bool, {
            "reason": "single_component",
            "pixels": n_pixels,
        }

    areas = stats[1:, cv2.CC_STAT_AREA]
    best_idx = 1 + int(np.argmax(areas))
    best = labels == best_idx

    n_best = int(np.count_nonzero(best))

    if n_best < int(min_pixels):
        return mask_bool, {
            "reason": "largest_component_too_small_fallback",
            "pixels": n_pixels,
            "largest_component_pixels": n_best,
        }

    return best, {
        "reason": "ok",
        "n_components": int(n_labels - 1),
        "kept_px": n_best,
        "original_px": n_pixels,
    }


def build_geometry_mask_from_sam(
    sam_mask: np.ndarray | None,
    depth_map: np.ndarray,
    *,
    min_pixels: int = 50,
    q_ref: float = 15.0,
    tol_far_m: float = 0.35,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """
    Build the geometric mask used for 3D reconstruction.

    Pipeline:
    1. start from SAM mask
    2. trim far pixels using depth
    3. keep largest connected component

    Returns:
    - refined geometry mask
    - debug info
    """

    if sam_mask is None:
        return None, {
            "source": "none",
            "trim": {"reason": "no_sam"},
            "component": None,
            "pixels": 0,
        }

    trimmed, trim_info = trim_far_pixels_from_sam_mask(
        sam_mask=sam_mask,
        depth_map=depth_map,
        min_pixels=min_pixels,
        q_ref=q_ref,
        tol_far_m=tol_far_m,
    )

    refined, component_info = keep_largest_component(
        trimmed,
        min_pixels=min_pixels,
    )

    pixels = 0 if refined is None else int(np.count_nonzero(refined))

    return refined, {
        "source": "sam_trim_far_largest_component",
        "trim": trim_info,
        "component": component_info,
        "pixels": pixels,
    }