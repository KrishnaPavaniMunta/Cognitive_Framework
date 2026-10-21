from __future__ import annotations

import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import cv2
import numpy as np


_dimensions_config_cache: dict[str, dict[str, dict[str, float]]] = {}

RDF_ABOUT = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about"
RDFS_COMMENT = "{http://www.w3.org/2000/01/rdf-schema#}comment"
DIMENSION_NAMESPACE = "http://www.semanticweb.org/chevi/ontologies/2026/5/52-classes-ontology#"
PHYSICAL_DIMENSIONS_TAG = f"{{{DIMENSION_NAMESPACE}}}physicalDimensions"
DIMENSION_TAGS = {
    "min_w": f"{{{DIMENSION_NAMESPACE}}}hasMinWidthM",
    "max_w": f"{{{DIMENSION_NAMESPACE}}}hasMaxWidthM",
    "min_h": f"{{{DIMENSION_NAMESPACE}}}hasMinHeightM",
    "max_h": f"{{{DIMENSION_NAMESPACE}}}hasMaxHeightM",
}


def load_dimensions_config(dimensions_config_path: str | Path) -> dict[str, dict[str, float]]:
    cache_key = str(Path(dimensions_config_path).resolve())
    if cache_key in _dimensions_config_cache:
        return _dimensions_config_cache[cache_key]

    config_path = Path(dimensions_config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Ontology not found: {config_path}")

    try:
        root = ET.parse(config_path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"Invalid RDF/XML ontology in {config_path}: {exc}") from exc

    parsed: dict[str, dict[str, float]] = {}

    # 1. Parse explicit machine-readable physicalDimensions annotations
    for element in root.iter():
        class_uri = element.attrib.get(RDF_ABOUT)
        if not class_uri:
            continue

        values: dict[str, float] = {}
        for key, tag in DIMENSION_TAGS.items():
            node = element.find(tag)
            if node is not None and node.text is not None:
                values[key] = float(node.text)

        annotation = element.find(PHYSICAL_DIMENSIONS_TAG)
        if annotation is not None and annotation.text:
            spec = json.loads(annotation.text)
            range_spec = spec.get("range", {})
            width_range = range_spec.get("width", [])
            height_range = range_spec.get("height", [])
            if len(width_range) == 2 and len(height_range) == 2:
                values = {
                    "min_w": float(width_range[0]),
                    "max_w": float(width_range[1]),
                    "min_h": float(height_range[0]),
                    "max_h": float(height_range[1]),
                }
                typical = spec.get("typical", {})
                depth_range = range_spec.get("depth", [])
                values.update({
                    "width": float(typical["width"]) if "width" in typical else 0.0,
                    "depth": float(typical["depth"]) if "depth" in typical else 0.0,
                    "height": float(typical["height"]) if "height" in typical else 0.0,
                    "min_d": float(depth_range[0]) if len(depth_range) == 2 else 0.0,
                    "max_d": float(depth_range[1]) if len(depth_range) == 2 else 0.0,
                })

        if not all(key in values for key in ("min_w", "max_w", "min_h", "max_h")):
            continue

        class_name = class_uri.rsplit("#", 1)[-1]
        parsed[class_name] = values

    # 2. Parse human-curated typical dimensions from rdfs:comment in ontology.rdf
    for element in root.iter():
        class_uri = element.attrib.get(RDF_ABOUT)
        if not class_uri:
            continue
        class_name = class_uri.rsplit("#", 1)[-1]
        if class_name in parsed:
            continue

        comment_node = element.find(RDFS_COMMENT)
        if comment_node is None or not comment_node.text:
            continue
        comment = comment_node.text

        match_3d = re.search(
            r"(?:dimensions|size).*?([0-9]+(?:\.[0-9]+)?)\s*[x×]\s*([0-9]+(?:\.[0-9]+)?)\s*[x×]\s*([0-9]+(?:\.[0-9]+)?)\s*cm",
            comment,
            re.IGNORECASE,
        )
        if match_3d:
            n1, n2, n3 = [float(v) / 100.0 for v in match_3d.groups()]
            if "counter" in class_name.lower() or "W" in comment:
                w, h, d = n1, n2, n3
            else:
                w, d, h = n1, n3, n2

            parsed[class_name] = {
                "width": w,
                "height": h,
                "depth": d,
                "min_w": round(max(0.15, w * 0.50), 2),
                "max_w": round(w * 2.00, 2),
                "min_h": round(max(0.20, h * 0.70), 2),
                "max_h": round(h * 1.30, 2),
                "min_d": round(max(0.15, d * 0.50), 2),
                "max_d": round(d * 2.00, 2),
            }
            continue

        match_pole = re.search(
            r"([0-9]+(?:\.[0-9]+)?)\s*cm\s*(?:base|diameter).*?([0-9]+(?:\.[0-9]+)?)[-–]([0-9]+(?:\.[0-9]+)?)\s*cm\s*height",
            comment,
            re.IGNORECASE,
        )
        if match_pole:
            base_m = float(match_pole.group(1)) / 100.0
            h_min = float(match_pole.group(2)) / 100.0
            h_max = float(match_pole.group(3)) / 100.0
            parsed[class_name] = {
                "width": base_m,
                "height": round((h_min + h_max) / 2.0, 2),
                "depth": base_m,
                "min_w": round(max(0.15, base_m * 0.50), 2),
                "max_w": round(base_m * 1.40, 2),
                "min_h": round(h_min * 0.85, 2),
                "max_h": round(h_max * 1.15, 2),
                "min_d": round(max(0.15, base_m * 0.50), 2),
                "max_d": round(base_m * 1.40, 2),
            }

    if not parsed:
        raise ValueError(f"No physical-size annotations found in ontology: {config_path}")

    _dimensions_config_cache[cache_key] = parsed
    print(f"[DEPTH FILTER] Loaded ontology size limits for {len(parsed)} classes from {config_path.name}")
    return parsed


def _backproject_depth_points(depth_mm, x1, y1, x2, y2, intrinsics, min_points=10, erosion_pixels=2, mask=None):
    """
    Step 1 & 2 of the Architecture:
    1. Extract ROI (optionally constrained by neural segmentation mask) and isolate foreground depth cluster.
    2. Valid-depth masking: filter invalid/zero depth, use median depth to resist background bleed through mesh.
    3. Backproject physical surface points into 3D.
    """
    h_img, w_img = depth_mm.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w_img, int(x2)), min(h_img, int(y2))

    if x2 <= x1 or y2 <= y1:
        return None

    depth_roi = depth_mm[y1:y2, x1:x2].astype(np.float32)
    x_indices, y_indices = np.meshgrid(np.arange(x1, x2), np.arange(y1, y2))

    valid_mask = np.isfinite(depth_roi) & (depth_roi >= 250.0) & (depth_roi <= 6500.0)
    if mask is not None:
        mask_roi = mask[y1:y2, x1:x2]
        valid_mask = valid_mask & (mask_roi > 0)

    if int(np.sum(valid_mask)) < int(min_points):
        return None

    valid_depths = depth_roi[valid_mask]
    z_med = float(np.median(valid_depths))
    fg_mask = valid_mask & (np.abs(depth_roi - z_med) <= 450.0)

    if mask is None and erosion_pixels > 0 and int(np.sum(fg_mask)) > 40:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        eroded = cv2.erode(fg_mask.astype(np.uint8), kernel, iterations=1)
        if int(np.sum(eroded)) >= int(min_points):
            fg_mask = (eroded > 0)

    if int(np.sum(fg_mask)) < int(min_points):
        fg_mask = valid_mask

    z_m = depth_roi[fg_mask] / 1000.0
    x_px = x_indices[fg_mask]
    y_px = y_indices[fg_mask]

    X = (x_px - float(intrinsics.cx)) * z_m / float(intrinsics.fx)
    Y = (y_px - float(intrinsics.cy)) * z_m / float(intrinsics.fy)
    Z = z_m
    return X, Y, Z


def get_oriented_3d_dimensions(depth_mm, x1, y1, x2, y2, intrinsics, mask=None):
    """
    Convert a 2D detection ROI (or neural mask) into rotation-robust 3D width/height estimates.
    """
    points = _backproject_depth_points(depth_mm, x1, y1, x2, y2, intrinsics, mask=mask)
    if points is None:
        return None

    X, Y, Z = points

    extent_height = float(np.max(Y) - np.min(Y))

    median_z = float(np.median(Z))
    std_z = float(np.std(Z))
    z_threshold = max(1.5 * std_z, 0.5)
    inlier_mask = np.abs(Z - median_z) < z_threshold

    X_filtered = X[inlier_mask]
    Z_filtered = Z[inlier_mask]
    if len(X_filtered) < 20:
        return None

    pts_2d = np.column_stack((X_filtered, Z_filtered))
    pts_centered = pts_2d - np.mean(pts_2d, axis=0)
    cov = np.cov(pts_centered, rowvar=False)
    _, eigenvectors = np.linalg.eigh(cov)

    pts_projected = pts_centered @ eigenvectors
    horizontal_dimensions = np.max(pts_projected, axis=0) - np.min(pts_projected, axis=0)
    extent_width = float(np.max(horizontal_dimensions))
    extent_depth = float(np.min(horizontal_dimensions))

    if extent_width <= 0 or extent_height <= 0:
        extent_width = float(np.max(X) - np.min(X))
        extent_depth = float(np.max(Z) - np.min(Z))
        if extent_width <= 0 or extent_height <= 0:
            return None

    return extent_width, extent_depth, extent_height


def extract_world_surface_points(
    depth_mm: np.ndarray,
    x1: int | float,
    y1: int | float,
    x2: int | float,
    y2: int | float,
    intrinsics,
    pose_matrix: np.ndarray,
    mask: np.ndarray | None = None,
    max_points: int = 300,
) -> np.ndarray | None:
    """
    Backprojects the foreground depth points (constrained by neural mask + 2px erosion)
    into 3D world coordinates.
    Returns: (N, 3) numpy array of world XYZ points, or None.
    """
    points = _backproject_depth_points(depth_mm, int(x1), int(y1), int(x2), int(y2), intrinsics, mask=mask)
    if points is None:
        return None
    X, Y, Z = points
    if len(X) == 0:
        return None

    if len(X) > max_points:
        indices = np.random.choice(len(X), size=max_points, replace=False)
        X, Y, Z = X[indices], Y[indices], Z[indices]

    pts_cam = np.stack([X, Y, Z, np.ones_like(Z)], axis=0)
    m = np.asarray(pose_matrix, dtype=np.float64)
    pts_world = (m @ pts_cam)[:3].T
    return pts_world.astype(np.float32)


def _fit_plane_from_points(points3d):
    p1, p2, p3 = points3d
    normal = np.cross(p2 - p1, p3 - p1)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-6:
        return None

    normal = normal / norm
    if abs(float(normal[1])) < 0.5:
        return None

    # In optical frame, Y is down. Orient normal pointing UP towards ceiling (normal[1] < 0)
    if normal[1] > 0.0:
        normal = -normal

    d = -float(np.dot(normal, p1))
    return normal, d


def estimate_floor_plane(depth_mm, intrinsics, *, lower_image_ratio=0.55, max_points=2500, ransac_iters=80, inlier_threshold_m=0.03):
    h_img, w_img = depth_mm.shape[:2]
    y_start = int(h_img * float(lower_image_ratio))
    if y_start >= h_img - 5:
        return None

    crop = _backproject_depth_points(depth_mm, 0, y_start, w_img, h_img, intrinsics, min_points=200)
    if crop is None:
        return None

    X, Y, Z = crop
    points = np.column_stack((X, Y, Z))
    if len(points) < 200:
        return None

    if len(points) > max_points:
        idx = np.linspace(0, len(points) - 1, max_points, dtype=int)
        points = points[idx]

    rng = np.random.default_rng(42)
    best_plane = None
    best_inliers = -1
    best_error = float("inf")

    for _ in range(int(ransac_iters)):
        sample_idx = rng.choice(len(points), size=3, replace=False)
        plane = _fit_plane_from_points(points[sample_idx])
        if plane is None:
            continue

        normal, d = plane
        distances = np.abs(points @ normal + d)
        inlier_mask = distances <= float(inlier_threshold_m)
        inlier_count = int(np.sum(inlier_mask))
        if inlier_count < 120:
            continue

        median_error = float(np.median(distances[inlier_mask]))
        if inlier_count > best_inliers or (inlier_count == best_inliers and median_error < best_error):
            best_plane = (normal, d)
            best_inliers = inlier_count
            best_error = median_error

    return best_plane


def estimate_box_floor_clearance(depth_mm, x1, y1, x2, y2, intrinsics, floor_plane, *, center_ratio=0.35, use_bottom=False):
    if floor_plane is None:
        return None

    box_w = max(1.0, float(x2) - float(x1))
    box_h = max(1.0, float(y2) - float(y1))
    cx = (float(x1) + float(x2)) * 0.5
    cy = (float(y1) + float(y2)) * 0.5 if not use_bottom else float(y2) - 0.15 * box_h
    half_w = box_w * float(center_ratio) * 0.5
    half_h = box_h * float(center_ratio) * 0.5

    points = _backproject_depth_points(
        depth_mm,
        cx - half_w,
        cy - half_h,
        cx + half_w,
        cy + half_h,
        intrinsics,
        min_points=8,
    )
    if points is None:
        return None

    X, Y, Z = points
    point = np.array([np.median(X), np.median(Y), np.median(Z)], dtype=np.float32)
    normal, d = floor_plane
    # Signed elevation above floor plane
    return float(np.dot(normal, point) + d)


def check_planar_backing(depth_mm: np.ndarray, x1: float, y1: float, x2: float, y2: float, *, max_gap_m: float = 0.35) -> bool:
    """
    Checks if a detected object (e.g. hazmat sign) has a physical vertical supporting
    surface (wall, door, cabinet) directly behind or adjacent to it, rejecting floating artifacts.
    """
    h_img, w_img = depth_mm.shape[:2]
    bx1, by1, bx2, by2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
    bw = max(1, bx2 - bx1)
    bh = max(1, by2 - by1)

    inner = depth_mm[max(0, by1 + int(0.2 * bh)):min(h_img, by2 - int(0.2 * bh)),
                     max(0, bx1 + int(0.2 * bw)):min(w_img, bx2 - int(0.2 * bw))]
    valid_inner = inner[(inner > 200) & (inner < 8000)]
    if len(valid_inner) < 10:
        return True

    d_obj_m = float(np.median(valid_inner)) / 1000.0

    margin_x = max(4, int(bw * 0.25))
    margin_y = max(4, int(bh * 0.25))

    border_samples = []
    if bx1 > margin_x:
        border_samples.append(depth_mm[max(0, by1):min(h_img, by2), max(0, bx1 - margin_x):bx1])
    if bx2 + margin_x < w_img:
        border_samples.append(depth_mm[max(0, by1):min(h_img, by2), bx2:min(w_img, bx2 + margin_x)])
    if by1 > margin_y:
        border_samples.append(depth_mm[max(0, by1 - margin_y):by1, max(0, bx1):min(w_img, bx2)])
    if by2 + margin_y < h_img:
        border_samples.append(depth_mm[by2:min(h_img, by2 + margin_y), max(0, bx1):min(w_img, bx2)])

    if not border_samples:
        return True

    border_depths = np.concatenate([s[(s > 200) & (s < 8000)] for s in border_samples if np.any((s > 200) & (s < 8000))])
    if len(border_depths) < 20:
        return True

    d_border_m = float(np.median(border_depths)) / 1000.0
    if (d_border_m - d_obj_m) > float(max_gap_m):
        return False

    return True


DEFAULT_HOSPITAL_PHYSICAL_LIMITS: dict[str, dict[str, float]] = {
    "reception_counter": {
        "min_w": 0.90, "max_w": 5.00,
        "min_h": 0.75, "max_h": 1.40,
        "min_d": 0.35, "max_d": 2.50,
    },
    "iv_stand": {
        "min_w": 0.25, "max_w": 0.85,
        "min_h": 1.20, "max_h": 2.50,
        "min_d": 0.20, "max_d": 0.85,
    },
}


def apply_depth_size_filter(
    predictions,
    depth_mm,
    intrinsics,
    *,
    dimensions_config_path,
    class_name_alias_candidates,
    enable_bin_physical_gating=True,
    enable_spillage_floor_gate=True,
    spillage_floor_clearance_m=0.10,
):
    if not predictions or depth_mm is None:
        return predictions

    size_limits = load_dimensions_config(dimensions_config_path)
    filtered = []
    removed_size = 0
    removed_spillage = 0
    removed_camera = 0
    removed_hazmat = 0

    needs_floor_plane = any(
        any(k in str(name).lower() for k in ("spillage", "spill", "camera", "hazmat"))
        for _, _, _, _, _, name in predictions
    )
    floor_plane = estimate_floor_plane(depth_mm, intrinsics) if needs_floor_plane else None

    for x1, y1, x2, y2, conf, name in predictions:
        base_name = str(name).replace("[DINO] ", "").lower()

        # ── 1. Spatial Common Sense: Security Camera ───────────────────────────
        # Must be high-mounted near ceiling (>= 1.70m above floor). Rejects desk/floor brackets.
        if "camera" in base_name and floor_plane is not None:
            clearance_m = estimate_box_floor_clearance(depth_mm, x1, y1, x2, y2, intrinsics, floor_plane)
            if clearance_m is not None and clearance_m < 1.70:
                removed_camera += 1
                continue

        # ── 2. Spatial Common Sense: Spillage ──────────────────────────────────
        # Must rest directly on the floor plane (|height| <= 0.12m). Rejects mid-air/wall spills.
        if base_name in ("spillage", "spill") and floor_plane is not None:
            clearance_m = estimate_box_floor_clearance(depth_mm, x1, y1, x2, y2, intrinsics, floor_plane, use_bottom=True)
            if clearance_m is not None and (clearance_m > float(spillage_floor_clearance_m) or clearance_m < -0.15):
                removed_spillage += 1
                continue

        # ── 3. Spatial Common Sense: Hazmat Sign ────────────────────────────────
        # Must be wall/door mounted at visual eye level (0.60m to 2.40m), not floating in mid-air.
        if "hazmat" in base_name:
            if floor_plane is not None:
                clearance_m = estimate_box_floor_clearance(depth_mm, x1, y1, x2, y2, intrinsics, floor_plane)
                if clearance_m is not None and (clearance_m < 0.60 or clearance_m > 2.40):
                    removed_hazmat += 1
                    continue
            if not check_planar_backing(depth_mm, x1, y1, x2, y2, max_gap_m=0.35):
                removed_hazmat += 1
                continue

        if not enable_bin_physical_gating and base_name in {"bin", "general_bin", "yellow_bin", "bin_tiger_stripe"}:
            filtered.append((x1, y1, x2, y2, conf, name))
            continue

        if base_name == "chair":
            filtered.append((x1, y1, x2, y2, conf, name))
            continue

        candidate_names = class_name_alias_candidates.get(base_name, [base_name])
        candidate_limits = [
            (candidate, size_limits[candidate])
            for candidate in candidate_names
            if candidate in size_limits
        ]
        if not candidate_limits and base_name in DEFAULT_HOSPITAL_PHYSICAL_LIMITS:
            candidate_limits = [(base_name, DEFAULT_HOSPITAL_PHYSICAL_LIMITS[base_name])]

        if not candidate_limits:
            filtered.append((x1, y1, x2, y2, conf, name))
            continue

        dims_3d = get_oriented_3d_dimensions(depth_mm, x1, y1, x2, y2, intrinsics)
        if dims_3d is None:
            filtered.append((x1, y1, x2, y2, conf, name))
            continue

        width_m, depth_m, height_m = dims_3d

        def _fits(lim: dict[str, float]) -> bool:
            if not (lim["min_w"] <= width_m <= lim["max_w"]):
                return False
            if not (lim["min_h"] <= height_m <= lim["max_h"]):
                return False
            min_d = lim.get("min_d", 0.0)
            max_d = lim.get("max_d", 0.0)
            if min_d > 0.0 and depth_m < min_d:
                return False
            if max_d > 0.0 and depth_m > max_d:
                return False
            return True

        fits_any = any(_fits(limits) for _, limits in candidate_limits)

        if not fits_any:
            removed_size += 1
            continue

        filtered.append((x1, y1, x2, y2, conf, name))

    if removed_camera > 0:
        print(f"  [SPATIAL 3D FILTER] Removed {removed_camera} fake security cameras below ceiling elevation (<1.70m).")
    if removed_spillage > 0:
        print(f"  [SPATIAL 3D FILTER] Removed {removed_spillage} spillage detections not resting on the floor plane.")
    if removed_hazmat > 0:
        print(f"  [SPATIAL 3D FILTER] Removed {removed_hazmat} hazmat signs floating or outside wall elevation.")
    if removed_size > 0:
        print(f"  [3D POINT CLOUD FILTER] Rotational PCA filter removed {removed_size} false-positive detections.")
    return filtered
