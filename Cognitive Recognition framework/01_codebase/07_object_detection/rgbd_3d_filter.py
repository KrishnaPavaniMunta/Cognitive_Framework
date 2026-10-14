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
    #    (e.g., reception_counter "200 × 110 × 70 cm", iv_stand "60 cm base × 150–220 cm height")
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

        # 3D bounding box pattern: W x H x D or W x D x H in cm
        match_3d = re.search(
            r"(?:dimensions|size).*?([0-9]+(?:\.[0-9]+)?)\s*[x×]\s*([0-9]+(?:\.[0-9]+)?)\s*[x×]\s*([0-9]+(?:\.[0-9]+)?)\s*cm",
            comment,
            re.IGNORECASE,
        )
        if match_3d:
            n1, n2, n3 = [float(v) / 100.0 for v in match_3d.groups()]
            # Reception counter: 200 x 110 x 70 cm -> W=2.00m, H=1.10m, D=0.70m
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

        # Pole / stand pattern: base diameter x height range in cm
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


def _backproject_depth_points(depth_mm, x1, y1, x2, y2, intrinsics, min_points=20, erosion_pixels=2, mask=None):
    """
    Step 1 & 2 of the Architecture:
    1. Extract ROI (optionally constrained by neural segmentation mask) and isolate foreground depth cluster.
    2. Erode binary mask by 2 pixels to strip flying depth pixels off object edges.
    3. Backproject only the clean surface points into 3D.
    """
    h_img, w_img = depth_mm.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w_img, int(x2)), min(h_img, int(y2))

    if x2 <= x1 or y2 <= y1:
        return None

    depth_roi = depth_mm[y1:y2, x1:x2].astype(np.float32)
    x_indices, y_indices = np.meshgrid(np.arange(x1, x2), np.arange(y1, y2))

    valid_mask = np.isfinite(depth_roi) & (depth_roi > 100) & (depth_roi < 8000)
    if mask is not None:
        mask_roi = mask[y1:y2, x1:x2]
        valid_mask = valid_mask & (mask_roi > 0)

    if int(np.sum(valid_mask)) < int(min_points):
        return None

    valid_depths = depth_roi[valid_mask]
    # Isolate foreground cluster (30th percentile represents nearest solid physical surface)
    z_fg = float(np.percentile(valid_depths, 30))
    fg_mask = valid_mask & (np.abs(depth_roi - z_fg) <= 300.0)

    # 2px morphological erosion to strip flying pixels and noisy boundary edges
    if erosion_pixels > 0 and int(np.sum(fg_mask)) > 40:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        eroded = cv2.erode(fg_mask.astype(np.uint8), kernel, iterations=int(erosion_pixels))
        if int(np.sum(eroded)) >= int(min_points):
            fg_mask = (eroded > 0)

    if int(np.sum(fg_mask)) < int(min_points):
        # Fallback to general valid mask if foreground cluster was too small
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

    # Subsample if too dense to keep downstream point cloud logging lightweight
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


def estimate_box_floor_clearance(depth_mm, x1, y1, x2, y2, intrinsics, floor_plane, *, center_ratio=0.35):
    if floor_plane is None:
        return None

    box_w = max(1.0, float(x2) - float(x1))
    box_h = max(1.0, float(y2) - float(y1))
    cx = (float(x1) + float(x2)) * 0.5
    cy = (float(y1) + float(y2)) * 0.5
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
    return float(abs(np.dot(normal, point) + d))


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

    needs_floor_plane = enable_spillage_floor_gate and any(
        str(name).replace("[DINO] ", "") == "spillage"
        for _, _, _, _, _, name in predictions
    )
    floor_plane = estimate_floor_plane(depth_mm, intrinsics) if needs_floor_plane else None

    for x1, y1, x2, y2, conf, name in predictions:
        base_name = str(name).replace("[DINO] ", "")

        if enable_spillage_floor_gate and base_name == "spillage" and floor_plane is not None:
            floor_clearance_m = estimate_box_floor_clearance(depth_mm, x1, y1, x2, y2, intrinsics, floor_plane)
            if floor_clearance_m is not None and floor_clearance_m > float(spillage_floor_clearance_m):
                removed_spillage += 1
                continue

        if not enable_bin_physical_gating and base_name in {"bin", "general_bin", "yellow_bin", "bin_tiger_stripe"}:
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

    if removed_spillage > 0:
        print(
            f"  [3D FLOOR FILTER] Removed {removed_spillage} spillage detections "
            f"floating more than {float(spillage_floor_clearance_m):.2f} m above the floor plane."
        )
    if removed_size > 0:
        print(f"  [3D POINT CLOUD FILTER] Rotational PCA filter removed {removed_size} false-positive detections.")
    return filtered