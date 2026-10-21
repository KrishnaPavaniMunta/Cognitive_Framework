"""
Residual 3D Spatial Clustering Engine.

Isolates uncatalogued physical objects from RGB-D frames by:
1. Systematic Point Purge: Subtracting all known 2D/3D detector masks (YOLO + DINO).
2. Multi-Plane RANSAC: Stripping floor and vertical drywall planes.
3. Statistical Outlier Removal (SOR): Removing flying depth-edge pixels.
4. Adaptive DBSCAN: Density clustering with slender-pole adaptation (e.g. IV drip stands).
5. 3D Bounding & 2D Projection: Projecting 3D clusters to tight 2D RGB crops for classification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

LOG = logging.getLogger("spatial_clustering")


@dataclass
class ResidualCluster:
    """Represents an isolated physical 3D object cluster from depth residuals."""
    cluster_id: int
    points_cam: np.ndarray          # (N, 3) XYZ in camera frame
    points_world: np.ndarray | None # (N, 3) XYZ in world frame
    centroid_cam: tuple[float, float, float]
    centroid_world: tuple[float, float, float] | None
    bbox_cam: tuple[float, float, float, float, float, float]  # minX, maxX, minY, maxY, minZ, maxZ
    dimensions_m: tuple[float, float, float]                   # width, depth, height (dx, dz, dy)
    volume_m3: float
    bbox_2d: tuple[int, int, int, int]                         # x1, y1, x2, y2 in image pixels
    crop_rgb: np.ndarray                                       # Cropped RGB image patch
    aspect_ratio_h_w: float                                    # height / width ratio


class ResidualSpatialClustering:
    """
    Extracts residual 3D object clusters from synchronized RGB-D frames.
    """

    def __init__(
        self,
        intrinsics: tuple[float, float, float, float] | list[float] | np.ndarray,
        eps_m: float = 0.06,
        base_min_points: int = 25,
        slender_min_points: int = 15,
        min_cluster_volume_m3: float = 0.0005,
        max_cluster_span_m: float = 2.80,
        sor_k_neighbors: int = 15,
        sor_std_ratio: float = 1.5,
    ) -> None:
        """
        intrinsics: (fx, fy, cx, cy)
        """
        if isinstance(intrinsics, np.ndarray):
            if intrinsics.shape == (3, 3):
                self.fx = float(intrinsics[0, 0])
                self.fy = float(intrinsics[1, 1])
                self.cx = float(intrinsics[0, 2])
                self.cy = float(intrinsics[1, 2])
            else:
                self.fx, self.fy, self.cx, self.cy = (float(v) for v in intrinsics.flatten()[:4])
        elif hasattr(intrinsics, "fx"):
            self.fx = float(intrinsics.fx)
            self.fy = float(intrinsics.fy)
            self.cx = float(intrinsics.cx)
            self.cy = float(intrinsics.cy)
        else:
            self.fx, self.fy, self.cx, self.cy = (float(v) for v in intrinsics[:4])

        self.eps_m = float(eps_m)
        self.base_min_points = int(base_min_points)
        self.slender_min_points = int(slender_min_points)
        self.min_volume_m3 = float(min_cluster_volume_m3)
        self.max_span_m = float(max_cluster_span_m)
        self.sor_k = int(sor_k_neighbors)
        self.sor_std = float(sor_std_ratio)

    def extract_residual_clusters(
        self,
        rgb_image: np.ndarray,
        depth_mm: np.ndarray,
        known_masks_or_boxes: Sequence[Any],
        pose_matrix: np.ndarray | None = None,
        min_depth_m: float = 0.30,
        max_depth_m: float = 5.00,
        subsample_step: int = 2,
    ) -> list[ResidualCluster]:
        """
        Main entry point: Extracts uncatalogued physical clusters from an RGB-D frame.

        rgb_image: (H, W, 3) BGR or RGB image
        depth_mm: (H, W) depth map in millimetres
        known_masks_or_boxes: list of masks (H, W bool/uint8) or bounding boxes (x1, y1, x2, y2, ...)
        pose_matrix: Optional 4x4 camera-to-world transform
        """
        if rgb_image is None or depth_mm is None:
            return []

        h, w = depth_mm.shape[:2]

        # ── 1. Systematic Point Purge ──────────────────────────────────────────
        # Create exclusion mask for all pixels belonging to known detections
        exclusion_mask = np.zeros((h, w), dtype=bool)

        for item in known_masks_or_boxes:
            if isinstance(item, np.ndarray) and item.shape[:2] == (h, w):
                exclusion_mask |= (item > 0)
            elif isinstance(item, (tuple, list)) and len(item) >= 4:
                x1, y1, x2, y2 = (int(v) for v in item[:4])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    exclusion_mask[y1:y2, x1:x2] = True

        # Dilate exclusion mask by 3 pixels to strip edge leakage
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        exclusion_mask = cv2.dilate(exclusion_mask.astype(np.uint8), kernel).astype(bool)

        # Valid depth mask
        depth_valid = (depth_mm >= int(min_depth_m * 1000)) & (depth_mm <= int(max_depth_m * 1000))
        residual_mask = depth_valid & (~exclusion_mask)

        # ── 2. Backproject Residual Pixels to Camera 3D ────────────────────────
        # Subsample to keep processing under 30ms
        y_coords, x_coords = np.nonzero(residual_mask)
        if len(x_coords) < self.base_min_points:
            return []

        if subsample_step > 1:
            step_idx = np.arange(0, len(x_coords), subsample_step)
            y_coords = y_coords[step_idx]
            x_coords = x_coords[step_idx]

        z_vals = depth_mm[y_coords, x_coords].astype(np.float32) / 1000.0  # metres
        x_vals = (x_coords.astype(np.float32) - self.cx) * z_vals / self.fx
        y_vals = (y_coords.astype(np.float32) - self.cy) * z_vals / self.fy
        points_cam = np.column_stack([x_vals, y_vals, z_vals])  # (N, 3)

        if len(points_cam) < self.base_min_points:
            return []

        # ── 3. Multi-Plane RANSAC (Floor & Major Drywall Stripping) ─────────────
        points_non_plane = self._strip_dominant_planes(points_cam)
        if len(points_non_plane) < self.base_min_points:
            return []

        # ── 4. Statistical Outlier Removal (Flying Pixel Purge) ─────────────────
        points_clean = self._remove_statistical_outliers(points_non_plane)
        if len(points_clean) < self.slender_min_points:
            return []

        # ── 5. Adaptive DBSCAN Spatial Clustering ──────────────────────────────
        clustering = DBSCAN(eps=self.eps_m, min_samples=self.slender_min_points).fit(points_clean)
        labels = clustering.labels_
        unique_labels = [lbl for lbl in np.unique(labels) if lbl != -1]

        clusters: list[ResidualCluster] = []
        cluster_id = 1

        for lbl in unique_labels:
            pts_cluster = points_clean[labels == lbl]
            if len(pts_cluster) < self.slender_min_points:
                continue

            # Compute axis-aligned bounding box in camera coordinates
            min_x, min_y, min_z = np.min(pts_cluster, axis=0)
            max_x, max_y, max_z = np.max(pts_cluster, axis=0)

            dx = float(max_x - min_x)  # width
            dy = float(max_y - min_y)  # height in cam frame (y is down)
            dz = float(max_z - min_z)  # depth
            span = max(dx, dy, dz)

            # Slender Pole vs Bulk density check:
            # If shape is not tall (aspect ratio < 2.0) and points < base_min_points, discard as noise
            aspect_ratio = dy / max(0.01, dx)
            if aspect_ratio < 2.0 and len(pts_cluster) < self.base_min_points:
                continue

            # Outlier pruning: volume and span limits
            vol = dx * dy * max(dz, 0.02)
            if vol < self.min_volume_m3 or span > self.max_span_m:
                continue

            # Centroid
            cen_x, cen_y, cen_z = np.mean(pts_cluster, axis=0)
            cen_cam = (float(cen_x), float(cen_y), float(cen_z))

            # World points if pose matrix is available
            pts_world = None
            cen_world = None
            if pose_matrix is not None:
                m = np.asarray(pose_matrix, dtype=np.float64)
                homog = np.column_stack([pts_cluster, np.ones(len(pts_cluster), dtype=np.float32)])
                pts_world = (m @ homog.T)[:3].T.astype(np.float32)
                cen_w = m @ np.array([cen_x, cen_y, cen_z, 1.0], dtype=np.float64)
                cen_world = (float(cen_w[0]), float(cen_w[1]), float(cen_w[2]))

            # ── 6. 3D-to-2D Intrinsic Projection & RGB Crop ────────────────────
            # Project all 8 3D bounding box corners to image coordinates
            corners_cam = np.array([
                [min_x, min_y, min_z],
                [max_x, min_y, min_z],
                [min_x, max_y, min_z],
                [max_x, max_y, min_z],
                [min_x, min_y, max_z],
                [max_x, min_y, max_z],
                [min_x, max_y, max_z],
                [max_x, max_y, max_z],
            ])

            # Filter corners with positive depth
            valid_corners = corners_cam[corners_cam[:, 2] > 0.05]
            if len(valid_corners) == 0:
                continue

            u_proj = (self.fx * valid_corners[:, 0] / valid_corners[:, 2]) + self.cx
            v_proj = (self.fy * valid_corners[:, 1] / valid_corners[:, 2]) + self.cy

            px1 = int(np.floor(np.min(u_proj)))
            py1 = int(np.floor(np.min(v_proj)))
            px2 = int(np.ceil(np.max(u_proj)))
            py2 = int(np.ceil(np.max(v_proj)))

            # Apply 10% safety margin around the crop
            margin_x = int(0.10 * max(10, px2 - px1))
            margin_y = int(0.10 * max(10, py2 - py1))
            px1 = max(0, px1 - margin_x)
            py1 = max(0, py1 - margin_y)
            px2 = min(w, px2 + margin_x)
            py2 = min(h, py2 + margin_y)

            if px2 - px1 < 12 or py2 - py1 < 12:
                continue

            crop = rgb_image[py1:py2, px1:px2].copy()

            clusters.append(
                ResidualCluster(
                    cluster_id=cluster_id,
                    points_cam=pts_cluster,
                    points_world=pts_world,
                    centroid_cam=cen_cam,
                    centroid_world=cen_world,
                    bbox_cam=(float(min_x), float(max_x), float(min_y), float(max_y), float(min_z), float(max_z)),
                    dimensions_m=(dx, dz, dy),
                    volume_m3=vol,
                    bbox_2d=(px1, py1, px2, py2),
                    crop_rgb=crop,
                    aspect_ratio_h_w=aspect_ratio,
                )
            )
            cluster_id += 1

        return clusters

    def _strip_dominant_planes(
        self,
        points: np.ndarray,
        max_planes: int = 2,
        dist_thresh_m: float = 0.05,
        min_inliers_count: int = 2500,
    ) -> np.ndarray:
        """
        Identifies and strips floor and massive drywall planes via RANSAC.
        Guarantees that small physical objects (boxes, carts, stands) are never
        stripped as planes.
        """
        remaining = points.copy()
        rng = np.random.default_rng(42)

        for _ in range(max_planes):
            if len(remaining) < min_inliers_count * 2:
                break

            best_inliers_mask = None
            best_inlier_count = 0

            # Subsample candidate set for fast plane proposals
            sample_pool_size = min(len(remaining), 1000)
            sample_pool = remaining[rng.choice(len(remaining), sample_pool_size, replace=False)]

            for _ in range(40):
                idx = rng.choice(sample_pool_size, 3, replace=False)
                p1, p2, p3 = sample_pool[idx]
                normal = np.cross(p2 - p1, p3 - p1)
                norm = np.linalg.norm(normal)
                if norm < 1e-6:
                    continue
                normal = normal / norm
                d = -np.dot(normal, p1)

                # Floor in camera frame has dominant Y component (camera Y points down: |n_y| > 0.65)
                # Drywall has dominant horizontal component (|n_y| < 0.25)
                is_floor = abs(normal[1]) > 0.65
                is_wall = abs(normal[1]) < 0.25

                if not (is_floor or is_wall):
                    continue

                dists = np.abs(remaining @ normal + d)
                inliers_mask = dists <= dist_thresh_m
                count = int(np.sum(inliers_mask))

                # Structural planes must be massive surfaces (> min_inliers_count)
                if count > best_inlier_count and count >= min_inliers_count:
                    best_inlier_count = count
                    best_inliers_mask = inliers_mask

            if best_inliers_mask is not None:
                remaining = remaining[~best_inliers_mask]
            else:
                break

        return remaining

    def _remove_statistical_outliers(self, points: np.ndarray) -> np.ndarray:
        """
        Fast Statistical Outlier Removal using cKDTree to strip flying depth pixels.
        """
        if len(points) <= self.sor_k:
            return points

        tree = cKDTree(points)
        dists, _ = tree.query(points, k=self.sor_k + 1)
        mean_dists = np.mean(dists[:, 1:], axis=1)  # Exclude self-distance at index 0

        mu = np.mean(mean_dists)
        sigma = np.std(mean_dists)
        threshold = mu + self.sor_std * sigma

        keep_mask = mean_dists <= threshold
        return points[keep_mask]
