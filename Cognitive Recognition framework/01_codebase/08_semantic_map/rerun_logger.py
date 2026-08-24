"""
Rerun scene logger for the semantic map builder.

Builds a navigable 3D scene: accumulated world point cloud from the depth stream,
the camera frustum with its live RGB image, the robot trajectory, and labelled
3D boxes for every landmark. Writes a .rrd recording that can be reopened later.

Entity layout:
    world/                     right-handed, Z up
    world/camera               camera optical pose per frame
    world/camera/image         pinhole + RGB frame
    world/map/cloud_<n>        accumulated depth points in world coordinates
    world/trajectory           robot path
    world/landmarks            labelled points + oriented boxes
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np
import rerun as rr

DEFAULT_BOX_HALF_SIZE_M = 0.25
MIN_CLOUD_DEPTH_M = 0.25
MAX_CLOUD_DEPTH_M = 6.0
DEFAULT_CLOUD_POINT_RADIUS_M = 0.006


def matrix_to_translation_quaternion(matrix) -> tuple[list[float], list[float]]:
    m = np.asarray(matrix, dtype=np.float64)
    rot = m[:3, :3]
    trace = float(np.trace(rot))

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rot[2, 1] - rot[1, 2]) / s
        qy = (rot[0, 2] - rot[2, 0]) / s
        qz = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        qw = (rot[2, 1] - rot[1, 2]) / s
        qx = 0.25 * s
        qy = (rot[0, 1] + rot[1, 0]) / s
        qz = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        qw = (rot[0, 2] - rot[2, 0]) / s
        qx = (rot[0, 1] + rot[1, 0]) / s
        qy = 0.25 * s
        qz = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        qw = (rot[1, 0] - rot[0, 1]) / s
        qx = (rot[0, 2] + rot[2, 0]) / s
        qy = (rot[1, 2] + rot[2, 1]) / s
        qz = 0.25 * s

    return [float(m[0, 3]), float(m[1, 3]), float(m[2, 3])], [float(qx), float(qy), float(qz), float(qw)]


def class_color(class_name: str) -> list[int]:
    h = abs(hash(class_name))
    return [80 + (h & 0x7F), 80 + ((h >> 8) & 0x7F), 80 + ((h >> 16) & 0x7F)]


def stable_landmark_path(landmark: dict, landmark_id: int | None = None) -> str:
    class_slug = re.sub(r"[^a-z0-9_-]+", "_", str(landmark["class_name"]).lower()).strip("_") or "unknown"
    resolved_id = landmark.get("landmark_id", landmark_id)
    if resolved_id is None:
        raise ValueError("A landmark_id is required for a stable Rerun entity path")
    return f"world/landmarks/{class_slug}_{int(resolved_id)}"


def landmark_metadata(landmark: dict, landmark_id: int | None = None) -> dict:
    ontology = landmark.get("ontology") or {}
    hierarchy = ontology.get("hierarchy") or []
    hierarchy_names = [item.get("name", "") if isinstance(item, dict) else str(item) for item in hierarchy]
    mean_confidence = landmark.get("mean_confidence")
    if mean_confidence is None and landmark.get("conf_sum") is not None:
        mean_confidence = landmark["conf_sum"] / max(1, landmark.get("hit_count", 1))

    values = {
        "landmark_id": landmark.get("landmark_id", landmark_id),
        "map_class": landmark["class_name"],
        "instance_id": landmark.get("instance_id"),
        "world_frame": landmark.get("world_frame"),
        "allowed_in_space": "UNKNOWN",
        "hit_count": landmark.get("hit_count"),
        "mean_confidence": mean_confidence,
        "max_confidence": landmark.get("max_confidence"),
        "first_observed": landmark.get("first_seen") or landmark.get("first_seen_ns"),
        "last_observed": landmark.get("last_seen") or landmark.get("last_seen_ns"),
        "ontology_class": ontology.get("resolved_name"),
        "ontology_hierarchy": " > ".join(hierarchy_names),
        "ontology_dimensions_json": json.dumps(ontology.get("dimensions") or {}, sort_keys=True),
        "ontology_comments": "\n".join(ontology.get("comments") or []),
        "ontology_properties_json": json.dumps(ontology.get("properties") or [], sort_keys=True),
    }
    return {key: value for key, value in values.items() if value is not None}


def log_landmark_entities(landmarks, size_lookup=None) -> None:
    items = landmarks.items() if isinstance(landmarks, dict) else (
        (landmark.get("landmark_id"), landmark) for landmark in landmarks
    )
    for landmark_id, landmark in items:
        center = np.asarray([[landmark["X"], landmark["Y"], landmark["Z"]]], dtype=np.float32)
        label = f"{landmark['class_name']} {landmark['instance_id']}"
        color = [class_color(landmark["class_name"])]
        extent = size_lookup(landmark["class_name"]) if size_lookup else None
        if extent is None:
            half_size = [[DEFAULT_BOX_HALF_SIZE_M] * 3]
        else:
            width_m, height_m = extent
            half_size = [[width_m / 2.0, width_m / 2.0, height_m / 2.0]]

        rr.log(
            stable_landmark_path(landmark, landmark_id),
            rr.Points3D(center, colors=color, labels=[label], radii=0.06),
            rr.Boxes3D(
                centers=center,
                half_sizes=np.asarray(half_size, dtype=np.float32),
                labels=[label],
                colors=color,
                fill_mode="TransparentFillMajorWireframe",
            ),
            rr.AnyValues(**landmark_metadata(landmark, landmark_id)),
            static=True,
        )


class RerunSceneLogger:
    def __init__(
        self,
        recording_path: Path,
        *,
        application_id: str,
        cloud_stride: int = 6,
        cloud_every_n_frames: int = 5,
        cloud_smoothing: bool = True,
        cloud_point_radius_m: float = DEFAULT_CLOUD_POINT_RADIUS_M,
        spawn_viewer: bool = False,
    ) -> None:
        self.recording_path = recording_path
        self.cloud_stride = max(1, int(cloud_stride))
        self.cloud_every_n_frames = max(1, int(cloud_every_n_frames))
        self.cloud_smoothing = bool(cloud_smoothing)
        self.cloud_point_radius_m = max(0.001, float(cloud_point_radius_m))
        self.trajectory: list[list[float]] = []
        self.cloud_chunks = 0
        self.points_logged = 0

        rr.init(application_id, spawn=spawn_viewer)
        recording_path.parent.mkdir(parents=True, exist_ok=True)
        rr.save(str(recording_path))
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    def _set_time(self, frame_index: int, timestamp_ns: int) -> None:
        try:
            rr.set_time("frame", sequence=frame_index)
            rr.set_time("bag_time", timestamp=timestamp_ns * 1e-9)
        except (AttributeError, TypeError):  # rerun < 0.23 API
            rr.set_time_sequence("frame", frame_index)

    def log_frame(
        self,
        frame_index: int,
        timestamp_ns: int,
        rgb_bgr: np.ndarray,
        depth_mm: np.ndarray,
        intrinsics,
        pose_matrix,
    ) -> None:
        self._set_time(frame_index, timestamp_ns)

        translation, quaternion = matrix_to_translation_quaternion(pose_matrix)
        self.trajectory.append(translation)
        rr.log("world/camera", rr.Transform3D(translation=translation, rotation=rr.Quaternion(xyzw=quaternion)))

        height, width = rgb_bgr.shape[:2]
        k = np.array(
            [[intrinsics.fx, 0.0, intrinsics.cx], [0.0, intrinsics.fy, intrinsics.cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        rr.log("world/camera/image", rr.Pinhole(image_from_camera=k, resolution=[width, height]))
        rr.log("world/camera/image", rr.Image(cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)))

        if frame_index % self.cloud_every_n_frames == 0:
            self._log_cloud(rgb_bgr, depth_mm, intrinsics, pose_matrix)

    def _log_cloud(self, rgb_bgr: np.ndarray, depth_mm: np.ndarray, intrinsics, pose_matrix) -> None:
        step = self.cloud_stride
        depth_full = depth_mm.astype(np.float32) / 1000.0
        valid_full = (depth_full > MIN_CLOUD_DEPTH_M) & (depth_full < MAX_CLOUD_DEPTH_M) & np.isfinite(depth_full)
        if self.cloud_smoothing:
            # Preserve depth edges while removing isolated sensor noise before back-projection.
            filtered = cv2.bilateralFilter(depth_full, d=5, sigmaColor=0.08, sigmaSpace=2.0)
            depth_full = np.where(valid_full & (np.abs(filtered - depth_full) <= 0.15), filtered, 0.0)

        depth = depth_full[::step, ::step]
        rows, cols = depth.shape
        us = (np.arange(cols) * step).astype(np.float32)
        vs = (np.arange(rows) * step).astype(np.float32)
        grid_u, grid_v = np.meshgrid(us, vs)

        valid = (depth > MIN_CLOUD_DEPTH_M) & (depth < MAX_CLOUD_DEPTH_M) & np.isfinite(depth)
        if not np.any(valid):
            return

        z = depth[valid]
        x = (grid_u[valid] - intrinsics.cx) * z / intrinsics.fx
        y = (grid_v[valid] - intrinsics.cy) * z / intrinsics.fy

        points_cam = np.stack([x, y, z, np.ones_like(z)], axis=0)
        points_world = (np.asarray(pose_matrix, dtype=np.float64) @ points_cam)[:3].T

        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)[::step, ::step]
        colors = rgb[valid]

        rr.log(
            f"world/map/cloud_{self.cloud_chunks:05d}",
            rr.Points3D(points_world.astype(np.float32), colors=colors, radii=self.cloud_point_radius_m),
            static=True,
        )
        self.cloud_chunks += 1
        self.points_logged += int(points_world.shape[0])

    def log_landmarks(self, landmarks: dict[int, dict], size_lookup=None) -> None:
        """Static labelled points and boxes; call once the map is final."""
        if not landmarks:
            return
        log_landmark_entities(landmarks, size_lookup=size_lookup)

    def finish(self) -> None:
        if len(self.trajectory) >= 2:
            rr.log(
                "world/trajectory",
                rr.LineStrips3D([np.asarray(self.trajectory, dtype=np.float32)], colors=[[255, 220, 0]], radii=0.02),
                static=True,
            )
        rr.rerun_shutdown()
