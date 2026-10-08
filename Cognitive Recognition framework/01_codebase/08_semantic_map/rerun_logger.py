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

    real_w = landmark.get("measured_width") or landmark.get("real_width")
    real_d = landmark.get("measured_depth") or landmark.get("real_depth")
    real_h = landmark.get("measured_height") or landmark.get("real_height")
    has_real = False
    if real_w is not None and real_h is not None:
        try:
            has_real = float(real_w) > 0.02 and float(real_h) > 0.02
        except (ValueError, TypeError):
            has_real = False

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
        "dimension_source": "camera_measured" if has_real else "ontology_typical",
        "measured_width_m": round(float(real_w), 3) if has_real and real_w else None,
        "measured_depth_m": round(float(real_d), 3) if has_real and real_d else None,
        "measured_height_m": round(float(real_h), 3) if has_real and real_h else None,
        "ontology_class": ontology.get("resolved_name"),
        "ontology_hierarchy": " > ".join(hierarchy_names),
        "ontology_dimensions_json": json.dumps(ontology.get("dimensions") or {}, sort_keys=True),
        "ontology_comments": "\n".join(ontology.get("comments") or []),
        "ontology_properties_json": json.dumps(ontology.get("properties") or [], sort_keys=True),
    }
    return {key: value for key, value in values.items() if value is not None}


def resolve_box_half_size(landmark: dict, size_lookup=None) -> list[list[float]]:
    """Resolve 3D bounding box half-extents [half_x, half_y, half_z] in metres.

    In ROS world coordinates (X forward / depth, Y lateral / width, Z vertical / height),
    box half sizes correspond to [depth / 2, width / 2, height / 2].

    Prioritizes real camera-measured physical dimensions when available;
    falls back to typical ontology dimensions or nominal limits.
    """
    # 1. Real measured camera dimensions
    real_w = landmark.get("measured_width") or landmark.get("real_width")
    real_d = landmark.get("measured_depth") or landmark.get("real_depth")
    real_h = landmark.get("measured_height") or landmark.get("real_height")
    if real_w is not None and real_h is not None:
        try:
            rw = float(real_w)
            rh = float(real_h)
            rd = float(real_d) if (real_d is not None and float(real_d) > 0) else rw
            if rw > 0.02 and rh > 0.02:
                return [[rd / 2.0, rw / 2.0, rh / 2.0]]
        except (ValueError, TypeError):
            pass

    # 2. Ontology typical dimensions
    ontology = landmark.get("ontology") or {}
    dims = ontology.get("dimensions") or {}
    if not dims:
        try:
            from ontology_knowledge import OntologyKnowledgeBase

            kb = OntologyKnowledgeBase()
            dims = kb.resolve(landmark["class_name"]).get("dimensions") or {}
        except Exception:
            dims = {}

    width = dims.get("width")
    if width is None and dims.get("min_width") is not None and dims.get("max_width") is not None:
        width = (dims["min_width"] + dims["max_width"]) / 2.0

    depth = dims.get("depth")
    if depth is None and dims.get("min_depth") is not None and dims.get("max_depth") is not None:
        depth = (dims["min_depth"] + dims["max_depth"]) / 2.0

    height = dims.get("height")
    if height is None and dims.get("min_height") is not None and dims.get("max_height") is not None:
        height = (dims["min_height"] + dims["max_height"]) / 2.0

    if (width is None or height is None) and size_lookup:
        extent = size_lookup(landmark["class_name"])
        if extent is not None:
            if len(extent) == 3:
                w, d, h = extent
                width = width or w
                depth = depth or d
                height = height or h
            elif len(extent) == 2:
                w, h = extent
                width = width or w
                height = height or h

    w = float(width) if (width is not None and width > 0) else DEFAULT_BOX_HALF_SIZE_M * 2.0
    d = float(depth) if (depth is not None and depth > 0) else w
    h = float(height) if (height is not None and height > 0) else DEFAULT_BOX_HALF_SIZE_M * 2.0

    return [[d / 2.0, w / 2.0, h / 2.0]]


class AnnotationManager:
    """Manages class ID mappings and Rerun AnnotationContext."""

    def __init__(self, initial_classes: list[str] | None = None) -> None:
        self.class_to_id: dict[str, int] = {}
        self.id_to_name: dict[int, str] = {0: "Background"}
        self._next_id = 1
        if initial_classes:
            for name in initial_classes:
                self.get_id(name)

    def get_id(self, class_name: str) -> int:
        clean_name = str(class_name).strip()
        if not clean_name:
            return 0
        if clean_name not in self.class_to_id:
            cid = self._next_id
            self._next_id += 1
            self.class_to_id[clean_name] = cid
            self.id_to_name[cid] = clean_name
        return self.class_to_id[clean_name]

    def get_name(self, class_id: int) -> str:
        return self.id_to_name.get(class_id, "unknown")

    def build_context(self) -> rr.AnnotationContext:
        entries = [(0, "Background", (160, 160, 160))]
        for name, cid in self.class_to_id.items():
            entries.append((cid, name, class_color(name)))
        return rr.AnnotationContext(entries)


def get_default_annotation_manager(classes: list[str] | None = None) -> AnnotationManager:
    initial: list[str] = list(classes) if classes else []
    try:
        from ontology_knowledge import CLASS_ALIASES, OntologyKnowledgeBase, local_name
        from rdflib import OWL, RDF

        kb = OntologyKnowledgeBase()
        onto_classes = sorted({local_name(s) for s in kb.graph.subjects(RDF.type, OWL.Class) if "#" in str(s)})
        for name in onto_classes:
            if name not in initial:
                initial.append(name)
        for alias in sorted(CLASS_ALIASES.keys()):
            if alias not in initial:
                initial.append(alias)
    except Exception:
        pass
    return AnnotationManager(initial)


def log_landmark_entities(landmarks, size_lookup=None) -> None:
    items = landmarks.items() if isinstance(landmarks, dict) else (
        (landmark.get("landmark_id"), landmark) for landmark in landmarks
    )
    for landmark_id, landmark in items:
        center = np.asarray([[landmark["X"], landmark["Y"], landmark["Z"]]], dtype=np.float32)
        label = f"{landmark['class_name']} {landmark['instance_id']}"
        color = [class_color(landmark["class_name"])]
        half_size = resolve_box_half_size(landmark, size_lookup=size_lookup)

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
        self.annotation_manager = get_default_annotation_manager()

        rr.init(application_id, spawn=spawn_viewer)
        recording_path.parent.mkdir(parents=True, exist_ok=True)
        rr.save(str(recording_path))
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rr.log("world", self.annotation_manager.build_context(), static=True)

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
        detections: list[dict] | None = None,
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

        annotated_rgb = rgb_bgr.copy()
        for detection in detections or []:
            x1, y1, x2, y2 = (int(round(value)) for value in detection["bbox_xyxy"])
            if x2 <= x1 or y2 <= y1:
                continue
            class_name = str(detection["class_name"])
            confidence = float(detection.get("confidence") or 0.0)
            color = tuple(int(channel) for channel in class_color(class_name))
            label = f"{class_name} {confidence:.0%}"
            cv2.rectangle(annotated_rgb, (x1, y1), (x2, y2), color, 2)
            text_y = y1 - 7 if y1 >= 22 else y1 + 18
            cv2.putText(annotated_rgb, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2, cv2.LINE_AA)
        rr.log("world/camera/annotated_image", rr.Image(cv2.cvtColor(annotated_rgb, cv2.COLOR_BGR2RGB)))

        if frame_index % self.cloud_every_n_frames == 0:
            self._log_cloud(rgb_bgr, depth_mm, intrinsics, pose_matrix, detections=detections)

    def _log_cloud(
        self,
        rgb_bgr: np.ndarray,
        depth_mm: np.ndarray,
        intrinsics,
        pose_matrix,
        detections: list[dict] | None = None,
    ) -> None:
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
        u_pts = grid_u[valid]
        v_pts = grid_v[valid]
        x = (u_pts - intrinsics.cx) * z / intrinsics.fx
        y = (v_pts - intrinsics.cy) * z / intrinsics.fy

        points_cam = np.stack([x, y, z, np.ones_like(z)], axis=0)
        points_world = (np.asarray(pose_matrix, dtype=np.float64) @ points_cam)[:3].T

        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)[::step, ::step]
        colors = rgb[valid]

        class_ids = np.zeros(len(z), dtype=np.uint16)
        if detections:
            for det in detections:
                if det.get("reject_reason"):
                    continue
                bbox = det.get("bbox_xyxy")
                if not bbox:
                    continue
                x1, y1, x2, y2 = bbox
                class_name = str(det.get("class_name", ""))
                cid = self.annotation_manager.get_id(class_name)

                in_box = (u_pts >= x1) & (u_pts <= x2) & (v_pts >= y1) & (v_pts <= y2)
                if not np.any(in_box):
                    continue

                det_z = det.get("depth_m")
                if det_z is not None and det_z > 0:
                    in_depth = np.abs(z - det_z) <= 0.40
                    mask = in_box & in_depth
                else:
                    box_z = z[in_box]
                    if len(box_z) > 0:
                        med_z = float(np.median(box_z))
                        mask = in_box & (np.abs(z - med_z) <= 0.40)
                    else:
                        mask = in_box

                class_ids[mask] = cid

        rr.log(
            f"world/map/cloud_{self.cloud_chunks:05d}",
            rr.Points3D(
                points_world.astype(np.float32),
                colors=colors,
                radii=self.cloud_point_radius_m,
                class_ids=class_ids,
            ),
            static=True,
        )

        semantic_mask = class_ids > 0
        if np.any(semantic_mask):
            sem_pts = points_world[semantic_mask].astype(np.float32)
            sem_cids = class_ids[semantic_mask]
            sem_colors = np.array([class_color(self.annotation_manager.get_name(cid)) for cid in sem_cids], dtype=np.uint8)
            rr.log(
                f"world/map/semantic_cloud_{self.cloud_chunks:05d}",
                rr.Points3D(
                    sem_pts,
                    colors=sem_colors,
                    radii=self.cloud_point_radius_m * 1.25,
                    class_ids=sem_cids,
                ),
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
