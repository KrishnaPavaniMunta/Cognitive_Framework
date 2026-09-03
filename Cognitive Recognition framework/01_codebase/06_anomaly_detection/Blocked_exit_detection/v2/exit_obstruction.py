"""V2 blocked-exit monitor used by the semantic-map bag replay.

V1 remains the detector authority: this module reuses its YOLO, DINO, SAM, and
RGB-D mask-intersection implementation. V2 owns only the semantic-map-facing
per-frame result and the operator video overlay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
V1_DIR = BASE_DIR.parent / "v1"
OBJECT_DETECTION_DIR = BASE_DIR.parents[2] / "07_object_detection"
KEEP_CLEAR_ARC_SAMPLES = 32


@dataclass
class ExitObstructionResult:
    door_camera_xyz: tuple[float, float, float] | None = None
    door_top_y: float | None = None
    door_bottom_y: float | None = None
    door_mask_distortion: float | None = None
    door_shape_obstruction: bool = False
    floor_y: float | None = None
    floor_world_z: float | None = None
    door_confirmed: bool = False
    zone_strips_world: list[list[list[float]]] = field(default_factory=list)
    blockers: list[dict[str, Any]] = field(default_factory=list)

    @property
    def obstruction_flag(self) -> bool:
        return self.door_shape_obstruction or bool(self.blockers)


def point_inside_keep_clear_zone(
    point_xyz: tuple[float, float, float],
    door_xyz: tuple[float, float, float],
    radius_m: float,
    door_top_y: float,
    door_bottom_y: float,
) -> bool:
    """Return whether a camera-frame point lies in the front door half-cylinder."""
    point_x, point_y, point_z = point_xyz
    door_x, _, door_z = door_xyz
    if point_y < min(door_top_y, door_bottom_y) or point_y > max(door_top_y, door_bottom_y):
        return False
    delta_x = point_x - door_x
    delta_z = point_z - door_z
    return delta_z <= 0.0 and (delta_x * delta_x + delta_z * delta_z) <= radius_m * radius_m


def mask_intersects_keep_clear_zone(
    mask: np.ndarray,
    depth_mm: np.ndarray,
    intrinsics,
    door_xyz: tuple[float, float, float],
    radius_m: float,
    door_top_y: float,
    door_bottom_y: float,
) -> bool:
    """Vectorized depth-mask intersection matching the v1 geometric rule."""
    valid_mask = (mask > 0) & (depth_mm > 0)
    rows, columns = np.where(valid_mask)
    if rows.size == 0:
        return False

    depth_m = depth_mm[rows, columns].astype(np.float32) / 1000.0
    x = (columns.astype(np.float32) - float(intrinsics.cx)) * depth_m / float(intrinsics.fx)
    y = (rows.astype(np.float32) - float(intrinsics.cy)) * depth_m / float(intrinsics.fy)
    door_x, _, door_z = door_xyz
    delta_x = x - float(door_x)
    delta_z = depth_m - float(door_z)
    inside = (
        (y >= min(door_top_y, door_bottom_y))
        & (y <= max(door_top_y, door_bottom_y))
        & (delta_z <= 0.0)
        & ((delta_x * delta_x + delta_z * delta_z) <= radius_m * radius_m)
    )
    return bool(np.any(inside))


def world_keep_clear_strips(result: ExitObstructionResult, pose_matrix, radius_m: float) -> list[list[list[float]]]:
    """Return sampled keep-clear wireframe strips transformed into world coordinates."""
    if result.zone_strips_world:
        return result.zone_strips_world
    if result.door_camera_xyz is None or result.door_top_y is None or result.door_bottom_y is None:
        return []
    x_door, _, z_door = result.door_camera_xyz
    matrix = np.asarray(pose_matrix, dtype=np.float64)
    floor_camera_y = result.floor_y if result.floor_y is not None else result.door_bottom_y
    floor_world = (matrix @ np.asarray([x_door, floor_camera_y, z_door, 1.0]))[:3]
    floor_world[2] = result.floor_world_z if result.floor_world_z is not None else matrix[2, 3]
    right = matrix[:3, :3] @ np.asarray([1.0, 0.0, 0.0])
    forward = matrix[:3, :3] @ np.asarray([0.0, 0.0, 1.0])
    right[2] = 0.0
    forward[2] = 0.0
    right /= max(np.linalg.norm(right), 1e-6)
    forward /= max(np.linalg.norm(forward), 1e-6)
    door_height = abs(result.door_bottom_y - result.door_top_y)
    bottom = []
    top = []
    for index in range(KEEP_CLEAR_ARC_SAMPLES + 1):
        theta = np.pi * index / KEEP_CLEAR_ARC_SAMPLES
        offset = radius_m * np.cos(theta) * right - radius_m * np.sin(theta) * forward
        bottom.append((floor_world + offset).tolist())
        top.append((floor_world + offset + np.asarray([0.0, 0.0, door_height])).tolist())
    bottom_world = np.asarray(bottom, dtype=np.float64)
    top_world = np.asarray(top, dtype=np.float64)
    strips = [bottom_world, top_world]
    strips.extend(np.stack([bottom_world[index], top_world[index]]) for index in (0, KEEP_CLEAR_ARC_SAMPLES))
    strips.extend(np.stack([bottom_world[index], top_world[index]]) for index in range(0, KEEP_CLEAR_ARC_SAMPLES + 1, 4))
    return [[[float(value) for value in point] for point in strip] for strip in strips]


class ExitObstructionMonitor:
    """Stateful V1 detector bridge for one synchronized semantic-map frame stream."""

    def __init__(self, radius_m: float, use_sam: bool = False, min_door_hits: int = 3) -> None:
        if radius_m <= 0.0:
            raise ValueError("radius_m must be positive")
        self.radius_m = float(radius_m)
        self.use_sam = bool(use_sam)
        self.min_door_hits = max(1, int(min_door_hits))
        self._legacy = None
        self._models: dict[str, Any] | None = None
        self._device = ""
        self._door_state: dict[str, Any] = {}
        self._object_state: dict[str, Any] = {}
        self._held_door_boxes: list[tuple] = []
        self._held_door_age = 10**9
        self._door_hits = 0
        self._candidate_geometry: tuple | None = None
        self._stable_geometry: tuple | None = None
        self._world_anchor: dict[str, Any] | None = None
        self._floor_estimator = None

    def load(self) -> None:
        """Load the v1 YOLO+DINO+SAM stack once, only when monitoring is enabled."""
        if self._models is not None:
            return
        if str(V1_DIR) not in sys.path:
            sys.path.insert(0, str(V1_DIR))
        if str(OBJECT_DETECTION_DIR) not in sys.path:
            sys.path.insert(0, str(OBJECT_DETECTION_DIR))
        import Obstruction_detection as legacy
        try:
            from rgbd_3d_filter import estimate_floor_plane
            self._floor_estimator = estimate_floor_plane
        except Exception:
            self._floor_estimator = None

        if not legacy.V1_PATH.exists() or not legacy.V3_PATH.exists():
            raise FileNotFoundError("V1 blocked-exit YOLO weights are unavailable")

        device = legacy.det._assert_gpu()
        v1 = legacy.det.YOLO(str(legacy.V1_PATH))
        v1.to(device)
        v3 = legacy.det.YOLO(str(legacy.V3_PATH))
        v3.to(device)
        processor = legacy.det.AutoProcessor.from_pretrained(legacy.det.DINO_MODEL_ID)
        dino = legacy.det.AutoModelForZeroShotObjectDetection.from_pretrained(legacy.det.DINO_MODEL_ID).to(device)
        dino.eval()
        sam = None
        if self.use_sam:
            sam = legacy.det.GroundedSAMRefiner(
                ckpt_path=legacy.det.SAM_CKPT_PATH,
                model_type="vit_h",
                device=device,
            )
        self._legacy = legacy
        self._models = {"v1": v1, "v3": v3, "proc": processor, "dino": dino, "sam": sam}
        self._device = device
        self._door_state = {"held_doors": [], "held_age": legacy.HOLD_FRAMES + 1}
        self._object_state = {"held_objs": [], "held_age": legacy.HOLD_FRAMES + 1}
        self._held_door_boxes = []
        self._held_door_age = legacy.HOLD_FRAMES + 1

    def set_world_anchor(self, pose_matrix, result: ExitObstructionResult) -> None:
        """Freeze the confirmed keep-clear zone in the semantic-map world frame."""
        if self._world_anchor is not None or result.door_camera_xyz is None:
            return
        matrix = np.asarray(pose_matrix, dtype=np.float64)
        floor_camera_y = result.floor_y if result.floor_y is not None else result.door_bottom_y
        floor_world = (matrix @ np.asarray([result.door_camera_xyz[0], floor_camera_y, result.door_camera_xyz[2], 1.0]))[:3]
        floor_world[2] = matrix[2, 3]
        door_height = abs(result.door_bottom_y - result.door_top_y)
        self._world_anchor = {
            "center": np.asarray((matrix @ np.asarray([*result.door_camera_xyz, 1.0]))[:3], dtype=np.float64),
            "rotation": np.asarray(pose_matrix, dtype=np.float64)[:3, :3],
            "top_z": float(floor_world[2] + door_height),
            "bottom_z": float(floor_world[2]),
        }
        result.floor_world_z = float(floor_world[2])
        result.zone_strips_world = world_keep_clear_strips(result, pose_matrix, self.radius_m)
        self._world_anchor["strips"] = result.zone_strips_world

    def _floor_y(self, depth_mm: np.ndarray, intrinsics, point_xyz: tuple[float, float, float]) -> float | None:
        if self._floor_estimator is None:
            return None
        plane = self._floor_estimator(depth_mm, intrinsics)
        if plane is None:
            return None
        normal, offset = plane
        if abs(float(normal[1])) < 1e-6:
            return None
        return float(-(normal[0] * point_xyz[0] + normal[2] * point_xyz[2] + offset) / normal[1])

    def evaluate(self, rgb_bgr: np.ndarray, depth_mm: np.ndarray, intrinsics, frame_index: int, pose_matrix=None) -> ExitObstructionResult:
        """Evaluate one synchronized RGB-D frame using the archived v1 detector stack."""
        self.load()
        assert self._legacy is not None and self._models is not None
        legacy = self._legacy
        height, width = rgb_bgr.shape[:2]
        if self.use_sam:
            doors = legacy.dz._detect_door_mask(
                rgb_bgr,
                frame_index,
                {"yolo": self._models["v3"], "proc": self._models["proc"], "dino": self._models["dino"], "sam": self._models["sam"]},
                self._door_state,
                self._device,
            )
            objects = legacy._detect_obstruction_masks(
                rgb_bgr, frame_index, self._models, self._object_state, self._device
            )
        else:
            door_boxes, _ = legacy.det.yolo_detect(self._models["v3"], rgb_bgr, self._device)
            object_boxes = legacy._yolo_non_door_boxes(
                self._models["v1"], self._models["v3"], rgb_bgr, self._device
            )
            if frame_index > 1 and (frame_index - 1) % legacy.DINO_INTERVAL == 0:
                dino_doors, _ = legacy.det.dino_detect(
                    self._models["proc"], self._models["dino"], rgb_bgr, self._device
                )
                door_boxes = legacy.det._nms_merge(door_boxes + dino_doors, iou_thr=0.30)
                dino_objects = legacy._dino_non_door_boxes(
                    self._models["proc"], self._models["dino"], rgb_bgr, self._device
                )
                object_boxes = legacy.det._nms_merge(object_boxes + dino_objects, iou_thr=0.30)
            if door_boxes:
                self._held_door_boxes = door_boxes[:2]
                self._held_door_age = 0
            else:
                self._held_door_age += 1
            doors = self._held_door_boxes if self._held_door_age <= legacy.HOLD_FRAMES else []
            objects = object_boxes
        result = ExitObstructionResult()
        detected_geometry = None

        if doors:
            door = max(doors, key=lambda item: item[4])
            door_mask = legacy.dz._mask_from_det(door, (height, width))
            if door_mask is not None:
                result.door_mask_distortion = legacy._door_mask_distortion(door_mask)
                result.door_shape_obstruction = bool(
                    result.door_mask_distortion is not None
                    and result.door_mask_distortion > legacy.DOOR_MASK_DISTORTION_THRESHOLD
                )
                centroid = legacy.dz._mask_centroid(door_mask)
                depth_m = legacy.dz._median_depth_m(depth_mm, door_mask)
                if centroid is not None and depth_m is not None and depth_m > 0.0:
                    center_x, center_y, center_z = legacy.dz._backproject(*centroid, depth_m, intrinsics)
                    _, top_y, _ = legacy.dz._backproject(centroid[0], float(door[1]), depth_m, intrinsics)
                    _, bottom_y, _ = legacy.dz._backproject(centroid[0], float(door[3]), depth_m, intrinsics)
                    door_height = abs(bottom_y - top_y)
                    floor_y = self._floor_y(depth_mm, intrinsics, (center_x, center_y, center_z))
                    bottom_y = floor_y if floor_y is not None else center_y + door_height / 2.0
                    top_y = bottom_y - door_height
                    detected_geometry = ((center_x, center_y, center_z), top_y, bottom_y, floor_y)

        if detected_geometry is not None:
            if self._candidate_geometry is not None:
                previous_center = np.asarray(self._candidate_geometry[0])
                current_center = np.asarray(detected_geometry[0])
                consistent = (
                    float(np.linalg.norm(previous_center - current_center)) <= 0.35
                    and abs(float(self._candidate_geometry[2]) - float(detected_geometry[2])) <= 0.45
                )
            else:
                consistent = False
            self._door_hits = self._door_hits + 1 if consistent else 1
            self._candidate_geometry = detected_geometry
            if self._door_hits >= self.min_door_hits:
                self._stable_geometry = detected_geometry
        elif self._stable_geometry is None:
            self._door_hits = 0

        geometry = self._stable_geometry
        if geometry is None:
            return result
        result.door_camera_xyz, result.door_top_y, result.door_bottom_y, result.floor_y = geometry
        result.door_confirmed = True

        if pose_matrix is not None and self._world_anchor is None:
            self.set_world_anchor(pose_matrix, result)
        if self._world_anchor is not None:
            result.zone_strips_world = self._world_anchor["strips"]

        for object_index, detection in enumerate(objects, start=1):
            object_mask = legacy.dz._mask_from_det(detection, (height, width))
            if object_mask is None:
                continue
            if self._world_anchor is not None and pose_matrix is not None:
                blocked = self._mask_intersects_world_anchor(
                    object_mask, depth_mm, intrinsics, pose_matrix
                )
            else:
                blocked = mask_intersects_keep_clear_zone(
                    object_mask,
                    depth_mm,
                    intrinsics,
                    result.door_camera_xyz,
                    self.radius_m,
                    result.door_top_y,
                    result.door_bottom_y,
                )
            if not blocked:
                continue
            depth_m = legacy.dz._median_depth_m(depth_mm, object_mask)
            result.blockers.append(
                {
                    "object_index": object_index,
                    "bbox_xyxy": [float(value) for value in detection[:4]],
                    "confidence": float(detection[4]),
                    "depth_m": depth_m,
                    "mask": object_mask,
                }
            )
        return result

    def _mask_intersects_world_anchor(self, mask: np.ndarray, depth_mm: np.ndarray, intrinsics, pose_matrix) -> bool:
        """Test current object depth points against the fixed world-frame zone."""
        ys, xs = np.where((mask > 0) & (depth_mm > 0))
        if ys.size == 0 or self._world_anchor is None:
            return False
        depth_m = depth_mm[ys, xs].astype(np.float64) / 1000.0
        cam = np.stack([
            (xs - float(intrinsics.cx)) * depth_m / float(intrinsics.fx),
            (ys - float(intrinsics.cy)) * depth_m / float(intrinsics.fy),
            depth_m,
            np.ones_like(depth_m),
        ])
        world = (np.asarray(pose_matrix, dtype=np.float64) @ cam)[:3].T
        anchor = self._world_anchor
        relative = world - anchor["center"]
        forward = anchor["rotation"] @ np.asarray([0.0, 0.0, 1.0])
        forward[2] = 0.0
        norm = np.linalg.norm(forward)
        if norm <= 1e-6:
            return False
        forward /= norm
        horizontal = relative.copy()
        horizontal[:, 2] = 0.0
        return bool(np.any(
            (world[:, 2] >= min(anchor["top_z"], anchor["bottom_z"]))
            & (world[:, 2] <= max(anchor["top_z"], anchor["bottom_z"]))
            & ((horizontal @ forward) <= 0.0)
            & (np.sum(horizontal * horizontal, axis=1) <= self.radius_m ** 2)
        ))

    def draw_overlay(self, frame: np.ndarray, result: ExitObstructionResult, intrinsics, pose_matrix=None) -> np.ndarray:
        """Draw the keep-clear zone, blockers, and operator-facing alert on a video frame."""
        image = frame.copy()
        if result.zone_strips_world and pose_matrix is not None:
            camera_from_world = np.linalg.inv(np.asarray(pose_matrix, dtype=np.float64))
            projected_strips = []
            for strip in result.zone_strips_world:
                pixels = []
                for point in strip:
                    camera_point = camera_from_world @ np.asarray([*point, 1.0], dtype=np.float64)
                    pixel = self._legacy.dz._project(*camera_point[:3], intrinsics)
                    if pixel is not None:
                        pixels.append(pixel)
                if len(pixels) >= 2:
                    projected_strips.append(np.asarray(pixels, dtype=np.int32))
            if projected_strips:
                overlay = image.copy()
                color = (0, 0, 220) if result.obstruction_flag else (0, 180, 60)
                for strip in projected_strips:
                    cv2.polylines(overlay, [strip], False, color, 2)
                cv2.addWeighted(overlay, 0.15, image, 0.85, 0.0, image)
                for strip in projected_strips:
                    cv2.polylines(image, [strip], False, color, 2)
        elif result.door_camera_xyz is not None and result.door_top_y is not None and result.door_bottom_y is not None:
            image = self._legacy.dz._draw_zone(
                image,
                result.door_camera_xyz,
                self.radius_m,
                result.door_bottom_y,
                result.door_top_y,
                intrinsics,
            )
        for blocker in result.blockers:
            x1, y1, x2, y2 = (int(round(value)) for value in blocker["bbox_xyxy"])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(image, "BLOCKING EXIT", (x1, max(22, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
        if result.obstruction_flag:
            cv2.rectangle(image, (0, 30), (image.shape[1], 72), (0, 0, 180), -1)
            cv2.putText(image, "EXIT BLOCKED", (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
        return image
