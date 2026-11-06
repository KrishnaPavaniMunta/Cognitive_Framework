from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


DEFAULT_RGB_TOPIC = "/camera/rgb/image_rect_color"
DEFAULT_DEPTH_TOPIC = "/camera/depth_registered/image_raw"
DEFAULT_CAMERA_INFO_TOPIC = "/camera/rgb/camera_info"
DEFAULT_TF_TOPIC = "/tf"
DEFAULT_ODOM_TOPIC = "/odom"
DEFAULT_DETECTION_TOPIC = "/simulated_detections"
DEFAULT_DETECTION_BACKEND = "topic"


@dataclass
class TimedMessage:
    timestamp_ns: int
    message: Any


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion_model: str
    d: list[float]


@dataclass
class ExtrinsicsRecord:
    source: str
    frame_id: str
    child_frame_id: str
    timestamp_ns: int
    matrix_4x4: list[list[float]]
    translation_xyz: list[float]
    quaternion_xyzw: list[float]


class YoloEnsembleDinoDetector:
    def __init__(
        self,
        module_path: Path,
        device: str,
        use_dino: bool,
        apply_depth_gate: bool,
        v1_path: str,
        v2_path: str,
        v3_path: str,
    ) -> None:
        self.module_path = module_path
        self.device = device
        self.use_dino = use_dino
        self.apply_depth_gate = apply_depth_gate
        self.v1_path = v1_path
        self.v2_path = v2_path
        self.v3_path = v3_path

        self.module = None
        self.v1 = None
        self.v2 = None
        self.v3 = None

    def load(self) -> None:
        module_dir = str(self.module_path.parent)
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)

        spec = importlib.util.spec_from_file_location("yolo_ensemble_dino_module", str(self.module_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load module from: {self.module_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.module = module

        try:
            import torch  # noqa: PLC0415
        except Exception as exc:
            raise RuntimeError(f"torch is required for YOLO inference mode: {exc}") from exc

        yolo_cls = getattr(module, "YOLO", None)
        if yolo_cls is None:
            raise RuntimeError("YOLO class not found in YOLO ensemble module")

        actual_device = self.device
        if actual_device == "cuda" and not torch.cuda.is_available():
            print("[WARN] CUDA requested but not available. Falling back to CPU.")
            actual_device = "cpu"
        self.device = actual_device

        print(f"[INFO] Loading YOLO ensemble models on device: {self.device}")
        self.v1 = yolo_cls(str(self.v1_path)).to(self.device)
        self.v2 = yolo_cls(str(self.v2_path)).to(self.device)
        self.v3 = yolo_cls(str(self.v3_path)).to(self.device)

    def infer(
        self,
        rgb_img: np.ndarray,
        depth_img_mm: np.ndarray,
        intrinsics: CameraIntrinsics,
    ) -> list[dict[str, Any]]:
        if self.module is None or self.v1 is None or self.v2 is None or self.v3 is None:
            raise RuntimeError("Detector is not loaded")

        module = self.module
        preds = module.run_yolo_ensemble(self.v1, self.v2, self.v3, rgb_img)

        if self.use_dino:
            from PIL import Image  # noqa: PLC0415

            pil_img = Image.fromarray(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
            dino_seen_conf = float(getattr(module, "DINO_SEEN_CONF_THRESH", 0.45))
            dino_fallback = getattr(module, "DINO_FALLBACK", {})
            seen_classes = {p[5].replace("[DINO] ", "") for p in preds if float(p[4]) >= dino_seen_conf}
            missing_targets = [c for c in dino_fallback.keys() if c not in seen_classes]
            if missing_targets:
                dino_preds = module.run_dino_fallback(pil_img, missing_targets)
                preds.extend(dino_preds)

        iou_thresh = float(getattr(module, "IOU_THRESH", 0.45))
        final = module.apply_global_nms(preds, iou_thresh)
        final = module.apply_common_sense_rules(final, rgb_img.shape[0], rgb_img.shape[1])
        final = module.refine_bin_detections(rgb_img, final)

        if self.apply_depth_gate and hasattr(module, "apply_depth_size_filter"):
            final = module.apply_depth_size_filter(final, depth_img_mm, intrinsics)

        parsed: list[dict[str, Any]] = []
        for x1, y1, x2, y2, conf, label in final:
            class_label = str(label).replace("[DINO] ", "")
            parsed.append(
                {
                    "class_label": class_label,
                    "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(conf),
                    "track_id": None,
                }
            )
        return parsed


def _header_stamp_ns(msg: Any) -> int | None:
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return None
    sec = getattr(stamp, "sec", None)
    nanosec = getattr(stamp, "nanosec", None)
    if sec is None or nanosec is None:
        return None
    return int(sec) * 1_000_000_000 + int(nanosec)


def _message_timestamp_ns(msg: Any, fallback_ns: int) -> int:
    return _header_stamp_ns(msg) or int(fallback_ns)


def _decode_color_image(msg: Any) -> np.ndarray | None:
    height = int(msg.height)
    width = int(msg.width)
    encoding = str(msg.encoding).lower()
    data = np.frombuffer(msg.data, dtype=np.uint8)

    if encoding == "rgb8":
        return cv2.cvtColor(data.reshape((height, width, 3)), cv2.COLOR_RGB2BGR)
    if encoding == "bgr8":
        return data.reshape((height, width, 3)).copy()
    if encoding == "rgba8":
        return cv2.cvtColor(data.reshape((height, width, 4)), cv2.COLOR_RGBA2BGR)
    if encoding == "bgra8":
        return cv2.cvtColor(data.reshape((height, width, 4)), cv2.COLOR_BGRA2BGR)
    if encoding in {"mono8", "8uc1"}:
        return cv2.cvtColor(data.reshape((height, width)), cv2.COLOR_GRAY2BGR)
    return None


def _decode_depth_image_mm(msg: Any) -> np.ndarray | None:
    height = int(msg.height)
    width = int(msg.width)
    encoding = str(msg.encoding).lower()

    if encoding in {"16uc1", "mono16"}:
        return np.frombuffer(msg.data, dtype=np.uint16).reshape((height, width)).copy()

    if encoding == "32fc1":
        arr_m = np.frombuffer(msg.data, dtype=np.float32).reshape((height, width)).copy()
        arr_m = np.nan_to_num(arr_m, nan=0.0, posinf=0.0, neginf=0.0)
        arr_m = np.clip(arr_m, 0.0, 65.535)
        return (arr_m * 1000.0).astype(np.uint16)

    return None


def _intrinsics_from_camera_info(msg: Any) -> CameraIntrinsics:
    k = [float(v) for v in msg.k]
    d = [float(v) for v in msg.d]
    return CameraIntrinsics(
        fx=k[0],
        fy=k[4],
        cx=k[2],
        cy=k[5],
        width=int(msg.width),
        height=int(msg.height),
        distortion_model=str(getattr(msg, "distortion_model", "")),
        d=d,
    )


def _quat_to_matrix_xyzw(x: float, y: float, z: float, w: float) -> list[list[float]]:
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy), 0.0],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx), 0.0],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _pose_to_extrinsics(
    *,
    source: str,
    timestamp_ns: int,
    frame_id: str,
    child_frame_id: str,
    tx: float,
    ty: float,
    tz: float,
    qx: float,
    qy: float,
    qz: float,
    qw: float,
) -> ExtrinsicsRecord:
    matrix = _quat_to_matrix_xyzw(qx, qy, qz, qw)
    matrix[0][3] = tx
    matrix[1][3] = ty
    matrix[2][3] = tz
    return ExtrinsicsRecord(
        source=source,
        frame_id=frame_id,
        child_frame_id=child_frame_id,
        timestamp_ns=int(timestamp_ns),
        matrix_4x4=matrix,
        translation_xyz=[float(tx), float(ty), float(tz)],
        quaternion_xyzw=[float(qx), float(qy), float(qz), float(qw)],
    )


def _closest_index(sorted_timestamps_ns: list[int], target_ns: int, start_idx: int = 0) -> int:
    if not sorted_timestamps_ns:
        return -1
    idx = bisect_left(sorted_timestamps_ns, target_ns, lo=max(0, start_idx))
    if idx == 0:
        return 0
    if idx >= len(sorted_timestamps_ns):
        return len(sorted_timestamps_ns) - 1

    prev_idx = idx - 1
    if abs(sorted_timestamps_ns[prev_idx] - target_ns) <= abs(sorted_timestamps_ns[idx] - target_ns):
        return prev_idx
    return idx


def _bbox_xyxy_from_detection(detection: Any) -> list[float] | None:
    if hasattr(detection, "bbox"):
        bbox = detection.bbox
        if hasattr(bbox, "center") and hasattr(bbox, "size_x") and hasattr(bbox, "size_y"):
            cx = float(getattr(bbox.center, "x", 0.0))
            cy = float(getattr(bbox.center, "y", 0.0))
            w = float(getattr(bbox, "size_x", 0.0))
            h = float(getattr(bbox, "size_y", 0.0))
            return [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h]

    candidate_attrs = [
        ("xmin", "ymin", "xmax", "ymax"),
        ("x_min", "y_min", "x_max", "y_max"),
        ("left", "top", "right", "bottom"),
        ("x1", "y1", "x2", "y2"),
    ]
    for attrs in candidate_attrs:
        if all(hasattr(detection, name) for name in attrs):
            x1, y1, x2, y2 = (float(getattr(detection, name)) for name in attrs)
            return [x1, y1, x2, y2]

    return None


def _label_and_score_from_detection(detection: Any) -> tuple[str, float | None, int | None]:
    label = ""
    score: float | None = None
    track_id: int | None = None

    if hasattr(detection, "id"):
        raw_id = getattr(detection, "id")
        if raw_id is not None:
            label = str(raw_id)

    if hasattr(detection, "class_id"):
        class_id = getattr(detection, "class_id")
        if class_id is not None:
            label = str(class_id)

    if hasattr(detection, "Class"):
        class_name = getattr(detection, "Class")
        if class_name is not None:
            label = str(class_name)

    if hasattr(detection, "label"):
        class_label = getattr(detection, "label")
        if class_label is not None:
            label = str(class_label)

    if hasattr(detection, "probability"):
        try:
            score = float(getattr(detection, "probability"))
        except Exception:
            score = None

    if hasattr(detection, "score"):
        try:
            score = float(getattr(detection, "score"))
        except Exception:
            pass

    if hasattr(detection, "track_id"):
        try:
            track_id = int(getattr(detection, "track_id"))
        except Exception:
            pass

    # vision_msgs/Detection2D style
    results = getattr(detection, "results", None)
    if isinstance(results, list) and results:
        top = results[0]
        hypothesis = getattr(top, "hypothesis", None)
        if hypothesis is not None:
            class_id = getattr(hypothesis, "class_id", None)
            if class_id is not None:
                label = str(class_id)
            if hasattr(hypothesis, "score"):
                try:
                    score = float(getattr(hypothesis, "score"))
                except Exception:
                    pass
        if hasattr(top, "score"):
            try:
                score = float(getattr(top, "score"))
            except Exception:
                pass

    if not label:
        label = "unknown"

    return label, score, track_id


def _parse_detections_message(msg: Any, image_width: int, image_height: int) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []

    # Common patterns: vision_msgs Detection2DArray, darknet_msgs BoundingBoxes, custom arrays
    if hasattr(msg, "detections") and isinstance(msg.detections, list):
        detections = msg.detections
    elif hasattr(msg, "bounding_boxes") and isinstance(msg.bounding_boxes, list):
        detections = msg.bounding_boxes
    elif hasattr(msg, "boxes") and isinstance(msg.boxes, list):
        detections = msg.boxes
    elif hasattr(msg, "objects") and isinstance(msg.objects, list):
        detections = msg.objects
    else:
        detections = []

    for det in detections:
        bbox = _bbox_xyxy_from_detection(det)
        if bbox is None:
            continue

        label, score, track_id = _label_and_score_from_detection(det)
        x1, y1, x2, y2 = bbox

        # Clamp while preserving ordering.
        x1 = float(max(0.0, min(x1, image_width - 1.0)))
        y1 = float(max(0.0, min(y1, image_height - 1.0)))
        x2 = float(max(0.0, min(x2, image_width - 1.0)))
        y2 = float(max(0.0, min(y2, image_height - 1.0)))
        if x2 <= x1 or y2 <= y1:
            continue

        parsed.append(
            {
                "class_label": str(label),
                "bbox_xyxy": [x1, y1, x2, y2],
                "confidence": score,
                "track_id": track_id,
            }
        )

    # Custom schema fallback: arrays on the top-level message.
    if not parsed and hasattr(msg, "class_labels") and hasattr(msg, "bboxes_xyxy"):
        labels = list(getattr(msg, "class_labels"))
        boxes = list(getattr(msg, "bboxes_xyxy"))
        scores = list(getattr(msg, "scores", []))
        n = min(len(labels), len(boxes))
        for i in range(n):
            box = boxes[i]
            if len(box) != 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            x1 = float(max(0.0, min(x1, image_width - 1.0)))
            y1 = float(max(0.0, min(y1, image_height - 1.0)))
            x2 = float(max(0.0, min(x2, image_width - 1.0)))
            y2 = float(max(0.0, min(y2, image_height - 1.0)))
            if x2 <= x1 or y2 <= y1:
                continue
            score = float(scores[i]) if i < len(scores) else None
            parsed.append(
                {
                    "class_label": str(labels[i]),
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "confidence": score,
                    "track_id": None,
                }
            )

    return parsed


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _ensure_output_dir(path: Path, overwrite: bool) -> None:
    _ = overwrite
    path.mkdir(parents=True, exist_ok=True)


def _topic_message_map(reader: AnyReader, selected_topics: set[str]) -> tuple[dict[str, list[TimedMessage]], dict[str, int]]:
    topic_messages: dict[str, list[TimedMessage]] = {}
    skipped = {
        "deserialize_errors": 0,
        "stream_errors": 0,
    }

    selected_connections = [c for c in reader.connections if str(c.topic) in selected_topics]
    try:
        for connection, timestamp_ns, raw_data in reader.messages(connections=selected_connections):
            topic = str(connection.topic)
            try:
                message = reader.deserialize(raw_data, connection.msgtype)
            except Exception:
                skipped["deserialize_errors"] += 1
                continue

            msg_ts = _message_timestamp_ns(message, int(timestamp_ns))
            topic_messages.setdefault(topic, []).append(TimedMessage(msg_ts, message))
    except Exception:
        skipped["stream_errors"] += 1

    for topic, entries in topic_messages.items():
        entries.sort(key=lambda item: item.timestamp_ns)
        topic_messages[topic] = entries
    return topic_messages, skipped


def _extract_tf_records(tf_messages: list[TimedMessage]) -> dict[str, list[ExtrinsicsRecord]]:
    per_child: dict[str, list[ExtrinsicsRecord]] = {}
    for item in tf_messages:
        tf_msg = item.message
        transforms = getattr(tf_msg, "transforms", [])
        if not isinstance(transforms, list):
            continue
        for tr in transforms:
            header = getattr(tr, "header", None)
            frame_id = str(getattr(header, "frame_id", ""))
            child_frame_id = str(getattr(tr, "child_frame_id", ""))
            transform = getattr(tr, "transform", None)
            if transform is None:
                continue
            trans = getattr(transform, "translation", None)
            rot = getattr(transform, "rotation", None)
            if trans is None or rot is None:
                continue

            ts_ns = _header_stamp_ns(tr)
            if ts_ns is None:
                ts_ns = item.timestamp_ns

            record = _pose_to_extrinsics(
                source="tf",
                timestamp_ns=ts_ns,
                frame_id=frame_id,
                child_frame_id=child_frame_id,
                tx=float(getattr(trans, "x", 0.0)),
                ty=float(getattr(trans, "y", 0.0)),
                tz=float(getattr(trans, "z", 0.0)),
                qx=float(getattr(rot, "x", 0.0)),
                qy=float(getattr(rot, "y", 0.0)),
                qz=float(getattr(rot, "z", 0.0)),
                qw=float(getattr(rot, "w", 1.0)),
            )
            per_child.setdefault(child_frame_id, []).append(record)

    for child_frame_id, records in per_child.items():
        records.sort(key=lambda item: item.timestamp_ns)
        per_child[child_frame_id] = records

    return per_child


def _extract_odom_records(odom_messages: list[TimedMessage]) -> list[ExtrinsicsRecord]:
    records: list[ExtrinsicsRecord] = []
    for item in odom_messages:
        msg = item.message
        pose = getattr(msg, "pose", None)
        pose_pose = getattr(pose, "pose", None)
        if pose_pose is None:
            continue
        position = getattr(pose_pose, "position", None)
        orientation = getattr(pose_pose, "orientation", None)
        if position is None or orientation is None:
            continue

        header = getattr(msg, "header", None)
        frame_id = str(getattr(header, "frame_id", ""))
        child_frame_id = str(getattr(msg, "child_frame_id", ""))

        records.append(
            _pose_to_extrinsics(
                source="odom",
                timestamp_ns=item.timestamp_ns,
                frame_id=frame_id,
                child_frame_id=child_frame_id,
                tx=float(getattr(position, "x", 0.0)),
                ty=float(getattr(position, "y", 0.0)),
                tz=float(getattr(position, "z", 0.0)),
                qx=float(getattr(orientation, "x", 0.0)),
                qy=float(getattr(orientation, "y", 0.0)),
                qz=float(getattr(orientation, "z", 0.0)),
                qw=float(getattr(orientation, "w", 1.0)),
            )
        )

    records.sort(key=lambda item: item.timestamp_ns)
    return records


def _choose_extrinsics(
    timestamp_ns: int,
    camera_frame_id: str,
    tf_records_by_child: dict[str, list[ExtrinsicsRecord]],
    odom_records: list[ExtrinsicsRecord],
    max_tf_delta_ns: int,
    max_odom_delta_ns: int,
) -> tuple[dict[str, Any] | None, str]:
    tf_candidates = tf_records_by_child.get(camera_frame_id, [])

    # Fallback: if exact frame not found, allow suffix match (e.g., namespaced frames).
    if not tf_candidates and camera_frame_id:
        for child_name, records in tf_records_by_child.items():
            if child_name.endswith(camera_frame_id):
                tf_candidates = records
                break

    if tf_candidates:
        ts_list = [r.timestamp_ns for r in tf_candidates]
        idx = _closest_index(ts_list, timestamp_ns)
        if idx >= 0:
            rec = tf_candidates[idx]
            if abs(rec.timestamp_ns - timestamp_ns) <= max_tf_delta_ns:
                return (
                    {
                        "source": rec.source,
                        "frame_id": rec.frame_id,
                        "child_frame_id": rec.child_frame_id,
                        "timestamp_ns": rec.timestamp_ns,
                        "translation_xyz": rec.translation_xyz,
                        "quaternion_xyzw": rec.quaternion_xyzw,
                        "matrix_4x4": rec.matrix_4x4,
                    },
                    "ok",
                )

    if odom_records:
        ts_list = [r.timestamp_ns for r in odom_records]
        idx = _closest_index(ts_list, timestamp_ns)
        if idx >= 0:
            rec = odom_records[idx]
            if abs(rec.timestamp_ns - timestamp_ns) <= max_odom_delta_ns:
                return (
                    {
                        "source": rec.source,
                        "frame_id": rec.frame_id,
                        "child_frame_id": rec.child_frame_id,
                        "timestamp_ns": rec.timestamp_ns,
                        "translation_xyz": rec.translation_xyz,
                        "quaternion_xyzw": rec.quaternion_xyzw,
                        "matrix_4x4": rec.matrix_4x4,
                    },
                    "ok_fallback_odom",
                )

    return None, "missing"


def _as_intrinsics_payload(intr: CameraIntrinsics) -> dict[str, Any]:
    return {
        "fx": intr.fx,
        "fy": intr.fy,
        "cx": intr.cx,
        "cy": intr.cy,
        "width": intr.width,
        "height": intr.height,
        "distortion_model": intr.distortion_model,
        "distortion_coeffs": intr.d,
    }


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")


def _is_intrinsics_sane(intr: CameraIntrinsics) -> bool:
    return (
        intr.fx > 0.0
        and intr.fy > 0.0
        and intr.width > 0
        and intr.height > 0
        and math.isfinite(intr.cx)
        and math.isfinite(intr.cy)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture synchronized RGBD + intrinsics/extrinsics + simulated detections from ROS bag."
    )
    parser.add_argument("--bag", required=True, type=str, help="Path to ROS bag folder (sqlite3 or mcap)")
    parser.add_argument("--output-root", required=True, type=str, help="Output directory for captured dataset")

    parser.add_argument("--rgb-topic", type=str, default=DEFAULT_RGB_TOPIC)
    parser.add_argument("--depth-topic", type=str, default=DEFAULT_DEPTH_TOPIC)
    parser.add_argument("--camera-info-topic", type=str, default=DEFAULT_CAMERA_INFO_TOPIC)
    parser.add_argument(
        "--detection-backend",
        type=str,
        default=DEFAULT_DETECTION_BACKEND,
        choices=["topic", "yolo_ensemble_dino"],
        help="Source for detections: ROS topic or local YOLO ensemble+DINO inference",
    )
    parser.add_argument("--detection-topic", type=str, default=DEFAULT_DETECTION_TOPIC)
    parser.add_argument("--tf-topic", type=str, default=DEFAULT_TF_TOPIC)
    parser.add_argument("--odom-topic", type=str, default=DEFAULT_ODOM_TOPIC)
    parser.add_argument(
        "--camera-frame-id",
        type=str,
        default="",
        help="Camera child frame id for tf matching. If empty, inferred from CameraInfo header.frame_id.",
    )

    parser.add_argument("--batch-size", type=int, default=64, help="Synchronized frames per batch")
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Skip the first N synchronized frames before starting capture",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N synchronized frames (0=all)")
    parser.add_argument("--max-rgb-depth-delta-ms", type=float, default=50.0)
    parser.add_argument("--max-det-delta-ms", type=float, default=75.0)
    parser.add_argument("--max-tf-delta-ms", type=float, default=100.0)
    parser.add_argument("--max-odom-delta-ms", type=float, default=100.0)
    parser.add_argument(
        "--detector-module-path",
        type=str,
        default="",
        help="Path to YOLO_ensemble+DINO.py (used when detection backend is yolo_ensemble_dino)",
    )
    parser.add_argument("--detector-device", type=str, default="cuda", help="Detector device: cuda or cpu")
    parser.add_argument("--disable-dino", action="store_true", help="Disable DINO fallback in inference mode")
    parser.add_argument(
        "--disable-depth-gate",
        action="store_true",
        help="Disable depth/size gating in inference mode",
    )
    parser.add_argument("--v1-path", type=str, default="", help="Override v1 model path in inference mode")
    parser.add_argument("--v2-path", type=str, default="", help="Override v2 model path in inference mode")
    parser.add_argument("--v3-path", type=str, default="", help="Override v3 model path in inference mode")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty output folder")
    parser.add_argument("--dry-run", action="store_true", help="Scan and validate without writing frame files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.start_frame < 0:
        raise ValueError("--start-frame must be >= 0")

    bag_path = Path(args.bag).resolve()
    output_root = Path(args.output_root).resolve()
    _ensure_output_dir(output_root, overwrite=bool(args.overwrite))

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    with AnyReader([bag_path], default_typestore=typestore) as reader:
        available_topics = {str(c.topic): str(c.msgtype) for c in reader.connections}

        required_topics = [args.rgb_topic, args.depth_topic, args.camera_info_topic]
        missing_required = [t for t in required_topics if t not in available_topics]
        if missing_required:
            raise RuntimeError(f"Missing required topics in bag: {missing_required}")

        selected_topics = {
            args.rgb_topic,
            args.depth_topic,
            args.camera_info_topic,
            args.tf_topic,
            args.odom_topic,
        }
        if args.detection_backend == "topic":
            selected_topics.add(args.detection_topic)
        topic_messages, skipped_messages = _topic_message_map(reader, selected_topics)

    rgb_messages = topic_messages.get(args.rgb_topic, [])
    depth_messages = topic_messages.get(args.depth_topic, [])
    camera_info_messages = topic_messages.get(args.camera_info_topic, [])
    detection_messages = topic_messages.get(args.detection_topic, []) if args.detection_backend == "topic" else []
    tf_messages = topic_messages.get(args.tf_topic, [])
    odom_messages = topic_messages.get(args.odom_topic, [])

    if not rgb_messages:
        raise RuntimeError(f"No messages found for rgb topic: {args.rgb_topic}")
    if not depth_messages:
        raise RuntimeError(f"No messages found for depth topic: {args.depth_topic}")
    if not camera_info_messages:
        raise RuntimeError(f"No messages found for camera info topic: {args.camera_info_topic}")

    tf_records_by_child = _extract_tf_records(tf_messages)
    odom_records = _extract_odom_records(odom_messages)

    detector: YoloEnsembleDinoDetector | None = None
    if args.detection_backend == "yolo_ensemble_dino":
        default_module = Path(__file__).with_name("YOLO_ensemble+DINO.py")
        module_path = Path(args.detector_module_path).resolve() if args.detector_module_path else default_module
        if not module_path.exists():
            raise FileNotFoundError(f"Detector module not found: {module_path}")

        module_spec = importlib.util.spec_from_file_location("detector_probe", str(module_path))
        if module_spec is None or module_spec.loader is None:
            raise RuntimeError(f"Unable to probe detector module at: {module_path}")
        probe = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(probe)

        v1_path = str(args.v1_path) if args.v1_path else str(getattr(probe, "V1_PATH"))
        v2_path = str(args.v2_path) if args.v2_path else str(getattr(probe, "V2_PATH"))
        v3_path = str(args.v3_path) if args.v3_path else str(getattr(probe, "V3_PATH"))

        detector = YoloEnsembleDinoDetector(
            module_path=module_path,
            device=str(args.detector_device),
            use_dino=not bool(args.disable_dino),
            apply_depth_gate=not bool(args.disable_depth_gate),
            v1_path=v1_path,
            v2_path=v2_path,
            v3_path=v3_path,
        )
        detector.load()

    depth_ts = [m.timestamp_ns for m in depth_messages]
    cam_ts = [m.timestamp_ns for m in camera_info_messages]
    det_ts = [m.timestamp_ns for m in detection_messages] if args.detection_backend == "topic" else []

    max_rgb_depth_delta_ns = int(float(args.max_rgb_depth_delta_ms) * 1_000_000.0)
    max_det_delta_ns = int(float(args.max_det_delta_ms) * 1_000_000.0)
    max_tf_delta_ns = int(float(args.max_tf_delta_ms) * 1_000_000.0)
    max_odom_delta_ns = int(float(args.max_odom_delta_ms) * 1_000_000.0)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / f"capture_{run_id}"
    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Run directory already exists and is not empty: {run_dir}. Use --overwrite to reuse it."
        )
    frames_root = run_dir / "batches"
    metadata_root = run_dir / "metadata"
    if not args.dry_run:
        frames_root.mkdir(parents=True, exist_ok=True)
        metadata_root.mkdir(parents=True, exist_ok=True)

    run_stats = {
        "rgb_messages": len(rgb_messages),
        "depth_messages": len(depth_messages),
        "camera_info_messages": len(camera_info_messages),
        "detection_messages": len(detection_messages),
        "tf_messages": len(tf_messages),
        "odom_messages": len(odom_messages),
        "synced_frames": 0,
        "skipped_no_depth": 0,
        "skipped_depth_delta": 0,
        "skipped_no_intrinsics": 0,
        "frames_missing_extrinsics": 0,
        "frames_without_detection_match": 0,
        "detections_total": 0,
        "batches_written": 0,
        "inference_frames": 0,
        "skipped_deserialize_messages": int(skipped_messages.get("deserialize_errors", 0)),
        "stream_error_events": int(skipped_messages.get("stream_errors", 0)),
    }

    class_histogram: dict[str, int] = {}
    batch_frames: list[dict[str, Any]] = []

    depth_cursor = 0
    cam_cursor = 0
    det_cursor = 0

    matched_frame_counter = 0
    captured_frame_counter = 0

    for rgb_item in rgb_messages:
        if args.max_frames > 0 and captured_frame_counter >= int(args.max_frames):
            break

        rgb_img = _decode_color_image(rgb_item.message)
        if rgb_img is None:
            continue

        depth_idx = _closest_index(depth_ts, rgb_item.timestamp_ns, start_idx=depth_cursor)
        if depth_idx < 0:
            run_stats["skipped_no_depth"] += 1
            continue
        depth_cursor = max(depth_cursor, depth_idx)

        depth_item = depth_messages[depth_idx]
        depth_delta = abs(depth_item.timestamp_ns - rgb_item.timestamp_ns)
        if depth_delta > max_rgb_depth_delta_ns:
            run_stats["skipped_depth_delta"] += 1
            continue

        depth_img_mm = _decode_depth_image_mm(depth_item.message)
        if depth_img_mm is None:
            run_stats["skipped_no_depth"] += 1
            continue

        cam_idx = _closest_index(cam_ts, rgb_item.timestamp_ns, start_idx=cam_cursor)
        if cam_idx < 0:
            run_stats["skipped_no_intrinsics"] += 1
            continue
        cam_cursor = max(cam_cursor, cam_idx)

        intrinsics = _intrinsics_from_camera_info(camera_info_messages[cam_idx].message)
        if not _is_intrinsics_sane(intrinsics):
            run_stats["skipped_no_intrinsics"] += 1
            continue

        # This frame is synchronized and valid across RGB/depth/intrinsics.
        if matched_frame_counter < int(args.start_frame):
            matched_frame_counter += 1
            continue

        camera_frame_id = args.camera_frame_id.strip()
        if not camera_frame_id:
            header = getattr(camera_info_messages[cam_idx].message, "header", None)
            camera_frame_id = str(getattr(header, "frame_id", "")).strip()

        extrinsics, extrinsics_status = _choose_extrinsics(
            timestamp_ns=rgb_item.timestamp_ns,
            camera_frame_id=camera_frame_id,
            tf_records_by_child=tf_records_by_child,
            odom_records=odom_records,
            max_tf_delta_ns=max_tf_delta_ns,
            max_odom_delta_ns=max_odom_delta_ns,
        )
        if extrinsics is None:
            run_stats["frames_missing_extrinsics"] += 1

        detections: list[dict[str, Any]] = []
        det_delta_ns: int | None = None
        if args.detection_backend == "topic":
            det_idx = _closest_index(det_ts, rgb_item.timestamp_ns, start_idx=det_cursor)
            if det_idx >= 0:
                det_cursor = max(det_cursor, det_idx)
                det_item = detection_messages[det_idx]
                det_delta_ns = abs(det_item.timestamp_ns - rgb_item.timestamp_ns)
                if det_delta_ns <= max_det_delta_ns:
                    detections = _parse_detections_message(
                        det_item.message, image_width=rgb_img.shape[1], image_height=rgb_img.shape[0]
                    )
                else:
                    run_stats["frames_without_detection_match"] += 1
            else:
                run_stats["frames_without_detection_match"] += 1
        else:
            if detector is None:
                raise RuntimeError("Detector backend is yolo_ensemble_dino but detector was not initialized")
            detections = detector.infer(rgb_img, depth_img_mm, intrinsics)
            run_stats["inference_frames"] += 1

        for det in detections:
            label = det["class_label"]
            class_histogram[label] = class_histogram.get(label, 0) + 1

        run_stats["detections_total"] += len(detections)

        frame_index = captured_frame_counter
        frame_key = f"frame_{frame_index:06d}_{rgb_item.timestamp_ns}"

        frame_payload = {
            "frame_index": frame_index,
            "frame_key": frame_key,
            "timestamp_ns": rgb_item.timestamp_ns,
            "timestamp_sec": rgb_item.timestamp_ns / 1e9,
            "rgb_shape_hwc": [int(rgb_img.shape[0]), int(rgb_img.shape[1]), int(rgb_img.shape[2])],
            "depth_shape_hw": [int(depth_img_mm.shape[0]), int(depth_img_mm.shape[1])],
            "depth_units": "mm",
            "intrinsics": _as_intrinsics_payload(intrinsics),
            "extrinsics": extrinsics,
            "extrinsics_status": extrinsics_status,
            "detection_match_delta_ms": None if det_delta_ns is None else float(det_delta_ns) / 1e6,
            "class_labels": [d["class_label"] for d in detections],
            "bboxes_xyxy": [d["bbox_xyxy"] for d in detections],
            "detections": detections,
            "sync_deltas_ms": {
                "rgb_depth": float(depth_delta) / 1e6,
                "rgb_detection": None if det_delta_ns is None else float(det_delta_ns) / 1e6,
            },
            "artifacts": {},
        }

        if not args.dry_run:
            batch_id = frame_index // int(args.batch_size)
            batch_dir = frames_root / f"batch_{batch_id:05d}"
            rgb_dir = batch_dir / "rgb"
            depth_dir = batch_dir / "depth"
            frames_dir = batch_dir / "frames"
            rgb_dir.mkdir(parents=True, exist_ok=True)
            depth_dir.mkdir(parents=True, exist_ok=True)
            frames_dir.mkdir(parents=True, exist_ok=True)

            rgb_path = rgb_dir / f"{frame_key}.png"
            depth_path = depth_dir / f"{frame_key}.png"
            frame_json_path = frames_dir / f"{frame_key}.json"

            cv2.imwrite(str(rgb_path), rgb_img)
            cv2.imwrite(str(depth_path), depth_img_mm)

            frame_payload["artifacts"] = {
                "rgb": _relative(rgb_path, run_dir),
                "depth": _relative(depth_path, run_dir),
                "frame_manifest": _relative(frame_json_path, run_dir),
            }
            _json_dump(frame_json_path, frame_payload)

        batch_frames.append(frame_payload)
        run_stats["synced_frames"] += 1
        captured_frame_counter += 1

        # Flush one complete batch immediately to keep memory bounded.
        if len(batch_frames) >= int(args.batch_size):
            if not args.dry_run:
                _flush_batch(batch_frames, frames_root, run_stats, int(args.batch_size))
            batch_frames.clear()

    if batch_frames and not args.dry_run:
        _flush_batch(batch_frames, frames_root, run_stats, int(args.batch_size))

    run_manifest = {
        "schema_version": "1.0.0",
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "bag_path": str(bag_path),
        "topics": {
            "rgb": args.rgb_topic,
            "depth": args.depth_topic,
            "camera_info": args.camera_info_topic,
            "detections": args.detection_topic if args.detection_backend == "topic" else "local_inference",
            "tf": args.tf_topic,
            "odom": args.odom_topic,
        },
        "camera_frame_id": args.camera_frame_id,
        "available_topics": available_topics,
        "configuration": {
            "batch_size": int(args.batch_size),
            "detection_backend": str(args.detection_backend),
            "max_frames": int(args.max_frames),
            "max_rgb_depth_delta_ms": float(args.max_rgb_depth_delta_ms),
            "max_det_delta_ms": float(args.max_det_delta_ms),
            "max_tf_delta_ms": float(args.max_tf_delta_ms),
            "max_odom_delta_ms": float(args.max_odom_delta_ms),
            "detector_module_path": str(Path(args.detector_module_path).resolve()) if args.detector_module_path else "",
            "detector_device": str(args.detector_device),
            "dino_enabled": not bool(args.disable_dino),
            "depth_gate_enabled": not bool(args.disable_depth_gate),
            "v1_path": str(args.v1_path),
            "v2_path": str(args.v2_path),
            "v3_path": str(args.v3_path),
            "dry_run": bool(args.dry_run),
        },
        "stats": run_stats,
        "class_histogram": class_histogram,
    }

    if not args.dry_run:
        _json_dump(metadata_root / "run_manifest.json", run_manifest)

    print("=" * 72)
    print("RGBD capture pipeline completed")
    print(f"Run directory: {run_dir}")
    print(f"Synced frames: {run_stats['synced_frames']}")
    print(f"Detections parsed: {run_stats['detections_total']}")
    print(f"Batches written: {run_stats['batches_written']}")
    print("=" * 72)


def _flush_batch(
    batch_frames: list[dict[str, Any]],
    frames_root: Path,
    run_stats: dict[str, Any],
    batch_size: int,
) -> None:
    if not batch_frames:
        return

    first_frame = batch_frames[0]
    batch_id = int(first_frame["frame_index"]) // max(1, int(batch_size))
    batch_dir = frames_root / f"batch_{batch_id:05d}"

    detections_count = sum(len(frame["detections"]) for frame in batch_frames)
    missing_extrinsics = sum(1 for frame in batch_frames if frame["extrinsics"] is None)
    without_det_match = sum(
        1 for frame in batch_frames if frame["sync_deltas_ms"]["rgb_detection"] is None
    )

    batch_manifest = {
        "schema_version": "1.0.0",
        "batch_id": batch_id,
        "frame_count": len(batch_frames),
        "frame_index_start": int(batch_frames[0]["frame_index"]),
        "frame_index_end": int(batch_frames[-1]["frame_index"]),
        "timestamp_ns_start": int(batch_frames[0]["timestamp_ns"]),
        "timestamp_ns_end": int(batch_frames[-1]["timestamp_ns"]),
        "detections_total": detections_count,
        "frames_missing_extrinsics": missing_extrinsics,
        "frames_without_detection_match": without_det_match,
        "frame_manifests": [
            frame["artifacts"].get("frame_manifest", "") for frame in batch_frames
        ],
    }

    _json_dump(batch_dir / "batch_manifest.json", batch_manifest)
    run_stats["batches_written"] += 1


if __name__ == "__main__":
    main()
