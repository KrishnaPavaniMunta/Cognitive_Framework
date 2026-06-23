"""
RGB-D ROS bag reader  (Saxon hallway bags)
──────────────────────────────────────────
Step 1 only: read RGB + depth frames and the aligned camera intrinsics
from a ROS2 (sqlite3 / zstd) bag. No detection, no zone projection.

The depth topic (`/camera/depth_registered/image_raw`) is already registered
to the RGB frame, and the RGB topic (`/camera/rgb/image_rect_color`) is
rectified — so a single CameraInfo (`/camera/rgb/camera_info`) describes both.

Usage:
    python door_rgbd_reader.py --bag "..../saxon/hallway 1"
    python door_rgbd_reader.py --bag "..../saxon/hallway 1" --preview
    python door_rgbd_reader.py --bag "..../saxon/hallway 1" --max-frames 100
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

# ── Default topics for the Saxon hallway bag ──────────────────────────────────
RGB_TOPIC         = "/camera/rgb/image_rect_color"
DEPTH_TOPIC       = "/camera/depth_registered/image_raw"
CAMERA_INFO_TOPIC = "/camera/rgb/camera_info"

# Max RGB↔depth timestamp delta (seconds) to accept a synchronized pair.
MAX_TIME_DIFF = 0.05


# ── Data containers ───────────────────────────────────────────────────────────

@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def __str__(self) -> str:
        return (
            f"fx={self.fx:.2f} fy={self.fy:.2f} "
            f"cx={self.cx:.2f} cy={self.cy:.2f} "
            f"({self.width}x{self.height})"
        )


@dataclass
class RGBDFrame:
    timestamp: float          # RGB timestamp (seconds)
    rgb: np.ndarray           # HxWx3 BGR uint8
    depth_mm: np.ndarray      # HxW uint16, millimetres (0 = invalid)


# ── ROS message helpers ───────────────────────────────────────────────────────

def _ros_time_to_float(sec: int, nanosec: int) -> float:
    return float(sec) + float(nanosec) * 1e-9


def _decode_color_image(msg) -> np.ndarray | None:
    """Decode a sensor_msgs/Image colour message to BGR uint8."""
    h = int(msg.height)
    w = int(msg.width)
    enc = str(msg.encoding).lower()
    data = np.frombuffer(msg.data, dtype=np.uint8)

    if enc == "rgb8":
        return cv2.cvtColor(data.reshape((h, w, 3)), cv2.COLOR_RGB2BGR)
    if enc == "bgr8":
        return data.reshape((h, w, 3)).copy()
    if enc == "rgba8":
        return cv2.cvtColor(data.reshape((h, w, 4)), cv2.COLOR_RGBA2BGR)
    if enc == "bgra8":
        return cv2.cvtColor(data.reshape((h, w, 4)), cv2.COLOR_BGRA2BGR)
    if enc == "mono8":
        return cv2.cvtColor(data.reshape((h, w)), cv2.COLOR_GRAY2BGR)
    return None


def _decode_depth_image_mm(msg) -> np.ndarray | None:
    """Decode a sensor_msgs/Image depth message to uint16 millimetres."""
    h = int(msg.height)
    w = int(msg.width)
    enc = str(msg.encoding).lower()

    if enc == "16uc1":
        return np.frombuffer(msg.data, dtype=np.uint16).reshape((h, w)).copy()

    if enc == "32fc1":
        arr_m = np.frombuffer(msg.data, dtype=np.float32).reshape((h, w)).copy()
        arr_m = np.nan_to_num(arr_m, nan=0.0, posinf=0.0, neginf=0.0)
        arr_m = np.clip(arr_m, 0.0, 65.535)
        return (arr_m * 1000.0).astype(np.uint16)

    return None


def _camera_info_to_intrinsics(msg) -> CameraIntrinsics:
    """Extract fx, fy, cx, cy from a sensor_msgs/CameraInfo message."""
    k = [float(v) for v in msg.k]   # row-major 3x3
    return CameraIntrinsics(
        fx=k[0], fy=k[4], cx=k[2], cy=k[5],
        width=int(msg.width), height=int(msg.height),
    )


# ── Bag reading ───────────────────────────────────────────────────────────────

def read_intrinsics(
    bag_dir: Path,
    camera_info_topic: str = CAMERA_INFO_TOPIC,
) -> CameraIntrinsics:
    """Read the first CameraInfo message and return aligned intrinsics."""
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    with AnyReader([bag_dir], default_typestore=typestore) as reader:
        conns = [c for c in reader.connections if c.topic == camera_info_topic]
        if not conns:
            raise RuntimeError(f"CameraInfo topic not found: {camera_info_topic}")
        for conn, _, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            return _camera_info_to_intrinsics(msg)
    raise RuntimeError(f"No CameraInfo messages on {camera_info_topic}")


def iter_rgbd_frames(
    bag_dir: Path,
    rgb_topic: str = RGB_TOPIC,
    depth_topic: str = DEPTH_TOPIC,
    max_time_diff: float = MAX_TIME_DIFF,
    max_frames: int = 0,
) -> Iterator[RGBDFrame]:
    """
    Yield time-synchronized RGBDFrame objects.

    Messages are read in log-time order; each RGB frame is paired with the most
    recent depth frame within `max_time_diff` seconds.
    """
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    with AnyReader([bag_dir], default_typestore=typestore) as reader:
        topics = {c.topic for c in reader.connections}
        missing = [t for t in (rgb_topic, depth_topic) if t not in topics]
        if missing:
            raise RuntimeError(f"Missing required topics in bag: {missing}")

        conns = [c for c in reader.connections if c.topic in (rgb_topic, depth_topic)]

        pending_depth_ts: float | None = None
        pending_depth_img: np.ndarray | None = None
        yielded = 0

        for conn, _, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            ts = _ros_time_to_float(msg.header.stamp.sec, msg.header.stamp.nanosec)

            if conn.topic == depth_topic:
                depth = _decode_depth_image_mm(msg)
                if depth is not None:
                    pending_depth_ts = ts
                    pending_depth_img = depth
                continue

            # conn.topic == rgb_topic
            rgb = _decode_color_image(msg)
            if rgb is None or pending_depth_img is None:
                continue
            if abs(ts - pending_depth_ts) > max_time_diff:
                continue

            yield RGBDFrame(timestamp=ts, rgb=rgb, depth_mm=pending_depth_img)
            yielded += 1
            if max_frames > 0 and yielded >= max_frames:
                return


def _depth_to_colormap(depth_mm: np.ndarray, max_mm: int = 5000) -> np.ndarray:
    """Colourise a uint16 mm depth map for visualization."""
    clipped = np.clip(depth_mm, 0, max_mm).astype(np.float32)
    norm = (clipped / float(max_mm) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_JET)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read RGB + depth + aligned intrinsics from a Saxon ROS2 bag (step 1)."
    )
    parser.add_argument("--bag", type=str, required=True, help="Path to bag folder (contains metadata.yaml)")
    parser.add_argument("--rgb-topic", type=str, default=RGB_TOPIC)
    parser.add_argument("--depth-topic", type=str, default=DEPTH_TOPIC)
    parser.add_argument("--camera-info-topic", type=str, default=CAMERA_INFO_TOPIC)
    parser.add_argument("--max-time-diff", type=float, default=MAX_TIME_DIFF)
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N synced frames (0 = all)")
    parser.add_argument("--preview", action="store_true", help="Show RGB | depth side-by-side window")
    args = parser.parse_args()

    bag_dir = Path(args.bag)
    if not (bag_dir / "metadata.yaml").exists():
        raise FileNotFoundError(f"metadata.yaml not found in bag dir: {bag_dir}")

    # ── Aligned camera intrinsics ─────────────────────────────────────────────
    intr = read_intrinsics(bag_dir, camera_info_topic=args.camera_info_topic)
    print(f"[INTRINSICS] {intr}")

    # ── Stream synchronized RGB-D frames ──────────────────────────────────────
    count = 0
    last_shape_rgb = None
    last_shape_depth = None
    for frame in iter_rgbd_frames(
        bag_dir,
        rgb_topic=args.rgb_topic,
        depth_topic=args.depth_topic,
        max_time_diff=args.max_time_diff,
        max_frames=args.max_frames,
    ):
        count += 1
        last_shape_rgb = frame.rgb.shape
        last_shape_depth = frame.depth_mm.shape

        if args.preview:
            depth_vis = _depth_to_colormap(frame.depth_mm)
            if depth_vis.shape[:2] != frame.rgb.shape[:2]:
                depth_vis = cv2.resize(depth_vis, (frame.rgb.shape[1], frame.rgb.shape[0]))
            combo = np.hstack([frame.rgb, depth_vis])
            cv2.imshow("RGB | Depth", combo)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if count % 100 == 0:
            print(f"[READ] synced frames: {count}  t={frame.timestamp:.3f}s")

    if args.preview:
        cv2.destroyAllWindows()

    print(f"\n[DONE] total synced RGB-D frames : {count}")
    print(f"[DONE] rgb shape                 : {last_shape_rgb}")
    print(f"[DONE] depth shape               : {last_shape_depth}")


if __name__ == "__main__":
    main()