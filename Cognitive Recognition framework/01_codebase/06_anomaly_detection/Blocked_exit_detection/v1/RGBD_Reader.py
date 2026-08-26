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
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from rosbags.typesys import Stores, get_typestore

try:
    import zstandard as zstd
except Exception:  # pragma: no cover - zstd should normally be available
    zstd = None

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


def _resolve_db3_files(bag_path: Path) -> list[Path]:
    """Return rosbag2 sqlite files from either a bag directory or a single .db3 file."""
    if bag_path.is_file():
        if bag_path.suffix.lower() != ".db3":
            raise RuntimeError(f"Unsupported bag file type (expected .db3): {bag_path}")
        return [bag_path]

    db3_files = sorted(bag_path.glob("*.db3"))
    if not db3_files:
        raise RuntimeError(f"No .db3 files found in bag path: {bag_path}")
    return db3_files


def _maybe_decompress_zstd(payload: bytes) -> bytes:
    """Decompress zstd-compressed rosbag payloads when needed."""
    zstd_magic = b"\x28\xb5\x2f\xfd"
    if payload.startswith(zstd_magic):
        if zstd is None:
            raise RuntimeError("zstandard is required to read compressed rosbag payloads")
        return zstd.ZstdDecompressor().decompress(payload)
    return payload


# ── Bag reading ───────────────────────────────────────────────────────────────

def read_intrinsics(
    bag_dir: Path,
    camera_info_topic: str = CAMERA_INFO_TOPIC,
) -> CameraIntrinsics:
    """Read the first CameraInfo message and return aligned intrinsics."""
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    db3_files = _resolve_db3_files(Path(bag_dir))

    for db3 in db3_files:
        conn = sqlite3.connect(str(db3))
        try:
            cur = conn.cursor()
            topics = {
                int(topic_id): (name, msgtype)
                for topic_id, name, msgtype in cur.execute("SELECT id, name, type FROM topics")
            }
            camera_topic_ids = [tid for tid, (name, _) in topics.items() if name == camera_info_topic]
            if not camera_topic_ids:
                continue

            placeholders = ",".join("?" for _ in camera_topic_ids)
            query = (
                f"SELECT topic_id, data FROM messages "
                f"WHERE topic_id IN ({placeholders}) ORDER BY timestamp"
            )
            for topic_id, data in cur.execute(query, camera_topic_ids):
                raw = bytes(data)
                _, msgtype = topics[int(topic_id)]
                try:
                    raw = _maybe_decompress_zstd(raw)
                    msg = typestore.deserialize_cdr(raw, msgtype)
                except Exception:
                    continue
                return _camera_info_to_intrinsics(msg)
        finally:
            conn.close()

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
    db3_files = _resolve_db3_files(Path(bag_dir))

    all_topics: set[str] = set()
    topic_map_by_db: list[dict[int, tuple[str, str]]] = []
    for db3 in db3_files:
        conn = sqlite3.connect(str(db3))
        try:
            cur = conn.cursor()
            topic_map = {
                int(topic_id): (name, msgtype)
                for topic_id, name, msgtype in cur.execute("SELECT id, name, type FROM topics")
            }
            topic_map_by_db.append(topic_map)
            all_topics.update(name for name, _ in topic_map.values())
        finally:
            conn.close()

    missing = [t for t in (rgb_topic, depth_topic) if t not in all_topics]
    if missing:
        raise RuntimeError(f"Missing required topics in bag: {missing}")

    pending_depth_ts: float | None = None
    pending_depth_img: np.ndarray | None = None
    yielded = 0
    skipped_decompress = 0
    skipped_deserialize = 0

    for db3_idx, db3 in enumerate(db3_files):
        topic_map = topic_map_by_db[db3_idx]
        wanted_ids = [tid for tid, (name, _) in topic_map.items() if name in (rgb_topic, depth_topic)]
        if not wanted_ids:
            continue

        placeholders = ",".join("?" for _ in wanted_ids)
        query = (
            f"SELECT topic_id, timestamp, data FROM messages "
            f"WHERE topic_id IN ({placeholders}) ORDER BY timestamp"
        )

        conn = sqlite3.connect(str(db3))
        try:
            cur = conn.cursor()
            for topic_id, timestamp_ns, data in cur.execute(query, wanted_ids):
                topic, msgtype = topic_map[int(topic_id)]
                raw = bytes(data)

                try:
                    raw = _maybe_decompress_zstd(raw)
                except Exception as exc:
                    skipped_decompress += 1
                    if skipped_decompress <= 10:
                        print(
                            f"[RGBD] Warning: skipped corrupted compressed message "
                            f"({db3.name}, topic={topic}): {exc}"
                        )
                    continue

                try:
                    msg = typestore.deserialize_cdr(raw, msgtype)
                except Exception as exc:
                    skipped_deserialize += 1
                    if skipped_deserialize <= 10:
                        print(
                            f"[RGBD] Warning: skipped undecodable message "
                            f"({db3.name}, topic={topic}): {exc}"
                        )
                    continue

                try:
                    ts = _ros_time_to_float(msg.header.stamp.sec, msg.header.stamp.nanosec)
                except Exception:
                    ts = float(timestamp_ns) * 1e-9

                if topic == depth_topic:
                    depth = _decode_depth_image_mm(msg)
                    if depth is not None:
                        pending_depth_ts = ts
                        pending_depth_img = depth
                    continue

                # topic == rgb_topic
                rgb = _decode_color_image(msg)
                if rgb is None or pending_depth_img is None or pending_depth_ts is None:
                    continue
                if abs(ts - pending_depth_ts) > max_time_diff:
                    continue

                yield RGBDFrame(timestamp=ts, rgb=rgb, depth_mm=pending_depth_img)
                yielded += 1
                if max_frames > 0 and yielded >= max_frames:
                    return
        finally:
            conn.close()

    if skipped_decompress > 10:
        print(f"[RGBD] Warning: skipped {skipped_decompress} corrupted compressed messages in total.")
    if skipped_deserialize > 10:
        print(f"[RGBD] Warning: skipped {skipped_deserialize} undecodable messages in total.")


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