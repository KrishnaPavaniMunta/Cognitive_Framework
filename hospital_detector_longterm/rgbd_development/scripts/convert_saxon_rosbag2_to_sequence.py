from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


@dataclass
class Entry:
    ts: float
    relpath: str


def _ros_time_to_float(sec: int, nanosec: int) -> float:
    return float(sec) + float(nanosec) * 1e-9


def _decode_color_image(msg) -> np.ndarray | None:
    h = int(msg.height)
    w = int(msg.width)
    enc = str(msg.encoding).lower()
    data = np.frombuffer(msg.data, dtype=np.uint8)

    if enc == "rgb8":
        arr = data.reshape((h, w, 3))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if enc == "bgr8":
        return data.reshape((h, w, 3)).copy()
    if enc == "rgba8":
        arr = data.reshape((h, w, 4))
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    if enc == "bgra8":
        arr = data.reshape((h, w, 4))
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    if enc == "mono8":
        arr = data.reshape((h, w))
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return None


def _decode_depth_image_mm(msg) -> np.ndarray | None:
    h = int(msg.height)
    w = int(msg.width)
    enc = str(msg.encoding).lower()

    if enc == "16uc1":
        arr = np.frombuffer(msg.data, dtype=np.uint16).reshape((h, w)).copy()
        return arr

    if enc == "32fc1":
        arr_m = np.frombuffer(msg.data, dtype=np.float32).reshape((h, w)).copy()
        arr_m = np.nan_to_num(arr_m, nan=0.0, posinf=0.0, neginf=0.0)
        arr_m = np.clip(arr_m, 0.0, 65.535)
        arr_mm = (arr_m * 1000.0).astype(np.uint16)
        return arr_mm

    return None


def _write_assoc(path: Path, entries: list[Entry]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# timestamp path\n")
        for e in entries:
            f.write(f"{e.ts:.9f} {e.relpath}\n")


def _pair_and_write_associations(root: Path, rgb_entries: list[Entry], depth_entries: list[Entry], max_time_diff: float) -> int:
    rgb_entries = sorted(rgb_entries, key=lambda x: x.ts)
    depth_entries = sorted(depth_entries, key=lambda x: x.ts)

    pairs: list[tuple[Entry, Entry]] = []
    j = 0
    dn = len(depth_entries)

    for r in rgb_entries:
        while j + 1 < dn and depth_entries[j + 1].ts <= r.ts:
            j += 1
        cands = [depth_entries[j]]
        if j + 1 < dn:
            cands.append(depth_entries[j + 1])
        d = min(cands, key=lambda x: abs(x.ts - r.ts))
        if abs(d.ts - r.ts) <= max_time_diff:
            pairs.append((r, d))

    rgb_out: list[Entry] = [p[0] for p in pairs]
    depth_out: list[Entry] = [p[1] for p in pairs]
    _write_assoc(root / "rgb.txt", rgb_out)
    _write_assoc(root / "depth.txt", depth_out)
    return len(pairs)


def convert_bag(
    bag_dir: Path,
    out_root: Path,
    max_frames: int,
    max_time_diff: float,
    rgb_topic: str,
    depth_topic: str,
    odom_topic: str,
    camera_info_topic: str,
) -> Path:
    if not (bag_dir / "metadata.yaml").exists():
        raise FileNotFoundError(f"metadata.yaml not found in bag dir: {bag_dir}")

    seq_root = out_root / bag_dir.name
    rgb_dir = seq_root / "rgb"
    depth_dir = seq_root / "depth"
    seq_root.mkdir(parents=True, exist_ok=True)
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    rgb_entries: list[Entry] = []
    depth_entries: list[Entry] = []
    odom_rows: list[list[float]] = []
    camera_info_saved = False

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    with AnyReader([bag_dir], default_typestore=typestore) as reader:
        topic_to_conn = {c.topic: c for c in reader.connections}
        required = [rgb_topic, depth_topic, odom_topic]
        missing = [t for t in required if t not in topic_to_conn]
        if missing:
            raise RuntimeError(f"Missing required topics in bag: {missing}")

        selected_topics = {rgb_topic, depth_topic, odom_topic}
        if camera_info_topic in topic_to_conn:
            selected_topics.add(camera_info_topic)

        selected = [c for c in reader.connections if c.topic in selected_topics]

        rgb_idx = 0
        depth_idx = 0

        for conn, _, raw in reader.messages(connections=selected):
            msg = reader.deserialize(raw, conn.msgtype)
            topic = conn.topic

            if topic == rgb_topic and rgb_idx < max_frames:
                t = _ros_time_to_float(msg.header.stamp.sec, msg.header.stamp.nanosec)
                bgr = _decode_color_image(msg)
                if bgr is None:
                    continue
                rel = f"rgb/{rgb_idx:06d}.png"
                cv2.imwrite(str(seq_root / rel), bgr)
                rgb_entries.append(Entry(t, rel))
                rgb_idx += 1
                continue

            if topic == depth_topic and depth_idx < max_frames:
                t = _ros_time_to_float(msg.header.stamp.sec, msg.header.stamp.nanosec)
                depth_mm = _decode_depth_image_mm(msg)
                if depth_mm is None:
                    continue
                rel = f"depth/{depth_idx:06d}.png"
                cv2.imwrite(str(seq_root / rel), depth_mm)
                depth_entries.append(Entry(t, rel))
                depth_idx += 1
                continue

            if topic == odom_topic:
                t = _ros_time_to_float(msg.header.stamp.sec, msg.header.stamp.nanosec)
                p = msg.pose.pose.position
                q = msg.pose.pose.orientation
                odom_rows.append([
                    t,
                    float(p.x), float(p.y), float(p.z),
                    float(q.x), float(q.y), float(q.z), float(q.w),
                ])
                continue

            if topic == camera_info_topic and not camera_info_saved:
                cam_info = {
                    "topic": camera_info_topic,
                    "width": int(msg.width),
                    "height": int(msg.height),
                    "k": [float(v) for v in msg.k],
                    "d": [float(v) for v in msg.d],
                    "distortion_model": str(msg.distortion_model),
                }
                (seq_root / "camera_info.json").write_text(json.dumps(cam_info, indent=2), encoding="utf-8")
                camera_info_saved = True

            if rgb_idx >= max_frames and depth_idx >= max_frames:
                # still keep reading odom for temporal context is not required for this first experiment
                # we can stop early for a quick one-bag trial
                break

    if not rgb_entries or not depth_entries:
        raise RuntimeError("No RGB/depth frames were decoded from bag.")

    pair_count = _pair_and_write_associations(seq_root, rgb_entries, depth_entries, max_time_diff=max_time_diff)

    with (seq_root / "odom.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"])
        w.writerows(odom_rows)

    summary = {
        "bag_dir": str(bag_dir),
        "rgb_topic": rgb_topic,
        "depth_topic": depth_topic,
        "odom_topic": odom_topic,
        "rgb_frames_saved": len(rgb_entries),
        "depth_frames_saved": len(depth_entries),
        "paired_frames": pair_count,
        "odom_messages": len(odom_rows),
        "max_time_diff": max_time_diff,
    }
    (seq_root / "export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return seq_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert one Saxon ROS2 bag folder to RGB-D sequence format.")
    parser.add_argument("--bag-dir", type=str, required=True, help="Path to a bag folder containing metadata.yaml")
    parser.add_argument("--out-root", type=str, required=True, help="Output root for converted sequence")
    parser.add_argument("--max-frames", type=int, default=600, help="Max RGB and depth frames to export")
    parser.add_argument("--max-time-diff", type=float, default=0.03, help="Max RGB-depth timestamp delta")
    parser.add_argument("--rgb-topic", type=str, default="/camera/rgb/image_rect_color")
    parser.add_argument("--depth-topic", type=str, default="/camera/depth_registered/image_raw")
    parser.add_argument("--odom-topic", type=str, default="/odom")
    parser.add_argument("--camera-info-topic", type=str, default="/camera/rgb/camera_info")
    args = parser.parse_args()

    seq_root = convert_bag(
        bag_dir=Path(args.bag_dir).resolve(),
        out_root=Path(args.out_root).resolve(),
        max_frames=int(args.max_frames),
        max_time_diff=float(args.max_time_diff),
        rgb_topic=args.rgb_topic,
        depth_topic=args.depth_topic,
        odom_topic=args.odom_topic,
        camera_info_topic=args.camera_info_topic,
    )
    print(f"[OK] Sequence exported: {seq_root}")


if __name__ == "__main__":
    main()
