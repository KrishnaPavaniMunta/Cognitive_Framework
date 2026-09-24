"""
rgbd_clean_to_sidebyside_video.py

Convert an already-extracted "rgbd_clean_*" sequence folder (produced from a
rosbag2 recording: rgb/*.png + depth_exr/*.exr + frames.csv) into a single
side-by-side RGB | colourised-depth MP4.

Reuses the same normalize -> colourmap -> hstack -> VideoWriter pattern as
rgbd_spatial_twin.py (this same directory), adapted for the frames.csv/EXR
clean-sequence format instead of TUM rgb.txt/depth.txt associations.

Usage:
  python rgbd_clean_to_sidebyside_video.py --sequence-root "<path to rgbd_clean_* folder>"
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Optional

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np


def load_frames_csv(root: Path) -> list[dict]:
    csv_path = root / "frames.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"frames.csv not found in: {root}")

    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    rows.sort(key=lambda r: int(r["frame"]))
    return rows


def colourise_depth_exr(depth: np.ndarray, depth_max_m: float) -> np.ndarray:
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    clipped = np.clip(depth, 0.0, depth_max_m)
    normalised = (clipped / depth_max_m * 255.0).astype(np.uint8)
    coloured = cv2.applyColorMap(normalised, cv2.COLORMAP_TURBO)
    coloured[depth <= 0.0] = (0, 0, 0)
    return coloured


def build_sidebyside_video(
    sequence_root: Path,
    output_path: Path,
    fps: float,
    depth_max_m: float,
    max_frames: Optional[int],
) -> None:
    rows = load_frames_csv(sequence_root)
    if not rows:
        raise RuntimeError(f"No frame rows found in frames.csv under {sequence_root}")

    writer: Optional[cv2.VideoWriter] = None
    written = 0

    for row in rows:
        if max_frames is not None and written >= max_frames:
            break

        rgb_path = sequence_root / row["rgb"]
        depth_path = sequence_root / row["depth"]

        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None:
            print(f"[WARN] Skipping frame {row.get('frame')}: missing rgb/depth image")
            continue

        depth_vis = colourise_depth_exr(depth, depth_max_m)
        if depth_vis.shape[:2] != rgb.shape[:2]:
            depth_vis = cv2.resize(depth_vis, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

        combined = np.hstack([rgb, depth_vis])
        cv2.putText(
            combined,
            f"frame={row['frame']}/{len(rows)}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if writer is None:
            h, w = combined.shape[:2]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open video writer for {output_path}")
            print(f"[INFO] Writing side-by-side video to {output_path} at {fps} FPS, size {w}x{h}")

        writer.write(combined)
        written += 1
        if written % 50 == 0:
            print(f"[INFO] Written {written}/{len(rows)} frames")

    if writer is not None:
        writer.release()
    if written == 0:
        raise RuntimeError("No frames were written; check sequence_root contents.")

    print(f"[SUCCESS] Wrote {written} frames -> {output_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export rgbd_clean_* sequence (frames.csv + rgb/ + depth_exr/) as side-by-side MP4")
    parser.add_argument("--sequence-root", type=str, required=True, help="Path to the rgbd_clean_* folder containing frames.csv")
    parser.add_argument("--output", type=str, default=None, help="Output MP4 path (default: <sequence_root>_sidebyside.mp4 next to the folder)")
    parser.add_argument("--fps", type=float, default=10.0, help="Output video FPS (default: 10)")
    parser.add_argument("--depth-max-m", type=float, default=6.0, help="Depth clip range in metres for colour normalisation (default: 6.0)")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on number of frames to export")
    args = parser.parse_args()

    sequence_root = Path(args.sequence_root)
    if not sequence_root.exists():
        raise FileNotFoundError(f"sequence_root does not exist: {sequence_root}")

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = sequence_root.parent / f"{sequence_root.name}_sidebyside.mp4"

    build_sidebyside_video(
        sequence_root=sequence_root,
        output_path=output_path,
        fps=args.fps,
        depth_max_m=args.depth_max_m,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
