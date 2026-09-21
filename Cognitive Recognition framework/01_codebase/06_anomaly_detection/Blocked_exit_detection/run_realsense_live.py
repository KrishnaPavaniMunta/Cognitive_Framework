"""Live RealSense Runner for Exit Obstruction Detection (V1/V2 detector).

Streams live RGB-D from an Intel RealSense camera (e.g. D455),
aligns depth to color, and runs ExitObstructionMonitor to display
the 3D door keep-clear zone and any detected obstructions in real time.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import pyrealsense2 as rs

# Add v2 and v1 directories to Python path
CURRENT_DIR = Path(__file__).resolve().parent
V2_DIR = CURRENT_DIR / "v2"
V1_DIR = CURRENT_DIR / "v1"
sys.path.insert(0, str(V2_DIR))
sys.path.insert(0, str(V1_DIR))

from exit_obstruction import ExitObstructionMonitor


class CameraIntrinsicsWrapper:
    """Wrapper matching the interface expected by v1/v2 geometry functions."""

    def __init__(self, fx: float, fy: float, cx: float, cy: float) -> None:
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live RealSense Exit Obstruction Detector")
    parser.add_argument("--radius", type=float, default=1.2, help="Keep-clear zone radius in meters (default: 1.2)")
    parser.add_argument("--use-sam", action="store_true", help="Enable SAM pixel-level refinement")
    parser.add_argument("--width", type=int, default=640, help="Stream width")
    parser.add_argument("--height", type=int, default=480, help="Stream height")
    parser.add_argument("--fps", type=int, default=30, help="Stream FPS")
    args = parser.parse_args()

    print(f"[LIVE] Initializing RealSense pipeline ({args.width}x{args.height} @ {args.fps}fps)...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    # Fetch camera intrinsics from color stream
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    raw_intr = color_stream.get_intrinsics()
    intrinsics = CameraIntrinsicsWrapper(
        fx=raw_intr.fx,
        fy=raw_intr.fy,
        cx=raw_intr.ppx,
        cy=raw_intr.ppy,
    )
    print(f"[LIVE] Intrinsics: fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}, cx={intrinsics.cx:.2f}, cy={intrinsics.cy:.2f}")

    print("[LIVE] Initializing ExitObstructionMonitor (YOLO + DINO + SAM)...")
    monitor = ExitObstructionMonitor(radius_m=args.radius, use_sam=args.use_sam)
    monitor.load()
    print("[LIVE] Detector ready! Starting streaming loop. Press 'q' or ESC in display window to exit.")

    frame_index = 0
    fps_history = []

    try:
        while True:
            t0 = time.perf_counter()
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            frame_index += 1
            rgb_bgr = np.asanyarray(color_frame.get_data())
            depth_mm = np.asanyarray(depth_frame.get_data())  # raw z16 values are millimeters

            # Run detection & 3D keep-clear zone check
            result = monitor.evaluate(
                rgb_bgr=rgb_bgr,
                depth_mm=depth_mm,
                intrinsics=intrinsics,
                frame_index=frame_index,
            )

            # Draw 3D zone & obstruction alerts
            annotated = monitor.draw_overlay(rgb_bgr, result, intrinsics)

            t1 = time.perf_counter()
            fps = 1.0 / max(1e-5, (t1 - t0))
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)

            # Draw HUD Info
            status_text = "BLOCKED" if result.obstruction_flag else "CLEAR"
            status_color = (0, 0, 255) if result.obstruction_flag else (0, 255, 0)
            door_text = "CONFIRMED" if result.door_confirmed else "SEARCHING"
            cv2.putText(annotated, f"EXIT: {status_text} | DOOR: {door_text} | FPS: {avg_fps:.1f}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA)

            if result.door_camera_xyz:
                dx, dy, dz = result.door_camera_xyz
                cv2.putText(annotated, f"Door 3D: X={dx:.2f}m Y={dy:.2f}m Z={dz:.2f}m",
                            (10, annotated.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("RealSense Blocked Exit Monitor", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break

    finally:
        print("[LIVE] Stopping pipeline and closing windows...")
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
