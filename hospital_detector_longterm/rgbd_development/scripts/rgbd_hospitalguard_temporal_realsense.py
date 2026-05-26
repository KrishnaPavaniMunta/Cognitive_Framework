from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from rgbd_hospitalguard_temporal import (
    DEFAULT_DB_PATH,
    DETECTIONS_DIR,
    LOGS_DIR,
    ROOT_DIR,
    _resolve_output_arg,
    run_realsense_temporal,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HospitalGuard Temporal live runner for Intel RealSense"
    )
    parser.add_argument("--expected-class", type=str, default="hospital_room", help="Expected class for Excel summary")
    parser.add_argument("--camera-name", type=str, default="realsense", help="Session label only")
    parser.add_argument("--output-video", type=str, default=None, help="Output annotated video path")
    parser.add_argument("--csv-path", type=str, default=None, help="Output CSV path")
    parser.add_argument("--db-path", type=str, default=str(DEFAULT_DB_PATH), help="SQLite DB path")
    parser.add_argument(
        "--log-path",
        type=str,
        default=str(LOGS_DIR / "hospitalguard_temporal_rgbd_log.xlsx"),
        help="Excel log path",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame cap")
    parser.add_argument("--fps", type=float, default=30.0, help="Tracker/writer FPS")
    parser.add_argument("--live-width", type=int, default=640, help="RealSense color/depth width")
    parser.add_argument("--live-height", type=int, default=480, help="RealSense color/depth height")
    parser.add_argument("--live-fps", type=int, default=30, help="RealSense stream FPS")
    parser.add_argument("--live-detect-every", type=int, default=1, help="Run YOLO every N live frames")
    parser.add_argument("--live-dino-interval-sec", type=float, default=3.5, help="DINO interval in seconds")
    parser.add_argument("--live-disable-dino", action="store_true", help="Disable DINO fallback")
    parser.add_argument("--db-commit-every", type=int, default=1, help="Commit DB every N inserts")
    parser.add_argument("--gpu-required", action="store_true", help="Fail if CUDA is unavailable")
    parser.add_argument("--yolo-half", action="store_true", help="Use FP16 YOLO on CUDA")
    parser.add_argument("--live-ultra-smooth", action="store_true", help="Preset for smoother live playback")
    parser.add_argument(
        "--v1-path",
        type=str,
        default=str(ROOT_DIR / "outputs/runs/hospital/phase2_neck_head/weights/best.pt"),
        help="Path to V1 YOLO weights",
    )
    parser.add_argument(
        "--v3-path",
        type=str,
        default=str(ROOT_DIR / "outputs/runs/hospital_v3/phase2_neck_head/weights/best.pt"),
        help="Path to V3 YOLO weights",
    )
    args = parser.parse_args()

    if args.live_ultra_smooth:
        args.live_detect_every = max(2, args.live_detect_every)
        args.live_disable_dino = True
        args.db_commit_every = max(20, args.db_commit_every)
        args.gpu_required = True
        args.yolo_half = True

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video = _resolve_output_arg(
        args.output_video,
        DETECTIONS_DIR / f"hospitalguard_realsense_temporal_{ts}.mp4",
    )
    csv_path = _resolve_output_arg(
        args.csv_path,
        LOGS_DIR / f"spatial_realsense_temporal_{ts}.csv",
    )
    db_path = _resolve_output_arg(args.db_path, DEFAULT_DB_PATH)
    log_path = _resolve_output_arg(args.log_path, LOGS_DIR / "hospitalguard_temporal_rgbd_log.xlsx")

    run_realsense_temporal(
        output_video=output_video,
        csv_path=csv_path,
        db_path=db_path,
        log_path=log_path,
        expected_class=args.expected_class,
        max_frames=args.max_frames,
        fps=args.fps,
        v1_path=Path(args.v1_path).resolve(),
        v3_path=Path(args.v3_path).resolve(),
        camera_name=args.camera_name,
        live_width=args.live_width,
        live_height=args.live_height,
        live_fps=args.live_fps,
        live_detect_every=args.live_detect_every,
        live_dino_interval_sec=args.live_dino_interval_sec,
        live_disable_dino=args.live_disable_dino,
        db_commit_every=args.db_commit_every,
        gpu_required=args.gpu_required,
        yolo_half=args.yolo_half,
    )


if __name__ == "__main__":
    main()
