from __future__ import annotations

import argparse
import csv
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image


class IoUTracker:
    """
    Lightweight IoU-based tracker.
    Assigns stable integer tracker_ids to detections across frames.
    One tracker instance per class (or shared with class_name key).
    """

    def __init__(self, iou_threshold: float = 0.3, max_missing_frames: int = 15):
        self.iou_threshold = iou_threshold
        self.max_missing_frames = max_missing_frames
        # {tracker_id: {"bbox": (x1,y1,x2,y2), "class": str, "missing": int}}
        self._tracks: dict[int, dict] = {}
        self._next_id = 1

    @staticmethod
    def _iou(a: tuple, b: tuple) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / (area_a + area_b - inter)

    def update(self, detections: list[dict]) -> list[dict]:
        """
        Match detections to existing tracks by IoU (same class only).
        Returns detections with 'tracker_id' field added.
        Unmatched old tracks age out after max_missing_frames.
        """
        matched_track_ids: set[int] = set()
        result: list[dict] = []

        for det in detections:
            cls = det["class_name"]
            bbox = det["bbox"]
            best_id, best_iou = None, 0.0
            for tid, track in self._tracks.items():
                if track["class"] != cls:
                    continue
                iou = self._iou(bbox, track["bbox"])
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou, best_id = iou, tid

            if best_id is not None:
                self._tracks[best_id]["bbox"] = bbox
                self._tracks[best_id]["missing"] = 0
                matched_track_ids.add(best_id)
                result.append({**det, "tracker_id": best_id})
            else:
                new_id = self._next_id
                self._next_id += 1
                self._tracks[new_id] = {"bbox": bbox, "class": cls, "missing": 0}
                matched_track_ids.add(new_id)
                result.append({**det, "tracker_id": new_id})

        # Age out unmatched tracks
        dead = []
        for tid, t in self._tracks.items():
            if tid not in matched_track_ids:
                t["missing"] += 1
                if t["missing"] > self.max_missing_frames:
                    dead.append(tid)
        for tid in dead:
            del self._tracks[tid]

        return result

SCRIPT_DIR = Path(__file__).resolve().parent
RGBD_DEV_DIR = SCRIPT_DIR.parent
ROOT_DIR = RGBD_DEV_DIR.parent.parent
OUTPUT_DIR = RGBD_DEV_DIR / "output"
DETECTIONS_DIR = OUTPUT_DIR / "detections"
LOGS_DIR = OUTPUT_DIR / "logs"
DEFAULT_LOG_PATH = LOGS_DIR / "hospitalguard_rgbd_log.xlsx"
DEFAULT_DB_PATH = OUTPUT_DIR / "hospital_twin.db"

sys.path.insert(0, str(SCRIPT_DIR))   # ensure local rgbd_spatial_twin takes priority
sys.path.insert(1, str(ROOT_DIR))
import infer_hospitalguard as hospitalguard  # noqa: E402
from rgbd_spatial_twin import (  # noqa: E402
    build_sequence,
    depth_to_xyz,
    init_db,
    insert_spatial_memory,
    select_intrinsics,
)


def _iter_detections(yolo_dets: dict, dino_dets: dict) -> list[dict]:
    detections: list[dict] = []
    for source, det_map in (("yolo", yolo_dets), ("dino", dino_dets)):
        for class_name, det_list in det_map.items():
            for det in det_list:
                x1, y1, x2, y2, conf = det
                detections.append(
                    {
                        "class_name": class_name,
                        "source": source,
                        "bbox": (float(x1), float(y1), float(x2), float(y2)),
                        "confidence": float(conf),
                    }
                )
    return detections


def _center_xyz(depth_img, bbox: tuple[float, float, float, float], intr) -> tuple[int, int, tuple[float, float, float] | None, float | None]:
    if depth_img.ndim == 3:
        depth_img = depth_img[:, :, 0]

    height, width = depth_img.shape[:2]
    x1, y1, x2, y2 = bbox
    center_u = min(max(int(round((x1 + x2) * 0.5)), 0), width - 1)
    center_v = min(max(int(round((y1 + y2) * 0.5)), 0), height - 1)
    depth_raw = float(depth_img[center_v, center_u])
    if depth_raw <= 0:
        return center_u, center_v, None, None
    return center_u, center_v, depth_to_xyz(center_u, center_v, depth_raw, intr), depth_raw / intr.depth_scale


def _annotate_spatial_points(frame, spatial_records: list[dict]) -> None:
    for record in spatial_records:
        u = record["center_u"]
        v = record["center_v"]
        tid = record.get("tracker_id", "?")
        cv2.circle(frame, (u, v), 4, (0, 255, 0), -1)
        if record["xyz"] is None:
            label = f"#{tid} {record['class_name']} Z=NA"
        else:
            x, y, z = record["xyz"]
            label = f"#{tid} {record['class_name']} X={x:.2f} Y={y:.2f} Z={z:.2f}"
        cv2.putText(
            frame,
            label,
            (u + 6, max(v - 6, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )


def run_rgbd_sequence_detection(
    sequence_root: Path,
    expected_class: str,
    output_video: Path,
    csv_path: Path,
    log_path: Path,
    db_path: Path,
    max_frames: int | None = None,
    max_time_diff: float = 0.02,
) -> tuple[Path, Path, str]:
    DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    hospitalguard.OUT_DIR = output_video.parent
    hospitalguard.EXCEL_PATH = log_path
    hospitalguard.OUT_DIR.mkdir(parents=True, exist_ok=True)
    hospitalguard.EXCEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    init_db(db_path)
    frames = build_sequence(sequence_root, max_time_diff=max_time_diff)
    if max_frames is not None:
        frames = frames[:max_frames]
    if not frames:
        raise RuntimeError(f"No RGB-D frame pairs found in {sequence_root}")

    intr = select_intrinsics(sequence_root.as_posix().lower())

    print("Loading V1 (106-class hospital)...")
    v1 = YOLO(str(hospitalguard.V1_PATH))
    print("Loading V3 (109-class)...")
    v3 = YOLO(str(hospitalguard.V3_PATH))

    session_id = str(uuid.uuid4())
    print(f"Session ID: {session_id}")
    print(f"Running RGB-D HospitalGuard on {sequence_root}...")
    print(f"Total frames to process: {len(frames)}")

    tracker = IoUTracker(iou_threshold=0.3, max_missing_frames=15)

    # Get first valid frame to determine dimensions for video writer
    writer = None
    video_width = None
    video_height = None
    for frame in frames:
        test_rgb = cv2.imread(str(frame.rgb_path), cv2.IMREAD_COLOR)
        if test_rgb is not None:
            video_height, video_width = test_rgb.shape[:2]
            break
    
    if video_width is not None and video_height is not None:
        writer = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*"MJPG"),
            30.0,
            (video_width, video_height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open output video writer for {output_video}")
    
    all_confs: dict[str, list[float]] = {}
    spatial_rows: list[dict] = []

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        csv_writer = csv.writer(handle)
        csv_writer.writerow([
            "frame_index",
            "timestamp",
            "class_name",
            "tracker_id",
            "source",
            "confidence",
            "center_u",
            "center_v",
            "depth_m",
            "X_m",
            "Y_m",
            "Z_m",
        ])

        conn = None
        try:
            import sqlite3

            conn = sqlite3.connect(db_path)
            for frame_index, frame in enumerate(frames, start=1):
                rgb = cv2.imread(str(frame.rgb_path), cv2.IMREAD_COLOR)
                depth = cv2.imread(str(frame.depth_path), cv2.IMREAD_UNCHANGED)
                if rgb is None or depth is None:
                    continue

                if frame_index % 50 == 0:
                    print(f"  Frame {frame_index}/{len(frames)} ...")

                yolo_dets = hospitalguard._yolo_on_frame(v1, v3, rgb)
                active_dino: dict = {}
                if frame_index % hospitalguard.DINO_VIDEO_INTERVAL == 1 or frame_index == 1:
                    detected_cls = set(yolo_dets.keys())
                    all_dino_targets = set(hospitalguard.DINO_FALLBACK) | set(hospitalguard.DINO_SAHI)
                    missing_weak = [c for c in all_dino_targets if c not in detected_cls]
                    if missing_weak:
                        pil_img = Image.fromarray(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
                        active_dino = hospitalguard.dino_infer(pil_img, missing_weak)

                annotated = hospitalguard.annotate_image(rgb, yolo_dets, active_dino)
                raw_dets = _iter_detections(yolo_dets, active_dino)
                tracked_dets = tracker.update(raw_dets)
                frame_records: list[dict] = []
                for detection in tracked_dets:
                    class_name = detection["class_name"]
                    tracker_id = detection["tracker_id"]
                    all_confs.setdefault(class_name, []).append(detection["confidence"])
                    center_u, center_v, xyz, depth_m = _center_xyz(depth, detection["bbox"], intr)
                    frame_records.append(
                        {
                            "class_name": class_name,
                            "tracker_id": tracker_id,
                            "center_u": center_u,
                            "center_v": center_v,
                            "xyz": xyz,
                        }
                    )

                    x_m = xyz[0] if xyz is not None else None
                    y_m = xyz[1] if xyz is not None else None
                    z_m = xyz[2] if xyz is not None else None
                    csv_writer.writerow([
                        frame_index,
                        f"{frame.timestamp:.6f}",
                        class_name,
                        tracker_id,
                        detection["source"],
                        f"{detection['confidence']:.6f}",
                        center_u,
                        center_v,
                        "" if depth_m is None else f"{depth_m:.6f}",
                        "" if x_m is None else f"{x_m:.6f}",
                        "" if y_m is None else f"{y_m:.6f}",
                        "" if z_m is None else f"{z_m:.6f}",
                    ])

                    if xyz is not None:
                        insert_spatial_memory(conn, f"{frame.timestamp:.6f}", class_name, tracker_id, xyz, session_id)

                _annotate_spatial_points(annotated, frame_records)
                
                if writer is not None:
                    writer.write(annotated)
                spatial_rows.extend(frame_records)
        finally:
            if writer is not None:
                writer.release()
            if conn is not None:
                conn.close()

    flat_dets = {cls: [max(confs)] for cls, confs in all_confs.items()}
    _, conf_str, result_type, _ = hospitalguard.classify_result(expected_class, flat_dets)
    hospitalguard.log_entry(str(sequence_root), expected_class, flat_dets, f"[RGBD-3D {sequence_root.name}]")

    print(f"Saved annotated video: {output_video}")
    print(f"Saved 3D detection CSV: {csv_path}")
    print(f"Saved Excel log: {log_path}")
    print(f"Updated spatial DB: {db_path}")
    print(f"Result: {result_type} | conf: {conf_str}")
    return output_video, csv_path, result_type


def run_video_detection(input_video: Path, expected_class: str, output_video: Path, log_path: Path) -> tuple[Path, dict[str, list[float]], str]:
    DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    hospitalguard.OUT_DIR = output_video.parent
    hospitalguard.EXCEL_PATH = log_path
    hospitalguard.OUT_DIR.mkdir(parents=True, exist_ok=True)
    hospitalguard.EXCEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading V1 (106-class hospital)...")
    v1 = YOLO(str(hospitalguard.V1_PATH))
    print("Loading V3 (109-class)...")
    v3 = YOLO(str(hospitalguard.V3_PATH))

    print(f"Running HospitalGuard on {input_video}...")
    all_confs = hospitalguard.run_video(v1, v3, input_video, output_video)
    flat_dets = {cls: [max(confs)] for cls, confs in all_confs.items()}

    print(f"Saved annotated video: {output_video}")
    print(f"Classes seen across video ({len(flat_dets)}):")
    for cls in sorted(flat_dets, key=lambda name: flat_dets[name][0], reverse=True):
        print(f"  {cls}: max_conf={flat_dets[cls][0]:.3f} detections={len(all_confs[cls])}")

    _, conf_str, result_type, _ = hospitalguard.classify_result(expected_class, flat_dets)
    hospitalguard.log_entry(str(input_video), expected_class, flat_dets, f"[RGBD-DETECT {input_video.stem}]")
    print(f"Result: {result_type} | conf: {conf_str}")
    print(f"Log saved: {log_path}")
    return output_video, all_confs, result_type


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HospitalGuard on RGB-only video or RGB-D sequence and store outputs in rgbd_development/output")
    parser.add_argument("--input-video", type=str, default=None, help="Path to RGB-only MP4 input")
    parser.add_argument("--sequence-root", type=str, default=None, help="Path to TUM RGB-D sequence root containing rgb.txt and depth.txt")
    parser.add_argument("--expected-class", type=str, default="[None]", help="Expected class for result logging")
    parser.add_argument("--output-video", type=str, default=None, help="Annotated output video path")
    parser.add_argument("--log-path", type=str, default=str(DEFAULT_LOG_PATH), help="Excel log path")
    parser.add_argument("--csv-path", type=str, default=None, help="CSV path for per-detection 3D coordinates (sequence mode)")
    parser.add_argument("--db-path", type=str, default=str(DEFAULT_DB_PATH), help="SQLite spatial memory DB path (sequence mode)")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap for number of RGB-D frames to process")
    parser.add_argument("--max-time-diff", type=float, default=0.02, help="Max RGB-depth timestamp difference (seconds)")
    args = parser.parse_args()

    if not args.input_video and not args.sequence_root:
        raise ValueError("Specify either --input-video or --sequence-root.")
    if args.input_video and args.sequence_root:
        raise ValueError("Specify only one of --input-video or --sequence-root.")

    log_path = Path(args.log_path).resolve()

    if args.sequence_root:
        sequence_root = Path(args.sequence_root).resolve()
        if not sequence_root.exists():
            raise FileNotFoundError(f"Sequence root not found: {sequence_root}")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.output_video:
            output_video = Path(args.output_video).resolve()
        else:
            output_video = DETECTIONS_DIR / f"hospitalguard_{sequence_root.name}_rgbd_detected_{ts}.avi"

        if args.csv_path:
            csv_path = Path(args.csv_path).resolve()
        else:
            csv_path = LOGS_DIR / f"spatial_detections_{sequence_root.name}_{ts}.csv"

        db_path = Path(args.db_path).resolve()
        run_rgbd_sequence_detection(
            sequence_root,
            args.expected_class,
            output_video,
            csv_path,
            log_path,
            db_path,
            max_frames=args.max_frames,
            max_time_diff=args.max_time_diff,
        )
        return

    input_video = Path(args.input_video).resolve()
    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    if args.output_video:
        output_video = Path(args.output_video).resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_video = DETECTIONS_DIR / f"hospitalguard_{input_video.stem}_detected_{ts}.mp4"

    run_video_detection(input_video, args.expected_class, output_video, log_path)


if __name__ == "__main__":
    main()
