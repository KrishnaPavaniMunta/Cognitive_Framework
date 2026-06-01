from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from PIL import Image
from ultralytics import YOLO

try:
    import torch
except ImportError:  # pragma: no cover - torch is expected with ultralytics
    torch = None

try:
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover - optional hardware dependency
    rs = None

SCRIPT_DIR = Path(__file__).resolve().parent
RGBD_DEV_DIR = SCRIPT_DIR.parent
ROOT_DIR = RGBD_DEV_DIR.parent.parent
OUTPUT_DIR = RGBD_DEV_DIR / "output"
DETECTIONS_DIR = OUTPUT_DIR / "detections"
LOGS_DIR = OUTPUT_DIR / "logs"
DEFAULT_DB_PATH = OUTPUT_DIR / "hospital_twin.db"

sys.path.insert(0, str(SCRIPT_DIR))
import hospitalguard_temporal_core as temporal  # noqa: E402
from rgbd_spatial_twin import (  # noqa: E402
    build_sequence,
    CameraIntrinsics,
    depth_to_xyz,
    init_db,
    insert_spatial_memory,
    select_intrinsics,
)
from hospital_constants import STATIC_CLASS_NAMES  # noqa: E402


def _center_xyz(depth_img, bbox: tuple[float, float, float, float], intr):
    if depth_img.ndim == 3:
        depth_img = depth_img[:, :, 0]

    h, w = depth_img.shape[:2]
    x1, y1, x2, y2 = bbox
    u = min(max(int(round((x1 + x2) * 0.5)), 0), w - 1)
    v = min(max(int(round((y1 + y2) * 0.5)), 0), h - 1)

    # Robust depth at center using a 5x5 median neighborhood
    y0, y1w = max(0, v - 2), min(h, v + 3)
    x0, x1w = max(0, u - 2), min(w, u + 3)
    win = depth_img[y0:y1w, x0:x1w].astype("float32")
    valid = win[win > 0]
    if valid.size == 0:
        return u, v, None, None

    depth_raw = float(np.median(valid))
    if depth_raw / intr.depth_scale > 8.0:  # reject readings beyond 8 m
        return u, v, None, None
    xyz = depth_to_xyz(u, v, depth_raw, intr)
    return u, v, xyz, depth_raw / intr.depth_scale


def _resolve_output_arg(path_str: str | None, default: Path) -> Path:
    if not path_str:
        return default.resolve()

    path = Path(path_str)
    if path.is_absolute():
        return path.resolve()

    parts = path.parts
    if parts and parts[0].lower() in {"output", "outputs"}:
        path = Path(*parts[1:]) if len(parts) > 1 else Path(path.name)

    if path.parent == Path("."):
        suffix = path.suffix.lower()
        if suffix in {".avi", ".mp4", ".mov", ".mkv"}:
            return (DETECTIONS_DIR / path.name).resolve()
        if suffix in {".csv", ".txt", ".xlsx"}:
            return (LOGS_DIR / path.name).resolve()
        if suffix == ".db":
            return (OUTPUT_DIR / path.name).resolve()

    return (OUTPUT_DIR / path).resolve()


def _open_video_writer(output_video: Path, fps: float, frame_size: tuple[int, int]) -> cv2.VideoWriter:
    suffix = output_video.suffix.lower()
    if suffix in {".mp4", ".m4v", ".mov"}:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    elif suffix == ".avi":
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(str(output_video), fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open writer: {output_video}")
    return writer


def _reuse_recent_tracks(
    tracked_sv: sv.Detections | None,
    last_tracked_sv: sv.Detections | None,
    frame_index: int,
    last_tracked_frame: int,
    fps: float,
) -> sv.Detections | None:
    if tracked_sv is not None and len(tracked_sv) > 0:
        return tracked_sv
    if last_tracked_sv is None or len(last_tracked_sv) == 0:
        return tracked_sv

    carry_frames = max(1, round(fps * 0.25))
    if (frame_index - last_tracked_frame) <= carry_frames:
        return last_tracked_sv
    return tracked_sv


# Anchors not seen for this many frames are considered stale and discarded.
# At 30 fps this is ~3 seconds — long enough to survive brief occlusions.
_STATIC_ANCHOR_EXPIRY_FRAMES = 90


def _stabilise_static_object_ids(
    tracked_sv: sv.Detections | None,
    anchor_state: dict,
    frame_index: int,
    max_match_dist_px: float = 90.0,
) -> sv.Detections | None:
    import copy as _copy

    if tracked_sv is None or len(tracked_sv) == 0 or tracked_sv.tracker_id is None:
        return tracked_sv

    # Expire anchors that have not been updated for _STATIC_ANCHOR_EXPIRY_FRAMES
    for _cls_key in list(anchor_state.keys()):
        _anc = anchor_state[_cls_key].get("anchors", {})
        stale_ids = [aid for aid, a in _anc.items()
                     if (frame_index - a["last_seen"]) > _STATIC_ANCHOR_EXPIRY_FRAMES]
        for _aid in stale_ids:
            del _anc[_aid]

    class_names = tracked_sv.data.get("class_name", np.array([]))
    tracker_ids = tracked_sv.tracker_id.copy()
    updated = False

    for i, box in enumerate(tracked_sv.xyxy):
        class_name = str(class_names[i]) if len(class_names) > i else ""
        if class_name not in STATIC_CLASS_NAMES:
            continue

        state = anchor_state.setdefault(class_name, {"next_id": 0, "anchors": {}})
        anchors = state["anchors"]
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        center_xy = np.asarray([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=float)

        best_anchor_id: int | None = None
        best_dist = float("inf")
        for anchor_id, anchor in anchors.items():
            anchor_xy = np.asarray(anchor["center_xy"], dtype=float)
            dist = float(np.linalg.norm(center_xy - anchor_xy))
            if dist < best_dist:
                best_dist = dist
                best_anchor_id = int(anchor_id)

        if best_anchor_id is not None and best_dist <= max_match_dist_px:
            stable_id = best_anchor_id
        else:
            state["next_id"] += 1
            stable_id = int(state["next_id"])

        anchors[stable_id] = {"center_xy": center_xy.tolist(), "last_seen": frame_index}
        tracker_ids[i] = stable_id
        updated = True

    if not updated:
        return tracked_sv

    stabilised = _copy.copy(tracked_sv)
    stabilised.tracker_id = tracker_ids
    return stabilised


def run_rgbd_temporal(
    sequence_root: Path,
    output_video: Path,
    csv_path: Path,
    db_path: Path,
    log_path: Path,
    expected_class: str,
    max_frames: int | None,
    max_time_diff: float,
    fps: float,
    v1_path: Path,
    v3_path: Path,
    gpu_required: bool = False,
    yolo_half: bool = False,
) -> tuple[Path, Path, Path]:
    DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Keep temporal outputs local to this single folder
    temporal.OUT_DIR = DETECTIONS_DIR
    temporal.EXCEL_PATH = log_path

    init_db(db_path)
    frames = build_sequence(sequence_root, max_time_diff=max_time_diff)
    if max_frames is not None:
        frames = frames[:max_frames]
    if not frames:
        raise RuntimeError(f"No RGB-D frame pairs found in {sequence_root}")

    intr = select_intrinsics(sequence_root.as_posix().lower())
    session_id = str(uuid.uuid4())
    print(f"Session ID: {session_id}")
    print(f"Frames to process: {len(frames)}")

    print("Loading V1 model...")
    v1 = YOLO(str(v1_path))
    print("Loading V3 model...")
    v3 = YOLO(str(v3_path))

    has_cuda = bool(torch is not None and torch.cuda.is_available())
    if gpu_required and not has_cuda:
        raise RuntimeError("CUDA GPU required but not available for sequence mode.")
    yolo_device = "cuda:0" if has_cuda else None
    if has_cuda:
        print("[GPU] Using CUDA for sequence inference (cuda:0)")
    else:
        print("[GPU] CUDA unavailable; running sequence inference on CPU")

    # Video writer (portable codec)
    first_rgb = cv2.imread(str(frames[0].rgb_path), cv2.IMREAD_COLOR)
    if first_rgb is None:
        raise RuntimeError(f"Failed to read first RGB frame: {frames[0].rgb_path}")
    h, w = first_rgb.shape[:2]
    writer = _open_video_writer(output_video, fps, (w, h))

    byte_tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=max(1, round(fps * 3)),
        minimum_matching_threshold=0.6,
        frame_rate=int(fps),
    )

    all_confs: dict[str, list[float]] = defaultdict(list)
    room_state: dict[str, dict] = {}
    id_remap: dict[str, dict] = {}
    static_anchor_state: dict[str, dict] = {}
    motion_state: dict = {}
    last_tracked_sv: sv.Detections | None = None
    last_tracked_frame = 0
    dino_frame_interval = max(1, round(fps * temporal.DINO_VIDEO_INTERVAL_SEC))

    stop_reason = "end_of_sequence"

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
            "session_id",
        ])

        import sqlite3

        conn = sqlite3.connect(db_path)
        try:
            frame_index = 0
            for frame_index, frame in enumerate(frames, start=1):
                rgb = cv2.imread(str(frame.rgb_path), cv2.IMREAD_COLOR)
                depth = cv2.imread(str(frame.depth_path), cv2.IMREAD_UNCHANGED)
                if rgb is None or depth is None:
                    continue

                if frame_index % 50 == 0:
                    print(f"  Frame {frame_index}/{len(frames)}")

                is_glare = temporal._is_overexposed(rgb, threshold=220.0)
                if is_glare and last_tracked_sv is not None:
                    tracked_sv = last_tracked_sv
                else:
                    yolo_dets = temporal._yolo_on_frame(
                        v1,
                        v3,
                        rgb,
                        device=yolo_device,
                        half=bool(yolo_half and has_cuda),
                    )
                    yolo_dets = temporal._promote_worker_aliases(yolo_dets)

                    active_dino: dict = {}
                    if (frame_index - 1) % dino_frame_interval == 0:
                        detected_cls = set(yolo_dets.keys())
                        all_dino_targets = set(temporal.DINO_FALLBACK) | set(temporal.DINO_SAHI)
                        missing_weak = [c for c in all_dino_targets if c not in detected_cls]
                        if missing_weak:
                            pil_img = Image.fromarray(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
                            active_dino = temporal.dino_infer(pil_img, missing_weak)

                    all_dets = {**active_dino, **yolo_dets}
                    combined_sv = temporal._dets_to_sv(all_dets, dino_classes=set(active_dino.keys()))
                    tracked_sv = byte_tracker.update_with_detections(
                        combined_sv if combined_sv is not None else sv.Detections.empty()
                    )

                    reuse_w = max(1, round(fps * 3))
                    tracked_sv = temporal._stabilise_tracker_ids(tracked_sv, id_remap, frame_index, reuse_w)
                    tracked_sv = _stabilise_static_object_ids(tracked_sv, static_anchor_state, frame_index)
                    tracked_sv = temporal._stabilise_ppe_with_worker_motion(
                        tracked_sv,
                        motion_state,
                        frame_index,
                        fps,
                        rgb.shape[:2],
                    )

                    for cls, dets in all_dets.items():
                        for det in dets:
                            all_confs[cls].append(float(det[4]))

                tracked_sv = _reuse_recent_tracks(
                    tracked_sv,
                    last_tracked_sv,
                    frame_index,
                    last_tracked_frame,
                    fps,
                )
                if tracked_sv is not None and len(tracked_sv) > 0:
                    last_tracked_sv = tracked_sv
                    last_tracked_frame = frame_index
                temporal._update_room_state(room_state, tracked_sv, frame_index)

                annotated = temporal._annotate_tracked(rgb, tracked_sv)

                if tracked_sv is not None and len(tracked_sv) > 0:
                    names = tracked_sv.data.get("class_name", [])
                    sources = tracked_sv.data.get("source", ["yolo"] * len(tracked_sv))
                    tids = tracked_sv.tracker_id
                    for i, box in enumerate(tracked_sv.xyxy):
                        class_name = str(names[i]) if len(names) > i else "unknown"
                        source = str(sources[i]) if len(sources) > i else "yolo"
                        conf = float(tracked_sv.confidence[i])
                        tracker_id = int(tids[i]) if tids is not None else (i + 1)
                        u, v, xyz, depth_m = _center_xyz(depth, tuple(box.tolist()), intr)
                        x_m = xyz[0] if xyz is not None else None
                        y_m = xyz[1] if xyz is not None else None
                        z_m = xyz[2] if xyz is not None else None


                        # Only store static/anchor/placed objects in DB
                        if xyz is not None and class_name in STATIC_CLASS_NAMES:
                            insert_spatial_memory(
                                conn,
                                f"{frame.timestamp:.6f}",
                                class_name,
                                tracker_id,
                                xyz,
                                session_id,
                            )

                        csv_writer.writerow([
                            frame_index,
                            f"{frame.timestamp:.6f}",
                            class_name,
                            tracker_id,
                            source,
                            f"{conf:.6f}",
                            u,
                            v,
                            "" if depth_m is None else f"{depth_m:.6f}",
                            "" if x_m is None else f"{x_m:.6f}",
                            "" if y_m is None else f"{y_m:.6f}",
                            "" if z_m is None else f"{z_m:.6f}",
                            session_id,
                        ])

                        if xyz is not None:
                            cv2.putText(
                                annotated,
                                f"X={x_m:.2f} Y={y_m:.2f} Z={z_m:.2f}",
                                (u + 6, max(v - 6, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.45,
                                (0, 255, 0),
                                1,
                                cv2.LINE_AA,
                            )

                writer.write(annotated)
            if max_frames is not None and frame_index >= max_frames:
                stop_reason = f"max_frames({max_frames})"
            else:
                stop_reason = "end_of_sequence"
        finally:
            writer.release()
            conn.close()

    flat_dets = {cls: confs for cls, confs in all_confs.items() if confs}
    temporal.log_entry(str(sequence_root), expected_class, flat_dets, f"[RGBD+Temporal session={session_id}]")

    print(f"Saved video: {output_video}")
    print(f"Saved CSV: {csv_path}")
    print(f"Updated DB: {db_path}")
    print(f"Updated Excel: {log_path}")
    print(f"Stop reason: {stop_reason}, frames written: {frame_index}")
    print(f"Session ID: {session_id}")
    return output_video, csv_path, db_path


def run_realsense_temporal(
    output_video: Path,
    csv_path: Path,
    db_path: Path,
    log_path: Path,
    expected_class: str,
    max_frames: int | None,
    fps: float,
    v1_path: Path,
    v3_path: Path,
    camera_name: str = "freiburg1",
    live_width: int = 640,
    live_height: int = 480,
    live_fps: int = 30,
    live_detect_every: int = 1,
    live_dino_interval_sec: float | None = None,
    live_disable_dino: bool = False,
    db_commit_every: int = 1,
    gpu_required: bool = False,
    yolo_half: bool = False,
) -> tuple[Path, Path, Path]:
    if rs is None:
        raise RuntimeError(
            "pyrealsense2 is not installed in this Python environment. "
            "Install it first, then re-run with --realsense."
        )

    DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    temporal.OUT_DIR = DETECTIONS_DIR
    temporal.EXCEL_PATH = log_path

    init_db(db_path)
    session_id = f"realsense_{camera_name}_{uuid.uuid4()}"
    print(f"Session ID: {session_id}")
    print(f"Starting RealSense live stream {live_width}x{live_height} @ {live_fps}fps")

    has_cuda = bool(torch is not None and torch.cuda.is_available())
    if gpu_required and not has_cuda:
        raise RuntimeError("CUDA GPU is required (--gpu-required), but no CUDA device is available.")
    yolo_device = "cuda:0" if has_cuda else "cpu"
    print(f"YOLO device: {yolo_device}")

    print("Loading V1 model...")
    v1 = YOLO(str(v1_path))
    print("Loading V3 model...")
    v3 = YOLO(str(v3_path))
    if has_cuda:
        v1.to(yolo_device)
        v3.to(yolo_device)
        if yolo_half:
            print("YOLO precision: FP16 (inference mode)")
        else:
            print("YOLO precision: FP32")

    byte_tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=max(1, round(fps * 3)),
        minimum_matching_threshold=0.6,
        frame_rate=int(fps),
    )

    all_confs: dict[str, list[float]] = defaultdict(list)
    room_state: dict[str, dict] = {}
    id_remap: dict[str, dict] = {}
    static_anchor_state: dict[str, dict] = {}
    motion_state: dict = {}
    last_tracked_sv: sv.Detections | None = None
    last_tracked_frame = 0
    dino_sec = live_dino_interval_sec if (live_dino_interval_sec is not None and live_dino_interval_sec > 0) else temporal.DINO_VIDEO_INTERVAL_SEC
    dino_frame_interval = max(1, round(fps * dino_sec))
    detect_every = max(1, int(live_detect_every))
    commit_every = max(1, int(db_commit_every))
    dino_desc = "disabled" if live_disable_dino else f"~{dino_sec:.2f}s"
    print(f"Live smoothness: detect every {detect_every} frame(s), DINO every {dino_desc}, DB commit every {commit_every} inserts")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, live_width, live_height, rs.format.bgr8, live_fps)
    config.enable_stream(rs.stream.depth, live_width, live_height, rs.format.z16, live_fps)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    # Read actual camera intrinsics from the connected RealSense device
    _rs_intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    _rs_depth_m = profile.get_device().first_depth_sensor().get_depth_scale()
    intr = CameraIntrinsics(
        fx=_rs_intr.fx, fy=_rs_intr.fy,
        cx=_rs_intr.ppx, cy=_rs_intr.ppy,
        depth_scale=1.0 / _rs_depth_m,
    )
    print(f"RealSense intrinsics: fx={intr.fx:.1f} fy={intr.fy:.1f} cx={intr.cx:.1f} cy={intr.cy:.1f} depth_scale={intr.depth_scale:.0f}")

    writer: cv2.VideoWriter | None = None
    conn: sqlite3.Connection | None = sqlite3.connect(db_path)

    stop_reason = "manual_stop"

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
            "session_id",
        ])

        frame_index = 0
        pending_db_writes = 0
        try:
            while True:
                frames = align.process(pipeline.wait_for_frames())
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                rgb = np.asanyarray(color_frame.get_data())
                depth = np.asanyarray(depth_frame.get_data())
                frame_index += 1

                if frame_index % 10 == 0:
                    print(f"  Live frame {frame_index}")

                is_glare = temporal._is_overexposed(rgb, threshold=220.0)
                run_inference = (frame_index == 1) or (detect_every <= 1) or ((frame_index - 1) % detect_every == 0)
                if is_glare and last_tracked_sv is not None:
                    tracked_sv = last_tracked_sv
                elif (not run_inference) and last_tracked_sv is not None:
                    # Reuse tracks on skipped inference frames for smoother playback.
                    tracked_sv = last_tracked_sv
                else:
                    yolo_dets = temporal._yolo_on_frame(
                        v1,
                        v3,
                        rgb,
                        device=yolo_device,
                        half=bool(yolo_half and has_cuda),
                    )
                    yolo_dets = temporal._promote_worker_aliases(yolo_dets)

                    active_dino: dict = {}
                    if (not live_disable_dino) and ((frame_index - 1) % dino_frame_interval == 0):
                        detected_cls = set(yolo_dets.keys())
                        all_dino_targets = set(temporal.DINO_FALLBACK) | set(temporal.DINO_SAHI)
                        missing_weak = [c for c in all_dino_targets if c not in detected_cls]
                        if missing_weak:
                            pil_img = Image.fromarray(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
                            active_dino = temporal.dino_infer(pil_img, missing_weak)

                    all_dets = {**active_dino, **yolo_dets}
                    combined_sv = temporal._dets_to_sv(all_dets, dino_classes=set(active_dino.keys()))
                    tracked_sv = byte_tracker.update_with_detections(
                        combined_sv if combined_sv is not None else sv.Detections.empty()
                    )

                    reuse_w = max(1, round(fps * 3))
                    tracked_sv = temporal._stabilise_tracker_ids(tracked_sv, id_remap, frame_index, reuse_w)
                    tracked_sv = _stabilise_static_object_ids(tracked_sv, static_anchor_state, frame_index)
                    tracked_sv = temporal._stabilise_ppe_with_worker_motion(
                        tracked_sv,
                        motion_state,
                        frame_index,
                        fps,
                        rgb.shape[:2],
                    )

                    for cls, dets in all_dets.items():
                        for det in dets:
                            all_confs[cls].append(float(det[4]))

                tracked_sv = _reuse_recent_tracks(
                    tracked_sv,
                    last_tracked_sv,
                    frame_index,
                    last_tracked_frame,
                    fps,
                )
                wrote_db = False
                if tracked_sv is not None and len(tracked_sv) > 0:
                    last_tracked_sv = tracked_sv
                    last_tracked_frame = frame_index
                temporal._update_room_state(room_state, tracked_sv, frame_index)
                annotated = temporal._annotate_tracked(rgb, tracked_sv)

                if tracked_sv is not None and len(tracked_sv) > 0:
                    names = tracked_sv.data.get("class_name", [])
                    sources = tracked_sv.data.get("source", ["yolo"] * len(tracked_sv))
                    tids = tracked_sv.tracker_id
                    for i, box in enumerate(tracked_sv.xyxy):
                        class_name = str(names[i]) if len(names) > i else "unknown"
                        source = str(sources[i]) if len(sources) > i else "yolo"
                        conf = float(tracked_sv.confidence[i])
                        tracker_id = int(tids[i]) if tids is not None else (i + 1)
                        u, v, xyz, depth_m = _center_xyz(depth, tuple(box.tolist()), intr)
                        x_m = xyz[0] if xyz is not None else None
                        y_m = xyz[1] if xyz is not None else None
                        z_m = xyz[2] if xyz is not None else None


                        # Only store static/anchor/placed objects in DB
                        if xyz is not None and class_name in STATIC_CLASS_NAMES:
                            insert_spatial_memory(
                                conn,
                                datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                                class_name,
                                tracker_id,
                                xyz,
                                session_id,
                                auto_commit=False,
                            )
                            wrote_db = True
                            pending_db_writes += 1

                        csv_writer.writerow([
                            frame_index,
                            datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                            class_name,
                            tracker_id,
                            source,
                            f"{conf:.6f}",
                            u,
                            v,
                            "" if depth_m is None else f"{depth_m:.6f}",
                            "" if x_m is None else f"{x_m:.6f}",
                            "" if y_m is None else f"{y_m:.6f}",
                            "" if z_m is None else f"{z_m:.6f}",
                            session_id,
                        ])

                        if xyz is not None:
                            cv2.putText(
                                annotated,
                                f"X={x_m:.2f} Y={y_m:.2f} Z={z_m:.2f}",
                                (u + 6, max(v - 6, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.45,
                                (0, 255, 0),
                                1,
                                cv2.LINE_AA,
                            )

                if wrote_db and conn is not None and pending_db_writes >= commit_every:
                    conn.commit()
                    pending_db_writes = 0

                # Convert depth data to a displayable format
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)

                # Ensure depth_colormap matches annotated dimensions
                if depth_colormap.shape[:2] != annotated.shape[:2]:
                    depth_colormap = cv2.resize(depth_colormap, (annotated.shape[1], annotated.shape[0]))

                # Combine annotated RGB (with coordinates) and depth side by side
                combined_view = np.hstack((annotated, depth_colormap))

                if writer is None:
                    h, w = combined_view.shape[:2]
                    writer = _open_video_writer(output_video, fps, (w, h))

                writer.write(combined_view)
                cv2.imshow("HospitalGuard RealSense Live", combined_view)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    stop_reason = "manual_stop"
                    break
                if max_frames is not None and frame_index >= max_frames:
                    stop_reason = f"max_frames({max_frames})"
                    break
        finally:
            if conn is not None and pending_db_writes > 0:
                conn.commit()
            pipeline.stop()
            if writer is not None:
                writer.release()
            if conn is not None:
                conn.close()
            cv2.destroyAllWindows()

    flat_dets = {cls: confs for cls, confs in all_confs.items() if confs}
    temporal.log_entry("realsense_live", expected_class, flat_dets, f"[RealSense RGBD+Temporal session={session_id}]")

    print(f"Saved video: {output_video}")
    print(f"Saved CSV: {csv_path}")
    print(f"Updated DB: {db_path}")
    print(f"Updated Excel: {log_path}")
    print(f"Stop reason: {stop_reason}, frames written: {frame_index}")
    print(f"Session ID: {session_id}")
    return output_video, csv_path, db_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified RGB-D + HospitalGuard Temporal (ByteTrack) runner")
    parser.add_argument("--sequence-root", type=str, default=None, help="Path to RGB-D sequence folder")
    parser.add_argument("--realsense", action="store_true", help="Use a connected Intel RealSense camera instead of a sequence folder")
    parser.add_argument("--camera-name", type=str, default="freiburg1", help="Camera intrinsics profile to use for live RealSense")
    parser.add_argument("--expected-class", type=str, default="hospital_room", help="Expected class for Excel summary")
    parser.add_argument("--output-video", type=str, default=None, help="Output annotated AVI path")
    parser.add_argument("--csv-path", type=str, default=None, help="Output CSV path")
    parser.add_argument("--db-path", type=str, default=str(DEFAULT_DB_PATH), help="SQLite DB path")
    parser.add_argument("--log-path", type=str, default=str(LOGS_DIR / "hospitalguard_temporal_rgbd_log.xlsx"), help="Excel log path")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame cap")
    parser.add_argument("--max-time-diff", type=float, default=0.02, help="RGB-depth timestamp max delta")
    parser.add_argument("--fps", type=float, default=30.0, help="Output FPS used by tracker/writer")
    parser.add_argument("--live-detect-every", type=int, default=1, help="Run YOLO every N live frames (>=1). Higher = smoother UI, lower detection refresh")
    parser.add_argument("--live-dino-interval-sec", type=float, default=3.5, help="Live DINO interval in seconds. Higher = fewer latency spikes")
    parser.add_argument("--live-disable-dino", action="store_true", help="Disable DINO fallback in live mode for maximum smoothness")
    parser.add_argument("--db-commit-every", type=int, default=1, help="Commit DB every N inserts in live mode. Higher = smoother playback")
    parser.add_argument("--gpu-required", action="store_true", help="Fail if CUDA GPU is not available")
    parser.add_argument("--yolo-half", action="store_true", help="Use FP16 for YOLO on CUDA for faster inference")
    parser.add_argument("--live-ultra-smooth", action="store_true", help="Preset: GPU required + FP16 + detect every 2 frames + DINO off + batched DB commits")
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

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    db_path = _resolve_output_arg(args.db_path, DEFAULT_DB_PATH)
    log_path = _resolve_output_arg(args.log_path, LOGS_DIR / "hospitalguard_temporal_rgbd_log.xlsx")

    if args.live_ultra_smooth:
        args.live_detect_every = max(2, args.live_detect_every)
        args.live_disable_dino = True
        args.db_commit_every = max(20, args.db_commit_every)
        args.gpu_required = True
        args.yolo_half = True

    if args.realsense:
        output_video = _resolve_output_arg(args.output_video, DETECTIONS_DIR / f"hospitalguard_realsense_temporal_{ts}.mp4")
        csv_path = _resolve_output_arg(args.csv_path, LOGS_DIR / f"spatial_realsense_temporal_{ts}.csv")
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
            live_detect_every=args.live_detect_every,
            live_dino_interval_sec=args.live_dino_interval_sec,
            live_disable_dino=args.live_disable_dino,
            db_commit_every=args.db_commit_every,
            gpu_required=args.gpu_required,
            yolo_half=args.yolo_half,
        )
        return

    if not args.sequence_root:
        raise ValueError("Specify either --sequence-root or --realsense.")

    sequence_root = Path(args.sequence_root).resolve()
    output_video = _resolve_output_arg(args.output_video, DETECTIONS_DIR / f"hospitalguard_temporal_rgbd_{sequence_root.name}_{ts}.mp4")
    csv_path = _resolve_output_arg(args.csv_path, LOGS_DIR / f"spatial_temporal_{sequence_root.name}_{ts}.csv")

    run_rgbd_temporal(
        sequence_root=sequence_root,
        output_video=output_video,
        csv_path=csv_path,
        db_path=db_path,
        log_path=log_path,
        expected_class=args.expected_class,
        max_frames=args.max_frames,
        max_time_diff=args.max_time_diff,
        fps=args.fps,
        v1_path=Path(args.v1_path).resolve(),
        v3_path=Path(args.v3_path).resolve(),
        gpu_required=args.gpu_required,
        yolo_half=args.yolo_half,
    )


if __name__ == "__main__":
    main()
