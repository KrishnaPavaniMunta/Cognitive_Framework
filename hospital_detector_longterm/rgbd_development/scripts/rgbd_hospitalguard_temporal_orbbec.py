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
except ImportError:  # pragma: no cover
    torch = None

try:
    from pyorbbecsdk import (
        AlignFilter,
        Config,
        OBFormat,
        OBSensorType,
        OBStreamType,
        Pipeline,
    )
except ImportError:  # pragma: no cover
    AlignFilter = None
    Config = None
    OBFormat = None
    OBSensorType = None
    OBStreamType = None
    Pipeline = None

SCRIPT_DIR = Path(__file__).resolve().parent
RGBD_DEV_DIR = SCRIPT_DIR.parent
ROOT_DIR = RGBD_DEV_DIR.parent.parent
OUTPUT_DIR = RGBD_DEV_DIR / "output"
DETECTIONS_DIR = OUTPUT_DIR / "detections"
LOGS_DIR = OUTPUT_DIR / "logs"
DEFAULT_DB_PATH = OUTPUT_DIR / "hospital_twin.db"

sys.path.insert(0, str(SCRIPT_DIR))
import hospitalguard_temporal_core as temporal  # noqa: E402
from hospital_constants import STATIC_CLASS_NAMES  # noqa: E402
from rgbd_hospitalguard_temporal import (  # noqa: E402
    _center_xyz,
    _open_video_writer,
    _resolve_output_arg,
    _reuse_recent_tracks,
    _stabilise_static_object_ids,
)
from rgbd_spatial_twin import (  # noqa: E402
    CameraIntrinsics,
    init_db,
    insert_spatial_memory,
    select_intrinsics,
)

try:
    from openni import openni2
except ImportError:  # pragma: no cover
    openni2 = None


def _find_profile(profile_list, width: int, height: int, fmt=None):
    profiles = profile_list.get_video_stream_profile_list()
    for profile in profiles:
        if profile.get_width() == width and profile.get_height() == height:
            if fmt is None or profile.get_format() == fmt:
                return profile
    return None


def _to_bgr(color_frame) -> np.ndarray | None:
    width = color_frame.get_width()
    height = color_frame.get_height()
    color_format = color_frame.get_format()
    data = np.asanyarray(color_frame.get_data())

    if color_format == OBFormat.RGB:
        img = np.resize(data, (height, width, 3))
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if color_format == OBFormat.BGR:
        return np.resize(data, (height, width, 3))
    if color_format == OBFormat.YUYV:
        img = np.resize(data, (height, width, 2))
        return cv2.cvtColor(img, cv2.COLOR_YUV2BGR_YUYV)
    if color_format == OBFormat.UYVY:
        img = np.resize(data, (height, width, 2))
        return cv2.cvtColor(img, cv2.COLOR_YUV2BGR_UYVY)
    if color_format == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    return None


def _depth_scale_to_intr(depth_scale: float) -> float:
    if depth_scale <= 0:
        raise ValueError(f"Invalid Orbbec depth scale: {depth_scale}")
    # SDKs vary by device family. Handle both common conventions:
    # - meters per depth unit (small value e.g. 0.001)
    # - millimeters per depth unit (value >= 0.01)
    if depth_scale < 0.01:
        return 1.0 / depth_scale
    return 1000.0 / depth_scale


def _resolve_openni_redist(openni_redist: str | None) -> str:
    if openni_redist:
        return openni_redist
    return (
        r"C:\Users\Krishna.Munta\Downloads\Orbbec_OpenNI_v2.3.0.86-beta6_windows_release"
        r"\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows"
        r"\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows"
        r"\Win64-Release\sdk\libs"
    )


def _open_color_camera(color_index: int) -> cv2.VideoCapture:
    # Prefer DSHOW over MSMF on Windows to reduce camera lockups on Astra UVC.
    backend_order = (cv2.CAP_DSHOW, cv2.CAP_ANY, cv2.CAP_MSMF)

    if color_index < 0:
        indices = (1, 2, 3, 4)
    else:
        indices = (color_index,)

    for idx in indices:
        for backend in backend_order:
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                cap.release()
                continue
            ok, frame = cap.read()
            if ok and frame is not None:
                print(f"Opened RGB camera index={idx} backend={backend}")
                return cap
            cap.release()

    idx_msg = "auto-scan (1..4)" if color_index < 0 else str(color_index)
    raise RuntimeError(
        f"Failed to open Orbbec RGB camera at index {idx_msg}. "
        "Close apps using camera and retry with --color-index 1 or 2."
    )


def run_orbbec_temporal_openni(
    output_video: Path,
    csv_path: Path,
    db_path: Path,
    log_path: Path,
    expected_class: str,
    max_frames: int | None,
    fps: float,
    v1_path: Path,
    v3_path: Path,
    camera_name: str = "orbbec_openni",
    live_width: int = 640,
    live_height: int = 480,
    live_fps: int = 30,
    live_detect_every: int = 1,
    live_dino_interval_sec: float | None = None,
    live_disable_dino: bool = False,
    db_commit_every: int = 1,
    gpu_required: bool = False,
    yolo_half: bool = False,
    openni_redist: str | None = None,
    color_index: int = -1,
    intrinsics_profile: str = "freiburg1",
) -> tuple[Path, Path, Path]:
    if openni2 is None:
        raise RuntimeError(
            "openni package is not installed in this Python environment. "
            "Use your OpenNI environment and install openni."
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
    session_id = f"orbbec_openni_{camera_name}_{uuid.uuid4()}"
    print(f"Session ID: {session_id}")
    print(f"Starting Orbbec OpenNI stream {live_width}x{live_height} @ {live_fps}fps")

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
        print("YOLO precision: FP16 (inference mode)" if yolo_half else "YOLO precision: FP32")

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

    redist_path = _resolve_openni_redist(openni_redist)
    openni2_inited = False
    dev = None
    depth_stream = None
    cap = None
    try:
        openni2.initialize(redist_path)
        openni2_inited = True
        dev = openni2.Device.open_any()
        depth_stream = dev.create_depth_stream()
        depth_stream.start()
        cap = _open_color_camera(color_index)
    except Exception:
        # Ensure hardware handles are released even if setup fails before the main loop.
        if cap is not None:
            cap.release()
        if depth_stream is not None:
            try:
                depth_stream.stop()
            except Exception:
                pass
        if dev is not None:
            try:
                dev.close()
            except Exception:
                pass
        if openni2_inited:
            try:
                openni2.unload()
            except Exception:
                pass
        raise

    intr_seed = select_intrinsics(intrinsics_profile)
    intr = CameraIntrinsics(
        fx=intr_seed.fx,
        fy=intr_seed.fy,
        cx=intr_seed.cx,
        cy=intr_seed.cy,
        depth_scale=1000.0,
    )
    print(
        "OpenNI intrinsics profile: "
        f"{intrinsics_profile} (fx={intr.fx:.1f} fy={intr.fy:.1f} cx={intr.cx:.1f} cy={intr.cy:.1f} depth_scale={intr.depth_scale:.0f})"
    )

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
                ok, rgb = cap.read()
                if not ok or rgb is None:
                    continue
                if rgb.shape[1] != live_width or rgb.shape[0] != live_height:
                    rgb = cv2.resize(rgb, (live_width, live_height))

                depth_frame = depth_stream.read_frame()
                depth = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16).reshape(
                    (depth_frame.height, depth_frame.width)
                )
                if depth.shape[1] != live_width or depth.shape[0] != live_height:
                    depth = cv2.resize(depth, (live_width, live_height), interpolation=cv2.INTER_NEAREST)

                frame_index += 1
                if frame_index % 10 == 0:
                    print(f"  Live frame {frame_index}")

                is_glare = temporal._is_overexposed(rgb, threshold=220.0)
                run_inference = (frame_index == 1) or (detect_every <= 1) or ((frame_index - 1) % detect_every == 0)
                if is_glare and last_tracked_sv is not None:
                    tracked_sv = last_tracked_sv
                elif (not run_inference) and last_tracked_sv is not None:
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

                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
                if depth_colormap.shape[:2] != annotated.shape[:2]:
                    depth_colormap = cv2.resize(depth_colormap, (annotated.shape[1], annotated.shape[0]))
                combined_view = np.hstack((annotated, depth_colormap))

                if writer is None:
                    h, w = combined_view.shape[:2]
                    writer = _open_video_writer(output_video, fps, (w, h))

                writer.write(combined_view)
                cv2.imshow("HospitalGuard Orbbec OpenNI Live", combined_view)
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
            if writer is not None:
                writer.release()
            if conn is not None:
                conn.close()
            if cap is not None:
                cap.release()
            if depth_stream is not None:
                try:
                    depth_stream.stop()
                except Exception:
                    pass
            if dev is not None:
                try:
                    dev.close()
                except Exception:
                    pass
            if openni2_inited:
                try:
                    openni2.unload()
                except Exception:
                    pass
            cv2.destroyAllWindows()

    flat_dets = {cls: confs for cls, confs in all_confs.items() if confs}
    temporal.log_entry("orbbec_openni_live", expected_class, flat_dets, f"[Orbbec OpenNI RGBD+Temporal session={session_id}]")

    print(f"Saved video: {output_video}")
    print(f"Saved CSV: {csv_path}")
    print(f"Updated DB: {db_path}")
    print(f"Updated Excel: {log_path}")
    print(f"Stop reason: {stop_reason}, frames written: {frame_index}")
    print(f"Session ID: {session_id}")
    return output_video, csv_path, db_path


def run_orbbec_temporal(
    output_video: Path,
    csv_path: Path,
    db_path: Path,
    log_path: Path,
    expected_class: str,
    max_frames: int | None,
    fps: float,
    v1_path: Path,
    v3_path: Path,
    camera_name: str = "orbbec",
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
    if Pipeline is None or Config is None:
        raise RuntimeError(
            "pyorbbecsdk is not installed in this Python environment. "
            "Activate your Orbbec environment and install pyorbbecsdk2."
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
    session_id = f"orbbec_{camera_name}_{uuid.uuid4()}"
    print(f"Session ID: {session_id}")
    print(f"Starting Orbbec live stream {live_width}x{live_height} @ {live_fps}fps")

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
        print("YOLO precision: FP16 (inference mode)" if yolo_half else "YOLO precision: FP32")

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

    pipeline = Pipeline()
    config = Config()

    color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)

    color_profile = _find_profile(color_profiles, live_width, live_height, OBFormat.RGB)
    if color_profile is None:
        color_profile = color_profiles.get_default_video_stream_profile()
    config.enable_stream(color_profile)

    depth_profile = _find_profile(depth_profiles, live_width, live_height)
    if depth_profile is None:
        depth_profile = depth_profiles.get_default_video_stream_profile()
    config.enable_stream(depth_profile)

    pipeline.start(config)
    align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

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
        intr: CameraIntrinsics | None = None
        try:
            while True:
                frames = pipeline.wait_for_frames(200)
                if frames is None:
                    continue
                frames = align_filter.process(frames)
                if frames is None:
                    continue

                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                rgb = _to_bgr(color_frame)
                if rgb is None:
                    continue

                depth = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
                    (depth_frame.get_height(), depth_frame.get_width())
                )

                if intr is None:
                    cam_param = pipeline.get_camera_param()
                    rgb_intr = cam_param.rgb_intrinsic
                    depth_scale = float(depth_frame.get_depth_scale())
                    intr = CameraIntrinsics(
                        fx=float(rgb_intr.fx),
                        fy=float(rgb_intr.fy),
                        cx=float(rgb_intr.cx),
                        cy=float(rgb_intr.cy),
                        depth_scale=_depth_scale_to_intr(depth_scale),
                    )
                    print(
                        "Orbbec intrinsics: "
                        f"fx={intr.fx:.1f} fy={intr.fy:.1f} cx={intr.cx:.1f} cy={intr.cy:.1f} depth_scale={intr.depth_scale:.0f}"
                    )

                frame_index += 1
                if frame_index % 10 == 0:
                    print(f"  Live frame {frame_index}")

                is_glare = temporal._is_overexposed(rgb, threshold=220.0)
                run_inference = (frame_index == 1) or (detect_every <= 1) or ((frame_index - 1) % detect_every == 0)
                if is_glare and last_tracked_sv is not None:
                    tracked_sv = last_tracked_sv
                elif (not run_inference) and last_tracked_sv is not None:
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

                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
                if depth_colormap.shape[:2] != annotated.shape[:2]:
                    depth_colormap = cv2.resize(depth_colormap, (annotated.shape[1], annotated.shape[0]))
                combined_view = np.hstack((annotated, depth_colormap))

                if writer is None:
                    h, w = combined_view.shape[:2]
                    writer = _open_video_writer(output_video, fps, (w, h))

                writer.write(combined_view)
                cv2.imshow("HospitalGuard Orbbec Live", combined_view)
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
    temporal.log_entry("orbbec_live", expected_class, flat_dets, f"[Orbbec RGBD+Temporal session={session_id}]")

    print(f"Saved video: {output_video}")
    print(f"Saved CSV: {csv_path}")
    print(f"Updated DB: {db_path}")
    print(f"Updated Excel: {log_path}")
    print(f"Stop reason: {stop_reason}, frames written: {frame_index}")
    print(f"Session ID: {session_id}")
    return output_video, csv_path, db_path


def main() -> None:
    parser = argparse.ArgumentParser(description="HospitalGuard Temporal live runner for Orbbec RGB-D cameras")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "pyorbbec", "openni"], help="Camera backend")
    parser.add_argument("--expected-class", type=str, default="hospital_room", help="Expected class for Excel summary")
    parser.add_argument("--camera-name", type=str, default="orbbec", help="Session label only")
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
    parser.add_argument("--live-width", type=int, default=640, help="Orbbec color/depth width")
    parser.add_argument("--live-height", type=int, default=480, help="Orbbec color/depth height")
    parser.add_argument("--live-fps", type=int, default=30, help="Orbbec stream FPS")
    parser.add_argument("--live-detect-every", type=int, default=1, help="Run YOLO every N live frames")
    parser.add_argument("--live-dino-interval-sec", type=float, default=3.5, help="DINO interval in seconds")
    parser.add_argument("--live-disable-dino", action="store_true", help="Disable DINO fallback")
    parser.add_argument("--db-commit-every", type=int, default=1, help="Commit DB every N inserts")
    parser.add_argument("--openni-redist", type=str, default=None, help="Path to OpenNI2 redist libs folder")
    parser.add_argument("--color-index", type=int, default=-1, help="OpenCV camera index for Orbbec RGB in OpenNI mode (-1 = auto scan 1..4)")
    parser.add_argument("--intrinsics-profile", type=str, default="freiburg1", help="Intrinsics profile for OpenNI depth-to-XYZ")
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
        DETECTIONS_DIR / f"hospitalguard_orbbec_temporal_{ts}.mp4",
    )
    csv_path = _resolve_output_arg(
        args.csv_path,
        LOGS_DIR / f"spatial_orbbec_temporal_{ts}.csv",
    )
    db_path = _resolve_output_arg(args.db_path, DEFAULT_DB_PATH)
    log_path = _resolve_output_arg(args.log_path, LOGS_DIR / "hospitalguard_temporal_rgbd_log.xlsx")

    run_kwargs = dict(
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

    if args.backend == "pyorbbec":
        run_orbbec_temporal(**run_kwargs)
        return
    if args.backend == "openni":
        run_orbbec_temporal_openni(
            **run_kwargs,
            openni_redist=args.openni_redist,
            color_index=args.color_index,
            intrinsics_profile=args.intrinsics_profile,
        )
        return

    try:
        run_orbbec_temporal(**run_kwargs)
    except Exception as ex:
        print(f"pyorbbec backend failed ({ex}). Falling back to OpenNI backend...")
        run_orbbec_temporal_openni(
            **run_kwargs,
            openni_redist=args.openni_redist,
            color_index=args.color_index,
            intrinsics_profile=args.intrinsics_profile,
        )


if __name__ == "__main__":
    main()
