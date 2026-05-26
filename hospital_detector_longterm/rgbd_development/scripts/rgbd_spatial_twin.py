"""
rgbd_spatial_twin.py

Minimal RGB-D prototype for long-term spatial memory.

What it does:
- Creates a local SQLite database with a spatial_memory table.
- Replays a TUM RGB-D dataset folder (rgb.txt + depth.txt + groundtruth/associations).
- Displays RGB and depth frames side by side.
- Stores per-frame 3D points for a selected pixel or future detector output.

This is intentionally small so it can be extended later with the hospital YOLO + DINO + ByteTrack pipeline.
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover - optional hardware dependency
    rs = None

SCRIPT_DIR = Path(__file__).resolve().parent
RGBD_DEV_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = RGBD_DEV_DIR / "output"
EXPORTS_DIR = OUTPUT_DIR / "exports"
DB_DEFAULT = OUTPUT_DIR / "hospital_twin.db"
TABLE_NAME = "spatial_memory"


@dataclass
class RGBDFrame:
    timestamp: float
    rgb_path: Path
    depth_path: Path


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale: float = 5000.0


TUM_INTRINSICS = {
    "freiburg1": CameraIntrinsics(fx=517.3, fy=516.5, cx=318.6, cy=255.3),
    "freiburg2": CameraIntrinsics(fx=520.9, fy=521.0, cx=325.1, cy=249.7),
    "freiburg3": CameraIntrinsics(fx=535.4, fy=539.2, cx=320.1, cy=247.6),
}


def init_db(db_path: str | Path = DB_DEFAULT) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL DEFAULT '',
                timestamp TEXT NOT NULL,
                class_name TEXT NOT NULL,
                tracker_id INTEGER NOT NULL,
                X REAL NOT NULL,
                Y REAL NOT NULL,
                Z REAL NOT NULL,
                last_seen TEXT NOT NULL
            )
            """
        )
        # Add columns to existing DBs that pre-date this schema
        for col, typedef in (("session_id", "TEXT NOT NULL DEFAULT ''"),
                             ("last_seen", "TEXT NOT NULL DEFAULT ''")):
            try:
                conn.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN {col} {typedef}")
            except sqlite3.OperationalError:
                pass  # column already exists
        conn.commit()
    finally:
        conn.close()


def load_assoc_file(path: Path) -> list[tuple[float, str]]:
    entries: list[tuple[float, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            ts, rel_path = raw.split(maxsplit=1)
            entries.append((float(ts), rel_path))
    return entries


def build_sequence(root: Path, max_time_diff: float = 0.02) -> list[RGBDFrame]:
    rgb_entries = load_assoc_file(root / "rgb.txt")
    depth_entries = load_assoc_file(root / "depth.txt")

    # TUM RGB and depth timestamps are not exactly equal; pair nearest timestamps.
    frames: list[RGBDFrame] = []
    if not rgb_entries or not depth_entries:
        return frames

    j = 0
    depth_n = len(depth_entries)
    for ts, rel in rgb_entries:
        while j + 1 < depth_n and depth_entries[j + 1][0] <= ts:
            j += 1

        candidates: list[tuple[float, str]] = [depth_entries[j]]
        if j + 1 < depth_n:
            candidates.append(depth_entries[j + 1])

        depth_ts, depth_rel = min(candidates, key=lambda item: abs(item[0] - ts))
        if abs(depth_ts - ts) > max_time_diff:
            continue

        frames.append(RGBDFrame(timestamp=ts, rgb_path=root / rel, depth_path=root / depth_rel))
    return frames


def select_intrinsics(sequence_name: str) -> CameraIntrinsics:
    for key, intrinsics in TUM_INTRINSICS.items():
        if key in sequence_name:
            return intrinsics
    return TUM_INTRINSICS["freiburg1"]


def depth_to_xyz(u: int, v: int, depth_raw: float, intr: CameraIntrinsics) -> tuple[float, float, float]:
    z = depth_raw / intr.depth_scale
    x = (u - intr.cx) * z / intr.fx
    y = (v - intr.cy) * z / intr.fy
    return x, y, z


def sample_depth(depth_img: np.ndarray, u: int, v: int, intr: CameraIntrinsics, window: int = 5) -> tuple[float, float, float] | None:
    h, w = depth_img.shape[:2]
    x0 = max(0, u - window)
    y0 = max(0, v - window)
    x1 = min(w, u + window + 1)
    y1 = min(h, v + window + 1)
    patch = depth_img[y0:y1, x0:x1]
    valid = patch[patch > 0]
    if valid.size == 0:
        return None
    depth_raw = float(np.median(valid))
    return depth_to_xyz(u, v, depth_raw, intr)


def run_realsense_live(
    db_path: str | Path = DB_DEFAULT,
    write_db: bool = True,
    max_frames: Optional[int] = None,
    output_video: Optional[str | Path] = None,
    camera_name: str = "freiburg1",
) -> None:
    if rs is None:
        raise RuntimeError(
            "pyrealsense2 is not installed in this Python environment. "
            "Install it first, then re-run with --realsense."
        )

    db_path = Path(db_path)
    if output_video is not None:
        output_video = Path(output_video)

    if write_db:
        init_db(db_path)

    intr = TUM_INTRINSICS.get(camera_name, TUM_INTRINSICS["freiburg1"])
    conn: Optional[sqlite3.Connection] = sqlite3.connect(db_path) if write_db else None
    writer: Optional[cv2.VideoWriter] = None
    session_id = f"realsense_{camera_name}"

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    try:
        frame_index = 0
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())
            depth_color = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_color, cv2.COLORMAP_TURBO)
            if depth_color.shape[:2] != color.shape[:2]:
                depth_color = cv2.resize(depth_color, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST)

            frame_index += 1
            combined = np.hstack([color, depth_color])
            cv2.putText(
                combined,
                f"RealSense RGB-D live | frame={frame_index}",
                (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if writer is None and output_video is not None:
                output_video.parent.mkdir(parents=True, exist_ok=True)
                h, w = combined.shape[:2]
                writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open video writer for {output_video}")

            if writer is not None:
                writer.write(combined)

            h, w = depth.shape[:2]
            u, v = w // 2, h // 2
            xyz = sample_depth(depth, u, v, intr)
            if xyz is not None and conn is not None:
                stamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
                insert_spatial_memory(conn, stamp, "center_pixel", frame_index, xyz, session_id)

            cv2.imshow("RealSense RGB-D Live", combined)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if max_frames is not None and frame_index >= max_frames:
                break
    finally:
        pipeline.stop()
        if writer is not None:
            writer.release()
        if conn is not None:
            conn.close()
        cv2.destroyAllWindows()


def insert_spatial_memory(
    conn: sqlite3.Connection,
    timestamp: str,
    class_name: str,
    tracker_id: int,
    xyz: tuple[float, float, float],
    session_id: str = "",
    auto_commit: bool = True,
) -> None:
    """Insert or update the latest known position for this (session, tracker_id, class_name)."""
    import datetime as _dt
    last_seen = _dt.datetime.now(_dt.timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")
    # Upsert: update X/Y/Z and last_seen if same tracker in this session; insert if new
    existing = conn.execute(
        f"SELECT id FROM {TABLE_NAME} WHERE session_id=? AND tracker_id=? AND class_name=?",
        (session_id, tracker_id, class_name),
    ).fetchone()
    if existing:
        conn.execute(
            f"UPDATE {TABLE_NAME} SET X=?, Y=?, Z=?, last_seen=?, timestamp=? WHERE id=?",
            (xyz[0], xyz[1], xyz[2], last_seen, timestamp, existing[0]),
        )
    else:
        conn.execute(
            f"INSERT INTO {TABLE_NAME} (session_id, timestamp, class_name, tracker_id, X, Y, Z, last_seen) "
            f"VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, timestamp, class_name, tracker_id, xyz[0], xyz[1], xyz[2], last_seen),
        )
    if auto_commit:
        conn.commit()


def replay_dataset(
    sequence_root: Path,
    db_path: str | Path = DB_DEFAULT,
    write_db: bool = True,
    max_frames: Optional[int] = None,
    wait_ms: int = 1,
    max_time_diff: float = 0.02,
    output_video: Optional[str | Path] = None,
) -> None:
    db_path = Path(db_path)
    if output_video is not None:
        output_video = Path(output_video)

    if write_db:
        init_db(db_path)
    frames = build_sequence(sequence_root, max_time_diff=max_time_diff)
    if not frames:
        raise RuntimeError(
            f"No RGB-depth frame pairs were found in {sequence_root}. "
            "Check path and increase --max-time-diff if needed."
        )
    intr = select_intrinsics(sequence_root.as_posix().lower())

    conn: Optional[sqlite3.Connection] = sqlite3.connect(db_path) if write_db else None
    writer: Optional[cv2.VideoWriter] = None
    video_frame_size: Optional[tuple[int, int]] = None
    
    try:
        for i, frame in enumerate(frames):
            if max_frames is not None and i >= max_frames:
                break

            rgb = cv2.imread(str(frame.rgb_path), cv2.IMREAD_COLOR)
            depth = cv2.imread(str(frame.depth_path), cv2.IMREAD_UNCHANGED)
            if rgb is None or depth is None:
                continue

            display_depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            display_depth = cv2.applyColorMap(display_depth, cv2.COLORMAP_TURBO)
            if display_depth.shape[:2] != rgb.shape[:2]:
                display_depth = cv2.resize(display_depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

            combined = np.hstack([rgb, display_depth])
            cv2.putText(combined, f"frame={i+1}/{len(frames)}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Initialize VideoWriter on first valid frame
            if output_video is not None and writer is None:
                output_video.parent.mkdir(parents=True, exist_ok=True)
                h, w = combined.shape[:2]
                video_frame_size = (w, h)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = 30.0
                writer = cv2.VideoWriter(str(output_video), fourcc, fps, video_frame_size)
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open video writer for {output_video}")
                print(f"Writing video to {output_video} at {fps} FPS, size {w}x{h}")
            
            # Write to video or display
            if writer is not None:
                writer.write(combined)
            else:
                cv2.imshow("RGB-D Twin Prototype", combined)

            h, w = depth.shape[:2]
            u, v = w // 2, h // 2
            xyz = sample_depth(depth, u, v, intr)
            if xyz is not None:
                stamp = f"{frame.timestamp:.6f}"
                if conn is not None:
                    insert_spatial_memory(conn, stamp, "center_pixel", 0, xyz)
                print(f"{stamp} center_pixel: X={xyz[0]:.2f}, Y={xyz[1]:.2f}, Z={xyz[2]:.2f}")

            if writer is None:
                key = cv2.waitKey(wait_ms) & 0xFF
                if key in (ord('q'), 27):
                    break
    finally:
        if conn is not None:
            conn.close()
        if writer is not None:
            writer.release()
            print(f"Video export complete: {output_video}")
        cv2.destroyAllWindows()


def download_and_extract_tum_sequence(url: str, extract_to: Path) -> None:
    """Download and extract a TUM RGB-D sequence from a given URL."""
    print(f"Downloading TUM RGB-D sequence from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        print(f"Extracting sequence to {extract_to}...")
        z.extractall(extract_to)

    print("Download and extraction complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal RGB-D spatial memory prototype")
    parser.add_argument("--sequence-root", type=str, help="Path to a TUM RGB-D sequence folder containing rgb.txt and depth.txt")
    parser.add_argument("--db", type=str, default=str(DB_DEFAULT), help="SQLite database path")
    parser.add_argument("--online-url", type=str, help="URL to download a TUM RGB-D sequence")
    parser.add_argument("--realsense", action="store_true", help="Use a connected Intel RealSense camera instead of a TUM dataset")
    parser.add_argument("--realsense-camera-name", type=str, default="freiburg1", help="Camera intrinsics profile to use for live RealSense depth to XYZ")
    parser.add_argument("--no-db", action="store_true", help="Viewer mode: do not write points to SQLite")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap for number of frames to replay")
    parser.add_argument("--wait-ms", type=int, default=1, help="Delay between frames in ms (higher = slower playback)")
    parser.add_argument("--max-time-diff", type=float, default=0.02, help="Max RGB-depth timestamp difference (seconds)")
    parser.add_argument("--output-video", type=str, default=None, help="Optional MP4 file path to export side-by-side RGB-depth video")
    args = parser.parse_args()

    if args.online_url:
        if not args.sequence_root:
            raise ValueError("--sequence-root must be specified to extract the downloaded sequence.")
        sequence_root = Path(args.sequence_root)
        sequence_root.mkdir(parents=True, exist_ok=True)
        download_and_extract_tum_sequence(args.online_url, sequence_root)

    if args.realsense:
        run_realsense_live(
            db_path=args.db,
            write_db=not args.no_db,
            max_frames=args.max_frames,
            output_video=args.output_video,
            camera_name=args.realsense_camera_name,
        )
        return

    if not args.sequence_root:
        raise ValueError("Either --sequence-root or --online-url must be specified.")

    replay_dataset(
        Path(args.sequence_root),
        args.db,
        write_db=not args.no_db,
        max_frames=args.max_frames,
        wait_ms=args.wait_ms,
        max_time_diff=args.max_time_diff,
        output_video=args.output_video,
    )


if __name__ == "__main__":
    main()
