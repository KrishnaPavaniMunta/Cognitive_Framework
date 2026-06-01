from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RGBD_DEV_DIR = SCRIPT_DIR.parent
LOGS_DIR = RGBD_DEV_DIR / "output" / "logs"
PLOTS_DIR = RGBD_DEV_DIR / "output" / "plots"

STATIC_CLASS_NAMES: frozenset[str] = frozenset({
    "hospital_bed", "infusion_pump", "iv_stand", "monitor_hosp",
    "patient_monitor", "surgical_light", "vending_machines", "wheelchair",
    "hospital_stretcher", "cabinet", "bench_hosp", "door", "reception_counter",
    "radiator", "bathroom_labels", "fire_extinguisher", "security_camera",
    "exit_sign", "iv_bag", "test_tube", "surgical_scissor", "spillage",
    "bench", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "microwave", "oven", "toaster", "sink", "refrigerator",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "bag",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake",
})


def _latest_csv(logs_dir: Path) -> Path:
    candidates = sorted(logs_dir.glob("spatial_realsense_temporal_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No spatial_realsense_temporal_*.csv files found in {logs_dir}")
    return candidates[0]


def _to_float(v: str) -> float | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_frame_data(csv_path: Path, session_id: str | None) -> dict[int, dict[tuple[str, int], np.ndarray]]:
    frames: dict[int, dict[tuple[str, int], np.ndarray]] = defaultdict(dict)
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if session_id and str(row.get("session_id", "")).strip() != session_id:
                continue

            frame_idx_raw = row.get("frame_index")
            class_name = str(row.get("class_name", "")).strip()
            tracker_id_raw = row.get("tracker_id")

            if not frame_idx_raw or not tracker_id_raw or not class_name:
                continue

            x = _to_float(row.get("X_m", ""))
            y = _to_float(row.get("Y_m", ""))
            z = _to_float(row.get("Z_m", ""))
            if x is None or y is None or z is None:
                continue

            try:
                frame_idx = int(frame_idx_raw)
                tracker_id = int(float(tracker_id_raw))
            except ValueError:
                continue

            key = (class_name, tracker_id)
            frames[frame_idx][key] = np.asarray([x, y, z], dtype=float)

    return dict(sorted(frames.items(), key=lambda kv: kv[0]))


def _kabsch_rt(
    src: np.ndarray,
    dst: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Kabsch (SVD) rigid-body alignment: find R, t such that dst ≈ R @ src + t.

    Parameters
    ----------
    src : (N, 3) camera-frame XYZ of matched anchor points
    dst : (N, 3) world-frame XYZ of the same anchors

    Returns
    -------
    R : (3, 3) rotation matrix
    t : (3,)   translation vector
    """
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    A = (src - src_mean).T @ (dst - dst_mean)
    U, _, Vt = np.linalg.svd(A)
    # Reflection correction to ensure a proper rotation (det = +1)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = dst_mean - R @ src_mean
    return R, t


def estimate_world_and_camera(
    frames: dict[int, dict[tuple[str, int], np.ndarray]],
) -> tuple[dict[tuple[str, int], np.ndarray], dict[int, tuple[np.ndarray, np.ndarray]]]:
    """Estimate world-frame object positions and camera poses.

    Uses the Kabsch SVD algorithm when ≥3 anchor correspondences are visible
    in a frame, falling back to translation-only when fewer are available.

    Returns
    -------
    world_anchors : {(class, id): world_xyz}
    camera_poses  : {frame_idx: (R_3x3, t_3)}  where  world_pt = R @ cam_pt + t
    """
    world_anchors: dict[tuple[str, int], np.ndarray] = {}
    camera_poses: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    prev_R = np.eye(3, dtype=float)
    prev_t = np.zeros(3, dtype=float)

    for frame_idx, objs in frames.items():
        static_obs = {k: v for k, v in objs.items() if k[0] in STATIC_CLASS_NAMES}

        src_pts: list[np.ndarray] = []
        dst_pts: list[np.ndarray] = []
        for key, cam_xyz in static_obs.items():
            if key in world_anchors:
                src_pts.append(cam_xyz)
                dst_pts.append(world_anchors[key])

        if len(src_pts) >= 3:
            R, t = _kabsch_rt(np.stack(src_pts), np.stack(dst_pts))
        elif src_pts:
            # Not enough points for rotation; refine translation only
            R = prev_R.copy()
            t = np.mean(
                [d - R @ s for s, d in zip(src_pts, dst_pts)], axis=0
            )
        else:
            R, t = prev_R.copy(), prev_t.copy()

        for key, cam_xyz in static_obs.items():
            if key not in world_anchors:
                world_anchors[key] = R @ cam_xyz + t

        camera_poses[frame_idx] = (R, t)
        prev_R, prev_t = R, t

    return world_anchors, camera_poses


def latest_world_points(
    frames: dict[int, dict[tuple[str, int], np.ndarray]],
    camera_poses: dict[int, tuple[np.ndarray, np.ndarray]],
) -> dict[tuple[str, int], np.ndarray]:
    latest: dict[tuple[str, int], np.ndarray] = {}
    for frame_idx, objs in frames.items():
        pose = camera_poses.get(frame_idx)
        if pose is None:
            continue
        R, t = pose
        for key, cam_xyz in objs.items():
            latest[key] = R @ cam_xyz + t
    return latest


def save_camera_path_csv(camera_poses: dict[int, tuple[np.ndarray, np.ndarray]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_index", "cam_x_m", "cam_y_m", "cam_z_m"])
        for frame_idx in sorted(camera_poses):
            _, t = camera_poses[frame_idx]
            w.writerow([frame_idx, f"{t[0]:.6f}", f"{t[1]:.6f}", f"{t[2]:.6f}"])


def plot_map(
    world_points: dict[tuple[str, int], np.ndarray],
    camera_pos_by_frame: dict[int, np.ndarray],
    save_path: Path,
    title: str,
    show: bool,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    class_groups: dict[str, list[np.ndarray]] = defaultdict(list)
    for (class_name, _tid), xyz in world_points.items():
        class_groups[class_name].append(xyz)

    cmap = plt.get_cmap("tab20")
    classes = sorted(class_groups.keys())
    for i, cls in enumerate(classes):
        pts = np.stack(class_groups[cls], axis=0)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=35, color=cmap(i % 20), label=cls, alpha=0.9)

    if camera_pos_by_frame:
        frames_sorted = sorted(camera_pos_by_frame)
        cam_pts = np.stack([camera_pos_by_frame[f] for f in frames_sorted], axis=0)
        ax.plot(cam_pts[:, 0], cam_pts[:, 1], cam_pts[:, 2], color="black", linewidth=2.0, label="camera path")
        ax.scatter(cam_pts[:, 0], cam_pts[:, 1], cam_pts[:, 2], color="black", s=10, alpha=0.7)
        ax.scatter(cam_pts[-1, 0], cam_pts[-1, 1], cam_pts[-1, 2], color="red", s=70, marker="^", label="camera (last)")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)

    all_pts: list[np.ndarray] = []
    if world_points:
        all_pts.extend(world_points.values())
    if camera_pos_by_frame:
        all_pts.extend(camera_pos_by_frame.values())
    if all_pts:
        arr = np.stack(all_pts, axis=0)
        mins = arr.min(axis=0)
        maxs = arr.max(axis=0)
        center = (mins + maxs) / 2.0
        span = float(np.max(maxs - mins))
        span = max(span, 0.5)
        half = span / 2.0
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)

    if len(classes) <= 25:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    print(f"Saved 3D map image: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize 3D object map and estimated camera path from RGB-D detections")
    parser.add_argument("--csv-path", type=str, default=None, help="Path to spatial_realsense_temporal_*.csv (default: latest)")
    parser.add_argument("--session-id", type=str, default=None, help="Optional session_id filter")
    parser.add_argument("--save-path", type=str, default=None, help="Output image path (default: output/plots/spatial_map_latest.png)")
    parser.add_argument("--show", action="store_true", help="Open interactive matplotlib window")
    args = parser.parse_args()

    csv_path = Path(args.csv_path).resolve() if args.csv_path else _latest_csv(LOGS_DIR)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    save_path = Path(args.save_path).resolve() if args.save_path else (PLOTS_DIR / "spatial_map_latest.png").resolve()
    cam_csv_path = (save_path.parent / "camera_path_latest.csv").resolve()

    frames = load_frame_data(csv_path, args.session_id)
    if not frames:
        raise RuntimeError(f"No usable XYZ detections found in: {csv_path}")

    world_anchors, camera_poses = estimate_world_and_camera(frames)
    world_points = latest_world_points(frames, camera_poses)
    cam_positions = {f: t for f, (_, t) in camera_poses.items()}

    save_camera_path_csv(camera_poses, cam_csv_path)
    print(f"Saved camera path CSV: {cam_csv_path}")
    print(f"Frames used: {len(frames)} | anchors seeded: {len(world_anchors)} | objects plotted: {len(world_points)}")

    title = f"3D Spatial Map + Estimated Camera Path\nsource={csv_path.name}"
    plot_map(world_points, cam_positions, save_path, title, args.show)


if __name__ == "__main__":
    main()
