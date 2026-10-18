"""
Export a lightweight, instant-loading RRD file from world_map.db.

Contains:
- All consensus landmarks with physical 3D surface point clouds
- Trajectory path
- Navigable floor boundary

Excludes raw video image frames, reducing file size from ~4 GB to <1 MB.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
import rerun as rr

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from rerun_logger import class_color, landmark_metadata, stable_landmark_path


def export_clean_rrd(db_path: Path, out_rrd: Path) -> Path:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    rr.init("SemanticMapClean", recording_id="semantic_map_clean")
    rr.save(str(out_rrd))

    # 1. Log Trajectory
    poses = conn.execute("SELECT x, y, z FROM camera_poses ORDER BY frame_index").fetchall()
    if poses:
        traj_pts = np.array([[p["x"], p["y"], p["z"]] for p in poses], dtype=np.float32)
        step = max(1, len(traj_pts) // 1500)
        sampled_traj = traj_pts[::step]
        rr.log(
            "world/trajectory",
            rr.LineStrips3D([sampled_traj], colors=[[80, 160, 240]], radii=0.025),
            static=True,
        )

    # 2. Log Navigable Floor Boundary
    zones = conn.execute("SELECT * FROM navigable_zones WHERE zone_name = 'Floor'").fetchall()
    for z in zones:
        min_x, max_x = z["min_x"], z["max_x"]
        min_y, max_y = z["min_y"], z["max_y"]
        ez = z["elevation_z"]
        box_rect = np.array([
            [min_x, min_y, ez],
            [max_x, min_y, ez],
            [max_x, max_y, ez],
            [min_x, max_y, ez],
            [min_x, min_y, ez],
        ], dtype=np.float32)
        rr.log(
            "world/zones/navigable_boundary",
            rr.LineStrips3D([box_rect], colors=[[100, 180, 140]], radii=0.035, labels=["Floor"]),
            static=True,
        )

    # 3. Log Landmarks with real surface point clouds
    landmarks = conn.execute("SELECT * FROM semantic_map").fetchall()
    count = 0
    for row in landmarks:
        lm = dict(row)
        landmark_id = lm["landmark_id"]
        class_name = lm["class_name"]
        instance_id = lm.get("instance_id") or landmark_id
        label = f"{class_name} {instance_id}"
        color = [class_color(class_name)]
        path = stable_landmark_path(lm, landmark_id)
        center = np.asarray([[lm["X"], lm["Y"], lm["Z"]]], dtype=np.float32)

        surf_raw = lm.get("surface_points_json")
        pts = None
        if surf_raw:
            try:
                pts = np.asarray(json.loads(surf_raw), dtype=np.float32)
            except Exception:
                pts = None

        if pts is not None and len(pts) >= 5:
            rr.log(
                path,
                rr.Points3D(
                    pts,
                    colors=[color[0]] * len(pts),
                    labels=[label],
                    radii=0.028,
                ),
                rr.AnyValues(**landmark_metadata(lm, landmark_id)),
                static=True,
            )
        else:
            rr.log(
                path,
                rr.Points3D(
                    center,
                    colors=color,
                    labels=[label],
                    radii=0.045,
                ),
                rr.AnyValues(**landmark_metadata(lm, landmark_id)),
                static=True,
            )
        count += 1

    conn.close()
    print(f"Exported {count} landmarks and trajectory to {out_rrd} (size: {out_rrd.stat().st_size / 1024:.1f} KB)")
    return out_rrd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export clean RRD from world_map.db")
    parser.add_argument("--db", type=Path, default=Path("04_outputs_runs_and_logs/outputs/semantic_maps/rgbd_clean_20260925_134830/world_map.db"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out_file = args.out or args.db.with_name("semantic_map_clean.rrd")
    export_clean_rrd(args.db, out_file)

