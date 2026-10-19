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

    # Set right-hand Z-up coordinate system
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

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
    lm_by_label: dict[str, tuple[float, float, float]] = {}
    all_pts_list = []
    all_colors_list = []
    count = 0
    for row in landmarks:
        lm = dict(row)
        landmark_id = lm["landmark_id"]
        class_name = lm["class_name"]
        instance_id = lm.get("instance_id") or landmark_id
        label = f"{class_name} {instance_id}"
        center = np.asarray([[lm["X"], lm["Y"], lm["Z"]]], dtype=np.float32)
        lm_by_label[label] = (float(lm["X"]), float(lm["Y"]), float(lm["Z"]))

        # Special styling for novel/residual assets
        if class_name.startswith("novel_") or class_name == "unidentified_obstacle":
            color = [[245, 166, 35]]  # Amber
        else:
            color = [class_color(class_name)]

        path = stable_landmark_path(lm, landmark_id)

        surf_raw = lm.get("surface_points_json")
        pts = None
        if surf_raw:
            try:
                pts = np.asarray(json.loads(surf_raw), dtype=np.float32)
            except Exception:
                pts = None

        if pts is not None and len(pts) >= 5:
            all_pts_list.append(pts)
            all_colors_list.append(np.tile(color[0], (len(pts), 1)))
            rr.log(
                path,
                rr.Points3D(
                    pts,
                    colors=[color[0]] * len(pts),
                    radii=0.035,
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
                    radii=0.055,
                ),
                rr.AnyValues(**landmark_metadata(lm, landmark_id)),
                static=True,
            )
        count += 1

    # Log unified high-visibility 3D point cloud
    if all_pts_list:
        combined_pts = np.vstack(all_pts_list)
        combined_colors = np.vstack(all_colors_list)
        rr.log(
            "world/pointcloud",
            rr.Points3D(combined_pts, colors=combined_colors, radii=0.038),
            static=True,
        )

    # 4. Log 3D Compliance Hazard Lines (Red Proximity Strips)
    report_path = db_path.with_name("hospital_compliance_report.json")
    if report_path.exists():
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                comp_data = json.load(f)
            violations = comp_data.get("violations", [])
            for v_idx, v in enumerate(violations):
                rule = v.get("rule_id", "HAZARD")
                source_label = v.get("object")
                target_label = v.get("target") or v.get("door")
                d_val = v.get("measured_distance_m")

                p1 = lm_by_label.get(source_label)
                p2 = lm_by_label.get(target_label) if target_label else None

                hazard_path = f"world/compliance_hazards/{rule}_{v_idx+1}"
                if p1 and p2:
                    # Draw connecting red hazard line between violating object and reference landmark (e.g. door)
                    strip = np.array([p1, p2], dtype=np.float32)
                    dist_text = f" ({d_val:.2f}m)" if d_val is not None else ""
                    rr.log(
                        hazard_path,
                        rr.LineStrips3D([strip], colors=[[240, 45, 45]], radii=0.035, labels=[f"VIOLATION: {rule}{dist_text}"]),
                        static=True,
                    )
                elif p1:
                    # Vertical red hazard alert post above single offending object
                    p_top = (p1[0], p1[1], p1[2] + 0.40)
                    strip = np.array([p1, p_top], dtype=np.float32)
                    rr.log(
                        hazard_path,
                        rr.LineStrips3D([strip], colors=[[240, 45, 45]], radii=0.03, labels=[f"VIOLATION: {rule}"]),
                        static=True,
                    )
        except Exception as exc:
            print(f"Warning: could not log compliance hazards to RRD: {exc}")

    # 5. Log 3D Door Keep-Clear Egress Zones (Andrea's 1.0m half-cylinder swing zones)
    graph_path = db_path.with_name("spatial_scene_graph.json")
    if graph_path.exists():
        try:
            with open(graph_path, "r", encoding="utf-8") as f:
                graph_data = json.load(f)
            door_zones = graph_data.get("door_keep_clear_zones", [])
            for dz in door_zones:
                did = dz.get("door_id")
                arc = dz.get("arc_samples_3d", [])
                if arc and len(arc) >= 3:
                    center_pt = (dz["center"][0], dz["center"][1], dz["z_bounds"][0])
                    closed_arc = [center_pt] + [tuple(p) for p in arc] + [center_pt]
                    rr.log(
                        f"world/door_zones/door_{did}_zone",
                        rr.LineStrips3D(
                            [np.asarray(closed_arc, dtype=np.float32)],
                            colors=[[220, 70, 70]],
                            radii=0.018,
                            labels=[f"Door {did} Keep-Clear Zone (r={dz.get('radius_m', 1.0)}m)"],
                        ),
                        static=True,
                    )
        except Exception as exc:
            print(f"Warning: could not log door keep-clear zones to RRD: {exc}")

    conn.close()
    print(f"Exported {count} landmarks and trajectory to {out_rrd} (size: {out_rrd.stat().st_size / 1024:.1f} KB)")
    return out_rrd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export clean RRD from world_map.db")
    parser.add_argument("--db", type=Path, default=Path("04_outputs_runs_and_logs/outputs/semantic_maps/rgbd_clean_20260925_134830/world_map.db"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out_file = args.out or args.db.with_name("world_map.rrd")
    export_clean_rrd(args.db, out_file)

