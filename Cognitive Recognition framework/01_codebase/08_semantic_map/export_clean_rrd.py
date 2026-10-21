"""
Export a lightweight, instant-loading canonical RRD file from world_map.db.

Contains:
- Full 3D environment point cloud (hallway walls, floor, structural background)
- All consensus landmarks with physical 3D surface point clouds
- Camera trajectory path
- Navigable floor boundary
- 3D Semi-Cylindrical Keep-Clear Egress Volumes on Emergency Fire Exit doors (Green/Red)
- Compliance hazard lines

Excludes heavy raw video frames (1080p images), maintaining fast loading (~5-7 MB).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys

import numpy as np
import rerun as rr

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from rerun_logger import class_color, landmark_metadata, stable_landmark_path


def export_clean_rrd(db_path: Path, out_rrd: Path) -> Path:
    db_path = Path(db_path).resolve()
    out_rrd = Path(out_rrd).resolve()
    run_dir = db_path.parent
    history_dir = run_dir / "run_history"

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # 1. Look for existing dense archive RRD with environment point clouds
    dense_cloud_extracted = False
    temp_cloud_rrd = run_dir / "_temp_env_cloud.rrd"
    temp_semantic_rrd = run_dir / "_temp_semantic.rrd"

    # Search for latest archive in run_history
    archive_candidates = sorted(history_dir.glob("world_map_*.rrd"), key=lambda p: p.stat().st_mtime, reverse=True)
    rerun_bin = sys.executable.replace("python.exe", "rerun.exe")

    if archive_candidates and Path(rerun_bin).exists():
        latest_archive = archive_candidates[0]
        try:
            # Preserve the existing dense environment point clouds and camera video feeds
            shutil.copy2(latest_archive, temp_cloud_rrd)
            if temp_cloud_rrd.exists() and temp_cloud_rrd.stat().st_size > 10000:
                dense_cloud_extracted = True
        except Exception as exc:
            print(f"Warning: could not preserve archive point cloud: {exc}")

    # 2. Write semantic landmarks, door keep-clear volumes, and hazards
    target_write_rrd = temp_semantic_rrd if dense_cloud_extracted else out_rrd
    if target_write_rrd.exists():
        target_write_rrd.unlink()

    app_id = f"semantic_map/{run_dir.name}"
    rr.init(app_id, recording_id=f"canonical_{run_dir.name}")
    rr.save(str(target_write_rrd))

    # Send strict single-map layout blueprint
    try:
        import rerun.blueprint as rrb
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(origin="world", name="3D Semantic Map"),
                rrb.Spatial2DView(origin="world/camera/image", name="Annotated 2D Video"),
                column_shares=[3, 2],
            ),
            collapse_panels=False,
        )
        rr.send_blueprint(blueprint)
    except Exception:
        pass

    # Set right-hand Z-up coordinate system
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # Log Trajectory
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

    # Log Navigable Floor Boundary
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

    # Log Landmarks with real surface point clouds
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

    # Log unified object surface point cloud
    if all_pts_list:
        combined_pts = np.vstack(all_pts_list)
        combined_colors = np.vstack(all_colors_list)
        rr.log(
            "world/pointcloud",
            rr.Points3D(combined_pts, colors=combined_colors, radii=0.038),
            static=True,
        )

    # Log 3D Compliance Hazard Lines
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
                    strip = np.array([p1, p2], dtype=np.float32)
                    dist_text = f" ({d_val:.2f}m)" if d_val is not None else ""
                    rr.log(
                        hazard_path,
                        rr.LineStrips3D([strip], colors=[[240, 45, 45]], radii=0.035, labels=[f"VIOLATION: {rule}{dist_text}"]),
                        static=True,
                    )
                elif p1:
                    p_top = (p1[0], p1[1], p1[2] + 0.40)
                    strip = np.array([p1, p_top], dtype=np.float32)
                    rr.log(
                        hazard_path,
                        rr.LineStrips3D([strip], colors=[[240, 45, 45]], radii=0.03, labels=[f"VIOLATION: {rule}"]),
                        static=True,
                    )
        except Exception as exc:
            print(f"Warning: could not log compliance hazards to RRD: {exc}")

    # Log 3D Semi-Cylindrical Keep-Clear Volumes (ONLY Emergency Fire Exits with Exit Signs on top)
    graph_path = db_path.with_name("spatial_scene_graph.json")
    if graph_path.exists():
        try:
            with open(graph_path, "r", encoding="utf-8") as f:
                graph_data = json.load(f)
            door_zones = graph_data.get("door_keep_clear_zones", [])
            for dz in door_zones:
                did = dz.get("door_id")
                is_obstructed = dz.get("is_obstructed", False)
                status_label = dz.get("status", "CLEAR")
                # Dynamic green / red status
                zone_color = [235, 50, 50] if is_obstructed else [46, 204, 113]

                cage = dz.get("cage_samples_3d") or {}
                bot_arc = cage.get("bottom_arc", [])
                top_arc = cage.get("top_arc", [])
                ribs = cage.get("vertical_ribs", [])
                back_wall = cage.get("back_wall", [])

                strips_list = []
                # 1. Closed bottom arc at floor
                if bot_arc and len(bot_arc) >= 3:
                    center_bot = (dz["center"][0], dz["center"][1], dz["z_bounds"][0])
                    strips_list.append(np.asarray([center_bot] + [tuple(p) for p in bot_arc] + [center_bot], dtype=np.float32))

                # 2. Closed top arc at door header
                if top_arc and len(top_arc) >= 3:
                    center_top = (dz["center"][0], dz["center"][1], dz["z_bounds"][1])
                    strips_list.append(np.asarray([center_top] + [tuple(p) for p in top_arc] + [center_top], dtype=np.float32))

                # 3. Vertical corner and wall ribs connecting bottom to top
                for rib in ribs:
                    if len(rib) == 2:
                        strips_list.append(np.asarray([tuple(rib[0]), tuple(rib[1])], dtype=np.float32))

                # 4. Back-wall door opening rectangle
                if back_wall and len(back_wall) >= 4:
                    strips_list.append(np.asarray([tuple(p) for p in back_wall], dtype=np.float32))

                if strips_list:
                    rr.log(
                        f"world/door_zones/fire_exit_door_{did}_zone",
                        rr.LineStrips3D(
                            strips_list,
                            colors=[zone_color] * len(strips_list),
                            radii=0.022,
                            labels=[f"Fire Exit Door {did} 3D Zone [{status_label}] (r={dz.get('radius_m', 1.0)}m, h={dz.get('height_m', 2.0)}m)"],
                        ),
                        static=True,
                    )
        except Exception as exc:
            print(f"Warning: could not log door keep-clear zones to RRD: {exc}")

    conn.close()

    # Flush Rerun writer
    rr.disconnect()

    # 3. Merge dense environment point cloud and semantic landmarks into canonical out_rrd
    if dense_cloud_extracted and temp_cloud_rrd.exists() and temp_semantic_rrd.exists():
        try:
            cmd_merge = [
                str(rerun_bin), "rrd", "merge",
                "-o", str(out_rrd),
                str(temp_cloud_rrd),
                str(temp_semantic_rrd)
            ]
            subprocess.run(cmd_merge, capture_output=True, text=True, timeout=30)
            if temp_cloud_rrd.exists():
                temp_cloud_rrd.unlink()
            if temp_semantic_rrd.exists():
                temp_semantic_rrd.unlink()
        except Exception as exc:
            print(f"Warning: could not merge environment cloud: {exc}")
            if temp_semantic_rrd.exists():
                shutil.move(str(temp_semantic_rrd), str(out_rrd))

    file_size_mb = out_rrd.stat().st_size / (1024 * 1024) if out_rrd.exists() else 0.0
    print(f"Exported {count} landmarks, full point cloud, and trajectory to {out_rrd} (size: {file_size_mb:.2f} MB)")
    return out_rrd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export clean RRD from world_map.db")
    parser.add_argument("--db", type=Path, default=Path("04_outputs_runs_and_logs/outputs/semantic_maps/rgbd_clean_20260925_134830/world_map.db"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out_file = args.out or args.db.with_name("world_map.rrd")
    export_clean_rrd(args.db, out_file)
