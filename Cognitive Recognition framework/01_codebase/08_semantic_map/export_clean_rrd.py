"""
Export a lightweight, instant-loading canonical RRD file from world_map.db.

Contains:
- Full 3D structural perimeter wall points from 2D/3D LiDAR (corridor boundaries)
- Navigable floor boundary polygon
- Camera trajectory path
- All consensus landmarks with physical 3D surface point clouds and wireframe bounding boxes
- Unified object surface point cloud
- 3D Semi-Cylindrical Keep-Clear Egress Volumes on Emergency Fire Exit doors and Exit Signs (Green/Red)
- Compliance hazard lines

Excludes heavy raw video frames (1080p images), maintaining fast loading (~2-5 MB).
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

from rerun_logger import class_color, landmark_metadata, resolve_box_half_size, stable_landmark_path


def export_clean_rrd(db_path: Path, out_rrd: Path) -> Path:
    db_path = Path(db_path).resolve()
    out_rrd = Path(out_rrd).resolve()
    run_dir = db_path.parent

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    if out_rrd.exists():
        try:
            out_rrd.unlink()
        except Exception:
            pass

    app_id = f"semantic_map/{run_dir.name}"
    rr.init(app_id, recording_id=f"canonical_{run_dir.name}")
    rr.save(str(out_rrd))

    # Send strict single 3D Semantic Map blueprint
    try:
        import rerun.blueprint as rrb
        blueprint = rrb.Blueprint(
            rrb.Spatial3DView(origin="world", name="3D Semantic Map"),
            collapse_panels=False,
        )
        rr.send_blueprint(blueprint)
    except Exception:
        pass

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

    # 2. Log Navigable Floor Boundary and LiDAR Perimeter Walls
    zones = conn.execute("SELECT * FROM navigable_zones WHERE zone_name = 'Floor'").fetchall()
    for z in zones:
        ez = float(z["elevation_z"]) if z["elevation_z"] is not None else 0.0
        keys = z.keys()

        # Floor boundary polygon
        poly_logged = False
        if "polygon_json" in keys and z["polygon_json"]:
            try:
                poly_pts = json.loads(z["polygon_json"])
                if poly_pts and len(poly_pts) >= 3:
                    poly_3d = np.array([[p[0], p[1], ez] for p in poly_pts] + [[poly_pts[0][0], poly_pts[0][1], ez]], dtype=np.float32)
                    rr.log(
                        "world/zones/navigable_boundary",
                        rr.LineStrips3D([poly_3d], colors=[[100, 180, 140]], radii=0.035, labels=["Floor"]),
                        static=True,
                    )
                    poly_logged = True
            except Exception:
                pass

        if not poly_logged:
            min_x, max_x = float(z["min_x"]), float(z["max_x"])
            min_y, max_y = float(z["min_y"]), float(z["max_y"])
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

        # Structural LiDAR Perimeter Walls (hallway layout)
        if "lidar_perimeter_json" in keys and z["lidar_perimeter_json"]:
            try:
                lidar_pts = json.loads(z["lidar_perimeter_json"])
                if lidar_pts:
                    pts_3d = []
                    for pt in lidar_pts:
                        if len(pt) == 2:
                            pts_3d.append([float(pt[0]), float(pt[1]), ez + 0.35])
                        elif len(pt) >= 3:
                            pts_3d.append([float(pt[0]), float(pt[1]), float(pt[2])])
                    if pts_3d:
                        pts_arr = np.array(pts_3d, dtype=np.float32)
                        rr.log(
                            "world/lidar/perimeter_points",
                            rr.Points3D(pts_arr, colors=[[160, 175, 195]], radii=0.018),
                            static=True,
                        )
            except Exception:
                pass

    # 3. Log Consolidated Consensus Landmarks with Surface Point Clouds and 3D Bounding Boxes
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
        is_floor = str(class_name).strip().lower() in ("floor", "navigable_floor")
        label = "Floor" if is_floor else f"{class_name} {instance_id}"
        center = np.asarray([[lm["X"], lm["Y"], lm["Z"]]], dtype=np.float32)
        lm_by_label[label] = (float(lm["X"]), float(lm["Y"]), float(lm["Z"]))

        if class_name.startswith("novel_") or class_name == "unidentified_obstacle":
            color = [[245, 166, 35]]  # Amber
        else:
            color = [class_color(class_name)]

        path = stable_landmark_path(lm, landmark_id)
        half_size = resolve_box_half_size(lm)

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
                    labels=[label],
                    radii=0.032,
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

        # 3D Bounding Box wireframe
        if not is_floor:
            rr.log(
                f"{path}/bounds",
                rr.Boxes3D(
                    centers=center,
                    half_sizes=np.asarray(half_size, dtype=np.float32),
                    colors=color,
                    fill_mode="MajorWireframe",
                ),
                static=True,
            )

        count += 1

    # 4. Log Unified Object Surface Point Cloud
    if all_pts_list:
        combined_pts = np.vstack(all_pts_list)
        combined_colors = np.vstack(all_colors_list)
        rr.log(
            "world/pointcloud",
            rr.Points3D(combined_pts, colors=combined_colors, radii=0.035),
            static=True,
        )

    # 5. Log 3D Compliance Hazard Lines
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

    # 6. Log 3D Semi-Cylindrical Keep-Clear Egress Volumes (Fire Exits and Exit Signs)
    zones_source_path = report_path if report_path.exists() else db_path.with_name("spatial_scene_graph.json")
    if zones_source_path.exists():
        try:
            with open(zones_source_path, "r", encoding="utf-8") as f:
                zones_data = json.load(f)
            door_zones = zones_data.get("door_keep_clear_zones", [])
            for dz in door_zones:
                if not dz.get("is_fire_exit", True):
                    continue
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
                    parent_cls = str(dz.get("parent_class", "door")).lower()
                    parent_inst = dz.get("parent_instance") or did
                    cls_title = "Exit Sign" if "exit" in parent_cls else "Door"
                    slug = "exit_sign" if "exit" in parent_cls else "door"
                    zone_label = f"{cls_title} {parent_inst} 3D Egress Zone [{status_label}] (r={dz.get('radius_m', 1.0)}m, h={dz.get('height_m', 2.0)}m)"
                    strips_comp = rr.LineStrips3D(
                        strips_list,
                        colors=[zone_color] * len(strips_list),
                        radii=0.022,
                        labels=[zone_label],
                    )
                    # Log under both clear descriptive path and backwards-compatible path
                    rr.log(f"world/egress_zones/{slug}_{parent_inst}_egress_zone", strips_comp, static=True)
                    rr.log(f"world/door_zones/fire_exit_door_{did}_zone", strips_comp, static=True)
        except Exception as exc:
            print(f"Warning: could not log door keep-clear zones to RRD: {exc}")

    conn.close()

    # Flush Rerun writer
    rr.disconnect()

    file_size_mb = out_rrd.stat().st_size / (1024 * 1024) if out_rrd.exists() else 0.0
    print(f"Exported {count} consolidated landmarks, full point clouds, LiDAR boundaries, and trajectory to {out_rrd} (size: {file_size_mb:.2f} MB)")
    return out_rrd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export clean canonical RRD from world_map.db")
    parser.add_argument("--db", type=Path, default=Path("04_outputs_runs_and_logs/outputs/semantic_maps/rgbd_clean_20260925_134830/world_map.db"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out_file = args.out or args.db.with_name("world_map.rrd")
    export_clean_rrd(args.db, out_file)
