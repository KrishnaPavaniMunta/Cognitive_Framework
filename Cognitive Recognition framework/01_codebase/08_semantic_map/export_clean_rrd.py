"""
Export a complete canonical RRD world map from world_map.db, satisfying all
mandatory specifications defined in WORLD_MAP_REQUIREMENTS.md:

1. Environmental 3D Point Cloud (RGB-D structural points from bag/cache)
2. Environmental LiDAR perimeter wall points (corridor boundaries from /scan)
3. Synchronized 2D/3D Annotated Video Stream (preview.mp4 with timeline scrubbing)
4. Health & Safety Keep-Clear Egress Cylinders (3D green/red semi-cylinders)
5. Exact Landmark Labels & Physical Surface Points (ontology metadata attached)
6. STRICT PROHIBITION: Zero Wireframe Cuboids (rr.Boxes3D strictly removed)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sqlite3
import sys
import time

import cv2
import numpy as np
import rerun as rr

BASE_DIR = Path(__file__).resolve().parent
CODEBASE_DIR = BASE_DIR.parent
OBJECT_DETECTION_DIR = CODEBASE_DIR / "07_object_detection"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(OBJECT_DETECTION_DIR) not in sys.path:
    sys.path.insert(0, str(OBJECT_DETECTION_DIR))

from rerun_logger import class_color, landmark_metadata, matrix_to_translation_quaternion, stable_landmark_path


def ensure_environmental_point_cloud(run_dir: Path, db_path: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load cached environmental point cloud or extract it from the source ROS2 bag."""
    cache_path = run_dir / "environment_point_cloud.npz"
    if cache_path.exists():
        try:
            data = np.load(cache_path)
            return data["points"], data["colors"]
        except Exception as exc:
            print(f"[RRD] Warning: failed loading point cloud cache {cache_path.name}: {exc}")

    # Fallback: extract from source bag if available
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    bag_path_row = conn.execute("SELECT bag_path FROM map_runs ORDER BY rowid DESC LIMIT 1").fetchone()
    bag_path = Path(bag_path_row["bag_path"]) if bag_path_row and bag_path_row["bag_path"] else None

    if not bag_path or not bag_path.exists():
        conn.close()
        return None, None

    poses = conn.execute("SELECT frame_index, matrix_json FROM camera_poses ORDER BY frame_index").fetchall()
    pose_by_frame = {int(p["frame_index"]): p for p in poses if p["matrix_json"]}
    conn.close()

    try:
        from rosbags.highlevel import AnyReader
        from rosbags.typesys import Stores, get_typestore
        import rosbag_rgbd_sim_capture as capture

        typestore = get_typestore(Stores.ROS2_HUMBLE)
        fx, fy, cx, cy = 525.0, 525.0, 320.0, 240.0
        depth_step = 6
        grid_u, grid_v = np.meshgrid(np.arange(0, 640, depth_step), np.arange(0, 480, depth_step))
        grid_u = grid_u.astype(np.float32)
        grid_v = grid_v.astype(np.float32)

        stride_step = 20
        env_points = []
        env_colors = []

        with AnyReader([bag_path], default_typestore=typestore) as reader:
            rgb_conns = [c for c in reader.connections if c.topic == "/camera/rgb/image_rect_color"]
            depth_conns = [c for c in reader.connections if c.topic == "/camera/depth_registered/image_raw"]
            if not rgb_conns or not depth_conns:
                return None, None

            conns = rgb_conns + depth_conns
            latest_depth_raw = None
            latest_depth_msgtype = depth_conns[0].msgtype
            frame_idx = 0

            for conn_item, _, raw in reader.messages(connections=conns):
                if conn_item.topic == "/camera/depth_registered/image_raw":
                    latest_depth_raw = raw
                elif conn_item.topic == "/camera/rgb/image_rect_color":
                    frame_idx += 1
                    if frame_idx % 4 != 0:
                        continue
                    db_frame = frame_idx // 4
                    if db_frame % stride_step != 0:
                        continue

                    p = pose_by_frame.get(db_frame)
                    if not p or latest_depth_raw is None:
                        continue

                    dmsg = reader.deserialize(latest_depth_raw, latest_depth_msgtype)
                    rgb_msg = reader.deserialize(raw, conn_item.msgtype)

                    depth_mm = capture._decode_depth_image_mm(dmsg)
                    if depth_mm is None:
                        continue

                    depth_sample = depth_mm[::depth_step, ::depth_step].astype(np.float32) / 1000.0
                    valid = (depth_sample > 0.40) & (depth_sample < 6.0) & np.isfinite(depth_sample)
                    if not np.any(valid):
                        continue

                    z = depth_sample[valid]
                    u = grid_u[valid]
                    v = grid_v[valid]
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy

                    pts_cam = np.stack([x, y, z], axis=1)
                    M = np.array(json.loads(p["matrix_json"]), dtype=np.float32)
                    R = M[:3, :3]
                    T = M[:3, 3]
                    pts_world = (pts_cam @ R.T) + T

                    rgb_bgr = capture._decode_color_image(rgb_msg)
                    if rgb_bgr is not None:
                        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
                        colors = rgb[::depth_step, ::depth_step][valid]
                    else:
                        colors = np.full((len(z), 3), 180, dtype=np.uint8)

                    env_points.append(pts_world)
                    env_colors.append(colors)

        if env_points:
            all_pts = np.vstack(env_points).astype(np.float32)
            all_col = np.vstack(env_colors).astype(np.uint8)
            voxel_size = 0.04
            voxel_keys = np.floor(all_pts / voxel_size).astype(np.int32)
            _, unique_idx = np.unique(voxel_keys, axis=0, return_index=True)
            filt_pts = all_pts[unique_idx]
            filt_col = all_col[unique_idx]

            try:
                np.savez_compressed(cache_path, points=filt_pts, colors=filt_col)
            except Exception:
                pass
            return filt_pts, filt_col

    except Exception as exc:
        print(f"[RRD] Warning: failed on-the-fly point cloud extraction: {exc}")

    return None, None


def export_clean_rrd(db_path: Path, out_rrd: Path | None = None) -> Path:
    """Generate the complete canonical world_map.rrd adhering to WORLD_MAP_REQUIREMENTS.md."""
    db_path = Path(db_path).resolve()
    run_dir = db_path.parent
    out_rrd = Path(out_rrd).resolve() if out_rrd is not None else db_path.with_name("world_map.rrd")

    print("=== BUILDING COMPLETE CANONICAL WORLD MAP (RRD) ===")
    print(f"Source DB : {db_path}")
    print(f"Output RRD: {out_rrd}")

    if out_rrd.exists():
        try:
            out_rrd.unlink()
        except Exception:
            pass

    app_id = f"semantic_map/{run_dir.name}"
    rr.init(app_id, recording_id=f"canonical_{run_dir.name}")
    rr.save(str(out_rrd))

    # 1. Dual Blueprint: 3D Digital Twin + 2D Synchronized Annotated Video
    try:
        import rerun.blueprint as rrb
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(origin="world", name="3D Semantic Digital Twin"),
                rrb.Spatial2DView(origin="world/camera/annotated_image", name="Annotated Camera Stream"),
            ),
            collapse_panels=False,
        )
        rr.send_blueprint(blueprint)
    except Exception:
        pass

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # 2. Camera Trajectory Path
    poses = conn.execute("SELECT frame_index, timestamp_ns, X, Y, Z, matrix_json FROM camera_poses ORDER BY frame_index").fetchall()
    pose_by_frame = {int(p["frame_index"]): p for p in poses}
    if poses:
        traj_pts = np.array([[float(p["X"]), float(p["Y"]), float(p["Z"])] for p in poses], dtype=np.float32)
        step = max(1, len(traj_pts) // 1500)
        rr.log(
            "world/trajectory",
            rr.LineStrips3D([traj_pts[::step]], colors=[[80, 160, 240]], radii=0.025),
            static=True,
        )

    # 3. Environmental LiDAR Perimeter Walls & Navigable Floor Boundary
    zones = conn.execute("SELECT * FROM navigable_zones WHERE zone_name = 'Floor'").fetchall()
    for z in zones:
        ez = float(z["elevation_z"]) if z["elevation_z"] is not None else 0.0
        keys = z.keys()

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
                            rr.Points3D(pts_arr, colors=[[140, 190, 230]], radii=0.022),
                            static=True,
                        )
            except Exception:
                pass

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
            except Exception:
                pass

    # 4. Consensus Landmarks (NO CUBOIDS, EXACT LABELS + SURFACE POINTS)
    landmarks = conn.execute("SELECT * FROM semantic_map").fetchall()
    lm_by_label: dict[str, tuple[float, float, float]] = {}

    for row in landmarks:
        lm = dict(row)
        landmark_id = lm["landmark_id"]
        class_name = lm["class_name"]
        instance_id = lm.get("instance_id") or landmark_id
        is_floor = str(class_name).strip().lower() in ("floor", "navigable_floor")
        label = "Floor" if is_floor else f"{class_name} #{instance_id}"
        center = np.asarray([[float(lm["X"]), float(lm["Y"]), float(lm["Z"])]], dtype=np.float32)
        lm_by_label[label] = (float(lm["X"]), float(lm["Y"]), float(lm["Z"]))

        color = [[245, 166, 35]] if (class_name.startswith("novel_") or class_name == "unidentified_obstacle") else [class_color(class_name)]
        path = stable_landmark_path(lm, landmark_id)

        surf_raw = lm.get("surface_points_json")
        pts = None
        if surf_raw:
            try:
                pts = np.asarray(json.loads(surf_raw), dtype=np.float32)
            except Exception:
                pts = None

        conf = lm.get("consensus_score", 1.0)
        display_label = f"{label} ({conf:.0%})" if conf is not None else label

        if pts is not None and len(pts) >= 5:
            rr.log(
                path,
                rr.Points3D(
                    pts,
                    colors=[color[0]] * len(pts),
                    labels=[display_label],
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
                    labels=[display_label],
                    radii=0.055,
                ),
                rr.AnyValues(**landmark_metadata(lm, landmark_id)),
                static=True,
            )

        # STRICT PROHIBITION: Zero cuboids logged! rr.Boxes3D is completely excluded.

    # 5. Health & Safety Keep-Clear Egress Cylinders & Compliance Hazards
    report_path = db_path.with_name("hospital_compliance_report.json")
    if not report_path.exists():
        report_path = db_path.with_name("spatial_scene_graph.json")

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
                dist_text = f" ({d_val:.2f}m)" if d_val is not None else ""

                if p1 and p2:
                    strip = np.array([p1, p2], dtype=np.float32)
                    rr.log(hazard_path, rr.LineStrips3D([strip], colors=[[240, 45, 45]], radii=0.035, labels=[f"VIOLATION: {rule}{dist_text}"]), static=True)
                elif p1:
                    p_top = (p1[0], p1[1], p1[2] + 0.40)
                    strip = np.array([p1, p_top], dtype=np.float32)
                    rr.log(hazard_path, rr.LineStrips3D([strip], colors=[[240, 45, 45]], radii=0.03, labels=[f"VIOLATION: {rule}"]), static=True)

            door_zones = comp_data.get("door_keep_clear_zones", [])
            for dz in door_zones:
                did = dz.get("door_id")
                parent_cls = str(dz.get("parent_class", "door")).lower()
                parent_inst = dz.get("parent_instance") or did
                is_obstructed = dz.get("is_obstructed", False)
                status_label = dz.get("status", "CLEAR")
                zone_color = [235, 50, 50] if is_obstructed else [46, 204, 113]

                cage = dz.get("cage_samples_3d") or {}
                bot_arc = cage.get("bottom_arc", [])
                top_arc = cage.get("top_arc", [])
                ribs = cage.get("vertical_ribs", [])
                back_wall = cage.get("back_wall", [])

                strips_list = []
                if bot_arc and len(bot_arc) >= 3 and top_arc and len(top_arc) >= 3:
                    center_bot = (dz["center"][0], dz["center"][1], dz["z_bounds"][0])
                    strips_list.append(np.asarray([center_bot] + [tuple(p) for p in bot_arc] + [center_bot], dtype=np.float32))

                    center_top = (dz["center"][0], dz["center"][1], dz["z_bounds"][1])
                    strips_list.append(np.asarray([center_top] + [tuple(p) for p in top_arc] + [center_top], dtype=np.float32))

                    for rib in ribs:
                        if len(rib) == 2:
                            strips_list.append(np.asarray([tuple(rib[0]), tuple(rib[1])], dtype=np.float32))

                    if back_wall and len(back_wall) >= 4:
                        strips_list.append(np.asarray([tuple(p) for p in back_wall], dtype=np.float32))
                else:
                    cx, cy, cz = dz["center"]
                    radius = float(dz.get("radius_m") or dz.get("radius") or 1.00)
                    z_min = float(dz.get("z_bounds", [cz - 0.9, cz + 1.2])[0])
                    z_max = float(dz.get("z_bounds", [cz - 0.9, cz + 1.2])[1])
                    normal = dz.get("egress_normal") or [0.0, 1.0]

                    theta_mid = math.atan2(normal[1], normal[0])
                    angles = np.linspace(theta_mid - math.pi / 2.0, theta_mid + math.pi / 2.0, 16)

                    arc_bot = np.array([[cx + radius * math.cos(a), cy + radius * math.sin(a), z_min] for a in angles], dtype=np.float32)
                    arc_top = np.array([[cx + radius * math.cos(a), cy + radius * math.sin(a), z_max] for a in angles], dtype=np.float32)

                    strips_list = [arc_bot, arc_top]
                    strips_list.append(np.array([[cx, cy, z_min], arc_bot[0]], dtype=np.float32))
                    strips_list.append(np.array([[cx, cy, z_min], arc_bot[-1]], dtype=np.float32))
                    strips_list.append(np.array([[cx, cy, z_max], arc_top[0]], dtype=np.float32))
                    strips_list.append(np.array([[cx, cy, z_max], arc_top[-1]], dtype=np.float32))
                    strips_list.append(np.array([[cx, cy, z_min], [cx, cy, z_max]], dtype=np.float32))
                    for i in [0, 5, 10, 15]:
                        strips_list.append(np.array([arc_bot[i], arc_top[i]], dtype=np.float32))

                if strips_list:
                    cls_title = "Exit Sign" if "exit" in parent_cls else "Door"
                    slug = "exit_sign" if "exit" in parent_cls else "door"
                    zone_label = f"FIRE EXIT KEEP-CLEAR [{cls_title} #{parent_inst}] ({status_label})"
                    strips_comp = rr.LineStrips3D(strips_list, colors=[zone_color] * len(strips_list), radii=0.022, labels=[zone_label])
                    rr.log(f"world/door_zones/{slug}_{parent_inst}_egress_zone", strips_comp, static=True)
                    rr.log(f"world/egress_zones/{slug}_{parent_inst}_egress_zone", strips_comp, static=True)
        except Exception as e:
            print(f"[RRD] Warning: could not log door egress zones: {e}")

    # 6. Environmental 3D Point Cloud (RGB-D Structural points)
    env_pts, env_cols = ensure_environmental_point_cloud(run_dir, db_path)
    if env_pts is not None and env_cols is not None:
        rr.log(
            "world/environment_point_cloud",
            rr.Points3D(env_pts, colors=env_cols, radii=0.020),
            static=True,
        )
        print(f"[RRD] Logged {len(env_pts)} structural point cloud points.")

    # 7. Synchronized Annotated Video Stream (preview.mp4 on timeline)
    preview_path = run_dir / "preview.mp4"
    if preview_path.exists():
        cap = cv2.VideoCapture(str(preview_path))
        fx, fy, cx, cy = 525.0, 525.0, 320.0, 240.0
        K_cam = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        v_frame_idx = 0
        logged_frames = 0
        video_stride = 2

        while cap.isOpened():
            ret, frame_bgr = cap.read()
            if not ret:
                break
            v_frame_idx += 1
            if v_frame_idx % video_stride != 0:
                continue

            p = pose_by_frame.get(v_frame_idx)
            if p is not None and p["matrix_json"]:
                try:
                    rr.set_time("frame", sequence=v_frame_idx)
                    rr.set_time("bag_time", timestamp=int(p["timestamp_ns"]) * 1e-9)
                except Exception:
                    pass

                M = np.array(json.loads(p["matrix_json"]), dtype=np.float32)
                translation, quaternion = matrix_to_translation_quaternion(M)

                rr.log("world/camera", rr.Transform3D(translation=translation, rotation=rr.Quaternion(xyzw=quaternion)))

                _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                img_bytes = buf.tobytes()
                rr.log("world/camera/image", rr.Pinhole(image_from_camera=K_cam, resolution=[frame_bgr.shape[1], frame_bgr.shape[0]]))
                rr.log("world/camera/image", rr.EncodedImage(contents=img_bytes, media_type="image/jpeg"))
                rr.log("world/camera/annotated_image", rr.EncodedImage(contents=img_bytes, media_type="image/jpeg"))
                logged_frames += 1

        cap.release()
        print(f"[RRD] Logged {logged_frames} synchronized annotated video frames with camera poses.")

    conn.close()
    rr.disconnect()

    file_size_mb = out_rrd.stat().st_size / (1024 * 1024) if out_rrd.exists() else 0.0
    print(f"[RRD] SUCCESS: Canonical World Map generated at: {out_rrd} ({file_size_mb:.2f} MB)")
    return out_rrd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export clean canonical RRD from world_map.db")
    parser.add_argument("--db", type=Path, default=Path("04_outputs_runs_and_logs/outputs/semantic_maps/rgbd_pose_20261103_143812/world_map.db"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out_file = args.out or args.db.with_name("world_map.rrd")
    export_clean_rrd(args.db, out_file)
