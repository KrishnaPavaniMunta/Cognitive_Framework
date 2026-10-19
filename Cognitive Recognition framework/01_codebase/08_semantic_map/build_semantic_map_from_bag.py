"""
Semantic map builder: recorded ROS2 bag -> world-frame 3D object pins in SQLite.

Pipeline per synchronized frame (reuses existing modules, nothing re-implemented):
  Step 1  2D perception on RGB only            -> YOLO ensemble + DINO fallback
  Step 2  depth association                    -> 5x5 median depth patch at bbox midpoint
  Step 3  de-projection with camera intrinsics -> XYZ in the camera frame
  Step 4  ego-motion compensation              -> multiply by TF/odom 4x4 extrinsics
  Step 5  semantic digital twin                -> merge + upsert into a per-bag SQLite DB

Every run creates its own folder: <out-root>/semanticmap_<bag>_<runid>/
    semantic_map.db      landmarks + raw observations
    run.log              full debug log (mirrors the console)
    preview.mp4          annotated live-preview recording
    run_manifest.json    topics, config, stats

Usage:
    python build_semantic_map_from_bag.py --bag "D:/.../saxon/hallway 1"
    python build_semantic_map_from_bag.py --bag "..." --max-frames 300 --no-preview
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

BASE_DIR = Path(__file__).resolve().parent
CODEBASE_DIR = BASE_DIR.parent
PROJECT_ROOT = CODEBASE_DIR.parent
OBJECT_DETECTION_DIR = CODEBASE_DIR / "07_object_detection"
ONTOLOGY_DIR = CODEBASE_DIR / "09_ontology"
DEFAULT_ONTOLOGY = ONTOLOGY_DIR / "ontology.rdf"

if str(OBJECT_DETECTION_DIR) not in sys.path:
    sys.path.insert(0, str(OBJECT_DETECTION_DIR))
if str(ONTOLOGY_DIR) not in sys.path:
    sys.path.insert(0, str(ONTOLOGY_DIR))

import rosbag_rgbd_sim_capture as capture  # noqa: E402  (path bootstrap must run first)
from fast_sam_segmenter import FastSAMSegmenter, overlay_mask_on_frame  # noqa: E402
from ontology_knowledge import OntologyKnowledgeBase  # noqa: E402
from rgbd_3d_filter import (  # noqa: E402
    extract_world_surface_points,
    get_oriented_3d_dimensions,
    load_dimensions_config,
)
from tf_tree import TFTree, normalize_frame  # noqa: E402


DEFAULT_OUT_ROOT = PROJECT_ROOT / "04_outputs_runs_and_logs" / "outputs" / "semantic_maps"

DEPTH_PATCH_HALF = 2          # 2 -> 5x5 median patch around the bbox midpoint
MIN_VALID_DEPTH_M = 0.20
MAX_VALID_DEPTH_M = 8.00

# Landmarks outside this height band almost always mean a broken TF chain.
SANE_WORLD_Z_MIN_M = -0.50
SANE_WORLD_Z_MAX_M = 3.00

DEFAULT_EXCLUDED_CLASSES = "person"
DEFAULT_DYNAMIC_CLASSES = "person,wheelchair"

LOG = logging.getLogger("semantic_map")


# ── Step 2: depth association ─────────────────────────────────────────────────
def sample_depth_m(depth_mm: np.ndarray, u: int, v: int, half: int = DEPTH_PATCH_HALF) -> float | None:
    """Median depth (metres) of the patch around (u, v); None when no valid pixels."""
    h, w = depth_mm.shape[:2]
    if not (0 <= u < w and 0 <= v < h):
        return None

    patch = depth_mm[max(0, v - half): min(h, v + half + 1), max(0, u - half): min(w, u + half + 1)]
    valid = patch[(patch > 0) & np.isfinite(patch)]
    if valid.size == 0:
        return None

    z_m = float(np.median(valid)) / 1000.0
    if not (MIN_VALID_DEPTH_M <= z_m <= MAX_VALID_DEPTH_M):
        return None
    return z_m


def sample_box_foreground_depth_m(
    depth_mm: np.ndarray,
    bbox_xyxy: list[float] | tuple[float, float, float, float],
    u: float,
    v: float,
    half: int = DEPTH_PATCH_HALF,
    mask: np.ndarray | None = None,
) -> tuple[float | None, float, float]:
    """
    Valid-depth masking for 3D depth association:
    Instead of relying on the geometric midpoint, extracts all depth values within
    the specific segmentation mask, filters out invalid or zero-depth pixels, and
    calculates the median depth of the remaining pixels.
    The median naturally resists extreme outliers like the background wall shining
    through mesh, giving the true distance to the physical material.
    Re-centers (u, v) on the median pixel coordinates of the valid physical surface.
    Returns: (z_m, adjusted_u, adjusted_v)
    """
    h_img, w_img = depth_mm.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    bx1, by1 = max(0, int(x1)), max(0, int(y1))
    bx2, by2 = min(w_img, int(x2)), min(h_img, int(y2))
    bw = bx2 - bx1
    bh = by2 - by1
    if bw < 4 or bh < 4:
        return None, u, v

    min_d_mm = MIN_VALID_DEPTH_M * 1000.0
    max_d_mm = MAX_VALID_DEPTH_M * 1000.0

    # 1. Neural segmentation mask valid-depth extraction
    if mask is not None:
        depth_roi = depth_mm[by1:by2, bx1:bx2].astype(np.float32)
        mask_roi = mask[by1:by2, bx1:bx2] > 0
        valid_mask = mask_roi & np.isfinite(depth_roi) & (depth_roi >= min_d_mm) & (depth_roi <= max_d_mm)
        if int(np.sum(valid_mask)) >= 10:
            valid_depths = depth_roi[valid_mask]
            median_d_mm = float(np.median(valid_depths))
            z_m = median_d_mm / 1000.0

            # Re-center (u, v) on the physical surface cluster around the median depth
            surface_cluster = valid_mask & (np.abs(depth_roi - median_d_mm) <= 350.0)
            if np.any(surface_cluster):
                ys, xs = np.where(surface_cluster)
            else:
                ys, xs = np.where(valid_mask)
            u_adj = float(bx1 + np.median(xs))
            v_adj = float(by1 + np.median(ys))
            return z_m, u_adj, v_adj

    # 2. Fallback if no segmentation mask or mask has insufficient valid depth pixels:
    # Use bounding box inner region to resist boundary bleeding
    inner_x1 = max(0, int(bx1 + 0.10 * bw))
    inner_x2 = min(w_img, int(bx2 - 0.10 * bw))
    inner_y1 = max(0, int(by1 + 0.10 * bh))
    inner_y2 = min(h_img, int(by2 - 0.05 * bh))
    if inner_x2 <= inner_x1 or inner_y2 <= inner_y1:
        inner_x1, inner_x2 = bx1, bx2
        inner_y1, inner_y2 = by1, by2

    depth_roi = depth_mm[inner_y1:inner_y2, inner_x1:inner_x2].astype(np.float32)
    valid_mask = np.isfinite(depth_roi) & (depth_roi >= min_d_mm) & (depth_roi <= max_d_mm)
    offset_x, offset_y = inner_x1, inner_y1

    if int(np.sum(valid_mask)) < 10:
        # Try full bounding box
        depth_roi = depth_mm[by1:by2, bx1:bx2].astype(np.float32)
        valid_mask = np.isfinite(depth_roi) & (depth_roi >= min_d_mm) & (depth_roi <= max_d_mm)
        offset_x, offset_y = bx1, by1
        if int(np.sum(valid_mask)) < 10:
            return None, u, v

    valid_depths = depth_roi[valid_mask]
    median_d_mm = float(np.median(valid_depths))
    z_m = median_d_mm / 1000.0

    surface_cluster = valid_mask & (np.abs(depth_roi - median_d_mm) <= 350.0)
    if np.any(surface_cluster):
        ys, xs = np.where(surface_cluster)
    else:
        ys, xs = np.where(valid_mask)
    u_adj = float(offset_x + np.median(xs))
    v_adj = float(offset_y + np.median(ys))
    return z_m, u_adj, v_adj


# ── Step 3: de-projection ─────────────────────────────────────────────────────
def deproject(u: float, v: float, z_m: float, intr: capture.CameraIntrinsics) -> tuple[float, float, float]:
    x = (float(u) - intr.cx) * z_m / intr.fx
    y = (float(v) - intr.cy) * z_m / intr.fy
    return x, y, z_m


# ── Step 4: ego-motion compensation ───────────────────────────────────────────
def transform_point(matrix_4x4, point_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    m = np.asarray(matrix_4x4, dtype=np.float64)
    p = np.array([point_xyz[0], point_xyz[1], point_xyz[2], 1.0], dtype=np.float64)
    w = m @ p
    return float(w[0]), float(w[1]), float(w[2])


# ── Step 5: semantic digital twin store ───────────────────────────────────────
class SemanticMapDB:
    """One SQLite file per bag: merged landmarks plus every raw observation."""

    def __init__(self, db_path: Path, merge_radius_m: float, dynamic_classes: set[str] | None = None) -> None:
        self.db_path = db_path
        self.merge_radius_m = float(merge_radius_m)
        self.dynamic_classes = dynamic_classes or set()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._create_schema()
        self._landmarks: dict[int, dict[str, Any]] = {}
        self._next_id = 1
        self._load_landmarks()

    def _create_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS semantic_map (
                landmark_id   INTEGER PRIMARY KEY,
                class_name    TEXT NOT NULL,
                instance_id   INTEGER NOT NULL,
                world_frame   TEXT NOT NULL,
                X             REAL NOT NULL,
                Y             REAL NOT NULL,
                Z             REAL NOT NULL,
                hit_count     INTEGER NOT NULL,
                mean_confidence REAL NOT NULL,
                max_confidence  REAL NOT NULL,
                first_seen_ns INTEGER NOT NULL,
                last_seen_ns  INTEGER NOT NULL,
                first_seen    TEXT NOT NULL,
                last_seen     TEXT NOT NULL,
                measured_width REAL,
                measured_depth REAL,
                measured_height REAL,
                class_votes_json TEXT,
                consensus_score REAL
            );

            CREATE TABLE IF NOT EXISTS observations (
                obs_id        INTEGER PRIMARY KEY AUTOINCREMENT,
                landmark_id   INTEGER,
                frame_index   INTEGER NOT NULL,
                timestamp_ns  INTEGER NOT NULL,
                source_class_name TEXT,
                class_name    TEXT NOT NULL,
                confidence    REAL,
                u             REAL NOT NULL,
                v             REAL NOT NULL,
                depth_m       REAL NOT NULL,
                cam_X REAL NOT NULL, cam_Y REAL NOT NULL, cam_Z REAL NOT NULL,
                world_X REAL, world_Y REAL, world_Z REAL,
                world_frame   TEXT,
                extrinsics_source TEXT,
                extrinsics_status TEXT,
                real_width    REAL,
                real_depth    REAL,
                real_height   REAL
            );

            CREATE INDEX IF NOT EXISTS idx_map_class ON semantic_map(class_name);
            CREATE INDEX IF NOT EXISTS idx_obs_landmark ON observations(landmark_id);

            CREATE TABLE IF NOT EXISTS camera_poses (
                frame_index   INTEGER PRIMARY KEY,
                timestamp_ns  INTEGER NOT NULL,
                world_frame   TEXT NOT NULL,
                camera_frame  TEXT NOT NULL,
                status        TEXT NOT NULL,
                X REAL NOT NULL, Y REAL NOT NULL, Z REAL NOT NULL,
                matrix_json   TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS map_runs (
                run_id        TEXT PRIMARY KEY,
                bag_path      TEXT NOT NULL,
                started_utc   TEXT NOT NULL,
                completed_utc TEXT,
                interrupted   INTEGER,
                stats_json    TEXT
            );

            CREATE TABLE IF NOT EXISTS manual_edits (
                edit_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                edited_utc    TEXT NOT NULL,
                action        TEXT NOT NULL,
                class_name    TEXT NOT NULL,
                instance_id   INTEGER NOT NULL,
                details       TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS class_remaps (
                source_class_name TEXT PRIMARY KEY,
                map_class_name    TEXT NOT NULL,
                updated_utc       TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS navigable_zones (
                zone_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                zone_name     TEXT NOT NULL,
                world_frame   TEXT NOT NULL,
                min_x         REAL NOT NULL,
                max_x         REAL NOT NULL,
                min_y         REAL NOT NULL,
                max_y         REAL NOT NULL,
                elevation_z   REAL NOT NULL,
                area_m2       REAL NOT NULL,
                point_count   INTEGER NOT NULL,
                polygon_json  TEXT,
                created_utc   TEXT NOT NULL
            );
            """
        )
        columns = {row[1] for row in self.conn.execute("PRAGMA table_info(semantic_map)")}
        if "instance_id" not in columns:
            self.conn.execute("ALTER TABLE semantic_map ADD COLUMN instance_id INTEGER NOT NULL DEFAULT 0")
            rows = self.conn.execute(
                "SELECT landmark_id, class_name FROM semantic_map ORDER BY class_name, landmark_id"
            ).fetchall()
            counts: dict[str, int] = {}
            for landmark_id, class_name in rows:
                counts[class_name] = counts.get(class_name, 0) + 1
                self.conn.execute(
                    "UPDATE semantic_map SET instance_id=? WHERE landmark_id=?",
                    (counts[class_name], landmark_id),
                )
        if "measured_width" not in columns:
            self.conn.execute("ALTER TABLE semantic_map ADD COLUMN measured_width REAL")
            self.conn.execute("ALTER TABLE semantic_map ADD COLUMN measured_depth REAL")
            self.conn.execute("ALTER TABLE semantic_map ADD COLUMN measured_height REAL")
        if "class_votes_json" not in columns:
            self.conn.execute("ALTER TABLE semantic_map ADD COLUMN class_votes_json TEXT")
        if "consensus_score" not in columns:
            self.conn.execute("ALTER TABLE semantic_map ADD COLUMN consensus_score REAL")
        if "surface_points_json" not in columns:
            self.conn.execute("ALTER TABLE semantic_map ADD COLUMN surface_points_json TEXT")
        self.conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_map_class_instance ON semantic_map(class_name, instance_id)"
        )
        observation_columns = {row[1] for row in self.conn.execute("PRAGMA table_info(observations)")}
        if "source_class_name" not in observation_columns:
            self.conn.execute("ALTER TABLE observations ADD COLUMN source_class_name TEXT")
            self.conn.execute("UPDATE observations SET source_class_name=class_name WHERE source_class_name IS NULL")
        if "real_width" not in observation_columns:
            self.conn.execute("ALTER TABLE observations ADD COLUMN real_width REAL")
            self.conn.execute("ALTER TABLE observations ADD COLUMN real_depth REAL")
            self.conn.execute("ALTER TABLE observations ADD COLUMN real_height REAL")
        self.conn.commit()

    def _load_landmarks(self) -> None:
        columns = {row[1] for row in self.conn.execute("PRAGMA table_info(semantic_map)")}
        has_measured = "measured_width" in columns
        has_consensus = "class_votes_json" in columns
        has_surface = "surface_points_json" in columns
        select_cols = (
            "landmark_id, class_name, instance_id, world_frame, X, Y, Z, "
            "hit_count, mean_confidence, max_confidence, first_seen_ns, last_seen_ns"
        )
        if has_measured:
            select_cols += ", measured_width, measured_depth, measured_height"
        if has_consensus:
            select_cols += ", class_votes_json, consensus_score"
        if has_surface:
            select_cols += ", surface_points_json"
        rows = self.conn.execute(f"SELECT {select_cols} FROM semantic_map").fetchall()
        for row in rows:
            lid, class_name, instance_id, world_frame, x, y, z, hits, mean_conf, max_conf, first_ns, last_ns = row[:12]
            record = {
                "class_name": class_name,
                "instance_id": int(instance_id),
                "world_frame": world_frame,
                "X": float(x), "Y": float(y), "Z": float(z),
                "hit_count": int(hits),
                "conf_sum": float(mean_conf) * int(hits),
                "max_confidence": float(max_conf),
                "first_seen_ns": int(first_ns), "last_seen_ns": int(last_ns),
            }
            idx = 12
            if has_measured and len(row) >= idx + 3:
                mw, md, mh = row[idx], row[idx + 1], row[idx + 2]
                record["measured_width"] = float(mw) if mw is not None else None
                record["measured_depth"] = float(md) if md is not None else None
                record["measured_height"] = float(mh) if mh is not None else None
                record["measured_count"] = int(hits) if (mw and mh) else 0
                idx += 3
            if has_consensus and len(row) >= idx + 2:
                votes_raw, score = row[idx], row[idx + 1]
                try:
                    record["class_votes"] = json.loads(votes_raw) if votes_raw else {class_name: float(mean_conf) * int(hits)}
                except Exception:
                    record["class_votes"] = {class_name: float(mean_conf) * int(hits)}
                record["consensus_score"] = float(score) if score is not None else 1.0
                idx += 2
            else:
                record["class_votes"] = {class_name: float(mean_conf) * int(hits)}
                record["consensus_score"] = 1.0
            if has_surface and len(row) > idx:
                surf_raw = row[idx]
                if surf_raw:
                    try:
                        record["surface_points"] = np.asarray(json.loads(surf_raw), dtype=np.float32)
                    except Exception:
                        record["surface_points"] = None
                else:
                    record["surface_points"] = None
            else:
                record["surface_points"] = None
            self._landmarks[int(lid)] = record
        self._next_id = max(self._landmarks, default=0) + 1
        self._class_remaps = dict(self.conn.execute("SELECT source_class_name, map_class_name FROM class_remaps"))

    def map_class_name(self, detector_class_name: str) -> str:
        mapped_name = detector_class_name
        visited: set[str] = set()
        while mapped_name in self._class_remaps and mapped_name not in visited:
            visited.add(mapped_name)
            mapped_name = self._class_remaps[mapped_name]
        return mapped_name

    def start_run(self, run_id: str, bag_path: Path) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO map_runs (run_id, bag_path, started_utc) VALUES (?,?,?)",
            (run_id, str(bag_path), datetime.now(timezone.utc).isoformat()),
        )
        self.conn.commit()

    def finish_run(self, run_id: str, interrupted: bool, stats: dict[str, Any]) -> None:
        self.conn.execute(
            "UPDATE map_runs SET completed_utc=?, interrupted=?, stats_json=? WHERE run_id=?",
            (datetime.now(timezone.utc).isoformat(), int(interrupted), json.dumps(stats), run_id),
        )
        self.conn.commit()

    def add_camera_pose(self, frame_index: int, timestamp_ns: int, world_frame: str,
                        camera_frame: str, status: str, matrix) -> None:
        m = np.asarray(matrix, dtype=np.float64)
        self.conn.execute(
            "INSERT OR REPLACE INTO camera_poses VALUES (?,?,?,?,?,?,?,?,?)",
            (int(frame_index), int(timestamp_ns), world_frame, camera_frame, status,
             float(m[0, 3]), float(m[1, 3]), float(m[2, 3]), json.dumps(m.tolist())),
        )

    def save_navigable_zone(self, zone_info: dict, world_frame: str) -> None:
        self.conn.execute(
            """
            INSERT INTO navigable_zones (
                zone_name, world_frame, min_x, max_x, min_y, max_y,
                elevation_z, area_m2, point_count, polygon_json, created_utc
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                zone_info.get("zone_name", "Floor"),
                world_frame,
                float(zone_info["min_x"]), float(zone_info["max_x"]),
                float(zone_info["min_y"]), float(zone_info["max_y"]),
                float(zone_info["elevation_z"]), float(zone_info["area_m2"]),
                int(zone_info.get("point_count", 0)),
                json.dumps(zone_info.get("polygon", [])),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self.conn.commit()

    def add_observation(self, record: dict[str, Any]) -> tuple[int | None, int | None]:
        """Merge into the nearest same-class landmark within the radius, else create a new one."""
        landmark_id = None
        if record["world_X"] is not None:
            landmark_id = self._merge(record)

        self.conn.execute(
            """
            INSERT INTO observations (
                landmark_id, frame_index, timestamp_ns, source_class_name, class_name, confidence,
                u, v, depth_m, cam_X, cam_Y, cam_Z,
                world_X, world_Y, world_Z, world_frame, extrinsics_source, extrinsics_status,
                real_width, real_depth, real_height
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                landmark_id, record["frame_index"], record["timestamp_ns"], record["source_class_name"],
                record["class_name"], record["confidence"], record["u"], record["v"], record["depth_m"],
                record["cam_X"], record["cam_Y"], record["cam_Z"],
                record["world_X"], record["world_Y"], record["world_Z"],
                record["world_frame"], record["extrinsics_source"], record["extrinsics_status"],
                record.get("real_width"), record.get("real_depth"), record.get("real_height"),
            ),
        )
        instance_id = self._landmarks[landmark_id]["instance_id"] if landmark_id is not None else None
        return landmark_id, instance_id

    def _merge(self, record: dict[str, Any]) -> int:
        wx, wy, wz = record["world_X"], record["world_Y"], record["world_Z"]
        conf = float(record["confidence"] or 0.0)
        det_class = record["class_name"]

        rw = record.get("real_width")
        rd = record.get("real_depth")
        rh = record.get("real_height")

        # Dynamic physical footprint calculated directly from the detected dimensions
        det_extent = max(rw or 0.45, rd or 0.45, 0.40)
        det_radius_xy = max(self.merge_radius_m, det_extent * 0.70)
        det_height = max(rh or 0.70, 0.70)

        # 1. Search for existing landmarks:
        # Priority 1: Same-class match within dynamic footprint
        best_same_id, best_same_dist_xy = None, float("inf")
        # Priority 2: Different-class occupant at same spot (for handling replacement e.g. bench replacing table)
        best_occupant_id, best_occupant_dist_xy = None, float("inf")

        for lid, lm in self._landmarks.items():
            dist_xy = math.dist((lm["X"], lm["Y"]), (wx, wy))
            dist_z = abs(lm["Z"] - wz)

            # Landmark dynamic footprint
            lm_extent = max(lm.get("measured_width") or 0.45, lm.get("measured_depth") or 0.45, 0.40)
            lm_radius_xy = max(det_radius_xy, lm_extent * 0.70)
            lm_height = max(lm.get("measured_height") or 0.70, det_height)

            # Height compatibility allows different viewing angles along the object's height
            z_compatible = dist_z <= (lm_height * 0.85 + 0.35)

            if dist_xy <= lm_radius_xy and z_compatible:
                if lm["class_name"] == det_class:
                    if dist_xy < best_same_dist_xy:
                        best_same_id, best_same_dist_xy = lid, dist_xy
                else:
                    if dist_xy < best_occupant_dist_xy:
                        best_occupant_id, best_occupant_dist_xy = lid, dist_xy

        # Target selection: prioritize same-class landmark, else fall back to spatial occupant
        best_id = best_same_id if best_same_id is not None else best_occupant_id
        best_dist = best_same_dist_xy if best_same_id is not None else best_occupant_dist_xy

        # 2. Fuse into existing 3D node if within dynamic envelope
        if best_id is not None:
            lm = self._landmarks[best_id]
            n = lm["hit_count"]
            if det_class in self.dynamic_classes:
                # Averaging a moving object smears it; keep the most recent fix.
                lm["X"], lm["Y"], lm["Z"] = wx, wy, wz
            else:
                lm["X"] = (lm["X"] * n + wx) / (n + 1)
                lm["Y"] = (lm["Y"] * n + wy) / (n + 1)
                lm["Z"] = (lm["Z"] * n + wz) / (n + 1)
            lm["conf_sum"] += conf
            lm["max_confidence"] = max(lm["max_confidence"], conf)
            lm["hit_count"] = n + 1
            lm["last_seen_ns"] = record["timestamp_ns"]

            # Multi-View Consensus: accumulate class confidence votes
            votes = lm.setdefault("class_votes", {})
            votes[det_class] = votes.get(det_class, 0.0) + max(0.01, conf)

            # Consensus winner is argmax(votes)
            winner_class = max(votes, key=votes.get)
            total_votes = sum(votes.values())
            old_class = lm["class_name"]
            lm["class_name"] = winner_class
            lm["consensus_score"] = float(votes[winner_class] / max(0.001, total_votes))

            # Maintain instance ID consistency if winning class shifted
            if old_class != winner_class:
                max_inst = max(
                    (other["instance_id"] for other_id, other in self._landmarks.items()
                     if other_id != best_id and other["class_name"] == winner_class),
                    default=0,
                )
                lm["instance_id"] = max_inst + 1

            # Progressive 3D physical dimension refinement from multi-view depth
            if rw is not None and rh is not None and rw > 0.02 and rh > 0.02:
                cur_w = lm.get("measured_width")
                if cur_w is not None and cur_w > 0:
                    mc = lm.get("measured_count", n)
                    lm["measured_width"] = (lm["measured_width"] * mc + rw) / (mc + 1)
                    lm["measured_depth"] = (lm["measured_depth"] * mc + (rd or rw)) / (mc + 1)
                    lm["measured_height"] = (lm["measured_height"] * mc + rh) / (mc + 1)
                    lm["measured_count"] = mc + 1
                else:
                    lm["measured_width"] = rw
                    lm["measured_depth"] = rd or rw
                    lm["measured_height"] = rh
                    lm["measured_count"] = 1

            # Multi-view point cloud aggregation for Rerun 3D surface visualization
            pts = record.get("surface_points")
            if pts is not None and len(pts) > 0:
                existing_pts = lm.get("surface_points")
                if existing_pts is not None and len(existing_pts) > 0:
                    merged_pts = np.vstack([existing_pts, pts])
                    if len(merged_pts) > 1500:
                        keep_idx = np.random.choice(len(merged_pts), size=1500, replace=False)
                        merged_pts = merged_pts[keep_idx]
                    lm["surface_points"] = merged_pts
                else:
                    lm["surface_points"] = pts

            LOG.debug(
                "      merged into landmark #%d (%s, cons=%.0f%%) d_xy=%.3fm -> (%.3f, %.3f, %.3f) hits=%d",
                best_id, lm["class_name"], lm["consensus_score"] * 100, best_dist,
                lm["X"], lm["Y"], lm["Z"], lm["hit_count"],
            )
            return best_id

        # 3. Spawn a new persistent 3D node
        lid = self._next_id
        self._next_id += 1
        instance_id = 1 + max(
            (lm["instance_id"] for lm in self._landmarks.values()
             if lm["class_name"] == det_class),
            default=0,
        )
        pts = record.get("surface_points")
        self._landmarks[lid] = {
            "class_name": det_class,
            "instance_id": instance_id,
            "world_frame": record["world_frame"],
            "X": wx, "Y": wy, "Z": wz,
            "hit_count": 1,
            "conf_sum": conf,
            "max_confidence": conf,
            "first_seen_ns": record["timestamp_ns"],
            "last_seen_ns": record["timestamp_ns"],
            "measured_width": rw if (rw and rw > 0.02) else None,
            "measured_depth": rd if (rd and rd > 0.02) else (rw if (rw and rw > 0.02) else None),
            "measured_height": rh if (rh and rh > 0.02) else None,
            "measured_count": 1 if (rw and rh and rw > 0.02 and rh > 0.02) else 0,
            "surface_points": pts if pts is not None else None,
            "class_votes": {det_class: max(0.01, conf)},
            "consensus_score": 1.0,
        }
        LOG.info(
            "      NEW landmark #%d '%s %d' at world (%.3f, %.3f, %.3f)",
            lid, det_class, instance_id, wx, wy, wz,
        )
        return lid

    def consolidate_landmarks(self) -> int:
        """
        Post-run agglomerative deduplication pass:
        Consolidates any landmarks of the same class whose physical footprints overlap on the ground plane.
        Ensures zero duplicate door/radiator/counter numbers in the final map.
        Returns the number of merged/removed duplicate landmarks.
        """
        merged_count = 0
        lids = sorted(self._landmarks.keys())
        to_delete: set[int] = set()

        for i, lid1 in enumerate(lids):
            if lid1 in to_delete or lid1 not in self._landmarks:
                continue
            lm1 = self._landmarks[lid1]

            for lid2 in lids[i + 1:]:
                if lid2 in to_delete or lid2 not in self._landmarks:
                    continue
                lm2 = self._landmarks[lid2]

                if lm1["class_name"] != lm2["class_name"]:
                    continue

                dist_xy = math.dist((lm1["X"], lm1["Y"]), (lm2["X"], lm2["Y"]))
                dist_z = abs(lm1["Z"] - lm2["Z"])

                extent1 = max(lm1.get("measured_width") or 0.45, lm1.get("measured_depth") or 0.45, 0.40)
                extent2 = max(lm2.get("measured_width") or 0.45, lm2.get("measured_depth") or 0.45, 0.40)
                radius_xy = max(self.merge_radius_m, max(extent1, extent2) * 0.70)
                height = max(lm1.get("measured_height") or 0.70, lm2.get("measured_height") or 0.70)

                if dist_xy <= radius_xy and dist_z <= (height * 0.85 + 0.35):
                    n1, n2 = lm1["hit_count"], lm2["hit_count"]
                    tot = n1 + n2
                    lm1["X"] = (lm1["X"] * n1 + lm2["X"] * n2) / tot
                    lm1["Y"] = (lm1["Y"] * n1 + lm2["Y"] * n2) / tot
                    lm1["Z"] = (lm1["Z"] * n1 + lm2["Z"] * n2) / tot
                    lm1["hit_count"] = tot
                    lm1["conf_sum"] += lm2["conf_sum"]
                    lm1["max_confidence"] = max(lm1["max_confidence"], lm2["max_confidence"])
                    lm1["first_seen_ns"] = min(lm1["first_seen_ns"], lm2["first_seen_ns"])
                    lm1["last_seen_ns"] = max(lm1["last_seen_ns"], lm2["last_seen_ns"])

                    votes1 = lm1.setdefault("class_votes", {})
                    for cls_name, v_score in lm2.get("class_votes", {}).items():
                        votes1[cls_name] = votes1.get(cls_name, 0.0) + v_score
                    winner_class = max(votes1, key=votes1.get)
                    lm1["class_name"] = winner_class
                    lm1["consensus_score"] = float(votes1[winner_class] / max(0.001, sum(votes1.values())))

                    mw1, mw2 = lm1.get("measured_width"), lm2.get("measured_width")
                    if mw1 and mw2:
                        lm1["measured_width"] = (mw1 * n1 + mw2 * n2) / tot
                        lm1["measured_depth"] = ((lm1.get("measured_depth") or mw1) * n1 + (lm2.get("measured_depth") or mw2) * n2) / tot
                        lm1["measured_height"] = ((lm1.get("measured_height") or 1.0) * n1 + (lm2.get("measured_height") or 1.0) * n2) / tot
                    elif mw2:
                        lm1["measured_width"] = lm2.get("measured_width")
                        lm1["measured_depth"] = lm2.get("measured_depth")
                        lm1["measured_height"] = lm2.get("measured_height")

                    pts1, pts2 = lm1.get("surface_points"), lm2.get("surface_points")
                    if pts1 is not None and pts2 is not None:
                        merged_pts = np.vstack([pts1, pts2])
                        if len(merged_pts) > 1500:
                            merged_pts = merged_pts[np.random.choice(len(merged_pts), size=1500, replace=False)]
                        lm1["surface_points"] = merged_pts
                    elif pts2 is not None:
                        lm1["surface_points"] = pts2

                    to_delete.add(lid2)
                    merged_count += 1
                    LOG.info(
                        "      [DEDUP] Consolidated duplicate landmark #%d into #%d '%s %d' (d_xy=%.3fm)",
                        lid2, lid1, lm1["class_name"], lm1["instance_id"], dist_xy,
                    )

        for lid in to_delete:
            del self._landmarks[lid]

        class_counters: dict[str, int] = {}
        for lid, lm in sorted(self._landmarks.items(), key=lambda kv: (kv[1]["class_name"], kv[1]["first_seen_ns"])):
            cname = lm["class_name"]
            class_counters[cname] = class_counters.get(cname, 0) + 1
            lm["instance_id"] = class_counters[cname]

        return merged_count

    def flush_landmarks(self) -> None:
        self.conn.execute("DELETE FROM semantic_map")
        rows = []
        for lid, lm in self._landmarks.items():
            pts = lm.get("surface_points")
            if pts is not None and isinstance(pts, np.ndarray):
                pts_json = json.dumps(pts.tolist())
            elif pts is not None and isinstance(pts, list):
                pts_json = json.dumps(pts)
            else:
                pts_json = None
            rows.append((
                lid, lm["class_name"], lm["instance_id"], lm["world_frame"], lm["X"], lm["Y"], lm["Z"],
                lm["hit_count"], lm["conf_sum"] / max(1, lm["hit_count"]), lm["max_confidence"],
                lm["first_seen_ns"], lm["last_seen_ns"],
                _ns_to_iso(lm["first_seen_ns"]), _ns_to_iso(lm["last_seen_ns"]),
                lm.get("measured_width"), lm.get("measured_depth"), lm.get("measured_height"),
                json.dumps(lm.get("class_votes", {lm["class_name"]: 1.0})),
                float(lm.get("consensus_score", 1.0)),
                pts_json,
            ))
        self.conn.executemany(
            """
            INSERT INTO semantic_map (
                landmark_id, class_name, instance_id, world_frame, X, Y, Z,
                hit_count, mean_confidence, max_confidence, first_seen_ns, last_seen_ns,
                first_seen, last_seen, measured_width, measured_depth, measured_height,
                class_votes_json, consensus_score, surface_points_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            rows,
        )
        self.conn.commit()

    @property
    def landmarks(self) -> dict[int, dict[str, Any]]:
        return self._landmarks

    def close(self) -> None:
        try:
            self.conn.commit()
            self.conn.close()
        except sqlite3.ProgrammingError:
            pass


def _ns_to_iso(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1e9, tz=timezone.utc).isoformat(timespec="milliseconds")


# ── Rendering ─────────────────────────────────────────────────────────────────
def get_class_bgr(class_name: str) -> tuple[int, int, int]:
    palette: dict[str, tuple[int, int, int]] = {
        "door": (240, 180, 50),            # Cyan / Sky blue
        "radiator": (40, 165, 255),         # Orange
        "dining table": (200, 80, 180),     # Purple
        "table": (200, 80, 180),
        "chair": (180, 100, 240),
        "bench": (160, 120, 200),
        "bench_hosp": (160, 120, 200),
        "hazmat_sign": (0, 215, 255),       # Amber / Yellow
        "exit_sign": (0, 230, 200),
        "power_socket": (255, 120, 40),     # Azure / Blue
        "general_bin": (60, 220, 60),       # Lime Green
        "hospital_stretcher": (220, 60, 60), # Blue-Red
        "infusion_pump": (80, 200, 220),
        "iv_stand": (220, 180, 100),
        "medical_tray": (140, 220, 120),
        "spillage": (180, 20, 255),         # Deep Pink
        "surgical_light": (40, 230, 230),   # Yellow
        "security_camera": (100, 100, 255), # Red
    }
    c_lower = class_name.lower().strip()
    if c_lower in palette:
        return palette[c_lower]
    h = abs(hash(c_lower))
    return (60 + (h & 0x7F), 60 + ((h >> 8) & 0x7F), 60 + ((h >> 16) & 0x7F))


def annotate(frame: np.ndarray, frame_index: int, timestamp_ns: int, pins: list[dict[str, Any]],
             landmark_total: int, extrinsics_status: str) -> np.ndarray:
    img = frame.copy()
    for pin in pins:
        x1, y1, x2, y2 = (int(v) for v in pin["bbox_xyxy"])
        ok = pin["world_X"] is not None
        excluded = pin["reject_reason"] == "excluded class"
        color = (150, 150, 150) if excluded else get_class_bgr(pin["class_name"]) if ok else (0, 100, 255)

        mask = pin.get("mask")
        if mask is not None and np.any(mask):
            overlay_mask_on_frame(img, mask, color, alpha=0.45, draw_contour=True)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.circle(img, (int(pin["u"]), int(pin["v"])), 4, (0, 0, 255), -1)

        is_floor = pin["class_name"].strip().lower() in ("floor", "navigable_floor")
        if is_floor:
            head = f"Floor {float(pin['confidence'] or 0):.0%}"
        else:
            instance_label = f" {pin['instance_id']}" if pin.get("instance_id") is not None else ""
            head = f"{pin['class_name']}{instance_label} {float(pin['confidence'] or 0):.0%}"
        body = (
            f"W({pin['world_X']:.2f},{pin['world_Y']:.2f},{pin['world_Z']:.2f}) d={pin['depth_m']:.2f}m"
            if ok else f"NO 3D: {pin['reject_reason']}"
        )

        (tw1, th1), _ = cv2.getTextSize(head, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        (tw2, th2), _ = cv2.getTextSize(body, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        max_tw = max(tw1, tw2)
        badge_y1 = max(0, y1 - th1 - th2 - 12) if y1 > 35 else y2 + 2
        badge_y2 = badge_y1 + th1 + th2 + 10
        cv2.rectangle(img, (x1, badge_y1), (min(img.shape[1], x1 + max_tw + 8), badge_y2), (20, 20, 20), -1)
        cv2.rectangle(img, (x1, badge_y1), (min(img.shape[1], x1 + max_tw + 8), badge_y2), color, 1)

        cv2.putText(img, head, (x1 + 4, badge_y1 + th1 + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        cv2.putText(img, body, (x1 + 4, badge_y1 + th1 + th2 + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)

    banner = (
        f"frame {frame_index} | t={timestamp_ns / 1e9:.3f}s | dets={len(pins)} "
        f"| landmarks={landmark_total} | tf={extrinsics_status}"
    )
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, banner, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logging(log_path: Path, verbose: bool) -> None:
    LOG.setLevel(logging.DEBUG)
    LOG.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    LOG.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    stream_handler.setFormatter(fmt)
    LOG.addHandler(stream_handler)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a semantic map (3D object pins) from a recorded ROS2 bag.")
    p.add_argument("--bag", required=True, help="Path to the ROS2 bag folder (sqlite3 or mcap)")
    p.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT), help="Root folder containing one persistent map folder per bag")

    p.add_argument("--rgb-topic", default=capture.DEFAULT_RGB_TOPIC)
    p.add_argument("--depth-topic", default=capture.DEFAULT_DEPTH_TOPIC)
    p.add_argument("--camera-info-topic", default=capture.DEFAULT_CAMERA_INFO_TOPIC)
    p.add_argument("--tf-topic", default=capture.DEFAULT_TF_TOPIC)
    p.add_argument("--tf-static-topic", default="/tf_static")
    p.add_argument("--odom-topic", default=capture.DEFAULT_ODOM_TOPIC)
    p.add_argument("--camera-frame-id", default="", help="Defaults to CameraInfo header.frame_id")
    p.add_argument("--world-frame", default="odom", help="Frame the map is anchored in (TF chain target)")

    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
    p.add_argument("--frame-stride", type=int, default=1, help="Process every Nth synchronized frame")
    p.add_argument("--max-rgb-depth-delta-ms", type=float, default=50.0)
    p.add_argument("--max-tf-delta-ms", type=float, default=100.0)
    p.add_argument("--max-odom-delta-ms", type=float, default=100.0)

    p.add_argument("--merge-radius-m", type=float, default=0.5,
                   help="Spatial clustering radius in metres (detections within this distance are fused into a 3D node)")
    p.add_argument("--reset-map", action="store_true",
                   help="Clear existing persistent world_map.db and start a fresh map without legacy seeding")
    p.add_argument("--min-consensus-hits", type=int, default=1,
                   help="Minimum multi-view hits required to export a landmark to Rerun / HTML (default 1)")
    p.add_argument("--min-confidence", type=float, default=0.0, help="Drop detections below this confidence")
    p.add_argument("--exclude-classes", default=DEFAULT_EXCLUDED_CLASSES,
                   help="Comma-separated classes never written to the map (still shown in preview)")
    p.add_argument("--dynamic-classes", default=DEFAULT_DYNAMIC_CLASSES,
                   help="Comma-separated classes tracked by last-known position instead of a running average")
    p.add_argument("--allow-odom-fallback", action="store_true",
                   help="When the TF chain fails, approximate the camera pose with raw /odom (base_link pose)")
    p.add_argument("--allow-no-extrinsics", action="store_true",
                   help="Fall back to identity pose (camera frame == world frame) when TF/odom is missing")

    p.add_argument("--detector-device", default="cuda")
    p.add_argument("--dino-interval", type=int, default=15,
                   help="Run the DINO fallback every Nth processed frame (1 = every frame)")
    p.add_argument("--disable-dino", action="store_true")
    p.add_argument("--disable-depth-gate", action="store_true")
    p.add_argument("--disable-fast-sam", action="store_true", help="Disable FastSAM neural instance segmentation")

    p.add_argument("--fast-sam-weights", default="", help="Path to FastSAM weights (default: FastSAM-s.pt)")
    p.add_argument("--detector-module-path", default="")
    p.add_argument("--v1-path", default="")
    p.add_argument("--v2-path", default="")
    p.add_argument("--v3-path", default="")

    p.add_argument("--no-preview", action="store_true", help="Disable the live OpenCV preview window")
    p.add_argument("--no-video", action="store_true", help="Do not record preview.mp4")
    p.add_argument("--rerun", action="store_true",
                   help="Export a new full world_map.rrd snapshot")
    p.add_argument("--no-rerun", action="store_false", dest="rerun",
                   help="Skip Rerun generation for this run")
    p.add_argument("--rerun-spawn", action="store_true", help="Open the Rerun viewer live while processing")
    p.add_argument("--rerun-cloud-stride", type=int, default=6,
                   help="Pixel stride when back-projecting depth into the scene cloud (higher = sparser)")
    p.add_argument("--rerun-cloud-every", type=int, default=5,
                   help="Add a depth cloud chunk every Nth processed frame")
    p.add_argument("--rerun-cloud-radius-m", type=float, default=0.006,
                   help="Rerun point-splat radius in metres (smaller reduces bubble appearance)")
    p.add_argument("--no-rerun-depth-smoothing", action="store_false", dest="rerun_depth_smoothing",
                   help="Use raw depth instead of edge-preserving bilateral smoothing")
    p.add_argument("--ontology", default=str(DEFAULT_ONTOLOGY),
                   help="RDF/OWL ontology embedded as landmark metadata in Rerun")
    p.add_argument("--enable-residual-discovery", action="store_true", default=True,
                   help="Run residual 3D DBSCAN clustering and two-tier cognition on keyframes")
    p.add_argument("--no-residual-discovery", action="store_false", dest="enable_residual_discovery",
                   help="Disable residual 3D clustering")
    p.add_argument("--residual-interval", type=int, default=35,
                   help="Run residual clustering every Nth processed frame (default: 35)")
    p.add_argument("--verbose", action="store_true", help="Print DEBUG lines to the console too")
    args = p.parse_args()
    if args.rerun_only:
        args.rerun = True
    return args


def build_detector(args: argparse.Namespace) -> capture.YoloEnsembleDinoDetector:
    module_path = (
        Path(args.detector_module_path).resolve()
        if args.detector_module_path
        else OBJECT_DETECTION_DIR / "YOLO_ensemble+DINO.py"
    )
    if not module_path.exists():
        raise FileNotFoundError(f"Detector module not found: {module_path}")

    import importlib.util

    spec = importlib.util.spec_from_file_location("detector_probe", str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to probe detector module at: {module_path}")
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)

    detector = capture.YoloEnsembleDinoDetector(
        module_path=module_path,
        device=str(args.detector_device),
        use_dino=not args.disable_dino,
        apply_depth_gate=not args.disable_depth_gate,
        v1_path=args.v1_path or str(getattr(probe, "V1_PATH")),
        v2_path=args.v2_path or str(getattr(probe, "V2_PATH")),
        v3_path=args.v3_path or str(getattr(probe, "V3_PATH")),
    )
    detector.load()
    return detector


def build_size_lookup():
    """Nominal (width, depth, height) in metres per class, from the ontology."""
    try:
        from rgbd_3d_filter import load_dimensions_config

        detector_module = OBJECT_DETECTION_DIR / "YOLO_ensemble+DINO.py"
        import importlib.util

        spec = importlib.util.spec_from_file_location("dims_probe", str(detector_module))
        probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(probe)

        limits = load_dimensions_config(probe.DIMENSIONS_CONFIG_PATH)
        aliases = probe.CLASS_NAME_ALIAS_CANDIDATES
    except Exception as exc:
        LOG.warning("[RERUN] Could not load object dimensions, using default box size: %s", exc)
        return None

    def lookup(class_name: str):
        for candidate in aliases.get(class_name, [class_name]):
            spec = limits.get(candidate)
            if spec:
                w = float(spec.get("width") or (spec["min_w"] + spec["max_w"]) / 2.0)
                d = float(spec.get("depth") or (spec.get("min_d", 0) + spec.get("max_d", 0)) / 2.0) or w
                h = float(spec.get("height") or (spec["min_h"] + spec["max_h"]) / 2.0)
                return w, d, h
        return None

    return lookup


def seed_persistent_map_from_legacy(out_root: Path, safe_label: str, db_path: Path) -> Path | None:
    """Carry forward the latest timestamped map the first time a bag gets persistent storage."""
    existing_count = 0
    if db_path.exists():
        try:
            with sqlite3.connect(db_path) as conn:
                existing_count = int(conn.execute("SELECT COUNT(*) FROM semantic_map").fetchone()[0])
        except sqlite3.Error:
            existing_count = 0
    if existing_count > 0:
        return None

    candidates = sorted(
        out_root.glob(f"semanticmap_{safe_label}_*/semantic_map.db"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for source in candidates:
        try:
            with sqlite3.connect(source) as conn:
                has_landmarks = int(conn.execute("SELECT COUNT(*) FROM semantic_map").fetchone()[0]) > 0
            if has_landmarks:
                shutil.copy2(source, db_path)
                return source
        except sqlite3.Error:
            continue
    return None


def main() -> None:
    args = parse_args()

    bag_path = Path(args.bag).resolve()
    if not bag_path.exists():
        raise FileNotFoundError(f"Bag not found: {bag_path}")
    if bag_path.is_dir() and not (bag_path / "metadata.yaml").exists():
        nested = [p.parent for p in bag_path.glob("*/metadata.yaml")]
        if nested:
            bag_path = nested[0]
    bag_label = bag_path.stem if bag_path.is_file() else bag_path.name
    safe_label = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in bag_label)

    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = Path(args.out_root).resolve() / safe_label
    run_dir.mkdir(parents=True, exist_ok=True)
    history_dir = run_dir / "run_history"
    history_dir.mkdir(exist_ok=True)

    setup_logging(run_dir / "run.log", args.verbose)
    LOG.info("=" * 78)
    LOG.info("SEMANTIC MAP BUILDER")
    LOG.info("Bag        : %s", bag_path)
    LOG.info("World map  : %s", run_dir)
    LOG.info("Run ID     : %s", run_id)
    LOG.info("Merge radius: %.2f m | stride: %d | max-frames: %s",
             args.merge_radius_m, args.frame_stride, args.max_frames or "all")

    excluded_classes = {c.strip() for c in args.exclude_classes.split(",") if c.strip()}
    dynamic_classes = {c.strip() for c in args.dynamic_classes.split(",") if c.strip()}
    LOG.info("World frame : %s | excluded: %s | dynamic: %s",
             args.world_frame, sorted(excluded_classes) or "none", sorted(dynamic_classes) or "none")
    LOG.info("=" * 78)

    # ── Read the bag once (topics are buffered by timestamp, same as the capture tool) ──
    LOG.info("[BAG] Opening bag and indexing topics ...")
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    with AnyReader([bag_path], default_typestore=typestore) as reader:
        available_topics = {str(c.topic): str(c.msgtype) for c in reader.connections}
        LOG.info("[BAG] %d topics found:", len(available_topics))
        for topic, msgtype in sorted(available_topics.items()):
            LOG.debug("        %-52s %s", topic, msgtype)

        missing = [t for t in (args.rgb_topic, args.depth_topic, args.camera_info_topic)
                   if t not in available_topics]
        if missing:
            raise RuntimeError(f"Missing required topics in bag: {missing}")

        selected = {args.rgb_topic, args.depth_topic, args.camera_info_topic,
                    args.tf_topic, args.tf_static_topic, args.odom_topic}
        topic_messages, skipped = capture._topic_message_map(reader, selected)

    rgb_messages = topic_messages.get(args.rgb_topic, [])
    depth_messages = topic_messages.get(args.depth_topic, [])
    caminfo_messages = topic_messages.get(args.camera_info_topic, [])
    tf_messages = topic_messages.get(args.tf_topic, [])
    tf_static_messages = topic_messages.get(args.tf_static_topic, [])
    odom_messages = topic_messages.get(args.odom_topic, [])

    LOG.info("[BAG] rgb=%d depth=%d camera_info=%d tf=%d tf_static=%d odom=%d (deserialize errors=%d)",
             len(rgb_messages), len(depth_messages), len(caminfo_messages),
             len(tf_messages), len(tf_static_messages), len(odom_messages),
             skipped.get("deserialize_errors", 0))

    if not rgb_messages or not depth_messages or not caminfo_messages:
        raise RuntimeError("Bag is missing RGB, depth or camera_info messages; cannot build a map.")

    tf_dynamic = capture._extract_tf_records(tf_messages)
    tf_static = capture._extract_tf_records(tf_static_messages)
    odom_records = capture._extract_odom_records(odom_messages)

    max_depth_delta_ns = int(args.max_rgb_depth_delta_ms * 1e6)
    max_tf_delta_ns = int(args.max_tf_delta_ms * 1e6)
    max_odom_delta_ns = int(args.max_odom_delta_ms * 1e6)

    tf_tree = TFTree(tf_dynamic, tf_static, max_tf_delta_ns)
    LOG.info("[TF ] dynamic frames: %s", sorted(tf_dynamic.keys()) or "(none)")
    LOG.info("[TF ] static  frames: %s", sorted(tf_static.keys()) or "(none)")
    if not tf_dynamic and not tf_static and not odom_records:
        LOG.warning("[TF ] No TF and no odometry in this bag. Step 4 cannot run. "
                    "Use --allow-no-extrinsics to map in the camera frame instead.")

    detector = None if args.rerun_only else build_detector(args)

    segmenter: FastSAMSegmenter | None = None
    if not args.rerun_only and not args.disable_fast_sam:
        try:
            weights_file = (
                Path(args.fast_sam_weights).resolve()
                if args.fast_sam_weights
                else (PROJECT_ROOT / "FastSAM-s.pt")
            )
            if not weights_file.exists():
                weights_file = Path("FastSAM-s.pt")
            segmenter = FastSAMSegmenter(weights_path=weights_file, device=args.detector_device)
            LOG.info("[FastSAM] Neural instance segmentation enabled (device: %s, weights: %s)",
                     args.detector_device, weights_file.name)
        except Exception as exc:
            LOG.warning("[FastSAM] Could not load FastSAM segmenter: %s. Continuing in bounding box mode.", exc)
            segmenter = None

    cognition_engine = None
    residual_clusterer = None
    if getattr(args, "enable_residual_discovery", False) and not args.rerun_only:
        try:
            from spatial_clustering import ResidualSpatialClustering
            from two_tier_classifier import TwoTierCognitionEngine
            cognition_engine = TwoTierCognitionEngine(
                ontology_path=Path(args.ontology).resolve(),
                device=args.detector_device,
            )
            LOG.info("[DISCOVERY] Two-Tier Residual Cognition active (interval: %d frames)", args.residual_interval)
        except Exception as exc:
            LOG.warning("[DISCOVERY] Could not initialize residual cognition: %s", exc)
    cam_ts = [m.timestamp_ns for m in caminfo_messages]

    db_path = run_dir / "world_map.db"
    if args.reset_map and db_path.exists():
        db_path.unlink()
        LOG.info("[MAP ] Reset map requested: deleted existing %s", db_path)

    migrated_from = None if args.reset_map else seed_persistent_map_from_legacy(Path(args.out_root).resolve(), safe_label, db_path)
    if migrated_from is not None:
        LOG.info("[MAP ] Seeded persistent map from legacy run: %s", migrated_from.parent.name)
    db = SemanticMapDB(db_path, args.merge_radius_m, dynamic_classes)
    db.start_run(run_id, bag_path)
    LOG.info("[MAP ] Loaded %d persistent landmarks from %s", len(db.landmarks), db.db_path.name)
    if args.rerun_only:
        LOG.info("[RERUN] Geometry-only replay: skipping YOLO and DINO inference.")

    scene = None
    rrd_path = run_dir / "world_map.rrd"
    if args.rerun:
        from rerun_logger import RerunSceneLogger

        if rrd_path.exists():
            archive_path = history_dir / f"world_map_{run_id}.rrd"
            rrd_path.replace(archive_path)
            LOG.info("[RERUN] Preserved previous recording at %s", archive_path)
        scene = RerunSceneLogger(
            rrd_path,
            application_id=f"semantic_map/{safe_label}",
            cloud_stride=args.rerun_cloud_stride,
            cloud_every_n_frames=args.rerun_cloud_every,
            cloud_smoothing=args.rerun_depth_smoothing,
            cloud_point_radius_m=args.rerun_cloud_radius_m,
            spawn_viewer=args.rerun_spawn,
        )
        LOG.info("[RERUN] Logging 3D scene to %s", rrd_path)
    writer: cv2.VideoWriter | None = None
    video_path = run_dir / "preview.mp4"
    if not args.no_preview:
        cv2.namedWindow("Semantic Map Builder", cv2.WINDOW_NORMAL)

    stats = {
        "synced_frames": 0, "processed_frames": 0, "skipped_stride": 0,
        "skipped_no_depth": 0, "skipped_depth_delta": 0, "skipped_bad_intrinsics": 0,
        "detections_total": 0, "pins_created": 0,
        "rejected_excluded_class": 0, "rejected_low_conf": 0,
        "rejected_no_depth_value": 0, "rejected_no_extrinsics": 0,
        "rejected_physical_dims": 0,
        "frames_missing_extrinsics": 0, "odom_fallback_frames": 0, "identity_pose_frames": 0,
        "implausible_height_pins": 0,
    }
    ontology_limits = load_dimensions_config(DEFAULT_ONTOLOGY)
    LOG.info("[ONTOLOGY] Loaded physical dimension limits for %d classes from %s",
             len(ontology_limits), DEFAULT_ONTOLOGY.name)
    class_histogram: dict[str, int] = {}
    world_frame_seen: set[str] = set()
    odom_ts = [r.timestamp_ns for r in odom_records]
    dino_enabled = not args.disable_dino
    chain_logged = False

    depth_cursor = cam_cursor = 0
    matched_counter = processed_counter = 0
    interrupted = False

    try:
        for rgb_item in rgb_messages:
            if args.max_frames > 0 and processed_counter >= args.max_frames:
                break

            rgb_img = capture._decode_color_image(rgb_item.message)
            if rgb_img is None:
                continue

            depth_idx = capture._closest_index(depth_ts, rgb_item.timestamp_ns, start_idx=depth_cursor)
            if depth_idx < 0:
                stats["skipped_no_depth"] += 1
                continue
            depth_cursor = max(depth_cursor, depth_idx)
            depth_delta = abs(depth_messages[depth_idx].timestamp_ns - rgb_item.timestamp_ns)
            if depth_delta > max_depth_delta_ns:
                stats["skipped_depth_delta"] += 1
                continue

            depth_mm = capture._decode_depth_image_mm(depth_messages[depth_idx].message)
            if depth_mm is None:
                stats["skipped_no_depth"] += 1
                continue

            cam_idx = capture._closest_index(cam_ts, rgb_item.timestamp_ns, start_idx=cam_cursor)
            cam_cursor = max(cam_cursor, cam_idx)
            intr = capture._intrinsics_from_camera_info(caminfo_messages[cam_idx].message)
            if not capture._is_intrinsics_sane(intr):
                stats["skipped_bad_intrinsics"] += 1
                continue

            stats["synced_frames"] += 1
            if matched_counter < args.start_frame:
                matched_counter += 1
                continue
            if (matched_counter - args.start_frame) % max(1, args.frame_stride) != 0:
                matched_counter += 1
                stats["skipped_stride"] += 1
                continue
            matched_counter += 1

            camera_frame_id = args.camera_frame_id.strip()
            if not camera_frame_id:
                header = getattr(caminfo_messages[cam_idx].message, "header", None)
                camera_frame_id = str(getattr(header, "frame_id", "")).strip()

            if not chain_logged:
                LOG.info("[TF ] camera frame  : %s", normalize_frame(camera_frame_id))
                LOG.info("[TF ] resolved chain: %s",
                         tf_tree.describe_chain(camera_frame_id, args.world_frame))
                chain_logged = True

            # Step 4: pose of the camera optical frame in the world at this exact timestamp.
            pose_matrix, extrinsics_status = tf_tree.lookup(
                camera_frame_id, args.world_frame, rgb_item.timestamp_ns
            )
            world_frame = normalize_frame(args.world_frame)
            extrinsics_source = "tf_chain"

            if pose_matrix is None:
                stats["frames_missing_extrinsics"] += 1
                if args.allow_odom_fallback and odom_records:
                    odom_idx = capture._closest_index(odom_ts, rgb_item.timestamp_ns)
                    odom_rec = odom_records[odom_idx]
                    if abs(odom_rec.timestamp_ns - rgb_item.timestamp_ns) <= max_odom_delta_ns:
                        pose_matrix = np.asarray(odom_rec.matrix_4x4, dtype=np.float64)
                        world_frame = normalize_frame(odom_rec.frame_id or "odom")
                        extrinsics_status = "odom_fallback_approx"
                        extrinsics_source = "odom"
                        stats["odom_fallback_frames"] += 1
                        if stats["odom_fallback_frames"] == 1:
                            LOG.warning("[TF ] TF chain unavailable (%s). Falling back to raw /odom: this is the "
                                        "base_link pose and ignores the camera mount offset and axis convention.",
                                        extrinsics_status)
                if pose_matrix is None and args.allow_no_extrinsics:
                    pose_matrix = np.eye(4)
                    world_frame = "camera_identity"
                    extrinsics_status = "identity_fallback"
                    extrinsics_source = "identity"
                    stats["identity_pose_frames"] += 1

            # Step 1: detection on RGB only.
            if detector is None:
                detections = []
            else:
                detector.use_dino = dino_enabled and (processed_counter % max(1, args.dino_interval) == 0)
                detections = detector.infer(rgb_img, depth_mm, intr)
            stats["detections_total"] += len(detections)
            processed_counter += 1
            stats["processed_frames"] += 1

            # Step 1.5: FastSAM neural instance segmentation (box-prompted mode)
            masks: list[tuple[np.ndarray, np.ndarray]] = []
            if segmenter is not None and detections:
                try:
                    boxes = [det["bbox_xyxy"] for det in detections]
                    masks = segmenter.segment_boxes(rgb_img, boxes)
                except Exception as exc:
                    LOG.warning("[FastSAM] Segmentation error at frame %d: %s", processed_counter, exc)
                    masks = []

            LOG.debug(
                "[FRAME %05d] t=%.3fs depth_dt=%.1fms dets=%d tf=%s (%s)",
                processed_counter, rgb_item.timestamp_ns / 1e9, depth_delta / 1e6,
                len(detections), extrinsics_status,
                camera_frame_id or "unknown-frame",
            )

            pins: list[dict[str, Any]] = []
            for det_idx, det in enumerate(detections):
                x1, y1, x2, y2 = det["bbox_xyxy"]
                source_class_name = det["class_label"]
                class_name = db.map_class_name(source_class_name)
                conf = det.get("confidence")
                class_histogram[source_class_name] = class_histogram.get(source_class_name, 0) + 1

                raw_mask = masks[det_idx][0] if det_idx < len(masks) else None
                eroded_mask = masks[det_idx][1] if det_idx < len(masks) else None

                pin: dict[str, Any] = {
                    "frame_index": processed_counter,
                    "timestamp_ns": rgb_item.timestamp_ns,
                    "class_name": class_name,
                    "source_class_name": source_class_name,
                    "confidence": conf,
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "u": (x1 + x2) / 2.0,
                    "v": (y1 + y2) / 2.0,
                    "depth_m": 0.0,
                    "cam_X": 0.0, "cam_Y": 0.0, "cam_Z": 0.0,
                    "world_X": None, "world_Y": None, "world_Z": None,
                    "world_frame": None,
                    "instance_id": None,
                    "extrinsics_source": extrinsics_source if pose_matrix is not None else None,
                    "extrinsics_status": extrinsics_status,
                    "reject_reason": "",
                    "mask": raw_mask,
                    "eroded_mask": eroded_mask,
                    "surface_points": None,
                }

                if class_name in excluded_classes:
                    stats["rejected_excluded_class"] += 1
                    pin["reject_reason"] = "excluded class"
                    pins.append(pin)
                    continue

                if conf is not None and conf < args.min_confidence:
                    stats["rejected_low_conf"] += 1
                    pin["reject_reason"] = "low confidence"
                    pins.append(pin)
                    continue

                # Step 2: robust foreground depth association (handles hollow furniture/chairs)
                z_m, adj_u, adj_v = sample_box_foreground_depth_m(
                    depth_mm, [x1, y1, x2, y2], pin["u"], pin["v"], mask=raw_mask
                )
                if z_m is None:
                    stats["rejected_no_depth_value"] += 1
                    pin["reject_reason"] = "no valid depth"
                    LOG.debug("      %-18s bbox=(%.0f,%.0f,%.0f,%.0f) -> no valid depth in bbox",
                              class_name, x1, y1, x2, y2)
                    pins.append(pin)
                    continue
                pin["depth_m"] = z_m
                pin["u"] = adj_u
                pin["v"] = adj_v

                # Measure real 3D physical dimensions from camera depth point cloud (guided by raw mask)
                dims_3d = get_oriented_3d_dimensions(depth_mm, x1, y1, x2, y2, intr, mask=raw_mask)
                if dims_3d is not None:
                    pin["real_width"], pin["real_depth"], pin["real_height"] = dims_3d

                    # Step 3: Physical 3D Geometric Gatekeeper directly from Ontology
                    # Checks measured (rw, rd, rh) against limits loaded from ontology.rdf
                    # Chairs are exempt from physical dimension gating (variable orientations, occlusions, side views)
                    if class_name in ontology_limits and class_name != "chair":
                        lim = ontology_limits[class_name]
                        rw, rd, rh = dims_3d
                        w_ok = lim["min_w"] <= rw <= lim["max_w"]
                        h_ok = lim["min_h"] <= rh <= lim["max_h"]
                        min_d = lim.get("min_d", 0.0)
                        max_d = lim.get("max_d", 0.0)
                        d_ok = True
                        if min_d > 0.0 and rd < min_d:
                            d_ok = False
                        if max_d > 0.0 and rd > max_d:
                            d_ok = False

                        if not (w_ok and h_ok and d_ok):
                            stats["rejected_physical_dims"] = stats.get("rejected_physical_dims", 0) + 1
                            pin["reject_reason"] = f"failed ontology dimension gate ({rw:.2f}x{rd:.2f}x{rh:.2f}m vs ontology W[{lim['min_w']}-{lim['max_w']}], H[{lim['min_h']}-{lim['max_h']}])"
                            LOG.debug("      %-18s dropped: failed ontology dimension gate (%.2fx%.2fx%.2fm vs ontology)",
                                      class_name, rw, rd, rh)
                            pins.append(pin)
                            continue
                else:
                    pin["real_width"] = pin["real_depth"] = pin["real_height"] = None

                # Step 3: stays in the camera optical frame; the TF chain handles the axis convention.
                cx, cy, cz = deproject(pin["u"], pin["v"], z_m, intr)
                pin["cam_X"], pin["cam_Y"], pin["cam_Z"] = cx, cy, cz

                if pose_matrix is None:
                    stats["rejected_no_extrinsics"] += 1
                    pin["reject_reason"] = extrinsics_status
                    LOG.debug("      %-18s cam=(%.2f,%.2f,%.2f) d=%.2fm -> dropped: %s",
                              class_name, cx, cy, cz, z_m, extrinsics_status)
                    pins.append(pin)
                    continue

                # Step 4
                wx, wy, wz = transform_point(pose_matrix, (cx, cy, cz))
                world_frame_seen.add(world_frame)
                pin["world_X"], pin["world_Y"], pin["world_Z"] = wx, wy, wz
                pin["world_frame"] = world_frame

                LOG.debug("      %-18s conf=%.2f d=%.2fm cam=(%.2f,%.2f,%.2f) -> world[%s]=(%.3f,%.3f,%.3f)",
                          class_name, float(conf or 0), z_m, cx, cy, cz, world_frame, wx, wy, wz)

                if not (SANE_WORLD_Z_MIN_M <= wz <= SANE_WORLD_Z_MAX_M):
                    stats["implausible_height_pins"] += 1
                    if stats["implausible_height_pins"] <= 5:
                        LOG.warning("      %-18s world Z=%.2fm is outside [%.1f, %.1f] - check the TF chain "
                                    "(is depth leaking into the height axis?)",
                                    class_name, wz, SANE_WORLD_Z_MIN_M, SANE_WORLD_Z_MAX_M)

                # Extract 3D physical surface points (guided by raw mask)
                pin["surface_points"] = extract_world_surface_points(
                    depth_mm, x1, y1, x2, y2, intr, pose_matrix, mask=raw_mask, max_points=300
                )

                # Step 5
                _, pin["instance_id"] = db.add_observation(pin)
                stats["pins_created"] += 1
                pins.append(pin)

            # Residual 3D DBSCAN Clustering & Two-Tier Cognition (Keyframes)
            if (cognition_engine is not None and pose_matrix is not None
                    and (processed_counter % max(1, args.residual_interval) == 0)):
                if residual_clusterer is None:
                    residual_clusterer = ResidualSpatialClustering(intr)

                known_regions = []
                for det_idx, det in enumerate(detections):
                    if det_idx < len(masks) and masks[det_idx][0] is not None:
                        known_regions.append(masks[det_idx][0])
                    else:
                        known_regions.append(det["bbox_xyxy"])

                try:
                    res_clusters = residual_clusterer.extract_residual_clusters(
                        rgb_img, depth_mm, known_regions, pose_matrix=pose_matrix
                    )
                    for cluster in res_clusters:
                        c_res = cognition_engine.classify_crop(cluster.crop_rgb)
                        if c_res.is_novel and c_res.class_name != "unidentified_obstacle":
                            cognition_engine.ingest_novel_concept(c_res.class_name, dimensions_m=cluster.dimensions_m)

                        c_pin = {
                            "frame_index": processed_counter,
                            "timestamp_ns": rgb_item.timestamp_ns,
                            "class_name": c_res.class_name,
                            "source_class_name": f"[{c_res.tier}] {c_res.class_name}",
                            "confidence": c_res.confidence,
                            "bbox_xyxy": list(cluster.bbox_2d),
                            "u": (cluster.bbox_2d[0] + cluster.bbox_2d[2]) / 2.0,
                            "v": (cluster.bbox_2d[1] + cluster.bbox_2d[3]) / 2.0,
                            "depth_m": cluster.centroid_cam[2],
                            "cam_X": cluster.centroid_cam[0],
                            "cam_Y": cluster.centroid_cam[1],
                            "cam_Z": cluster.centroid_cam[2],
                            "world_X": cluster.centroid_world[0],
                            "world_Y": cluster.centroid_world[1],
                            "world_Z": cluster.centroid_world[2],
                            "world_frame": world_frame,
                            "instance_id": None,
                            "extrinsics_source": extrinsics_source,
                            "extrinsics_status": extrinsics_status,
                            "reject_reason": "",
                            "mask": None,
                            "eroded_mask": None,
                            "surface_points": cluster.points_world,
                            "real_width": cluster.dimensions_m[0],
                            "real_depth": cluster.dimensions_m[1],
                            "real_height": cluster.dimensions_m[2],
                        }
                        _, c_pin["instance_id"] = db.add_observation(c_pin)
                        pins.append(c_pin)
                        stats["pins_created"] += 1
                        stats["residual_discovered"] = stats.get("residual_discovered", 0) + 1
                        LOG.debug("      [RESIDUAL] Discovered %s at (%.2f, %.2f, %.2f) dims=%.2fx%.2fx%.2fm",
                                  c_res.class_name, cluster.centroid_world[0], cluster.centroid_world[1], cluster.centroid_world[2],
                                  cluster.dimensions_m[0], cluster.dimensions_m[1], cluster.dimensions_m[2])
                except Exception as exc:
                    LOG.debug("[DISCOVERY] Error in residual clustering: %s", exc)
                db.conn.commit()

            if pose_matrix is not None:
                db.add_camera_pose(processed_counter, rgb_item.timestamp_ns, world_frame,
                                   normalize_frame(camera_frame_id), extrinsics_status, pose_matrix)
                if scene is not None:
                    scene.log_frame(processed_counter, rgb_item.timestamp_ns,
                                    rgb_img, depth_mm, intr, pose_matrix, detections=pins)

            annotated = annotate(rgb_img, processed_counter, rgb_item.timestamp_ns,
                                 pins, len(db.landmarks), extrinsics_status)

            if not args.no_video:
                if writer is None:
                    h, w = annotated.shape[:2]
                    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (w, h))
                writer.write(annotated)

            if not args.no_preview:
                cv2.imshow("Semantic Map Builder", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    LOG.warning("[USER] 'q' pressed - stopping early.")
                    interrupted = True
                    break

            if processed_counter % 25 == 0:
                db.flush_landmarks()
                LOG.info("[PROGRESS] frames=%d dets=%d pins=%d landmarks=%d",
                         processed_counter, stats["detections_total"],
                         stats["pins_created"], len(db.landmarks))
    except KeyboardInterrupt:
        LOG.warning("[USER] KeyboardInterrupt - stopping early.")
        interrupted = True
    finally:
        dedup_count = db.consolidate_landmarks()
        if dedup_count > 0:
            LOG.info("[MAP ] Consolidated %d duplicate landmarks across runs.", dedup_count)
        db.flush_landmarks()
        if scene is not None:
            zone_info = scene.log_navigable_zone()
            if zone_info is not None:
                primary_frame = sorted(world_frame_seen)[0] if world_frame_seen else args.world_frame
                db.save_navigable_zone(zone_info, primary_frame)
                LOG.info("[ZONE ] Navigable floor zone saved: area=%.2f m², bounds=[%.2f, %.2f]x[%.2f, %.2f]",
                         zone_info["area_m2"], zone_info["min_x"], zone_info["max_x"], zone_info["min_y"], zone_info["max_y"])
            consensus_landmarks = {
                lid: lm for lid, lm in db.landmarks.items()
                if lm["hit_count"] >= max(1, args.min_consensus_hits)
            }
            knowledge_base = OntologyKnowledgeBase(Path(args.ontology).resolve())
            knowledge_by_class = {
                class_name: knowledge_base.resolve(class_name)
                for class_name in {landmark["class_name"] for landmark in consensus_landmarks.values()}
            }
            for landmark in consensus_landmarks.values():
                landmark["ontology"] = knowledge_by_class[landmark["class_name"]]
            scene.log_landmarks(consensus_landmarks, size_lookup=build_size_lookup())
            scene.finish()
            LOG.info("[RERUN] Scene written: %d cloud chunks, %d points, %d trajectory poses, %d consensus landmarks",
                     scene.cloud_chunks, scene.points_logged, len(scene.trajectory), len(consensus_landmarks))
        if writer is not None:
            writer.release()
            if video_path.exists() and shutil.which("ffmpeg"):
                try:
                    tmp_h264 = video_path.with_name("preview_h264.mp4")
                    cmd = [
                        "ffmpeg", "-y", "-i", str(video_path),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "22", "-preset", "fast", str(tmp_h264)
                    ]
                    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
                    if res.returncode == 0 and tmp_h264.exists():
                        tmp_h264.replace(video_path)
                        LOG.info("[VIDEO] Converted preview to web-compatible H.264: %s", video_path)
                except Exception as exc:
                    LOG.debug("ffmpeg post-encoding skipped: %s", exc)
        cv2.destroyAllWindows()

    from view_semantic_map import refresh_semantic_map_html
    from export_clean_rrd import export_clean_rrd
    from hospital_compliance import HospitalComplianceEvaluator, generate_markdown_report

    # 1. Hospital Health & Safety Compliance Audit (KMi Spatial Reasoner + Andrea's Egress Zones)
    comp_md_path = db_path.with_name("hospital_compliance_report.md")
    comp_json_path = db_path.with_name("hospital_compliance_report.json")
    try:
        compliance_eval = HospitalComplianceEvaluator(db_path)
        comp_report = compliance_eval.evaluate_all()
        with open(comp_json_path, "w", encoding="utf-8") as f:
            json.dump(comp_report, f, indent=2)
        with open(comp_md_path, "w", encoding="utf-8") as f:
            f.write(generate_markdown_report(comp_report))
        comp_status = "COMPLIANT" if comp_report["overall_compliant"] else "NON-COMPLIANT"
        LOG.info("[AUDIT] Hospital H&S Compliance: %s (%d violations detected) -> %s",
                 comp_status, comp_report["total_violations"], comp_md_path)
    except Exception as exc:
        LOG.warning("Failed to evaluate hospital compliance: %s", exc)

    # 2. Export canonical lightweight RRD (< 3 MB) for instant sub-second visualization
    canonical_rrd_path = db_path.with_name("world_map.rrd")
    try:
        export_clean_rrd(db_path, canonical_rrd_path)
        LOG.info("[RERUN] Canonical 3D digital twin written: %s", canonical_rrd_path)
    except Exception as exc:
        LOG.warning("Failed to export canonical RRD: %s", exc)

    # 3. Export canonical interactive HTML 3D Digital Twin
    html_path = refresh_semantic_map_html(db_path, Path(args.ontology).resolve())
    LOG.info("[HTML ] Current semantic-map viewer written: %s", html_path)

    # ── Report ────────────────────────────────────────────────────────────────
    LOG.info("=" * 78)
    LOG.info("RUN COMPLETE%s", " (interrupted)" if interrupted else "")
    for key, value in stats.items():
        LOG.info("  %-28s %s", key, value)
    LOG.info("  %-28s %d", "landmarks", len(db.landmarks))
    LOG.info("-" * 78)
    LOG.info("SEMANTIC MAP (world frame: %s | min consensus: >= %d hits)",
             ", ".join(sorted(world_frame_seen)) or "none", args.min_consensus_hits)
    for lid, lm in sorted(db.landmarks.items(), key=lambda kv: kv[1]["class_name"]):
        score_str = f" cons={lm['consensus_score']:.0%}" if lm.get("consensus_score") is not None else ""
        is_floor = lm["class_name"].strip().lower() in ("floor", "navigable_floor")
        name_str = "Floor" if is_floor else f"{lm['class_name']} {lm['instance_id']}"
        LOG.info("  %-22s (%8.3f, %8.3f, %8.3f)  hits=%-4d conf=%.2f%s",
                 name_str, lm["X"], lm["Y"], lm["Z"],
                 lm["hit_count"], lm["conf_sum"] / max(1, lm["hit_count"]), score_str)
    LOG.info("=" * 78)

    manifest = {
        "schema_version": "1.0.0",
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "bag_path": str(bag_path),
        "run_dir": str(run_dir),
        "interrupted": interrupted,
        "topics": {
            "rgb": args.rgb_topic, "depth": args.depth_topic,
            "camera_info": args.camera_info_topic, "tf": args.tf_topic,
            "tf_static": args.tf_static_topic, "odom": args.odom_topic,
        },
        "available_topics": available_topics,
        "camera_frame_id": args.camera_frame_id,
        "world_frame": args.world_frame,
        "world_frames": sorted(world_frame_seen),
        "configuration": {
            "merge_radius_m": args.merge_radius_m,
            "frame_stride": args.frame_stride,
            "start_frame": args.start_frame,
            "max_frames": args.max_frames,
            "min_confidence": args.min_confidence,
            "excluded_classes": sorted(excluded_classes),
            "dynamic_classes": sorted(dynamic_classes),
            "allow_odom_fallback": args.allow_odom_fallback,
            "allow_no_extrinsics": args.allow_no_extrinsics,
            "dino_enabled": dino_enabled,
            "dino_interval": args.dino_interval,
            "depth_gate_enabled": not args.disable_depth_gate,
            "detector_device": None if detector is None else detector.device,
            "rerun_only": args.rerun_only,
            "min_consensus_hits": args.min_consensus_hits,
        },
        "stats": stats,
        "landmark_count": len(db.landmarks),
        "class_histogram": class_histogram,
    }
    (history_dir / f"{run_id}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (run_dir / "latest_run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    db.finish_run(run_id, interrupted, stats)
    db.close()
    LOG.info("Database : %s", run_dir / "world_map.db")
    LOG.info("Log file : %s", run_dir / "run.log")
    LOG.info("History  : %s", history_dir / f"{run_id}.json")
    if args.rerun:
        LOG.info("Rerun 3D : %s", canonical_rrd_path)
    LOG.info("HTML 3D  : %s", html_path)
    LOG.info("Audit    : %s", comp_md_path)
    if not args.no_video:
        LOG.info("Preview  : %s", video_path)


if __name__ == "__main__":
    main()
