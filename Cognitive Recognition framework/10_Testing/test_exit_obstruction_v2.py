"""CPU-only tests for v2 blocked-exit RGB-D geometry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import tempfile
import unittest
import sqlite3

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2_DIR = PROJECT_ROOT / "01_codebase" / "06_anomaly_detection" / "Blocked_exit_detection" / "v2"
SEMANTIC_MAP_DIR = PROJECT_ROOT / "01_codebase" / "08_semantic_map"
sys.path.insert(0, str(V2_DIR))
sys.path.insert(0, str(SEMANTIC_MAP_DIR))

from exit_obstruction import ExitObstructionResult, mask_intersects_keep_clear_zone, point_inside_keep_clear_zone, world_keep_clear_strips
from build_semantic_map_from_bag import SemanticMapDB


@dataclass
class Intrinsics:
    fx: float = 100.0
    fy: float = 100.0
    cx: float = 2.0
    cy: float = 2.0


class ExitObstructionGeometryTests(unittest.TestCase):
    def test_point_in_front_of_door_is_inside_zone(self) -> None:
        self.assertTrue(point_inside_keep_clear_zone((0.0, 0.0, 1.5), (0.0, 0.0, 2.0), 1.0, -1.0, 1.0))

    def test_point_behind_door_is_clear(self) -> None:
        self.assertFalse(point_inside_keep_clear_zone((0.0, 0.0, 2.1), (0.0, 0.0, 2.0), 1.0, -1.0, 1.0))

    def test_world_zone_bottom_is_on_transformed_floor(self) -> None:
        result = ExitObstructionResult(
            door_camera_xyz=(0.0, 0.0, 2.0),
            door_top_y=-1.0,
            door_bottom_y=1.0,
            floor_y=0.0,
        )
        camera_to_world = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=float,
        )
        strips = world_keep_clear_strips(result, camera_to_world, 1.0)
        bottom_z = [point[2] for point in strips[0]]
        self.assertTrue(all(abs(value) < 1e-6 for value in bottom_z))

    def test_mask_depth_pixel_inside_zone_blocks(self) -> None:
        depth_mm = np.zeros((5, 5), dtype=np.uint16)
        depth_mm[2, 2] = 1500
        mask = np.zeros_like(depth_mm, dtype=np.uint8)
        mask[2, 2] = 255
        self.assertTrue(mask_intersects_keep_clear_zone(mask, depth_mm, Intrinsics(), (0.0, 0.0, 2.0), 1.0, -1.0, 1.0))

    def test_mask_without_valid_depth_is_clear(self) -> None:
        depth_mm = np.zeros((5, 5), dtype=np.uint16)
        mask = np.zeros_like(depth_mm, dtype=np.uint8)
        mask[2, 2] = 255
        self.assertFalse(mask_intersects_keep_clear_zone(mask, depth_mm, Intrinsics(), (0.0, 0.0, 2.0), 1.0, -1.0, 1.0))

    def test_semantic_map_persists_blocked_event(self) -> None:
        result = ExitObstructionResult(
            door_camera_xyz=(0.0, 0.0, 2.0),
            door_top_y=-1.0,
            door_bottom_y=1.0,
            blockers=[{"object_index": 1, "bbox_xyxy": [1.0, 1.0, 3.0, 3.0], "confidence": 0.9, "depth_m": 1.5}],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            database = SemanticMapDB(Path(temp_dir) / "world_map.db", 0.5)
            database.add_egress_obstruction_event(
                "test-run", 7, 123456789, "odom", result, (3.0, 4.0, 1.0), 1.0
            )
            row = database.conn.execute(
                "SELECT frame_index, world_frame, door_world_X, obstruction_flag, blockers_json "
                "FROM egress_obstruction_events"
            ).fetchone()
            database.close()

        self.assertEqual(row[:4], (7, "odom", 3.0, 1))
        self.assertIn('"object_index": 1', row[4])

    def test_flush_landmarks_handles_legacy_instance_column_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "world_map.db"
            with sqlite3.connect(db_path) as connection:
                connection.execute(
                    """
                    CREATE TABLE semantic_map (
                        landmark_id INTEGER PRIMARY KEY, class_name TEXT NOT NULL, world_frame TEXT NOT NULL,
                        X REAL NOT NULL, Y REAL NOT NULL, Z REAL NOT NULL, hit_count INTEGER NOT NULL,
                        mean_confidence REAL NOT NULL, max_confidence REAL NOT NULL,
                        first_seen_ns INTEGER NOT NULL, last_seen_ns INTEGER NOT NULL,
                        first_seen TEXT NOT NULL, last_seen TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    "INSERT INTO semantic_map VALUES (1, 'door', 'odom', 1, 2, 3, 1, 0.9, 0.9, 1, 1, 'a', 'a')"
                )
            connection.close()

            database = SemanticMapDB(db_path, 0.5)
            database.flush_landmarks()
            row = database.conn.execute(
                "SELECT class_name, instance_id, world_frame, X, Y, Z FROM semantic_map"
            ).fetchone()
            database.close()

        self.assertEqual(row, ("door", 1, "odom", 1.0, 2.0, 3.0))


if __name__ == "__main__":
    unittest.main()
