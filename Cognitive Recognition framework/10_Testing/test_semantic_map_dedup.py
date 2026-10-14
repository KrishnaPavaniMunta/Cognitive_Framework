"""
Unit tests for SemanticMapDB dynamic tolerance envelope ("beach ball"),
XY-decoupled merging, object replacement, and post-run agglomerative deduplication.
"""

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np

from build_semantic_map_from_bag import SemanticMapDB


class TestSemanticMapDedupAndTolerance(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp_dir.name) / "test_map.db"
        self.db = SemanticMapDB(self.db_path, merge_radius_m=0.5, dynamic_classes={"person", "wheelchair"})
        self.db.start_run("test_run_1", Path("test_bag"))

    def tearDown(self):
        self.db.close()
        self.tmp_dir.cleanup()

    def test_door_height_variation_merges_into_single_landmark(self):
        """
        Robot tilts down, sees door handle at Z=0.51m.
        Robot tilts up, sees top of door frame at Z=1.15m (delta Z = 0.64m > legacy 0.5m threshold).
        Horizontal delta XY is 0.12m.
        Should merge into Door 1, NOT create Door 2!
        """
        obs1 = {
            "frame_index": 1,
            "timestamp_ns": 1000,
            "source_class_name": "door",
            "class_name": "door",
            "confidence": 0.85,
            "u": 320, "v": 240, "depth_m": 2.0,
            "cam_X": 0.1, "cam_Y": 0.0, "cam_Z": 2.0,
            "world_X": 3.0, "world_Y": 1.5, "world_Z": 0.51,
            "world_frame": "odom",
            "extrinsics_source": "tf_chain",
            "extrinsics_status": "ok",
            "real_width": 0.90, "real_depth": 0.20, "real_height": 2.05,
            "surface_points": np.array([[3.0, 1.5, 0.51], [3.05, 1.52, 0.55]], dtype=np.float32),
        }
        lid1, inst1 = self.db.add_observation(obs1)
        self.assertEqual(inst1, 1)

        # Observation 2: same door doorway, delta Z = 0.64m, delta XY = 0.12m
        obs2 = {
            "frame_index": 2,
            "timestamp_ns": 2000,
            "source_class_name": "door",
            "class_name": "door",
            "confidence": 0.90,
            "u": 320, "v": 200, "depth_m": 2.1,
            "cam_X": 0.15, "cam_Y": 0.1, "cam_Z": 2.1,
            "world_X": 3.12, "world_Y": 1.50, "world_Z": 1.15,
            "world_frame": "odom",
            "extrinsics_source": "tf_chain",
            "extrinsics_status": "ok",
            "real_width": 0.92, "real_depth": 0.20, "real_height": 2.00,
            "surface_points": np.array([[3.12, 1.50, 1.15]], dtype=np.float32),
        }
        lid2, inst2 = self.db.add_observation(obs2)
        # Must merge into the same landmark and maintain instance 1!
        self.assertEqual(lid1, lid2)
        self.assertEqual(inst2, 1)
        self.assertEqual(len(self.db.landmarks), 1)
        lm = self.db.landmarks[lid1]
        self.assertEqual(lm["hit_count"], 2)
        self.assertAlmostEqual(lm["X"], (3.0 + 3.12) / 2.0, places=3)
        self.assertAlmostEqual(lm["Z"], (0.51 + 1.15) / 2.0, places=3)

    def test_dynamic_tolerance_scales_with_object_dimensions(self):
        """
        Large object (e.g. conference table width=2.4m):
        Dynamic radius R_xy = max(0.5, 2.4 * 0.70) = 1.68m.
        Detections 1.2m apart should merge because of the large physical extent.
        """
        obs1 = {
            "frame_index": 10, "timestamp_ns": 10000,
            "source_class_name": "table", "class_name": "table",
            "confidence": 0.80, "u": 300, "v": 200, "depth_m": 3.0,
            "cam_X": 0.0, "cam_Y": 0.0, "cam_Z": 3.0,
            "world_X": 5.0, "world_Y": 2.0, "world_Z": 0.75,
            "world_frame": "odom", "extrinsics_source": "tf_chain", "extrinsics_status": "ok",
            "real_width": 2.40, "real_depth": 1.20, "real_height": 0.75,
        }
        lid1, _ = self.db.add_observation(obs1)

        obs2 = {
            "frame_index": 11, "timestamp_ns": 11000,
            "source_class_name": "table", "class_name": "table",
            "confidence": 0.85, "u": 320, "v": 200, "depth_m": 3.1,
            "cam_X": 0.1, "cam_Y": 0.0, "cam_Z": 3.1,
            "world_X": 6.1, "world_Y": 2.2, "world_Z": 0.76, # dist_xy = sqrt(1.1^2 + 0.2^2) = 1.118m < 1.68m
            "world_frame": "odom", "extrinsics_source": "tf_chain", "extrinsics_status": "ok",
            "real_width": 2.40, "real_depth": 1.20, "real_height": 0.75,
        }
        lid2, _ = self.db.add_observation(obs2)
        self.assertEqual(lid1, lid2, "Large table detections within dynamic extent must merge")

    def test_object_replacement_transition(self):
        """
        Run 1: Table placed at (4.0, 3.0). Hit count = 2.
        Run 2: Bench replaces table at exact same location.
        When bench gets strong evidence (e.g. 3 hits), consensus score flips to bench!
        """
        # Table detections
        for i in range(2):
            self.db.add_observation({
                "frame_index": i, "timestamp_ns": 1000 * (i + 1),
                "source_class_name": "table", "class_name": "table",
                "confidence": 0.75, "u": 320, "v": 240, "depth_m": 2.0,
                "cam_X": 0.0, "cam_Y": 0.0, "cam_Z": 2.0,
                "world_X": 4.0, "world_Y": 3.0, "world_Z": 0.5,
                "world_frame": "odom", "extrinsics_source": "tf_chain", "extrinsics_status": "ok",
                "real_width": 1.0, "real_depth": 0.8, "real_height": 0.75,
            })
        lm_id = list(self.db.landmarks.keys())[0]
        self.assertEqual(self.db.landmarks[lm_id]["class_name"], "table")

        # Now bench replaces table: 3 strong observations
        for i in range(3):
            self.db.add_observation({
                "frame_index": 10 + i, "timestamp_ns": 50000 + 1000 * i,
                "source_class_name": "bench", "class_name": "bench",
                "confidence": 0.95, "u": 320, "v": 240, "depth_m": 2.0,
                "cam_X": 0.0, "cam_Y": 0.0, "cam_Z": 2.0,
                "world_X": 4.05, "world_Y": 3.02, "world_Z": 0.48,
                "world_frame": "odom", "extrinsics_source": "tf_chain", "extrinsics_status": "ok",
                "real_width": 1.2, "real_depth": 0.6, "real_height": 0.50,
            })

        # Landmark should be updated to bench because bench vote sum (3 * 0.95 = 2.85) > table (2 * 0.75 = 1.5)
        self.assertEqual(self.db.landmarks[lm_id]["class_name"], "bench")
        self.assertGreater(self.db.landmarks[lm_id]["consensus_score"], 0.5)

    def test_post_run_agglomerative_consolidation(self):
        """
        Simulate legacy map with duplicate Door 1 and Door 2 in same doorway.
        consolidate_landmarks() must merge them into a single Door 1.
        """
        self.db.landmarks[1] = {
            "class_name": "door", "instance_id": 1, "world_frame": "odom",
            "X": 2.0, "Y": 1.0, "Z": 0.50,
            "hit_count": 5, "conf_sum": 4.0, "max_confidence": 0.85,
            "first_seen_ns": 1000, "last_seen_ns": 2000,
            "measured_width": 0.90, "measured_depth": 0.15, "measured_height": 2.0,
            "measured_count": 5, "surface_points": np.array([[2.0, 1.0, 0.5]]),
            "class_votes": {"door": 4.0}, "consensus_score": 1.0,
        }
        self.db.landmarks[2] = {
            "class_name": "door", "instance_id": 2, "world_frame": "odom",
            "X": 2.1, "Y": 1.05, "Z": 1.20,
            "hit_count": 4, "conf_sum": 3.5, "max_confidence": 0.90,
            "first_seen_ns": 3000, "last_seen_ns": 4000,
            "measured_width": 0.90, "measured_depth": 0.15, "measured_height": 2.0,
            "measured_count": 4, "surface_points": np.array([[2.1, 1.05, 1.2]]),
            "class_votes": {"door": 3.5}, "consensus_score": 1.0,
        }
        # Another door further away (5m away)
        self.db.landmarks[3] = {
            "class_name": "door", "instance_id": 3, "world_frame": "odom",
            "X": 7.0, "Y": 1.0, "Z": 0.80,
            "hit_count": 10, "conf_sum": 9.0, "max_confidence": 0.95,
            "first_seen_ns": 5000, "last_seen_ns": 6000,
            "measured_width": 0.90, "measured_depth": 0.15, "measured_height": 2.0,
            "measured_count": 10, "surface_points": None,
            "class_votes": {"door": 9.0}, "consensus_score": 1.0,
        }

        merged = self.db.consolidate_landmarks()
        self.assertEqual(merged, 1)
        self.assertEqual(len(self.db.landmarks), 2)
        # Instances reindexed consecutively
        instances = [lm["instance_id"] for lm in self.db.landmarks.values()]
        self.assertEqual(instances, [1, 2])

    def test_surface_points_persistence_in_db(self):
        """Surface point clouds must survive flush and reload from SQLite."""
        pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        self.db.landmarks[1] = {
            "class_name": "chair", "instance_id": 1, "world_frame": "odom",
            "X": 1.0, "Y": 2.0, "Z": 0.5,
            "hit_count": 1, "conf_sum": 0.9, "max_confidence": 0.9,
            "first_seen_ns": 1000, "last_seen_ns": 1000,
            "measured_width": 0.5, "measured_depth": 0.5, "measured_height": 0.8,
            "measured_count": 1, "surface_points": pts,
            "class_votes": {"chair": 0.9}, "consensus_score": 1.0,
        }
        self.db.flush_landmarks()
        self.db.close()

        # Reload DB
        reloaded_db = SemanticMapDB(self.db_path, merge_radius_m=0.5)
        self.assertIn(1, reloaded_db.landmarks)
        reloaded_pts = reloaded_db.landmarks[1].get("surface_points")
        self.assertIsNotNone(reloaded_pts)
        np.testing.assert_array_almost_equal(pts, reloaded_pts)
        reloaded_db.close()


if __name__ == "__main__":
    unittest.main()

