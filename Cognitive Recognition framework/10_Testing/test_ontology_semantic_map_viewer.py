from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_MAP_DIR = PROJECT_ROOT / "01_codebase" / "08_semantic_map"
ONTOLOGY_DIR = PROJECT_ROOT / "01_codebase" / "09_ontology"
OBJECT_DETECTION_DIR = PROJECT_ROOT / "01_codebase" / "07_object_detection"
for module_dir in (SEMANTIC_MAP_DIR, ONTOLOGY_DIR, OBJECT_DETECTION_DIR):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

from build_semantic_map_from_bag import sample_box_foreground_depth_m
from ontology_knowledge import OntologyKnowledgeBase
from rgbd_3d_filter import apply_depth_size_filter
from rosbag_rgbd_sim_capture import CameraIntrinsics
from rerun_logger import (
    AnnotationManager,
    RerunSceneLogger,
    get_default_annotation_manager,
    landmark_metadata,
    resolve_box_half_size,
    stable_landmark_path,
)
from semantic_map_html import _landmark_payload, build_figure, write_html
from view_semantic_map import (
    CURRENT_HTML_NAME,
    LEGACY_HTML_NAME,
    attach_ontology_knowledge,
    find_latest_db,
    load_landmarks,
    load_navigable_zones,
    refresh_semantic_map_html,
)


class OntologyKnowledgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.knowledge_base = OntologyKnowledgeBase(ONTOLOGY_DIR / "ontology.rdf")

    def test_resolution_modes_and_knowledge(self) -> None:
        door = self.knowledge_base.resolve("door")
        self.assertEqual(door["resolution"], "exact")
        self.assertTrue(door["dimensions"])
        self.assertIn("Infrastructure", [item["name"] for item in door["hierarchy"]])

        power_socket = self.knowledge_base.resolve("power_socket")
        self.assertEqual(power_socket["resolution"], "exact")
        self.assertTrue(power_socket["dimensions"])

        self.assertEqual(self.knowledge_base.resolve("general_bin")["resolution"], "aliased")
        self.assertEqual(self.knowledge_base.resolve("medical_tray")["resolution"], "extension")
        self.assertEqual(self.knowledge_base.resolve("floor")["resolution"], "extension")
        self.assertEqual(self.knowledge_base.resolve("navigable_floor")["resolution"], "aliased")
        fallback = self.knowledge_base.resolve("not_a_real_class")
        self.assertEqual(fallback["resolution"], "fallback")
        self.assertEqual(fallback["resolved_name"], "PhysicalObject")

        trolley = self.knowledge_base.resolve("utility_trolley")
        self.assertEqual(trolley["dimensions"]["width"], 0.5)
        self.assertEqual(trolley["dimensions"]["depth"], 0.8)
        self.assertEqual(trolley["dimensions"]["height"], 0.95)

        expected = {
            "fork": (0.03, 0.18, 0.01, 0.02, 0.05, 0.15, 0.22),
            "spoon": (0.04, 0.18, 0.01, 0.03, 0.06, 0.15, 0.22),
            "scissors": (0.06, 0.20, 0.01, 0.05, 0.09, 0.15, 0.25),
            "surgical_scissor": (0.05, 0.16, 0.01, 0.04, 0.08, 0.12, 0.23),
            "nasal_cannula": (0.15, 0.15, 0.05, 0.05, 0.25, 0.05, 0.25),
        }
        for class_name, values in expected.items():
            dimensions = self.knowledge_base.resolve(class_name)["dimensions"]
            self.assertEqual(
                tuple(dimensions[key] for key in ("width", "height", "depth", "min_width", "max_width", "min_height", "max_height")),
                values,
            )


class SemanticMapViewerTests(unittest.TestCase):
    def test_database_discovery_supports_current_and_legacy_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            current = root / "bag" / "world_map.db"
            legacy = root / "semanticmap_old" / "semantic_map.db"
            current.parent.mkdir()
            legacy.parent.mkdir()
            current.touch()
            legacy.touch()
            os.utime(current, (1, 1))
            os.utime(legacy, (2, 2))
            self.assertEqual(find_latest_db(root), legacy)

    def test_evidence_ontology_html_and_rerun_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            database = root / "world_map.db"
            connection = sqlite3.connect(database)
            try:
                connection.execute(
                    """
                    CREATE TABLE semantic_map (
                        landmark_id INTEGER, class_name TEXT, instance_id INTEGER,
                        world_frame TEXT, X REAL, Y REAL, Z REAL, hit_count INTEGER,
                        mean_confidence REAL, max_confidence REAL, first_seen_ns INTEGER,
                        last_seen_ns INTEGER, first_seen TEXT, last_seen TEXT
                    )
                    """
                )
                connection.execute(
                    "INSERT INTO semantic_map VALUES (7, 'door', 2, 'odom', 1, 2, 0.4, 5, 0.8, 0.9, 10, 20, '<first>', 'last')"
                )
                landmarks = load_landmarks(connection, 1, None)
            finally:
                connection.close()

            attach_ontology_knowledge(landmarks, OntologyKnowledgeBase(ONTOLOGY_DIR / "ontology.rdf"))
            landmark = landmarks[0]
            self.assertEqual(landmark["ontology"]["resolution"], "exact")
            self.assertEqual(stable_landmark_path(landmark), "world/landmarks/door_7")
            metadata = landmark_metadata(landmark)
            self.assertEqual(metadata["hit_count"], 5)
            self.assertIn("Infrastructure", metadata["ontology_hierarchy"])
            self.assertTrue(json.loads(metadata["ontology_dimensions_json"]))

            figure = build_figure(landmarks, [], "test")
            payload = json.loads(figure.data[0].customdata[0][0])
            self.assertEqual(payload["map"]["Landmark ID"], 7)
            self.assertEqual(payload["ontology"]["uri"], landmark["ontology"]["uri"])
            self.assertEqual(payload["map"]["Instance ID"], 2)
            self.assertEqual(payload["map"]["Dimension source"], "ontology_typical")
            self.assertNotIn("predicate_uri", payload["ontology"]["properties"][0])

            html_path = write_html(figure, root / "viewer.html", "<map source>")
            html = html_path.read_text(encoding="utf-8")
            self.assertIn("plotly_click", html)
            self.assertIn("Physical Dimensions", html)
            self.assertIn("&lt;map source&gt;", html)
            self.assertIn("const esc =", html)

    def test_current_html_refresh_uses_one_fixed_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            database = root / "world_map.db"
            connection = sqlite3.connect(database)
            try:
                connection.execute(
                    """
                    CREATE TABLE semantic_map (
                        landmark_id INTEGER, class_name TEXT, instance_id INTEGER,
                        world_frame TEXT, X REAL, Y REAL, Z REAL, hit_count INTEGER,
                        mean_confidence REAL, max_confidence REAL, first_seen_ns INTEGER,
                        last_seen_ns INTEGER, first_seen TEXT, last_seen TEXT
                    )
                    """
                )
                connection.execute(
                    "INSERT INTO semantic_map VALUES (1, 'door', 1, 'odom', 1, 2, 0.4, 5, 0.8, 0.9, 10, 20, 'first', 'last')"
                )
                connection.commit()
            finally:
                connection.close()

            legacy_path = root / LEGACY_HTML_NAME
            legacy_path.write_text("old viewer", encoding="utf-8")
            output_path = refresh_semantic_map_html(database, ONTOLOGY_DIR / "ontology.rdf")
            self.assertEqual(output_path.name, CURRENT_HTML_NAME)
            self.assertTrue(output_path.exists())
            self.assertFalse(legacy_path.exists())

    def test_rerun_path_sanitizes_class_name(self) -> None:
        landmark = {"landmark_id": 3, "class_name": "Power Socket/Test"}
        self.assertEqual(stable_landmark_path(landmark), "world/landmarks/power_socket_test_3")

    def test_resolve_box_half_size_uses_ontology_dimensions(self) -> None:
        kb = OntologyKnowledgeBase(ONTOLOGY_DIR / "ontology.rdf")
        trolley_lm = {"class_name": "utility_trolley", "ontology": kb.resolve("utility_trolley")}
        half_size = resolve_box_half_size(trolley_lm)
        # utility_trolley: depth=0.8, width=0.5, height=0.95 -> [depth/2, width/2, height/2] = [0.4, 0.25, 0.475]
        self.assertAlmostEqual(half_size[0][0], 0.4, places=2)
        self.assertAlmostEqual(half_size[0][1], 0.25, places=2)
        self.assertAlmostEqual(half_size[0][2], 0.475, places=2)

        door_lm = {"class_name": "door", "ontology": kb.resolve("door")}
        door_half_size = resolve_box_half_size(door_lm)
        # door: depth=0.05, width=1.0, height=2.1 -> [0.025, 0.5, 1.05]
        self.assertAlmostEqual(door_half_size[0][0], 0.025, places=2)
        self.assertAlmostEqual(door_half_size[0][1], 0.5, places=2)
        self.assertAlmostEqual(door_half_size[0][2], 1.05, places=2)

        fallback_lm = {"class_name": "unknown_object"}
        fallback_half_size = resolve_box_half_size(fallback_lm)
        self.assertEqual(fallback_half_size, [[0.25, 0.25, 0.25]])

    def test_annotation_manager_and_semantic_context(self) -> None:
        manager = get_default_annotation_manager(["door", "utility_trolley"])
        door_id = manager.get_id("door")
        trolley_id = manager.get_id("utility_trolley")
        self.assertGreater(door_id, 0)
        self.assertGreater(trolley_id, 0)
        self.assertNotEqual(door_id, trolley_id)
        self.assertEqual(manager.get_name(door_id), "door")
        context = manager.build_context()
        self.assertIsNotNone(context)

    def test_resolve_box_half_size_prioritizes_camera_measured_dimensions(self) -> None:
        kb = OntologyKnowledgeBase(ONTOLOGY_DIR / "ontology.rdf")
        lm_with_real = {
            "landmark_id": 10,
            "class_name": "utility_trolley",
            "ontology": kb.resolve("utility_trolley"),  # ontology dims: 0.8 x 0.5 x 0.95
            "measured_width": 0.62,
            "measured_depth": 0.74,
            "measured_height": 1.10,
        }
        half_size = resolve_box_half_size(lm_with_real)
        # Must use camera measurements: [0.74/2, 0.62/2, 1.10/2] = [0.37, 0.31, 0.55]
        self.assertAlmostEqual(half_size[0][0], 0.37, places=2)
        self.assertAlmostEqual(half_size[0][1], 0.31, places=2)
        self.assertAlmostEqual(half_size[0][2], 0.55, places=2)

        meta = landmark_metadata(lm_with_real)
        self.assertEqual(meta["dimension_source"], "camera_measured")
        self.assertEqual(meta["measured_width_m"], 0.62)
        self.assertEqual(meta["measured_depth_m"], 0.74)
        self.assertEqual(meta["measured_height_m"], 1.10)

    def test_load_landmarks_reads_measured_dimensions_and_observations_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "world_map.db"
            conn = sqlite3.connect(db_path)
            conn.executescript("""
                CREATE TABLE semantic_map (
                    landmark_id INTEGER PRIMARY KEY,
                    class_name TEXT NOT NULL,
                    instance_id INTEGER NOT NULL,
                    world_frame TEXT NOT NULL,
                    X REAL, Y REAL, Z REAL,
                    hit_count INTEGER NOT NULL,
                    mean_confidence REAL NOT NULL,
                    measured_width REAL,
                    measured_depth REAL,
                    measured_height REAL
                );
                CREATE TABLE observations (
                    obs_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    landmark_id INTEGER,
                    real_width REAL,
                    real_depth REAL,
                    real_height REAL
                );
                -- Landmark 1: has direct measured dimensions
                INSERT INTO semantic_map VALUES (1, 'door', 1, 'map', 1.0, 2.0, 0.0, 5, 0.9, 0.95, 0.08, 2.05);
                -- Landmark 2: has NULL measured dimensions in semantic_map, but observations exist
                INSERT INTO semantic_map VALUES (2, 'utility_trolley', 1, 'map', 3.0, 4.0, 0.0, 3, 0.85, NULL, NULL, NULL);
                INSERT INTO observations (landmark_id, real_width, real_depth, real_height) VALUES (2, 0.52, 0.82, 0.96);
                INSERT INTO observations (landmark_id, real_width, real_depth, real_height) VALUES (2, 0.48, 0.78, 0.94);
            """)
            conn.commit()

            lms = load_landmarks(conn, min_hits=1, classes=None)
            conn.close()

            lm1 = next(lm for lm in lms if lm["landmark_id"] == 1)
            self.assertAlmostEqual(lm1["measured_width"], 0.95)
            self.assertAlmostEqual(lm1["measured_depth"], 0.08)
            self.assertAlmostEqual(lm1["measured_height"], 2.05)

            lm2 = next(lm for lm in lms if lm["landmark_id"] == 2)
            self.assertAlmostEqual(lm2["measured_width"], 0.50, places=2)
            self.assertAlmostEqual(lm2["measured_depth"], 0.80, places=2)
            self.assertAlmostEqual(lm2["measured_height"], 0.95, places=2)

    def test_multiview_consensus_fusion(self) -> None:
        from build_semantic_map_from_bag import SemanticMapDB

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "world_map.db"
            db = SemanticMapDB(db_path, merge_radius_m=0.50)

            # Observation 1: Transient false positive: 'security_camera' at (1.0, 2.0, 0.5)
            obs1 = {
                "frame_index": 1, "timestamp_ns": 1000,
                "source_class_name": "security_camera", "class_name": "security_camera",
                "confidence": 0.55, "u": 100, "v": 100, "depth_m": 2.0,
                "cam_X": 0.1, "cam_Y": 0.2, "cam_Z": 2.0,
                "world_X": 1.0, "world_Y": 2.0, "world_Z": 0.5,
                "world_frame": "odom", "extrinsics_source": "tf", "extrinsics_status": "ok",
                "real_width": 0.20, "real_depth": 0.20, "real_height": 0.15,
            }
            lid1, inst1 = db.add_observation(obs1)
            self.assertEqual(len(db.landmarks), 1)
            self.assertEqual(db.landmarks[lid1]["class_name"], "security_camera")

            # Observation 2: Stable ground truth: 'doorstop' at (1.02, 1.98, 0.51)
            obs2 = {
                "frame_index": 2, "timestamp_ns": 2000,
                "source_class_name": "doorstop", "class_name": "doorstop",
                "confidence": 0.85, "u": 102, "v": 98, "depth_m": 1.98,
                "cam_X": 0.11, "cam_Y": 0.19, "cam_Z": 1.98,
                "world_X": 1.02, "world_Y": 1.98, "world_Z": 0.51,
                "world_frame": "odom", "extrinsics_source": "tf", "extrinsics_status": "ok",
                "real_width": 0.22, "real_depth": 0.21, "real_height": 0.16,
            }
            lid2, inst2 = db.add_observation(obs2)
            self.assertEqual(lid1, lid2, "Must fuse into the same 3D spatial node despite class mismatch")
            self.assertEqual(len(db.landmarks), 1)
            # Winning class flips to 'doorstop' because 0.85 > 0.55
            self.assertEqual(db.landmarks[lid1]["class_name"], "doorstop")

            # Observation 3: Another view confirms 'doorstop' (conf=0.90)
            obs3 = {
                "frame_index": 3, "timestamp_ns": 3000,
                "source_class_name": "doorstop", "class_name": "doorstop",
                "confidence": 0.90, "u": 101, "v": 100, "depth_m": 1.99,
                "cam_X": 0.10, "cam_Y": 0.20, "cam_Z": 1.99,
                "world_X": 1.01, "world_Y": 2.00, "world_Z": 0.49,
                "world_frame": "odom", "extrinsics_source": "tf", "extrinsics_status": "ok",
                "real_width": 0.21, "real_depth": 0.20, "real_height": 0.15,
            }
            lid3, inst3 = db.add_observation(obs3)
            self.assertEqual(lid1, lid3)
            self.assertEqual(db.landmarks[lid1]["hit_count"], 3)
            self.assertEqual(db.landmarks[lid1]["class_name"], "doorstop")
            self.assertAlmostEqual(db.landmarks[lid1]["consensus_score"], (0.85 + 0.90) / (0.55 + 0.85 + 0.90), places=2)

            # Flush to disk and reload
            db.flush_landmarks()
            db.close()

            db_reloaded = SemanticMapDB(db_path, merge_radius_m=0.50)
            self.assertEqual(len(db_reloaded.landmarks), 1)
            reloaded_lm = db_reloaded.landmarks[lid1]
            self.assertEqual(reloaded_lm["class_name"], "doorstop")
            self.assertEqual(reloaded_lm["hit_count"], 3)
            self.assertIn("security_camera", reloaded_lm["class_votes"])
            self.assertIn("doorstop", reloaded_lm["class_votes"])
            self.assertAlmostEqual(reloaded_lm["consensus_score"], (0.85 + 0.90) / (0.55 + 0.85 + 0.90), places=2)
            db_reloaded.close()

    def test_floor_semantic_labeling_and_navigable_zone(self) -> None:
        # 1. AnnotationManager reserves ID 1 for 'Floor' and names it 'Floor' without any instance ID
        mgr = AnnotationManager()
        self.assertEqual(mgr.get_id("Floor"), 1)
        self.assertEqual(mgr.get_id("floor"), 1)
        self.assertEqual(mgr.get_name(1), "Floor")
        self.assertEqual(mgr.get_name(0), "Background")
        # Other classes get IDs >= 2
        fire_id = mgr.get_id("fire_extinguisher")
        self.assertGreaterEqual(fire_id, 2)

        # 2. Stable landmark path and metadata for floor: must be 'Floor' with no instance_id
        floor_landmark = {
            "class_name": "floor",
            "landmark_id": 99,
            "instance_id": 1,
            "world_frame": "odom",
            "X": 0.0, "Y": 0.0, "Z": 0.0,
            "hit_count": 20,
        }
        self.assertEqual(stable_landmark_path(floor_landmark), "world/zones/floor")
        meta = landmark_metadata(floor_landmark)
        self.assertEqual(meta["map_class"], "Floor")
        self.assertNotIn("instance_id", meta, "Floor must not be given an instance ID")
        self.assertIsNone(meta.get("instance_id"))

        # 3. HTML landmark payload: omits 'Instance ID' for Floor
        payload = _landmark_payload(floor_landmark)
        self.assertEqual(payload["map"]["Class"], "Floor")
        self.assertNotIn("Instance ID", payload["map"])
        self.assertNotIn("Landmark ID", payload["map"])

        # 4. Database schema and navigable_zones persistence
        with tempfile.TemporaryDirectory() as temp_dir:
            from build_semantic_map_from_bag import SemanticMapDB
            db_path = Path(temp_dir) / "world_map.db"
            db = SemanticMapDB(db_path, merge_radius_m=0.50)

            zone_info = {
                "zone_name": "Floor",
                "min_x": -5.0, "max_x": 5.0,
                "min_y": -2.0, "max_y": 2.0,
                "elevation_z": 0.0,
                "area_m2": 40.0,
                "point_count": 1500,
                "polygon": [[-5.0, -2.0], [5.0, -2.0], [5.0, 2.0], [-5.0, 2.0]],
            }
            db.save_navigable_zone(zone_info, "odom")
            db.close()

            conn = sqlite3.connect(db_path)
            zones = load_navigable_zones(conn)
            conn.close()

            self.assertEqual(len(zones), 1)
            self.assertEqual(zones[0]["zone_name"], "Floor")
            self.assertAlmostEqual(zones[0]["area_m2"], 40.0)
            self.assertEqual(len(zones[0]["polygon"]), 4)

            # 5. build_figure renders navigable zone trace
            fig = build_figure([floor_landmark], [], "Test Map", navigable_zones=zones)
            trace_names = [t.name for t in fig.data]
            self.assertIn("Floor (boundary)", trace_names)
            self.assertIn("Floor", trace_names)
            # Ensure no trace is called 'floor 1' or 'Floor 1'
            self.assertNotIn("floor 1", trace_names)
            self.assertNotIn("Floor 1", trace_names)

        # 6. RerunSceneLogger.log_navigable_zone computes valid geometry
        with tempfile.TemporaryDirectory() as temp_dir:
            import numpy as np
            rrd_path = Path(temp_dir) / "test.rrd"
            logger = RerunSceneLogger(rrd_path, application_id="test_app", spawn_viewer=False)
            pts = np.array([
                [-2.0, -1.0], [-1.5, -1.0], [-1.0, -1.0], [0.0, -1.0], [1.0, -1.0], [2.0, -1.0],
                [-2.0, 1.0], [-1.0, 1.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0],
                [-2.0, 0.0], [-1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
            ], dtype=np.float32)
            logger.floor_points_xy.append(pts)
            logger.floor_z_sum = 0.0
            logger.floor_z_count = len(pts)
            zone_info = logger.log_navigable_zone()
            self.assertIsNotNone(zone_info)
            self.assertEqual(zone_info["zone_name"], "Floor")
            self.assertAlmostEqual(zone_info["min_x"], -2.0)
            self.assertAlmostEqual(zone_info["max_x"], 2.0)
            self.assertGreater(zone_info["area_m2"], 0)
            logger.finish()


class PerceptionGatingTests(unittest.TestCase):
    def test_hollow_furniture_foreground_depth_recovery(self) -> None:
        """Hollow furniture (e.g. office chair) with 0 depth at midpoint must recover foreground seat depth."""
        import numpy as np
        depth = np.zeros((480, 640), dtype=np.uint16)
        # Bbox from frame 2101
        bbox = [160, 220, 250, 350]
        u_mid, v_mid = (160 + 250) / 2.0, (220 + 350) / 2.0
        # Ensure midpoint 5x5 patch is 0
        depth[int(v_mid) - 2: int(v_mid) + 3, int(u_mid) - 2: int(u_mid) + 3] = 0

        # Simulate solid seat cushion / frame
        depth[230:260, 180:230] = 4400  # 4.4 metres

        z_m, adj_u, adj_v = sample_box_foreground_depth_m(depth, bbox, u_mid, v_mid)
        self.assertIsNotNone(z_m)
        self.assertAlmostEqual(z_m, 4.4, places=1)
        # Adjusted center should shift towards the seat cushion
        self.assertGreater(adj_u, 175)
        self.assertLess(adj_u, 235)

    def test_solid_object_midpoint_depth_preserved(self) -> None:
        """Objects with solid center (like beds or doors) should keep standard center depth sampling."""
        import numpy as np
        depth = np.zeros((480, 640), dtype=np.uint16)
        bbox = [100, 100, 300, 300]
        u_mid, v_mid = 200.0, 200.0
        depth[195:206, 195:206] = 2100  # 2.1 metres

        z_m, adj_u, adj_v = sample_box_foreground_depth_m(depth, bbox, u_mid, v_mid)
        self.assertIsNotNone(z_m)
        self.assertAlmostEqual(z_m, 2.1, places=2)
        self.assertEqual(adj_u, u_mid)
        self.assertEqual(adj_v, v_mid)

    def test_reception_counter_and_iv_stand_dimension_gating(self) -> None:
        """Thin partition walls or tall ceilings must not pass as reception_counter; wide fixtures not as iv_stand."""
        import numpy as np
        depth = np.zeros((480, 640), dtype=np.uint16)
        intr = CameraIntrinsics(
            fx=600.0, fy=600.0, cx=320.0, cy=240.0,
            width=640, height=480, distortion_model="none", d=[]
        )
        # 1. Thin flat partition wall (H=2.5m, D=0.10m) labeled as reception_counter
        depth[50:400, 150:500] = 2000  # 2.0m flat wall
        preds_counter = [(150, 50, 500, 400, 0.45, "reception_counter")]
        filtered = apply_depth_size_filter(
            preds_counter, depth, intr,
            dimensions_config_path=ONTOLOGY_DIR / "ontology.rdf",
            class_name_alias_candidates={},
        )
        # Flat tall wall should be rejected as reception_counter
        self.assertEqual(len(filtered), 0)

        # 2. Wide fixture (W=1.6m, H=0.8m) labeled as iv_stand
        preds_iv = [(150, 100, 500, 300, 0.40, "iv_stand")]
        filtered_iv = apply_depth_size_filter(
            preds_iv, depth, intr,
            dimensions_config_path=ONTOLOGY_DIR / "ontology.rdf",
            class_name_alias_candidates={},
        )
        # Horizontal wide fixture must be rejected as iv_stand
        self.assertEqual(len(filtered_iv), 0)


class FastSAMAnd3DSurfaceExtractionTests(unittest.TestCase):
    def test_overlay_mask_on_frame(self) -> None:
        from fast_sam_segmenter import overlay_mask_on_frame
        import numpy as np

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:50, 20:50] = 1

        color = (0, 255, 0)
        out = overlay_mask_on_frame(img.copy(), mask, color, alpha=0.50, draw_contour=True)

        # Pixels inside mask should be blended with green (approx 0.5 * 255 = 127)
        self.assertGreater(out[30, 30, 1], 100)
        # Pixels outside mask should remain 0
        self.assertEqual(int(out[10, 10, 1]), 0)

    def test_sample_box_foreground_depth_with_mask(self) -> None:
        import numpy as np
        depth = np.zeros((100, 100), dtype=np.uint16)
        # Background wall at 5.0m
        depth[:, :] = 5000
        # Foreground object at 1.8m in upper-left of bbox
        depth[20:40, 20:40] = 1800

        # Mask only covers the foreground 1.8m region
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:40, 20:40] = 1

        bbox = [10, 10, 80, 80]
        u_mid, v_mid = 45.0, 45.0  # Center is at 5.0m (wall)

        # Without mask, center patch would sample the 5.0m wall
        z_no_mask, _, _ = sample_box_foreground_depth_m(depth, bbox, u_mid, v_mid)
        self.assertAlmostEqual(z_no_mask, 5.0, places=1)

        # With neural mask, center patch is outside mask so it strictly isolates the 1.8m object
        z_mask, adj_u, adj_v = sample_box_foreground_depth_m(depth, bbox, u_mid, v_mid, mask=mask)
        self.assertIsNotNone(z_mask)
        self.assertAlmostEqual(z_mask, 1.8, places=1)
        # Centroid adjusted towards (30, 30)
        self.assertLess(adj_u, 40.0)
        self.assertLess(adj_v, 40.0)

    def test_extract_world_surface_points(self) -> None:
        from rgbd_3d_filter import extract_world_surface_points
        import numpy as np

        intr = CameraIntrinsics(
            fx=500.0, fy=500.0, cx=50.0, cy=50.0,
            width=100, height=100, distortion_model="none", d=[]
        )
        depth = np.full((100, 100), 2000, dtype=np.uint16)  # 2.0m flat plane
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 1

        # Camera pose: shifted 1.0m in X world
        pose = np.eye(4)
        pose[0, 3] = 1.0

        pts_world = extract_world_surface_points(
            depth, 30, 30, 70, 70, intr, pose, mask=mask, max_points=100
        )
        self.assertIsNotNone(pts_world)
        self.assertGreater(len(pts_world), 20)
        # Z depth in camera is 2.0m
        self.assertAlmostEqual(float(np.median(pts_world[:, 2])), 2.0, places=1)
        # World X has 1.0m translation added
        self.assertAlmostEqual(float(np.median(pts_world[:, 0])), 1.0, places=1)

    def test_fast_sam_segmenter_inference(self) -> None:
        weights_path = PROJECT_ROOT / "FastSAM-s.pt"
        if not weights_path.exists():
            weights_path = Path("FastSAM-s.pt")
        if not weights_path.exists():
            self.skipTest("FastSAM-s.pt weights not found")

        from fast_sam_segmenter import FastSAMSegmenter
        import cv2
        import numpy as np

        segmenter = FastSAMSegmenter(weights_path=weights_path)
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.circle(img, (160, 120), 40, (255, 255, 255), -1)

        boxes = [[110.0, 70.0, 210.0, 170.0]]
        masks = segmenter.segment_boxes(img, boxes)

        self.assertEqual(len(masks), 1)
        raw_mask, eroded_mask = masks[0]
        self.assertEqual(raw_mask.shape, (240, 320))
        self.assertEqual(eroded_mask.shape, (240, 320))
        self.assertGreater(int(np.sum(raw_mask)), 100)
        self.assertLessEqual(int(np.sum(eroded_mask)), int(np.sum(raw_mask)))


if __name__ == "__main__":
    unittest.main()


