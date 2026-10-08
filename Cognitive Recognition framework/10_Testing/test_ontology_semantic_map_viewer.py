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
for module_dir in (SEMANTIC_MAP_DIR, ONTOLOGY_DIR):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

from ontology_knowledge import OntologyKnowledgeBase
from rerun_logger import (
    AnnotationManager,
    get_default_annotation_manager,
    landmark_metadata,
    resolve_box_half_size,
    stable_landmark_path,
)
from semantic_map_html import build_figure, write_html
from view_semantic_map import (
    CURRENT_HTML_NAME,
    LEGACY_HTML_NAME,
    attach_ontology_knowledge,
    find_latest_db,
    load_landmarks,
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


if __name__ == "__main__":
    unittest.main()
