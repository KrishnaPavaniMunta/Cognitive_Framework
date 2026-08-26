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
from rerun_logger import landmark_metadata, stable_landmark_path
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


if __name__ == "__main__":
    unittest.main()
