"""
Deterministic Spatial Scene Graph Query Engine & Robot View Pipeline.

Provides a lightweight, zero-daemon, pure Python/NumPy spatial query interface
over the 3D Qualitative Spatial Scene Graph stored in world_map.db and
spatial_scene_graph.json.

Answers spatial queries such as:
1. "Given a robot view / location, identify objects and relations"
2. "Find all objects on the floor, along the wall, or near the ceiling"
3. "Find all movable obstacles or trip hazards in the circulation walkway"
4. "Get all topological, directional, proximity, and structural relations for asset X"
5. "Query fire exit keep-clear zone status and active safety hazard infractions"

Usage Examples:
    python query_spatial_map.py --find on_floor
    python query_spatial_map.py --find in_walkway
    python query_spatial_map.py --find along_wall
    python query_spatial_map.py --relations "dining table 1"
    python query_spatial_map.py --view -5.5,-0.1 --radius 3.0
    python query_spatial_map.py --hazards
    python query_spatial_map.py --fire-exits
    python query_spatial_map.py --view -5.5,-0.1 --json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sqlite3
import sys
from typing import Any

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
DEFAULT_OUT_ROOT = PROJECT_ROOT / "04_outputs_runs_and_logs" / "outputs" / "semantic_maps"


def find_latest_db(out_root: Path) -> Path:
    candidates = list(out_root.glob("*/world_map.db"))
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No world_map.db found under {out_root}")
    return candidates[0]


class SemanticMapQuery:
    """
    Lightweight, declarative spatial query engine for 3D digital twins.
    Directly queries SQLite world_map.db and associated JSON scene graphs.
    """

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path).resolve()
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

        # Load optional spatial_scene_graph.json if available
        self.scene_graph_json: dict[str, Any] = {}
        sg_path = self.db_path.with_name("spatial_scene_graph.json")
        if sg_path.exists():
            try:
                with open(sg_path, "r", encoding="utf-8") as f:
                    self.scene_graph_json = json.load(f)
            except Exception:
                self.scene_graph_json = {}

        # Load optional compliance report
        self.compliance_report: dict[str, Any] = {}
        comp_path = self.db_path.with_name("hospital_compliance_report.json")
        if comp_path.exists():
            try:
                with open(comp_path, "r", encoding="utf-8") as f:
                    self.compliance_report = json.load(f)
            except Exception:
                self.compliance_report = {}

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> SemanticMapQuery:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ── Core Query 1: Get Landmark by ID or Name ─────────────────────────────
    def get_landmark(self, identifier: str | int) -> dict[str, Any] | None:
        """Find a landmark by integer landmark_id or name (e.g. 'dining table 1')."""
        cur = self.conn.cursor()
        if isinstance(identifier, int) or (isinstance(identifier, str) and identifier.strip().isdigit()):
            lid = int(identifier)
            row = cur.execute("SELECT * FROM semantic_map WHERE landmark_id = ?", (lid,)).fetchone()
            if row:
                return dict(row)

        target_str = str(identifier).strip().lower()
        rows = cur.execute("SELECT * FROM semantic_map").fetchall()
        for r in rows:
            d = dict(r)
            full_name = f"{d['class_name']} {d.get('instance_id', '')}".strip().lower()
            if full_name == target_str or target_str == str(d['class_name']).lower():
                return d
        for r in rows:
            d = dict(r)
            if target_str in str(d['class_name']).lower():
                return d
        return None

    # ── Core Query 2: Get All Relations for an Asset ─────────────────────────
    def get_relations(self, identifier: str | int) -> dict[str, Any]:
        """
        Query all spatial relationships (outgoing and incoming) for a given landmark.
        Includes structural relations (on_floor, along_wall, near_ceiling, in_walkway).
        """
        lm = self.get_landmark(identifier)
        if not lm:
            return {"error": f"Landmark not found matching '{identifier}'"}

        lid = int(lm["landmark_id"])
        lm_name = f"{lm['class_name']} {lm.get('instance_id', '')}".strip()

        cur = self.conn.cursor()
        outgoing_rows = cur.execute("""
            SELECT edge_id, target_id, target_class, predicate, category, distance_m, lateral_distance_m, is_violation, evidence
            FROM spatial_edges WHERE source_id = ? ORDER BY category, predicate
        """, (lid,)).fetchall()

        incoming_rows = cur.execute("""
            SELECT edge_id, source_id, source_class, predicate, category, distance_m, lateral_distance_m, is_violation, evidence
            FROM spatial_edges WHERE target_id = ? ORDER BY category, predicate
        """, (lid,)).fetchall()

        return {
            "landmark": {
                "landmark_id": lid,
                "name": lm_name,
                "class_name": lm["class_name"],
                "coordinates": (round(float(lm["X"]), 3), round(float(lm["Y"]), 3), round(float(lm["Z"]), 3)),
                "dimensions": {
                    "width": round(float(lm.get("measured_width") or 0.45), 3),
                    "depth": round(float(lm.get("measured_depth") or 0.45), 3),
                    "height": round(float(lm.get("measured_height") or 0.60), 3),
                },
            },
            "outgoing_relations": [dict(r) for r in outgoing_rows],
            "incoming_relations": [dict(r) for r in incoming_rows],
            "total_relations": len(outgoing_rows) + len(incoming_rows),
        }

    # ── Core Query 3: Find Objects by Spatial Predicate ──────────────────────
    def find_objects(
        self,
        predicate: str | None = None,
        category: str | None = None,
        class_name: str | None = None,
        is_violation: bool | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query objects satisfying a spatial relation (e.g. predicate='on_floor', 'in_walkway', 'along_wall').
        """
        query = """
            SELECT e.edge_id, e.source_id, e.source_class, e.target_id, e.target_class,
                   e.predicate, e.category, e.distance_m, e.is_violation, e.evidence,
                   s.X, s.Y, s.Z, s.instance_id
            FROM spatial_edges e
            JOIN semantic_map s ON e.source_id = s.landmark_id
            WHERE 1=1
        """
        params: list[Any] = []
        if predicate:
            query += " AND e.predicate = ?"
            params.append(predicate.strip().lower())
        if category:
            query += " AND e.category = ?"
            params.append(category.strip().lower())
        if class_name:
            query += " AND LOWER(e.source_class) LIKE ?"
            params.append(f"%{class_name.strip().lower()}%")
        if is_violation is not None:
            query += " AND e.is_violation = ?"
            params.append(1 if is_violation else 0)

        query += " ORDER BY e.category, e.source_class, e.source_id"
        cur = self.conn.cursor()
        rows = cur.execute(query, params).fetchall()

        results = []
        for r in rows:
            results.append({
                "landmark_id": r["source_id"],
                "name": f"{r['source_class']} {r['instance_id'] or ''}".strip(),
                "class_name": r["source_class"],
                "predicate": r["predicate"],
                "category": r["category"],
                "target": r["target_class"],
                "target_id": r["target_id"],
                "distance_m": round(float(r["distance_m"]), 2) if r["distance_m"] is not None else None,
                "coordinates": (round(float(r["X"]), 3), round(float(r["Y"]), 3), round(float(r["Z"]), 3)),
                "is_violation": bool(r["is_violation"]),
                "evidence": r["evidence"],
            })
        return results

    # ── Core Query 4: Given a Robot View, Query Objects & Relations ──────────
    def query_robot_view(
        self,
        robot_x: float,
        robot_y: float,
        robot_z: float = 0.35,
        view_radius_m: float = 3.0,
        heading_deg: float | None = None,
        fov_deg: float | None = None,
    ) -> dict[str, Any]:
        """
        Evaluates the local spatial scene from the robot's vantage point.
        Identifies all visible objects within range, their metric distance and bearing,
        and extracts all active relationships between those objects and structural room anchors.
        """
        cur = self.conn.cursor()
        rows = cur.execute("SELECT * FROM semantic_map").fetchall()

        visible_objects: list[dict[str, Any]] = []
        visible_ids: set[int] = set()

        for r in rows:
            d = dict(r)
            lx, ly, lz = float(d["X"]), float(d["Y"]), float(d["Z"])
            dx = lx - robot_x
            dy = ly - robot_y
            dz = lz - robot_z
            dist_2d = math.hypot(dx, dy)
            dist_3d = math.sqrt(dx * dx + dy * dy + dz * dz)

            if dist_2d > view_radius_m:
                continue

            bearing_deg = math.degrees(math.atan2(dy, dx))
            # Check FOV if heading and FOV specified
            if heading_deg is not None and fov_deg is not None:
                angle_diff = (bearing_deg - heading_deg + 180) % 360 - 180
                if abs(angle_diff) > (fov_deg / 2.0):
                    continue

            lid = int(d["landmark_id"])
            visible_ids.add(lid)
            visible_objects.append({
                "landmark_id": lid,
                "name": f"{d['class_name']} {d.get('instance_id', '')}".strip(),
                "class_name": d["class_name"],
                "coordinates": (round(lx, 3), round(ly, 3), round(lz, 3)),
                "distance_2d_m": round(dist_2d, 2),
                "distance_3d_m": round(dist_3d, 2),
                "relative_bearing_deg": round(bearing_deg, 1),
            })

        visible_objects.sort(key=lambda o: o["distance_2d_m"])

        # Fetch all spatial relations among visible objects or with room structures
        relations_in_view: list[dict[str, Any]] = []
        hazards_in_view: list[dict[str, Any]] = []

        if visible_ids:
            placeholders = ",".join("?" for _ in visible_ids)
            edge_query = f"""
                SELECT edge_id, source_id, source_class, target_id, target_class,
                       predicate, category, distance_m, is_violation, evidence
                FROM spatial_edges
                WHERE source_id IN ({placeholders})
                  AND (target_id IN ({placeholders}) OR target_id <= 0)
                ORDER BY category, predicate
            """
            edge_rows = cur.execute(edge_query, list(visible_ids) + list(visible_ids)).fetchall()

            for er in edge_rows:
                ed = dict(er)
                relations_in_view.append(ed)
                if ed["is_violation"]:
                    hazards_in_view.append(ed)

        return {
            "robot_vantage_point": {
                "x": round(robot_x, 3),
                "y": round(robot_y, 3),
                "z": round(robot_z, 3),
                "view_radius_m": view_radius_m,
                "heading_deg": heading_deg,
                "fov_deg": fov_deg,
            },
            "total_objects_in_view": len(visible_objects),
            "visible_objects": visible_objects,
            "total_relations_in_view": len(relations_in_view),
            "spatial_relations": relations_in_view,
            "hazards_in_view": hazards_in_view,
        }

    # ── Core Query 5: Get All Safety Hazards ─────────────────────────────────
    def get_hazards(self) -> list[dict[str, Any]]:
        """Retrieve all active compliance infractions and safety hazards."""
        if self.compliance_report and "violations" in self.compliance_report:
            return self.compliance_report["violations"]

        cur = self.conn.cursor()
        rows = cur.execute("""
            SELECT e.source_id, e.source_class, e.target_id, e.target_class,
                   e.predicate, e.category, e.distance_m, e.evidence,
                   s.X, s.Y, s.Z, s.instance_id
            FROM spatial_edges e
            JOIN semantic_map s ON e.source_id = s.landmark_id
            WHERE e.is_violation = 1
        """).fetchall()

        hazards = []
        for r in rows:
            hazards.append({
                "rule_id": f"HAZARD-{r['predicate'].upper()}",
                "object": f"{r['source_class']} {r['instance_id'] or ''}".strip(),
                "predicate": r["predicate"],
                "target": r["target_class"],
                "evidence": r["evidence"],
                "location": [round(float(r["X"]), 2), round(float(r["Y"]), 2), round(float(r["Z"]), 2)],
            })
        return hazards

    # ── Core Query 6: Fire Exit Clearance Status ─────────────────────────────
    def get_fire_exit_status(self, door_id: int | None = None) -> list[dict[str, Any]]:
        """Query fire exit door keep-clear 3D egress zones."""
        if self.scene_graph_json and "door_keep_clear_zones" in self.scene_graph_json:
            zones = self.scene_graph_json["door_keep_clear_zones"]
            if door_id is not None:
                zones = [z for z in zones if int(z["door_id"]) == int(door_id)]
            return zones

        # Fallback to DB check
        cur = self.conn.cursor()
        blocking_edges = cur.execute("""
            SELECT source_id, source_class, target_id, target_class, distance_m, evidence
            FROM spatial_edges WHERE predicate = 'is_blocking'
        """).fetchall()
        return [dict(r) for r in blocking_edges]


# ── CLI Interface ─────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Query 3D Semantic Map & Qualitative Spatial Relations (QSR) Pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--db", default="", help="Path to world_map.db (default: latest run)")
    p.add_argument("--relations", default="", help="Get all spatial relations for asset name or ID (e.g. 'dining table 1')")
    p.add_argument("--find", default="", help="Find objects satisfying predicate (e.g. 'on_floor', 'along_wall', 'in_walkway')")
    p.add_argument("--view", default="", help="Evaluate robot view at coordinates 'X,Y' (e.g. '-5.5,-0.1')")
    p.add_argument("--radius", type=float, default=3.0, help="Robot view radius in meters (default: 3.0m)")
    p.add_argument("--hazards", action="store_true", help="List all safety hazards and active infractions")
    p.add_argument("--fire-exits", action="store_true", help="List fire exit door keep-clear egress status")
    p.add_argument("--json", action="store_true", help="Output machine-readable JSON format")

    args = p.parse_args()

    db_path = Path(args.db).resolve() if args.db else find_latest_db(DEFAULT_OUT_ROOT)

    with SemanticMapQuery(db_path) as query_engine:
        if args.relations:
            res = query_engine.get_relations(args.relations)
            if args.json:
                print(json.dumps(res, indent=2))
            else:
                if "error" in res:
                    print(f"[ERROR] {res['error']}")
                    sys.exit(1)
                lm = res["landmark"]
                print(f"\n=======================================================")
                print(f"  SPATIAL RELATIONS: {lm['name']} (ID: {lm['landmark_id']})")
                print(f"  Position: {lm['coordinates']} m | Extents: {lm['dimensions']}")
                print(f"=======================================================")
                out_rels = res["outgoing_relations"]
                in_rels = res["incoming_relations"]
                print(f"\nOutgoing Relations ({len(out_rels)}):")
                for r in out_rels:
                    viol = " [!] VIOLATION" if r["is_violation"] else ""
                    dist_str = f" ({r['distance_m']:.2f}m)" if r["distance_m"] is not None else ""
                    print(f"  - [{r['category'].upper():<11}] {r['predicate']:<15} -> {r['target_class']:<25}{dist_str}{viol}")
                    if r.get("evidence"):
                        print(f"      Evidence: {r['evidence']}")
                if in_rels:
                    print(f"\nIncoming Relations ({len(in_rels)}):")
                    for r in in_rels:
                        viol = " [!] VIOLATION" if r["is_violation"] else ""
                        print(f"  - [{r['category'].upper():<11}] {r['source_class']} #{r['source_id']} -> is_{r['predicate']}_to -> this asset{viol}")
                print()

        elif args.find:
            res = query_engine.find_objects(predicate=args.find)
            if args.json:
                print(json.dumps(res, indent=2))
            else:
                print(f"\nFound {len(res)} objects matching predicate '{args.find}':")
                print(f"{'ID':<6} {'Name':<24} {'Predicate':<14} {'Target Anchor / Asset':<28} {'Coords (X,Y,Z)':<20}")
                print("-" * 96)
                for item in res:
                    coords = f"({item['coordinates'][0]:.2f}, {item['coordinates'][1]:.2f}, {item['coordinates'][2]:.2f})"
                    print(f"{item['landmark_id']:<6} {item['name']:<24} {item['predicate']:<14} {item['target']:<28} {coords:<20}")
                print()

        elif args.view:
            try:
                coords = [float(c.strip()) for c in args.view.split(",")]
                rx, ry = coords[0], coords[1]
                rz = coords[2] if len(coords) > 2 else 0.35
            except Exception:
                print("[ERROR] --view requires 'X,Y' or 'X,Y,Z' float coordinates (e.g. --view -5.5,-0.1)")
                sys.exit(1)

            res = query_engine.query_robot_view(rx, ry, rz, view_radius_m=args.radius)
            if args.json:
                print(json.dumps(res, indent=2))
            else:
                vp = res["robot_vantage_point"]
                print(f"\n=======================================================")
                print(f"  ROBOT VIEW QUERY at ({vp['x']:.2f}, {vp['y']:.2f}, {vp['z']:.2f}) [Radius: {vp['view_radius_m']}m]")
                print(f"=======================================================")
                print(f"Visible Landmarks: {res['total_objects_in_view']}")
                for obj in res["visible_objects"]:
                    print(f"  * #{obj['landmark_id']:<3} {obj['name']:<22} Dist: {obj['distance_2d_m']:.2f}m | Bearing: {obj['relative_bearing_deg']:+5.1f} deg | Coords: {obj['coordinates']}")

                print(f"\nSpatial Relations In View: {res['total_relations_in_view']}")
                for r in res["spatial_relations"]:
                    viol = " [!] HAZARD" if r["is_violation"] else ""
                    print(f"  - {r['source_class']} #{r['source_id']} -> [{r['predicate']}] -> {r['target_class']} #{r['target_id']}{viol}")

                if res["hazards_in_view"]:
                    print(f"\n[!] WARNING: {len(res['hazards_in_view'])} Active Safety Hazard(s) In Robot View:")
                    for h in res["hazards_in_view"]:
                        print(f"  ! {h['source_class']} #{h['source_id']}: {h['evidence']}")
                print()

        elif args.hazards:
            res = query_engine.get_hazards()
            if args.json:
                print(json.dumps(res, indent=2))
            else:
                print(f"\nTotal Active Safety Hazards / Infractions: {len(res)}")
                for i, h in enumerate(res, start=1):
                    ev = h.get('evidence') or h.get('reason', '')
                    loc = h.get('coordinates') or h.get('location', '')
                    print(f"[{i}] Rule: {h.get('rule_id', 'HAZARD')} | Offending Asset: {h.get('object', h.get('offending_object', ''))}")
                    if ev:
                        print(f"    Evidence: {ev}")
                    if loc:
                        print(f"    Location: {loc}")
                print()

        elif args.fire_exits:
            res = query_engine.get_fire_exit_status()
            if args.json:
                print(json.dumps(res, indent=2))
            else:
                print(f"\nEmergency Fire Exit Egress Status ({len(res)} Doors):")
                for dz in res:
                    st = dz.get("status", "UNKNOWN")
                    badge = "[OK] CLEAR" if st == "CLEAR" else "[FAIL] OBSTRUCTED"
                    print(f"  * Door #{dz.get('door_id')}: {badge}")
                    print(f"      Center: {dz.get('center')} | Required Width: {dz.get('required_clear_width_m', 0.85)}m | Keep-Clear Radius: {dz.get('radius_m', 1.0)}m")
                print()

        else:
            p.print_help()


if __name__ == "__main__":
    main()
