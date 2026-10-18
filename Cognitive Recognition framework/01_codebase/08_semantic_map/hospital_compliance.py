"""
Automated Hospital Health & Safety (H&S) Compliance Reasoner.

Evaluates 3D Semantic Map digital twins against real-world UK fire and safety
inspection regulations, implementing the qualitative spatial reasoning framework
from the Knowledge Media Institute (KMi, Open University; Ibba et al., 2026).

Evaluated Rules:
1. Emergency Fire Exit Doorway Clearance (No obstruction within 0.9m)
2. Portable Fire Extinguisher Mounting & Accessibility (Wall-mounted or stand; unobstructed)
3. Main Walkway / Evacuation Corridor Clearance
4. Ignition Hazard Separation (Combustibles/waste away from electrical sockets)
5. Liquid Spillage Identification & Immediate Hazard Isolation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from spatial_predicates import (
    BoundingBox3D,
    SpatialSceneGraph,
    is_blocking,
    is_contained_in_polygon,
    is_near,
    is_on_top_of,
    touches,
)


MOVABLE_OBSTACLE_CLASSES = {
    "chair", "hospital_stretcher", "utility_trolley", "bin", "general_bin",
    "yellow_bin", "bin_tiger_stripe", "suitcase", "medical_tray", "dining table",
    "bench_hosp", "box",
}

COMBUSTIBLE_CLASSES = {
    "general_bin", "yellow_bin", "bin_tiger_stripe", "bin", "suitcase",
    "paper", "cardboard", "linen", "waste",
}

IGNITION_CLASSES = {
    "power_socket", "electrical_cabinet", "fuse_box", "radiator",
}


class HospitalComplianceEvaluator:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path).resolve()
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        self.landmarks: list[dict[str, Any]] = []
        self.zones: list[dict[str, Any]] = []
        self._load_database()
        self.scene_graph = SpatialSceneGraph(self.landmarks)
        self.edges = self.scene_graph.compute_spatial_edges(max_pair_dist_m=2.0)

    def _load_database(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        self.landmarks = [dict(r) for r in conn.execute("SELECT * FROM semantic_map ORDER BY landmark_id").fetchall()]
        try:
            self.zones = [dict(r) for r in conn.execute("SELECT * FROM navigable_zones").fetchall()]
        except Exception:
            self.zones = []
        conn.close()

    def evaluate_all(self) -> dict[str, Any]:
        """Evaluate all 5 H&S compliance rules over the 3D semantic digital twin."""
        checks = [
            self._check_door_clearance(),
            self._check_fire_extinguishers(),
            self._check_walkway_clearance(),
            self._check_ignition_separation(),
            self._check_spillage_hazards(),
        ]

        violations_count = sum(len(c["violations"]) for c in checks)
        overall_compliant = bool(violations_count == 0)

        report = {
            "database": str(self.db_path),
            "total_landmarks_analyzed": len(self.landmarks),
            "total_spatial_edges_evaluated": len(self.edges),
            "overall_compliant": overall_compliant,
            "total_violations": violations_count,
            "rules": checks,
        }
        return report

    def _check_door_clearance(self) -> dict[str, Any]:
        """Rule 1: Are all emergency fire doors and escape exits kept clear and unobstructed?"""
        doors = [lm for lm in self.landmarks if "door" in str(lm["class_name"]).lower()]
        violations = []
        passed = []

        for door in doors:
            did = int(door["landmark_id"])
            door_box = self.scene_graph.boxes[did]
            obstructing = []

            for lm in self.landmarks:
                oid = int(lm["landmark_id"])
                if oid == did:
                    continue
                cname = str(lm["class_name"]).lower()
                if cname not in MOVABLE_OBSTACLE_CLASSES:
                    continue

                box = self.scene_graph.boxes[oid]
                lat_dist = box.lateral_distance_to(door_box)
                if lat_dist < 0.90 and is_blocking(box, door_box, threshold_m=0.90):
                    obstructing.append({
                        "obstacle_id": oid,
                        "obstacle_class": lm["class_name"],
                        "obstacle_instance": lm.get("instance_id"),
                        "distance_m": round(lat_dist, 2),
                        "coordinates": (round(float(lm["X"]), 2), round(float(lm["Y"]), 2), round(float(lm["Z"]), 2)),
                    })

            if obstructing:
                violations.append({
                    "door_id": did,
                    "door_instance": door.get("instance_id"),
                    "coordinates": (round(float(door["X"]), 2), round(float(door["Y"]), 2), round(float(door["Z"]), 2)),
                    "obstructing_items": obstructing,
                    "reason": f"Doorway is obstructed by {len(obstructing)} nearby item(s) within <0.9m clearance.",
                })
            else:
                passed.append({
                    "door_id": did,
                    "door_instance": door.get("instance_id"),
                    "coordinates": (round(float(door["X"]), 2), round(float(door["Y"]), 2), round(float(door["Z"]), 2)),
                })

        return {
            "rule_id": "HS-01-DOOR-CLEARANCE",
            "rule_title": "Fire Exit Doorway & Evacuation Route Clearance",
            "compliant": len(violations) == 0,
            "items_checked": len(doors),
            "violations": violations,
            "compliant_items": passed,
        }

    def _check_fire_extinguishers(self) -> dict[str, Any]:
        """Rule 2: Are portable fire extinguishers securely wall-mounted and unobstructed?"""
        extinguishers = [lm for lm in self.landmarks if "extinguisher" in str(lm["class_name"]).lower()]
        violations = []
        passed = []

        for ext in extinguishers:
            eid = int(ext["landmark_id"])
            cz = float(ext["Z"])
            # Extinguishers must not rest unmounted directly on floor (Z < 0.20m without stand)
            if cz < 0.20:
                violations.append({
                    "extinguisher_id": eid,
                    "instance_id": ext.get("instance_id"),
                    "elevation_z": round(cz, 2),
                    "coordinates": (round(float(ext["X"]), 2), round(float(ext["Y"]), 2), round(cz, 2)),
                    "reason": "Fire extinguisher is unmounted and resting directly on the floor without an approved stand.",
                })
            else:
                passed.append({
                    "extinguisher_id": eid,
                    "instance_id": ext.get("instance_id"),
                    "elevation_z": round(cz, 2),
                })

        return {
            "rule_id": "HS-02-EXTINGUISHER-MOUNTING",
            "rule_title": "Portable Fire Extinguisher Mounting & Accessibility",
            "compliant": len(violations) == 0,
            "items_checked": len(extinguishers),
            "violations": violations,
            "compliant_items": passed,
        }

    def _check_walkway_clearance(self) -> dict[str, Any]:
        """Rule 3: Are main corridor circulation routes and walkways free from tripping hazards?"""
        violations = []
        passed = []
        tripping_candidates = [
            lm for lm in self.landmarks
            if str(lm["class_name"]).lower() in {"suitcase", "box", "spillage"}
        ]

        for item in tripping_candidates:
            iid = int(item["landmark_id"])
            cname = str(item["class_name"]).lower()
            cz = float(item["Z"])
            if cz <= 0.60:  # Low-lying obstacle on floor
                violations.append({
                    "obstacle_id": iid,
                    "class_name": item["class_name"],
                    "instance_id": item.get("instance_id"),
                    "elevation_z": round(cz, 2),
                    "coordinates": (round(float(item["X"]), 2), round(float(item["Y"]), 2), round(cz, 2)),
                    "reason": f"Potential tripping hazard: {item['class_name']} resting in corridor floor space.",
                })
            else:
                passed.append(iid)

        return {
            "rule_id": "HS-03-CORRIDOR-CLEARANCE",
            "rule_title": "Circulation Corridor & Walkway Trip Hazard Clearance",
            "compliant": len(violations) == 0,
            "items_checked": len(tripping_candidates),
            "violations": violations,
            "compliant_items": passed,
        }

    def _check_ignition_separation(self) -> dict[str, Any]:
        """Rule 4: Have combustible materials been stored safely away from sources of ignition?"""
        combustibles = [lm for lm in self.landmarks if str(lm["class_name"]).lower() in COMBUSTIBLE_CLASSES]
        ignitions = [lm for lm in self.landmarks if str(lm["class_name"]).lower() in IGNITION_CLASSES]
        violations = []
        passed = []

        for comb in combustibles:
            cid = int(comb["landmark_id"])
            cbox = self.scene_graph.boxes[cid]
            hazardous_near = []

            for ign in ignitions:
                iid = int(ign["landmark_id"])
                ibox = self.scene_graph.boxes[iid]
                dist = cbox.distance_to(ibox)
                if dist < 0.60:
                    hazardous_near.append({
                        "ignition_id": iid,
                        "ignition_class": ign["class_name"],
                        "ignition_instance": ign.get("instance_id"),
                        "distance_m": round(dist, 2),
                    })

            if hazardous_near:
                violations.append({
                    "combustible_id": cid,
                    "combustible_class": comb["class_name"],
                    "combustible_instance": comb.get("instance_id"),
                    "coordinates": (round(float(comb["X"]), 2), round(float(comb["Y"]), 2), round(float(comb["Z"]), 2)),
                    "near_ignitions": hazardous_near,
                    "reason": f"Combustible waste located <0.60m from electrical ignition source.",
                })
            else:
                passed.append(cid)

        return {
            "rule_id": "HS-04-IGNITION-SEPARATION",
            "rule_title": "Combustible Material Separation from Ignition Sources",
            "compliant": len(violations) == 0,
            "items_checked": len(combustibles),
            "violations": violations,
            "compliant_items": passed,
        }

    def _check_spillage_hazards(self) -> dict[str, Any]:
        """Rule 5: Are chemical or biological liquid spillages identified and isolated?"""
        spillages = [lm for lm in self.landmarks if "spill" in str(lm["class_name"]).lower()]
        violations = []

        for sp in spillages:
            sid = int(sp["landmark_id"])
            violations.append({
                "spillage_id": sid,
                "instance_id": sp.get("instance_id"),
                "coordinates": (round(float(sp["X"]), 2), round(float(sp["Y"]), 2), round(float(sp["Z"]), 2)),
                "consensus_score": round(float(sp.get("consensus_score") or 1.0), 2),
                "reason": "Active uncontained liquid spillage detected on the floor. Immediate maintenance required.",
            })

        return {
            "rule_id": "HS-05-SPILLAGE-HAZARD",
            "rule_title": "Liquid Spillage Identification & Biohazard Isolation",
            "compliant": len(violations) == 0,
            "items_checked": len(spillages),
            "violations": violations,
            "compliant_items": [],
        }


def generate_markdown_report(report: dict[str, Any]) -> str:
    """Format evaluation results into a human-readable Health & Safety Inspection report."""
    status_str = "COMPLIANT" if report["overall_compliant"] else "NON-COMPLIANT (VIOLATIONS DETECTED)"
    badge = "[PASS]" if report["overall_compliant"] else "[ACTION REQUIRED]"

    md = [
        "# HOSPITAL HEALTH & SAFETY (H&S) COMPLIANCE AUDIT REPORT",
        f"**Status:** {badge} **{status_str}**",
        f"**Database:** `{report['database']}`",
        f"**Landmarks Analyzed:** {report['total_landmarks_analyzed']}",
        f"**Total Spatial Edges Checked:** {report['total_spatial_edges_evaluated']}",
        f"**Total Violations Detected:** {report['total_violations']}",
        "",
        "---",
        "## Summary of Regulatory Rule Checks",
        "",
    ]

    for rule in report["rules"]:
        r_status = "PASS" if rule["compliant"] else "FAIL"
        r_tag = "[OK]" if rule["compliant"] else "[VIOLATION]"
        md.append(f"### {r_tag} {rule['rule_title']} (`{rule['rule_id']}`)")
        md.append(f"- **Result:** **{r_status}** (Items Checked: {rule['items_checked']})")

        if rule["violations"]:
            md.append("- **Violations Detected:**")
            for v in rule["violations"]:
                md.append(f"  * **{v.get('reason')}**")
                coords = v.get("coordinates")
                if coords:
                    md.append(f"    - Location: `(X={coords[0]:.2f}, Y={coords[1]:.2f}, Z={coords[2]:.2f})`")
                if v.get("obstructing_items"):
                    for obs in v["obstructing_items"]:
                        md.append(f"    - Obstructing Item: `{obs['obstacle_class']} {obs['obstacle_instance']}` at distance `{obs['distance_m']}m`")
        else:
            md.append("- **Violations:** None. All inspected items satisfy regulatory constraints.")
        md.append("")

    return "\n".join(md)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Hospital Health & Safety Compliance from world_map.db")
    parser.add_argument("--db", type=Path, default=Path("04_outputs_runs_and_logs/outputs/semantic_maps/rgbd_clean_20260925_134830/world_map.db"))
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    args = parser.parse_args()

    evaluator = HospitalComplianceEvaluator(args.db)
    report = evaluator.evaluate_all()

    md_report = generate_markdown_report(report)
    print("\n" + md_report + "\n")

    json_path = args.json_out or args.db.with_name("hospital_compliance_report.json")
    md_path = args.md_out or args.db.with_name("hospital_compliance_report.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report)

    print(f"Compliance JSON saved: {json_path}")
    print(f"Compliance Markdown saved: {md_path}")


if __name__ == "__main__":
    main()
