"""
Bridge: semantic map SQLite landmarks -> populated RDF/OWL individuals aligned to ontology.rdf.

For every landmark in a `semantic_map.db` (produced by 08_semantic_map/build_semantic_map_from_bag.py)
this script asserts:
  - an individual typed with the matching ontology class (falls back to PhysicalObject)
  - a Location individual carrying its world-frame X/Y/Z (bridge extension datatype properties)
  - a reified Statement (ontology's Statement/hasSubject/hasPredicate/hasObject pattern) recording
    the detection confidence and timestamp

Optionally (--block-radius > 0), nearby "obstacle" objects (trolleys, bins, bags, ...) close to
"protected" objects (doors, exit signs, extinguishers, switchboards, ...) are linked with the
ontology's `blocks` property and wrapped in an `Obstacle` Role + evidence Statement, seeding the
Hazard/TrippingHazard/FireHazard axioms already defined in ontology.rdf for downstream OWL reasoning.

Usage:
    python semantic_map_to_rdf.py --db "...\\semantic_map.db" --out "...\\semantic_map.ttl"
    python semantic_map_to_rdf.py                     # uses newest run under 04_outputs_runs_and_logs
"""

from __future__ import annotations

import argparse
import math
import sqlite3
from pathlib import Path

from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef
from rdflib.namespace import OWL, XSD

BASE_DIR = Path(__file__).resolve().parent
CODEBASE_DIR = BASE_DIR.parent
PROJECT_ROOT = CODEBASE_DIR.parent

DEFAULT_ONTOLOGY = BASE_DIR / "ontology.rdf"
DEFAULT_MAPS_ROOT = PROJECT_ROOT / "04_outputs_runs_and_logs" / "outputs" / "semantic_maps"

CO = Namespace("http://www.semanticweb.org/chevi/ontologies/2026/5/52-classes-ontology#")
UO = Namespace("http://www.semanticweb.org/chevi/ontologies/2026/5/untitled-ontology-6#")
BR = Namespace("http://www.semanticweb.org/chevi/ontologies/2026/5/semantic-map-bridge#")

# Detector class_name -> ontology local class name, for names that don't match 1:1.
CLASS_ALIASES = {
    "power_socket": "switchboard",
    "general_bin": "waste_bin",
    "yellow_bin": "waste_bin",
    "bin_tiger_stripe": "waste_bin",
    "patient": "Patient",
}

# Classes not present in ontology.rdf, added into the populated graph as small bridge extensions.
EXTENSION_CLASSES = {
    "medical_tray": ("MedicalDevice", "Steel/plastic instrument tray (semantic-map-bridge extension class)."),
    "waste_bin": ("Infrastructure", "General/yellow/tiger-stripe hospital waste bin (semantic-map-bridge extension class)."),
}

# Heuristic seed for the ontology's Obstacle/blocks reasoning (not a substitute for an OWL reasoner).
OBSTACLE_CLASSES = {"utility_trolley", "wheelchair", "suitcase", "bag", "backpack", "handbag", "waste_bin", "cabinet"}
PROTECTED_CLASSES = {"door", "exit_sign", "fire_extinguisher", "fire_hydrant", "switchboard"}


def _sanitize(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(text))


def add_extension_classes(graph: Graph) -> None:
    for local_name, (super_local, comment) in EXTENSION_CLASSES.items():
        cls_uri = BR[local_name]
        graph.add((cls_uri, RDF.type, OWL.Class))
        graph.add((cls_uri, RDFS.subClassOf, UO[super_local]))
        graph.add((cls_uri, RDFS.comment, Literal(comment)))


def resolve_class(graph: Graph, class_name: str) -> URIRef:
    """Map a detector class_name to an ontology (or bridge extension) class URI."""
    local_name = CLASS_ALIASES.get(class_name, class_name)
    for namespace in (CO, UO, BR):
        candidate = namespace[local_name]
        if (candidate, RDF.type, OWL.Class) in graph:
            return candidate
    print(f"[semantic_map_to_rdf] no ontology class for '{class_name}', falling back to PhysicalObject")
    return UO.PhysicalObject


def find_latest_db(maps_root: Path) -> Path:
    candidates = sorted(maps_root.glob("*/semantic_map.db"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No semantic_map.db found under {maps_root}")
    return candidates[-1]


def load_landmarks(db_path: Path) -> list[dict]:
    conn = sqlite3.connect(db_path)
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(semantic_map)")}
        has_instance_id = "instance_id" in columns
        select_cols = "landmark_id, class_name, world_frame, X, Y, Z, mean_confidence, last_seen"
        if has_instance_id:
            select_cols = "landmark_id, class_name, instance_id, world_frame, X, Y, Z, mean_confidence, last_seen"
        rows = conn.execute(f"SELECT {select_cols} FROM semantic_map ORDER BY class_name, landmark_id").fetchall()
    finally:
        conn.close()

    landmarks = []
    per_class_counter: dict[str, int] = {}
    for row in rows:
        if has_instance_id:
            landmark_id, class_name, instance_id, world_frame, x, y, z, mean_conf, last_seen = row
        else:
            landmark_id, class_name, world_frame, x, y, z, mean_conf, last_seen = row
            per_class_counter[class_name] = per_class_counter.get(class_name, 0) + 1
            instance_id = per_class_counter[class_name]
        landmarks.append(
            {
                "landmark_id": int(landmark_id),
                "class_name": class_name,
                "instance_id": int(instance_id),
                "world_frame": world_frame,
                "X": float(x),
                "Y": float(y),
                "Z": float(z),
                "mean_confidence": float(mean_conf),
                "last_seen": last_seen,
            }
        )
    return landmarks


def infer_obstacle_relations(graph: Graph, objects: dict[int, dict], radius_m: float) -> int:
    graph.add((BR.blocksPredicate, RDF.type, CO.Predicate))
    count = 0
    for lid, obstacle in objects.items():
        if obstacle["class_name"] not in OBSTACLE_CLASSES:
            continue
        for pid, protected in objects.items():
            if protected["class_name"] not in PROTECTED_CLASSES:
                continue
            dist = math.hypot(obstacle["X"] - protected["X"], obstacle["Y"] - protected["Y"])
            if dist > radius_m:
                continue

            role_uri = BR[f"role_obstacle_{lid}_{pid}"]
            graph.add((role_uri, RDF.type, UO.Obstacle))
            graph.add((obstacle["uri"], UO.playsRole, role_uri))
            graph.add((role_uri, UO.blocks, protected["uri"]))

            stmt_uri = BR[f"stmt_block_{lid}_{pid}"]
            confidence = max(0.0, 1.0 - dist / radius_m)
            graph.add((stmt_uri, RDF.type, CO.Statement))
            graph.add((stmt_uri, CO.hasSubject, obstacle["uri"]))
            graph.add((stmt_uri, CO.hasPredicate, BR.blocksPredicate))
            graph.add((stmt_uri, CO.hasObject, protected["uri"]))
            graph.add((stmt_uri, CO.hasConfidence, Literal(confidence, datatype=XSD.float)))
            graph.add((stmt_uri, CO.isEvidenceFor, role_uri))
            count += 1
    return count


def build_graph(ontology_path: Path, db_path: Path, block_radius_m: float) -> Graph:
    graph = Graph()
    graph.parse(ontology_path, format="xml")
    graph.bind("classes-ontology", CO)
    graph.bind("untitled-ontology-6", UO)
    graph.bind("smap-bridge", BR)
    add_extension_classes(graph)

    graph.add((BR.detectedAsPredicate, RDF.type, CO.Predicate))

    landmarks = load_landmarks(db_path)
    objects: dict[int, dict] = {}
    for lm in landmarks:
        obj_uri = BR[f"obj_{_sanitize(lm['class_name'])}_{lm['instance_id']}"]
        cls_uri = resolve_class(graph, lm["class_name"])
        graph.add((obj_uri, RDF.type, cls_uri))
        graph.add((obj_uri, RDFS.label, Literal(f"{lm['class_name']} #{lm['instance_id']}")))

        loc_uri = BR[f"loc_{lm['landmark_id']}"]
        graph.add((loc_uri, RDF.type, UO.Location))
        graph.add((loc_uri, BR.hasX, Literal(lm["X"], datatype=XSD.float)))
        graph.add((loc_uri, BR.hasY, Literal(lm["Y"], datatype=XSD.float)))
        graph.add((loc_uri, BR.hasZ, Literal(lm["Z"], datatype=XSD.float)))
        graph.add((loc_uri, BR.worldFrame, Literal(lm["world_frame"])))
        graph.add((obj_uri, UO.hasLocation, loc_uri))

        stmt_uri = BR[f"stmt_detect_{lm['landmark_id']}"]
        graph.add((stmt_uri, RDF.type, CO.Statement))
        graph.add((stmt_uri, CO.hasSubject, obj_uri))
        graph.add((stmt_uri, CO.hasPredicate, BR.detectedAsPredicate))
        graph.add((stmt_uri, CO.hasObject, cls_uri))
        graph.add((stmt_uri, CO.hasConfidence, Literal(lm["mean_confidence"], datatype=XSD.float)))
        if lm["last_seen"]:
            graph.add((stmt_uri, CO.detectedAt, Literal(lm["last_seen"], datatype=XSD.dateTime)))

        objects[lm["landmark_id"]] = {**lm, "uri": obj_uri}

    if block_radius_m > 0:
        n = infer_obstacle_relations(graph, objects, block_radius_m)
        print(f"[semantic_map_to_rdf] asserted {n} candidate 'blocks' relation(s) within {block_radius_m} m")

    print(f"[semantic_map_to_rdf] {len(landmarks)} landmark(s) -> {len(graph)} triples total")
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=None, help="Path to a semantic_map.db (default: newest run).")
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY, help="Path to the base ontology.rdf.")
    parser.add_argument("--out", type=Path, default=None, help="Output RDF file (default: alongside --db).")
    parser.add_argument("--format", choices=["turtle", "xml"], default="turtle", help="Output serialization.")
    parser.add_argument(
        "--block-radius", type=float, default=1.2,
        help="Horizontal distance (m) within which an obstacle-class object is asserted to block a "
             "protected-class object. Set to 0 to disable this heuristic.",
    )
    args = parser.parse_args()

    db_path = args.db or find_latest_db(DEFAULT_MAPS_ROOT)
    out_path = args.out or db_path.with_name(f"semantic_map.{'ttl' if args.format == 'turtle' else 'rdf'}")

    print(f"[semantic_map_to_rdf] ontology : {args.ontology}")
    print(f"[semantic_map_to_rdf] database : {db_path}")
    graph = build_graph(args.ontology, db_path, args.block_radius)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    graph.serialize(destination=str(out_path), format=args.format)
    print(f"[semantic_map_to_rdf] wrote {out_path}")


if __name__ == "__main__":
    main()
