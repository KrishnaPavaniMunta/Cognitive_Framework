"""Shared ontology class resolution and viewer-friendly knowledge extraction."""

from __future__ import annotations

import json
import re
from pathlib import Path

from rdflib import BNode, Graph, Literal, Namespace, RDF, RDFS, URIRef
from rdflib.namespace import OWL

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ONTOLOGY = BASE_DIR / "ontology.rdf"

CO = Namespace("http://www.semanticweb.org/chevi/ontologies/2026/5/52-classes-ontology#")
UO = Namespace("http://www.semanticweb.org/chevi/ontologies/2026/5/untitled-ontology-6#")
BR = Namespace("http://www.semanticweb.org/chevi/ontologies/2026/5/semantic-map-bridge#")

CLASS_ALIASES = {
    "general_bin": "waste_bin",
    "yellow_bin": "waste_bin",
    "bin_tiger_stripe": "waste_bin",
    "patient": "Patient",
}

EXTENSION_CLASSES = {
    "medical_tray": ("MedicalDevice", "Steel/plastic instrument tray (semantic-map-bridge extension class)."),
    "waste_bin": ("Infrastructure", "General/yellow/tiger-stripe hospital waste bin (semantic-map-bridge extension class)."),
}


def local_name(value: object) -> str:
    text = str(value)
    return text.rsplit("#", 1)[-1].rsplit("/", 1)[-1]


def add_extension_classes(graph: Graph) -> None:
    for class_name, (superclass_name, comment) in EXTENSION_CLASSES.items():
        class_uri = BR[class_name]
        graph.add((class_uri, RDF.type, OWL.Class))
        graph.add((class_uri, RDFS.subClassOf, UO[superclass_name]))
        graph.add((class_uri, RDFS.comment, Literal(comment)))


def resolve_class(graph: Graph, class_name: str) -> URIRef:
    """Map a detector class name to an ontology or bridge-extension class URI."""
    resolved_name = CLASS_ALIASES.get(class_name, class_name)
    for namespace in (CO, UO, BR):
        candidate = namespace[resolved_name]
        if (candidate, RDF.type, OWL.Class) in graph:
            return candidate
    return UO.PhysicalObject


def load_ontology_graph(ontology_path: str | Path = DEFAULT_ONTOLOGY) -> Graph:
    graph = Graph()
    graph.parse(Path(ontology_path), format="xml")
    graph.bind("classes-ontology", CO)
    graph.bind("untitled-ontology-6", UO)
    graph.bind("smap-bridge", BR)
    add_extension_classes(graph)
    return graph


def _display_term(graph: Graph, term: object) -> str:
    if isinstance(term, URIRef):
        try:
            return graph.namespace_manager.normalizeUri(term)
        except Exception:
            return str(term)
    if isinstance(term, Literal):
        return str(term)
    if isinstance(term, BNode):
        return term.n3()
    return str(term)


def _hierarchy(graph: Graph, class_uri: URIRef) -> list[dict[str, str]]:
    hierarchy: list[dict[str, str]] = []
    visited = {class_uri}
    frontier = [class_uri]
    while frontier:
        current = frontier.pop(0)
        parents = sorted(
            (parent for parent in graph.objects(current, RDFS.subClassOf) if isinstance(parent, URIRef)),
            key=str,
        )
        for parent in parents:
            if parent in visited:
                continue
            visited.add(parent)
            hierarchy.append({"name": local_name(parent), "uri": str(parent)})
            frontier.append(parent)
    return hierarchy


def _dimensions(graph: Graph, class_uri: URIRef) -> dict:
    annotation = graph.value(class_uri, CO.physicalDimensions)
    if annotation is None:
        return _comment_dimensions(graph, class_uri)
    try:
        value = json.loads(str(annotation))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {"raw": str(annotation)}
    if not isinstance(value, dict):
        return {}

    typical = value.get("typical", {})
    ranges = value.get("range", {})
    dimensions = {}
    for axis in ("width", "depth", "height"):
        dimensions[axis] = typical.get(axis)
        axis_range = ranges.get(axis, [])
        dimensions[f"min_{axis}"] = axis_range[0] if len(axis_range) == 2 else None
        dimensions[f"max_{axis}"] = axis_range[1] if len(axis_range) == 2 else None
    return dimensions


def _comment_dimensions(graph: Graph, class_uri: URIRef) -> dict:
    comments = " ".join(str(value) for value in graph.objects(class_uri, RDFS.comment))
    match = re.search(r"(?:dimensions|size).*?([0-9]+(?:\.[0-9]+)?)\s*[x×]\s*"
                      r"([0-9]+(?:\.[0-9]+)?)\s*[x×]\s*([0-9]+(?:\.[0-9]+)?)\s*cm",
                      comments, re.IGNORECASE)
    if not match:
        return {}

    numbers = [float(number) / 100.0 for number in match.groups()]
    descriptor = comments[match.end():]
    if re.search(r"W\s*[×x]\s*H\s*[×x]\s*(?:L|D)", descriptor, re.IGNORECASE):
        width, height, depth = numbers
    else:
        width, depth, height = numbers
    return {
        "width": width,
        "depth": depth,
        "height": height,
        "min_width": None,
        "max_width": None,
        "min_depth": None,
        "max_depth": None,
        "min_height": None,
        "max_height": None,
    }


def _property_value(graph: Graph, predicate: URIRef, value: object) -> object:
    if predicate == CO.physicalDimensions:
        return None
    if isinstance(value, Literal):
        python_value = value.toPython()
        return python_value if isinstance(python_value, (str, int, float, bool)) else str(value)
    return _display_term(graph, value)


class OntologyKnowledgeBase:
    """Parse one ontology and resolve semantic-map classes into display records."""

    def __init__(self, ontology_path: str | Path = DEFAULT_ONTOLOGY) -> None:
        self.ontology_path = Path(ontology_path).resolve()
        self.graph = load_ontology_graph(self.ontology_path)
        self._cache: dict[str, dict] = {}

    def resolve(self, class_name: str) -> dict:
        if class_name in self._cache:
            return self._cache[class_name]

        mapped_name = CLASS_ALIASES.get(class_name, class_name)
        class_uri = resolve_class(self.graph, class_name)
        if class_uri == UO.PhysicalObject and mapped_name != "PhysicalObject":
            resolution = "fallback"
        elif class_name in CLASS_ALIASES:
            resolution = "aliased"
        elif str(class_uri).startswith(str(BR)):
            resolution = "extension"
        else:
            resolution = "exact"

        properties = [
            {
                "predicate": local_name(predicate),
                "value": property_value,
            }
            for predicate, value in self.graph.predicate_objects(class_uri)
            for property_value in [_property_value(self.graph, predicate, value)]
            if property_value is not None
        ]
        dimensions = _dimensions(self.graph, class_uri)
        properties.extend(
            {
                "predicate": f"{key}M",
                "value": value,
            }
            for key, value in dimensions.items()
            if value is not None
        )
        properties.sort(key=lambda item: (item["predicate"], str(item["value"])))

        record = {
            "map_class": class_name,
            "object_type": local_name(class_uri).replace("_", " ").title(),
            "resolved_name": local_name(class_uri),
            "uri": str(class_uri),
            "resolution": resolution,
            "hierarchy": _hierarchy(self.graph, class_uri),
            "dimensions": dimensions,
            "comments": sorted(str(value) for value in self.graph.objects(class_uri, RDFS.comment)),
            "properties": properties,
        }
        self._cache[class_name] = record
        return record
