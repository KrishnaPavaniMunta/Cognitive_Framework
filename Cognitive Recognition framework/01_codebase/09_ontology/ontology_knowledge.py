"""Shared ontology class resolution and viewer-friendly knowledge extraction."""

from __future__ import annotations

import json
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
        return {}
    try:
        value = json.loads(str(annotation))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {"raw": str(annotation)}
    return value if isinstance(value, dict) else {"raw": value}


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
                "predicate_uri": str(predicate),
                "value": _display_term(self.graph, value),
            }
            for predicate, value in self.graph.predicate_objects(class_uri)
        ]
        properties.sort(key=lambda item: (item["predicate_uri"], item["value"]))

        record = {
            "map_class": class_name,
            "resolved_name": local_name(class_uri),
            "uri": str(class_uri),
            "resolution": resolution,
            "hierarchy": _hierarchy(self.graph, class_uri),
            "dimensions": _dimensions(self.graph, class_uri),
            "comments": sorted(str(value) for value in self.graph.objects(class_uri, RDFS.comment)),
            "properties": properties,
        }
        self._cache[class_name] = record
        return record
