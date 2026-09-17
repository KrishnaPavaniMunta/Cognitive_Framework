import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RDF_PATH = ROOT / "01_codebase" / "09_ontology" / "ontology.rdf"
OUT_PATH = Path(__file__).resolve().parents[1] / "public" / "data" / "ontology.json"

RDFS = "http://www.w3.org/2000/01/rdf-schema#"
OWL = "http://www.w3.org/2002/07/owl#"
RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"


def local_name(uri):
    return uri.rsplit("#", 1)[-1].rsplit("/", 1)[-1]


def resource(element, tag):
    child = element.find(tag)
    return child.attrib.get(f"{{{RDF}}}resource") if child is not None else None


def main():
    tree = ET.parse(RDF_PATH)
    root = tree.getroot()
    classes = []
    class_ids = set()
    for element in root.findall(f"{{{OWL}}}Class"):
        uri = element.attrib.get(f"{{{RDF}}}about", "")
        name = local_name(uri)
        if not name:
            continue
        parents = [local_name(parent.attrib.get(f"{{{RDF}}}resource", "")) for parent in element.findall(f"{{{RDFS}}}subClassOf")]
        classes.append({"id": name, "label": re.sub(r"([a-z])([A-Z])", r"\1 \2", name).replace("_", " "), "uri": uri, "parents": [parent for parent in parents if parent]})
        class_ids.add(name)

    properties = []
    for property_type, tag in (("object", "ObjectProperty"), ("datatype", "DatatypeProperty")):
        for element in root.findall(f"{{{OWL}}}{tag}"):
            uri = element.attrib.get(f"{{{RDF}}}about", "")
            name = local_name(uri)
            if not name:
                continue
            properties.append({"id": name, "label": re.sub(r"([a-z])([A-Z])", r"\1 \2", name), "type": property_type, "domain": local_name(resource(element, f"{{{RDFS}}}domain") or ""), "range": local_name(resource(element, f"{{{RDFS}}}range") or "")})

    payload = {"source": "01_codebase/09_ontology/ontology.rdf", "classCount": len(classes), "propertyCount": len(properties), "classes": classes, "properties": properties}
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Exported {len(classes)} classes and {len(properties)} properties to {OUT_PATH}")


if __name__ == "__main__":
    main()
