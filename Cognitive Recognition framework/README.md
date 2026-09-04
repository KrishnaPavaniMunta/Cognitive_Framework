# Cognitive Recognition Framework

This repository contains the conference-submission code and resources for a hospital cognitive-recognition framework that combines object detection, RGB-D semantic mapping, ontology grounding, RDF export, and ontology-aware visualization.

## Start Here: Ontology

The ontology documentation is here:

[01_codebase/09_ontology/README.md](01_codebase/09_ontology/README.md)

That README explains how the ontology is connected across the project and how to use it with Protege, Python resolution, semantic-map RDF export, the HTML viewer, and Rerun.

Key ontology files:

| File | Purpose |
|---|---|
| [01_codebase/09_ontology/ontology.rdf](01_codebase/09_ontology/ontology.rdf) | Authoritative RDF/XML OWL ontology. |
| [01_codebase/09_ontology/ONTOLOGY_INFORMATION.md](01_codebase/09_ontology/ONTOLOGY_INFORMATION.md) | Detailed class inventory, dimensions, aliases, and integration notes. |
| [01_codebase/09_ontology/ontology_knowledge.py](01_codebase/09_ontology/ontology_knowledge.py) | Runtime resolver that maps detector labels to ontology classes. |
| [01_codebase/09_ontology/semantic_map_to_rdf.py](01_codebase/09_ontology/semantic_map_to_rdf.py) | Converts semantic-map SQLite landmarks into populated RDF. |
| [01_codebase/09_ontology/Run_commands.txt](01_codebase/09_ontology/Run_commands.txt) | Reproducible commands for ontology checks, export, and visualization. |

## Repository Layout

| Folder | Contents |
|---|---|
| `01_codebase` | Training, inference, data preparation, RGB-D mapping, object detection, semantic map, and ontology code. |
| `02_datasets` | Dataset inputs and archives used by the detection pipeline. |
| `03_models_and_weights` | Trained detector weights and model assets. |
| `04_outputs_runs_and_logs` | Generated outputs, semantic-map databases, RDF exports, Rerun recordings, and validation logs. |
| `05_documents_and_presentations` | Paper-related documents and presentation material. |
| `10_Testing` | Focused validation scripts and tests. |

## Ontology Pipeline Summary

```text
YOLO / RGB-D detections
  -> ontology class resolution
  -> physical-size filtering
  -> persistent semantic map
  -> RDF/OWL individuals and statements
  -> HTML and Rerun ontology-aware viewers
```

For detailed usage, open [01_codebase/09_ontology/README.md](01_codebase/09_ontology/README.md).