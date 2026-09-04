# Ontology Module

This folder contains the hospital cognitive-recognition ontology and the bridge code that connects object detection, RGB-D semantic mapping, RDF export, and ontology-aware visualization.

For conference reviewers, the main files are:

| File | Purpose |
|---|---|
| [ontology.rdf](ontology.rdf) | Authoritative RDF/XML OWL ontology. Open this in Protege or any OWL/RDF tool. |
| [ontology_knowledge.py](ontology_knowledge.py) | Runtime ontology resolver used by the semantic-map viewer and downstream tools. |
| [semantic_map_to_rdf.py](semantic_map_to_rdf.py) | Converts persistent semantic-map SQLite landmarks into populated RDF individuals. |
| [ONTOLOGY_INFORMATION.md](ONTOLOGY_INFORMATION.md) | Detailed class inventory, aliases, dimensions, and implementation notes. |
| [Run_commands.txt](Run_commands.txt) | Copy-ready PowerShell commands for validation, export, and visualization. |

## How The Ontology Is Connected

The framework uses the ontology as the semantic layer between perception outputs and higher-level hospital-scene reasoning.

```text
YOLO / RGB-D detections
        |
        v
Detector labels and 3D observations
        |
        v
ontology_knowledge.py
  - loads ontology.rdf
  - resolves detector labels to OWL classes
  - adds semantic-map bridge classes
  - exposes hierarchy, comments, and physical dimensions
        |
        +-----------------------------+
        |                             |
        v                             v
Physical-size filtering          Semantic map viewer
01_codebase/07_object_detection  01_codebase/08_semantic_map
        |                             |
        v                             v
Validated detections             Ontology-enriched landmarks
        |                             |
        +-------------+---------------+
                      |
                      v
Persistent semantic map SQLite database
04_outputs_runs_and_logs/outputs/semantic_maps/<run>/world_map.db
                      |
                      v
semantic_map_to_rdf.py
                      |
                      v
Populated RDF graph with objects, locations, statements, confidence, and obstacle relations
```

The base ontology is [ontology.rdf](ontology.rdf). It declares hospital object classes, people, roles, locations, contexts, hazard concepts, semantic statement concepts, object properties, and physical-dimension annotations. Runtime code loads this ontology and uses it to attach semantic meaning to detections and mapped landmarks.

## Ontology Structure

The ontology contains 93 named OWL classes in the base RDF file. At runtime, two bridge classes are added for detector and map labels that are not present in the base namespace, giving 95 runtime classes.

Main class groups:

```text
PhysicalObject
+-- Furniture
+-- Infrastructure
+-- MedicalDevice
+-- PersonalBelonging
+-- SafetyEquipment

Person
+-- Patient
+-- Staff
|   +-- healthcare_worker
+-- Visitor

Context
+-- EnvironmentContext
+-- SpatialContext
+-- TemporalContext

Location
+-- Environment
+-- IndoorSpace

Event
+-- Alert
+-- HazardEvent

Role
+-- Hazard
|   +-- ContaminationHazard
|   +-- FireHazard
|   +-- TrippingHazard
+-- MonitoredEntity
+-- Obstacle
+-- SafetyZone
```

Runtime bridge classes:

| Bridge class | Parent class | Why it exists |
|---|---|---|
| `medical_tray` | `MedicalDevice` | Used by semantic-map/object labels but absent from the base RDF class list. |
| `waste_bin` | `Infrastructure` | Shared ontology target for `general_bin`, `yellow_bin`, and `bin_tiger_stripe`. |

## Namespaces

| Prefix | Namespace | Role |
|---|---|---|
| `classes-ontology` / `CO` | `http://www.semanticweb.org/chevi/ontologies/2026/5/52-classes-ontology#` | Main ontology metadata, statement vocabulary, and physical-dimension annotation. |
| `untitled-ontology-6` / `UO` | `http://www.semanticweb.org/chevi/ontologies/2026/5/untitled-ontology-6#` | Most hospital-domain classes and relationships. |
| `smap-bridge` / `BR` | `http://www.semanticweb.org/chevi/ontologies/2026/5/semantic-map-bridge#` | Runtime extension namespace for semantic-map individuals and bridge classes. |

## Detector Label Resolution

Object detectors and semantic maps often use labels that do not exactly match OWL class names. The resolver in [ontology_knowledge.py](ontology_knowledge.py) normalizes these labels before ontology lookup.

| Detector/map label | Ontology target | Resolution type |
|---|---|---|
| `general_bin` | `waste_bin` | Alias plus runtime extension |
| `yellow_bin` | `waste_bin` | Alias plus runtime extension |
| `bin_tiger_stripe` | `waste_bin` | Alias plus runtime extension |
| `patient` | `Patient` | Alias |
| `medical_tray` | `medical_tray` | Runtime extension |
| Unknown class | `PhysicalObject` | Fallback |

Each resolved record includes the final ontology URI, display name, hierarchy, comments, physical dimensions, and other RDF properties.

## Physical Dimensions

The ontology stores physical-size knowledge as class-level annotations using the custom property `classes-ontology:physicalDimensions`. Values are JSON in metres.

Example:

```xml
<classes-ontology:physicalDimensions>
{"typical":{"width":1.0,"height":2.1,"depth":0.05},"range":{"width":[0.8,1.4],"height":[1.9,2.4]}}
</classes-ontology:physicalDimensions>
```

These dimensions are used by the RGB-D physical-size gate to reject detections whose measured 3D width or height is outside the ontology range. When no size range exists, the detection is kept because missing ontology data is not treated as evidence that an object is invalid.

## Semantic Map To RDF

The semantic map is stored as SQLite under:

```text
04_outputs_runs_and_logs/outputs/semantic_maps/<run-name>/world_map.db
```

[semantic_map_to_rdf.py](semantic_map_to_rdf.py) converts each landmark row into RDF triples:

| Semantic-map item | RDF/OWL representation |
|---|---|
| Landmark object | Individual typed with the resolved ontology class. |
| 3D position | `Location` individual with `hasX`, `hasY`, `hasZ`, and `worldFrame`. |
| Detection evidence | Reified `Statement` with subject, predicate, object, confidence, and timestamp. |
| Nearby obstruction | Optional `Obstacle` role and `blocks` relation when obstacle classes are near protected classes. |

The exporter can also seed candidate obstruction relations. For example, if a `bag`, `waste_bin`, `wheelchair`, `cabinet`, or `utility_trolley` is close to a protected object such as a `door`, `exit_sign`, `fire_extinguisher`, `fire_hydrant`, or `switchboard`, it asserts a `blocks` relation within the configured radius.

## How To Use

Run commands from the project root:

```powershell
cd "D:\Object Detection Model\yolo_tr\yolo_tr\Cognitive Recognition framework"
```

### 1. Inspect the ontology in Protege

Open this file directly:

```text
01_codebase/09_ontology/ontology.rdf
```

Use Protege to inspect classes, subclass relationships, object properties, datatype properties, annotation properties, and physical-dimension annotations.

### 2. Check ontology resolution from Python

```powershell
& ".\07_environment_and_project_meta\.venv-gpu311\Scripts\python.exe" `
  -c "import sys; sys.path.insert(0, r'.\01_codebase\09_ontology'); from ontology_knowledge import OntologyKnowledgeBase; kb=OntologyKnowledgeBase(); print('triples=', len(kb.graph)); [print(name, kb.resolve(name)['resolution'], kb.resolve(name)['resolved_name'], kb.resolve(name)['dimensions']) for name in ('door', 'utility_trolley', 'power_socket', 'general_bin', 'medical_tray', 'not_a_real_class')]"
```

This verifies that the ontology loads, bridge classes are added, aliases resolve correctly, and representative dimensions are available.

### 3. Export semantic-map landmarks to RDF

```powershell
& ".\07_environment_and_project_meta\.venv-gpu311\Scripts\python.exe" `
  ".\01_codebase\09_ontology\semantic_map_to_rdf.py" `
  --db ".\04_outputs_runs_and_logs\outputs\semantic_maps\rgbd_clean_20260521_142555\world_map.db" `
  --out ".\04_outputs_runs_and_logs\outputs\semantic_maps\rgbd_clean_20260521_142555\semantic_map.rdf"
```

To export Turtle instead of RDF/XML:

```powershell
& ".\07_environment_and_project_meta\.venv-gpu311\Scripts\python.exe" `
  ".\01_codebase\09_ontology\semantic_map_to_rdf.py" `
  --format turtle
```

### 4. Open the ontology-enriched HTML semantic-map viewer

```powershell
& ".\07_environment_and_project_meta\.venv-gpu311\Scripts\python.exe" `
  ".\01_codebase\08_semantic_map\view_semantic_map.py" `
  --db ".\04_outputs_runs_and_logs\outputs\semantic_maps\rgbd_clean_20260521_142555\world_map.db" `
  --html
```

Click a landmark in the HTML viewer to inspect its detected class, resolved ontology class, hierarchy, coordinates, dimensions, confidence, comments, and ontology properties.

### 5. Open the Rerun semantic-map recording

```powershell
$env:Path = (Resolve-Path ".\07_environment_and_project_meta\.venv-gpu311\Scripts").Path + ";" + $env:Path
& ".\07_environment_and_project_meta\.venv-gpu311\Scripts\rerun.exe" `
  ".\04_outputs_runs_and_logs\outputs\semantic_maps\rgbd_clean_20260521_142555\world_map.rrd"
```

In Rerun, select an entity under `world/landmarks` and inspect ontology metadata in the Data Inspector.

## Validation

Run the focused ontology and semantic-map viewer tests:

```powershell
& ".\07_environment_and_project_meta\.venv-gpu311\Scripts\python.exe" `
  -m unittest ".\10_Testing\test_ontology_semantic_map_viewer.py" -v
```

## Recommended GitHub Navigation For Reviewers

Start here:

1. [ontology.rdf](ontology.rdf) for the OWL/RDF ontology.
2. [ONTOLOGY_INFORMATION.md](ONTOLOGY_INFORMATION.md) for the complete class inventory, dimensions, aliases, and integration details.
3. [ontology_knowledge.py](ontology_knowledge.py) to see how detector labels are resolved at runtime.
4. [semantic_map_to_rdf.py](semantic_map_to_rdf.py) to see how mapped objects become RDF individuals and semantic statements.
5. [Run_commands.txt](Run_commands.txt) for exact commands to reproduce ontology checks, RDF export, and visualization.