# Hospital Cognitive Recognition Ontology

## 1. Overview

The authoritative ontology file is:

`01_codebase/09_ontology/ontology.rdf`

It is an RDF/XML OWL ontology used by the object-detection, physical-size filtering, semantic-map, RDF-export, and viewer components.

The ontology has two related class counts:

| Scope | Count | Meaning |
|---|---:|---|
| Base ontology file | 93 | Named OWL classes declared in `ontology.rdf` |
| Runtime knowledge graph | 95 | The 93 base classes plus 2 semantic-map bridge classes |

The two runtime bridge classes are:

- `medical_tray`, subclass of `MedicalDevice`
- `waste_bin`, subclass of `Infrastructure`

The bridge classes are added in `ontology_knowledge.py` because these detector/map labels do not have matching base classes in the original ontology namespace.

## 2. Ontology Namespaces

| Prefix | Namespace | Purpose |
|---|---|---|
| `CO` | `http://www.semanticweb.org/chevi/ontologies/2026/5/52-classes-ontology#` | Main classes ontology and physical-dimension annotation |
| `UO` | `http://www.semanticweb.org/chevi/ontologies/2026/5/untitled-ontology-6#` | Original ontology classes |
| `BR` | `http://www.semanticweb.org/chevi/ontologies/2026/5/semantic-map-bridge#` | Runtime semantic-map extension classes |

Protégé should open the RDF/XML file directly. The class hierarchy is represented with `rdfs:subClassOf` relationships.

## 3. Top-Level Concept Groups

### Physical-object hierarchy

```text
PhysicalObject
├── Furniture
├── Infrastructure
├── MedicalDevice
├── PersonalBelonging
└── SafetyEquipment
```

### Person hierarchy

```text
Person
├── Patient
├── Staff
│   └── healthcare_worker
└── Visitor
```

### Context and location hierarchy

```text
Context
├── EnvironmentContext
├── SpatialContext
└── TemporalContext

Location
├── Environment
└── IndoorSpace
```

### Event, hazard, and role hierarchy

```text
Event
├── Alert
└── HazardEvent

Role
├── Hazard
│   ├── ContaminationHazard
│   ├── FireHazard
│   └── TrippingHazard
├── MonitoredEntity
├── Obstacle
└── SafetyZone
```

Other standalone support concepts include `Predicate` and `Statement`.

## 4. Complete Class Inventory

The following is the inventory of the 93 named OWL classes in the base RDF file. The parent shown after each class is its direct superclass.

### Core concepts and roles

- `Alert` -> `Event`
- `ContaminationHazard` -> `Hazard`
- `Context`
- `Environment` -> `Location`
- `EnvironmentContext` -> `Context`
- `Event`
- `FireHazard` -> `Hazard`
- `Furniture` -> `PhysicalObject`
- `Hazard` -> `Role`
- `HazardEvent` -> `Event`
- `IndoorSpace` -> `Location`
- `Infrastructure` -> `PhysicalObject`
- `Location`
- `MedicalDevice` -> `PhysicalObject`
- `MonitoredEntity` -> `Role`
- `Obstacle` -> `Role`
- `Patient` -> `Person`
- `Person`
- `PersonalBelonging` -> `PhysicalObject`
- `PhysicalObject`
- `Predicate`
- `Role`
- `SafetyEquipment` -> `PhysicalObject`
- `SafetyZone` -> `Role`
- `SpatialContext` -> `Context`
- `Staff` -> `Person`
- `Statement`
- `TemporalContext` -> `Context`
- `TrippingHazard` -> `Hazard`
- `Visitor` -> `Person`

### Furniture

- `bed` -> `Furniture`
- `bench` -> `Furniture`
- `bench_hosp` -> `Furniture`
- `cabinet` -> `Furniture`
- `chair` -> `Furniture`
- `couch` -> `Furniture`
- `dining_table` -> `Furniture`
- `reception_counter` -> `Furniture`
- `utility_trolley` -> `Furniture`
- `vending_machines` -> `Furniture`

### Infrastructure

- `bathroom_labels` -> `Infrastructure`
- `clock` -> `Infrastructure`
- `door` -> `Infrastructure`
- `hand_sanitizer` -> `Infrastructure`
- `large_bin` -> `Infrastructure`
- `power_socket` -> `Infrastructure`
- `radiator` -> `Infrastructure`
- `refrigerator` -> `Infrastructure`
- `sink` -> `Infrastructure`
- `small_bin` -> `Infrastructure`
- `switchboard` -> `Infrastructure`
- `toilet` -> `Infrastructure`

Runtime extension:

- `waste_bin` -> `Infrastructure`

### Medical devices

- `hospital_bed` -> `MedicalDevice`
- `hospital_stretcher` -> `MedicalDevice`
- `infusion_pump` -> `MedicalDevice`
- `iv_bag` -> `MedicalDevice`
- `iv_stand` -> `MedicalDevice`
- `monitor_hosp` -> `MedicalDevice`
- `nasal_cannula` -> `MedicalDevice`
- `oxygen_cylinder` -> `MedicalDevice`
- `oxygen_pump` -> `MedicalDevice`
- `patient_monitor` -> `MedicalDevice`
- `surgical_light` -> `MedicalDevice`
- `surgical_scissor` -> `MedicalDevice`
- `test_tube` -> `MedicalDevice`
- `wheelchair` -> `MedicalDevice`
- `wheelchair_manual` -> `wheelchair`
- `wheelchair_powered` -> `wheelchair`

Runtime extension:

- `medical_tray` -> `MedicalDevice`

### Personal belongings

- `backpack` -> `PersonalBelonging`
- `bag` -> `PersonalBelonging`
- `baseball_glove` -> `PersonalBelonging`
- `book` -> `PersonalBelonging`
- `bottle` -> `PersonalBelonging`
- `cell_phone` -> `PersonalBelonging`
- `cup` -> `PersonalBelonging`
- `fork` -> `PersonalBelonging`
- `handbag` -> `PersonalBelonging`
- `knife` -> `PersonalBelonging`
- `scissors` -> `PersonalBelonging`
- `spoon` -> `PersonalBelonging`
- `suitcase` -> `PersonalBelonging`

### Safety equipment

- `exit_sign` -> `SafetyEquipment`
- `fire_extinguisher` -> `SafetyEquipment`
- `fire_hydrant` -> `SafetyEquipment`
- `glove` -> `SafetyEquipment`
- `hair_net` -> `SafetyEquipment`
- `mask` -> `SafetyEquipment`
- `security_camera` -> `SafetyEquipment`
- `security_camera_bullet` -> `security_camera`
- `security_camera_dome` -> `security_camera`

### People and worker classes

- `healthcare_worker` -> `Staff`
- `person` -> `Person`

## 5. Detector Aliases

The shared resolver maps detector labels to ontology classes as follows:

| Detector/map label | Ontology target | Resolution |
|---|---|---|
| `general_bin` | `waste_bin` | Alias plus runtime extension |
| `yellow_bin` | `waste_bin` | Alias plus runtime extension |
| `bin_tiger_stripe` | `waste_bin` | Alias plus runtime extension |
| `patient` | `Patient` | Alias |
| `medical_tray` | `medical_tray` | Runtime extension |
| Unknown class | `PhysicalObject` | Fallback only |

The resolver implementation is in `ontology_knowledge.py`.

## 6. Physical Dimensions

Structured physical-size information is stored with the custom RDF annotation property:

```xml
<owl:AnnotationProperty
  rdf:about="...#physicalDimensions"/>
```

The property is declared around line 218 of `ontology.rdf`. Class-level values are stored as JSON annotations in metres.

Example:

```xml
<classes-ontology:physicalDimensions>
{"typical":{"width":1.0,"height":2.1,"depth":0.05},"range":{"width":[0.8,1.4],"height":[1.9,2.4]}}
</classes-ontology:physicalDimensions>
```

The normalized application record exposes these fields:

- `width`
- `depth`
- `height`
- `min_width`
- `max_width`
- `min_depth`
- `max_depth`
- `min_height`
- `max_height`

All values are metres.

### Classes with structured dimensions

There are currently 36 classes with structured `physicalDimensions` values:

```text
door
exit_sign
fork
glove
hair_net
hand_sanitizer
hospital_stretcher
infusion_pump
iv_stand
knife
large_bin
mask
nasal_cannula
oxygen_cylinder
oxygen_pump
power_socket
radiator
security_camera_bullet
security_camera_dome
scissors
small_bin
spoon
surgical_light
surgical_scissor
utility_trolley
wheelchair_manual
wheelchair_powered
bag
cabinet
chair
cup
patient_monitor
suitcase
switchboard
vending_machines
wheelchair
```

`utility_trolley` is represented as:

```text
width  = 0.50 m
depth  = 0.80 m
height = 0.95 m
```

Its physical-size gate range is width `0.40-0.60 m`, depth `0.64-0.96 m`, and height `0.76-1.14 m`.

### Comment-only dimensions

Twenty-seven classes contain dimension descriptions in `rdfs:comment` text but do not yet have structured RDF dimension annotations:

```text
Patient
backpack
baseball_glove
bathroom_labels
bed
bench
bench_hosp
book
bottle
cell_phone
clock
couch
dining_table
fire_extinguisher
fire_hydrant
handbag
healthcare_worker
hospital_bed
iv_bag
monitor_hosp
person
reception_counter
refrigerator
security_camera
sink
test_tube
toilet
```

The following nine classes were migrated from comments into structured numeric RDF values:

```text
bag
cabinet
chair
cup
patient_monitor
suitcase
switchboard
vending_machines
wheelchair
```

The following five classes were added from standard or clinical sizing references:

```text
fork
nasal_cannula
scissors
spoon
surgical_scissor
```

The shared resolver can still extract simple centimeter triplets from comments for remaining legacy classes, but comment-derived values are not treated as strict physical-size gate ranges.

### Classes without dimensions

Thirty classes currently have no dimension data or usable dimension comment:

```text
Alert
ContaminationHazard
Context
Environment
EnvironmentContext
Event
FireHazard
Furniture
Hazard
HazardEvent
IndoorSpace
Infrastructure
Location
MedicalDevice
MonitoredEntity
Obstacle
Person
PersonalBelonging
PhysicalObject
Predicate
Role
SafetyEquipment
SafetyZone
SpatialContext
Staff
Statement
spillage
TemporalContext
TrippingHazard
Visitor
```

The full list should be regenerated with the ontology audit command when the ontology changes.

## 7. Physical-Size Gate

The physical-size gate is implemented in:

`01_codebase/07_object_detection/rgbd_3d_filter.py`

Processing flow:

1. Read structured dimensions from `ontology.rdf`.
2. Match the detector class, including configured aliases.
3. Back-project the detection region from RGB-D depth into 3D points.
4. Estimate oriented width and height using depth points and PCA.
5. Compare the measured width and height against ontology min/max ranges.
6. Reject detections outside the configured range.
7. Keep detections when no dimension limits are available, because lack of ontology data is not itself proof of a false positive.

The gate currently uses strict min/max width and height ranges. Typical depth values are available for many classes, but depth min/max ranges are not generally defined and depth is not currently used as a rejection axis.

## 8. Semantic-Map Integration

The persistent semantic map is stored in SQLite at:

```text
04_outputs_runs_and_logs/outputs/semantic_maps/<bag-name>/world_map.db
```

The map stores object identity, class, instance ID, world coordinates, confidence statistics, hit counts, and observation timestamps.

Ontology data is attached to each mapped landmark by `view_semantic_map.py` and is written to Rerun by `rerun_logger.py`.

The current Rerun entity metadata includes readable fields such as:

- `map_class`
- `ontology_class`
- `allowed_in_space` (`UNKNOWN` until space rules are supplied)
- `ontology_hierarchy`
- `ontology_dimensions_json`
- `ontology_comments`
- `ontology_properties_json`

Raw ontology URI fields are intentionally excluded from the user-facing Rerun metadata.

## 9. Viewers and Tools

### Protégé

Use Protégé to inspect and edit:

- OWL classes
- `rdfs:subClassOf` relationships
- Annotation properties
- Object properties
- Data properties
- Individuals and restrictions

Open:

`01_codebase/09_ontology/ontology.rdf`

The current physical dimensions appear under each class's **Annotations** section as `physicalDimensions` JSON.

### HTML semantic-map viewer

The Plotly viewer is generated with:

```powershell
& ".\07_environment_and_project_meta\.venv-gpu311\Scripts\python.exe" `
  ".\01_codebase\08_semantic_map\view_semantic_map.py" `
  --db ".\04_outputs_runs_and_logs\outputs\semantic_maps\rgbd_clean_20260521_142555\world_map.db" `
  --html
```

Click a landmark to inspect its class, type, coordinates, last observation, dimensions, hierarchy, properties, and current space-compatibility status.

### Rerun viewer

The Rerun scene contains the RGB-D cloud, camera trajectory, labels, boxes, observations, and ontology-bearing landmark entities.

Open the current recording with:

```powershell
$env:Path = (Resolve-Path ".\07_environment_and_project_meta\.venv-gpu311\Scripts").Path + ";" + $env:Path
& ".\07_environment_and_project_meta\.venv-gpu311\Scripts\rerun.exe" `
  ".\04_outputs_runs_and_logs\outputs\semantic_maps\rgbd_clean_20260521_142555\world_map.rrd"
```

Select an entity under `world/landmarks` and inspect its metadata in the Data Inspector.

## 10. Persistent Map and Rerun History

SQLite is persistent and is reopened on later bag runs. Existing landmarks are loaded and merged by class and spatial distance. New landmarks are added and existing landmarks receive updated coordinates, confidence, hit counts, and timestamps.

Rerun file sinks cannot append to an existing `.rrd` file. The builder therefore preserves the previous recording under:

```text
run_history/world_map_<run-id>.rrd
```

before writing a new current `world_map.rrd`. This prevents silent deletion of previous Rerun data. Multiple recordings can be opened together in the Rerun CLI.


## 11. Related Files

- [ontology.rdf](ontology.rdf) - authoritative RDF/XML ontology
- [ontology_knowledge.py](ontology_knowledge.py) - shared resolver and normalized knowledge extraction
- [semantic_map_to_rdf.py](semantic_map_to_rdf.py) - semantic-map RDF exporter
- [../08_semantic_map/view_semantic_map.py](../08_semantic_map/view_semantic_map.py) - Matplotlib, HTML, and Rerun viewer entry point
- [../08_semantic_map/semantic_map_html.py](../08_semantic_map/semantic_map_html.py) - Plotly HTML inspector
- [../08_semantic_map/rerun_logger.py](../08_semantic_map/rerun_logger.py) - Rerun scene and landmark metadata logging
- [../../10_Testing/test_ontology_semantic_map_viewer.py](../../10_Testing/test_ontology_semantic_map_viewer.py) - focused ontology/viewer tests
