# Object Categorization Reference (HospitalGuard)

## Why this file
This document gives a single place to check object categorization instead of digging through multiple scripts.

## Current Status in Codebase (As-Is)

### Explicitly defined category
Only one explicit category is currently defined in code:
- `STATIC_CLASS_NAMES`

Defined in:
- `01_codebase/04_rgbd_and_spatial_twin/hospital_detector_longterm/rgbd_development/scripts/hospital_constants.py`

Used by:
- `rgbd_hospitalguard_temporal.py` (static ID stabilization + spatial DB writes)
- `rgbd_hospitalguard_temporal_orbbec.py` (static spatial writes)
- `visualize_spatial_map_interactive.py` and `visualize_spatial_map.py` (static anchors for world alignment)

### Not explicitly defined in code
No explicit constants were found for:
- `DYNAMIC_CLASS_NAMES`
- `SEMI_STATIC_CLASS_NAMES`

So today, categorization is effectively:
- static = classes in `STATIC_CLASS_NAMES`
- non-static = everything else (implicit)

---

## Category Lists for Practical Use

## 1) Static Objects (explicit in code)
These are currently treated as spatial anchors / persistent scene objects.

- hospital_bed, infusion_pump, iv_stand, monitor_hosp, patient_monitor, surgical_light, vending_machines, wheelchair, hospital_stretcher
- cabinet, bench_hosp, door, reception_counter, radiator, bathroom_labels, fire_extinguisher, security_camera, exit_sign
- iv_bag, test_tube, surgical_scissor, spillage
- bench, chair, couch, potted plant, bed, dining table, toilet, tv, microwave, oven, toaster, sink, refrigerator
- bottle, wine glass, cup, fork, knife, spoon, bowl, laptop, mouse, remote, keyboard, cell phone, book, clock, vase, scissors, teddy bear, hair drier, toothbrush, bag
- traffic light, fire hydrant, stop sign, parking meter
- banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake

## 2) Dynamic Objects (recommended practical grouping)
These are typically moving actors/agents in scenes.

- person, healthcare_worker, patient
- bicycle, car, motorcycle, bus, train, truck, boat, airplane
- bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
- skateboard, surfboard, skis, snowboard, sports ball, kite

## 3) Semi-Static Objects (recommended practical grouping)
These can be stationary for long periods but are commonly moved/repositioned.

- wheelchair, hospital_stretcher
- iv_stand, infusion_pump, iv_bag
- surgical_scissor, test_tube, bag
- bottle, cup, bowl, laptop, cell phone, book
- spillage (transient hazard; not a stable anchor over time)

---

## Important Note
There is currently a mismatch between semantic behavior and current static anchors:
- Some objects in `STATIC_CLASS_NAMES` are physically movable (example: wheelchair, iv_stand, bag).

If you want strict long-term spatial stability, consider splitting into:
- true static anchors (walls/fixtures/fixed equipment)
- semi-static movable assets
- dynamic actors

This file can be used as your immediate reference until constants are formalized in code.
