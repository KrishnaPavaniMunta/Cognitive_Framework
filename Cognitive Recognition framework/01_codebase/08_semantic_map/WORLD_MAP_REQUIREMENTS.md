# MANDATORY SPECIFICATION: 3D DIGITAL TWIN & WORLD MAP (RRD) ARTIFACTS

This document defines the permanent, mandatory architectural contract for the 3D Digital Twin (`world_map.rrd`), interactive viewer (`semantic_map_ontology.html`), and compliance evaluation across the Cognitive Recognition framework.

All future development, reconstruction pipelines, and export scripts MUST strictly adhere to these rules without exception.

---

## 1. Required Core Elements in `world_map.rrd`

Every generated `world_map.rrd` must contain all six of the following constituent layers:

### A. Environmental 3D Point Cloud (`world/environment_point_cloud`)
- Full 3D RGB-D point cloud representation of the physical environment (walls, corridors, floor, ceiling).
- Smoothly filtered, outlier-rejected, and accurately registered in the `odom`/`world` frame.
- Provides immediate visual spatial context for human inspectors and robot path planners.

### B. Environmental LiDAR Perimeter (`world/lidar/perimeter_points`)
- 2D/3D LiDAR perimeter wall boundaries and laser scan points (`/scan` topic).
- Captures true structural wall geometry and corridor boundaries regardless of lighting conditions.

### C. Synchronized 2D/3D Annotated Video Stream (`world/camera/...`)
- The video stream (`world/camera/annotated_image` and `world/camera/image`) synchronized with the camera trajectory `world/camera`.
- Renders the robot's real-time visual perception:
  * 2D bounding boxes and class badges.
  * FastSAM neural instance segmentation masks with class-specific tinting.
  * EasyOCR transcribed text badges (e.g. `[ROOM: DEDJ]`, `[EXIT: EMERGENCY EXIT]`).
- Fully scrubbable along the timeline slider in the Rerun Viewer.

### D. Health & Safety (H&S) Egress Volumes & Labels (`world/door_zones`, `world/egress_zones`)
- **3D Semi-Cylindrical Keep-Clear Egress Volumes (1.0m radius arc, 2.1m height, green `#2ecc71`)**:
  * Placed on all primary Fire Exit Doors (`door 1`, `door 3`, `door 4`).
  * Placed directly beneath all Emergency Exit Signs (`exit_sign 1`, etc.).
- **3D Compliance Hazard Lines (`world/compliance_hazards/...`)**:
  * Dynamic red hazard vectors connecting trip hazards (e.g. spillage, movable chairs) to blocked circulation corridors or egress paths.

### E. Exact Landmark Labels & Physical Surface Points
- Every persistent landmark is represented by:
  * Its genuine 3D physical surface point cloud (`rr.Points3D`).
  * Exact display label: `"{class_name} #{instance_id} ({confidence:.0%})"` or transcribed room nameplate.
  * Rich ontology metadata attributes attached to the entity (`rr.AnyValues`).

### F. STRICT PROHIBITION: NO CUBOIDS (`rr.Boxes3D` REMOVED)
- **Do NOT render wireframe bounding boxes or cuboids (`rr.Boxes3D`)** over landmarks.
- Cuboids clutter the scene, obscure physical surface points, and distort irregular shapes.
- Landmarks must be displayed solely via their physical 3D surface points, exact badges/labels, and Keep-Clear Egress Cylinders.

---

## 2. Preservation of Previous Assets (Additive Development)

1. **Safety Signs & Exit Doors**:
   - Exit signs (`exit_sign`) and doors (`door`) are foundational safety assets.
   - Text detection engines (OCR) must NEVER demote or reclassify an exit sign into a generic `room_sign`.
   - Exit signs are physically mounted above door headers or high on walls ($Z \in [1.60, 2.90]\text{ m}$). Non-elevated surfaces (carpets at $Z \approx 0.0\text{ m}$, desks at $Z \approx 0.7\text{ m}$) must NEVER be classified as exit signs.
2. **Door Keep-Clear Zones (Egress Cylinders)**:
   - Must be preserved and rendered whenever exit doors or exit signs exist in the digital twin.
   - Both Rerun (`world_map.rrd`) and the Web viewer (`semantic_map_ontology.html`) must display the 3D semi-cylindrical volumes.
3. **Persistent Signage Observation Threshold ($\ge 10$ hits)**:
   - Signs (`exit_sign`, `hazmat_sign`, `sign_prohibition_access`, etc.) must accumulate at least 10 observations across multi-view reconstruction to be confirmed as genuine persistent environmental signage.
   - Transient 1–3 hit false positives (e.g. paper notices or badges taped to door glass) are rejected from creating safety landmarks or falsely designating interior doors as fire exits.

---

## 3. Implementation Verification Checklist

Before marking any reconstruction or export run as complete, verify:
- [x] Environmental 3D Point Cloud is present and rendered.
- [x] Environmental LiDAR wall boundary is present.
- [x] Annotated video stream is scrubbable on timeline.
- [x] 3D Keep-Clear Door Egress Cylinders are visible.
- [x] **Zero cuboids (`rr.Boxes3D`) are emitted.**
- [x] Exit signs and doors are preserved.
