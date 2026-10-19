"""
Formal Qualitative Spatial Reasoning (QSR) and 3D Scene Graph Engine.

Combines the open-vocabulary 3D scene graph representation principles of
ConceptGraphs (Guhur et al., ICRA 2024) with deterministic QSR spatial formalisms
(KMi Open University / Ibba et al., 2026; Chiatti, 2022) and NHS / UK Building
Regulations fire safety clearance rules (Andrea's egress half-cylinder zones).

Zero LLM coordinate hallucination: All spatial relationships and safety edges
are computed analytically with sensor-noise tolerance (+/- 8cm).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Sequence

import numpy as np

# ── Semantic Taxonomy & Affordance Dictionaries ──────────────────────────────

MOVABLE_OBSTACLE_CLASSES = {
    "chair", "hospital_stretcher", "utility_trolley", "bin", "general_bin",
    "yellow_bin", "bin_tiger_stripe", "suitcase", "medical_tray", "dining table",
    "bench_hosp", "box", "bed", "wheelchair", "unidentified_obstacle", "obstacle",
    "item", "thing",
}

COMBUSTIBLE_CLASSES = {
    "general_bin", "yellow_bin", "bin_tiger_stripe", "bin", "suitcase",
    "paper", "cardboard", "linen", "waste", "box",
}

IGNITION_CLASSES = {
    "power_socket", "electrical_cabinet", "fuse_box", "radiator", "distribution_board",
}

STRUCTURAL_CLASSES = {
    "wall", "door", "floor", "navigable_floor", "ceiling", "pillar", "column", "window",
}

EMERGENCY_ASSET_CLASSES = {
    "fire_extinguisher", "fire_alarm", "emergency_exit_sign", "exit_sign", "call_point", "first_aid_kit",
}


# ── 3D Bounding Box Geometry ──────────────────────────────────────────────────

@dataclass
class BoundingBox3D:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    @classmethod
    def from_center_extents(
        cls,
        cx: float,
        cy: float,
        cz: float,
        depth: float,
        width: float,
        height: float,
    ) -> BoundingBox3D:
        """Create a 3D bounding box from ROS world center (X forward, Y left, Z up)."""
        hd = max(0.02, float(depth)) / 2.0
        hw = max(0.02, float(width)) / 2.0
        hh = max(0.02, float(height)) / 2.0
        return cls(
            x_min=float(cx) - hd,
            x_max=float(cx) + hd,
            y_min=float(cy) - hw,
            y_max=float(cy) + hw,
            z_min=float(cz) - hh,
            z_max=float(cz) + hh,
        )

    @classmethod
    def from_points(cls, points: np.ndarray, padding_m: float = 0.02) -> BoundingBox3D:
        """Fit an axis-aligned 3D bounding box around an array of points (N, 3)."""
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] == 0:
            return cls(0.0, 0.1, 0.0, 0.1, 0.0, 0.1)
        mins = pts.min(axis=0) - padding_m
        maxs = pts.max(axis=0) + padding_m
        return cls(
            x_min=float(mins[0]), x_max=float(maxs[0]),
            y_min=float(mins[1]), y_max=float(maxs[1]),
            z_min=float(mins[2]), z_max=float(maxs[2]),
        )

    @property
    def center(self) -> tuple[float, float, float]:
        return (
            (self.x_min + self.x_max) / 2.0,
            (self.y_min + self.y_max) / 2.0,
            (self.z_min + self.z_max) / 2.0,
        )

    @property
    def extents(self) -> tuple[float, float, float]:
        return (
            max(0.02, self.x_max - self.x_min),
            max(0.02, self.y_max - self.y_min),
            max(0.02, self.z_max - self.z_min),
        )

    @property
    def volume(self) -> float:
        dx, dy, dz = self.extents
        return max(0.0, dx * dy * dz)

    def distance_to(self, other: BoundingBox3D) -> float:
        """Minimum 3D Euclidean distance between two axis-aligned bounding boxes."""
        dx = max(0.0, self.x_min - other.x_max, other.x_min - self.x_max)
        dy = max(0.0, self.y_min - other.y_max, other.y_min - self.y_max)
        dz = max(0.0, self.z_min - other.z_max, other.z_min - self.z_max)
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def lateral_distance_to(self, other: BoundingBox3D) -> float:
        """Minimum 2D Euclidean distance on the ground (XY) plane."""
        dx = max(0.0, self.x_min - other.x_max, other.x_min - self.x_max)
        dy = max(0.0, self.y_min - other.y_max, other.y_min - self.y_max)
        return math.sqrt(dx * dx + dy * dy)

    def intersection_volume(self, other: BoundingBox3D) -> float:
        """Calculate overlapping volume (m^3) between two 3D boxes."""
        ix = max(0.0, min(self.x_max, other.x_max) - max(self.x_min, other.x_min))
        iy = max(0.0, min(self.y_max, other.y_max) - max(self.y_min, other.y_min))
        iz = max(0.0, min(self.z_max, other.z_max) - max(self.z_min, other.z_min))
        return ix * iy * iz

    def iou_3d(self, other: BoundingBox3D) -> float:
        """3D Intersection-over-Union."""
        inter = self.intersection_volume(other)
        if inter <= 0.0:
            return 0.0
        union = self.volume + other.volume - inter
        return inter / union if union > 0.0 else 0.0

    def iou_2d(self, other: BoundingBox3D) -> float:
        """2D Intersection-over-Union on the XY ground plane."""
        inter_x = max(0.0, min(self.x_max, other.x_max) - max(self.x_min, other.x_min))
        inter_y = max(0.0, min(self.y_max, other.y_max) - max(self.y_min, other.y_min))
        inter_area = inter_x * inter_y
        if inter_area <= 0.0:
            return 0.0
        area_a = (self.x_max - self.x_min) * (self.y_max - self.y_min)
        area_b = (other.x_max - other.x_min) * (other.y_max - other.y_min)
        union_area = area_a + area_b - inter_area
        return inter_area / union_area if union_area > 0.0 else 0.0


# ── 3D Door Keep-Clear Egress Half-Cylinder Zone ──────────────────────────────
# Operationalizes Andrea's NHS Firecode / UK Building Regs Approved Document B
# and integrates geometry from 06_anomaly_detection/Blocked_exit_detection/door_zone_rgbd.py

@dataclass
class DoorKeepClearZone3D:
    door_id: int
    center: tuple[float, float, float]       # (xd, yd, zd) door center in world frame
    width: float = 0.90                      # door opening width
    height: float = 2.00                     # door opening height
    radius: float = 1.00                     # keep-clear egress radius (default 1.0m, or 1.2m)
    egress_normal: tuple[float, float] = (0.0, 1.0)  # unit vector pointing into room/corridor
    z_min: float = 0.00                      # floor elevation at door threshold
    z_max: float = 2.05                      # top of door header clearance
    required_clear_width: float = 0.85       # minimum legally required passage width (m)
    is_fire_exit: bool = True                # Only doors with an exit sign on top are emergency fire exits
    is_obstructed: bool = False              # True if any obstacle penetrates the keep-clear volume
    status: str = "CLEAR"                    # "CLEAR" (green) or "OBSTRUCTED" (red)
    color_hex: str = "#2ecc71"               # Default green
    obstructing_items: list[dict] = field(default_factory=list)

    def contains_point(self, x: float, y: float, z: float, tolerance_m: float = 0.05) -> bool:
        """
        Analytically checks if a 3D point falls inside the egress half-cylinder zone.
        Directly mirrors _point_inside_half_cylinder from Obstruction_detection.py,
        adapted to the world reference frame (Z up, XY planar).
        """
        # 1. Vertical extrusion check across the entire door height
        if z < (self.z_min - tolerance_m) or z > (self.z_max + tolerance_m):
            return False

        # 2. Horizontal distance from door center
        dx = x - self.center[0]
        dy = y - self.center[1]
        dist_sq = dx * dx + dy * dy
        if dist_sq > ((self.radius + tolerance_m) ** 2):
            return False

        # 3. Directional half-space (only in front of door along egress normal)
        nx, ny = self.egress_normal
        dot_fwd = dx * nx + dy * ny
        return dot_fwd >= -tolerance_m

    def intersects_box(self, box: BoundingBox3D, tolerance_m: float = 0.08) -> tuple[bool, float, float]:
        """
        Check if an obstacle bounding box penetrates the door keep-clear half-cylinder.
        Evaluates full vertical extrusion from floor to door header (hanging or floor obstacles).
        Returns:
            (intersects, penetration_distance_m, clear_width_deficit_m)
        """
        # Vertical overlap check across door height
        if box.z_max < (self.z_min - tolerance_m) or box.z_min > (self.z_max + tolerance_m):
            return False, 0.0, 0.0

        # Planar distance check
        cx, cy, _ = box.center
        dx = cx - self.center[0]
        dy = cy - self.center[1]
        planar_dist = math.sqrt(dx * dx + dy * dy)
        max_box_radius = max(box.extents[0], box.extents[1]) / 2.0

        if (planar_dist - max_box_radius) > self.radius:
            return False, 0.0, 0.0

        # Directional half-space check (in front of door)
        nx, ny = self.egress_normal
        dot_fwd = dx * nx + dy * ny
        if dot_fwd < (-max_box_radius - tolerance_m):
            return False, 0.0, 0.0

        # Obstruction detected: calculate intrusion metrics
        intrusion = max(0.0, self.radius - (planar_dist - max_box_radius))

        # Check remaining clear passage width
        obstacle_width = min(box.extents[0], box.extents[1])
        unblocked_width = max(0.0, self.width - obstacle_width)
        width_deficit = max(0.0, self.required_clear_width - unblocked_width)

        return True, round(intrusion, 3), round(width_deficit, 3)

    def intersects_points(self, points: np.ndarray, tolerance_m: float = 0.05) -> tuple[bool, int, float]:
        """
        Check if any points of an obstacle's 3D point cloud penetrate the keep-clear zone.
        Returns (is_intersecting, point_count_in_zone, min_dist_to_door).
        """
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] == 0:
            return False, 0, 999.0

        xd, yd, zd = self.center
        nx, ny = self.egress_normal

        # Vectorized check
        dx = pts[:, 0] - xd
        dy = pts[:, 1] - yd
        z = pts[:, 2]

        vert_mask = (z >= (self.z_min - tolerance_m)) & (z <= (self.z_max + tolerance_m))
        dist_sq = dx * dx + dy * dy
        rad_mask = dist_sq <= ((self.radius + tolerance_m) ** 2)
        fwd_mask = (dx * nx + dy * ny) >= -tolerance_m

        inside = vert_mask & rad_mask & fwd_mask
        count = int(np.count_nonzero(inside))

        if count > 0:
            min_dist = float(np.min(np.sqrt(dist_sq[inside])))
            return True, count, round(min_dist, 3)
        return False, 0, 999.0

    def sample_arc_3d(self, num_samples: int = 24) -> list[tuple[float, float, float]]:
        """Sample the semi-circular floor arc in 3D for digital twin visualization."""
        xd, yd, _ = self.center
        nx, ny = self.egress_normal
        base_angle = math.atan2(ny, nx)

        arc_pts: list[tuple[float, float, float]] = []
        for i in range(num_samples + 1):
            angle = base_angle - (math.pi / 2.0) + (math.pi * i / num_samples)
            px = xd + self.radius * math.cos(angle)
            py = yd + self.radius * math.sin(angle)
            arc_pts.append((round(px, 3), round(py, 3), round(self.z_min, 3)))
        return arc_pts

    def sample_cylinder_cage_3d(self, num_samples: int = 20) -> dict[str, Any]:
        """
        Samples the full extruded 3D semi-cylindrical volume (bottom arc, top arc,
        vertical corner posts, and ribs) matching door_zone_rgbd.py / Obstruction_detection.py.
        """
        xd, yd, _ = self.center
        nx, ny = self.egress_normal
        base_angle = math.atan2(ny, nx)

        bottom_arc: list[tuple[float, float, float]] = []
        top_arc: list[tuple[float, float, float]] = []

        for i in range(num_samples + 1):
            angle = base_angle - (math.pi / 2.0) + (math.pi * i / num_samples)
            px = xd + self.radius * math.cos(angle)
            py = yd + self.radius * math.sin(angle)
            bottom_arc.append((round(px, 3), round(py, 3), round(self.z_min, 3)))
            top_arc.append((round(px, 3), round(py, 3), round(self.z_max, 3)))

        # Ribs connecting bottom to top along curved wall
        vertical_ribs = []
        step = max(1, num_samples // 4)
        for i in range(0, num_samples + 1, step):
            vertical_ribs.append([bottom_arc[i], top_arc[i]])

        # Flat back-wall rectangle at door opening
        back_wall = [
            bottom_arc[0], bottom_arc[-1],
            top_arc[-1], top_arc[0],
            bottom_arc[0],
        ]

        return {
            "bottom_arc": bottom_arc,
            "top_arc": top_arc,
            "vertical_ribs": vertical_ribs,
            "back_wall": back_wall,
        }


# ── Directional Halfspace Predicates (Section III-B in Ibba et al.) ────────────

def is_above(a: BoundingBox3D, b: BoundingBox3D, tolerance_m: float = 0.08) -> bool:
    """True if bounding box A intersects the upward Z+ halfspace of B with horizontal alignment."""
    vert_ok = a.z_min >= (b.z_max - tolerance_m)
    # Check 2D ground alignment
    lat_dist = a.lateral_distance_to(b)
    return vert_ok and (lat_dist <= 0.35 or a.iou_2d(b) > 0.0)


def is_below(a: BoundingBox3D, b: BoundingBox3D, tolerance_m: float = 0.08) -> bool:
    """True if bounding box A intersects the downward Z- halfspace of B with horizontal alignment."""
    vert_ok = a.z_max <= (b.z_min + tolerance_m)
    lat_dist = a.lateral_distance_to(b)
    return vert_ok and (lat_dist <= 0.35 or a.iou_2d(b) > 0.0)


def is_in_front_of(a: BoundingBox3D, b: BoundingBox3D, normal: tuple[float, float] | None = None, tolerance_m: float = 0.08) -> bool:
    """True if A falls within the forward halfspace of B."""
    ca = a.center
    cb = b.center
    if normal is not None:
        nx, ny = normal
        dot = (ca[0] - cb[0]) * nx + (ca[1] - cb[1]) * ny
        return dot >= tolerance_m
    # Default: along global +X axis
    return ca[0] >= (cb[0] + tolerance_m)


def is_behind(a: BoundingBox3D, b: BoundingBox3D, normal: tuple[float, float] | None = None, tolerance_m: float = 0.08) -> bool:
    """True if A falls behind B along the viewing/normal axis."""
    return not is_in_front_of(a, b, normal=normal, tolerance_m=tolerance_m)


def is_left_of(a: BoundingBox3D, b: BoundingBox3D, normal: tuple[float, float] | None = None, tolerance_m: float = 0.08) -> bool:
    """True if A falls to the left of B relative to normal."""
    ca = a.center
    cb = b.center
    if normal is not None:
        # Cross product (normal x d) in 2D
        nx, ny = normal
        dx = ca[0] - cb[0]
        dy = ca[1] - cb[1]
        cross = nx * dy - ny * dx
        return cross >= tolerance_m
    # Default: along global +Y axis
    return ca[1] >= (cb[1] + tolerance_m)


def is_right_of(a: BoundingBox3D, b: BoundingBox3D, normal: tuple[float, float] | None = None, tolerance_m: float = 0.08) -> bool:
    """True if A falls to the right of B relative to normal."""
    return not is_left_of(a, b, normal=normal, tolerance_m=tolerance_m)


# ── Topological & Contact Predicates (Table I in Ibba et al.) ──────────────────

def touches(a: BoundingBox3D, b: BoundingBox3D, tolerance_m: float = 0.06) -> bool:
    """True if objects A and B physically touch or are adjacent within sensor margin."""
    return a.distance_to(b) <= tolerance_m


def is_on_top_of(a: BoundingBox3D, b: BoundingBox3D, tolerance_m: float = 0.08) -> bool:
    """
    True if object A rests on top of object B.
    Requires:
    1. Horizontal support overlap (XY alignment).
    2. Vertical support contact: bottom of A is at or near top of B.
    """
    # Horizontal footprint check: bounding boxes must align in XY
    horiz_overlap = (
        max(0.0, min(a.x_max, b.x_max) - max(a.x_min, b.x_min)) > -tolerance_m
        and max(0.0, min(a.y_max, b.y_max) - max(a.y_min, b.y_min)) > -tolerance_m
    )
    if not horiz_overlap:
        return False

    vertical_contact = abs(a.z_min - b.z_max) <= tolerance_m
    is_higher = a.z_min >= (b.z_min + 0.05)
    return vertical_contact and is_higher


def supports(b: BoundingBox3D, a: BoundingBox3D, tolerance_m: float = 0.08) -> bool:
    """Reciprocal predicate: True if B provides physical support to A."""
    return is_on_top_of(a, b, tolerance_m=tolerance_m)


def is_contained_in(a: BoundingBox3D, b: BoundingBox3D, tolerance_m: float = 0.05) -> bool:
    """True if bounding box A is topologically inside bounding box B."""
    return (
        a.x_min >= (b.x_min - tolerance_m) and a.x_max <= (b.x_max + tolerance_m) and
        a.y_min >= (b.y_min - tolerance_m) and a.y_max <= (b.y_max + tolerance_m) and
        a.z_min >= (b.z_min - tolerance_m) and a.z_max <= (b.z_max + tolerance_m)
    )


def contains(b: BoundingBox3D, a: BoundingBox3D, tolerance_m: float = 0.05) -> bool:
    """Reciprocal predicate: True if B topologically contains A."""
    return is_contained_in(a, b, tolerance_m=tolerance_m)


def is_affixed_to(a: BoundingBox3D, b: BoundingBox3D, tolerance_m: float = 0.12) -> bool:
    """True if object A is wall-mounted or affixed to supporting structure B."""
    lat_contact = a.lateral_distance_to(b) <= tolerance_m
    vert_overlap = a.z_max >= b.z_min and a.z_min <= b.z_max
    return lat_contact and vert_overlap


def is_near(a: BoundingBox3D, b: BoundingBox3D, threshold_m: float = 1.0) -> bool:
    """True if minimum 3D Euclidean distance between bounding boxes is within threshold."""
    return a.distance_to(b) <= float(threshold_m)


def is_adjacent(a: BoundingBox3D, b: BoundingBox3D, threshold_m: float = 0.35) -> bool:
    """True if objects are immediately adjacent in lateral plane."""
    return a.lateral_distance_to(b) <= float(threshold_m)


def is_blocking(
    a: BoundingBox3D,
    door_box: BoundingBox3D,
    door_zone: DoorKeepClearZone3D | None = None,
    threshold_m: float = 1.0,
) -> bool:
    """
    True if object A obstructs doorway egress.
    If a 3D keep-clear zone is provided, evaluates exact half-cylinder penetration.
    Otherwise uses conservative lateral threshold and vertical overlap.
    """
    if door_zone is not None:
        intersects, _, _ = door_zone.intersects_box(a)
        return intersects

    lat_dist = a.lateral_distance_to(door_box)
    if lat_dist > threshold_m:
        return False
    vert_overlap = min(a.z_max, door_box.z_max) >= max(a.z_min, door_box.z_min)
    return vert_overlap


def is_contained_in_polygon(x: float, y: float, polygon_xy: Sequence[Sequence[float]] | Sequence[tuple[float, float]]) -> bool:
    """Point-in-polygon ray casting algorithm."""
    n = len(polygon_xy)
    if n < 3:
        return False
    inside = False
    p1x, p1y = polygon_xy[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon_xy[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


# ── Scene Graph Data Structures (ConceptGraphs Open-Vocabulary Schema) ─────────

@dataclass
class SceneNode:
    node_id: int
    class_name: str
    instance_id: int
    centroid: tuple[float, float, float]
    extents: tuple[float, float, float]
    bbox: BoundingBox3D
    surface_points_count: int = 0
    hit_count: int = 1
    confidence: float = 1.0
    consensus_score: float = 1.0
    is_movable: bool = False
    is_combustible: bool = False
    is_ignition_source: bool = False
    is_emergency_asset: bool = False
    elevation_tier: str = "floor"  # floor, low, counter, wall, ceiling

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "class_name": self.class_name,
            "instance_id": self.instance_id,
            "centroid": [round(c, 3) for c in self.centroid],
            "extents": [round(e, 3) for e in self.extents],
            "bbox": {
                "x_min": round(self.bbox.x_min, 3), "x_max": round(self.bbox.x_max, 3),
                "y_min": round(self.bbox.y_min, 3), "y_max": round(self.bbox.y_max, 3),
                "z_min": round(self.bbox.z_min, 3), "z_max": round(self.bbox.z_max, 3),
            },
            "surface_points_count": self.surface_points_count,
            "hit_count": self.hit_count,
            "confidence": round(self.confidence, 3),
            "consensus_score": round(self.consensus_score, 3),
            "attributes": {
                "is_movable": self.is_movable,
                "is_combustible": self.is_combustible,
                "is_ignition_source": self.is_ignition_source,
                "is_emergency_asset": self.is_emergency_asset,
                "elevation_tier": self.elevation_tier,
            },
        }


@dataclass
class SceneEdge:
    source_id: int
    source_class: str
    target_id: int
    target_class: str
    predicate: str
    category: str                       # topological, directional, proximity, functional_safety
    metrics: dict[str, float]           # distance_m, lateral_distance_m, vertical_delta_m, overlap_volume_m3
    is_violation: bool = False
    evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_class": self.source_class,
            "target_id": self.target_id,
            "target_class": self.target_class,
            "predicate": self.predicate,
            "category": self.category,
            "metrics": {k: round(v, 3) for k, v in self.metrics.items()},
            "is_violation": self.is_violation,
            "evidence": self.evidence,
        }


# ── Spatial Scene Graph Engine ────────────────────────────────────────────────

class SpatialSceneGraph:
    """
    Constructs and maintains the 3D Qualitative Spatial Scene Graph.
    Integrates nodes (landmarks) and deterministic QSR edges with hospital safety rules.
    """

    def __init__(
        self,
        landmarks: list[dict[str, Any]],
        floor_elevation_z: float = 0.05,
        navigable_zones: list[dict[str, Any]] | None = None,
        camera_poses: list[tuple[float, float, float]] | None = None,
    ) -> None:
        self.raw_landmarks = landmarks
        self.floor_elevation_z = float(floor_elevation_z)
        self.navigable_zones = navigable_zones or []
        self.camera_poses = camera_poses or []

        self.nodes: dict[int, SceneNode] = {}
        self.door_zones: dict[int, DoorKeepClearZone3D] = {}
        self.edges: list[SceneEdge] = []

        self._build_nodes()
        self._build_door_zones()

    def _build_nodes(self) -> None:
        for lm in self.raw_landmarks:
            lid = int(lm["landmark_id"])
            cname = str(lm["class_name"]).lower().strip()
            cx, cy, cz = float(lm["X"]), float(lm["Y"]), float(lm["Z"])
            rw = float(lm.get("measured_width") or 0.45)
            rd = float(lm.get("measured_depth") or rw)
            rh = float(lm.get("measured_height") or 0.60)

            # Surface points
            surf_raw = lm.get("surface_points_json")
            surf_count = 0
            if surf_raw:
                try:
                    pts = json.loads(surf_raw)
                    surf_count = len(pts)
                except Exception:
                    surf_count = 0

            bbox = BoundingBox3D.from_center_extents(cx, cy, cz, depth=rd, width=rw, height=rh)

            # Elevation tier
            if cz <= (self.floor_elevation_z + 0.25):
                tier = "floor"
            elif cz <= 0.85:
                tier = "low"
            elif cz <= 1.40:
                tier = "counter"
            elif cz <= 2.20:
                tier = "wall"
            else:
                tier = "ceiling"

            self.nodes[lid] = SceneNode(
                node_id=lid,
                class_name=lm["class_name"],
                instance_id=int(lm.get("instance_id") or lid),
                centroid=(cx, cy, cz),
                extents=(rd, rw, rh),
                bbox=bbox,
                surface_points_count=surf_count,
                hit_count=int(lm.get("hit_count") or 1),
                confidence=float(lm.get("mean_confidence") or 1.0),
                consensus_score=float(lm.get("consensus_score") or 1.0),
                is_movable=cname in MOVABLE_OBSTACLE_CLASSES,
                is_combustible=cname in COMBUSTIBLE_CLASSES,
                is_ignition_source=cname in IGNITION_CLASSES,
                is_emergency_asset=cname in EMERGENCY_ASSET_CLASSES,
                elevation_tier=tier,
            )

    def _build_door_zones(self) -> None:
        """
        Builds Andrea's 3D half-cylinder keep-clear egress zones for all door landmarks.
        Infers the corridor-facing egress normal from the navigable floor polygon or camera trajectory.
        """
        # Calculate corridor center from navigable floor polygons if available
        corridor_center = None
        if self.navigable_zones:
            all_pts = []
            for z in self.navigable_zones:
                poly = z.get("polygon")
                if poly and isinstance(poly, list):
                    all_pts.extend(poly)
            if all_pts:
                pts_arr = np.asarray(all_pts, dtype=np.float32)
                corridor_center = (float(np.mean(pts_arr[:, 0])), float(np.mean(pts_arr[:, 1])))

        # 1. Identify all exit sign landmarks
        exit_sign_nodes = [
            n for n in self.nodes.values()
            if "exit_sign" in n.class_name.lower() or "fire_exit" in n.class_name.lower()
        ]

        for node in self.nodes.values():
            if "door" not in node.class_name.lower():
                continue

            cx, cy, cz = node.centroid
            rw = node.extents[1]  # measured width
            rh = node.extents[2]  # measured height

            # Check if this door has an exit sign above or on top of it (Emergency Fire Exit)
            is_fire_exit = False
            matched_sign_id = None
            for sign in exit_sign_nodes:
                dx = sign.centroid[0] - cx
                dy = sign.centroid[1] - cy
                lat_dist = math.hypot(dx, dy)
                vert_dz = sign.centroid[2] - cz
                # Exit sign must be within lateral reach of the door opening and at/above door height
                if lat_dist <= max(2.0, rw + 1.0) and vert_dz >= -0.25:
                    is_fire_exit = True
                    matched_sign_id = sign.node_id
                    break

            # Check if landmark votes explicitly classified this door as an exit
            if not is_fire_exit:
                for lm_raw in self.raw_landmarks:
                    if int(lm_raw.get("landmark_id", -1)) == node.node_id:
                        votes = lm_raw.get("class_votes_json") or ""
                        if "exit" in votes.lower():
                            is_fire_exit = True
                            break

            # ONLY activate keep-clear egress zone for Emergency Fire Exit doors!
            if not is_fire_exit:
                continue

            # Determine egress normal facing into the corridor
            if corridor_center is not None:
                dx = corridor_center[0] - cx
                dy = corridor_center[1] - cy
                dist = math.hypot(dx, dy)
                normal = (dx / dist, dy / dist) if dist > 1e-4 else (0.0, 1.0)
            elif self.camera_poses:
                poses_arr = np.asarray([(p[0], p[1]) for p in self.camera_poses], dtype=np.float32)
                diff = poses_arr - np.array([cx, cy], dtype=np.float32)
                dists = np.hypot(diff[:, 0], diff[:, 1])
                nearest_idx = int(np.argmin(dists))
                nearest_p = poses_arr[nearest_idx]
                dx = float(nearest_p[0] - cx)
                dy = float(nearest_p[1] - cy)
                dist = math.hypot(dx, dy)
                normal = (dx / dist, dy / dist) if dist > 1e-4 else (0.0, 1.0)
            else:
                normal = (0.0, 1.0)

            z_bottom = max(self.floor_elevation_z, cz - rh / 2.0)
            z_top = cz + rh / 2.0

            self.door_zones[node.node_id] = DoorKeepClearZone3D(
                door_id=node.node_id,
                center=(cx, cy, cz),
                width=max(0.75, rw),
                height=max(1.80, rh),
                radius=1.00,  # Andrea's fire door rule: 1.0m keep-clear arc
                egress_normal=normal,
                z_min=z_bottom,
                z_max=z_top,
                required_clear_width=0.85,
                is_fire_exit=True,
                status="CLEAR",
                color_hex="#2ecc71",
            )

    def compute_spatial_edges(self, max_pair_dist_m: float = 2.5) -> list[SceneEdge]:
        """
        Deterministically evaluates all active QSR relations between nearby landmark pairs.
        Categorizes relationships into topological, directional, proximity, and functional safety.
        """
        self.edges = []
        node_list = list(self.nodes.values())
        n = len(node_list)

        for i in range(n):
            node_a = node_list[i]
            box_a = node_a.bbox
            id_a = node_a.node_id
            cls_a = node_a.class_name.lower()

            for j in range(i + 1, n):
                node_b = node_list[j]
                box_b = node_b.bbox
                id_b = node_b.node_id
                cls_b = node_b.class_name.lower()

                dist_3d = box_a.distance_to(box_b)
                if dist_3d > max_pair_dist_m:
                    continue

                lat_dist = box_a.lateral_distance_to(box_b)
                vert_delta = node_a.centroid[2] - node_b.centroid[2]
                inter_vol = box_a.intersection_volume(box_b)

                base_metrics = {
                    "distance_m": dist_3d,
                    "lateral_distance_m": lat_dist,
                    "vertical_delta_m": vert_delta,
                    "overlap_volume_m3": inter_vol,
                }

                # ── 1. Topological Contact & Support ─────────────────────────
                if touches(box_a, box_b):
                    self.edges.append(SceneEdge(
                        source_id=id_a, source_class=node_a.class_name,
                        target_id=id_b, target_class=node_b.class_name,
                        predicate="touches", category="topological",
                        metrics=base_metrics,
                        evidence=f"3D bbox distance {dist_3d:.2f}m <= 0.06m margin",
                    ))

                if is_on_top_of(box_a, box_b):
                    self.edges.append(SceneEdge(
                        source_id=id_a, source_class=node_a.class_name,
                        target_id=id_b, target_class=node_b.class_name,
                        predicate="on_top_of", category="topological",
                        metrics=base_metrics,
                        evidence=f"{node_a.class_name} base at z={box_a.z_min:.2f}m rests on {node_b.class_name} top z={box_b.z_max:.2f}m",
                    ))
                    self.edges.append(SceneEdge(
                        source_id=id_b, source_class=node_b.class_name,
                        target_id=id_a, target_class=node_a.class_name,
                        predicate="supports", category="topological",
                        metrics=base_metrics,
                        evidence=f"{node_b.class_name} provides vertical support to {node_a.class_name}",
                    ))
                elif is_on_top_of(box_b, box_a):
                    self.edges.append(SceneEdge(
                        source_id=id_b, source_class=node_b.class_name,
                        target_id=id_a, target_class=node_a.class_name,
                        predicate="on_top_of", category="topological",
                        metrics=base_metrics,
                        evidence=f"{node_b.class_name} base at z={box_b.z_min:.2f}m rests on {node_a.class_name} top z={box_a.z_max:.2f}m",
                    ))
                    self.edges.append(SceneEdge(
                        source_id=id_a, source_class=node_a.class_name,
                        target_id=id_b, target_class=node_b.class_name,
                        predicate="supports", category="topological",
                        metrics=base_metrics,
                        evidence=f"{node_a.class_name} provides vertical support to {node_b.class_name}",
                    ))

                if is_contained_in(box_a, box_b):
                    self.edges.append(SceneEdge(
                        source_id=id_a, source_class=node_a.class_name,
                        target_id=id_b, target_class=node_b.class_name,
                        predicate="inside", category="topological",
                        metrics=base_metrics,
                        evidence=f"{node_a.class_name} 3D bbox fully contained within {node_b.class_name}",
                    ))
                elif is_contained_in(box_b, box_a):
                    self.edges.append(SceneEdge(
                        source_id=id_b, source_class=node_b.class_name,
                        target_id=id_a, target_class=node_a.class_name,
                        predicate="inside", category="topological",
                        metrics=base_metrics,
                        evidence=f"{node_b.class_name} 3D bbox fully contained within {node_a.class_name}",
                    ))

                # ── 2. Functional Safety & Emergency Exit Obstruction ────────
                # Check A blocking emergency fire door B
                if "door" in cls_b and node_a.is_movable:
                    dzone = self.door_zones.get(id_b)
                    if dzone is not None:
                        blocked, pen_m, def_m = dzone.intersects_box(box_a)
                        if blocked:
                            dzone.is_obstructed = True
                            dzone.status = "OBSTRUCTED"
                            dzone.color_hex = "#e74c3c"
                            dzone.obstructing_items.append({
                                "obstacle_id": id_a,
                                "obstacle_class": node_a.class_name,
                                "intrusion_depth_m": pen_m,
                                "deficit_m": def_m,
                            })
                            metrics = dict(base_metrics)
                            metrics["intrusion_depth_m"] = pen_m
                            metrics["clear_width_deficit_m"] = def_m
                            self.edges.append(SceneEdge(
                                source_id=id_a, source_class=node_a.class_name,
                                target_id=id_b, target_class=node_b.class_name,
                                predicate="is_blocking", category="functional_safety",
                                metrics=metrics, is_violation=True,
                                evidence=f"Obstacle intrudes {pen_m:.2f}m into 1.0m emergency fire exit keep-clear zone (width deficit: {def_m:.2f}m)",
                            ))

                # Check B blocking emergency fire door A
                if "door" in cls_a and node_b.is_movable:
                    dzone = self.door_zones.get(id_a)
                    if dzone is not None:
                        blocked, pen_m, def_m = dzone.intersects_box(box_b)
                        if blocked:
                            dzone.is_obstructed = True
                            dzone.status = "OBSTRUCTED"
                            dzone.color_hex = "#e74c3c"
                            dzone.obstructing_items.append({
                                "obstacle_id": id_b,
                                "obstacle_class": node_b.class_name,
                                "intrusion_depth_m": pen_m,
                                "deficit_m": def_m,
                            })
                            metrics = dict(base_metrics)
                            metrics["intrusion_depth_m"] = pen_m
                            metrics["clear_width_deficit_m"] = def_m
                            self.edges.append(SceneEdge(
                                source_id=id_b, source_class=node_b.class_name,
                                target_id=id_a, target_class=node_a.class_name,
                                predicate="is_blocking", category="functional_safety",
                                metrics=metrics, is_violation=True,
                                evidence=f"Obstacle intrudes {pen_m:.2f}m into 1.0m emergency fire exit keep-clear zone (width deficit: {def_m:.2f}m)",
                            ))

                # ── 3. Combustible Near Ignition Hazard ───────────────────────
                if (node_a.is_combustible and node_b.is_ignition_source) or (node_b.is_combustible and node_a.is_ignition_source):
                    comb_node = node_a if node_a.is_combustible else node_b
                    ign_node = node_b if node_a.is_combustible else node_a
                    if dist_3d < 0.60:
                        self.edges.append(SceneEdge(
                            source_id=comb_node.node_id, source_class=comb_node.class_name,
                            target_id=ign_node.node_id, target_class=ign_node.class_name,
                            predicate="hazardous_near", category="functional_safety",
                            metrics=base_metrics, is_violation=True,
                            evidence=f"Combustible waste located {dist_3d:.2f}m (<0.60m) from ignition source",
                        ))

                # ── 4. Directional Halfspaces ─────────────────────────────────
                if is_above(box_a, box_b):
                    self.edges.append(SceneEdge(
                        source_id=id_a, source_class=node_a.class_name,
                        target_id=id_b, target_class=node_b.class_name,
                        predicate="above", category="directional",
                        metrics=base_metrics,
                        evidence=f"z_min ({box_a.z_min:.2f}m) >= z_max ({box_b.z_max:.2f}m)",
                    ))
                    self.edges.append(SceneEdge(
                        source_id=id_b, source_class=node_b.class_name,
                        target_id=id_a, target_class=node_a.class_name,
                        predicate="below", category="directional",
                        metrics=base_metrics,
                        evidence=f"z_max ({box_b.z_max:.2f}m) <= z_min ({box_a.z_min:.2f}m)",
                    ))
                elif is_above(box_b, box_a):
                    self.edges.append(SceneEdge(
                        source_id=id_b, source_class=node_b.class_name,
                        target_id=id_a, target_class=node_a.class_name,
                        predicate="above", category="directional",
                        metrics=base_metrics,
                        evidence=f"z_min ({box_b.z_min:.2f}m) >= z_max ({box_a.z_max:.2f}m)",
                    ))
                    self.edges.append(SceneEdge(
                        source_id=id_a, source_class=node_a.class_name,
                        target_id=id_b, target_class=node_b.class_name,
                        predicate="below", category="directional",
                        metrics=base_metrics,
                        evidence=f"z_max ({box_a.z_max:.2f}m) <= z_min ({box_b.z_min:.2f}m)",
                    ))

                # ── 5. Proximity ──────────────────────────────────────────────
                if is_adjacent(box_a, box_b, threshold_m=0.35):
                    self.edges.append(SceneEdge(
                        source_id=id_a, source_class=node_a.class_name,
                        target_id=id_b, target_class=node_b.class_name,
                        predicate="adjacent", category="proximity",
                        metrics=base_metrics,
                        evidence=f"Lateral distance {lat_dist:.2f}m <= 0.35m",
                    ))
                elif is_near(box_a, box_b, threshold_m=1.0):
                    self.edges.append(SceneEdge(
                        source_id=id_a, source_class=node_a.class_name,
                        target_id=id_b, target_class=node_b.class_name,
                        predicate="near", category="proximity",
                        metrics=base_metrics,
                        evidence=f"3D Euclidean distance {dist_3d:.2f}m <= 1.0m",
                    ))

        return self.edges

    def export_json(self, out_path: Path | None = None) -> dict[str, Any]:
        """Export full 3D Scene Graph representation to ConceptGraphs-compatible JSON format."""
        graph_dict = {
            "schema_version": "2.0.0",
            "graph_type": "3D_Qualitative_Spatial_Scene_Graph",
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
            "door_keep_clear_zones": [
                {
                    "door_id": dz.door_id,
                    "center": [round(c, 3) for c in dz.center],
                    "width_m": round(dz.width, 3),
                    "height_m": round(dz.height, 3),
                    "radius_m": round(dz.radius, 3),
                    "egress_normal": [round(n, 3) for n in dz.egress_normal],
                    "z_bounds": [round(dz.z_min, 3), round(dz.z_max, 3)],
                    "required_clear_width_m": dz.required_clear_width,
                    "is_fire_exit": dz.is_fire_exit,
                    "is_obstructed": dz.is_obstructed,
                    "status": dz.status,
                    "color_hex": dz.color_hex,
                    "arc_samples_3d": dz.sample_arc_3d(num_samples=20),
                    "cage_samples_3d": dz.sample_cylinder_cage_3d(num_samples=20),
                }
                for dz in self.door_zones.values()
            ],
        }
        if out_path is not None:
            out_path = Path(out_path).resolve()
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(graph_dict, f, indent=2)
        return graph_dict

    def persist_to_sqlite(self, db_path: Path) -> int:
        """Persist all scene graph edges into world_map.db under 'spatial_edges' table."""
        db_path = Path(db_path).resolve()
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS spatial_edges (
                edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER,
                source_class TEXT,
                target_id INTEGER,
                target_class TEXT,
                predicate TEXT,
                category TEXT,
                distance_m REAL,
                lateral_distance_m REAL,
                is_violation INTEGER,
                metrics_json TEXT,
                evidence TEXT,
                FOREIGN KEY(source_id) REFERENCES semantic_map(landmark_id),
                FOREIGN KEY(target_id) REFERENCES semantic_map(landmark_id)
            )
        """)
        cur.execute("DELETE FROM spatial_edges")

        rows = [
            (
                e.source_id, e.source_class,
                e.target_id, e.target_class,
                e.predicate, e.category,
                e.metrics.get("distance_m", 0.0),
                e.metrics.get("lateral_distance_m", 0.0),
                1 if e.is_violation else 0,
                json.dumps(e.metrics),
                e.evidence,
            )
            for e in self.edges
        ]
        cur.executemany("""
            INSERT INTO spatial_edges (
                source_id, source_class, target_id, target_class,
                predicate, category, distance_m, lateral_distance_m,
                is_violation, metrics_json, evidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)

        conn.commit()
        conn.close()
        return len(rows)
