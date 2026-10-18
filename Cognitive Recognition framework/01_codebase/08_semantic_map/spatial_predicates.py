"""
Formal Qualitative Spatial Reasoning (QSR) Predicates for 3D Semantic Maps.

Based on the spatial formalisms from the Knowledge Media Institute (KMi, Open University)
and Chiatti (2022), operationalized for 3D robotics and digital twins.

Defines directional halfspaces (Above, Below, InFrontOf, Behind, RightOf, LeftOf)
with configurable noise-tolerant expansions (+/- 8cm) and topological contact predicates
(Touches, OnTopOf, AffixedTo, IsBlocking, Near, Contains).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable


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
            self.x_max - self.x_min,
            self.y_max - self.y_min,
            self.z_max - self.z_min,
        )

    @property
    def volume(self) -> float:
        dx, dy, dz = self.extents
        return max(0.0, dx * dy * dz)

    def distance_to(self, other: BoundingBox3D) -> float:
        """Minimum Euclidean distance between two 3D axis-aligned bounding boxes."""
        dx = max(0.0, self.x_min - other.x_max, other.x_min - self.x_max)
        dy = max(0.0, self.y_min - other.y_max, other.y_min - self.y_max)
        dz = max(0.0, self.z_min - other.z_max, other.z_min - self.z_max)
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def lateral_distance_to(self, other: BoundingBox3D) -> float:
        """Minimum 2D Euclidean distance on the ground (XY) plane."""
        dx = max(0.0, self.x_min - other.x_max, other.x_min - self.x_max)
        dy = max(0.0, self.y_min - other.y_max, other.y_min - self.y_max)
        return math.sqrt(dx * dx + dy * dy)

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


# ── Directional Halfspace Predicates (Section III-B in Ibba et al.) ────────────

def is_above(a: BoundingBox3D, b: BoundingBox3D, tolerance_m: float = 0.08) -> bool:
    """True if bounding box A intersects the upward Z+ halfspace of B."""
    return a.z_max >= (b.z_max - tolerance_m) and a.z_min >= (b.z_min - tolerance_m)


def is_below(a: BoundingBox3D, b: BoundingBox3D, tolerance_m: float = 0.08) -> bool:
    """True if bounding box A intersects the downward Z- halfspace of B."""
    return a.z_min <= (b.z_min + tolerance_m) and a.z_max <= (b.z_max + tolerance_m)


def is_in_front_of(a: BoundingBox3D, b: BoundingBox3D, tolerance_m: float = 0.08) -> bool:
    """True if A falls within the forward-facing (+Y in ROS optical or +X in robot frame)."""
    return a.x_max >= (b.x_max - tolerance_m)


def is_behind(a: BoundingBox3D, b: BoundingBox3D, tolerance_m: float = 0.08) -> bool:
    """True if A falls behind B along the longitudinal axis."""
    return a.x_min <= (b.x_min + tolerance_m)


def is_right_of(a: BoundingBox3D, b: BoundingBox3D, tolerance_m: float = 0.08) -> bool:
    """True if A falls to the right of B along lateral axis."""
    return a.y_min <= (b.y_min + tolerance_m)


def is_left_of(a: BoundingBox3D, b: BoundingBox3D, tolerance_m: float = 0.08) -> bool:
    """True if A falls to the left of B along lateral axis."""
    return a.y_max >= (b.y_max - tolerance_m)


# ── Topological & Contact Predicates (Table I in Ibba et al.) ──────────────────

def touches(a: BoundingBox3D, b: BoundingBox3D, tolerance_m: float = 0.06) -> bool:
    """
    True if objects A and B touch or are physically adjacent.
    Uses distance tolerance (equivalent to ST_3DDWithin in PostGIS) to absorb depth noise.
    """
    return a.distance_to(b) <= tolerance_m


def is_on_top_of(a: BoundingBox3D, b: BoundingBox3D, tolerance_m: float = 0.08) -> bool:
    """
    True if object A rests on top of object B.
    Requires:
    1. Horizontal alignment (2D overlap on XY plane or proximity).
    2. Vertical support: bottom of A is at or slightly above top of B.
    """
    # Horizontal footprint check: bounding boxes must align in XY
    horiz_overlap = (
        max(0.0, min(a.x_max, b.x_max) - max(a.x_min, b.x_min)) > -tolerance_m
        and max(0.0, min(a.y_max, b.y_max) - max(a.y_min, b.y_min)) > -tolerance_m
    )
    if not horiz_overlap:
        return False

    # Vertical contact: bottom of A rests near top of B
    vertical_contact = abs(a.z_min - b.z_max) <= tolerance_m
    is_higher = a.z_min >= (b.z_min + 0.05)
    return vertical_contact and is_higher


def is_affixed_to(a: BoundingBox3D, b: BoundingBox3D, tolerance_m: float = 0.12) -> bool:
    """
    True if object A is wall-mounted or affixed to supporting structure B.
    Allows for wall-mounting bracket clearance up to 12cm.
    """
    return a.lateral_distance_to(b) <= tolerance_m and (a.z_max >= b.z_min and a.z_min <= b.z_max)


def is_near(a: BoundingBox3D, b: BoundingBox3D, threshold_m: float = 1.0) -> bool:
    """True if minimum Euclidean distance between bounding boxes is within threshold."""
    return a.distance_to(b) <= float(threshold_m)


def is_blocking(a: BoundingBox3D, door_box: BoundingBox3D, threshold_m: float = 0.90) -> bool:
    """
    True if object A is positioned in front of or adjacent to a doorway,
    obstructing egress or emergency access.
    """
    # 2D ground distance from obstacle to door opening
    lat_dist = a.lateral_distance_to(door_box)
    if lat_dist > threshold_m:
        return False
    # Check vertical overlap (obstacle must be above floor level within door height)
    vert_overlap = min(a.z_max, door_box.z_max) >= max(a.z_min, door_box.z_min)
    return vert_overlap


def is_contained_in_polygon(x: float, y: float, polygon_xy: list[list[float]] | list[tuple[float, float]]) -> bool:
    """Point-in-polygon ray casting algorithm."""
    n = len(polygon_xy)
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


# ── Spatial Scene Graph Builder ───────────────────────────────────────────────

class SpatialSceneGraph:
    """
    Builds a 3D semantic scene graph from a list of landmark records.
    Constructs nodes (landmarks) and computes all active QSR edges between nearby pairs.
    """

    def __init__(self, landmarks: list[dict[str, Any]], floor_elevation_z: float = 0.05) -> None:
        self.landmarks = landmarks
        self.floor_elevation_z = float(floor_elevation_z)
        self.boxes: dict[int, BoundingBox3D] = {}
        self._build_boxes()

    def _build_boxes(self) -> None:
        for lm in self.landmarks:
            lid = int(lm["landmark_id"])
            cx, cy, cz = float(lm["X"]), float(lm["Y"]), float(lm["Z"])
            rw = float(lm.get("measured_width") or 0.45)
            rd = float(lm.get("measured_depth") or rw)
            rh = float(lm.get("measured_height") or 0.60)
            self.boxes[lid] = BoundingBox3D.from_center_extents(cx, cy, cz, depth=rd, width=rw, height=rh)

    def compute_spatial_edges(self, max_pair_dist_m: float = 2.5) -> list[dict[str, Any]]:
        """
        Compute active QSR spatial relations between all nearby pairs of landmarks.
        Returns a list of edge dicts: {'subject_id', 'predicate', 'object_id', 'distance_m'}
        """
        edges: list[dict[str, Any]] = []
        n = len(self.landmarks)

        for i in range(n):
            lm_a = self.landmarks[i]
            id_a = int(lm_a["landmark_id"])
            box_a = self.boxes[id_a]
            class_a = str(lm_a["class_name"]).lower()

            for j in range(i + 1, n):
                lm_b = self.landmarks[j]
                id_b = int(lm_b["landmark_id"])
                box_b = self.boxes[id_b]
                class_b = str(lm_b["class_name"]).lower()

                dist = box_a.distance_to(box_b)
                if dist > max_pair_dist_m:
                    continue

                # 1. Touches
                if touches(box_a, box_b):
                    edges.append({
                        "subject_id": id_a, "subject_class": class_a,
                        "predicate": "touches",
                        "object_id": id_b, "object_class": class_b,
                        "distance_m": round(dist, 3),
                    })

                # 2. OnTopOf
                if is_on_top_of(box_a, box_b):
                    edges.append({
                        "subject_id": id_a, "subject_class": class_a,
                        "predicate": "on_top_of",
                        "object_id": id_b, "object_class": class_b,
                        "distance_m": round(dist, 3),
                    })
                elif is_on_top_of(box_b, box_a):
                    edges.append({
                        "subject_id": id_b, "subject_class": class_b,
                        "predicate": "on_top_of",
                        "object_id": id_a, "object_class": class_a,
                        "distance_m": round(dist, 3),
                    })

                # 3. Blocking (if one is a door)
                if "door" in class_b and is_blocking(box_a, box_b):
                    edges.append({
                        "subject_id": id_a, "subject_class": class_a,
                        "predicate": "is_blocking",
                        "object_id": id_b, "object_class": class_b,
                        "distance_m": round(box_a.lateral_distance_to(box_b), 3),
                    })
                elif "door" in class_a and is_blocking(box_b, box_a):
                    edges.append({
                        "subject_id": id_b, "subject_class": class_b,
                        "predicate": "is_blocking",
                        "object_id": id_a, "object_class": class_a,
                        "distance_m": round(box_b.lateral_distance_to(box_a), 3),
                    })

                # 4. Near
                if is_near(box_a, box_b, threshold_m=1.0):
                    edges.append({
                        "subject_id": id_a, "subject_class": class_a,
                        "predicate": "near",
                        "object_id": id_b, "object_class": class_b,
                        "distance_m": round(dist, 3),
                    })

        return edges
