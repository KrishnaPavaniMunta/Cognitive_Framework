"""
Minimal TF tree resolver for recorded ROS2 bags.

`rosbag_rgbd_sim_capture._choose_extrinsics` matches a single transform by child
frame name, which cannot express `camera_rgb_optical_frame -> ... -> odom`.
This module composes the full chain instead, treating /tf_static transforms as
timeless and /tf transforms as time-indexed.
"""

from __future__ import annotations

from bisect import bisect_left
from typing import Any

import numpy as np

MAX_CHAIN_DEPTH = 32


def normalize_frame(frame_id: str) -> str:
    return str(frame_id).strip().lstrip("/")


class TFTree:
    def __init__(
        self,
        dynamic_records_by_child: dict[str, list[Any]],
        static_records_by_child: dict[str, list[Any]],
        max_delta_ns: int,
    ) -> None:
        self.max_delta_ns = int(max_delta_ns)
        self._dynamic: dict[str, list[Any]] = {}
        self._dynamic_ts: dict[str, list[int]] = {}
        self._static: dict[str, Any] = {}
        self._parent: dict[str, str] = {}

        for child, records in static_records_by_child.items():
            if not records:
                continue
            key = normalize_frame(child)
            self._static[key] = records[-1]
            self._parent.setdefault(key, normalize_frame(records[-1].frame_id))

        for child, records in dynamic_records_by_child.items():
            if not records:
                continue
            key = normalize_frame(child)
            merged = self._dynamic.setdefault(key, [])
            merged.extend(records)
            # A dynamic parent overrides a static one for the same child.
            self._parent[key] = normalize_frame(records[-1].frame_id)

        for key, records in self._dynamic.items():
            records.sort(key=lambda r: r.timestamp_ns)
            self._dynamic_ts[key] = [r.timestamp_ns for r in records]

    @property
    def frames(self) -> list[str]:
        return sorted(set(self._parent) | set(self._parent.values()))

    def resolve_chain(self, source_frame: str, target_frame: str) -> list[str] | None:
        """Frame names walking from source up to target, inclusive. None if unreachable."""
        source = normalize_frame(source_frame)
        target = normalize_frame(target_frame)
        chain = [source]
        current = source

        for _ in range(MAX_CHAIN_DEPTH):
            if current == target:
                return chain
            parent = self._parent.get(current)
            if parent is None or parent in chain:
                return None
            chain.append(parent)
            current = parent
        return None

    def _link_matrix(self, child: str, timestamp_ns: int) -> tuple[np.ndarray | None, str]:
        """Transform mapping points in `child` into its parent frame."""
        timestamps = self._dynamic_ts.get(child)
        if timestamps:
            idx = bisect_left(timestamps, timestamp_ns)
            candidates = [i for i in (idx - 1, idx) if 0 <= i < len(timestamps)]
            if candidates:
                best = min(candidates, key=lambda i: abs(timestamps[i] - timestamp_ns))
                if abs(timestamps[best] - timestamp_ns) <= self.max_delta_ns:
                    return np.asarray(self._dynamic[child][best].matrix_4x4, dtype=np.float64), "dynamic"

        static_record = self._static.get(child)
        if static_record is not None:
            return np.asarray(static_record.matrix_4x4, dtype=np.float64), "static"

        return None, "missing"

    def lookup(
        self, source_frame: str, target_frame: str, timestamp_ns: int
    ) -> tuple[np.ndarray | None, str]:
        """Composed 4x4 mapping points from `source_frame` into `target_frame`."""
        chain = self.resolve_chain(source_frame, target_frame)
        if chain is None:
            return None, f"no_chain:{normalize_frame(source_frame)}->{normalize_frame(target_frame)}"

        matrix = np.eye(4, dtype=np.float64)
        used_dynamic = False
        for child in chain[:-1]:
            link, kind = self._link_matrix(child, timestamp_ns)
            if link is None:
                return None, f"stale_or_missing_link:{child}"
            used_dynamic = used_dynamic or kind == "dynamic"
            matrix = link @ matrix

        return matrix, "ok" if used_dynamic else "ok_static_only"

    def describe_chain(self, source_frame: str, target_frame: str) -> str:
        chain = self.resolve_chain(source_frame, target_frame)
        if chain is None:
            return f"{normalize_frame(source_frame)} -> ??? -> {normalize_frame(target_frame)} (unreachable)"
        parts = []
        for child in chain[:-1]:
            kind = "static" if child in self._static and child not in self._dynamic else "dynamic"
            parts.append(f"{child} --[{kind}]--> ")
        return "".join(parts) + chain[-1]
