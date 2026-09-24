"""Interactive EKAW 2026 Egress Dashboard.

Run:
    streamlit run "01_codebase/06_anomaly_detection/Blocked_exit_detection/Live demo/app_demo.py"
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
from pathlib import Path
from urllib.parse import quote

import numpy as np
import rerun as rr
import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_ROOT = PROJECT_ROOT / "04_outputs_runs_and_logs" / "outputs" / "semantic_maps"
PREFERRED_RUN_DIR = (
    PROJECT_ROOT
    / "04_outputs_runs_and_logs"
    / "conference_demo"
    / "hallway1_door_merged_full_20260923_114938"
    / "hallway_1"
)

st.set_page_config(page_title="Sensemaking Robots: Live Egress Monitor", layout="wide", initial_sidebar_state="collapsed")


def newest_run_dir() -> Path | None:
    """Use the selected full merged-door run, with the old map as fallback."""
    if (PREFERRED_RUN_DIR / "world_map.db").exists() and (PREFERRED_RUN_DIR / "world_map.rrd").exists():
        return PREFERRED_RUN_DIR
    db_files = list(OUTPUT_ROOT.glob("*/world_map.db"))
    if not db_files:
        return None
    return max([p.parent for p in db_files], key=lambda p: p.stat().st_mtime)


@st.cache_data(show_spinner=False)
def load_map_data(db_path: str):
    """Load landmarks, door anchor, and door rotation matrix from SQLite."""
    conn = sqlite3.connect(db_path)
    
    # Load landmarks
    landmarks = [
        {"class_name": r[1], "instance_id": r[2], "X": r[3], "Y": r[4], "Z": r[5]}
        for r in conn.execute("SELECT landmark_id, class_name, instance_id, X, Y, Z FROM semantic_map")
    ]
    
    # Load the definitive door anchor
    door_event = conn.execute(
        """SELECT frame_index, door_world_X, door_world_Y, door_world_Z, 
                  door_top_cam_Y, door_bottom_cam_Y, zone_radius_m 
           FROM egress_obstruction_events WHERE door_world_X IS NOT NULL 
           ORDER BY frame_index DESC LIMIT 1"""
    ).fetchone()
    
    # Load the exact camera pose matrix to align the door zone correctly
    matrix_json = None
    if door_event:
        frame_idx = door_event[0]
        pose_row = conn.execute("SELECT matrix_json FROM camera_poses WHERE frame_index=?", (frame_idx,)).fetchone()
        if pose_row:
            matrix_json = pose_row[0]
            
    conn.close()
    return landmarks, door_event, matrix_json


def get_door_vectors(matrix_json: str):
    """Extract perfect forward and right vectors from the camera pose matrix."""
    matrix = np.array(json.loads(matrix_json))
    right = matrix[:3, :3] @ np.array([1.0, 0.0, 0.0])
    forward = matrix[:3, :3] @ np.array([0.0, 0.0, 1.0])
    right[2] = 0.0
    forward[2] = 0.0
    right /= max(np.linalg.norm(right), 1e-6)
    forward /= max(np.linalg.norm(forward), 1e-6)
    return forward, right


def zone_strips(center: np.ndarray, forward: np.ndarray, right: np.ndarray, radius: float, bottom_z: float, top_z: float):
    """Generate 3D wireframe arcs aligned to the door."""
    floor = np.array([center[0], center[1], bottom_z])
    bottom, top = [], []
    samples = 32
    for index in range(samples + 1):
        theta = math.pi * index / samples
        point = floor + radius * math.cos(theta) * right - radius * math.sin(theta) * forward
        bottom.append(point)
        top.append(point + np.array([0.0, 0.0, top_z - bottom_z]))
    bottom, top = np.asarray(bottom), np.asarray(top)
    return [bottom, top] + [np.stack([bottom[i], top[i]]) for i in range(0, samples + 1, 4)]


def find_blockers(landmarks, center, forward, radius, bottom_z, top_z):
    """Find mapped landmarks inside the dynamic half-cylinder."""
    hits = []
    for lm in landmarks:
        if lm["class_name"] == "door":
            continue
            
        # 1. Height check
        if not (min(bottom_z, top_z) <= lm["Z"] <= max(bottom_z, top_z)):
            continue
            
        # 2. Horizontal geometry check (must be in front of door and within radius)
        dx = lm["X"] - center[0]
        dy = lm["Y"] - center[1]
        dist_2d = math.hypot(dx, dy)
        rel_vec = np.array([dx, dy, 0.0])
        dot_product = np.dot(rel_vec, forward)
        
        if dist_2d <= radius and dot_product <= 0.0:
            hits.append(lm)
            
    return hits


def get_rrd_recording_info(rrd_path: Path) -> tuple[str, str]:
    """Extract the original application_id and recording_id from an .rrd file."""
    try:
        import rerun_bindings as b
        reader = b.RrdReaderInternal(str(rrd_path))
        for entry in reader.store_entries():
            if str(entry.kind).lower() == "recording":
                return entry.application_id, entry.recording_id
    except Exception:
        pass
    return "semantic_map-hallway_126fd", "c2e2619378bd4a5dbd4a1bf8f4338c8b"


@st.cache_resource(show_spinner="Booting Rerun Server with world_map.rrd...")
def start_rerun_server(rrd_path: Path) -> tuple[rr.RecordingStream, str]:
    """Boot one persistent recording+server; using the .rrd file's recording ID ensures only one recording exists."""
    app_id, rec_id = get_rrd_recording_info(rrd_path)
    stream = rr.RecordingStream(
        application_id=app_id,
        recording_id=rec_id,
    )
    # Buffer generously: recorded point-cloud .rrd files can be several hundred MB.
    grpc_uri = stream.serve_grpc(grpc_port=9877, server_memory_limit="8GiB")
    # Replay the recorded landmarks/cloud/camera/trajectory into this same live stream.
    stream.log_file_from_path(str(rrd_path))
    rr.serve_web_viewer(web_port=9090, open_browser=False, connect_to=grpc_uri)
    return stream, grpc_uri


def main():
    st.markdown("<h2 style='text-align: center; margin-top: -30px;'>Neurosymbolic Spatial Monitor</h2>", unsafe_allow_html=True)
    
    run_dir = newest_run_dir()
    if not run_dir:
        st.error(f"No semantic map runs found in {OUTPUT_ROOT}")
        st.stop()

    rrd_path = run_dir / "world_map.rrd"
    db_path = run_dir / "world_map.db"
    
    if not rrd_path.exists():
        st.error(f"world_map.rrd not found in {run_dir}. Did you run the builder with --rerun?")
        st.stop()

    # Load Database state
    landmarks, door_event, matrix_json = load_map_data(str(db_path))
    if not door_event or not matrix_json:
        st.error("No valid door geometry found in this map.")
        st.stop()

    # Calculate base geometry
    center = np.asarray(door_event[1:4], dtype=float)
    detected_radius = float(door_event[6])
    height = abs(float(door_event[5]) - float(door_event[4])) or 2.0
    bottom_z, top_z = center[2] - height / 2.0, center[2] + height / 2.0
    forward, right = get_door_vectors(matrix_json)

    # Reuse the single persistent recording/server across every script rerun; logging
    # straight onto this cached stream avoids re-touching the global default recording,
    # which would otherwise drop the RecordingStream that owns the gRPC server (see
    # rerun.serve_grpc docs) and kill it on every slider move.
    stream, grpc_uri = start_rerun_server(rrd_path)
    viewer_url = f"http://localhost:9090/?url={quote(grpc_uri, safe='')}"

    # ── Top Row: The Rerun Viewer ──────────────────────────────────────────────
    # This natively embeds both your RGB video and 3D map from the .rrd
    st.markdown("#### Live Telemetry & Digital Twin")
    components.iframe(viewer_url, height=550, scrolling=False)
    
    st.divider()

    # ── Bottom Row: Interactive Alert Dashboard ────────────────────────────────
    col_alert, col_slider = st.columns([1.5, 1], gap="large")

    with col_slider:
        st.markdown("#### Dynamic Rule Override")
        custom_radius = st.slider(
            "Keep-Clear Zone Radius (m)", 
            min_value=0.4, max_value=3.5, 
            value=float(round(detected_radius, 2)), step=0.05,
            help="Simulate regulatory rule changes on the fly."
        )

    # Evaluate Collisions
    hits = find_blockers(landmarks, center, forward, custom_radius, bottom_z, top_z)

    # Inject override zone into Rerun instantly
    zone_color = [220, 30, 30] if hits else [40, 190, 80]
    strips = zone_strips(center, forward, right, custom_radius, bottom_z, top_z)

    # static=True forces the zone to update visually across the entire recorded timeline
    stream.log(
        "world/exit_zone",
        rr.LineStrips3D(strips, colors=[zone_color] * len(strips), radii=0.035),
        static=True
    )

    with col_alert:
        st.markdown("#### Autonomous Obstruction Assessment")
        if hits:
            blocker_names = ", ".join([f"**{h['class_name']} {h['instance_id']}**" for h in hits])
            st.error(
                f"### 🚨 REGULATORY VIOLATION\n"
                f"{len(hits)} physical object(s) detected inside the {custom_radius:.2f}m egress envelope: {blocker_names}",
                icon="⚠️"
            )
        else:
            st.success(
                f"### ✅ EGRESS CLEAR\n"
                f"No objects detected within the {custom_radius:.2f}m safety envelope.",
                icon="🛡️"
            )

if __name__ == "__main__":
    main()