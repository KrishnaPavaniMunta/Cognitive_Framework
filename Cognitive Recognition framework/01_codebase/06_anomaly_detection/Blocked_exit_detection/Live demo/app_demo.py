"""Hospital Exit Safety Dashboard.

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

st.set_page_config(
    page_title="Hospital Exit Safety Monitor",
    page_icon="🚪",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Clean, modern, impressive dark styling
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        padding-left: clamp(1rem, 3vw, 2.5rem);
        padding-right: clamp(1rem, 3vw, 2.5rem);
        max-width: 100% !important;
    }
    .header-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 18px 24px;
        text-align: center;
        margin-bottom: 16px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
    }
    .viewer-frame {
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        overflow: hidden;
        background: #0f172a;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    }
    .status-clear {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid #10b981;
        border-radius: 10px;
        padding: 16px;
    }
    .status-blocked {
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid #ef4444;
        border-radius: 10px;
        padding: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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
    
    landmarks = [
        {"class_name": r[1], "instance_id": r[2], "X": r[3], "Y": r[4], "Z": r[5]}
        for r in conn.execute("SELECT landmark_id, class_name, instance_id, X, Y, Z FROM semantic_map")
    ]
    
    door_event = conn.execute(
        """SELECT frame_index, door_world_X, door_world_Y, door_world_Z, 
                  door_top_cam_Y, door_bottom_cam_Y, zone_radius_m 
           FROM egress_obstruction_events WHERE door_world_X IS NOT NULL 
           ORDER BY frame_index DESC LIMIT 1"""
    ).fetchone()
    
    matrix_json = None
    if door_event:
        frame_idx = door_event[0]
        pose_row = conn.execute("SELECT matrix_json FROM camera_poses WHERE frame_index=?", (frame_idx,)).fetchone()
        if pose_row:
            matrix_json = pose_row[0]
            
    conn.close()
    return landmarks, door_event, matrix_json


def get_door_vectors(matrix_json: str):
    """Extract forward and right vectors from the camera pose matrix."""
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
            
        if not (min(bottom_z, top_z) <= lm["Z"] <= max(bottom_z, top_z)):
            continue
            
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


@st.cache_resource(show_spinner="Loading 3D world map...")
def start_rerun_server(rrd_path: Path) -> tuple[rr.RecordingStream, str]:
    """Boot one persistent recording+server using the exact recording ID from the file."""
    app_id, rec_id = get_rrd_recording_info(rrd_path)
    stream = rr.RecordingStream(
        application_id=app_id,
        recording_id=rec_id,
    )
    grpc_uri = stream.serve_grpc(grpc_port=9877, server_memory_limit="8GiB")
    stream.log_file_from_path(str(rrd_path))
    rr.serve_web_viewer(web_port=9090, open_browser=False, connect_to=grpc_uri)
    return stream, grpc_uri


def main():
    run_dir = newest_run_dir()
    if not run_dir:
        st.error(f"No semantic map runs found in {OUTPUT_ROOT}")
        st.stop()

    rrd_path = run_dir / "world_map.rrd"
    db_path = run_dir / "world_map.db"
    
    if not rrd_path.exists():
        st.error(f"world_map.rrd not found in {run_dir}.")
        st.stop()

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

    # Reuse persistent stream
    stream, grpc_uri = start_rerun_server(rrd_path)
    viewer_url = f"http://localhost:9090/?url={quote(grpc_uri, safe='')}"

    # ── Top Title Header ──────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="header-card">
            <h1 style="color: #f8fafc; font-size: 26px; font-weight: 700; margin: 0 0 6px 0;">
                Hospital Hallway: Exit Door Safety Monitor
            </h1>
            <p style="color: #94a3b8; font-size: 15px; margin: 0;">
                Live 3D map showing whether the emergency exit is clear or blocked by obstacles.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── 3D Viewer ─────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <span style="font-weight: 600; font-size: 15px; color: #e2e8f0;">📹 3D Map & Camera View</span>
            <span style="font-size: 12px; color: #94a3b8;">Left-click to rotate &bull; Right-click to pan &bull; Scroll to zoom</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown('<div class="viewer-frame">', unsafe_allow_html=True)
    st.iframe(viewer_url, height=580, width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # ── Bottom Row: Clean Two-Column Controls & Status ─────────────────────────
    col_alert, col_slider = st.columns([1.2, 1.0], gap="medium")

    with col_slider:
        with st.container(border=True):
            st.markdown("<h3 style='margin-top:0; font-size: 19px;'>📏 Required Clear Distance</h3>", unsafe_allow_html=True)
            st.write("Adjust the required safety distance in front of the door:")
            
            custom_radius = st.slider(
                "Clearance distance (meters)", 
                min_value=0.40, max_value=3.50, 
                value=float(round(detected_radius, 2)), step=0.05,
                help="Move this slider to change how much space must remain clear in front of the door.",
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
        static=True,
    )

    with col_alert:
        with st.container(border=True):
            st.markdown("<h3 style='margin-top:0; font-size: 19px;'>🛡️ Exit Door Status</h3>", unsafe_allow_html=True)
            
            if hits:
                blocker_names = ", ".join([f"**{h['class_name'].replace('_', ' ').title()} #{h['instance_id']}**" for h in hits])
                st.markdown(
                    f"""
                    <div class="status-blocked">
                        <div style="color: #ef4444; font-size: 18px; font-weight: 700; margin-bottom: 6px;">
                            🚨 Exit is Blocked!
                        </div>
                        <div style="color: #fca5a5; font-size: 14px;">
                            <b>{len(hits)} obstacle(s)</b> are inside the <b>{custom_radius:.2f} meter</b> safety zone:
                            <br><br>
                            {blocker_names}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="status-clear">
                        <div style="color: #10b981; font-size: 18px; font-weight: 700; margin-bottom: 6px;">
                            ✅ Exit is Clear
                        </div>
                        <div style="color: #86efac; font-size: 14px;">
                            No obstacles are blocking the exit door within <b>{custom_radius:.2f} meters</b>.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


if __name__ == "__main__":
    main()