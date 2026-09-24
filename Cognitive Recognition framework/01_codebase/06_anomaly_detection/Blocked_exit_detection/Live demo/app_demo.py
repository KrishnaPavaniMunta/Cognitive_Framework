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
import pandas as pd
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
    page_title="Sensemaking Robots: Live Egress Spatial Twin",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for dark glassmorphism, refined typography, and sleek cards
st.markdown(
    """
    <style>
    /* Global layout tightening */
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 98% !important;
    }
    
    /* Header card */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px 24px;
        margin-bottom: 14px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    
    /* Sleek badge pill */
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 11px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.3px;
        text-transform: uppercase;
    }
    .badge-green {
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.35);
    }
    .badge-blue {
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.35);
    }
    .badge-purple {
        background: rgba(168, 85, 247, 0.15);
        color: #c084fc;
        border: 1px solid rgba(168, 85, 247, 0.35);
    }
    
    /* Card containers */
    .control-card {
        background: #0f172a;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 14px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    
    /* Viewer wrapper */
    .viewer-frame {
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        overflow: hidden;
        background: #0b0f19;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
    }
    
    /* Metric overrides */
    div[data-testid="stMetric"] {
        background: #0f172a;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 10px 14px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 12px !important;
        font-weight: 500 !important;
        color: #94a3b8 !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 20px !important;
        font-weight: 700 !important;
        color: #f8fafc !important;
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
    
    # Load landmarks with hit counts and confidences
    landmarks = [
        {
            "landmark_id": r[0],
            "class_name": r[1],
            "instance_id": r[2],
            "X": r[3],
            "Y": r[4],
            "Z": r[5],
            "hit_count": r[6] if len(r) > 6 else 1,
            "confidence": r[7] if len(r) > 7 else 1.0,
        }
        for r in conn.execute(
            "SELECT landmark_id, class_name, instance_id, X, Y, Z, hit_count, mean_confidence FROM semantic_map"
        )
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
    """Extract forward and right vectors from camera pose matrix."""
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


def find_blockers_and_proximity(landmarks, center, forward, radius, bottom_z, top_z):
    """Find mapped landmarks inside the dynamic half-cylinder and compute proximities."""
    hits = []
    proximity_list = []
    
    for lm in landmarks:
        if lm["class_name"] == "door":
            continue
            
        dx = lm["X"] - center[0]
        dy = lm["Y"] - center[1]
        dist_2d = math.hypot(dx, dy)
        rel_vec = np.array([dx, dy, 0.0])
        dot_product = np.dot(rel_vec, forward)
        
        in_height = min(bottom_z, top_z) <= lm["Z"] <= max(bottom_z, top_z)
        in_front = dot_product <= 0.0
        is_hit = in_height and in_front and (dist_2d <= radius)
        
        if is_hit:
            hits.append(lm)
            
        proximity_list.append({
            "landmark": lm,
            "dist_2d": dist_2d,
            "in_front": in_front,
            "in_height": in_height,
            "is_hit": is_hit,
        })
        
    proximity_list.sort(key=lambda x: x["dist_2d"])
    return hits, proximity_list


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


@st.cache_resource(show_spinner="Booting Rerun Digital Twin with world_map.rrd...")
def start_rerun_server(rrd_path: Path) -> tuple[rr.RecordingStream, str]:
    """Boot one persistent recording+server matching the .rrd's exact recording ID."""
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

    # Initialize session state for the slider
    if "radius_slider" not in st.session_state:
        st.session_state.radius_slider = float(round(detected_radius, 2))

    # Reuse the single persistent stream
    stream, grpc_uri = start_rerun_server(rrd_path)
    viewer_url = f"http://localhost:9090/?url={quote(grpc_uri, safe='')}"

    # ── Header Banner ────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="hero-banner">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px;">
                <div>
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 26px;">🛡️</span>
                        <h2 style="margin: 0; color: #f8fafc; font-size: 22px; font-weight: 700; letter-spacing: -0.4px;">
                            Sensemaking Robots &bull; Live Egress Spatial Monitor
                        </h2>
                    </div>
                    <div style="margin: 4px 0 0 36px; color: #94a3b8; font-size: 13px;">
                        EKAW 2026 Interactive Demonstration &bull; Real-time Neurosymbolic Anomaly Detection &bull; Hospital Hallway 1
                    </div>
                </div>
                <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                    <span class="badge-pill badge-green">● gRPC Synchronized</span>
                    <span class="badge-pill badge-blue">SOSA/SSN Ontology</span>
                    <span class="badge-pill badge-purple">Spatial Twin</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Top KPI Metric Bar ───────────────────────────────────────────────────
    # Temporary evaluate to get current blocker count for top metrics
    temp_hits, proximity_list = find_blockers_and_proximity(
        landmarks, center, forward, st.session_state.radius_slider, bottom_z, top_z
    )

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric(
            label="Monitored Door Anchor",
            value="Door #1",
            delta=f"Z: {center[2]:.2f}m",
            delta_color="off",
        )
    with kpi2:
        st.metric(
            label="Standard Code Baseline",
            value=f"{detected_radius:.2f} m",
            delta="OSHA §1910.36",
            delta_color="off",
        )
    with kpi3:
        delta_m = st.session_state.radius_slider - detected_radius
        st.metric(
            label="Active Rule Envelope",
            value=f"{st.session_state.radius_slider:.2f} m",
            delta=f"{delta_m:+.2f} m override",
            delta_color="normal" if abs(delta_m) < 0.01 else "off",
        )
    with kpi4:
        if temp_hits:
            st.metric(
                label="Compliance Status",
                value="🚨 VIOLATION",
                delta=f"-{len(temp_hits)} Blocker(s)",
                delta_color="inverse",
            )
        else:
            st.metric(
                label="Compliance Status",
                value="✅ CLEAR",
                delta="Safe Envelope",
                delta_color="normal",
            )

    # ── Main Two-Column Layout ───────────────────────────────────────────────
    col_viewer, col_panel = st.columns([1.75, 1.0], gap="medium")

    # ── Left: Rerun 3D Web Viewer ──
    with col_viewer:
        st.markdown(
            """
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="font-weight: 600; font-size: 14px; color: #e2e8f0;">
                    🌐 3D Semantic Twin & Telemetry View
                </span>
                <span style="font-size: 12px; color: #64748b;">
                    Recording: <code>hallway_1</code>
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Embed Web Viewer
        st.markdown('<div class="viewer-frame">', unsafe_allow_html=True)
        components.iframe(viewer_url, height=560, scrolling=False)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Navigation helper bar
        st.markdown(
            """
            <div style="background: #0f172a; border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 6px 14px; margin-top: 6px; font-size: 11px; color: #94a3b8; display: flex; justify-content: space-between;">
                <span>🖱️ <b>Left-Click:</b> Orbit View</span>
                <span>✋ <b>Right-Click:</b> Pan</span>
                <span>🔍 <b>Scroll:</b> Zoom</span>
                <span>⏱️ <b>Timeline:</b> Scrub Synced Bag Playback</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Right: Control & Safety Intelligence Console ──
    with col_panel:
        # Card 1: Interactive Rule Override
        st.markdown('<div class="control-card">', unsafe_allow_html=True)
        st.markdown("#### ⚡ Dynamic Rule Override")
        st.caption("Simulate safety standards and emergency clearance envelope expansion live.")
        
        # Preset buttons
        st.markdown("<p style='font-size: 12px; font-weight: 600; color: #94a3b8; margin-bottom: 6px;'>Regulatory Presets:</p>", unsafe_allow_html=True)
        btn1, btn2, btn3, btn4 = st.columns(4)
        if btn1.button("0.81m\nOSHA", use_container_width=True):
            st.session_state.radius_slider = 0.81
            st.rerun()
        if btn2.button("1.20m\nADA", use_container_width=True):
            st.session_state.radius_slider = 1.20
            st.rerun()
        if btn3.button("1.80m\nGurney", use_container_width=True):
            st.session_state.radius_slider = 1.80
            st.rerun()
        if btn4.button("2.50m\nEvac", use_container_width=True):
            st.session_state.radius_slider = 2.50
            st.rerun()

        # Dynamic Slider
        custom_radius = st.slider(
            "Keep-Clear Zone Radius (meters)",
            min_value=0.40,
            max_value=3.50,
            value=float(st.session_state.radius_slider),
            step=0.05,
            key="radius_slider_input",
            help="Drag to adjust keep-clear radius. The 3D cylinder redraws live in the viewer above.",
        )
        if custom_radius != st.session_state.radius_slider:
            st.session_state.radius_slider = custom_radius
            st.rerun()
            
        st.markdown('</div>', unsafe_allow_html=True)

        # Evaluate Collisions with selected radius
        hits, proximity_list = find_blockers_and_proximity(
            landmarks, center, forward, custom_radius, bottom_z, top_z
        )

        # Inject override zone into Rerun instantly
        zone_color = [220, 30, 30] if hits else [40, 190, 80]
        strips = zone_strips(center, forward, right, custom_radius, bottom_z, top_z)

        # static=True forces the zone to update visually across the entire recorded timeline
        stream.log(
            "world/exit_zone",
            rr.LineStrips3D(strips, colors=[zone_color] * len(strips), radii=0.035),
            static=True,
        )

        # Card 2: Autonomous Obstruction Assessment
        st.markdown('<div class="control-card">', unsafe_allow_html=True)
        st.markdown("#### 🚨 Autonomous Assessment")
        
        if hits:
            st.markdown(
                f"""
                <div style="background: rgba(239, 68, 68, 0.12); border: 1px solid #ef4444; border-radius: 10px; padding: 14px; margin-bottom: 10px;">
                    <div style="display: flex; align-items: center; gap: 8px; color: #ef4444; font-weight: 700; font-size: 15px;">
                        <span>⚠️</span> REGULATORY VIOLATION DETECTED
                    </div>
                    <div style="color: #fca5a5; font-size: 13px; margin-top: 6px;">
                        <b>{len(hits)} physical object(s)</b> infringe upon the <b>{custom_radius:.2f}m</b> egress safety boundary.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            for h in hits:
                dx = h["X"] - center[0]
                dy = h["Y"] - center[1]
                dist = math.hypot(dx, dy)
                st.markdown(
                    f"&bull; **{h['class_name'].replace('_', ' ').title()} #{h['instance_id']}** &mdash; "
                    f"<code>{dist:.2f}m</code> from door (infringes by {custom_radius - dist:+.2f}m)"
                )
        else:
            closest = proximity_list[0] if proximity_list else None
            closest_info = f"Nearest obstacle is **{closest['landmark']['class_name']} #{closest['landmark']['instance_id']}** at **{closest['dist_2d']:.2f}m**." if closest else "No objects nearby."
            st.markdown(
                f"""
                <div style="background: rgba(34, 197, 94, 0.12); border: 1px solid #22c55e; border-radius: 10px; padding: 14px;">
                    <div style="display: flex; align-items: center; gap: 8px; color: #22c55e; font-weight: 700; font-size: 15px;">
                        <span>🛡️</span> EGRESS CLEAR & COMPLIANT
                    </div>
                    <div style="color: #86efac; font-size: 13px; margin-top: 6px;">
                        No physical obstacles detected within the {custom_radius:.2f}m safety envelope.<br>
                        <span style="color: #94a3b8; font-size: 12px;">{closest_info}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
        st.markdown('</div>', unsafe_allow_html=True)

        # Card 3: Proximity Radar (Top 4 closest)
        st.markdown('<div class="control-card">', unsafe_allow_html=True)
        st.markdown("<p style='font-size: 13px; font-weight: 600; color: #cbd5e1; margin-bottom: 8px;'>🎯 Proximity Radar (Nearest Landmarks)</p>", unsafe_allow_html=True)
        
        radar_items = []
        for item in proximity_list[:4]:
            lm = item["landmark"]
            status_badge = "🔴 Inside Zone" if item["is_hit"] else "🟢 Safe"
            radar_items.append({
                "Object": f"{lm['class_name'].replace('_', ' ').title()} #{lm['instance_id']}",
                "Dist (m)": f"{item['dist_2d']:.2f}",
                "Status": status_badge,
            })
        if radar_items:
            st.dataframe(pd.DataFrame(radar_items), hide_index=True, use_container_width=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Bottom Section: Structured Information Tabs ──────────────────────────
    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
    tab_landmarks, tab_ontology, tab_pipeline = st.tabs([
        "📋 Semantic Map Landmarks Inventory",
        "🧠 Neurosymbolic Spatial Logic",
        "⚙️ Pipeline Architecture & Telemetry",
    ])

    with tab_landmarks:
        st.markdown("##### Persistent Spatial Landmarks in Hallway 1")
        st.caption("Registered using RGB-D SLAM, YOLOv8/v11 detections, and Grounded SAM multi-view clustering.")
        
        table_rows = []
        for item in proximity_list:
            lm = item["landmark"]
            table_rows.append({
                "Landmark ID": lm["landmark_id"],
                "Class": lm["class_name"].replace("_", " ").title(),
                "Instance": f"#{lm['instance_id']}",
                "Distance to Exit (m)": round(item["dist_2d"], 2),
                "Position (X, Y, Z)": f"({lm['X']:.2f}, {lm['Y']:.2f}, {lm['Z']:.2f})",
                "Hit Count": lm["hit_count"],
                "Confidence": f"{float(lm['confidence']):.2f}" if lm["confidence"] else "N/A",
                "Egress Status": "🚨 VIOLATION (Inside Zone)" if item["is_hit"] else "✅ Clear",
            })
        
        df_landmarks = pd.DataFrame(table_rows)
        st.dataframe(df_landmarks, hide_index=True, use_container_width=True)

    with tab_ontology:
        st.markdown("##### Neurosymbolic Spatial Reasoning & Rule Formulation")
        st.markdown(
            r"""
            The egress clearance monitor implements a hybrid perception-reasoning pipeline adhering to international building safety regulations:
            
            1. **Geometric Anchor Extraction:**  
               The exit door plane is detected in camera space $\mathbf{T}_{cam}$ and mapped to world coordinates using camera pose matrix $\mathbf{T}_{world \leftarrow cam}$:
               $$\mathbf{p}_{world} = \mathbf{T}_{world \leftarrow cam} \begin{bmatrix} \mathbf{p}_{cam} \\ 1 \end{bmatrix}$$
            
            2. **Half-Cylindrical Keep-Clear Safety Envelope:**  
               Given door anchor center $\mathbf{c} = [x_d, y_d, z_d]^T$ and egress outward normal $\hat{\mathbf{f}}$, the envelope volume $\mathcal{B}(r)$ for radius $r$ is defined as:
               $$\mathcal{B}(r) = \left\{ \mathbf{p} \in \mathbb{R}^3 \;\middle|\; \|\mathbf{p}_{xy} - \mathbf{c}_{xy}\|_2 \le r, \quad (\mathbf{p} - \mathbf{c}) \cdot \hat{\mathbf{f}} \le 0, \quad z_{min} \le p_z \le z_{max} \right\}$$
            
            3. **Regulatory Decision Function:**  
               An obstruction event is triggered if any persistent semantic landmark $\mathcal{L}_i \notin \{\text{door}\}$ intersects $\mathcal{B}(r)$:
               $$\text{ObstructionFlag}(r) = \bigvee_{i} \left( \mathbf{p}_{\mathcal{L}_i} \in \mathcal{B}(r) \right)$$
               
            4. **Ontological Grounding:**  
               Classes and relations conform to **SOSA/SSN** (`sosa:Observation`, `sosa:FeatureOfInterest`) and **BuildingSMART IFC** (`IfcDoor`, `IfcSpace`, `IfcFlowTerminal`).
            """
        )

    with tab_pipeline:
        st.markdown("##### System Telemetry & gRPC Stream Topology")
        meta_col1, meta_col2 = st.columns(2)
        with meta_col1:
            st.markdown(f"**Data Run Directory:** `{run_dir}`")
            st.markdown(f"**Semantic Map DB:** `{db_path.name}` ({os.path.getsize(db_path) / 1024:.1f} KB)")
            st.markdown(f"**Rerun Archive (.rrd):** `{rrd_path.name}` ({os.path.getsize(rrd_path) / (1024 * 1024):.1f} MB)")
        with meta_col2:
            st.markdown(f"**Live gRPC Port:** `9877` (Memory Buffer: `8GiB`)")
            st.markdown(f"**Rerun Web Viewer Port:** `9090`")
            st.markdown(f"**Door Anchor World XYZ:** `({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})`")


if __name__ == "__main__":
    main()