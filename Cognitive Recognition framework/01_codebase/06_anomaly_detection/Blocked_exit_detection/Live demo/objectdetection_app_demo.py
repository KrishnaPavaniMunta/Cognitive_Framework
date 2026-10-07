"""3D Object Detection & Semantic Map Dashboard.

Run:
    streamlit run "01_codebase/06_anomaly_detection/Blocked_exit_detection/Live demo/objectdetection_app_demo.py"
"""
from __future__ import annotations

import sqlite3
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import rerun as rr
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[4]
MAP_DIR = (
    PROJECT_ROOT
    / "04_outputs_runs_and_logs"
    / "outputs"
    / "semantic_maps"
    / "rgbd_clean_20260521_142555"
)
RRD_PATH = MAP_DIR / "world_map.rrd"
DB_PATH = MAP_DIR / "world_map.db"

st.set_page_config(
    page_title="3D Object Detection Demo",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Clean, modern dark styling matching app_demo.py
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
    .class-badge {
        display: inline-block;
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.35);
        color: #e0f2fe;
        border-radius: 8px;
        padding: 6px 12px;
        margin: 4px;
        font-size: 13px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_detected_objects(db_path: str, db_mtime: float = 0.0) -> list[dict]:
    """Load mapped 3D object detections from world_map.db."""
    if not Path(db_path).exists():
        return []
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT landmark_id, class_name, instance_id, hit_count, mean_confidence, X, Y, Z
            FROM semantic_map
            ORDER BY class_name, instance_id
            """
        ).fetchall()
    finally:
        conn.close()

    return [
        {
            "landmark_id": r[0],
            "class_name": r[1],
            "instance_id": r[2],
            "hit_count": r[3],
            "mean_confidence": r[4],
            "X": r[5],
            "Y": r[6],
            "Z": r[7],
        }
        for r in rows
    ]


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
    return "semantic_map-rgbd_clean_20260521_1425551aaf", "907492cf5afb406fab9712b71c607c49"


@st.cache_resource(show_spinner="Loading 3D object detection map...")
def start_rerun_server(rrd_path: Path) -> tuple[rr.RecordingStream, str]:
    """Boot one persistent recording+server using the exact recording ID from the file."""
    app_id, rec_id = get_rrd_recording_info(rrd_path)
    stream = rr.RecordingStream(
        application_id=app_id,
        recording_id=rec_id,
    )
    grpc_uri = stream.serve_grpc(grpc_port=9878, server_memory_limit="8GiB")
    stream.log_file_from_path(str(rrd_path))
    rr.serve_web_viewer(web_port=9091, open_browser=False, connect_to=grpc_uri)
    return stream, grpc_uri


def main():
    if not RRD_PATH.exists():
        st.error(f"Rerun recording not found: {RRD_PATH}")
        st.stop()

    landmarks = (
        load_detected_objects(str(DB_PATH), DB_PATH.stat().st_mtime)
        if DB_PATH.exists()
        else []
    )

    _, grpc_uri = start_rerun_server(RRD_PATH)
    viewer_url = f"http://localhost:9091/?url={quote(grpc_uri, safe='')}"

    # ── Top Title Header ──────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="header-card">
            <h1 style="color: #f8fafc; font-size: 26px; font-weight: 700; margin: 0 0 6px 0;">
                Hospital Environment: 3D Object Detection &amp; Semantic Map
            </h1>
            <p style="color: #94a3b8; font-size: 15px; margin: 0;">
                Interactive 3D semantic map and camera view showing detected objects localized in world coordinates.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── 3D Viewer ─────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <span style="font-weight: 600; font-size: 15px; color: #e2e8f0;">📹 3D Map &amp; Camera View</span>
            <span style="font-size: 12px; color: #94a3b8;">Left-click to rotate &bull; Right-click to pan &bull; Scroll to zoom</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="viewer-frame">', unsafe_allow_html=True)
    st.iframe(viewer_url, height=620, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

    if landmarks:
        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
        col_summary, col_table = st.columns([1.0, 1.4], gap="medium")

        class_counts = Counter(lm["class_name"] for lm in landmarks)

        with col_summary:
            with st.container(border=True):
                st.markdown(
                    f"<h3 style='margin-top:0; font-size: 19px;'>🎯 Detected Object Classes ({len(landmarks)} total)</h3>",
                    unsafe_allow_html=True,
                )
                badges_html = "".join(
                    f'<span class="class-badge">{cls.replace("_", " ").title()}: {count}</span>'
                    for cls, count in sorted(class_counts.items())
                )
                st.markdown(f"<div>{badges_html}</div>", unsafe_allow_html=True)

        with col_table:
            with st.container(border=True):
                st.markdown(
                    "<h3 style='margin-top:0; font-size: 19px;'>📍 Mapped 3D Object Instances</h3>",
                    unsafe_allow_html=True,
                )
                table_rows = [
                    {
                        "Object": f"{lm['class_name'].replace('_', ' ').title()} #{lm['instance_id']}",
                        "Confidence": f"{lm['mean_confidence']:.0%}",
                        "Observations": lm["hit_count"],
                        "World (X, Y, Z) [m]": f"({lm['X']:.2f}, {lm['Y']:.2f}, {lm['Z']:.2f})",
                    }
                    for lm in landmarks
                ]
                st.dataframe(table_rows, use_container_width=True, hide_index=True, height=220)


if __name__ == "__main__":
    main()
