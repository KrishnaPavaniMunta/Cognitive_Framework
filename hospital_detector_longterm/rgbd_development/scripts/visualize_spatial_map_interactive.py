"""visualize_spatial_map_interactive.py
Interactive 3-D spatial map viewer for HospitalGuard spatial detections.

Features
--------
- Full 3-D rotate / pan / zoom (Plotly, opens automatically in browser)
- Objects coloured and sized by class / confidence
- Camera trajectory rendered as a 3-D line with Start/End markers
- Hover any object point → right-hand panel shows the actual video frame
  plus class, tracker ID, confidence, depth and world XYZ
- Self-contained HTML (no web server required — works offline)

Usage
-----
    python visualize_spatial_map_interactive.py
    python visualize_spatial_map_interactive.py --csv-path PATH --video-path PATH
    python visualize_spatial_map_interactive.py --session-id ID
    python visualize_spatial_map_interactive.py --no-video          # skip frame extraction
    python visualize_spatial_map_interactive.py --no-open           # don't auto-open browser
"""

from __future__ import annotations

import argparse
import base64
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go
from hospital_constants import STATIC_CLASS_NAMES

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
SCRIPT_DIR    = Path(__file__).resolve().parent
RGBD_DEV_DIR  = SCRIPT_DIR.parent
LOGS_DIR      = RGBD_DEV_DIR / "output" / "logs"
DETECTIONS_DIR = RGBD_DEV_DIR / "output" / "detections"
PLOTS_DIR     = RGBD_DEV_DIR / "output" / "plots"

# ---------------------------------------------------------------------------
# Colourblind-friendly 22-colour palette
_PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9A6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
    "#ffffff", "#66ccff",
]

# 1×1 transparent PNG used when no thumbnail is available
_EMPTY_IMG = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_float(v: str | None) -> float | None:
    s = str(v).strip() if v is not None else ""
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _latest_csv(logs_dir: Path) -> Path:
    candidates = sorted(
        logs_dir.glob("spatial_realsense_temporal_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No spatial_realsense_temporal_*.csv files in {logs_dir}")
    return candidates[0]


def _matching_video(csv_path: Path, detections_dir: Path) -> Path | None:
    """Best-effort: find the video whose timestamp suffix matches the CSV."""
    suffix = csv_path.stem.split("spatial_realsense_temporal_")[-1]  # e.g. 20260513_153939
    for ext in ("mp4", "avi"):
        cand = detections_dir / f"hospitalguard_realsense_temporal_{suffix}.{ext}"
        if cand.exists():
            return cand
    # Fallback to the most recent mp4
    fallbacks = sorted(
        detections_dir.glob("hospitalguard_realsense_temporal_*.mp4"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return fallbacks[0] if fallbacks else None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_detections(csv_path: Path, session_id: str | None) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if session_id and str(row.get("session_id", "")).strip() != session_id:
                continue
            x = _to_float(row.get("X_m"))
            y = _to_float(row.get("Y_m"))
            z = _to_float(row.get("Z_m"))
            if x is None or y is None or z is None:
                continue
            try:
                frame_idx  = int(row["frame_index"])
                tracker_id = int(float(row["tracker_id"]))
            except (ValueError, KeyError):
                continue
            rows.append({
                "frame_idx":  frame_idx,
                "timestamp":  row.get("timestamp", ""),
                "class_name": str(row.get("class_name", "")).strip(),
                "tracker_id": tracker_id,
                "confidence": _to_float(row.get("confidence")) or 0.0,
                "center_u":   _to_float(row.get("center_u")) or 0.0,
                "center_v":   _to_float(row.get("center_v")) or 0.0,
                "depth_m":    _to_float(row.get("depth_m")) or 0.0,
                "x": x, "y": y, "z": z,
            })
    return rows


def _kabsch_rt(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kabsch (SVD) rigid-body alignment: find R, t such that dst ≈ R @ src + t.

    Parameters
    ----------
    src, dst : (N, 3)  N ≥3 point correspondences

    Returns
    -------
    R : (3, 3) rotation matrix
    t : (3,)   translation vector
    """
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    A = (src - src_mean).T @ (dst - dst_mean)
    U, _, Vt = np.linalg.svd(A)
    d = np.linalg.det(Vt.T @ U.T)  # reflection correction
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = dst_mean - R @ src_mean
    return R, t


def compute_world_coords(rows: list[dict]) -> list[dict]:
    """
    Rotation-aware world-frame estimator using the Kabsch SVD algorithm.
    Falls back to translation-only when fewer than 3 anchor correspondences
    are visible.  Adds wx/wy/wz (world) and cam_x/cam_y/cam_z (camera origin
    in world frame) to every row in-place.
    """
    frames: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        frames[r["frame_idx"]].append(r)

    world_anchors: dict[tuple[str, int], np.ndarray] = {}
    prev_R = np.eye(3, dtype=float)
    prev_t = np.zeros(3, dtype=float)

    for frame_idx in sorted(frames.keys()):
        objs = frames[frame_idx]
        static_obs = [
            (r, np.asarray([r["x"], r["y"], r["z"]], dtype=float))
            for r in objs
            if r["class_name"] in STATIC_CLASS_NAMES
        ]

        src_pts, dst_pts = [], []
        for r, cam_xyz in static_obs:
            key = (r["class_name"], r["tracker_id"])
            if key in world_anchors:
                src_pts.append(cam_xyz)
                dst_pts.append(world_anchors[key])

        if len(src_pts) >= 3:
            R, t = _kabsch_rt(np.stack(src_pts), np.stack(dst_pts))
        elif src_pts:
            R = prev_R.copy()
            t = np.mean([d - R @ s for s, d in zip(src_pts, dst_pts)], axis=0)
        else:
            R, t = prev_R.copy(), prev_t.copy()

        for r, cam_xyz in static_obs:
            key = (r["class_name"], r["tracker_id"])
            if key not in world_anchors:
                world_anchors[key] = R @ cam_xyz + t

        for r in objs:
            cam_xyz = np.asarray([r["x"], r["y"], r["z"]], dtype=float)
            world_pt = R @ cam_xyz + t
            r["wx"]    = float(world_pt[0])
            r["wy"]    = float(world_pt[1])
            r["wz"]    = float(world_pt[2])
            r["cam_x"] = float(t[0])
            r["cam_y"] = float(t[1])
            r["cam_z"] = float(t[2])

        prev_R, prev_t = R, t

    return rows


# ---------------------------------------------------------------------------
# Frame thumbnail extraction
# ---------------------------------------------------------------------------
def _encode_frame(frame: np.ndarray, max_w: int = 320) -> str:
    """Resize frame and encode as base64 JPEG data-URI."""
    h, w = frame.shape[:2]
    if w > max_w:
        scale = max_w / w
        frame = cv2.resize(frame, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
    if not ok:
        return _EMPTY_IMG
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def extract_frame_thumbnails(video_path: Path, frame_indices: list[int]) -> dict[int, str]:
    """
    Seek into *video_path* and extract one thumbnail per index in
    *frame_indices*.  Returns {frame_idx: data_uri}.
    """
    if not video_path.exists():
        return {}
    needed = sorted(set(frame_indices))
    result: dict[int, str] = {}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[spatial-viewer] Warning: could not open video {video_path}")
        return {}
    current = -1
    for fi in needed:
        if fi != current + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if ok:
            result[fi] = _encode_frame(frame)
        current = fi
    cap.release()
    return result


# ---------------------------------------------------------------------------
# Plotly figure
# ---------------------------------------------------------------------------
def build_figure(rows: list[dict], frame_thumbs: dict[int, str]) -> go.Figure:
    all_classes = sorted({r["class_name"] for r in rows})
    color_map   = {cls: _PALETTE[i % len(_PALETTE)] for i, cls in enumerate(all_classes)}

    # Camera trajectory — one position per frame
    cam_by_frame: dict[int, tuple[float, float, float]] = {}
    for r in rows:
        fi = r["frame_idx"]
        if fi not in cam_by_frame:
            cam_by_frame[fi] = (r["cam_x"], r["cam_y"], r["cam_z"])
    sorted_frames = sorted(cam_by_frame.keys())
    cam_xs = [cam_by_frame[fi][0] for fi in sorted_frames]
    cam_ys = [cam_by_frame[fi][1] for fi in sorted_frames]
    cam_zs = [cam_by_frame[fi][2] for fi in sorted_frames]

    # One representative row per unique (class, tracker_id) — last seen
    latest: dict[tuple[str, int], dict] = {}
    for r in rows:
        latest[(r["class_name"], r["tracker_id"])] = r
    unique_rows = list(latest.values())

    traces: list[go.BaseTraceType] = []

    # ---- Camera path ----
    traces.append(go.Scatter3d(
        x=cam_xs, y=cam_ys, z=cam_zs,
        mode="lines+markers",
        name="Camera path",
        line=dict(color="#aaaaaa", width=2, dash="dot"),
        marker=dict(size=1.5, color="#aaaaaa"),
        hoverinfo="skip",
        legendgroup="camera",
    ))
    if sorted_frames:
        traces.append(go.Scatter3d(
            x=[cam_xs[0]], y=[cam_ys[0]], z=[cam_zs[0]],
            mode="markers+text", name="Start",
            marker=dict(size=9, color="#00ff88", symbol="diamond"),
            text=["Start"], textposition="top center",
            hoverinfo="skip", showlegend=True,
        ))
        traces.append(go.Scatter3d(
            x=[cam_xs[-1]], y=[cam_ys[-1]], z=[cam_zs[-1]],
            mode="markers+text", name="End",
            marker=dict(size=9, color="#ff4444", symbol="diamond"),
            text=["End"], textposition="top center",
            hoverinfo="skip", showlegend=True,
        ))

    # ---- One trace per class ----
    for cls in all_classes:
        cls_rows = [r for r in unique_rows if r["class_name"] == cls]
        xs    = [r["wx"] for r in cls_rows]
        ys    = [r["wy"] for r in cls_rows]
        zs    = [r["wz"] for r in cls_rows]
        sizes = [max(5, min(16, int(r["confidence"] * 16))) for r in cls_rows]

        # customdata columns: class, tracker_id, frame_idx, confidence,
        #                     depth_m, img_uri
        custom = [
            [
                r["class_name"],
                r["tracker_id"],
                r["frame_idx"],
                round(r["confidence"], 3),
                round(r["depth_m"], 3),
                frame_thumbs.get(r["frame_idx"], _EMPTY_IMG),
            ]
            for r in cls_rows
        ]

        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="markers",
            name=cls,
            marker=dict(
                size=sizes,
                color=color_map[cls],
                opacity=0.88,
                line=dict(width=0.4, color="rgba(0,0,0,0.5)"),
            ),
            customdata=custom,
            hovertemplate=(
                "<b>%{customdata[0]}</b> #%{customdata[1]}<br>"
                "Frame %{customdata[2]}<br>"
                "Conf: %{customdata[3]:.1%}<br>"
                "Depth: %{customdata[4]} m<br>"
                "World (X,Y,Z): (%{x:.3f}, %{y:.3f}, %{z:.3f}) m"
                "<extra></extra>"
            ),
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#1a1a2e",
        font=dict(color="white", family="'Courier New', monospace"),
        title=dict(
            text="HospitalGuard — Interactive 3-D Spatial Map",
            font=dict(size=17, color="#7ec8e3"),
        ),
        legend=dict(
            bgcolor="rgba(10,10,30,0.75)",
            bordercolor="#2a2a4a",
            borderwidth=1,
            font=dict(size=10),
            itemsizing="constant",
        ),
        scene=dict(
            bgcolor="#0d0d1a",
            xaxis=dict(
                title="X (m)", gridcolor="#2a2a4a",
                zerolinecolor="#555", tickfont=dict(size=9),
            ),
            yaxis=dict(
                title="Y (m)", gridcolor="#2a2a4a",
                zerolinecolor="#555", tickfont=dict(size=9),
            ),
            zaxis=dict(
                title="Z (m)", gridcolor="#2a2a4a",
                zerolinecolor="#555", tickfont=dict(size=9),
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.8),
            ),
        ),
        margin=dict(l=0, r=0, t=42, b=0),
    )
    return fig


# ---------------------------------------------------------------------------
# Self-contained HTML builder
# ---------------------------------------------------------------------------
# NOTE: this is a plain string, NOT a Python f-string, so no {{}} escaping is
#       needed.  Two sentinel strings are replaced with str.replace() at save
#       time: __PLOTLY_JSON__ and __SESSION_INFO__.
_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>HospitalGuard — 3-D Spatial Map</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: #0d0d1a;
  color: #e0e0e0;
  font-family: 'Courier New', monospace;
  display: flex;
  height: 100vh;
  overflow: hidden;
}
#map-container { flex: 1; min-width: 0; }
#side-panel {
  width: 330px;
  flex-shrink: 0;
  background: #11112a;
  border-left: 1px solid #2a2a4a;
  display: flex;
  flex-direction: column;
  padding: 14px 12px;
  gap: 12px;
  overflow-y: auto;
}
#panel-title {
  font-size: 12px;
  color: #7ec8e3;
  font-weight: bold;
  text-transform: uppercase;
  letter-spacing: 1.5px;
}
#frame-wrap {
  width: 100%;
  background: #0a0a1a;
  border: 1px solid #2a2a4a;
  border-radius: 4px;
  min-height: 180px;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}
#frame-wrap img { width: 100%; display: block; border-radius: 3px; }
#placeholder-text { color: #555; font-size: 12px; text-align: center; padding: 20px; }
#obj-info { font-size: 12px; line-height: 1.9; }
#obj-info table { width: 100%; border-collapse: collapse; }
#obj-info .lbl { color: #7ec8e3; padding: 2px 6px 2px 2px; white-space: nowrap; }
#obj-info .val { color: #ffffff; padding: 2px 2px; }
#obj-info tr:nth-child(odd) { background: rgba(255,255,255,0.035); }
#instructions {
  font-size: 10px;
  color: #555;
  margin-top: auto;
  padding-top: 10px;
  border-top: 1px solid #2a2a4a;
  line-height: 1.7;
}
#session-meta {
  font-size: 10px;
  color: #555;
  border-top: 1px solid #2a2a4a;
  padding-top: 8px;
}
</style>
</head>
<body>

<div id="map-container">
  <div id="plotly-div" style="width:100%;height:100%;"></div>
</div>

<div id="side-panel">
  <div id="panel-title">&#x1F50D; Object Inspector</div>

  <div id="frame-wrap">
    <img id="preview-img" alt="frame" style="display:none"/>
    <div id="placeholder-text">Hover a point to inspect</div>
  </div>

  <div id="obj-info">
    <table id="info-table"></table>
  </div>

  <div id="instructions">
    <b>Rotate:</b> left-drag &nbsp;|&nbsp; <b>Pan:</b> right-drag<br>
    <b>Zoom:</b> scroll wheel &nbsp;|&nbsp; <b>Reset:</b> double-click<br>
    <b>Inspect:</b> hover any coloured dot<br>
    <b>Toggle class:</b> click legend entry
  </div>

  <div id="session-meta">__SESSION_INFO__</div>
</div>

<script>
var PDATA = __PLOTLY_JSON__;
</script>
<script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
<script>
(function () {
  var cfg = {
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ["toImage"],
    scrollZoom: true
  };

  Plotly.newPlot("plotly-div", PDATA.data, PDATA.layout, cfg);

  var plotDiv = document.getElementById("plotly-div");

  plotDiv.on("plotly_hover", function (evt) {
    var pts = evt.points;
    if (!pts || !pts.length) return;
    var pt = pts[0];
    var cd = pt.customdata;
    if (!cd) return;

    // customdata: [class_name, tracker_id, frame_idx, confidence, depth_m, img_uri]
    var cls   = cd[0];
    var tid   = cd[1];
    var fidx  = cd[2];
    var conf  = cd[3];
    var depth = cd[4];
    var imgURI = cd[5];

    // Frame preview
    var imgEl   = document.getElementById("preview-img");
    var holder  = document.getElementById("placeholder-text");
    if (imgURI && imgURI.length > 40) {
      imgEl.src = imgURI;
      imgEl.style.display = "block";
      holder.style.display = "none";
    } else {
      imgEl.style.display = "none";
      holder.style.display = "block";
      holder.textContent = "No frame available";
    }

    // Info table
    var rows = [
      ["Class",       "<b>" + cls + "</b>"],
      ["Tracker ID",  "#" + tid],
      ["Frame",       fidx],
      ["Confidence",  (conf * 100).toFixed(1) + "%"],
      ["Depth",       depth + " m"],
      ["World X",     pt.x.toFixed(3) + " m"],
      ["World Y",     pt.y.toFixed(3) + " m"],
      ["World Z",     pt.z.toFixed(3) + " m"],
    ];
    document.getElementById("info-table").innerHTML = rows.map(function (r) {
      return '<tr><td class="lbl">' + r[0] + '</td><td class="val">' + r[1] + '</td></tr>';
    }).join("");
  });

  plotDiv.on("plotly_unhover", function () {
    // keep last inspection visible — do nothing on unhover
  });
}());
</script>
</body>
</html>
"""


def save_html(fig: go.Figure, out_path: Path, session_info: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    html = _HTML.replace("__PLOTLY_JSON__", fig.to_json())
    html = html.replace("__SESSION_INFO__", session_info)
    out_path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive 3-D spatial map viewer for HospitalGuard"
    )
    parser.add_argument("--csv-path",   type=Path, default=None,
                        help="Spatial CSV (default: latest spatial_realsense_temporal_*.csv)")
    parser.add_argument("--video-path", type=Path, default=None,
                        help="Annotated video for frame thumbnails (default: auto-detect)")
    parser.add_argument("--session-id", default=None,
                        help="Filter rows by session_id")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output HTML path (default: output/plots/spatial_map_interactive.html)")
    parser.add_argument("--no-open",  action="store_true",
                        help="Do not open the browser automatically")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip video frame extraction (faster, no hover images)")
    args = parser.parse_args()

    # ---- CSV ----
    csv_path = args.csv_path or _latest_csv(LOGS_DIR)
    print(f"[viewer] CSV    : {csv_path}")

    rows = load_detections(csv_path, args.session_id)
    if not rows:
        print("[viewer] No data rows found — check --session-id or --csv-path.")
        return

    rows = compute_world_coords(rows)
    total_objects = len({(r["class_name"], r["tracker_id"]) for r in rows})
    total_frames  = len({r["frame_idx"] for r in rows})
    print(f"[viewer] Rows   : {len(rows)} | Objects: {total_objects} | Frames: {total_frames}")

    # ---- Video thumbnails ----
    frame_thumbs: dict[int, str] = {}
    if not args.no_video:
        video_path = args.video_path or _matching_video(csv_path, DETECTIONS_DIR)
        if video_path:
            print(f"[viewer] Video  : {video_path}")
            # Only extract the representative frame for each unique object
            # (the last frame it appeared in, stored in latest[key])
            latest: dict[tuple[str, int], dict] = {}
            for r in rows:
                latest[(r["class_name"], r["tracker_id"])] = r
            needed_frames = sorted({r["frame_idx"] for r in latest.values()})
            print(f"[viewer] Extracting {len(needed_frames)} unique object frames …")
            frame_thumbs = extract_frame_thumbnails(video_path, needed_frames)
            print(f"[viewer] Thumbnails: {len(frame_thumbs)} extracted")
        else:
            print("[viewer] Video  : not found — hover images disabled")

    # ---- Build & save ----
    fig      = build_figure(rows, frame_thumbs)
    out_path = args.out or PLOTS_DIR / "spatial_map_interactive.html"
    unique_classes = sorted({r["class_name"] for r in rows})
    session_info = (
        f"CSV: {csv_path.name}<br>"
        f"Frames: {total_frames} &nbsp;|&nbsp; Objects: {total_objects}<br>"
        f"Classes ({len(unique_classes)}): {', '.join(unique_classes)}"
    )
    save_html(fig, out_path, session_info)
    print(f"[viewer] Saved  : {out_path}")

    if not args.no_open:
        import webbrowser
        webbrowser.open(out_path.as_uri())
        print("[viewer] Opened in browser.")


if __name__ == "__main__":
    main()
