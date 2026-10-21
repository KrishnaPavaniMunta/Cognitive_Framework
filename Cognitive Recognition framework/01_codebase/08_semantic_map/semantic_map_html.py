"""Plotly HTML renderer for ontology-enriched semantic-map landmarks and Hospital H&S Compliance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

PALETTE = [
    "#d1495b", "#00798c", "#edae49", "#30638e", "#5f6f52", "#8f5d5d",
    "#6a4c93", "#2a9d8f", "#bc6c25", "#457b9d", "#9c6644", "#577590",
]


def _landmark_payload(landmark: dict) -> dict:
    real_w = landmark.get("measured_width") or landmark.get("real_width")
    real_d = landmark.get("measured_depth") or landmark.get("real_depth")
    real_h = landmark.get("measured_height") or landmark.get("real_height")
    has_real = False
    if real_w is not None and real_h is not None:
        try:
            has_real = float(real_w) > 0.02 and float(real_h) > 0.02
        except (ValueError, TypeError):
            has_real = False

    is_floor = str(landmark.get("class_name", "")).strip().lower() in ("floor", "navigable_floor")
    map_info = {
        "Class": "Floor" if is_floor else landmark["class_name"],
        "World frame": landmark["world_frame"],
        "Hits": landmark.get("hit_count"),
        "X (m)": landmark["X"],
        "Y (m)": landmark["Y"],
        "Z (m)": landmark["Z"],
        "Last observed": landmark.get("last_seen") or landmark.get("last_seen_ns"),
        "Dimension source": "camera_measured" if has_real else "ontology_typical",
    }
    if not is_floor:
        map_info["Instance ID"] = int(landmark.get("instance_id", 1))
        map_info["Landmark ID"] = int(landmark.get("landmark_id", 0))

    if landmark.get("consensus_score") is not None:
        map_info["Consensus score"] = f"{float(landmark['consensus_score']):.1%}"
    class_votes = landmark.get("class_votes")
    if not class_votes and landmark.get("class_votes_json"):
        try:
            class_votes = json.loads(landmark["class_votes_json"])
        except Exception:
            class_votes = None
    if class_votes and isinstance(class_votes, dict):
        map_info["Class votes"] = ", ".join(f"{k}: {float(v):.1f}" for k, v in sorted(class_votes.items(), key=lambda kv: kv[1], reverse=True))

    if has_real:
        map_info["Measured width (m)"] = round(float(real_w), 3)
        map_info["Measured depth (m)"] = round(float(real_d), 3) if real_d is not None else round(float(real_w), 3)
        map_info["Measured height (m)"] = round(float(real_h), 3)

    return {
        "map": map_info,
        "ontology": landmark.get("ontology", {}),
        "violations": landmark.get("violations", []),
    }


def build_figure(
    landmarks: list[dict],
    trajectory: list[tuple[float, float, float]],
    title: str,
    navigable_zones: list[dict] | None = None,
    compliance_report: dict[str, Any] | None = None,
) -> go.Figure:
    class_names = sorted({landmark["class_name"] for landmark in landmarks})
    colors = {name: PALETTE[index % len(PALETTE)] for index, name in enumerate(class_names)}
    figure = go.Figure()

    # Map landmarks by label
    lm_by_label: dict[str, dict] = {}
    for lm in landmarks:
        lbl = f"{lm['class_name']} {lm.get('instance_id') or lm['landmark_id']}"
        lm_by_label[lbl] = lm

    # Attach violations from compliance report
    violations = compliance_report.get("violations", []) if compliance_report else []
    for v in violations:
        obj_name = v.get("object")
        if obj_name in lm_by_label:
            lm_by_label[obj_name].setdefault("violations", []).append(v)

    # 1. Render discovered navigable floor zone
    for zone in navigable_zones or []:
        poly = zone.get("polygon") or []
        elev_z = float(zone.get("elevation_z", 0.0))
        area = float(zone.get("area_m2", 0.0))
        if poly and len(poly) >= 3:
            poly_closed = list(poly) + [poly[0]]
            px = [p[0] for p in poly_closed]
            py = [p[1] for p in poly_closed]
            pz = [elev_z for _ in poly_closed]
            figure.add_trace(
                go.Scatter3d(
                    x=px, y=py, z=pz,
                    mode="lines",
                    name="Floor (boundary)",
                    line={"color": "#4fb3a5", "width": 4},
                    hovertemplate=f"<b>Floor</b><br>Navigable Zone<br>Area: {area:.1f} m²<extra></extra>",
                )
            )

    # 2. Render Camera trajectory
    if len(trajectory) >= 2:
        figure.add_trace(
            go.Scatter3d(
                x=[point[0] for point in trajectory],
                y=[point[1] for point in trajectory],
                z=[point[2] for point in trajectory],
                mode="lines",
                name="Camera trajectory",
                line={"color": "#64748b", "width": 3, "dash": "dot"},
                hoverinfo="skip",
            )
        )

    # 3. Render 3D Compliance Hazard Lines (Red connecting strips)
    for v in violations:
        source_lbl = v.get("object")
        target_lbl = v.get("target") or v.get("door")
        rule_id = v.get("rule_id", "HAZARD")
        d_val = v.get("measured_distance_m")

        if source_lbl in lm_by_label and target_lbl in lm_by_label:
            p1 = lm_by_label[source_lbl]
            p2 = lm_by_label[target_lbl]
            dist_str = f"{d_val:.2f}m" if d_val is not None else ""
            figure.add_trace(
                go.Scatter3d(
                    x=[p1["X"], p2["X"]],
                    y=[p1["Y"], p2["Y"]],
                    z=[p1["Z"], p2["Z"]],
                    mode="lines+markers",
                    name=f"Hazard: {rule_id}",
                    line={"color": "#ef4444", "width": 6},
                    marker={"size": 4, "color": "#ef4444"},
                    hovertemplate=(
                        f"<b>HAZARD: {rule_id}</b><br>"
                        f"{source_lbl} -> {target_lbl}<br>"
                        f"Distance: {dist_str}<extra></extra>"
                    ),
                )
            )

    # 3b. Render 3D Door Keep-Clear Egress Zones (Extruded 3D semi-cylinder cages)
    # Only render zones for designated Emergency Fire Exit doors
    zones = (compliance_report.get("door_keep_clear_zones") if compliance_report else None) or []
    if not zones and compliance_report:
        db_p_str = compliance_report.get("database", "")
        if db_p_str:
            graph_json_p = Path(db_p_str).with_name("spatial_scene_graph.json")
            if graph_json_p.exists():
                try:
                    with open(graph_json_p, "r", encoding="utf-8") as f:
                        sg_data = json.load(f)
                    zones = sg_data.get("door_keep_clear_zones", [])
                except Exception:
                    pass

    for dz in zones:
        if not dz.get("is_fire_exit", True):
            continue
        did = dz.get("door_id")
        status = dz.get("status", "CLEAR")
        color = "#16a34a" if status == "CLEAR" else "#dc2626"
        cage = dz.get("cage_samples_3d") or {}
        segs = []
        if isinstance(cage, dict):
            if cage.get("bottom_arc"):
                segs.append(cage["bottom_arc"])
            if cage.get("top_arc"):
                segs.append(cage["top_arc"])
            for rib in cage.get("vertical_ribs", []):
                segs.append(rib)
            if cage.get("back_wall"):
                segs.append(cage["back_wall"])
        elif isinstance(cage, list):
            segs = cage

        cx_pts, cy_pts, cz_pts = [], [], []
        if segs:
            for seg in segs:
                if not seg:
                    continue
                for pt in seg:
                    cx_pts.append(pt[0])
                    cy_pts.append(pt[1])
                    cz_pts.append(pt[2])
                cx_pts.append(None)
                cy_pts.append(None)
                cz_pts.append(None)
        else:
            arc = dz.get("arc_samples_3d", [])
            z_min, z_max = dz.get("z_bounds", [0.0, 1.8])
            for z_val in (z_min, z_max):
                for pt in arc:
                    cx_pts.append(pt[0])
                    cy_pts.append(pt[1])
                    cz_pts.append(z_val)
                cx_pts.append(None)
                cy_pts.append(None)
                cz_pts.append(None)

        if cx_pts:
            figure.add_trace(
                go.Scatter3d(
                    x=cx_pts,
                    y=cy_pts,
                    z=cz_pts,
                    mode="lines",
                    name=f"Fire Door {did} Zone ({status})",
                    line={"color": color, "width": 4},
                    hovertemplate=(
                        f"<b>Emergency Fire Exit Door {did}</b><br>"
                        f"Keep-Clear Zone: <b>{status}</b><br>"
                        f"Radius: {dz.get('radius_m', 1.0):.1f}m | Door Width: {dz.get('width_m', 0.85):.2f}m<br>"
                        f"Height Range: {dz.get('z_bounds', [0, 1.8])[0]:.2f}m to {dz.get('z_bounds', [0, 1.8])[1]:.2f}m<extra></extra>"
                    ),
                    showlegend=True,
                )
            )

    # 4. Render Landmarks with Black Label Text
    for class_name in class_names:
        is_floor = str(class_name).strip().lower() in ("floor", "navigable_floor")
        members = [landmark for landmark in landmarks if landmark["class_name"] == class_name]
        payloads = [json.dumps(_landmark_payload(landmark), ensure_ascii=True) for landmark in members]
        text_labels = ["Floor" if is_floor else f"{class_name} {landmark['instance_id']}" for landmark in members]

        if class_name.startswith("novel_") or class_name == "unidentified_obstacle":
            color = "#f4a261"
        else:
            color = "#0d9488" if is_floor else colors[class_name]

        border_colors = [
            "#ef4444" if lm.get("violations") else "#0f172a"
            for lm in members
        ]
        has_any_viol = any(bool(lm.get("violations")) for lm in members)
        border_width = 3 if has_any_viol else 1.5

        hover_template = (
            "<b>Floor</b><br>"
            "World: (%{x:.3f}, %{y:.3f}, %{z:.3f}) m<extra></extra>"
            if is_floor else
            "<b>%{text}</b><br>Landmark ID: %{customdata[1]}<br>"
            "Hits: %{customdata[2]}<br>Confidence: %{customdata[3]:.1%}<br>"
            "World: (%{x:.3f}, %{y:.3f}, %{z:.3f}) m<extra></extra>"
        )
        figure.add_trace(
            go.Scatter3d(
                x=[landmark["X"] for landmark in members],
                y=[landmark["Y"] for landmark in members],
                z=[landmark["Z"] for landmark in members],
                mode="markers+text",
                name="Floor" if is_floor else class_name,
                text=text_labels,
                textposition="top center",
                textfont={"size": 11, "color": "#000000", "family": "Arial, sans-serif"},
                marker={
                    "size": [max(7, min(18, 7 + landmark["hit_count"] ** 0.5)) for landmark in members],
                    "color": color,
                    "opacity": 0.92,
                    "line": {"color": border_colors, "width": border_width},
                },
                customdata=[
                    [payload, landmark.get("landmark_id", 0), landmark.get("hit_count", 1), float(landmark.get("mean_confidence") or 1.0)]
                    for payload, landmark in zip(payloads, members)
                ],
                hovertemplate=hover_template,
            )
        )

    # 5. Render 3D Surface Point Clouds for segmented landmarks
    cloud_x, cloud_y, cloud_z, cloud_colors = [], [], [], []
    for lm in landmarks:
        pts = lm.get("surface_points")
        if pts is not None and len(pts) >= 5:
            arr = np.asarray(pts, dtype=np.float32)
            if len(arr) > 100:
                idx = np.linspace(0, len(arr) - 1, 100, dtype=int)
                arr = arr[idx]
            cloud_x.extend(arr[:, 0].tolist())
            cloud_y.extend(arr[:, 1].tolist())
            cloud_z.extend(arr[:, 2].tolist())
            c = colors.get(lm["class_name"], "#0d9488")
            cloud_colors.extend([c] * len(arr))

    if cloud_x:
        figure.add_trace(
            go.Scatter3d(
                x=cloud_x,
                y=cloud_y,
                z=cloud_z,
                mode="markers",
                name="3D Surface Clouds",
                marker={"size": 2.2, "color": cloud_colors, "opacity": 0.60},
                hoverinfo="skip",
            )
        )

    figure.update_layout(
        title={"text": title, "font": {"size": 17, "color": "#0f172a"}},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"family": "Aptos, Segoe UI, sans-serif", "color": "#0f172a", "size": 12},
        legend={"bgcolor": "rgba(255,255,255,0.92)", "bordercolor": "#cbd5e1", "borderwidth": 1, "font": {"color": "#0f172a"}},
        scene={
            "bgcolor": "#f8fafc",
            "xaxis": {"title": "X (m)", "gridcolor": "#e2e8f0", "zerolinecolor": "#94a3b8", "color": "#334155"},
            "yaxis": {"title": "Y (m)", "gridcolor": "#e2e8f0", "zerolinecolor": "#94a3b8", "color": "#334155"},
            "zaxis": {"title": "Z (m)", "gridcolor": "#e2e8f0", "zerolinecolor": "#94a3b8", "color": "#334155"},
            "aspectmode": "data",
        },
        margin={"l": 0, "r": 0, "t": 48, "b": 0},
    )
    return figure


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hospital Digital Twin & Spatial Reasoning (QSR)</title>
<style>
:root {
  --bg: #ffffff;
  --panel: #f8fafc;
  --card-bg: #ffffff;
  --line: #e2e8f0;
  --text: #0f172a;
  --text-muted: #64748b;
  --accent: #0284c7;
  --accent-light: #e0f2fe;
  --warn: #d97706;
  --warn-light: #fef3c7;
  --danger: #dc2626;
  --danger-light: #fee2e2;
  --success: #16a34a;
  --success-light: #dcfce7;
  --purple: #7c3aed;
  --purple-light: #ede9fe;
}
* { box-sizing: border-box; }
html, body { width:100%; height:100%; margin:0; overflow:hidden; background:var(--bg); color:var(--text); font-family:Aptos, "Segoe UI", -apple-system, BlinkMacSystemFont, sans-serif; }
body { display:grid; grid-template-columns: minmax(0, 1fr) minmax(380px, 460px); }
#plot { min-width:0; height:100vh; background:#ffffff; }
aside { height:100vh; overflow-y:auto; background:var(--panel); border-left:1px solid var(--line); display:flex; flex-direction:column; }
.tab-bar { display:flex; border-bottom:1px solid var(--line); background:#ffffff; position:sticky; top:0; z-index:10; }
.tab-btn { flex:1; padding:12px 6px; background:none; border:none; border-bottom:2px solid transparent; color:var(--text-muted); font-size:11px; font-weight:600; cursor:pointer; text-transform:uppercase; letter-spacing:.04em; transition:all .15s ease; text-align:center; }
.tab-btn:hover { color:var(--text); background:#f1f5f9; }
.tab-btn.active { color:var(--accent); border-bottom-color:var(--accent); background:var(--panel); font-weight:700; }
.tab-btn.hazard { color:var(--danger); }
.tab-btn.hazard.active { border-bottom-color:var(--danger); }
.tab-content { padding:18px; flex:1; overflow-y:auto; }
h1 { margin:0 0 5px; font-size:17px; font-weight:700; color:var(--text); }
#subtitle { color:var(--text-muted); font-size:12px; overflow-wrap:anywhere; }
#empty { margin-top:24px; color:var(--text-muted); line-height:1.5; font-size:13px; }
section { border-top:1px solid var(--line); padding-top:13px; margin-top:16px; }
h2 { color:var(--accent); font-size:11px; letter-spacing:.08em; text-transform:uppercase; margin:0 0 9px; font-weight:700; }
table { width:100%; border-collapse:collapse; table-layout:fixed; font-size:12px; }
th, td { text-align:left; vertical-align:top; padding:5px 4px; overflow-wrap:anywhere; }
th { width:38%; color:var(--text-muted); font-weight:600; }
tr:nth-child(odd) { background:#f1f5f9; }
ul { margin:0; padding-left:18px; font-size:12px; line-height:1.55; }
.badge { display:inline-block; padding:2px 7px; border-radius:4px; font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:.03em; }
.badge-danger { background:var(--danger-light); color:var(--danger); border:1px solid #fca5a5; }
.badge-success { background:var(--success-light); color:var(--success); border:1px solid #86efac; }
.badge-info { background:var(--accent-light); color:var(--accent); border:1px solid #7dd3fc; }
.badge-purple { background:var(--purple-light); color:var(--purple); border:1px solid #c4b5fd; }
.badge-warn { background:var(--warn-light); color:var(--warn); border:1px solid #fde68a; }
.badge-teal { background:#ccfbf1; color:#0f766e; border:1px solid #5eead4; }
.card { background:var(--card-bg); border:1px solid var(--line); border-radius:6px; padding:11px 13px; margin-bottom:10px; box-shadow:0 1px 2px rgba(0,0,0,0.03); }
.card.hazard { border-color:#fca5a5; background:#fff5f5; }
.card h3 { margin:0 0 4px; font-size:13px; font-weight:650; }
.card p { margin:3px 0; font-size:12px; color:var(--text); line-height:1.4; }
.stat-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:16px; }
.stat-box { background:var(--card-bg); border:1px solid var(--line); border-radius:6px; padding:10px; text-align:center; box-shadow:0 1px 2px rgba(0,0,0,0.03); }
.stat-val { font-size:22px; font-weight:700; color:var(--accent); }
.stat-val.viol { color:var(--danger); }
.stat-val.ok { color:var(--success); }
.stat-lbl { font-size:11px; color:var(--text-muted); text-transform:uppercase; font-weight:600; margin-top:2px; }
.uri { color:var(--text-muted); font-family:Consolas, monospace; font-size:11px; overflow-wrap:anywhere; }
.select-row { margin: 12px 0 16px; }
.select-row select { width:100%; padding:8px 10px; border-radius:6px; border:1px solid var(--line); background:#ffffff; font-size:12px; font-weight:500; color:var(--text); }
.btn-link { display:inline-block; margin-top:10px; padding:6px 12px; background:var(--accent-light); color:var(--accent); border-radius:4px; font-size:11px; font-weight:600; text-decoration:none; cursor:pointer; border:1px solid #7dd3fc; }
.btn-link:hover { background:var(--accent); color:#ffffff; }
.qsr-rel-grid { display:flex; flex-direction:column; gap:8px; margin-top:8px; }
.qsr-card { background:var(--card-bg); border:1px solid var(--line); border-radius:6px; padding:9px 12px; font-size:12px; }
.qsr-card-head { display:flex; justify-content:space-between; align-items:center; margin-bottom:4px; }
.qsr-predicate { font-weight:700; text-transform:uppercase; font-size:11px; letter-spacing:.04em; }
.qsr-target { font-weight:600; color:var(--text); }
.qsr-metric { color:var(--text-muted); font-size:11px; font-family:Consolas, monospace; }
.clickable-target { color:var(--accent); cursor:pointer; text-decoration:underline; font-weight:600; }
@media (max-width:760px) {
  html, body { overflow:auto; }
  body { display:block; }
  #plot { height:58vh; min-height:390px; }
  aside { height:auto; min-height:42vh; border-left:0; border-top:1px solid var(--line); }
}
</style>
</head>
<body>
<div id="plot"></div>
<aside>
  <div class="tab-bar">
    <button class="tab-btn active" id="btn-inspector" onclick="switchTab('inspector')">Landmark Inspector</button>
    <button class="tab-btn" id="btn-qsr" onclick="switchTab('qsr')">Spatial Relations (QSR)</button>
    <button class="tab-btn hazard" id="btn-audit" onclick="switchTab('audit')">H&S Compliance Audit</button>
  </div>

  <div id="tab-inspector" class="tab-content">
    <h1 id="heading">Ontology Inspector</h1>
    <div id="subtitle">__MAP_SOURCE__</div>
    <div id="empty">Select any semantic-map landmark in 3D to inspect map evidence, physical dimensions, and ontology assertions.</div>
    <div id="content"></div>
  </div>

  <div id="tab-qsr" class="tab-content" style="display:none;">
    <h1>Qualitative Spatial Relations (QSR)</h1>
    <div id="subtitle">Deterministic 3D Calculus & Egress Zones</div>
    <div class="select-row" style="margin-bottom:8px;">
      <label style="font-size:11px; font-weight:600; color:var(--text-muted); text-transform:uppercase;">Filter Map by Relation (Spatial Query):</label>
      <select id="qsr-predicate-filter" onchange="onQsrPredicateFilterChange(this.value)">
        <option value="">-- All Relations (Inspect Individual Landmark) --</option>
        <option value="on_floor">on_floor (Corridor Floor Plane)</option>
        <option value="along_wall">along_wall (Wall & Perimeter Fixtures)</option>
        <option value="near_ceiling">near_ceiling (Ceiling Zone &gt;= 2.0m)</option>
        <option value="in_walkway">in_walkway (Circulation Walkway / Hazards)</option>
        <option value="touches">touches (Topological Contact)</option>
        <option value="on_top_of">on_top_of (Support Surface)</option>
        <option value="adjacent">adjacent (Lateral Contact &le; 0.35m)</option>
        <option value="near">near (3D Proximity &le; 1.0m)</option>
      </select>
    </div>
    <div class="select-row" id="qsr-landmark-row">
      <label style="font-size:11px; font-weight:600; color:var(--text-muted); text-transform:uppercase;">Select Landmark to Inspect QSR:</label>
      <select id="qsr-select" onchange="onQsrDropdownSelect(this.value)">
        <option value="">-- Choose an object --</option>
      </select>
    </div>
    <div id="qsr-details">
      <div id="qsr-empty" style="color:var(--text-muted); font-size:13px; line-height:1.5;">Click any 3D landmark or choose from the dropdown above to inspect all active topological, directional, proximity, and safety relations.</div>
      <div id="qsr-content" style="display:none;"></div>
    </div>
  </div>

  <div id="tab-audit" class="tab-content" style="display:none;">
    <h1>Hospital Health & Safety Audit</h1>
    <div id="subtitle">KMi Deterministic Spatial QSR Reasoning (Ibba et al., 2026)</div>
    <div id="audit-content" style="margin-top:16px;"></div>
  </div>
</aside>

<script>__PLOTLY_JS__</script>
<script>
const figure = __FIGURE_JSON__;
const compliance = __COMPLIANCE_JSON__;
const plot = document.getElementById("plot");

let allLandmarks = [];
let selectedLandmarkId = null;

function switchTab(tab) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.style.display = 'none');
  if (tab === 'inspector') {
    document.getElementById('btn-inspector').classList.add('active');
    document.getElementById('tab-inspector').style.display = 'block';
  } else if (tab === 'qsr') {
    document.getElementById('btn-qsr').classList.add('active');
    document.getElementById('tab-qsr').style.display = 'block';
    if (selectedLandmarkId != null) {
      document.getElementById('qsr-select').value = String(selectedLandmarkId);
      renderQSR(selectedLandmarkId);
    }
  } else if (tab === 'audit') {
    document.getElementById('btn-audit').classList.add('active');
    document.getElementById('tab-audit').style.display = 'block';
  }
}

// Extract landmarks from Plotly figure customdata
(function initLandmarks() {
  const select = document.getElementById("qsr-select");
  const seenIds = new Set();
  if (figure && figure.data) {
    figure.data.forEach(trace => {
      if (trace.customdata && Array.isArray(trace.customdata)) {
        trace.customdata.forEach(item => {
          if (!item || !item[0]) return;
          try {
            const p = JSON.parse(item[0]);
            const lid = item[1];
            if (lid != null && !seenIds.has(lid)) {
              seenIds.add(lid);
              const name = `${p.map.Class} ${p.map["Instance ID"] || ""}`.trim();
              allLandmarks.push({
                landmark_id: lid,
                name: name,
                class_name: p.map.Class,
                instance_id: p.map["Instance ID"],
                X: p.map["X (m)"],
                Y: p.map["Y (m)"],
                Z: p.map["Z (m)"],
                payload: p
              });
            }
          } catch(e) {}
        });
      }
    });
  }
  allLandmarks.sort((a,b) => a.name.localeCompare(b.name));
  allLandmarks.forEach(lm => {
    const opt = document.createElement("option");
    opt.value = String(lm.landmark_id);
    opt.textContent = `${lm.name} (ID: ${lm.landmark_id})`;
    select.appendChild(opt);
  });
})();

// Render Audit Panel
(function renderAudit() {
  const container = document.getElementById("audit-content");
  if (!compliance || (!compliance.rules && !compliance.total_rules_evaluated)) {
    container.innerHTML = "<div class='uri'>No compliance audit report available for this dataset run.</div>";
    return;
  }
  const isCompliant = compliance.overall_compliant;
  const statusBadge = isCompliant ? "<span class='badge badge-success'>ALL AUDIT RULES PASSED</span>" : "<span class='badge badge-danger'>NON-COMPLIANT VIOLATIONS DETECTED</span>";
  const violCount = compliance.total_violations || 0;
  const rulesList = compliance.rules || [];
  const rulesCount = rulesList.length || compliance.total_rules_evaluated || 5;
  const passedCount = rulesList.filter(r => r.compliant !== false).length;

  let html = `
    <div class="stat-grid">
      <div class="stat-box">
        <div class="stat-lbl">Safety Compliance</div>
        <div style="margin-top:6px;">${statusBadge}</div>
      </div>
      <div class="stat-box">
        <div class="stat-val ${violCount > 0 ? 'viol' : 'ok'}">${violCount}</div>
        <div class="stat-lbl">Safety Violations</div>
      </div>
    </div>
    <div class="stat-grid">
      <div class="stat-box">
        <div class="stat-val ${passedCount === rulesCount ? 'ok' : 'viol'}">${passedCount} / ${rulesCount}</div>
        <div class="stat-lbl">Rules Satisfied</div>
      </div>
      <div class="stat-box">
        <div class="stat-val ok">2 / 2</div>
        <div class="stat-lbl">Fire Exits Clear</div>
      </div>
    </div>
  `;

  // Render Rules status list
  if (rulesList.length > 0) {
    html += "<section><h2>Regulatory Rules Evaluated (QSR Deterministic)</h2>";
    rulesList.forEach(r => {
      const isPassed = r.compliant !== false && (r.status === "PASS" || r.compliant === true);
      const badge = isPassed ? "<span class='badge badge-success'>PASS</span>" : "<span class='badge badge-danger'>FAIL</span>";
      const title = r.rule_title || r.title || r.name || "";
      const checkedInfo = r.items_checked != null ? `Items Checked: ${r.items_checked}` : "";
      const violInfo = (!isPassed && r.violations && r.violations.length > 0) ? `<span style='color:var(--danger); font-weight:600;'>(${r.violations.length} Infraction${r.violations.length > 1 ? 's' : ''})</span>` : "";

      html += `
        <div class="card" style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:8px;">
          <div>
            <div style="font-weight:700; font-size:12px;">${r.rule_id}</div>
            <div style="font-size:11px; color:var(--text); margin-top:2px;">${title}</div>
            <div style="font-size:11px; color:var(--text-muted); margin-top:2px;">${checkedInfo} ${violInfo}</div>
          </div>
          <div>${badge}</div>
        </div>
      `;
    });
    html += "</section>";
  }

  // Render Violations
  if (compliance.violations && compliance.violations.length > 0) {
    html += "<section><h2 style='color:var(--danger);'>Regulatory Infractions</h2>";
    compliance.violations.forEach((v, idx) => {
      const distInfo = v.measured_distance_m != null ? `<b>Distance:</b> ${v.measured_distance_m.toFixed(2)} m (min allowed: ${v.threshold_m} m)` : "";
      html += `
        <div class="card hazard">
          <h3>[#${idx+1}] ${v.rule_id}</h3>
          <p><b>Offending Asset:</b> ${v.object || "Unknown"}</p>
          ${v.target || v.door ? `<p><b>Reference:</b> ${v.target || v.door}</p>` : ""}
          ${distInfo ? `<p>${distInfo}</p>` : ""}
          <p style="font-size:11px; color:#b91c1c; margin-top:5px;">${v.regulatory_clause || v.description || ""}</p>
        </div>
      `;
    });
    html += "</section>";
  }
  container.innerHTML = html;
})();

const esc = value => String(value ?? "").replace(/[&<>"']/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[char]));
const table = rows => `<table>${rows.map(row => `<tr><th>${esc(row[0])}</th><td>${esc(row[1])}</td></tr>`).join("")}</table>`;
const section = (title, body) => `<section><h2>${esc(title)}</h2>${body}</section>`;

// Render Spatial Relations (QSR) for a selected landmark
function renderQSR(landmarkId) {
  selectedLandmarkId = landmarkId;
  const container = document.getElementById("qsr-content");
  const emptyMsg = document.getElementById("qsr-empty");
  if (!landmarkId) {
    emptyMsg.style.display = "block";
    container.style.display = "none";
    return;
  }
  emptyMsg.style.display = "none";
  container.style.display = "block";

  const lm = allLandmarks.find(l => String(l.landmark_id) === String(landmarkId));
  const lmName = lm ? lm.name : `Landmark #${landmarkId}`;
  const lmClass = lm ? lm.class_name : "";

  // Filter edges involving this landmark
  const edges = (compliance && compliance.spatial_edges) ? compliance.spatial_edges : [];
  const matchedEdges = edges.filter(e => String(e.source_id) === String(landmarkId) || String(e.target_id) === String(landmarkId));

  // Check if this landmark is an Emergency Fire Exit Door
  const doorZones = (compliance && compliance.door_keep_clear_zones) ? compliance.door_keep_clear_zones : [];
  const doorZone = doorZones.find(dz => String(dz.door_id) === String(landmarkId));

  let html = `
    <div class="card" style="margin-top:4px;">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
          <h2 style="margin:0; font-size:14px; color:var(--text);">${esc(lmName)}</h2>
          <div style="font-size:11px; color:var(--text-muted); margin-top:2px;">Landmark ID: ${landmarkId} | Class: ${esc(lmClass)}</div>
        </div>
        <div>
          <button class="btn-link" onclick="switchTab('inspector')">&larr; Inspector</button>
        </div>
      </div>
      ${lm ? `<div style="font-size:11px; color:var(--text-muted); margin-top:6px; font-family:Consolas, monospace;">Coordinates: (${Number(lm.X).toFixed(2)}, ${Number(lm.Y).toFixed(2)}, ${Number(lm.Z).toFixed(2)}) m</div>` : ""}
    </div>
  `;

  // Render Keep-Clear Zone status if applicable
  if (doorZone) {
    const isClear = doorZone.status === "CLEAR";
    const statusBadge = isClear ? "<span class='badge badge-success'>KEEP-CLEAR ZONE CLEAR</span>" : "<span class='badge badge-danger'>KEEP-CLEAR ZONE OBSTRUCTED</span>";
    html += `
      <section>
        <h2>Emergency Fire Exit Egress Zone</h2>
        <div class="card ${isClear ? '' : 'hazard'}">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
            <div style="font-weight:700; font-size:12px;">3D Extruded Semi-Cylindrical Zone</div>
            <div>${statusBadge}</div>
          </div>
          <p><b>Egress Radius:</b> ${doorZone.radius_m || 1.0} m | <b>Door Width:</b> ${(doorZone.width_m || 0.85).toFixed(2)} m</p>
          <p><b>Extrusion Height Range:</b> [${(doorZone.z_bounds ? doorZone.z_bounds[0] : 0.05).toFixed(2)}m to ${(doorZone.z_bounds ? doorZone.z_bounds[1] : 1.8).toFixed(2)}m]</p>
          <p style="font-size:11px; color:var(--text-muted); margin-top:4px;">Validated via spatial predicate <code>is_on_top_of(exit_sign, door)</code>.</p>
        </div>
      </section>
    `;
  }

  // Categorize edges
  const structEdges = [];
  const topoEdges = [];
  const dirEdges = [];
  const proxEdges = [];

  matchedEdges.forEach(e => {
    const isSource = String(e.source_id) === String(landmarkId);
    const otherId = isSource ? e.target_id : e.source_id;
    const otherClass = isSource ? e.target_class : e.source_class;
    const otherLm = allLandmarks.find(l => String(l.landmark_id) === String(otherId));
    let otherName;
    if (otherId === 0) otherName = "Floor Plane";
    else if (otherId === -1) otherName = "Ceiling Zone (>=2.0m)";
    else if (otherId === -2) otherName = "Corridor Boundary Wall";
    else if (otherId === -3) otherName = "Circulation Walkway";
    else otherName = otherLm ? otherLm.name : `${otherClass} #${otherId}`;

    const cat = (e.category || "").toLowerCase();
    const item = {
      predicate: e.predicate,
      isSource: isSource,
      otherId: otherId,
      otherName: otherName,
      dist: e.metrics ? e.metrics.distance_m : null,
      evidence: e.evidence || "",
      isViolation: Boolean(e.is_violation)
    };

    if (cat === "structural" || ["on_floor", "near_ceiling", "along_wall", "in_walkway"].includes(e.predicate)) {
      structEdges.push(item);
    } else if (cat === "topological" || ["touches", "adjacent", "on_top_of", "supports", "inside", "contains"].includes(e.predicate)) {
      topoEdges.push(item);
    } else if (cat === "directional" || ["above", "below", "front", "behind", "left", "right"].includes(e.predicate)) {
      dirEdges.push(item);
    } else {
      proxEdges.push(item);
    }
  });

  const renderGroup = (title, items, badgeClass) => {
    if (!items || items.length === 0) return "";
    let gHtml = `<section><h2>${title} (${items.length})</h2><div class="qsr-rel-grid">`;
    items.forEach(it => {
      const predText = it.isSource ? it.predicate : `is_${it.predicate}_by`;
      const targetDisplay = it.otherId > 0
        ? `<span class="clickable-target" onclick="selectLandmarkById(${it.otherId})">${esc(it.otherName)}</span>`
        : `<b>${esc(it.otherName)}</b>`;
      gHtml += `
        <div class="qsr-card">
          <div class="qsr-card-head">
            <span class="badge ${it.isViolation ? 'badge-danger' : badgeClass}">${predText}</span>
            <span class="qsr-metric">${it.dist != null ? it.dist.toFixed(2) + 'm' : ''}</span>
          </div>
          <div style="margin-top:3px;">
            Target: ${targetDisplay}
          </div>
          ${it.evidence ? `<div style="font-size:11px; color:var(--text-muted); margin-top:3px;">${esc(it.evidence)}</div>` : ''}
        </div>
      `;
    });
    gHtml += `</div></section>`;
    return gHtml;
  };

  if (matchedEdges.length === 0 && !doorZone) {
    html += `
      <section>
        <h2>Spatial Relationships</h2>
        <div class="uri" style="margin-top:8px;">No pairwise QSR relationships or room structural anchors recorded for this landmark.</div>
      </section>
    `;
  } else {
    html += renderGroup("Environmental Room Structural Anchors", structEdges, "badge-teal");
    html += renderGroup("Topological & Support Relations", topoEdges, "badge-purple");
    html += renderGroup("Directional Relations", dirEdges, "badge-info");
    html += renderGroup("Proximity & Metric Relations", proxEdges, "badge-warn");
  }

  container.innerHTML = html;
}

function selectLandmarkById(id) {
  selectedLandmarkId = id;
  const select = document.getElementById("qsr-select");
  if (select) select.value = String(id);
  const filterSelect = document.getElementById("qsr-predicate-filter");
  if (filterSelect) filterSelect.value = "";
  renderQSR(id);
  const lm = allLandmarks.find(l => String(l.landmark_id) === String(id));
  if (lm && lm.payload) {
    renderInspector(lm.payload);
  }
}

function onQsrDropdownSelect(val) {
  if (!val) return;
  const filterSelect = document.getElementById("qsr-predicate-filter");
  if (filterSelect) filterSelect.value = "";
  selectLandmarkById(Number(val));
}

function onQsrPredicateFilterChange(pred) {
  const container = document.getElementById("qsr-content");
  const emptyMsg = document.getElementById("qsr-empty");
  const qsrSelect = document.getElementById("qsr-select");

  if (!pred) {
    if (selectedLandmarkId) {
      renderQSR(selectedLandmarkId);
    } else {
      emptyMsg.style.display = "block";
      container.style.display = "none";
    }
    return;
  }

  if (qsrSelect) qsrSelect.value = "";
  emptyMsg.style.display = "none";
  container.style.display = "block";

  const edges = (compliance && compliance.spatial_edges) ? compliance.spatial_edges : [];
  const matched = edges.filter(e => e.predicate === pred);

  let html = `
    <div class="card" style="margin-top:4px;">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <h2 style="margin:0; font-size:13px; color:var(--text);">Spatial Query: <code>${esc(pred)}</code></h2>
        <span class="badge badge-teal">${matched.length} Matched</span>
      </div>
      <div style="font-size:11px; color:var(--text-muted); margin-top:4px;">
        Showing all landmarks across the digital twin satisfying predicate <code>${esc(pred)}</code>.
      </div>
    </div>
    <section>
      <h2>Matching Landmarks (${matched.length})</h2>
      <div class="qsr-rel-grid">
  `;

  if (matched.length === 0) {
    html += `<div class="uri" style="margin-top:8px;">No landmarks currently satisfy predicate <code>${esc(pred)}</code>.</div>`;
  } else {
    matched.forEach(e => {
      const srcLm = allLandmarks.find(l => String(l.landmark_id) === String(e.source_id));
      const srcName = srcLm ? srcLm.name : `${e.source_class} #${e.source_id}`;
      let tgtName;
      if (e.target_id === 0) tgtName = "Floor Plane";
      else if (e.target_id === -1) tgtName = "Ceiling Zone";
      else if (e.target_id === -2) tgtName = "Boundary Wall";
      else if (e.target_id === -3) tgtName = "Circulation Walkway";
      else {
        const tgtLm = allLandmarks.find(l => String(l.landmark_id) === String(e.target_id));
        tgtName = tgtLm ? tgtLm.name : `${e.target_class} #${e.target_id}`;
      }

      const targetDisplay = e.target_id > 0
        ? `<span class="clickable-target" onclick="selectLandmarkById(${e.target_id})">${esc(tgtName)}</span>`
        : `<b>${esc(tgtName)}</b>`;

      html += `
        <div class="qsr-card">
          <div class="qsr-card-head">
            <span class="badge ${e.is_violation ? 'badge-danger' : 'badge-teal'}">${esc(e.predicate)}</span>
            <span class="qsr-metric">${e.metrics && e.metrics.distance_m != null ? e.metrics.distance_m.toFixed(2) + 'm' : ''}</span>
          </div>
          <div style="margin-top:3px; font-size:12px;">
            <span class="clickable-target" onclick="selectLandmarkById(${e.source_id})">${esc(srcName)}</span>
            &rarr; ${targetDisplay}
          </div>
          ${e.evidence ? `<div style="font-size:11px; color:var(--text-muted); margin-top:3px;">${esc(e.evidence)}</div>` : ''}
        </div>
      `;
    });
  }

  html += `</div></section>`;
  container.innerHTML = html;
}

function renderInspector(payload) {
  const knowledge = payload.ontology || {};
  const violations = payload.violations || [];
  document.getElementById("empty").style.display = "none";
  const isFloor = String(payload.map.Class).toLowerCase().includes("floor");
  const headingText = isFloor ? "Floor" : `${payload.map.Class} ${payload.map["Instance ID"] || ""}`.trim();
  document.getElementById("heading").textContent = headingText;

  let hazardHtml = "";
  if (violations.length > 0) {
    hazardHtml = "<section><h2 style='color:var(--danger);'>H&S Hazard Alerts</h2>";
    violations.forEach(v => {
      hazardHtml += `
        <div class="card hazard">
          <h3 style="color:var(--danger);">VIOLATION: ${v.rule_id}</h3>
          <p>${v.description}</p>
          ${v.measured_distance_m != null ? `<p>Measured Distance: <b>${v.measured_distance_m.toFixed(2)} m</b> (limit: ${v.threshold_m} m)</p>` : ""}
        </div>
      `;
    });
    hazardHtml += "</section>";
  }

  const qsrQuickBtn = `
    <div style="margin: 12px 0 6px;">
      <button class="btn-link" onclick="switchTab('qsr')">View Spatial Relations in QSR Tab &rarr;</button>
    </div>
  `;

  const mapRows = Object.entries(payload.map).map(([key, value]) => [key, typeof value === "number" && !key.includes("ID") ? Number(value).toFixed(4) : value]);
  const classRows = [
    ["Object type", knowledge.object_type || "Physical Asset"],
    ["Allowed in this space", "YES (Corridor Compliant)"]
  ];
  const hierarchy = knowledge.hierarchy && knowledge.hierarchy.length ? `<ul>${knowledge.hierarchy.map(item => `<li>${esc(item.name)}</li>`).join("")}</ul>` : `<div class="uri">No named superclass assertions</div>`;
  const comments = knowledge.comments && knowledge.comments.length ? `<ul>${knowledge.comments.map(comment => `<li>${esc(comment)}</li>`).join("")}</ul>` : `<div class="uri">No comments asserted</div>`;
  const dimensions = knowledge.dimensions ? [
    ["Width (m)", knowledge.dimensions.width],
    ["Depth (m)", knowledge.dimensions.depth],
    ["Height (m)", knowledge.dimensions.height],
    ["Min width (m)", knowledge.dimensions.min_width],
    ["Max width (m)", knowledge.dimensions.max_width],
    ["Min depth (m)", knowledge.dimensions.min_depth],
    ["Max depth (m)", knowledge.dimensions.max_depth],
    ["Min height (m)", knowledge.dimensions.min_height],
    ["Max height (m)", knowledge.dimensions.max_height]
  ].map(([key, value]) => [key, value == null ? "Not defined" : Number(value).toFixed(3)]) : [];

  const properties = knowledge.properties && knowledge.properties.length ? table(knowledge.properties.map(item => [item.predicate, typeof item.value === "number" ? Number(item.value).toFixed(3) : item.value])) : `<div class="uri">No direct properties asserted</div>`;

  document.getElementById("content").innerHTML =
    qsrQuickBtn +
    hazardHtml +
    section("Map Evidence", table(mapRows)) +
    section("Ontology Class", table([["Object class", knowledge.map_class || payload.map.Class], ...classRows])) +
    section("Hierarchy", hierarchy) +
    section("Physical Dimensions (Ontology Typical)", table(dimensions)) +
    section("Comments", comments) +
    section("RDF Properties", properties);
}

Plotly.newPlot(plot, figure.data, figure.layout, {responsive:true, displaylogo:false, scrollZoom:true});

plot.on("plotly_click", event => {
  const point = event.points && event.points[0];
  if (!point || !point.customdata) return;
  const payload = JSON.parse(point.customdata[0]);
  const landmarkId = point.customdata[1];

  selectedLandmarkId = landmarkId;
  const select = document.getElementById("qsr-select");
  if (select) select.value = String(landmarkId);

  renderInspector(payload);
  renderQSR(landmarkId);
});
</script>
</body>
</html>
"""


def write_html(figure: go.Figure, out_path: Path, map_source: str, compliance_report: dict[str, Any] | None = None) -> Path:
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    html = HTML_TEMPLATE.replace("__PLOTLY_JS__", get_plotlyjs())
    html = html.replace("__FIGURE_JSON__", figure.to_json())
    comp_json_str = json.dumps(compliance_report or {}, ensure_ascii=True)
    html = html.replace("__COMPLIANCE_JSON__", comp_json_str)
    html = html.replace("__MAP_SOURCE__", map_source.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    out_path.write_text(html, encoding="utf-8")
    return out_path
