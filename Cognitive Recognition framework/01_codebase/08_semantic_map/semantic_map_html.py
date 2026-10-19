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
        map_info["Instance ID"] = int(landmark["instance_id"])
        map_info["Landmark ID"] = int(landmark["landmark_id"])

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
                line={"color": "#8b949e", "width": 3, "dash": "dot"},
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
                    line={"color": "#f2545b", "width": 6},
                    marker={"size": 4, "color": "#f2545b"},
                    hovertemplate=(
                        f"<b>HAZARD: {rule_id}</b><br>"
                        f"{source_lbl} -> {target_lbl}<br>"
                        f"Distance: {dist_str}<extra></extra>"
                    ),
                )
            )

    # 4. Render Landmarks
    for class_name in class_names:
        is_floor = str(class_name).strip().lower() in ("floor", "navigable_floor")
        members = [landmark for landmark in landmarks if landmark["class_name"] == class_name]
        payloads = [json.dumps(_landmark_payload(landmark), ensure_ascii=True) for landmark in members]
        text_labels = ["Floor" if is_floor else f"{class_name} {landmark['instance_id']}" for landmark in members]

        # Use amber color for novel/obstacle classes
        if class_name.startswith("novel_") or class_name == "unidentified_obstacle":
            color = "#f4a261"
        else:
            color = "#4fb3a5" if is_floor else colors[class_name]

        # Border color: red if landmark has violations
        border_colors = [
            "#f2545b" if lm.get("violations") else "#111820"
            for lm in members
        ]
        has_any_viol = any(bool(lm.get("violations")) for lm in members)
        border_width = 3 if has_any_viol else 1

        hover_template = (
            "<b>Floor</b><br>"
            "World: (%{x:.3f}, %{y:.3f}, %{z:.3f}) m<extra></extra>"
            if is_floor else
            "<b>%{text}</b><br>Landmark %{customdata[1]}<br>"
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
                textfont={"size": 10},
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
            c = colors.get(lm["class_name"], "#4fb3a5")
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
        title={"text": title, "font": {"size": 18, "color": "#dce7e9"}},
        paper_bgcolor="#101820",
        plot_bgcolor="#101820",
        font={"family": "Aptos, Segoe UI, sans-serif", "color": "#dce7e9", "size": 12},
        legend={"bgcolor": "rgba(16,24,32,0.82)", "bordercolor": "#30404d", "borderwidth": 1},
        scene={
            "bgcolor": "#101820",
            "xaxis": {"title": "X (m)", "gridcolor": "#30404d", "zerolinecolor": "#70808c"},
            "yaxis": {"title": "Y (m)", "gridcolor": "#30404d", "zerolinecolor": "#70808c"},
            "zaxis": {"title": "Z (m)", "gridcolor": "#30404d", "zerolinecolor": "#70808c"},
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
<title>Hospital Digital Twin & Health & Safety Audit</title>
<style>
:root { --bg:#101820; --panel:#17232c; --line:#30404d; --text:#dce7e9; --muted:#8fa3ad; --accent:#4fb3a5; --warn:#edae49; --danger:#f2545b; }
* { box-sizing:border-box; }
html, body { width:100%; height:100%; margin:0; overflow:hidden; background:var(--bg); color:var(--text); font-family:Aptos,"Segoe UI",sans-serif; }
body { display:grid; grid-template-columns:minmax(0,1fr) minmax(360px,440px); }
#plot { min-width:0; height:100vh; }
aside { height:100vh; overflow-y:auto; background:var(--panel); border-left:1px solid var(--line); display:flex; flex-direction:column; }
.tab-bar { display:flex; border-bottom:1px solid var(--line); background:#121c23; }
.tab-btn { flex:1; padding:12px 10px; background:none; border:none; color:var(--muted); font-size:12px; font-weight:600; cursor:pointer; text-transform:uppercase; letter-spacing:.05em; }
.tab-btn.active { color:var(--text); border-bottom:2px solid var(--accent); background:var(--panel); }
.tab-btn.hazard { color:var(--danger); }
.tab-content { padding:18px; flex:1; overflow-y:auto; }
h1 { margin:0 0 5px; font-size:18px; font-weight:650; }
#subtitle { color:var(--muted); font-size:12px; overflow-wrap:anywhere; }
#empty { margin-top:32px; color:var(--muted); line-height:1.5; }
section { border-top:1px solid var(--line); padding-top:13px; margin-top:16px; }
h2 { color:var(--accent); font-size:11px; letter-spacing:.08em; text-transform:uppercase; margin:0 0 9px; }
table { width:100%; border-collapse:collapse; table-layout:fixed; font-size:12px; }
th, td { text-align:left; vertical-align:top; padding:5px 3px; overflow-wrap:anywhere; }
th { width:39%; color:var(--muted); font-weight:500; }
tr:nth-child(odd) { background:rgba(255,255,255,.025); }
ul { margin:0; padding-left:18px; font-size:12px; line-height:1.55; }
.badge { display:inline-block; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:600; text-transform:uppercase; }
.badge-danger { background:rgba(242,84,91,0.2); color:#ff6b72; border:1px solid #f2545b; }
.badge-success { background:rgba(79,179,165,0.2); color:#4fb3a5; border:1px solid #4fb3a5; }
.hazard-card { background:#221518; border:1px solid var(--danger); border-radius:6px; padding:10px 12px; margin-bottom:12px; }
.hazard-card h3 { margin:0 0 4px; font-size:13px; color:#ff6b72; }
.hazard-card p { margin:3px 0; font-size:12px; color:#e0c0c2; }
.stat-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:16px; }
.stat-box { background:#121d25; border:1px solid var(--line); border-radius:6px; padding:10px; text-align:center; }
.stat-val { font-size:22px; font-weight:700; color:var(--accent); }
.stat-val.viol { color:var(--danger); }
.stat-lbl { font-size:11px; color:var(--muted); text-transform:uppercase; }
.uri { color:var(--muted); font-family:Consolas,monospace; font-size:11px; overflow-wrap:anywhere; }
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
    <button class="tab-btn active" onclick="switchTab('inspector')">Landmark Inspector</button>
    <button class="tab-btn hazard" id="audit-tab-btn" onclick="switchTab('audit')">H&S Compliance Audit</button>
  </div>

  <div id="tab-inspector" class="tab-content">
    <h1 id="heading">Ontology Inspector</h1>
    <div id="subtitle">__MAP_SOURCE__</div>
    <div id="empty">Select a semantic-map landmark in 3D to inspect evidence, dimensions, and ontology knowledge.</div>
    <div id="content"></div>
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

function switchTab(tab) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.style.display = 'none');
  if (tab === 'inspector') {
    document.querySelector('.tab-btn:first-child').classList.add('active');
    document.getElementById('tab-inspector').style.display = 'block';
  } else {
    document.getElementById('audit-tab-btn').classList.add('active');
    document.getElementById('tab-audit').style.display = 'block';
  }
}

// Render Audit Panel
(function renderAudit() {
  const container = document.getElementById("audit-content");
  if (!compliance || !compliance.total_rules_evaluated) {
    container.innerHTML = "<div class='uri'>No compliance audit report available for this dataset run.</div>";
    return;
  }
  const isCompliant = compliance.overall_compliant;
  const statusBadge = isCompliant ? "<span class='badge badge-success'>COMPLIANT</span>" : "<span class='badge badge-danger'>NON-COMPLIANT</span>";
  const violCount = compliance.total_violations || 0;
  
  let html = `
    <div class="stat-grid">
      <div class="stat-box">
        <div class="stat-lbl">Safety Status</div>
        <div style="margin-top:4px;">${statusBadge}</div>
      </div>
      <div class="stat-box">
        <div class="stat-val ${violCount > 0 ? 'viol' : ''}">${violCount}</div>
        <div class="stat-lbl">Violations Detected</div>
      </div>
    </div>
  `;

  if (compliance.violations && compliance.violations.length > 0) {
    html += "<section><h2>Regulatory Infractions</h2>";
    compliance.violations.forEach((v, idx) => {
      const distInfo = v.measured_distance_m != null ? `<b>Distance:</b> ${v.measured_distance_m.toFixed(2)} m (min allowed: ${v.threshold_m} m)` : "";
      html += `
        <div class="hazard-card">
          <h3>[#${idx+1}] ${v.rule_id}</h3>
          <p><b>Offending Asset:</b> ${v.object || "Unknown"}</p>
          ${v.target || v.door ? `<p><b>Reference:</b> ${v.target || v.door}</p>` : ""}
          ${distInfo ? `<p>${distInfo}</p>` : ""}
          <p style="font-size:11px; color:#ff9da3; margin-top:5px;">${v.regulatory_clause || v.description || ""}</p>
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

Plotly.newPlot(plot, figure.data, figure.layout, {responsive:true, displaylogo:false, scrollZoom:true});

plot.on("plotly_click", event => {
  const point = event.points && event.points[0];
  if (!point || !point.customdata) return;
  const payload = JSON.parse(point.customdata[0]);
  const knowledge = payload.ontology;
  const violations = payload.violations || [];
  
  switchTab('inspector');
  document.getElementById("empty").style.display = "none";
  const isFloor = String(payload.map.Class).toLowerCase().includes("floor");
  document.getElementById("heading").textContent = isFloor ? "Floor" : `${payload.map.Class} ${payload.map["Instance ID"] || ""}`.trim();

  let hazardHtml = "";
  if (violations.length > 0) {
    hazardHtml = "<section><h2 style='color:#f2545b;'>H&S Hazard Alerts</h2>";
    violations.forEach(v => {
      hazardHtml += `
        <div class="hazard-card">
          <h3>VIOLATION: ${v.rule_id}</h3>
          <p>${v.description}</p>
          ${v.measured_distance_m != null ? `<p>Measured Distance: <b>${v.measured_distance_m.toFixed(2)} m</b> (limit: ${v.threshold_m} m)</p>` : ""}
        </div>
      `;
    });
    hazardHtml += "</section>";
  }

  const mapRows = Object.entries(payload.map).map(([key, value]) => [key, typeof value === "number" && !key.includes("ID") ? Number(value).toFixed(4) : value]);
  const classRows = [
    ["Object type", knowledge.object_type],
    ["Allowed in this space", "UNKNOWN"]
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
    hazardHtml +
    section("Map Evidence", table(mapRows)) +
    section("Ontology Class", table([["Object class", knowledge.map_class], ...classRows])) +
    section("Hierarchy", hierarchy) +
    section("Physical Dimensions (Ontology Typical)", table(dimensions)) +
    section("Comments", comments) +
    section("RDF Properties", properties);
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
