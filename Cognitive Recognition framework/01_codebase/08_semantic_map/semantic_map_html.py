"""Plotly HTML renderer for ontology-enriched semantic-map landmarks."""

from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

PALETTE = [
    "#d1495b", "#00798c", "#edae49", "#30638e", "#5f6f52", "#8f5d5d",
    "#6a4c93", "#2a9d8f", "#bc6c25", "#457b9d", "#9c6644", "#577590",
]


def _landmark_payload(landmark: dict) -> dict:
    return {
        "map": {
            "Class": landmark["class_name"],
            "Instance ID": int(landmark["instance_id"]),
            "Landmark ID": int(landmark["landmark_id"]),
            "World frame": landmark["world_frame"],
            "X (m)": landmark["X"],
            "Y (m)": landmark["Y"],
            "Z (m)": landmark["Z"],
            "Last observed": landmark.get("last_seen") or landmark.get("last_seen_ns"),
        },
        "ontology": landmark["ontology"],
    }


def build_figure(
    landmarks: list[dict],
    trajectory: list[tuple[float, float, float]],
    title: str,
) -> go.Figure:
    class_names = sorted({landmark["class_name"] for landmark in landmarks})
    colors = {name: PALETTE[index % len(PALETTE)] for index, name in enumerate(class_names)}
    figure = go.Figure()

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

    for class_name in class_names:
        members = [landmark for landmark in landmarks if landmark["class_name"] == class_name]
        payloads = [json.dumps(_landmark_payload(landmark), ensure_ascii=True) for landmark in members]
        figure.add_trace(
            go.Scatter3d(
                x=[landmark["X"] for landmark in members],
                y=[landmark["Y"] for landmark in members],
                z=[landmark["Z"] for landmark in members],
                mode="markers+text",
                name=class_name,
                text=[f"{class_name} {landmark['instance_id']}" for landmark in members],
                textposition="top center",
                textfont={"size": 10},
                marker={
                    "size": [max(7, min(18, 7 + landmark["hit_count"] ** 0.5)) for landmark in members],
                    "color": colors[class_name],
                    "opacity": 0.92,
                    "line": {"color": "#111820", "width": 1},
                },
                customdata=[
                    [payload, landmark["landmark_id"], landmark["hit_count"], landmark["mean_confidence"]]
                    for payload, landmark in zip(payloads, members)
                ],
                hovertemplate=(
                    "<b>%{text}</b><br>Landmark %{customdata[1]}<br>"
                    "Hits: %{customdata[2]}<br>Confidence: %{customdata[3]:.1%}<br>"
                    "World: (%{x:.3f}, %{y:.3f}, %{z:.3f}) m<extra></extra>"
                ),
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
<title>Semantic Map Ontology Inspector</title>
<style>
:root { --bg:#101820; --panel:#17232c; --line:#30404d; --text:#dce7e9; --muted:#8fa3ad; --accent:#4fb3a5; --warn:#edae49; }
* { box-sizing:border-box; }
html, body { width:100%; height:100%; margin:0; overflow:hidden; background:var(--bg); color:var(--text); font-family:Aptos,"Segoe UI",sans-serif; }
body { display:grid; grid-template-columns:minmax(0,1fr) minmax(320px,390px); }
#plot { min-width:0; height:100vh; }
aside { height:100vh; overflow-y:auto; background:var(--panel); border-left:1px solid var(--line); padding:18px; }
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
.status { display:inline-block; color:#071315; background:var(--accent); padding:2px 6px; border-radius:3px; font-size:11px; text-transform:uppercase; }
.status.fallback { background:var(--warn); }
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
  <h1 id="heading">Ontology Inspector</h1>
  <div id="subtitle">__MAP_SOURCE__</div>
  <div id="empty">Select a semantic-map landmark to inspect its map evidence and ontology knowledge.</div>
  <div id="content"></div>
</aside>
<script>__PLOTLY_JS__</script>
<script>
const figure = __FIGURE_JSON__;
const plot = document.getElementById("plot");
const esc = value => String(value ?? "").replace(/[&<>"']/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[char]));
const table = rows => `<table>${rows.map(row => `<tr><th>${esc(row[0])}</th><td>${esc(row[1])}</td></tr>`).join("")}</table>`;
const section = (title, body) => `<section><h2>${esc(title)}</h2>${body}</section>`;
const flatten = (value, prefix="") => {
  const rows = [];
  if (Array.isArray(value)) value.forEach((item, index) => rows.push(...flatten(item, `${prefix}[${index}]`)));
  else if (value && typeof value === "object") Object.keys(value).sort().forEach(key => rows.push(...flatten(value[key], prefix ? `${prefix}.${key}` : key)));
  else rows.push([prefix, value]);
  return rows;
};
Plotly.newPlot(plot, figure.data, figure.layout, {responsive:true, displaylogo:false, scrollZoom:true});
plot.on("plotly_click", event => {
  const point = event.points && event.points[0];
  if (!point || !point.customdata) return;
  const payload = JSON.parse(point.customdata[0]);
  const knowledge = payload.ontology;
  document.getElementById("empty").style.display = "none";
  document.getElementById("heading").textContent = `${payload.map.Class} ${payload.map["Instance ID"]}`;
    const mapRows = Object.entries(payload.map).map(([key, value]) => [key, typeof value === "number" && !key.includes("ID") ? Number(value).toFixed(4) : value]);
  const classRows = [
        ["Object type", knowledge.object_type],
        ["Allowed in this space", "UNKNOWN"]
  ];
    const hierarchy = knowledge.hierarchy.length ? `<ul>${knowledge.hierarchy.map(item => `<li>${esc(item.name)}</li>`).join("")}</ul>` : `<div class="uri">No named superclass assertions</div>`;
    const comments = knowledge.comments.length ? `<ul>${knowledge.comments.map(comment => `<li>${esc(comment)}</li>`).join("")}</ul>` : `<div class="uri">No comments asserted</div>`;
    const dimensions = [
        ["Width (m)", knowledge.dimensions.width],
        ["Depth (m)", knowledge.dimensions.depth],
        ["Height (m)", knowledge.dimensions.height],
        ["Min width (m)", knowledge.dimensions.min_width],
        ["Max width (m)", knowledge.dimensions.max_width],
        ["Min depth (m)", knowledge.dimensions.min_depth],
        ["Max depth (m)", knowledge.dimensions.max_depth],
        ["Min height (m)", knowledge.dimensions.min_height],
        ["Max height (m)", knowledge.dimensions.max_height]
    ].map(([key, value]) => [key, value == null ? "Not defined" : Number(value).toFixed(3)]);
    const properties = knowledge.properties.length ? table(knowledge.properties.map(item => [item.predicate, typeof item.value === "number" ? Number(item.value).toFixed(3) : item.value])) : `<div class="uri">No direct properties asserted</div>`;
  document.getElementById("content").innerHTML =
    section("Map Evidence", table(mapRows)) +
        section("Ontology Class", table([["Object class", knowledge.map_class], ...classRows])) +
    section("Hierarchy", hierarchy) +
        section("Physical Dimensions", table(dimensions)) +
    section("Comments", comments) +
    section("RDF Properties", properties);
});
</script>
</body>
</html>
"""


def write_html(figure: go.Figure, out_path: Path, map_source: str) -> Path:
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    html = HTML_TEMPLATE.replace("__PLOTLY_JS__", get_plotlyjs())
    html = html.replace("__FIGURE_JSON__", figure.to_json())
    html = html.replace("__MAP_SOURCE__", map_source.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    out_path.write_text(html, encoding="utf-8")
    return out_path
