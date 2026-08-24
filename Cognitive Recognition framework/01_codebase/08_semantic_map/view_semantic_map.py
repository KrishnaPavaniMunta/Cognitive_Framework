"""
Viewer for semantic maps produced by build_semantic_map_from_bag.py.

Renders landmarks from a run's semantic_map.db as a 3D scatter with class labels,
plus a top-down floor plan. Coordinates are in the ROS world frame recorded in the
DB (X forward, Y left, Z up), so the top-down panel reads like a room layout.

Usage:
    python view_semantic_map.py                       # newest run under the default out-root
    python view_semantic_map.py --db <path to semantic_map.db>
    python view_semantic_map.py --min-hits 5 --observations --save map.png
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import webbrowser
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colormaps

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
DEFAULT_OUT_ROOT = PROJECT_ROOT / "04_outputs_runs_and_logs" / "outputs" / "semantic_maps"
ONTOLOGY_DIR = PROJECT_ROOT / "01_codebase" / "09_ontology"
DEFAULT_ONTOLOGY = ONTOLOGY_DIR / "ontology.rdf"

if str(ONTOLOGY_DIR) not in sys.path:
    sys.path.insert(0, str(ONTOLOGY_DIR))

from ontology_knowledge import OntologyKnowledgeBase  # noqa: E402


def find_latest_db(out_root: Path) -> Path:
    candidates = list(out_root.glob("*/world_map.db"))
    candidates.extend(out_root.glob("semanticmap_*/semantic_map.db"))
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No world_map.db or semantic_map.db found under {out_root}")
    return candidates[0]


def load_landmarks(conn: sqlite3.Connection, min_hits: int, classes: set[str] | None) -> list[dict]:
    available = {row[1] for row in conn.execute("PRAGMA table_info(semantic_map)")}
    required = ["landmark_id", "class_name", "world_frame", "X", "Y", "Z", "hit_count", "mean_confidence"]
    optional = ["instance_id", "max_confidence", "first_seen_ns", "last_seen_ns", "first_seen", "last_seen"]
    selected = required + [column for column in optional if column in available]
    rows = conn.execute(
        f"SELECT {', '.join(selected)} FROM semantic_map WHERE hit_count >= ? ORDER BY class_name, landmark_id",
        (int(min_hits),),
    ).fetchall()
    landmarks = [dict(zip(selected, row)) for row in rows]

    class_counts: dict[str, int] = {}
    for landmark in landmarks:
        class_name = landmark["class_name"]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        landmark.setdefault("instance_id", class_counts[class_name])
        landmark.setdefault("max_confidence", landmark["mean_confidence"])
        landmark.setdefault("first_seen_ns", None)
        landmark.setdefault("last_seen_ns", None)
        landmark.setdefault("first_seen", "")
        landmark.setdefault("last_seen", "")
    if classes:
        landmarks = [lm for lm in landmarks if lm["class_name"] in classes]
    return landmarks


def attach_ontology_knowledge(landmarks: list[dict], knowledge_base: OntologyKnowledgeBase) -> None:
    knowledge_by_class = {
        class_name: knowledge_base.resolve(class_name)
        for class_name in {landmark["class_name"] for landmark in landmarks}
    }
    for landmark in landmarks:
        landmark["ontology"] = knowledge_by_class[landmark["class_name"]]


def load_observations(conn: sqlite3.Connection, classes: set[str] | None) -> list[dict]:
    rows = conn.execute(
        "SELECT class_name, world_X, world_Y, world_Z FROM observations WHERE world_X IS NOT NULL"
    ).fetchall()
    obs = [{"class_name": r[0], "X": r[1], "Y": r[2], "Z": r[3]} for r in rows]
    if classes:
        obs = [o for o in obs if o["class_name"] in classes]
    return obs


def build_palette(class_names: list[str]) -> dict[str, tuple]:
    colormap = colormaps["tab20"].resampled(max(len(class_names), 1))
    return {name: colormap(i) for i, name in enumerate(class_names)}


def render(landmarks: list[dict], observations: list[dict], title: str, save_path: Path | None) -> None:
    class_names = sorted({lm["class_name"] for lm in landmarks})
    palette = build_palette(class_names)

    fig = plt.figure(figsize=(16, 8))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax2d = fig.add_subplot(1, 2, 2)

    for obs in observations:
        color = palette.get(obs["class_name"], (0.6, 0.6, 0.6, 1.0))
        ax3d.scatter(obs["X"], obs["Y"], obs["Z"], color=color, s=4, alpha=0.12, linewidths=0)
        ax2d.scatter(obs["X"], obs["Y"], color=color, s=4, alpha=0.12, linewidths=0)

    for lm in landmarks:
        color = palette[lm["class_name"]]
        size = 40 + 8 * min(lm["hit_count"], 40)
        label = f"{lm['class_name']} {lm['instance_id']}"

        ax3d.scatter(lm["X"], lm["Y"], lm["Z"], color=color, s=size, edgecolors="black", linewidths=0.6, depthshade=False)
        ax3d.plot([lm["X"], lm["X"]], [lm["Y"], lm["Y"]], [0, lm["Z"]], color=color, alpha=0.4, linewidth=1)
        ax3d.text(lm["X"], lm["Y"], lm["Z"] + 0.12, label, fontsize=7)

        ax2d.scatter(lm["X"], lm["Y"], color=color, s=size, edgecolors="black", linewidths=0.6)
        ax2d.annotate(label, (lm["X"], lm["Y"]), textcoords="offset points", xytext=(6, 4), fontsize=7)

    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z / height (m)")
    ax3d.set_title("Semantic map (3D)")

    ax2d.set_xlabel("X (m)")
    ax2d.set_ylabel("Y (m)")
    ax2d.set_title("Top-down floor plan")
    ax2d.set_aspect("equal", adjustable="datalim")
    ax2d.grid(True, alpha=0.3)

    if landmarks:
        xs = [lm["X"] for lm in landmarks]
        ys = [lm["Y"] for lm in landmarks]
        span = max(max(xs) - min(xs), max(ys) - min(ys), 2.0) * 0.6
        cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
        ax3d.set_xlim(cx - span, cx + span)
        ax3d.set_ylim(cy - span, cy + span)
        ax3d.set_zlim(0, max(2.5, max(lm["Z"] for lm in landmarks) + 0.5))

    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=palette[name], label=name)
               for name in class_names]
    if handles:
        ax2d.legend(handles=handles, loc="best", fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.show()


def load_camera_poses(conn: sqlite3.Connection) -> list[tuple[float, float, float]]:
    try:
        rows = conn.execute("SELECT X, Y, Z FROM camera_poses ORDER BY frame_index").fetchall()
    except sqlite3.OperationalError:  # DB written before camera_poses existed
        return []
    return [(r[0], r[1], r[2]) for r in rows]


def render_rerun(landmarks: list[dict], observations: list[dict],
                 trajectory: list[tuple[float, float, float]], application_id: str) -> None:
    import numpy as np
    import rerun as rr
    from rerun_logger import class_color, log_landmark_entities

    rr.init(application_id, spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    log_landmark_entities(landmarks)

    if observations:
        pts = np.asarray([[o["X"], o["Y"], o["Z"]] for o in observations], dtype=np.float32)
        obs_colors = [class_color(o["class_name"]) for o in observations]
        rr.log("world/observations", rr.Points3D(pts, colors=obs_colors, radii=0.015), static=True)

    if len(trajectory) >= 2:
        rr.log(
            "world/trajectory",
            rr.LineStrips3D([np.asarray(trajectory, dtype=np.float32)], colors=[[255, 220, 0]], radii=0.02),
            static=True,
        )


def main() -> None:
    p = argparse.ArgumentParser(description="View a semantic map in 3D with labels.")
    p.add_argument("--db", default="", help="Path to semantic_map.db (default: newest run)")
    p.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    p.add_argument("--min-hits", type=int, default=1, help="Hide landmarks seen fewer than N times")
    p.add_argument("--classes", default="", help="Comma-separated class whitelist")
    p.add_argument("--observations", action="store_true", help="Also plot raw per-frame observations")
    viewer = p.add_mutually_exclusive_group()
    viewer.add_argument("--rerun", action="store_true", help="Open in the Rerun 3D viewer instead of matplotlib")
    viewer.add_argument("--html", action="store_true", help="Write an interactive Plotly ontology viewer")
    p.add_argument("--html-out", default="", help="Output path for --html (default: beside the database)")
    p.add_argument("--no-open", action="store_true", help="Do not open the generated HTML in a browser")
    p.add_argument("--ontology", default=str(DEFAULT_ONTOLOGY), help="RDF/OWL ontology used by HTML and Rerun")
    p.add_argument("--save", default="", help="Write the figure to this PNG path (matplotlib only)")
    args = p.parse_args()

    if args.html_out and not args.html:
        p.error("--html-out requires --html")
    if args.no_open and not args.html:
        p.error("--no-open requires --html")
    if args.save and (args.html or args.rerun):
        p.error("--save is only available with the default matplotlib viewer")

    db_path = Path(args.db).resolve() if args.db else find_latest_db(Path(args.out_root).resolve())
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    classes = {c.strip() for c in args.classes.split(",") if c.strip()} or None

    conn = sqlite3.connect(db_path)
    try:
        landmarks = load_landmarks(conn, args.min_hits, classes)
        observations = load_observations(conn, classes) if args.observations else []
        trajectory = load_camera_poses(conn) if args.rerun or args.html else []
    finally:
        conn.close()

    if not landmarks:
        print(f"No landmarks in {db_path} matching the filters (min-hits={args.min_hits}).")
        return

    frames = {lm["world_frame"] for lm in landmarks}
    print(f"{db_path}\n{len(landmarks)} landmarks in frame(s): {', '.join(sorted(frames))}")
    for lm in landmarks:
        print(f"  {lm['class_name']:<20} {lm['instance_id']:<3} "
              f"({lm['X']:7.3f}, {lm['Y']:7.3f}, {lm['Z']:7.3f})  hits={lm['hit_count']}")

    if args.html or args.rerun:
        attach_ontology_knowledge(landmarks, OntologyKnowledgeBase(Path(args.ontology).resolve()))

    if args.html:
        from semantic_map_html import build_figure, write_html

        output_path = Path(args.html_out).resolve() if args.html_out else db_path.parent / "semantic_map_ontology.html"
        title = f"{db_path.parent.name} | ontology semantic map"
        written_path = write_html(build_figure(landmarks, trajectory, title), output_path, str(db_path))
        print(f"Saved: {written_path}")
        if not args.no_open:
            webbrowser.open(written_path.as_uri())
        return

    if args.rerun:
        render_rerun(landmarks, observations, trajectory, f"semantic_map/{db_path.parent.name}")
        return

    render(landmarks, observations, f"{db_path.parent.name}  |  frame: {', '.join(sorted(frames))}",
           Path(args.save).resolve() if args.save else None)


if __name__ == "__main__":
    main()
