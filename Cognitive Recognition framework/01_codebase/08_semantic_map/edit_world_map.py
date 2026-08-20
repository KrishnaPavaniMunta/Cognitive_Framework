"""Interactive curator for one persistent bag world map."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
DEFAULT_OUT_ROOT = PROJECT_ROOT / "04_outputs_runs_and_logs" / "outputs" / "semantic_maps"
OBJECT_DETECTION_DIR = PROJECT_ROOT / "01_codebase" / "07_object_detection"


def ensure_editor_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS class_remaps (
            source_class_name TEXT PRIMARY KEY,
            map_class_name    TEXT NOT NULL,
            updated_utc       TEXT NOT NULL
        )
        """
    )
    columns = {row[1] for row in conn.execute("PRAGMA table_info(observations)")}
    if "source_class_name" not in columns:
        conn.execute("ALTER TABLE observations ADD COLUMN source_class_name TEXT")
        conn.execute("UPDATE observations SET source_class_name=class_name WHERE source_class_name IS NULL")
    conn.commit()


def choose_map(out_root: Path) -> Path:
    maps = sorted(path for path in out_root.iterdir() if (path / "world_map.db").exists())
    if not maps:
        raise FileNotFoundError(f"No persistent world maps found under {out_root}")

    print("\nAvailable bag world maps:")
    for index, path in enumerate(maps, start=1):
        print(f"  {index}. {path.name}")

    while True:
        selected = input("Select a map number: ").strip()
        if selected.isdigit() and 1 <= int(selected) <= len(maps):
            return maps[int(selected) - 1]
        print("Enter one of the displayed numbers.")


def list_landmarks(conn: sqlite3.Connection) -> list[tuple]:
    return conn.execute(
        """
        SELECT landmark_id, class_name, instance_id, hit_count, mean_confidence, X, Y, Z
        FROM semantic_map
        ORDER BY class_name, instance_id
        """
    ).fetchall()


def print_landmarks(rows: list[tuple]) -> None:
    print("\nCurrent world-map objects:")
    if not rows:
        print("  (none)")
        return
    for _, class_name, instance_id, hits, confidence, x, y, z in rows:
        print(
            f"  {class_name} {instance_id:<3} conf={confidence:.2f}  "
            f"world=({x:.2f}, {y:.2f}, {z:.2f})"
        )


def detector_vocabulary(conn: sqlite3.Connection) -> set[str]:
    vocabulary = {row[0] for row in conn.execute("SELECT DISTINCT class_name FROM semantic_map")}
    try:
        vocabulary.update(row[0] for row in conn.execute("SELECT DISTINCT source_class_name FROM observations"))
    except sqlite3.OperationalError:
        vocabulary.update(row[0] for row in conn.execute("SELECT DISTINCT class_name FROM observations"))

    prompts_path = OBJECT_DETECTION_DIR / "dino_prompts.py"
    spec = importlib.util.spec_from_file_location("semantic_map_dino_prompts", prompts_path)
    if spec is not None and spec.loader is not None:
        prompts = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prompts)
        vocabulary.update(prompts.DINO_FALLBACK)
    return {label for label in vocabulary if label}


def renumber_class_instances(conn: sqlite3.Connection, class_name: str) -> None:
    rows = conn.execute(
        """
        SELECT landmark_id FROM semantic_map
        WHERE class_name=?
        ORDER BY first_seen_ns, landmark_id
        """,
        (class_name,),
    ).fetchall()
    for offset, (landmark_id,) in enumerate(rows, start=1):
        conn.execute(
            "UPDATE semantic_map SET instance_id=? WHERE landmark_id=?",
            (-offset, landmark_id),
        )
    for new_instance_id, (landmark_id,) in enumerate(rows, start=1):
        conn.execute(
            "UPDATE semantic_map SET instance_id=? WHERE landmark_id=?",
            (new_instance_id, landmark_id),
        )


def rebuild_rerun(map_dir: Path, conn: sqlite3.Connection) -> None:
    row = conn.execute(
        "SELECT bag_path FROM map_runs WHERE completed_utc IS NOT NULL ORDER BY completed_utc DESC LIMIT 1"
    ).fetchone()
    if row is None:
        raise RuntimeError("Cannot rebuild point cloud: this map has no completed source-bag run recorded.")

    builder = BASE_DIR / "build_semantic_map_from_bag.py"
    command = [
        sys.executable,
        str(builder),
        "--bag", str(row[0]),
        "--out-root", str(map_dir.parent),
        "--rerun-only",
        "--no-preview",
        "--no-video",
        "--rerun-cloud-stride", "6",
        "--rerun-cloud-every", "10",
    ]
    print("\nRebuilding the point cloud and updated landmark layer...")
    subprocess.run(command, check=True)


def delete_landmark(conn: sqlite3.Connection, class_name: str, instance_id: int) -> bool:
    row = conn.execute(
        "SELECT landmark_id FROM semantic_map WHERE class_name=? AND instance_id=?",
        (class_name, instance_id),
    ).fetchone()
    if row is None:
        print(f"No object named '{class_name} {instance_id}' exists in this map.")
        return False

    landmark_id = int(row[0])
    conn.execute("DELETE FROM observations WHERE landmark_id=?", (landmark_id,))
    conn.execute("DELETE FROM semantic_map WHERE landmark_id=?", (landmark_id,))
    renumber_class_instances(conn, class_name)
    conn.execute(
        """
        INSERT INTO manual_edits (edited_utc, action, class_name, instance_id, details)
        VALUES (?,?,?,?,?)
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            "delete_landmark",
            class_name,
            instance_id,
            json.dumps({"landmark_id": landmark_id}),
        ),
    )
    conn.commit()
    return True


def rename_class(conn: sqlite3.Connection, source_class: str, target_class: str) -> bool:
    if source_class == target_class:
        print("The source and target classes are the same.")
        return False
    if target_class not in detector_vocabulary(conn):
        print(f"'{target_class}' is not in this bag's YOLO/DINO vocabulary.")
        return False

    rows = conn.execute(
        "SELECT landmark_id FROM semantic_map WHERE class_name=? ORDER BY first_seen_ns, landmark_id",
        (source_class,),
    ).fetchall()
    if not rows:
        print(f"No mapped objects use class '{source_class}'.")
        return False

    # Negative temporary IDs avoid the unique (class_name, instance_id) index while moving classes.
    for offset, (landmark_id,) in enumerate(rows, start=1):
        conn.execute("UPDATE semantic_map SET instance_id=? WHERE landmark_id=?", (-offset, landmark_id))
    conn.execute("UPDATE semantic_map SET class_name=? WHERE class_name=?", (target_class, source_class))
    conn.execute("UPDATE observations SET class_name=? WHERE class_name=?", (target_class, source_class))
    conn.execute("UPDATE class_remaps SET map_class_name=? WHERE map_class_name=?", (target_class, source_class))
    conn.execute(
        "INSERT OR REPLACE INTO class_remaps VALUES (?,?,?)",
        (source_class, target_class, datetime.now(timezone.utc).isoformat()),
    )
    renumber_class_instances(conn, target_class)
    conn.execute(
        """
        INSERT INTO manual_edits (edited_utc, action, class_name, instance_id, details)
        VALUES (?,?,?,?,?)
        """,
        (datetime.now(timezone.utc).isoformat(), "rename_class", source_class, 0,
         json.dumps({"target_class": target_class, "landmark_count": len(rows)})),
    )
    conn.commit()
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactively delete false positives from one bag world map.")
    parser.add_argument("--map-dir", default="", help="Bag map folder containing world_map.db")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    args = parser.parse_args()

    map_dir = Path(args.map_dir).resolve() if args.map_dir else choose_map(Path(args.out_root).resolve())
    db_path = map_dir / "world_map.db"
    if not db_path.exists():
        raise FileNotFoundError(f"world_map.db not found in {map_dir}")

    conn = sqlite3.connect(db_path)
    try:
        ensure_editor_schema(conn)
        print(f"\nEditing: {map_dir.name}")
        while True:
            rows = list_landmarks(conn)
            print_landmarks(rows)
            entry = input("\nCommand: delete <class> <ID>, rename <old_class> <new_class>, or done: ").strip()
            if entry.lower() in {"done", "exit", "quit"}:
                break

            if entry.lower().startswith("rename "):
                parts = entry.split()
                if len(parts) != 3:
                    print("Example: rename hospital_stretcher utility_trolley")
                    continue
                _, source_class, target_class = parts
                confirm = input(f"Rename all '{source_class}' map objects to '{target_class}'? [y/N]: ").strip().lower()
                if confirm == "y" and rename_class(conn, source_class, target_class):
                    print(f"Renamed '{source_class}' to '{target_class}'.")
                continue

            if entry.lower().startswith("delete "):
                entry = entry[7:].strip()

            try:
                class_name, raw_id = entry.rsplit(maxsplit=1)
                instance_id = int(raw_id)
            except ValueError:
                print("Example: delete knife 1  or  rename hospital_stretcher utility_trolley")
                continue

            confirm = input(f"Delete '{class_name} {instance_id}'? [y/N]: ").strip().lower()
            if confirm != "y":
                print("Not deleted.")
                continue
            if delete_landmark(conn, class_name, instance_id):
                print(f"Deleted '{class_name} {instance_id}'.")

        rebuild_rerun(map_dir, conn)
        print(f"\nUpdated database: {db_path}")
        print(f"Updated Rerun map: {map_dir / 'world_map.rrd'}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()