"""
query_digital_twin.py

Digital Twin Permanence Query Tool
------------------------------------
Connects to the spatial_memory SQLite database and reports:

  1. All objects CURRENTLY visible in the live detection session (last 5 sec).
  2. All objects that are NO LONGER visible but have a known last position.

This demonstrates "digital twin permanence":
  - Turn the camera to a blank wall → live view shows nothing.
  - Query this tool → it still knows exactly where every object was last seen.

Usage:
    python query_digital_twin.py                          # live summary
    python query_digital_twin.py --session <uuid>         # specific session
    python query_digital_twin.py --class-name chair       # filter by class
    python query_digital_twin.py --watch                  # refresh every 2s
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RGBD_DEV_DIR = SCRIPT_DIR.parent
DEFAULT_DB = RGBD_DEV_DIR / "output" / "hospital_twin.db"
TABLE = "spatial_memory"

LIVE_WINDOW_SEC = 5       # seconds: detections newer than this = "currently visible"
RECENT_WINDOW_SEC = 300   # seconds: show objects seen in last 5 minutes


def _utc_now_naive() -> datetime:
    """Return current UTC time without tzinfo for consistent DB string comparisons."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _connect(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found: {db_path}\n"
            "Run rgbd_hospitalguard_detect.py first to build the twin."
        )
    return sqlite3.connect(db_path)


def query_last_known(
    db_path: Path,
    session_id: str | None = None,
    class_filter: str | None = None,
    recent_only: bool = True,
) -> list[dict]:
    """Return one row per (session, tracker_id, class_name) with its latest position."""
    conn = _connect(db_path)
    try:
        where_clauses = []
        params: list = []

        if session_id:
            where_clauses.append("session_id = ?")
            params.append(session_id)

        if class_filter:
            where_clauses.append("class_name LIKE ?")
            params.append(f"%{class_filter}%")

        if recent_only:
            cutoff = (_utc_now_naive() - timedelta(seconds=RECENT_WINDOW_SEC)).isoformat(timespec="seconds")
            where_clauses.append("last_seen >= ?")
            params.append(cutoff)

        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        rows = conn.execute(
            f"""
            SELECT session_id, class_name, tracker_id, X, Y, Z, last_seen, timestamp
            FROM {TABLE}
            {where_sql}
            ORDER BY last_seen DESC
            """,
            params,
        ).fetchall()

        results = []
        for r in rows:
            results.append({
                "session_id": r[0],
                "class_name": r[1],
                "tracker_id": r[2],
                "X": r[3],
                "Y": r[4],
                "Z": r[5],
                "last_seen": r[6],
                "timestamp": r[7],
            })
        return results
    finally:
        conn.close()


def _age_label(last_seen_str: str) -> str:
    """Return human-readable age like '3.2s ago' or '2m ago'."""
    try:
        ls = datetime.fromisoformat(last_seen_str)
        age = (_utc_now_naive() - ls).total_seconds()
        if age < 60:
            return f"{age:.1f}s ago"
        if age < 3600:
            return f"{age/60:.1f}m ago"
        return f"{age/3600:.1f}h ago"
    except Exception:
        return last_seen_str


def _is_live(last_seen_str: str) -> bool:
    try:
        ls = datetime.fromisoformat(last_seen_str)
        return (_utc_now_naive() - ls).total_seconds() <= LIVE_WINDOW_SEC
    except Exception:
        return False


def print_twin_report(
    db_path: Path,
    session_id: str | None = None,
    class_filter: str | None = None,
) -> None:
    rows = query_last_known(db_path, session_id=session_id, class_filter=class_filter, recent_only=False)

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"\n{'='*70}")
    print(f"  HOSPITAL DIGITAL TWIN  —  {now_str}")
    print(f"  Database: {db_path}")
    print(f"{'='*70}")

    if not rows:
        print("  No objects recorded in the spatial memory database yet.")
        print("  Run rgbd_hospitalguard_detect.py to populate it.")
        print(f"{'='*70}\n")
        return

    live_rows = [r for r in rows if _is_live(r["last_seen"])]
    memory_rows = [r for r in rows if not _is_live(r["last_seen"])]

    # --- LIVE OBJECTS ---
    print(f"\n  CURRENTLY VISIBLE  ({len(live_rows)} object{'s' if len(live_rows) != 1 else ''})")
    print(f"  {'Class':<22} {'#ID':<6} {'X (m)':>8} {'Y (m)':>8} {'Z (m)':>8}   Last Seen")
    print(f"  {'-'*68}")
    if live_rows:
        for r in live_rows:
            print(f"  {r['class_name']:<22} #{r['tracker_id']:<5} "
                  f"{r['X']:>8.3f} {r['Y']:>8.3f} {r['Z']:>8.3f}   {_age_label(r['last_seen'])}")
    else:
        print("  (nothing visible right now — camera may be looking away)")

    # --- REMEMBERED OBJECTS ---
    print(f"\n  SPATIAL MEMORY  —  NOT currently visible, but I know where they are")
    print(f"  ({len(memory_rows)} remembered object{'s' if len(memory_rows) != 1 else ''})")
    print(f"  {'Class':<22} {'#ID':<6} {'X (m)':>8} {'Y (m)':>8} {'Z (m)':>8}   Last Seen")
    print(f"  {'-'*68}")
    if memory_rows:
        for r in memory_rows:
            print(f"  {r['class_name']:<22} #{r['tracker_id']:<5} "
                  f"{r['X']:>8.3f} {r['Y']:>8.3f} {r['Z']:>8.3f}   {_age_label(r['last_seen'])}")
    else:
        print("  (no objects in memory older than live window)")

    # --- SUMMARY ---
    all_classes = sorted({r["class_name"] for r in rows})
    print(f"\n  TWIN SUMMARY  —  {len(rows)} total tracked object instances")
    print(f"  Classes in memory: {', '.join(all_classes)}")

    if memory_rows:
        print(f"\n  DIGITAL TWIN PERMANENCE PROOF:")
        for r in memory_rows[:5]:  # Show top 5
            print(f"    \"{r['class_name']}\" (ID #{r['tracker_id']}) was last seen "
                  f"{_age_label(r['last_seen'])} at "
                  f"X={r['X']:.3f}m, Y={r['Y']:.3f}m, Z={r['Z']:.3f}m — still in memory.")

    print(f"{'='*70}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the Hospital Digital Twin spatial memory database")
    parser.add_argument("--db-path", default=str(DEFAULT_DB), help="Path to hospital_twin.db")
    parser.add_argument("--session", default=None, help="Filter to a specific session UUID")
    parser.add_argument("--class-name", default=None, help="Filter by class name (partial match)")
    parser.add_argument("--watch", action="store_true", help="Refresh every 2 seconds (Ctrl+C to stop)")
    parser.add_argument("--list-sessions", action="store_true", help="List all recorded session IDs")
    args = parser.parse_args()

    db_path = Path(args.db_path).resolve()

    if args.list_sessions:
        conn = _connect(db_path)
        sessions = conn.execute(
            f"SELECT session_id, COUNT(*) as n, MIN(timestamp), MAX(last_seen) "
            f"FROM {TABLE} GROUP BY session_id ORDER BY MAX(last_seen) DESC"
        ).fetchall()
        conn.close()
        print(f"\n{'='*70}")
        print(f"  Sessions in {db_path.name}  ({len(sessions)} total)")
        print(f"  {'Session ID':<38} {'Objects':>7}  {'First Frame':<22}  Last Seen")
        print(f"  {'-'*68}")
        for s in sessions:
            print(f"  {s[0]:<38} {s[1]:>7}  {s[2]:<22}  {s[3]}")
        print(f"{'='*70}\n")
        return

    if args.watch:
        try:
            while True:
                import os
                os.system("cls" if os.name == "nt" else "clear")
                print_twin_report(db_path, session_id=args.session, class_filter=args.class_name)
                print("  [Press Ctrl+C to stop watching]")
                time.sleep(2)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        print_twin_report(db_path, session_id=args.session, class_filter=args.class_name)


if __name__ == "__main__":
    main()
