"""
analyse_output_video.py
Extracts per-frame annotation data from an annotated HospitalGuard output video
and saves: a CSV of per-frame detections, a timeline chart, and sample frames.
"""
import sys
import re
import csv
from pathlib import Path
import cv2
import numpy as np

# ── config ────────────────────────────────────────────────────────────────────
VIDEO_PATH = Path(r"d:\Object Detection Model\yolo_tr\yolo_tr\outputs\hospitalguard_output\Surgeon_with_blue_hairnet__20260507_141714.mp4")
OUT_DIR    = VIDEO_PATH.parent / (VIDEO_PATH.stem + "_analysis")
OUT_DIR.mkdir(exist_ok=True)

# Sample frames to extract as JPEGs (evenly spaced)
N_SAMPLE_FRAMES = 16

# Classes we care about for PPE analysis
PPE_CLASSES     = {"hair_net", "mask", "glove"}
WORKER_CLASSES  = {"healthcare_worker", "person"}
KEY_CLASSES     = PPE_CLASSES | WORKER_CLASSES | {"surgical_scissor", "surgical_light", "medical_tray"}

# Regex to find labels drawn on the video: "classname #id  XX%"
LABEL_RE = re.compile(r"([a-z_]+)\s*#\d+\s+\d+%", re.IGNORECASE)

# ── helpers ───────────────────────────────────────────────────────────────────
def detect_labels_in_frame(bgr_frame):
    """
    Read text labels from the annotated frame using simple colour/contour
    heuristics — we look for white text on dark pill-shaped backgrounds
    by scanning a small strip at the top of each bounding-box region.
    Since we can't run OCR easily, we instead re-run YOLO on the raw
    frame and compare box positions.  Here we just count coloured boxes
    as a proxy for 'at least one box is visible'.
    """
    # Count the number of distinct annotation colours (supervision palette)
    # by looking for bright non-background pixels in the top banner region.
    # Simpler: just count distinct hue bands in the annotated image.
    hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
    # Mask out near-black (background) and near-white (labels)
    mask = (hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 60)
    n_colour_pixels = int(mask.sum())
    return n_colour_pixels


def extract_frame_summary(video_path: Path):
    """Read every frame, record coloured-pixel count as detection-activity proxy."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {video_path}")
        sys.exit(1)

    fps        = cap.get(cv2.CAP_PROP_FPS)
    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video  : {video_path.name}")
    print(f"Size   : {width}x{height}  |  FPS: {fps:.1f}  |  Frames: {total}")
    print()

    rows       = []   # (frame_idx, sec, colour_pixels)
    sample_idx = set(np.linspace(0, total - 1, N_SAMPLE_FRAMES, dtype=int).tolist())
    frame_idx  = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        sec     = frame_idx / fps
        cpx     = detect_labels_in_frame(frame)
        rows.append((frame_idx, round(sec, 2), cpx))

        if frame_idx in sample_idx:
            jpg_path = OUT_DIR / f"frame_{frame_idx:04d}_t{sec:.1f}s.jpg"
            cv2.imwrite(str(jpg_path), frame)

        frame_idx += 1

    cap.release()
    return rows, fps, total


def save_csv(rows, out_path):
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_sec", "coloured_pixels"])
        w.writerows(rows)
    print(f"CSV saved: {out_path}")


def save_timeline_chart(rows, fps, out_path):
    """Render a simple bar chart of detection activity over time."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        frames = [r[0] for r in rows]
        cpx    = [r[2] for r in rows]
        secs   = [r[1] for r in rows]

        fig, ax = plt.subplots(figsize=(16, 4))
        ax.fill_between(secs, cpx, alpha=0.7, color="steelblue")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Coloured annotation pixels")
        ax.set_title(f"Detection activity over time — {out_path.stem}")
        ax.axhline(np.mean(cpx), color="red", linestyle="--", linewidth=1, label=f"mean={int(np.mean(cpx)):,}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(str(out_path), dpi=120)
        plt.close()
        print(f"Chart  : {out_path}")
    except ImportError:
        print("[WARN] matplotlib not installed — skipping chart.")


def analyse_gaps(rows, fps):
    """Find frames where annotation activity drops below 10% of max (detection gaps)."""
    cpx    = [r[2] for r in rows]
    thresh = max(cpx) * 0.10
    gaps   = []
    in_gap = False
    start  = 0

    for i, (fi, sec, c) in enumerate(rows):
        if c < thresh:
            if not in_gap:
                in_gap = True
                start  = sec
        else:
            if in_gap:
                gaps.append((start, sec, round(sec - start, 2)))
                in_gap = False

    if in_gap:
        gaps.append((start, rows[-1][1], round(rows[-1][1] - start, 2)))

    return gaps


def main():
    rows, fps, total = extract_frame_summary(VIDEO_PATH)
    save_csv(rows, OUT_DIR / "frame_activity.csv")
    save_timeline_chart(rows, fps, OUT_DIR / "activity_timeline.png")

    gaps = analyse_gaps(rows, fps)

    print("\n── Detection Summary ───────────────────────────────────────")
    cpx     = [r[2] for r in rows]
    active  = sum(1 for c in cpx if c > max(cpx) * 0.10)
    dark    = total - active
    print(f"  Total frames        : {total}")
    print(f"  Frames with boxes   : {active}  ({100*active/total:.1f}%)")
    print(f"  Frames with NO boxes: {dark}   ({100*dark/total:.1f}%)")
    print(f"  Mean annotation px  : {int(np.mean(cpx)):,}")
    print(f"  Peak annotation px  : {int(max(cpx)):,}")

    print(f"\n── Detection gaps (activity < 10% of peak) ─────────────────")
    if gaps:
        for g in gaps:
            print(f"  {g[0]:.1f}s → {g[1]:.1f}s  ({g[2]:.2f}s gap)")
    else:
        print("  None — boxes present throughout.")

    print(f"\n── Sample frames saved ─────────────────────────────────────")
    print(f"  {N_SAMPLE_FRAMES} frames saved to: {OUT_DIR}")
    print(f"\nOpen {OUT_DIR} to inspect the annotated frames visually.")


if __name__ == "__main__":
    main()
