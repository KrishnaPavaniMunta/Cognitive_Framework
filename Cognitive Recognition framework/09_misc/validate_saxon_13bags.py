from __future__ import annotations

import csv
import importlib.util
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
INFER_PATH = ROOT / "01_codebase" / "02_inference" / "infer_hospitalguard_v2.py"
SAXON_DIR = ROOT / "02_datasets" / "saxon"
OUT_DIR = ROOT / "04_outputs_runs_and_logs" / "outputs" / "saxon_OD_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
V1_WEIGHTS = ROOT / "04_outputs_runs_and_logs" / "outputs" / "runs" / "hospital" / "phase2_neck_head" / "weights" / "best.pt"
V3_WEIGHTS = ROOT / "04_outputs_runs_and_logs" / "outputs" / "runs" / "hospital_v3" / "phase2_neck_head" / "weights" / "best.pt"

# Validation settings (sampling-based to keep runtime practical)
SAMPLE_EVERY_SEC = 10.0
BASELINE_DINO_INTERVAL_SEC = 60.0
LOW_CONF_FP_THRESHOLD = 0.40
ENABLE_DINO_IN_VALIDATION = False


@dataclass
class FrameEval:
    frame_idx: int
    baseline_classes: set[str]
    baseline_max_conf: Dict[str, float]


def load_module(module_path: Path):
    spec = importlib.util.spec_from_file_location("infer_hg_v2", str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def max_conf_map(dets: Dict[str, List[Tuple[float, float, float, float, float]]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for cls, boxes in dets.items():
        if boxes:
            out[cls] = max(float(b[4]) for b in boxes)
    return out


def run_one_video(hg, v1: YOLO, v3: YOLO, video_path: Path) -> tuple[list[FrameEval], dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    sample_stride = max(1, round(fps * SAMPLE_EVERY_SEC))
    baseline_dino_stride = max(1, round(fps * BASELINE_DINO_INTERVAL_SEC))

    frame_idx = -1
    sampled: list[FrameEval] = []

    all_dino_targets = set(hg.DINO_FALLBACK) | set(hg.DINO_SAHI)

    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % sample_stride != 0:
            continue

        yolo = hg._yolo_on_frame(v1, v3, bgr)
        pil_img = None

        # Baseline pass (matches throughput-oriented production behavior)
        baseline_dino: dict = {}
        if ENABLE_DINO_IN_VALIDATION and frame_idx % baseline_dino_stride == 0:
            missing = [c for c in all_dino_targets if c not in yolo]
            missing = hg._context_gate(missing, yolo)
            if missing:
                if pil_img is None:
                    pil_img = hg.Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                baseline_dino = hg.dino_infer(pil_img, missing)

        baseline_all = {**yolo, **baseline_dino}

        sampled.append(
            FrameEval(
                frame_idx=frame_idx,
                baseline_classes=set(baseline_all.keys()),
                baseline_max_conf=max_conf_map(baseline_all),
            )
        )

    cap.release()

    info = {
        "fps": fps,
        "total_frames": total,
        "sample_stride": sample_stride,
        "sampled_frames": len(sampled),
        "baseline_dino_stride": baseline_dino_stride,
    }
    return sampled, info


def temporal_metrics(sampled: list[FrameEval], classes: list[str]) -> tuple[dict, dict, dict]:
    """
    Compute temporal-consistency based estimates.
    - likely_missed: class absent at t but present at t-1 and t+1 (dropout)
    - likely_false: class appears only at t (isolated spike) with low confidence
    """
    presence = {c: [0] * len(sampled) for c in classes}
    confs = {c: [0.0] * len(sampled) for c in classes}

    for i, fr in enumerate(sampled):
        for c in fr.baseline_classes:
            presence[c][i] = 1
            confs[c][i] = fr.baseline_max_conf.get(c, 0.0)

    hits = {c: sum(presence[c]) for c in classes}
    likely_missed = {c: 0 for c in classes}
    likely_false = {c: 0 for c in classes}

    for c in classes:
        p = presence[c]
        cf = confs[c]
        for i in range(1, len(p) - 1):
            if p[i - 1] == 1 and p[i] == 0 and p[i + 1] == 1:
                likely_missed[c] += 1
            if p[i - 1] == 0 and p[i] == 1 and p[i + 1] == 0 and cf[i] < LOW_CONF_FP_THRESHOLD:
                likely_false[c] += 1

    return hits, likely_missed, likely_false


def main() -> None:
    hg = load_module(INFER_PATH)
    hg.V1_PATH = V1_WEIGHTS
    hg.V3_PATH = V3_WEIGHTS

    print("Loading YOLO weights...")
    v1 = YOLO(str(hg.V1_PATH))
    v3 = YOLO(str(hg.V3_PATH))

    classes = [v3.names[i] for i in sorted(v3.names.keys())]

    videos = sorted(SAXON_DIR.glob("rgbd_clean_*.mp4"))
    if len(videos) != 13:
        print(f"WARNING: expected 13 videos, found {len(videos)}")

    global_baseline_hits = defaultdict(int)
    global_likely_missed = defaultdict(int)
    global_likely_false = defaultdict(int)

    per_bag_rows = []

    for idx, video in enumerate(videos, start=1):
        print(f"[{idx}/{len(videos)}] Validating {video.name}")
        sampled, info = run_one_video(hg, v1, v3, video)
        print(f"    sampled_frames={info['sampled_frames']} (every {SAMPLE_EVERY_SEC}s)")

        bag_baseline_hits, bag_likely_missed, bag_likely_false = temporal_metrics(sampled, classes)

        for c, n in bag_baseline_hits.items():
            global_baseline_hits[c] += n
        for c, n in bag_likely_missed.items():
            global_likely_missed[c] += n
        for c, n in bag_likely_false.items():
            global_likely_false[c] += n

        top_missed = sorted(bag_likely_missed.items(), key=lambda x: x[1], reverse=True)[:10]
        top_false = sorted(bag_likely_false.items(), key=lambda x: x[1], reverse=True)[:10]

        per_bag_rows.append(
            {
                "bag": video.name,
                "fps": f"{info['fps']:.2f}",
                "total_frames": info["total_frames"],
                "sampled_frames": info["sampled_frames"],
                "classes_detected": sum(1 for c in classes if bag_baseline_hits[c] > 0),
                "top_likely_missed": "; ".join(f"{k}:{v}" for k, v in top_missed),
                "top_likely_false": "; ".join(f"{k}:{v}" for k, v in top_false),
            }
        )

    class_report_path = OUT_DIR / "validation_13bags_by_class.csv"
    with class_report_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "class_name",
                "baseline_hits",
                "likely_missed_frames",
                "likely_false_frames",
                "dropout_rate_vs_hits",
                "spike_rate_vs_hits",
            ]
        )
        for c in classes:
            b = global_baseline_hits[c]
            m = global_likely_missed[c]
            fp = global_likely_false[c]
            dropout_rate = (m / b) if b else 0.0
            spike_rate = (fp / b) if b else 0.0
            w.writerow([c, b, m, fp, f"{dropout_rate:.4f}", f"{spike_rate:.4f}"])

    bag_report_path = OUT_DIR / "validation_13bags_by_bag.csv"
    with bag_report_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "bag",
                "fps",
                "total_frames",
                "sampled_frames",
                "classes_detected",
                "top_likely_missed",
                "top_likely_false",
            ],
        )
        w.writeheader()
        w.writerows(per_bag_rows)

    summary_path = OUT_DIR / "validation_13bags_summary.txt"
    top_missed_global = sorted(global_likely_missed.items(), key=lambda x: x[1], reverse=True)[:20]
    top_false_global = sorted(global_likely_false.items(), key=lambda x: x[1], reverse=True)[:20]

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Validation summary (sampling-based, no ground-truth labels)\n")
        f.write(f"SAMPLE_EVERY_SEC={SAMPLE_EVERY_SEC}\n")
        f.write(f"BASELINE_DINO_INTERVAL_SEC={BASELINE_DINO_INTERVAL_SEC}\n")
        f.write(f"ENABLE_DINO_IN_VALIDATION={ENABLE_DINO_IN_VALIDATION}\n")
        seen = sum(1 for c in classes if global_baseline_hits[c] > 0)
        unseen = [c for c in classes if global_baseline_hits[c] == 0]
        f.write(f"Detected classes (of 109): {seen}\n")
        f.write(f"Not detected classes (of 109): {len(unseen)}\n")
        if unseen:
            f.write("Not detected class list:\n")
            for c in unseen:
                f.write(f"- {c}\n")
        f.write("\nTop likely missed classes (global):\n")
        for cls, n in top_missed_global:
            if n > 0:
                f.write(f"- {cls}: {n}\n")
        f.write("\nTop likely false classes (global):\n")
        for cls, n in top_false_global:
            if n > 0:
                f.write(f"- {cls}: {n}\n")

    print("Done.")
    print(f"Class report: {class_report_path}")
    print(f"Bag report  : {bag_report_path}")
    print(f"Summary     : {summary_path}")


if __name__ == "__main__":
    main()
