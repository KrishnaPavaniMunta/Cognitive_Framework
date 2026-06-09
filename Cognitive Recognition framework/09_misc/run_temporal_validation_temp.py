from pathlib import Path
import sys
import time
import json
import csv
from datetime import datetime
import cv2

ws = Path(r"d:\Object Detection Model\yolo_tr\yolo_tr\Cognitive Recognition framework")
temporal_dir = ws / "01_codebase" / "04_rgbd_and_spatial_twin" / "hospital_detector_temporal"
sys.path.insert(0, str(temporal_dir))

import infer_hospitalguard_temporal as hgt
from ultralytics import YOLO

video_path = ws / "02_datasets" / "saxon" / "recordings" / "bags" / "recording_20260521_142201" / "recording.mp4"
v1_path = ws / "04_outputs_runs_and_logs" / "outputs" / "runs" / "hospital" / "phase2_neck_head" / "weights" / "best.pt"
v3_path = ws / "04_outputs_runs_and_logs" / "outputs" / "runs" / "hospital_v3" / "phase2_neck_head" / "weights" / "best.pt"

assert video_path.exists(), f"Missing video: {video_path}"
assert v1_path.exists(), f"Missing V1 weights: {v1_path}"
assert v3_path.exists(), f"Missing V3 weights: {v3_path}"

hgt.V1_PATH = v1_path
hgt.V3_PATH = v3_path
hgt.OUT_DIR = ws / "04_outputs_runs_and_logs" / "outputs" / "hospitalguard_output"
hgt.EXCEL_PATH = ws / "04_outputs_runs_and_logs" / "validation_results" / "hospitalguard_temporal_validation_log.xlsx"
hgt.DINO_FALLBACK = {}
hgt.DINO_SAHI = {}
hgt.OUT_DIR.mkdir(parents=True, exist_ok=True)
hgt.EXCEL_PATH.parent.mkdir(parents=True, exist_ok=True)

targets = ["hospital_bed", "monitor_hosp", "radiator", "fire_extinguisher", "exit_sign"]
sample_step = 60
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
sampled_video = hgt.OUT_DIR / f"recording_20260521_142201_sampled_x{sample_step}_{ts}.mp4"
out_video = hgt.OUT_DIR / f"recording_20260521_142201_temporal_validation_{ts}.mp4"
out_csv = hgt.EXCEL_PATH.parent / f"temporal_validation_metrics_{ts}.csv"
out_json = hgt.EXCEL_PATH.parent / f"temporal_validation_metrics_{ts}.json"
out_md = hgt.EXCEL_PATH.parent / f"temporal_validation_report_{ts}.md"

cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {video_path}")
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
cap.release()

print(f"Sampling video every {sample_step} frame(s)...")
cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    raise RuntimeError(f"Cannot reopen video: {video_path}")
sample_fps = fps / sample_step if fps else 0.0
sampled_writer = None
sampled_frames = 0
frame_idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if frame_idx % sample_step == 0:
        if sampled_writer is None:
            h, w = frame.shape[:2]
            sampled_writer = cv2.VideoWriter(str(sampled_video), cv2.VideoWriter_fourcc(*"mp4v"), max(sample_fps, 1.0), (w, h))
        sampled_writer.write(frame)
        sampled_frames += 1
    frame_idx += 1
cap.release()
if sampled_writer is not None:
    sampled_writer.release()
else:
    raise RuntimeError("No frames were sampled from the video")
print(f"Sampled video written: {sampled_video} ({sampled_frames} frames)")

print("Loading models...")
v1 = YOLO(str(v1_path))
v3 = YOLO(str(v3_path))
print("Running temporal inference...")
t0 = time.time()
all_confs = hgt.run_video(v1, v3, sampled_video, out_video)
elapsed = time.time() - t0

rows = []
detected_targets = 0
for cls in targets:
    confs = list(all_confs.get(cls, []))
    detections = len(confs)
    detected = detections > 0
    if detected:
        detected_targets += 1
    rows.append({
        "class": cls,
        "detected": detected,
        "detections": detections,
        "max_conf": round(max(confs), 4) if confs else 0.0,
        "mean_conf": round(sum(confs) / len(confs), 4) if confs else 0.0,
        "min_conf": round(min(confs), 4) if confs else 0.0,
    })

coverage = detected_targets / len(targets)
proc_fps = (frame_count / elapsed) if frame_count and elapsed else 0.0
summary = {
    "timestamp": ts,
    "video_path": str(video_path),
    "sampled_video_path": str(sampled_video),
    "sample_step": sample_step,
    "sampled_frames": sampled_frames,
    "output_video": str(out_video),
    "frame_count": frame_count,
    "video_fps": round(fps, 3),
    "sample_fps": round(sample_fps, 3),
    "inference_seconds": round(elapsed, 3),
    "inference_fps": round(proc_fps, 3),
    "validation_targets": targets,
    "targets_detected": detected_targets,
    "target_coverage": round(coverage, 4),
    "per_class": rows,
}

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["class", "detected", "detections", "max_conf", "mean_conf", "min_conf"])
    w.writeheader()
    w.writerows(rows)

with open(out_json, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

lines = []
lines.append("# HospitalGuard Temporal Validation Report")
lines.append("")
lines.append(f"- Timestamp: {ts}")
lines.append(f"- Video: {video_path}")
lines.append(f"- Sampled video: {sampled_video}")
lines.append(f"- Sample step: every {sample_step} frame(s)")
lines.append(f"- Sampled frames: {sampled_frames}")
lines.append(f"- Output video: {out_video}")
lines.append(f"- Frames: {frame_count}")
lines.append(f"- Source FPS: {fps:.2f}")
lines.append(f"- Sample FPS: {sample_fps:.2f}")
lines.append(f"- Inference time (s): {elapsed:.2f}")
lines.append(f"- Inference FPS: {proc_fps:.2f}")
lines.append(f"- Validation target coverage: {detected_targets}/{len(targets)} ({coverage*100:.1f}%)")
lines.append("")
lines.append("## Per-class")
lines.append("")
lines.append("| class | detected | detections | max_conf | mean_conf | min_conf |")
lines.append("|---|---:|---:|---:|---:|---:|")
for r in rows:
    lines.append(f"| {r['class']} | {str(r['detected'])} | {r['detections']} | {r['max_conf']:.4f} | {r['mean_conf']:.4f} | {r['min_conf']:.4f} |")
with open(out_md, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print("\n=== VALIDATION SUMMARY ===")
print(f"video={video_path}")
print(f"output_video={out_video}")
print(f"frames={frame_count} source_fps={fps:.2f}")
print(f"inference_seconds={elapsed:.2f} inference_fps={proc_fps:.2f}")
print(f"target_coverage={detected_targets}/{len(targets)} ({coverage*100:.1f}%)")
for r in rows:
    print(f"- {r['class']}: detected={r['detected']} detections={r['detections']} max={r['max_conf']:.4f} mean={r['mean_conf']:.4f}")
print(f"csv={out_csv}")
print(f"json={out_json}")
print(f"md={out_md}")