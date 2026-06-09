from pathlib import Path
import sys
import time
import json
from datetime import datetime

import cv2
import torch
from ultralytics import YOLO

ws = Path(r"d:\Object Detection Model\yolo_tr\yolo_tr\Cognitive Recognition framework")
temporal_dir = ws / "01_codebase" / "04_rgbd_and_spatial_twin" / "hospital_detector_temporal"
sys.path.insert(0, str(temporal_dir))

import infer_hospitalguard_temporal as hgt

video_path = ws / "02_datasets" / "saxon" / "recordings" / "bags" / "recording_20260521_142201" / "recording.mp4"
v1_path = ws / "04_outputs_runs_and_logs" / "outputs" / "runs" / "hospital" / "phase2_neck_head" / "weights" / "best.pt"
v3_path = ws / "04_outputs_runs_and_logs" / "outputs" / "runs" / "hospital_v3" / "phase2_neck_head" / "weights" / "best.pt"

assert video_path.exists(), f"Missing video: {video_path}"
assert v1_path.exists(), f"Missing V1 weights: {v1_path}"
assert v3_path.exists(), f"Missing V3 weights: {v3_path}"

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not available in this environment. Cannot force GPU run.")

hgt.V1_PATH = v1_path
hgt.V3_PATH = v3_path
hgt.OUT_DIR = ws / "04_outputs_runs_and_logs" / "outputs" / "hospitalguard_output"
hgt.EXCEL_PATH = ws / "04_outputs_runs_and_logs" / "validation_results" / "hospitalguard_temporal_fullgpu_log.xlsx"
hgt.DINO_DEVICE = "cuda"
# Run DINO once every 15 frames.
hgt.DINO_VIDEO_INTERVAL_FRAMES = 15
hgt.DINO_VIDEO_INTERVAL_SEC = 1.0

hgt.OUT_DIR.mkdir(parents=True, exist_ok=True)
hgt.EXCEL_PATH.parent.mkdir(parents=True, exist_ok=True)

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_video = hgt.OUT_DIR / f"recording_20260521_142201_temporal_fullgpu_{ts}.mp4"
out_json = hgt.EXCEL_PATH.parent / f"temporal_fullgpu_perf_{ts}.json"

cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {video_path}")
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
cap.release()

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
print("Loading YOLO models on GPU...")
v1 = YOLO(str(v1_path))
v3 = YOLO(str(v3_path))
v1.to("cuda")
v3.to("cuda")

print("Running temporal inference (YOLO + DINO every 15 frames, GPU mode)...")
t0 = time.time()
all_confs = hgt.run_video(v1, v3, video_path, out_video)
elapsed = time.time() - t0

processed_fps = (frame_count / elapsed) if frame_count > 0 and elapsed > 0 else 0.0
summary = {
    "timestamp": ts,
    "video_path": str(video_path),
    "output_video": str(out_video),
    "frame_count": frame_count,
    "source_fps": round(source_fps, 3),
    "inference_seconds": round(elapsed, 3),
    "processed_fps": round(processed_fps, 3),
    "dino_device": hgt.DINO_DEVICE,
    "dino_interval_frames": hgt.DINO_VIDEO_INTERVAL_FRAMES,
    "dino_interval_sec": hgt.DINO_VIDEO_INTERVAL_SEC,
    "mode": "YOLO + DINO every 15 frames",
    "classes_seen": len(all_confs),
}

with open(out_json, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n=== FULL GPU RUN SUMMARY ===")
print(f"output_video={out_video}")
print(f"frames={frame_count} source_fps={source_fps:.2f}")
print(f"inference_seconds={elapsed:.2f}")
print(f"processed_fps={processed_fps:.2f}")
print(f"classes_seen={len(all_confs)}")
print(f"perf_json={out_json}")