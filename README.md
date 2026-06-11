# HospitalGuard YOLO + RGB-D Workspace

This repository contains training, inference, evaluation, and RGB-D temporal development scripts for the HospitalGuard object detection pipeline.

---

## Model Weights

The inference pipeline uses an **ensemble of two custom-trained YOLO models** (V1 and V3) plus a zero-shot fallback from Grounding DINO.

### YOLO V1 — Hospital Safety Model (Phase 1)

Trained on core hospital safety objects including PPE, clinical equipment, furniture, and access-control fixtures (80 COCO classes + 26 hospital-specific classes = **106 classes total**).

Key classes: `glove`, `healthcare_worker`, `hospital_bed`, `infusion_pump`, `iv_stand`, `wheelchair`, `door`, `fire_extinguisher`, `mask`, `hair_net`, `surgical_scissor`, `patient`, `security_camera`, and more.

**Weight file:**
```
03_models_and_weights/models/yolo_trained_v1.pt
```

### YOLO V3 — Extended Hospital Model (Phase 2)

Builds on V1 with additional hazard and navigation classes introduced in the v3 dataset. Fully backward-compatible with V1 classes.

Additional classes over V1: `bag`, `exit_sign`, `spillage` (classes 106–108).

**Weight file:**
```
03_models_and_weights/models/yolo_trained_v3.pt
```

### Ensemble Routing Logic

| Classes | Source |
|---|---|
| `wheelchair`, `door`, `fire_extinguisher` | V1 + V3 detections pooled → NMS |
| `bag`, `exit_sign`, `spillage` | V3 only (not in V1 vocabulary) |
| All other classes | V1 primary, V3 as supplement |

### Grounding DINO — Zero-Shot Fallback

Used as a fallback layer for weak or missed detections on difficult classes (e.g. `surgical_scissor`, `hair_net`, `glove`).

**No local weight file.** The model is downloaded automatically from Hugging Face Hub at runtime:
```
IDEA-Research/grounding-dino-base
```
Weights are cached in your local HuggingFace cache (`~/.cache/huggingface/`). No `.pt` file is committed to this repository.

---

## Project Highlights

- YOLO-based training/inference for hospital safety classes
- Ensemble and comparison utilities for model validation
- Temporal tracking pipeline in `hospital_detector_temporal/`
- RGB-D spatial memory + temporal fusion pipeline in `hospital_detector_longterm/rgbd_development/`

## Repository Layout

- `train_hospital.py`, `train_hospital_v3.py`, `train_merged.py`: Training entrypoints
- `infer_hospitalguard.py`, `infer_hospitalguard_v2.py`, `infer_v3.py`, `infer_video.py`: Inference scripts
- `batch_test.py`, `batch_test_ensemble.py`, `compare_v1_v3.py`: Validation and comparison tools
- `hospital_detector_temporal/`: Temporal (ByteTrack-based) detection pipeline
- `hospital_detector_longterm/rgbd_development/`: RGB-D replay, spatial memory, and long-term experiments

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Quick Start

### 1) Train a model

```powershell
python train_hospital_v3.py
```

### 2) Run image/video inference

```powershell
python infer_hospitalguard.py
python infer_video.py
```

### 3) Run temporal detection

```powershell
python hospital_detector_temporal/infer_hospitalguard_temporal.py
```

### 4) Run RGB-D temporal pipeline

```powershell
cd hospital_detector_longterm/rgbd_development/scripts
python rgbd_hospitalguard_temporal.py --sequence-root "../data/rgbd_dataset_freiburg1_xyz"
```

## Notes

- Large datasets, model weights, and generated outputs are intentionally excluded via `.gitignore`.
- RGB-D detailed usage is documented in `hospital_detector_longterm/rgbd_development/README.md`.

## GitHub Actions CI

This repository includes a lightweight CI workflow that validates Python syntax for tracked `.py` files on every push and pull request.