"""
phase2_retry.py
───────────────
Re-runs phase 2 (neck + head) from the phase 1 best checkpoint with fixes:
  - warmup_bias_lr=lr0  → prevents the bias-LR spike that disrupted the first attempt
  - patience=50         → avoids premature early stopping
"""

from pathlib import Path
from ultralytics import YOLO

BASE_DIR      = Path(__file__).parent.resolve()
PHASE1_BEST   = BASE_DIR / "training_output" / "phase1_head" / "weights" / "best.pt"
OUTPUT_YAML   = BASE_DIR / "finetune_data.yaml"

assert PHASE1_BEST.exists(), f"Phase 1 checkpoint not found: {PHASE1_BEST}"

print(f"Loading phase 1 checkpoint: {PHASE1_BEST}")

model = YOLO(str(PHASE1_BEST))
model.train(
    data=str(OUTPUT_YAML),
    imgsz=640,
    epochs=70,
    batch=16,
    optimizer="AdamW",
    weight_decay=0.0005,
    lr0=0.0002,
    lrf=0.01,
    warmup_epochs=3,
    warmup_bias_lr=0.0002,   # match lr0 — fixes the spike from the first attempt
    freeze=10,
    patience=50,
    workers=8,
    plots=True,
    project=str(BASE_DIR / "runs" / "finetune"),
    name="phase2_neck_head3",
    pretrained=True,
)

best = BASE_DIR / "runs" / "finetune" / "phase2_neck_head3" / "weights" / "best.pt"
print(f"\nDone. Best weights: {best}")
