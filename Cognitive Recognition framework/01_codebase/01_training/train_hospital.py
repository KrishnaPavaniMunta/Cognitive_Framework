"""
train_hospital.py
─────────────────
Fine-tunes yolo26m.pt to detect 80 COCO + 26 hospital classes (106 total).
Uses already-downloaded COCO 2017 val + filtered hospital dataset.

Two-phase freeze strategy:
  Phase 1 (30 epochs, freeze=22): head-only — learns all 106 class outputs
  Phase 2 (70 epochs, freeze=10): neck+head at low LR — gentle adaptation

Hospital class mapping (filtered IDs 0-25 → merged space 80-105):
  0 cabinet            → 80    13 vending_machines   → 93
  1 glove              → 81    14 wheelchair         → 94
  2 healthcare_worker  → 82    15 bench_hosp         → 95
  3 hospital_bed       → 83    16 door               → 96
  4 infusion_pump      → 84    17 reception_counter  → 97
  5 iv_bag             → 85    18 radiator           → 98
  6 iv_stand           → 86    19 bathroom_labels    → 99
  7 monitor_hosp       → 87    20 fire_extinguisher  → 100
  8 nasal_cannula      → 88    21 hospital_stretcher → 101
  9 patient            → 89    22 security_camera    → 102
  10 patient_monitor   → 90    23 hair_net           → 103
  11 surgical_light    → 91    24 mask               → 104
  12 test_tube         → 92    25 surgical_scissor   → 105

Run:
    cd /home/kelvin/yolo_tr
    python train_hospital.py
"""

import random
from pathlib import Path

import yaml

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent.resolve()

HOSP_TRAIN_IMG = BASE_DIR / "Hospital_Dataset_filtered" / "images" / "train"
HOSP_TRAIN_LBL = BASE_DIR / "Hospital_Dataset_filtered" / "labels" / "train"
HOSP_VAL_IMG   = BASE_DIR / "Hospital_Dataset_filtered" / "images" / "val"
HOSP_VAL_LBL   = BASE_DIR / "Hospital_Dataset_filtered" / "labels" / "val"

COCO_IMG = BASE_DIR / "coco2017val" / "val2017"
COCO_LBL = BASE_DIR / "coco2017val" / "coco" / "labels" / "val2017"

# Ultralytics resolves labels by replacing 'images' → 'labels' in the path.
MERGED_TR_IMG = BASE_DIR / "hospital_merged" / "images" / "train"
MERGED_TR_LBL = BASE_DIR / "hospital_merged" / "labels" / "train"
MERGED_VA_IMG = BASE_DIR / "hospital_merged" / "images" / "val"
MERGED_VA_LBL = BASE_DIR / "hospital_merged" / "labels" / "val"

MODEL_WEIGHTS = BASE_DIR / "yolo26m.pt"
OUTPUT_YAML   = BASE_DIR / "hospital_data.yaml"

VALID_SPLIT = 0.20
RANDOM_SEED = 42

# ── Class mapping ──────────────────────────────────────────────────────────────
HOSPITAL_REMAP = {i: 80 + i for i in range(26)}

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]
assert len(COCO_NAMES) == 80

# Note: monitor_hosp / bench_hosp are named differently to avoid confusion
# with COCO's tv (62) / bench (13) during training.
HOSPITAL_NAMES = [
    "cabinet", "glove", "healthcare_worker", "hospital_bed", "infusion_pump",
    "iv_bag", "iv_stand", "monitor_hosp", "nasal_cannula", "patient",
    "patient_monitor", "surgical_light", "test_tube", "vending_machines",
    "wheelchair", "bench_hosp", "door", "reception_counter", "radiator",
    "bathroom_labels", "fire_extinguisher", "hospital_stretcher",
    "security_camera", "hair_net", "mask", "surgical_scissor",
]
assert len(HOSPITAL_NAMES) == 26

MERGED_NAMES = COCO_NAMES + HOSPITAL_NAMES
assert len(MERGED_NAMES) == 106

IMG_EXTS = [".jpg", ".jpeg", ".png", ".JPG", ".PNG", ".JPEG"]


def find_img(img_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = img_dir / (stem + ext)
        if p.exists():
            return p
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Validate prerequisites
# ═══════════════════════════════════════════════════════════════════════════════
def validate() -> None:
    print("\n" + "=" * 60)
    print("STEP 1: Validating prerequisites")
    print("=" * 60)

    missing = []
    if not HOSP_TRAIN_IMG.exists() or not any(HOSP_TRAIN_IMG.iterdir()):
        missing.append("Hospital_Dataset_filtered/ — run prepare_hospital_dataset.py first")
    if not COCO_IMG.exists() or not any(COCO_IMG.iterdir()):
        missing.append("coco2017val/val2017 — COCO images not found")
    if not COCO_LBL.exists() or not any(COCO_LBL.iterdir()):
        missing.append("coco2017val/coco/labels/val2017 — COCO labels not found")
    if not MODEL_WEIGHTS.exists():
        missing.append(f"{MODEL_WEIGHTS.name} — model weights not found")

    if missing:
        print("\nERROR — Missing prerequisites:")
        for m in missing:
            print(f"  ✗ {m}")
        raise SystemExit(1)

    print(f"  ✓ Hospital train : {sum(1 for _ in HOSP_TRAIN_IMG.iterdir())} images, "
          f"{sum(1 for _ in HOSP_TRAIN_LBL.glob('*.txt'))} labels")
    print(f"  ✓ Hospital val   : {sum(1 for _ in HOSP_VAL_IMG.iterdir())} images")
    print(f"  ✓ COCO val       : {sum(1 for _ in COCO_IMG.glob('*.jpg'))} images")
    print(f"  ✓ Model          : {MODEL_WEIGHTS.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Build merged train/val directories
# ═══════════════════════════════════════════════════════════════════════════════
def build_merged_dirs() -> None:
    print("\n" + "=" * 60)
    print("STEP 2: Building merged train/val directories")
    print("=" * 60)

    if MERGED_TR_IMG.exists() and any(MERGED_TR_IMG.iterdir()):
        print("  [skip] hospital_merged/ already populated.")
        return

    for d in (MERGED_TR_IMG, MERGED_TR_LBL, MERGED_VA_IMG, MERGED_VA_LBL):
        d.mkdir(parents=True, exist_ok=True)

    def _remap_and_write(src_lf: Path, dst_lf: Path) -> None:
        if dst_lf.exists():
            return
        lines = []
        for line in src_lf.read_text().splitlines():
            parts = line.strip().split()
            if parts:
                lines.append(f"{HOSPITAL_REMAP[int(parts[0])]} {' '.join(parts[1:])}")
        dst_lf.write_text("\n".join(lines) + "\n")

    def _link(src: Path, dst: Path) -> None:
        if not dst.exists():
            dst.symlink_to(src.resolve())

    # ── Hospital train ──────────────────────────────────────────────────
    n_tr = 0
    for lf in sorted(HOSP_TRAIN_LBL.glob("*.txt")):
        img = find_img(HOSP_TRAIN_IMG, lf.stem)
        if img is None:
            continue
        _remap_and_write(lf, MERGED_TR_LBL / lf.name)
        _link(img, MERGED_TR_IMG / img.name)
        n_tr += 1

    # ── Hospital val ────────────────────────────────────────────────────
    n_va = 0
    for lf in sorted(HOSP_VAL_LBL.glob("*.txt")):
        img = find_img(HOSP_VAL_IMG, lf.stem)
        if img is None:
            continue
        _remap_and_write(lf, MERGED_VA_LBL / lf.name)
        _link(img, MERGED_VA_IMG / img.name)
        n_va += 1

    # ── COCO 2017 val — 80% train / 20% val (deterministic) ────────────
    rng = random.Random(RANDOM_SEED)
    coco_imgs = sorted(COCO_IMG.glob("*.jpg"))
    rng.shuffle(coco_imgs)
    n_cv         = int(len(coco_imgs) * VALID_SPLIT)
    coco_val_set = coco_imgs[:n_cv]
    coco_tr_set  = coco_imgs[n_cv:]

    def _add_coco(img: Path, img_dir: Path, lbl_dir: Path) -> None:
        _link(img, img_dir / ("coco_" + img.name))
        lbl = COCO_LBL / (img.stem + ".txt")
        if lbl.exists():
            dst_lbl = lbl_dir / ("coco_" + lbl.name)
            if not dst_lbl.exists():
                dst_lbl.write_text(lbl.read_text())

    for img in coco_tr_set:
        _add_coco(img, MERGED_TR_IMG, MERGED_TR_LBL)
    for img in coco_val_set:
        _add_coco(img, MERGED_VA_IMG, MERGED_VA_LBL)

    print(f"  train : {n_tr} hospital + {len(coco_tr_set)} COCO = {n_tr + len(coco_tr_set)} total")
    print(f"  val   : {n_va} hospital + {len(coco_val_set)} COCO = {n_va + len(coco_val_set)} total")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Write hospital_data.yaml
# ═══════════════════════════════════════════════════════════════════════════════
def write_yaml() -> None:
    print("\n" + "=" * 60)
    print("STEP 3: Writing hospital_data.yaml")
    print("=" * 60)

    cfg = {
        "path":  str(BASE_DIR / "hospital_merged"),
        "train": "images/train",
        "val":   "images/val",
        "nc":    106,
        "names": MERGED_NAMES,
    }
    with open(OUTPUT_YAML, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"  Written : {OUTPUT_YAML}")
    print(f"  Classes : 106 (80 COCO + 26 hospital)")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Two-phase training
# ═══════════════════════════════════════════════════════════════════════════════
def train() -> None:
    print("\n" + "=" * 60)
    print("STEP 4: Two-phase training (yolo26m, 106 classes)")
    print("=" * 60)

    from ultralytics import YOLO

    common = dict(
        data=str(OUTPUT_YAML),
        imgsz=640,
        optimizer="AdamW",
        weight_decay=0.0005,
        warmup_epochs=3,
        batch=8,        # medium model on 6GB VRAM
        workers=4,
        amp=False,      # disabled for GTX 1660 Ti stability
        plots=True,
        project=str(BASE_DIR / "runs" / "hospital"),
        pretrained=True,
    )

    # ── Phase 1: head-only (freeze backbone + neck) ───────────────────────
    # Topology: layers 0-9 backbone | 10 C2PSA | 11-22 neck | 23 Detect
    # freeze=22 → only the Detect head (layer 23) trains.
    print("\n  Phase 1 — head-only (30 epochs, freeze=22, lr=0.001) …")
    model = YOLO(str(MODEL_WEIGHTS))
    model.train(
        **common,
        epochs=30,
        freeze=22,
        lr0=0.001,
        lrf=0.01,
        warmup_bias_lr=0.001,   # matches lr0 — prevents warmup spike on bias params
        patience=10,
        name="phase1_head",
    )

    # ── Phase 2: neck + head (unfreeze neck at low LR) ────────────────────
    # Resume from phase 1 best. Layers 11-22 (neck) unfreeze.
    phase1_best = BASE_DIR / "runs" / "hospital" / "phase1_head" / "weights" / "best.pt"
    print(f"\n  Phase 2 — neck+head (70 epochs, freeze=10, lr=0.0002) …")
    print(f"  Resuming from: {phase1_best}")
    model2 = YOLO(str(phase1_best))
    model2.train(
        **common,
        epochs=70,
        freeze=10,
        lr0=0.0002,
        lrf=0.01,
        warmup_bias_lr=0.0002,  # matches lr0 — fixes the 500× spike bug from prev runs
        patience=30,
        name="phase2_neck_head",
    )

    final = BASE_DIR / "runs" / "hospital" / "phase2_neck_head" / "weights" / "best.pt"
    print(f"\nFinal weights: {final}")
    print("Detects: 80 COCO + 26 hospital = 106 classes")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    validate()
    build_merged_dirs()
    write_yaml()
    train()
