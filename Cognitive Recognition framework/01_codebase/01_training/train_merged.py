"""
train_merged.py
───────────────
Fine-tunes yolo26n.pt to detect BOTH the original 80 COCO classes AND 10 new
classes, by mixing in COCO 2017 val (5,000 images) for genuine COCO supervision.

Strategy: COCO2017-val mixing + two-phase freeze
─────────────────────────────────────────────────
The core problem: if you train a new head on only 10 classes, the model forgets
all 80 COCO classes entirely (the head has no outputs for them).

Solution (uses COCO 2017 val — ~5,000 images, ~6.4 GB download on first run):
  1. Download COCO 2017 val images + YOLO-format labels (~6.3 GB images,
     ~100 MB labels).  Labels stay at their original indices 0-79.
  2. Remap the 10 new-class labels to indices 80-88 and merge them with the
     COCO val labels in a single flat directory.
  3. Train in two phases:
       Phase 1 (30 epochs, freeze=22): only the detection head trains —
         head learns all 89 class outputs while backbone + neck stay frozen.
       Phase 2 (70 epochs, freeze=10): neck unfreezes at a low LR —
         neck adapts gently with both COCO and new data well-represented.

Result: the output model detects all 80 original COCO classes + 9 new ones
(Mouse merges with COCO index 64).

Class index mapping (new dataset → 89-class space)
───────────────────────────────────────────────────
  0 Airpods        → 80      4 Mouse  → 64 (reuses COCO mouse)
  1 Glasses        → 81      5 Pen    → 84
  2 Key            → 82      6 Power Adapter → 85
  3 Lightning Cable→ 83      7 Telephone → 86
                             8 Wallet → 87
                             9 Watch  → 88

Run
───
    cd /home/kelvin/yolo_tr
    python train_merged.py
"""

import random
import shutil
import urllib.request
import zipfile
from pathlib import Path

import yaml

# ─── PATHS ────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent.resolve()   # /home/kelvin/yolo_tr
TRAIN_IMG       = BASE_DIR / "train" / "images"
TRAIN_LBL       = BASE_DIR / "train" / "labels"        # original 10-class labels
VALID_IMG       = BASE_DIR / "valid" / "images"
VALID_LBL       = BASE_DIR / "valid" / "labels"
# Ultralytics resolves labels by replacing 'images' → 'labels' in the path,
# so merged dirs must follow the same images/labels sibling convention.
MERGED_TRAIN_IMG  = BASE_DIR / "train_merged" / "images"
MERGED_TRAIN_LBL  = BASE_DIR / "train_merged" / "labels"
MERGED_VALID_IMG  = BASE_DIR / "valid_merged" / "images"
MERGED_VALID_LBL  = BASE_DIR / "valid_merged" / "labels"
MODEL_WEIGHTS   = BASE_DIR / "yolo26n.pt"
OUTPUT_YAML     = BASE_DIR / "finetune_data.yaml"

# COCO 2017 val — 5,000 real COCO images (~6.3 GB images + ~100 MB labels)
# NOTE: first run will download ~6.4 GB total; subsequent runs skip this step.
COCO_LABELS_URL  = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip"
COCO_VAL_IMG_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_DIR         = BASE_DIR / "coco2017val"
COCO_IMG         = COCO_DIR / "val2017"
COCO_LBL         = COCO_DIR / "coco" / "labels" / "val2017"

VALID_SPLIT     = 0.20
RANDOM_SEED     = 42

# ─── CLASS MAPPING ────────────────────────────────────────────────────────────
# New dataset (0-9) → merged 89-class index
REMAP = {0: 80, 1: 81, 2: 82, 3: 83, 4: 64,
         5: 84, 6: 85, 7: 86, 8: 87, 9: 88}

MERGED_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra",
    "giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove",
    "skateboard","surfboard","tennis racket","bottle","wine glass","cup",
    "fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop",
    "mouse",            # 64 — shared with new-dataset Mouse
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush",
    "Airpods","Glasses","Key","Lightning Cable",   # 80-83
    "Pen","Power Adapter","Telephone","Wallet","Watch",  # 84-88
]
assert len(MERGED_NAMES) == 89


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Create train / valid split
# ═══════════════════════════════════════════════════════════════════════════════
def create_valid_split() -> None:
    print("\n" + "=" * 60)
    print("STEP 1: Creating train/valid split (80/20)")
    print("=" * 60)

    if VALID_IMG.exists() and any(VALID_IMG.iterdir()):
        print("  [skip] valid/ already has files — split already done.")
        return

    VALID_IMG.mkdir(parents=True, exist_ok=True)
    VALID_LBL.mkdir(parents=True, exist_ok=True)

    all_images = sorted(TRAIN_IMG.glob("*.jpg")) or sorted(TRAIN_IMG.glob("*.png"))

    random.seed(RANDOM_SEED)
    random.shuffle(all_images)

    n_valid = int(len(all_images) * VALID_SPLIT)
    valid_images = all_images[:n_valid]

    moved = 0
    for img_path in valid_images:
        lbl_path = TRAIN_LBL / (img_path.stem + ".txt")
        shutil.move(str(img_path), VALID_IMG / img_path.name)
        if lbl_path.exists():
            shutil.move(str(lbl_path), VALID_LBL / lbl_path.name)
        moved += 1

    remaining = len(list(TRAIN_IMG.glob("*.jpg")))
    print(f"  Moved {moved} images → valid/")
    print(f"  Train: {remaining}  |  Valid: {moved}")



# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Download COCO 2017 val (~6.4 GB) if not already present
# ═══════════════════════════════════════════════════════════════════════════════
def download_coco2017val() -> None:
    print("\n" + "=" * 60)
    print("STEP 2: Downloading COCO 2017 val (~6.4 GB — only on first run)")
    print("=" * 60)

    COCO_DIR.mkdir(parents=True, exist_ok=True)

    # ── Labels (~100 MB) ────────────────────────────────────────────────
    if COCO_LBL.exists() and any(COCO_LBL.iterdir()):
        print(f"  [skip] COCO labels already present.")
    else:
        lbl_zip = COCO_DIR / "coco2017labels.zip"
        print(f"  Downloading COCO labels (~100 MB) …")
        urllib.request.urlretrieve(COCO_LABELS_URL, lbl_zip)
        print(f"  Extracting labels …")
        with zipfile.ZipFile(lbl_zip, "r") as zf:
            zf.extractall(COCO_DIR)
        lbl_zip.unlink()
        print(f"  Labels ready at {COCO_LBL}")

    # ── Images (~6.3 GB) ────────────────────────────────────────────────
    if COCO_IMG.exists() and any(COCO_IMG.iterdir()):
        n = sum(1 for _ in COCO_IMG.glob("*.jpg"))
        print(f"  [skip] COCO val images already present ({n} images).")
    else:
        img_zip = COCO_DIR / "val2017.zip"
        print(f"  Downloading COCO val images (~6.3 GB) — this may take a while …")
        urllib.request.urlretrieve(COCO_VAL_IMG_URL, img_zip)
        print(f"  Extracting images …")
        with zipfile.ZipFile(img_zip, "r") as zf:
            zf.extractall(COCO_DIR)
        img_zip.unlink()
        n = sum(1 for _ in COCO_IMG.glob("*.jpg"))
        print(f"  Done — {n} COCO val images at {COCO_IMG}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Build merged train/valid directories
# ═══════════════════════════════════════════════════════════════════════════════
def build_merged_dirs() -> None:
    """
    train_merged/:
      images/ — symlinks to new-dataset train images + ~4,000 COCO 2017 val images
      labels/ — new-dataset labels remapped (0-9 → 80-88) + COCO labels (0-79 unchanged)

    valid_merged/:
      images/ — symlinks to new-dataset valid images + ~1,000 COCO 2017 val images
      labels/ — new-dataset labels remapped (0-9 → 80-88) + COCO labels (0-79 unchanged)
    """
    print("\n" + "=" * 60)
    print("STEP 3: Building merged train / valid directories")
    print("=" * 60)

    for d in (MERGED_TRAIN_IMG, MERGED_TRAIN_LBL,
              MERGED_VALID_IMG, MERGED_VALID_LBL):
        d.mkdir(parents=True, exist_ok=True)

    # ── Helper: symlink an image, skip if link already exists ──────────
    def _link(src: Path, dst_dir: Path) -> None:
        dst = dst_dir / src.name
        if not dst.exists():
            dst.symlink_to(src.resolve())

    # ── Helper: remap a label file from old→new class index ────────────
    def _remap_label(src: Path, dst: Path, remap: dict) -> None:
        if dst.exists():
            return
        lines = []
        for line in src.read_text().splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            new_cls = remap.get(int(parts[0]), int(parts[0]))
            lines.append(f"{new_cls} {' '.join(parts[1:])}")
        dst.write_text("\n".join(lines) + "\n")

    # ── New-dataset train images & remapped labels ──────────────────────
    new_train_imgs = sorted(TRAIN_IMG.glob("*.jpg")) + sorted(TRAIN_IMG.glob("*.png"))
    for img in new_train_imgs:
        _link(img, MERGED_TRAIN_IMG)
        lbl = TRAIN_LBL / (img.stem + ".txt")
        if lbl.exists():
            _remap_label(lbl, MERGED_TRAIN_LBL / lbl.name, REMAP)

    # ── New-dataset valid images & remapped labels ──────────────────────
    new_valid_imgs = sorted(VALID_IMG.glob("*.jpg")) + sorted(VALID_IMG.glob("*.png"))
    for img in new_valid_imgs:
        _link(img, MERGED_VALID_IMG)
        lbl = VALID_LBL / (img.stem + ".txt")
        if lbl.exists():
            _remap_label(lbl, MERGED_VALID_LBL / lbl.name, REMAP)

    # ── COCO 2017 val images & labels — 80% train / 20% val (deterministic) ──
    rng = random.Random(RANDOM_SEED)
    coco_imgs = sorted(COCO_IMG.glob("*.jpg")) + sorted(COCO_IMG.glob("*.png"))
    rng.shuffle(coco_imgs)
    n_coco_val  = int(len(coco_imgs) * VALID_SPLIT)
    coco_val    = coco_imgs[:n_coco_val]    # ~1,000 images for validation
    coco_train  = coco_imgs[n_coco_val:]    # ~4,000 images for training

    def _add_coco(img: Path, img_dir: Path, lbl_dir: Path) -> None:
        """Symlink one COCO image and copy its label (prefixed to avoid clashes)."""
        dst = img_dir / ("coco_" + img.name)
        if not dst.exists():
            dst.symlink_to(img.resolve())
        lbl = COCO_LBL / (img.stem + ".txt")
        if lbl.exists():
            dst_lbl = lbl_dir / ("coco_" + lbl.name)
            if not dst_lbl.exists():
                dst_lbl.write_text(lbl.read_text())

    for img in coco_train:
        _add_coco(img, MERGED_TRAIN_IMG, MERGED_TRAIN_LBL)
    for img in coco_val:
        _add_coco(img, MERGED_VALID_IMG, MERGED_VALID_LBL)

    n_train_new = len(new_train_imgs)
    n_valid_new = len(new_valid_imgs)
    print(f"  train_merged : {n_train_new} new + {len(coco_train)} COCO = {n_train_new + len(coco_train)} total")
    print(f"  valid_merged : {n_valid_new} new + {len(coco_val)} COCO = {n_valid_new + len(coco_val)} total")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Write finetune_data.yaml
# ═══════════════════════════════════════════════════════════════════════════════
def write_yaml() -> None:
    print("\n" + "=" * 60)
    print("STEP 4: Writing finetune_data.yaml")
    print("=" * 60)

    # Ultralytics resolves labels by replacing 'images' → 'labels' in the path.
    # We point it at the merged image dirs (which symlink the real images);
    # the sibling 'labels' dirs hold the merged pseudo + real annotations.
    cfg = {
        "train": str(MERGED_TRAIN_IMG),
        "val":   str(MERGED_VALID_IMG),
        "nc":    89,
        "names": MERGED_NAMES,
    }

    with open(OUTPUT_YAML, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"  Written : {OUTPUT_YAML}")
    print(f"  Classes : 89 (80 COCO + 9 new)")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Train
# ═══════════════════════════════════════════════════════════════════════════════
def train() -> None:
    print("\n" + "=" * 60)
    print("STEP 5: Two-phase training of 89-class model")
    print("=" * 60)

    from ultralytics import YOLO

    common = dict(
        data=str(OUTPUT_YAML),
        imgsz=640,
        optimizer="AdamW",
        weight_decay=0.0005,
        warmup_epochs=3,
        batch=16,
        workers=8,
        plots=True,
        project=str(BASE_DIR / "runs" / "finetune"),
        pretrained=True,
    )

    # ── Phase 1: head-only (freeze backbone + neck) ──────────────────────
    # Architecture: 0-9 backbone | 10 C2PSA | 11-22 neck | 23 Detect
    # freeze=22 → only the Detect head (layer 23) trains.
    # The head learns all 89 class outputs from scratch while the entire
    # feature extractor stays perfectly COCO-faithful.
    print("\n  Phase 1 — head-only training (30 epochs, freeze=22) …")
    model = YOLO(str(MODEL_WEIGHTS))
    model.train(
        **common,
        epochs=30,
        freeze=22,
        lr0=0.001,
        lrf=0.01,
        patience=10,
        name="phase1_head",
    )

    # ── Phase 2: neck + head (gentle neck adaptation) ────────────────────
    # Resume from phase 1's best checkpoint. Unfreeze the neck (layers
    # 11-22) at a low LR so it adapts gently to both COCO and new-class
    # features without overwriting the backbone.
    phase1_best = BASE_DIR / "runs" / "finetune" / "phase1_head" / "weights" / "best.pt"
    print(f"\n  Phase 2 — neck+head training (70 epochs, freeze=10, lr=0.0002) …")
    print(f"  Resuming from: {phase1_best}")
    model2 = YOLO(str(phase1_best))
    model2.train(
        **common,
        epochs=70,
        freeze=10,
        lr0=0.0002,
        lrf=0.01,
        patience=20,
        name="phase2_neck_head",
    )

    weights_path = BASE_DIR / "runs" / "finetune" / "phase2_neck_head" / "weights" / "best.pt"
    print(f"\nFinal weights saved to: {weights_path}")
    print("Model detects: 80 original COCO classes + 9 new classes (89 total)")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    create_valid_split()
    download_coco2017val()
    build_merged_dirs()
    write_yaml()
    train()
