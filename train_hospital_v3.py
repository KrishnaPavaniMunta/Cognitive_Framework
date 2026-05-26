"""
train_hospital_v3.py
────────────────────
Extends the 106-class hospital model to 109 classes by incorporating
datasets/hospital_v2_dataset/ (6 external Roboflow classes).

New classes added (IDs 106-108):
  106  bag
  107  exit_sign
  108  spillage

Reinforced existing classes (already in 80-105 range):
  94   wheelchair        (v2 source class 4)
  96   door              (v2 source class 0)
  100  fire_extinguisher (v2 source class 1)

Two-phase freeze strategy:
  Phase 1 (30 epochs, freeze=22) : head-only  — adapts Detect head to 109 classes
  Phase 2 (70 epochs, freeze=15) : deeper neck+head — refined convergence

Phase 2 refinements vs. previous run:
  freeze        : 10   -> 15       (protects early neck layers 11-14)
  lr0           : 0.0002 -> 0.00005 (gentler adaptation for new head)
  warmup_epochs : 3    -> 5        (extra stabilisation at low LR)
  patience      : 30   -> 50       (more room to escape plateau)
  close_mosaic  : 10              (unchanged - already effective)

Oversampling (train split only - proportional to class rarity):
  spillage   -> 3x  (~1,900 source images)
  exit_sign  -> 2x  (~3,000 source images)
  others     -> 1x

Run:
    cd /home/kelvin/yolo_tr
    python train_hospital_v3.py
"""

import shutil
from pathlib import Path

import yaml

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent.resolve()

HOSP_MERGED  = BASE_DIR / "datasets" / "hospital_merged"
V2_DATASET   = BASE_DIR / "datasets" / "hospital_v2_dataset"
V3_MERGED    = BASE_DIR / "datasets" / "hospital_v3_merged"
OUTPUT_YAML  = BASE_DIR / "hospital_v3_data.yaml"
BASE_WEIGHTS = (BASE_DIR / "outputs" / "runs" / "hospital"
                / "phase2_neck_head" / "weights" / "best.pt")
RUNS_DIR     = BASE_DIR / "outputs" / "runs" / "hospital_v3"

# ── Class mapping: v2 source ID -> merged 109-class ID ────────────────────────
V2_CLASS_MAP = {
    0: 96,   # door              (existing)
    1: 100,  # fire_extinguisher (existing)
    2: 106,  # bag               (NEW)
    3: 107,  # exit_sign         (NEW)
    4: 94,   # wheelchair        (existing)
    5: 108,  # spillage          (NEW)
}

# Oversampling repeat count per v2 source class (train only).
# An image's repeat count = max over all classes present in that image.
V2_OVERSAMPLE = {
    0: 1,   # door
    1: 1,   # fire_extinguisher
    2: 1,   # bag
    3: 2,   # exit_sign
    4: 1,   # wheelchair
    5: 3,   # spillage
}

# ── 109-class name list ────────────────────────────────────────────────────────
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
HOSPITAL_NAMES = [
    "cabinet", "glove", "healthcare_worker", "hospital_bed", "infusion_pump",
    "iv_bag", "iv_stand", "monitor_hosp", "nasal_cannula", "patient",
    "patient_monitor", "surgical_light", "test_tube", "vending_machines",
    "wheelchair", "bench_hosp", "door", "reception_counter", "radiator",
    "bathroom_labels", "fire_extinguisher", "hospital_stretcher",
    "security_camera", "hair_net", "mask", "surgical_scissor",
]
NEW_NAMES = ["bag", "exit_sign", "spillage"]

MERGED_NAMES = COCO_NAMES + HOSPITAL_NAMES + NEW_NAMES
assert len(COCO_NAMES)     == 80,  f"Expected 80 COCO names, got {len(COCO_NAMES)}"
assert len(HOSPITAL_NAMES) == 26,  f"Expected 26 hospital names, got {len(HOSPITAL_NAMES)}"
assert len(MERGED_NAMES)   == 109, f"Expected 109 merged names, got {len(MERGED_NAMES)}"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".PNG", ".JPEG"}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Validate prerequisites
# ══════════════════════════════════════════════════════════════════════════════
def validate() -> None:
    print("\n" + "=" * 60)
    print("STEP 1: Validating prerequisites")
    print("=" * 60)

    checks = [
        (HOSP_MERGED / "images" / "train", "datasets/hospital_merged/images/train"),
        (HOSP_MERGED / "images" / "val",   "datasets/hospital_merged/images/val"),
        (V2_DATASET  / "images" / "train", "datasets/hospital_v2_dataset/images/train"),
        (V2_DATASET  / "labels" / "train", "datasets/hospital_v2_dataset/labels/train"),
        (V2_DATASET  / "images" / "val",   "datasets/hospital_v2_dataset/images/val"),
        (BASE_WEIGHTS,                     str(BASE_WEIGHTS.relative_to(BASE_DIR))),
    ]
    missing = [label for path, label in checks if not path.exists()]
    if missing:
        print("\nERROR — Missing prerequisites:")
        for m in missing:
            print(f"  x {m}")
        raise SystemExit(1)

    def _count(d: Path) -> int:
        return sum(1 for _ in d.iterdir())

    print(f"  OK  hospital_merged  train : {_count(HOSP_MERGED / 'images' / 'train')} entries")
    print(f"  OK  hospital_merged  val   : {_count(HOSP_MERGED / 'images' / 'val')} entries")
    print(f"  OK  hospital_v2      train : {_count(V2_DATASET  / 'images' / 'train')} images")
    print(f"  OK  hospital_v2      val   : {_count(V2_DATASET  / 'images' / 'val')} images")
    print(f"  OK  Base weights           : {BASE_WEIGHTS.name}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Build hospital_v3_merged/
# ══════════════════════════════════════════════════════════════════════════════
def build_merged_dirs() -> None:
    print("\n" + "=" * 60)
    print("STEP 2: Building hospital_v3_merged/")
    print("=" * 60)

    train_img_dir = V3_MERGED / "images" / "train"
    if train_img_dir.exists() and any(train_img_dir.iterdir()):
        print("  [skip] hospital_v3_merged/ already populated.")
        return

    for split in ("train", "val"):
        (V3_MERGED / "images" / split).mkdir(parents=True, exist_ok=True)
        (V3_MERGED / "labels" / split).mkdir(parents=True, exist_ok=True)

    # ── 2a: Symlink hospital_merged content ───────────────────────────────
    # Resolve existing symlinks so paths remain valid even if hospital_merged
    # itself contains symlinks into _archive/.
    for split in ("train", "val"):
        n = 0
        src_img_dir = HOSP_MERGED / "images" / split
        src_lbl_dir = HOSP_MERGED / "labels" / split
        dst_img_dir = V3_MERGED   / "images" / split
        dst_lbl_dir = V3_MERGED   / "labels" / split

        for src_img in src_img_dir.iterdir():
            dst_img = dst_img_dir / src_img.name
            if not dst_img.exists():
                dst_img.symlink_to(src_img.resolve())

            src_lbl = src_lbl_dir / (src_img.stem + ".txt")
            dst_lbl = dst_lbl_dir / (src_img.stem + ".txt")
            if src_lbl.exists() and not dst_lbl.exists():
                dst_lbl.symlink_to(src_lbl.resolve())
            n += 1

        print(f"  [{split}] hospital_merged symlinks : {n}")

    # ── 2b: Remap + copy v2 data with oversampling ────────────────────────
    for split in ("train", "val"):
        src_img_dir = V2_DATASET / "images" / split
        src_lbl_dir = V2_DATASET / "labels" / split
        dst_img_dir = V3_MERGED  / "images" / split
        dst_lbl_dir = V3_MERGED  / "labels" / split

        if not src_img_dir.exists():
            print(f"  [{split}] v2 images dir not found — skip")
            continue

        n_src   = 0
        n_total = 0

        for img_file in sorted(src_img_dir.iterdir()):
            if img_file.suffix.lower() not in IMG_EXTS:
                continue

            lbl_file = src_lbl_dir / (img_file.stem + ".txt")
            if not lbl_file.exists():
                continue

            # Remap class IDs; skip lines whose class is not in V2_CLASS_MAP
            src_classes = set()
            remapped_lines = []
            for line in lbl_file.read_text().splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                src_cls = int(parts[0])
                if src_cls not in V2_CLASS_MAP:
                    continue
                src_classes.add(src_cls)
                remapped_lines.append(
                    f"{V2_CLASS_MAP[src_cls]} " + " ".join(parts[1:])
                )

            if not remapped_lines:
                continue  # no relevant annotations after remapping

            n_src += 1
            label_text = "\n".join(remapped_lines) + "\n"

            # How many copies? Max oversample rate of any class in this image.
            # Val split always uses 1 copy (no oversampling).
            repeats = (
                max(V2_OVERSAMPLE.get(c, 1) for c in src_classes)
                if split == "train" else 1
            )

            for r in range(repeats):
                # r=0 -> v2_stem.ext
                # r>0 -> v2_stem_r1.ext, v2_stem_r2.ext, ...
                # Image and label are always created as a matched pair.
                suffix   = f"_r{r}" if r > 0 else ""
                new_stem = f"v2_{img_file.stem}{suffix}"

                dst_img = dst_img_dir / (new_stem + img_file.suffix)
                dst_lbl = dst_lbl_dir / (new_stem + ".txt")

                if not dst_img.exists():
                    shutil.copy2(img_file, dst_img)
                if not dst_lbl.exists():
                    dst_lbl.write_text(label_text)

                n_total += 1

        print(f"  [{split}] v2 source: {n_src} images  ->  {n_total} copies after oversampling")

    # Final summary
    print()
    for split in ("train", "val"):
        n_imgs = sum(1 for _ in (V3_MERGED / "images" / split).iterdir())
        n_lbls = sum(1 for _ in (V3_MERGED / "labels" / split).glob("*.txt"))
        print(f"  [{split}] TOTAL : {n_imgs} images,  {n_lbls} label files")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Write hospital_v3_data.yaml
# ══════════════════════════════════════════════════════════════════════════════
def write_yaml() -> None:
    print("\n" + "=" * 60)
    print("STEP 3: Writing hospital_v3_data.yaml")
    print("=" * 60)

    cfg = {
        "path":  str(V3_MERGED),
        "train": "images/train",
        "val":   "images/val",
        "nc":    109,
        "names": MERGED_NAMES,
    }
    with open(OUTPUT_YAML, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"  Written : {OUTPUT_YAML}")
    print(f"  Classes : 109  (80 COCO + 26 hospital + 3 new)")
    print(f"  New IDs : 106=bag   107=exit_sign   108=spillage")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Two-phase training
# ══════════════════════════════════════════════════════════════════════════════
def train() -> None:
    print("\n" + "=" * 60)
    print("STEP 4: Two-phase training (109 classes)")
    print("=" * 60)

    from ultralytics import YOLO

    common = dict(
        data=str(OUTPUT_YAML),
        imgsz=640,
        optimizer="AdamW",
        weight_decay=0.0005,
        batch=8,
        workers=0,       # 0 = main-process dataloader, avoids worker deadlocks
        amp=False,       # disabled for GTX 1660 Ti stability
        plots=True,
        project=str(RUNS_DIR),
        pretrained=True,
    )

    # ── Phase 1: head-only ────────────────────────────────────────────────
    # Load the 106-class best.pt. YOLO auto-rebuilds the Detect head for 109
    # classes, transferring all backbone + neck weights intact.
    # freeze=22 locks backbone (0-9) + C2PSA (10) + full neck (11-22).
    # Only the new 109-class Detect head (layer 23) trains.
    print(f"\n  Phase 1 — head-only  |  30 ep  |  freeze=22  |  lr=0.001")
    print(f"  Base: {BASE_WEIGHTS}")
    model = YOLO(str(BASE_WEIGHTS))
    model.train(
        **common,
        epochs=30,
        freeze=22,
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        warmup_bias_lr=0.001,
        close_mosaic=10,
        patience=10,
        name="phase1_head",
    )

    # ── Phase 2: deeper neck + head ───────────────────────────────────────
    # freeze=15: locks backbone (0-9) + C2PSA (10) + early neck (11-14).
    #            Unlocks deeper neck layers 15-22 and head.
    # lr0=0.00005: 4x lower than previous run — gentler neck adaptation.
    # warmup_bias_lr MUST match lr0 exactly to prevent the ~500x warmup spike.
    # warmup_epochs=5: extra ramp-up time in the low-LR regime.
    # close_mosaic=10: disable Mosaic for last 10 epochs -> cleaner geometry.
    # patience=50: more room for the optimizer to reach a deeper minimum.
    # YOLO auto-increments the run dir (phase1_head → phase1_head2, etc.)
    # find the most-recently-modified phase1_head* dir to get the real best.pt
    phase1_candidates = sorted(RUNS_DIR.glob("phase1_head*"),
                               key=lambda p: p.stat().st_mtime, reverse=True)
    if not phase1_candidates:
        raise FileNotFoundError(f"No phase1_head* dir found under {RUNS_DIR}")
    phase1_best = phase1_candidates[0] / "weights" / "best.pt"
    print(f"\n  Phase 2 — neck+head  |  70 ep  |  freeze=15  |  lr=0.00005")
    print(f"  Base: {phase1_best}")
    model2 = YOLO(str(phase1_best))
    model2.train(
        **common,
        epochs=70,
        freeze=15,
        lr0=0.00005,
        lrf=0.01,
        warmup_epochs=5,
        warmup_bias_lr=0.00005,
        close_mosaic=10,
        patience=50,
        name="phase2_neck_head",
    )

    final = RUNS_DIR / "phase2_neck_head" / "weights" / "best.pt"
    print(f"\n{'='*60}")
    print(f"Final weights : {final}")
    print(f"Detects       : 80 COCO + 26 hospital + 3 new = 109 classes")
    print(f"  New classes : 106=bag   107=exit_sign   108=spillage")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    validate()
    build_merged_dirs()
    write_yaml()
    train()
