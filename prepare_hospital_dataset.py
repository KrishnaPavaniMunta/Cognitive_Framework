#!/usr/bin/env python3
"""
prepare_hospital_dataset.py

1. Filters classes with < MIN_TRAIN train instances or < MIN_VAL val instances
2. Remaps remaining class IDs to 0..N-1
3. Drops images where all annotations are filtered out
4. Oversamples minority-class training images (via duplication + augmentation)
   to push each class toward TARGET instances
5. Outputs to Hospital_Dataset_filtered/ with a corrected dataset.yaml
"""

import random
import shutil
from collections import defaultdict
from pathlib import Path

try:
    import yaml
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml", "-q"])
    import yaml

# ── Config ────────────────────────────────────────────────────────────────────
SRC       = Path("/home/kelvin/yolo_tr/Hospital_Dataset")
DST       = Path("/home/kelvin/yolo_tr/Hospital_Dataset_filtered")
MIN_TRAIN = 10   # minimum train instances required to keep a class
MIN_VAL   = 1    # minimum val instances required to keep a class
TARGET    = 50   # target train instances per class (oversample up to this)
MAX_COPIES = 4   # max times a single image can appear in train (orig + dups)
SEED      = 42
# ─────────────────────────────────────────────────────────────────────────────

random.seed(SEED)

# ── Load original class names ─────────────────────────────────────────────────
with open(SRC / "dataset.yaml", "rb") as f:
    raw = f.read()
# Strip lines starting with '#' (comment with non-UTF-8 em dash) before parsing
clean = b"\n".join(
    line for line in raw.splitlines() if not line.strip().startswith(b"#")
)
orig_cfg = yaml.safe_load(clean.decode("utf-8", errors="replace"))
orig_names = orig_cfg["names"]

# ── Count instances per class in each split ───────────────────────────────────
def count_split(lbl_dir: Path):
    """Returns (counts_dict, {stem: [cls_id, ...]})"""
    counts = defaultdict(int)
    stem_cls = {}
    for lf in sorted(lbl_dir.glob("*.txt")):
        ids = []
        with open(lf) as f:
            for line in f:
                line = line.strip()
                if line:
                    ids.append(int(line.split()[0]))
        for c in ids:
            counts[c] += 1
        stem_cls[lf.stem] = ids
    return dict(counts), stem_cls

print("Counting instances…")
tr_counts, tr_stem_cls = count_split(SRC / "labels/train")
va_counts, va_stem_cls = count_split(SRC / "labels/val")

# ── Decide which classes survive ──────────────────────────────────────────────
kept = sorted(
    c for c in range(len(orig_names))
    if tr_counts.get(c, 0) >= MIN_TRAIN and va_counts.get(c, 0) >= MIN_VAL
)
kept_set  = set(kept)
remap     = {old: new for new, old in enumerate(kept)}
new_names = [orig_names[c] for c in kept]

print(f"\n{'ID':<4} {'Name':<35} {'Train':>7} {'Val':>5}  Keep")
print("─" * 62)
for c, name in enumerate(orig_names):
    tr = tr_counts.get(c, 0)
    va = va_counts.get(c, 0)
    mark = "✓" if c in kept_set else "✗"
    print(f"{c:<4} {name:<35} {tr:>7} {va:>5}  {mark}")

dropped = [orig_names[c] for c in range(len(orig_names)) if c not in kept_set]
print(f"\nKept   : {len(kept)} classes")
print(f"Dropped: {len(dropped)} classes")
print(f"  {dropped}")

# ── Helpers ───────────────────────────────────────────────────────────────────
IMG_EXTS = [".jpg", ".jpeg", ".png", ".JPG", ".PNG", ".JPEG"]

def find_img(img_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = img_dir / (stem + ext)
        if p.exists():
            return p
    return None

# ── Write a split (filter labels, copy images) ────────────────────────────────
def write_split(split: str, stem_cls_map: dict) -> list[str]:
    """Filter labels and copy matching images. Returns list of kept stems."""
    src_lbl = SRC / "labels" / split
    src_img = SRC / "images" / split
    dst_lbl = DST / "labels" / split
    dst_img = DST / "images" / split
    dst_lbl.mkdir(parents=True, exist_ok=True)
    dst_img.mkdir(parents=True, exist_ok=True)

    kept_stems = []
    for stem in sorted(stem_cls_map):
        src_lf = src_lbl / (stem + ".txt")
        if not src_lf.exists():
            continue

        new_lines = []
        with open(src_lf) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                c = int(parts[0])
                if c in kept_set:
                    parts[0] = str(remap[c])
                    new_lines.append(" ".join(parts))

        if not new_lines:
            continue  # all annotations were filtered out

        img = find_img(src_img, stem)
        if img is None:
            continue

        with open(dst_lbl / (stem + ".txt"), "w") as f:
            f.write("\n".join(new_lines) + "\n")
        shutil.copy2(img, dst_img / img.name)
        kept_stems.append(stem)

    return kept_stems

print("\nWriting val split…")
va_kept = write_split("val", va_stem_cls)
print(f"  {len(va_kept)} images kept")

print("Writing train split (base)…")
tr_kept = write_split("train", tr_stem_cls)
print(f"  {len(tr_kept)} images kept (before oversampling)")

# ── Count filtered train instances (new IDs) ─────────────────────────────────
dst_lbl_tr = DST / "labels/train"
dst_img_tr = DST / "images/train"

new_tr_counts  = defaultdict(int)
new_id_to_stems = defaultdict(list)  # new_id -> stems that contain it

for stem in tr_kept:
    with open(dst_lbl_tr / (stem + ".txt")) as f:
        for line in f:
            if line.strip():
                c = int(line.split()[0])
                new_tr_counts[c] += 1
                new_id_to_stems[c].append(stem)

print(f"\nFiltered train counts (new IDs) — TARGET={TARGET}:")
for new_id, name in enumerate(new_names):
    cnt = new_tr_counts[new_id]
    flag = " ← needs oversampling" if cnt < TARGET else ""
    print(f"  [{new_id:2d}] {name:<35} {cnt:4d}{flag}")

# ── Oversample minority classes ───────────────────────────────────────────────
print(f"\nOversampling (max {MAX_COPIES}x per image)…")
copy_counts = defaultdict(int)  # stem -> extra copies scheduled so far

# Process from most underrepresented to most represented
for new_id in sorted(range(len(new_names)), key=lambda x: new_tr_counts[x]):
    current = new_tr_counts[new_id]
    if current >= TARGET:
        continue

    candidate_stems = list(set(new_id_to_stems[new_id]))
    if not candidate_stems:
        print(f"  [{new_id}] {new_names[new_id]}: no training images, skipping")
        continue

    random.shuffle(candidate_stems)
    idx = 0
    attempts = 0
    max_attempts = len(candidate_stems) * MAX_COPIES * 10

    while current < TARGET and attempts < max_attempts:
        stem = candidate_stems[idx % len(candidate_stems)]
        if copy_counts[stem] < MAX_COPIES - 1:
            copy_counts[stem] += 1
            with open(dst_lbl_tr / (stem + ".txt")) as f:
                added = sum(1 for l in f if l.strip() and int(l.split()[0]) == new_id)
            current += added
        idx += 1
        attempts += 1

    status = f"reached {current}" if current < TARGET else f"ok → {current}"
    print(f"  [{new_id:2d}] {new_names[new_id]:<35} {status}")

# Write the duplicate files
total_extra = 0
for stem, n_extra in copy_counts.items():
    src_lf = dst_lbl_tr / (stem + ".txt")
    img = find_img(dst_img_tr, stem)
    if not src_lf.exists() or img is None:
        continue
    for i in range(1, n_extra + 1):
        new_stem = f"{stem}_dup{i}"
        shutil.copy2(src_lf, dst_lbl_tr / f"{new_stem}.txt")
        shutil.copy2(img,    dst_img_tr / f"{new_stem}{img.suffix}")
        total_extra += 1

print(f"\nAdded {total_extra} duplicate images")
print(f"Total train: {len(tr_kept) + total_extra} images")

# ── Final class balance report ────────────────────────────────────────────────
final_counts = defaultdict(int)
for lf in dst_lbl_tr.glob("*.txt"):
    with open(lf) as f:
        for line in f:
            if line.strip():
                final_counts[int(line.split()[0])] += 1

print("\nFinal train instance counts after oversampling:")
for new_id, name in enumerate(new_names):
    bar = "█" * (final_counts[new_id] // 5)
    print(f"  [{new_id:2d}] {name:<35} {final_counts[new_id]:4d}  {bar}")

# ── Write dataset.yaml ────────────────────────────────────────────────────────
new_cfg = {
    "path":  str(DST),
    "train": "images/train",
    "val":   "images/val",
    "nc":    len(new_names),
    "names": new_names,
}
with open(DST / "dataset.yaml", "w") as f:
    yaml.dump(new_cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

print(f"\nDone. Output: {DST}")
print(f"  nc={len(new_names)} classes | train={len(tr_kept)+total_extra} imgs | val={len(va_kept)} imgs")
print(f"  dataset.yaml written")
