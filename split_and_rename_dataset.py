"""
Split new_dataset/ train images into train/val (85/15),
then rename the folder to hospital_v2_dataset/.
"""

import random
import shutil
from pathlib import Path

SRC_DIR   = Path("new_dataset")
DST_DIR   = Path("hospital_v2_dataset")
VAL_RATIO = 0.15
SEED      = 42

random.seed(SEED)

img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

train_img_dir = SRC_DIR / "images" / "train"
train_lbl_dir = SRC_DIR / "labels" / "train"

# Collect all train images
all_imgs = sorted([p for p in train_img_dir.iterdir() if p.suffix.lower() in img_exts])
random.shuffle(all_imgs)

n_val = int(len(all_imgs) * VAL_RATIO)
val_imgs   = all_imgs[:n_val]
train_imgs = all_imgs[n_val:]

print(f"Total train images : {len(all_imgs)}")
print(f"→ keeping train    : {len(train_imgs)}")
print(f"→ moving to val    : {n_val}")

# Create destination dirs
for split in ("train", "val"):
    (DST_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (DST_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

def copy_pair(img_path: Path, dst_img_dir: Path, dst_lbl_dir: Path):
    shutil.copy2(img_path, dst_img_dir / img_path.name)
    lbl = train_lbl_dir / (img_path.stem + ".txt")
    dst_lbl = dst_lbl_dir / (img_path.stem + ".txt")
    if lbl.exists():
        shutil.copy2(lbl, dst_lbl)

print("\nCopying train split…")
for img in train_imgs:
    copy_pair(img, DST_DIR / "images" / "train", DST_DIR / "labels" / "train")

print("Copying val split…")
for img in val_imgs:
    copy_pair(img, DST_DIR / "images" / "val", DST_DIR / "labels" / "val")

# Write yaml
classes = ["door", "fire_extinguisher", "bag", "exit_sign", "wheelchair", "spillage"]
names_str = "\n".join(f"- {n}" for n in classes)
yaml_content = (
    f"path: {DST_DIR.resolve()}\n"
    f"train: images/train\n"
    f"val: images/val\n"
    f"nc: {len(classes)}\n"
    f"names:\n{names_str}\n"
)
(DST_DIR / "hospital_v2_dataset.yaml").write_text(yaml_content)
print(f"\nWrote {DST_DIR / 'hospital_v2_dataset.yaml'}")

# Count val labels for verification
n_val_lbls = sum(1 for f in (DST_DIR / "labels" / "val").iterdir() if f.suffix == ".txt")
n_train_lbls = sum(1 for f in (DST_DIR / "labels" / "train").iterdir() if f.suffix == ".txt")

print(f"\nDONE — {DST_DIR}/")
print(f"  train: {len(train_imgs)} images, {n_train_lbls} label files")
print(f"  val  : {len(val_imgs)} images, {n_val_lbls} label files")
print(f"\nClasses:")
for i, name in enumerate(classes):
    print(f"  {i}: {name}")
