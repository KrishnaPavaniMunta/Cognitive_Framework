"""
Download and merge 6 Roboflow datasets into a unified YOLO dataset.

Output directory : new_dataset/
Output classes   :
  0 - door
  1 - fire_extinguisher
  2 - bag            (Fashion Bag + Shopping Bag merged)
  3 - exit_sign      (all 6 directional exit-sign classes merged)
  4 - wheelchair
  5 - spillage       (all 6 spillage severity classes merged)

Usage:
    python download_new_dataset.py --api-key YOUR_ROBOFLOW_API_KEY

Get a free API key at https://app.roboflow.com/ → Settings → API Keys
"""

import argparse
import os
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Dataset definitions
# Each entry:
#   workspace   : Roboflow workspace slug
#   project     : Roboflow project slug
#   version     : dataset version number (int)
#   class_map   : {source_class_id (int): target_class_id (int)}
#                 any source class NOT in this dict is dropped
# ---------------------------------------------------------------------------
DATASETS = [
    {
        "name": "door",
        "workspace": "door-2wjcn",
        "project": "door-ocdh8",
        "version": 1,
        # source class 0 → target door (0)
        "class_map": {0: 0},
    },
    {
        "name": "fire_extinguisher",
        "workspace": "fireextworkspace",
        "project": "fire_ext-ioldk",
        "version": 1,
        # source class 0 → target fire_extinguisher (1)
        "class_map": {0: 1},
    },
    {
        "name": "bag",
        "workspace": "almas-mnl25",
        "project": "bag-detection-nmfca",
        "version": 1,
        # Fashion Bag (0) + Shopping Bag (1) → target bag (2)
        "class_map": {0: 2, 1: 2},
    },
    {
        "name": "exit_sign",
        "workspace": "emergency-exit-signs",
        "project": "emergency-exit-signs-v2",
        "version": 10,
        # Straight(0) Backwards(1) Left(2) Left-Right(3) Right(4) Straight-Backwards(5)
        # — all are exit signs, merge everything → exit_sign (3)
        "class_map": {0: 3, 1: 3, 2: 3, 3: 3, 4: 3, 5: 3},
    },
    {
        "name": "wheelchair",
        "workspace": "kpz2",
        "project": "wheelchair-grmuz",
        "version": 1,
        # source class 0 → target wheelchair (4)
        "class_map": {0: 4},
    },
    {
        "name": "spillage",
        "workspace": "spillage",
        "project": "spillage-detection",
        "version": 1,
        # severity classes 0-5 → target spillage (5)
        "class_map": {0: 5, 1: 5, 2: 5, 3: 5, 4: 5, 5: 5},
    },
]

TARGET_CLASSES = ["door", "fire_extinguisher", "bag", "exit_sign", "wheelchair", "spillage"]

OUTPUT_DIR = Path("new_dataset")


# ---------------------------------------------------------------------------

def remap_label_file(src_path: Path, dst_path: Path, class_map: dict) -> int:
    """
    Read a YOLO .txt label file, remap class IDs, write to dst_path.
    Returns number of kept annotations.
    """
    kept = 0
    lines_out = []
    if not src_path.exists():
        return 0
    with open(src_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            src_cls = int(parts[0])
            if src_cls not in class_map:
                continue  # drop this annotation
            tgt_cls = class_map[src_cls]
            lines_out.append(f"{tgt_cls} " + " ".join(parts[1:]))
            kept += 1
    if lines_out:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dst_path, "w") as f:
            f.write("\n".join(lines_out) + "\n")
    return kept


def process_split(raw_dir: Path, split: str, class_map: dict,
                  out_img_dir: Path, out_lbl_dir: Path, prefix: str):
    """Copy images and remapped labels for one split (train/val)."""
    img_src = raw_dir / split / "images"
    lbl_src = raw_dir / split / "labels"

    if not img_src.exists():
        print(f"    [skip] {img_src} not found")
        return 0, 0

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    copied_imgs = 0
    copied_anns = 0

    for img_file in img_src.iterdir():
        if img_file.suffix.lower() not in img_exts:
            continue

        new_name = f"{prefix}_{img_file.name}"
        dst_img = out_img_dir / new_name
        shutil.copy2(img_file, dst_img)
        copied_imgs += 1

        lbl_file = lbl_src / (img_file.stem + ".txt")
        dst_lbl = out_lbl_dir / (Path(new_name).stem + ".txt")
        kept = remap_label_file(lbl_file, dst_lbl, class_map)
        if kept > 0:
            copied_anns += kept

    return copied_imgs, copied_anns


def get_latest_version(workspace: str, project: str, api_key: str) -> int:
    """Use the Roboflow REST API to find the latest version number for a project."""
    import urllib.request, json
    url = f"https://api.roboflow.com/{workspace}/{project}?api_key={api_key}"
    with urllib.request.urlopen(url, timeout=15) as resp:
        data = json.loads(resp.read())
    versions_list = data.get("versions", [])
    if not versions_list:
        raise RuntimeError(f"No published versions for {workspace}/{project}. "
                           f"The dataset owner has not exported a downloadable version.")
    nums = []
    for v in versions_list:
        vid = v.get("id", "")
        try:
            nums.append(int(vid.split("/")[-1]))
        except ValueError:
            pass
    if not nums:
        raise RuntimeError(f"Could not parse version numbers for {workspace}/{project}")
    return max(nums)


def download_dataset(ds: dict, api_key: str, download_root: Path):
    """Download a single Roboflow dataset and return its local path."""
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(ds["workspace"]).project(ds["project"])

    # Use REST API to discover the real latest version, then SDK to download
    pinned = ds.get("version", None)
    version = None
    if pinned is not None:
        try:
            version = project.version(pinned)
        except Exception:
            pinned = None

    if version is None:
        latest = get_latest_version(ds["workspace"], ds["project"], api_key)
        print(f"  Detected latest version: v{latest}")
        version = project.version(latest)

    raw_dir = download_root / ds["name"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Downloading {ds['name']} (v{version.version}) …")
    version.download("yolov8", location=str(raw_dir), overwrite=True)
    # Roboflow may download into a nested subfolder — find the one with train/
    for candidate in sorted(raw_dir.rglob("train"), key=lambda p: len(p.parts)):
        if candidate.is_dir():
            return candidate.parent  # parent of train/ is the dataset root
    # fallback: raw_dir itself
    return raw_dir


def write_yaml(out_dir: Path):
    yaml_path = out_dir / "new_dataset.yaml"
    rel_train = "images/train"
    rel_val   = "images/val"
    names_str  = "\n".join(f"- {n}" for n in TARGET_CLASSES)
    content = (
        f"path: {out_dir.resolve()}\n"
        f"train: {rel_train}\n"
        f"val: {rel_val}\n"
        f"nc: {len(TARGET_CLASSES)}\n"
        f"names:\n{names_str}\n"
    )
    yaml_path.write_text(content)
    print(f"\nWrote {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description="Download & merge Roboflow datasets → new_dataset/")
    parser.add_argument("--api-key", required=True, help="Roboflow API key")
    parser.add_argument("--output", default="new_dataset", help="Output directory (default: new_dataset)")
    parser.add_argument("--download-root", default="rf_downloads",
                        help="Temp directory for raw Roboflow downloads (default: rf_downloads)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip downloading; use existing files in --download-root")
    args = parser.parse_args()

    out_dir   = Path(args.output)
    dl_root   = Path(args.download_root)

    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    total_imgs = {"train": 0, "val": 0}
    total_anns = {"train": 0, "val": 0}

    for ds in DATASETS:
        print(f"\n{'='*60}")
        print(f"Processing: {ds['name']}")

        if args.skip_download:
            raw_dir = dl_root / ds["name"]
            # find the actual downloaded subdir
            # find subfolder containing train/
            for candidate in sorted(raw_dir.rglob("train"), key=lambda p: len(p.parts)):
                if candidate.is_dir():
                    raw_dir = candidate.parent
                    break
        else:
            raw_dir = download_dataset(ds, args.api_key, dl_root)

        prefix = ds["name"]

        for split in ("train", "val"):
            out_img = out_dir / "images" / split
            out_lbl = out_dir / "labels" / split
            imgs, anns = process_split(raw_dir, split, ds["class_map"],
                                       out_img, out_lbl, prefix)
            total_imgs[split] += imgs
            total_anns[split] += anns
            print(f"  [{split}] {imgs} images, {anns} annotations kept")

    write_yaml(out_dir)

    print(f"\n{'='*60}")
    print("DONE")
    print(f"  Train: {total_imgs['train']} images, {total_anns['train']} annotations")
    print(f"  Val  : {total_imgs['val']} images, {total_anns['val']} annotations")
    print(f"  Output: {out_dir.resolve()}")
    print(f"\nClasses:")
    for i, name in enumerate(TARGET_CLASSES):
        print(f"  {i}: {name}")


if __name__ == "__main__":
    main()
