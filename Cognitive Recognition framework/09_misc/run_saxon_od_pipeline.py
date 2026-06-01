from pathlib import Path
import sys

from ultralytics import YOLO

# Add inference script folder to import path.
ROOT = Path(__file__).resolve().parents[1]
INFER_DIR = ROOT / "01_codebase" / "02_inference"
if str(INFER_DIR) not in sys.path:
    sys.path.insert(0, str(INFER_DIR))

import infer_hospitalguard as hg  # noqa: E402


def main() -> None:
    input_video = ROOT / "02_datasets" / "saxon" / "rgbd_clean_20260521_104913.mp4"

    v1_weights = ROOT / "04_outputs_runs_and_logs" / "outputs" / "runs" / "hospital" / "phase2_neck_head" / "weights" / "best.pt"
    v3_weights = ROOT / "04_outputs_runs_and_logs" / "outputs" / "runs" / "hospital_v3" / "phase2_neck_head" / "weights" / "best.pt"

    out_dir = ROOT / "04_outputs_runs_and_logs" / "outputs" / "saxon_OD_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "rgbd_clean_20260521_104913_OD.mp4"

    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")
    if not v1_weights.exists():
        raise FileNotFoundError(f"V1 weights not found: {v1_weights}")
    if not v3_weights.exists():
        raise FileNotFoundError(f"V3 weights not found: {v3_weights}")

    print(f"Loading V1: {v1_weights}")
    v1 = YOLO(str(v1_weights))
    print(f"Loading V3: {v3_weights}")
    v3 = YOLO(str(v3_weights))

    print(f"Running OD pipeline on: {input_video}")
    hg.run_video(v1, v3, input_video, out_path)
    print(f"Saved output video: {out_path}")


if __name__ == "__main__":
    main()
