import importlib.util
from pathlib import Path

from ultralytics import YOLO

script_path = Path(r"D:\Object Detection Model\yolo_tr\yolo_tr\Cognitive Recognition framework\01_codebase\04_rgbd_and_spatial_twin\hospital_detector_temporal\infer_hospitalguard_temporal.py")
video_path = Path(r"D:\Object Detection Model\yolo_tr\yolo_tr\Cognitive Recognition framework\10_Testing\Rules for AD\Person_blocking_hospital_exit_202606091138.mp4")
out_dir = Path(r"D:\Object Detection Model\yolo_tr\yolo_tr\Cognitive Recognition framework\04_outputs_runs_and_logs\AD_Rules_Outputs")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"{video_path.stem}_annotated_rerun7.mp4"

spec = importlib.util.spec_from_file_location("hgt", script_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

v1_path = Path(r"D:\Object Detection Model\yolo_tr\yolo_tr\Cognitive Recognition framework\04_outputs_runs_and_logs\outputs\runs\hospital\phase2_neck_head\weights\best.pt")
v3_path = Path(r"D:\Object Detection Model\yolo_tr\yolo_tr\Cognitive Recognition framework\04_outputs_runs_and_logs\outputs\runs\hospital_v3\phase2_neck_head\weights\best.pt")

v1 = YOLO(str(v1_path))
v3 = YOLO(str(v3_path))
v1.to("cuda")
v3.to("cuda")

confs, note = mod.run_video(v1, v3, video_path, out_path)
print("OUTPUT:", out_path)
print("NUM_CLASSES:", len(confs))
print("NOTE:", note)
