import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import tempfile
import argparse
import requests
from pathlib import Path
from datetime import datetime
import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from torchvision.ops import nms as tv_nms
import random
from common_sense_filter import apply_common_sense_rules
from bin_classifier import refine_bin_detections
from dino_prompts import DINO_FALLBACK
from dino_fallback import run_dino_fallback
from rgbd_bag_processing import process_rgbd_bag
from rgbd_3d_filter import apply_depth_size_filter as apply_rgbd_depth_size_filter

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()
REPO_ROOT = BASE_DIR.parents[2] if len(BASE_DIR.parents) >= 3 else BASE_DIR
PROJECT_ROOT = BASE_DIR.parents[1] if len(BASE_DIR.parents) >= 2 else BASE_DIR

V1_PATH = Path(r"D:\Object Detection Model\yolo_tr\yolo_tr\Cognitive Recognition framework\03_models_and_weights\models\yolo_trained_v1.pt")
V2_PATH = Path(r"D:\Object Detection Model\yolo_tr\yolo_tr\Cognitive Recognition framework\03_models_and_weights\yolo_trained_v2.pt")
V3_PATH = Path(r"D:\Object Detection Model\yolo_tr\yolo_tr\Cognitive Recognition framework\03_models_and_weights\models\yolo_trained_v3.pt")
OUT_DIR = Path(r"D:\Object Detection Model\yolo_tr\yolo_tr\Cognitive Recognition framework\04_outputs_runs_and_logs\OD_Outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DIMENSIONS_CONFIG_PATH = BASE_DIR / "hospital_object_dimensions_approx.yaml"

# ── Hyperparameters ────────────────────────────────────────────────────────────
DEFAULT_CONF_THRESH = 0.25
IOU_THRESH = 0.45
DINO_VIDEO_INTERVAL_FRAMES = 15
DINO_HOLD_FRAMES = 10
RGBD_MAX_FRAMES_DEFAULT = 0
DINO_SEEN_CONF_THRESH = 0.45
ENABLE_BIN_PHYSICAL_GATING = True
ENABLE_SPILLAGE_FLOOR_GATING = True
SPILLAGE_MAX_FLOOR_CLEARANCE_M = 0.10

# Class-specific confidence thresholds for native YOLO Ensemble to minimize false positives
YOLO_CLASS_THRESHOLDS = {
    "door": 0.30,
    "exit_sign": 0.35,
    "bin": 0.30,
}

# ── Core Filter List ───────────────────────────────────────────────────────────
CLASSES_TO_IGNORE = {
    "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "stop sign", "parking meter", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "tie", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "skateboard", "surfboard",
    "tennis racket", "wine glass", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "tv", "laptop", "mouse",
    "remote", "keyboard", "microwave", "oven", "toaster", "vase", "teddy bear",
    "hair drier", "toothbrush", "umbrella", "bowl", "potted plant", "cell phone", 
    "book", "clock", "refridgerator", "refrigerator"
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

CLASS_COLORS = {}

# Map detector output names to YAML object keys used by depth-size filtering.
# Generic classes map to multiple candidates and pass if any candidate fits.
CLASS_NAME_ALIAS_CANDIDATES = {
    "wheelchair": ["wheelchair_manual", "wheelchair_powered"],
    "security_camera": ["security_camera_dome", "security_camera_bullet"],
    "bin": ["small_bin", "large_bin"],
    "general_bin": ["small_bin", "large_bin"],
    "yellow_bin": ["small_bin", "large_bin"],
    "bin_tiger_stripe": ["small_bin", "large_bin"],
}


def _depth_to_colormap(depth_mm, max_mm=5000):
    clipped = np.clip(depth_mm, 0, max_mm).astype(np.float32)
    norm = (clipped / float(max_mm) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_JET)


def apply_depth_size_filter(predictions, depth_mm, intrinsics):
    return apply_rgbd_depth_size_filter(
        predictions,
        depth_mm,
        intrinsics,
        dimensions_config_path=DIMENSIONS_CONFIG_PATH,
        class_name_alias_candidates=CLASS_NAME_ALIAS_CANDIDATES,
        enable_bin_physical_gating=ENABLE_BIN_PHYSICAL_GATING,
        enable_spillage_floor_gate=ENABLE_SPILLAGE_FLOOR_GATING,
        spillage_floor_clearance_m=SPILLAGE_MAX_FLOOR_CLEARANCE_M,
    )

def get_yolo_color(class_name):
    if class_name not in CLASS_COLORS:
        CLASS_COLORS[class_name] = (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255)
        )
    return CLASS_COLORS[class_name]

# ── Ensemble Layer ─────────────────────────────────────────────────────────────
def run_yolo_ensemble(v1, v2, v3, frame):
    """ Runs inference and filters predictions using fine-grained class thresholds """
    all_boxes = []
    
    r1 = v1(frame, conf=DEFAULT_CONF_THRESH, iou=IOU_THRESH, verbose=False)[0]
    r2 = v2(frame, conf=DEFAULT_CONF_THRESH, iou=IOU_THRESH, verbose=False)[0]
    r3 = v3(frame, conf=DEFAULT_CONF_THRESH, iou=IOU_THRESH, verbose=False)[0]

    for model, result in [(v1, r1), (v2, r2), (v3, r3)]:
        if result.boxes is not None:
            for box in result.boxes:
                name = model.names[int(box.cls)]
                
                # Active filter check
                if name in CLASSES_TO_IGNORE:
                    continue
                
                conf = float(box.conf)
                required_conf = YOLO_CLASS_THRESHOLDS.get(name, DEFAULT_CONF_THRESH)
                if conf < required_conf:
                    continue
                    
                xyxy = box.xyxy[0].cpu().tolist()
                all_boxes.append((*xyxy, conf, name))
                
    return all_boxes

 

def apply_global_nms(predictions, iou_thresh=0.45):
    if not predictions:
        return []
        
    boxes = torch.tensor([[p[0], p[1], p[2], p[3]] for p in predictions], dtype=torch.float32)
    scores = torch.tensor([p[4] for p in predictions], dtype=torch.float32)
    base_labels = [p[5].replace("[DINO] ", "") for p in predictions]

    unique_labels = list(set(base_labels))
    label_to_id = {lbl: i for i, lbl in enumerate(unique_labels)}

    offsets = torch.tensor(
        [label_to_id[p[5].replace("[DINO] ", "")] * 4096.0 for p in predictions],
        dtype=torch.float32
    )
    
    offset_boxes = boxes + offsets.unsqueeze(1)
    kept_indices = tv_nms(offset_boxes, scores, iou_thresh).tolist()
    
    return [predictions[i] for i in kept_indices]

# ── Render Utilities ───────────────────────────────────────────────────────────
def draw_predictions(frame, final_detections):
    for x1, y1, x2, y2, conf, name in final_detections:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        
        box_color = get_yolo_color(name)
        text_color = (255, 255, 255)

        label = f"{name} {conf:.1%}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        label_h = th + baseline + 8
        label_w = tw + 10

        # Keep label fully visible: prefer above bbox, otherwise place below it.
        if y1 - label_h >= 0:
            bg_top = y1 - label_h
            bg_bottom = y1
            text_y = y1 - baseline - 4
        else:
            bg_top = y1
            bg_bottom = y1 + label_h
            text_y = y1 + th + 2

        frame_h, frame_w = frame.shape[:2]
        bg_left = max(0, min(x1, frame_w - 1))
        bg_right = min(frame_w - 1, bg_left + label_w)
        bg_top = max(0, min(bg_top, frame_h - 1))
        bg_bottom = max(0, min(bg_bottom, frame_h - 1))

        cv2.rectangle(frame, (bg_left, bg_top), (bg_right, bg_bottom), box_color, -1)
        cv2.putText(
            frame,
            label,
            (bg_left + 5, max(0, min(text_y, frame_h - 1))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            text_color,
            2,
            cv2.LINE_AA,
        )

    return frame

# ── Processing Pipelines ───────────────────────────────────────────────────────
def process_image(v1, v2, v3, img_path):
    frame = cv2.imread(str(img_path))
    pil_img = Image.open(img_path).convert("RGB")
    
    preds = run_yolo_ensemble(v1, v2, v3, frame)
    
    seen_classes = {p[5].replace("[DINO] ", "") for p in preds if p[4] >= DINO_SEEN_CONF_THRESH}
    missing_targets = [c for c in DINO_FALLBACK.keys() if c not in seen_classes]
    
    dino_preds = run_dino_fallback(pil_img, missing_targets)
    preds.extend(dino_preds)
    
    final_dets = apply_global_nms(preds, IOU_THRESH)
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    final_dets = apply_common_sense_rules(final_dets, frame_height, frame_width)
    final_dets = refine_bin_detections(frame, final_dets)
    annotated = draw_predictions(frame, final_dets)
    
    out_path = OUT_DIR / f"out_{datetime.now().strftime('%M%S')}_{img_path.name}"
    cv2.imwrite(str(out_path), annotated)
    
    print("\n" + "═"*70)
    print(f"[SUCCESS] Image artifact successfully generated on GPU path!")
    print(f" -> Saved to: {out_path.resolve()}")
    print("═"*70 + "\n")

    cv2.namedWindow("Inference Window (Resizable)", cv2.WINDOW_NORMAL)
    cv2.imshow("Inference Window (Resizable)", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(v1, v2, v3, vid_path):
    cap = cv2.VideoCapture(str(vid_path))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    
    out_path = OUT_DIR / f"out_{datetime.now().strftime('%H%M%S')}_{vid_path.stem}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    
    cv2.namedWindow("Inference Window (Resizable)", cv2.WINDOW_NORMAL)
    frame_idx = 0
    
    held_dino_detections = []
    held_dino_age = DINO_HOLD_FRAMES + 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        
        preds = run_yolo_ensemble(v1, v2, v3, frame)
        
        run_dino_now = (
            frame_idx > 1 and
            ((frame_idx - 1) % DINO_VIDEO_INTERVAL_FRAMES == 0)
        )

        if run_dino_now:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            seen_classes = {p[5].replace("[DINO] ", "") for p in preds if p[4] >= DINO_SEEN_CONF_THRESH}
            missing_targets = [c for c in DINO_FALLBACK.keys() if c not in seen_classes]

            dino_preds = run_dino_fallback(pil_img, missing_targets)
            held_dino_detections = dino_preds
            held_dino_age = 0
        else:
            held_dino_age += 1

        if held_dino_age <= DINO_HOLD_FRAMES:
            preds.extend(held_dino_detections) 
                
        final_dets = apply_global_nms(preds, IOU_THRESH)
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
        final_dets = apply_common_sense_rules(final_dets, frame_height, frame_width)
        final_dets = refine_bin_detections(frame, final_dets)
        annotated = draw_predictions(frame, final_dets)
            
        writer.write(annotated)
        cv2.imshow("Inference Window (Resizable)", annotated)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[INFO] Video processing interrupted early by user input key.")
            break
            
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    
    print("\n" + "═"*70)
    print(f"[SUCCESS] Video rendering complete!")
    print(f" -> Processed: {frame_idx} frames total.")
    print(f" -> Saved to: {out_path.resolve()}")
    print("═"*70 + "\n")


def _download(url):
    suffix = Path(url.split("?")[0]).suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    r.raise_for_status()
    tmp.write(r.content)
    tmp.close()
    return Path(tmp.name)

def main():
    parser = argparse.ArgumentParser(description="YOLO ensemble + DINO fallback (image/video/url + RGBD ROS2 bag)")
    parser.add_argument("--bag", type=str, default="", help="Path to RGBD bag directory or a file inside bag directory")
    parser.add_argument("--max-frames", type=int, default=RGBD_MAX_FRAMES_DEFAULT, help="RGBD mode: max synced frames to process (0 = all)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CRITICAL: CUDA execution environment not detected.")
        
    print("Initializing Core Models on CUDA GPU Devices...")
    v1 = YOLO(str(V1_PATH)).to("cuda")
    v2 = YOLO(str(V2_PATH)).to("cuda")
    v3 = YOLO(str(V3_PATH)).to("cuda")
    print("YOLO V1, V2, and V3 successfully registered in GPU device memory buffers.")

    if args.bag:
        process_rgbd_bag(
            v1,
            v2,
            v3,
            args.bag,
            args.max_frames,
            project_root=PROJECT_ROOT,
            out_dir=OUT_DIR,
            dino_video_interval_frames=DINO_VIDEO_INTERVAL_FRAMES,
            dino_hold_frames=DINO_HOLD_FRAMES,
            dino_seen_conf_thresh=DINO_SEEN_CONF_THRESH,
            iou_thresh=IOU_THRESH,
            dino_fallback=DINO_FALLBACK,
            run_yolo_ensemble=run_yolo_ensemble,
            run_dino_fallback=run_dino_fallback,
            apply_global_nms=apply_global_nms,
            apply_common_sense_rules=apply_common_sense_rules,
            apply_depth_size_filter=apply_depth_size_filter,
            draw_predictions=draw_predictions,
        )
        return

    while True:
        url_input = input("\nEnter Local File Path / URL String (or 'quit'): ").strip().strip('"' + "'")
        if url_input.lower() in ["quit", "q", "exit"]:
            break
        if not url_input:
            continue
            
        is_temp = False
        if Path(url_input).exists():
            media_path = Path(url_input)
        else:
            print("Processing download target parsing pipeline stream...")
            try:
                media_path = _download(url_input)
                is_temp = True
            except Exception as e:
                print(f"  [ERROR] Invalid local path or broken URL downpour: {e}")
                continue
            
        suffix = media_path.suffix.lower()
        if suffix in VIDEO_EXTS:
            process_video(v1, v2, v3, media_path)
        else:
            process_image(v1, v2, v3, media_path)
            
        if is_temp and media_path.exists():
            os.unlink(media_path)

if __name__ == "__main__":
    main()