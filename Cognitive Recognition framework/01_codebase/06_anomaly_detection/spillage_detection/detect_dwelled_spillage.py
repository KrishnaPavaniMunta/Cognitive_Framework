import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import tempfile
import requests
from pathlib import Path
from datetime import datetime
import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from torchvision.ops import nms as tv_nms
import random

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()
REPO_ROOT = BASE_DIR.parents[2] if len(BASE_DIR.parents) >= 3 else BASE_DIR

V1_PATH = Path(r"D:\Object Detection Model\yolo_tr\yolo_tr\Cognitive Recognition framework\03_models_and_weights\models\yolo_trained_v1.pt")
V2_PATH = Path(r"D:\Object Detection Model\yolo_tr\yolo_tr\Cognitive Recognition framework\03_models_and_weights\models\yolo_trained_v2.pt")
V3_PATH = Path(r"D:\Object Detection Model\yolo_tr\yolo_tr\Cognitive Recognition framework\03_models_and_weights\models\yolo_trained_v3.pt")
OUT_DIR = Path(r"D:\Object Detection Model\yolo_tr\yolo_tr\Cognitive Recognition framework\04_outputs_runs_and_logs\OD_Outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────
DEFAULT_CONF_THRESH = 0.25
IOU_THRESH = 0.45
DINO_VIDEO_INTERVAL_FRAMES = 15
DINO_HOLD_FRAMES = 10

# ── Spillage Dwell Configuration ───────────────────────────────────────────────
SPILLAGE_DWELL_THRESHOLD_SEC = 5.0

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
    "book", "clock", "refridgerator", "refrigerator", "door", "exit_sign", "bin",
    "surgical_scissor", "surgical_light", "glove", "mask", "hair_net", "radiator",
    "iv_stand", "medical_tray", "infusion_pump", "hand_sanitizer", "hazmat_sign"
}

# ── Grounding DINO Tuning Prompt Map ───────────────────────────────────────────
DINO_MODEL_ID = "IDEA-Research/grounding-dino-base"

# Enhanced language query optimized explicitly for specular reflections of clear liquids
DINO_FALLBACK = {
    "spillage": ("clear liquid puddle on floor reflection. splash. water spill. water spill. transparent spill splash.", 0.28),
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

_dino_processor = None
_dino_model = None
CLASS_COLORS = {}

# Stateless Global Continuous Variables (Replaces Trackers entirely)
_spillage_detected_frames_run = 0
_spillage_alert_triggered = False

def get_yolo_color(class_name):
    return (0, 0, 255)  # Pure Red for spillage anomalies

def _load_dino():
    global _dino_processor, _dino_model
    if _dino_model is None:
        print(f"  [DINO] Loading {DINO_MODEL_ID} directly on CUDA GPU...")
        _dino_processor = AutoProcessor.from_pretrained(DINO_MODEL_ID)
        _dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL_ID).to("cuda")
        _dino_model.eval()

# ── Ensemble Layer ─────────────────────────────────────────────────────────────
def run_yolo_ensemble(v1, v2, v3, frame):
    all_boxes = []
    
    r1 = v1(frame, conf=DEFAULT_CONF_THRESH, iou=IOU_THRESH, verbose=False)[0]
    r2 = v2(frame, conf=DEFAULT_CONF_THRESH, iou=IOU_THRESH, verbose=False)[0]
    r3 = v3(frame, conf=DEFAULT_CONF_THRESH, iou=IOU_THRESH, verbose=False)[0]

    for model, result in [(v1, r1), (v2, r2), (v3, r3)]:
        if result.boxes is not None:
            for box in result.boxes:
                name = model.names[int(box.cls)]
                
                if name in CLASSES_TO_IGNORE or name != "spillage":
                    continue
                
                conf = float(box.conf)
                xyxy = box.xyxy[0].cpu().tolist()
                all_boxes.append((*xyxy, conf, name))
                    
    return all_boxes

def run_dino_fallback(pil_image, target_classes):
    _load_dino()

    if not target_classes:
        return []

    dino_boxes = []

    for cls in target_classes:
        if cls not in DINO_FALLBACK:
            continue

        prompt, target_thresh = DINO_FALLBACK[cls]

        inputs = _dino_processor(
            images=pil_image,
            text=prompt,
            return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            outputs = _dino_model(**inputs)

        results = _dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=target_thresh,
            text_threshold=0.25,
            target_sizes=[pil_image.size[::-1]]
        )[0]

        for box, score in zip(
            results["boxes"].cpu().numpy(),
            results["scores"].cpu().numpy()
        ):
            dino_boxes.append(
                (*box.tolist(), float(score), f"[DINO] {cls}")
            )

    return dino_boxes

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

# ── Stateless Alert Logic ───────────────────────────────────────────────────────
def _process_stateless_timer_logic(final_detections, fps):
    """ Increments a single global counter frame-by-frame if any spillage is present """
    global _spillage_detected_frames_run, _spillage_alert_triggered
    
    spillage_present = any("spillage" in d[5].lower() for d in final_detections)
    
    if spillage_present:
        _spillage_detected_frames_run += 1
        current_dwell_time = _spillage_detected_frames_run / max(fps, 1.0)
        
        if current_dwell_time >= SPILLAGE_DWELL_THRESHOLD_SEC and not _spillage_alert_triggered:
            print(f"\n[ALERT SYSTEM] !!! CRITICAL ENVIRONMENTAL HAZARD !!!")
            print(f" -> Spillage presence has remaned visible for over {current_dwell_time:.2f} seconds continuously.")
            print(f" -> System Timestamp: {datetime.now().strftime('%H:%M:%S')} | Action Logged.\n")
            _spillage_alert_triggered = True
    else:
        # Puddle gone or moved out of frame — reset counter blocks immediately
        _spillage_detected_frames_run = 0
        _spillage_alert_triggered = False

# ── Render Utilities ───────────────────────────────────────────────────────────
def draw_predictions(frame, final_detections, fps=25.0):
    global _spillage_detected_frames_run
    
    for x1, y1, x2, y2, conf, name in final_detections:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        
        box_color = get_yolo_color(name)
        text_color = (255, 255, 255)
        
        clean_name = name.replace("[DINO] ", "")
        current_dwell = _spillage_detected_frames_run / max(fps, 1.0)
        label = f"{clean_name} ({current_dwell:.1f}s) {conf:.1%}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        
        # Thicken box and flash warning string across matrix if threshold breached
        if current_dwell >= SPILLAGE_DWELL_THRESHOLD_SEC:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
            cv2.putText(frame, "CRITICAL: DWELLED LIQUID HAZARD", (x1, max(y1 - 30, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), box_color, -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2, cv2.LINE_AA)

    return frame

# ── Processing Pipelines ───────────────────────────────────────────────────────
def process_image(v1, v2, v3, img_path):
    frame = cv2.imread(str(img_path))
    pil_img = Image.open(img_path).convert("RGB")
    
    preds = run_yolo_ensemble(v1, v2, v3, frame)
    seen_classes = set([p[4] for p in preds])
    missing_targets = [c for c in DINO_FALLBACK.keys() if c not in seen_classes]
    
    dino_preds = run_dino_fallback(pil_img, missing_targets)
    preds.extend(dino_preds)
    
    final_dets = apply_global_nms(preds, IOU_THRESH)
    annotated = draw_predictions(frame, final_dets, fps=1.0)
    
    out_path = OUT_DIR / f"out_{datetime.now().strftime('%M%S')}_{img_path.name}"
    cv2.imwrite(str(out_path), annotated)
    print(f"[SUCCESS] Image parsed cleanly: {out_path.resolve()}")

def process_video(v1, v2, v3, vid_path):
    global _spillage_detected_frames_run, _spillage_alert_triggered
    cap = cv2.VideoCapture(str(vid_path))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    
    out_path = OUT_DIR / f"out_{datetime.now().strftime('%H%M%S')}_{vid_path.stem}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    
    cv2.namedWindow("Inference Window (Resizable)", cv2.WINDOW_NORMAL)
    frame_idx = 0
    
    # Flush global state variables before beginning stream execution
    _spillage_detected_frames_run = 0
    _spillage_alert_triggered = False
    
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
            seen_classes = {p[5].replace("[DINO] ", "") for p in preds}
            missing_targets = [c for c in DINO_FALLBACK.keys() if c not in seen_classes]

            dino_preds = run_dino_fallback(pil_img, missing_targets)
            held_dino_detections = dino_preds
            held_dino_age = 0
        else:
            held_dino_age += 1

        if held_dino_age <= DINO_HOLD_FRAMES:
            preds.extend(held_dino_detections) 
                
        final_dets = apply_global_nms(preds, IOU_THRESH)
        
        # Calculate stateless continuous duration
        _process_stateless_timer_logic(final_dets, fps)
        annotated = draw_predictions(frame, final_dets, fps=fps)
            
        writer.write(annotated)
        cv2.imshow("Inference Window (Resizable)", annotated)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"[SUCCESS] Video parsed cleanly: {out_path.resolve()}")

def _download(url):
    suffix = Path(url.split("?")[0]).suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    r.raise_for_status()
    tmp.write(r.content)
    tmp.close()
    return Path(tmp.name)

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CRITICAL: CUDA execution environment not detected.")
        
    print("Initializing Core Models on CUDA GPU Devices (STATELESS LIQUID ENGINE)...")
    v1 = YOLO(str(V1_PATH)).to("cuda")
    v2 = YOLO(str(V2_PATH)).to("cuda")
    v3 = YOLO(str(V3_PATH)).to("cuda")
    print("YOLO Models and specialized Grounding DINO settings successfully mapped.")

    while True:
        url_input = input("\nEnter Local Spillage File Path / URL String (or 'quit'): ").strip().strip('"' + "'")
        if url_input.lower() in ["quit", "q", "exit"]:
            break
        if not url_input:
            continue
            
        is_temp = False
        if Path(url_input).exists():
            media_path = Path(url_input)
        else:
            try:
                media_path = _download(url_input)
                is_temp = True
            except Exception as e:
                print(f"  [ERROR] Broken URL: {e}")
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