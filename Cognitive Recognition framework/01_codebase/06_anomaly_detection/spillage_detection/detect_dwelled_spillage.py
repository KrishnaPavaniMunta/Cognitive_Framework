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
DINO_HOLD_IOU_THRESH = 0.12
DINO_HOLD_MIN_CONF = 0.08
DINO_MAX_HOLD_BOXES = 3
DINO_TO_YOLO_SUPPRESS_IOU = 0.50

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
    "spillage": ("water on floor. splash. water spill. transparent spill splash.", 0.10),
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

_dino_processor = None
_dino_model = None
CLASS_COLORS = {}

# Stateless Global Continuous Variables (Replaces Trackers entirely)
_spillage_present_last_frame = False

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

def compute_iou_xyxy(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area

def suppress_dino_with_yolo(predictions, iou_thresh=DINO_TO_YOLO_SUPPRESS_IOU):
    if not predictions:
        return []

    yolo_preds = [p for p in predictions if not p[5].startswith("[DINO] ")]
    dino_preds = [p for p in predictions if p[5].startswith("[DINO] ")]
    if not yolo_preds or not dino_preds:
        return predictions

    kept_dino = []
    for d in dino_preds:
        d_cls = d[5].replace("[DINO] ", "")
        d_box = d[:4]

        suppress = False
        for y in yolo_preds:
            y_cls = y[5].replace("[DINO] ", "")
            if y_cls != d_cls:
                continue
            if compute_iou_xyxy(d_box, y[:4]) >= iou_thresh and y[4] >= d[4]:
                suppress = True
                break

        if not suppress:
            kept_dino.append(d)

    return yolo_preds + kept_dino

def build_held_dino_predictions(held_dino_detections, held_dino_age, yolo_preds):
    if not held_dino_detections or held_dino_age > DINO_HOLD_FRAMES:
        return []

    decay = max(0.0, 1.0 - (held_dino_age / max(float(DINO_HOLD_FRAMES), 1.0)))
    yolo_boxes = [p[:4] for p in yolo_preds]

    hold_preds = []
    for x1, y1, x2, y2, conf, name in held_dino_detections:
        decayed_conf = conf * decay
        if decayed_conf < DINO_HOLD_MIN_CONF:
            continue

        if yolo_boxes:
            max_iou = max(compute_iou_xyxy((x1, y1, x2, y2), y_box) for y_box in yolo_boxes)
            if max_iou < DINO_HOLD_IOU_THRESH:
                continue

        hold_preds.append((x1, y1, x2, y2, decayed_conf, name))

    hold_preds.sort(key=lambda p: p[4], reverse=True)
    return hold_preds[:DINO_MAX_HOLD_BOXES]

def apply_global_nms(predictions, iou_thresh=0.45):
    if not predictions:
        return []

    predictions = suppress_dino_with_yolo(predictions)
        
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
def _process_instant_alert_logic(final_detections):
    """Raise alert immediately when spillage appears in frame (no timer/dwell)."""
    global _spillage_present_last_frame

    spillage_present = any("spillage" in d[5].lower() for d in final_detections)

    if spillage_present and not _spillage_present_last_frame:
        print(f"\n[ALERT SYSTEM] !!! CRITICAL ENVIRONMENTAL HAZARD !!!")
        print(" -> Spillage detected in frame. Immediate alert raised.")
        print(f" -> System Timestamp: {datetime.now().strftime('%H:%M:%S')} | Action Logged.\n")

    _spillage_present_last_frame = spillage_present

# ── Render Utilities ───────────────────────────────────────────────────────────
def draw_predictions(frame, final_detections, fps=25.0):
    for x1, y1, x2, y2, conf, name in final_detections:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        
        box_color = get_yolo_color(name)
        text_color = (255, 255, 255)
        
        clean_name = name.replace("[DINO] ", "")
        label = f"{clean_name} {conf:.1%}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
        cv2.putText(frame, "CRITICAL: SPILLAGE DETECTED", (x1, max(y1 - 30, 20)),
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
    seen_classes = {p[5].replace("[DINO] ", "") for p in preds}
    missing_targets = [c for c in DINO_FALLBACK.keys() if c not in seen_classes]
    
    dino_preds = run_dino_fallback(pil_img, missing_targets)
    preds.extend(dino_preds)
    
    final_dets = apply_global_nms(preds, IOU_THRESH)
    annotated = draw_predictions(frame, final_dets, fps=1.0)
    
    out_path = OUT_DIR / f"out_{datetime.now().strftime('%M%S')}_{img_path.name}"
    cv2.imwrite(str(out_path), annotated)
    print(f"[SUCCESS] Image parsed cleanly: {out_path.resolve()}")

def process_video(v1, v2, v3, vid_path):
    global _spillage_present_last_frame
    cap = cv2.VideoCapture(str(vid_path))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    
    out_path = OUT_DIR / f"out_{datetime.now().strftime('%H%M%S')}_{vid_path.stem}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    
    cv2.namedWindow("Inference Window (Resizable)", cv2.WINDOW_NORMAL)
    frame_idx = 0
    
    # Flush global state variables before beginning stream execution
    _spillage_present_last_frame = False
    
    held_dino_detections = []
    held_dino_age = DINO_HOLD_FRAMES + 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        
        preds = run_yolo_ensemble(v1, v2, v3, frame)
        yolo_preds = list(preds)
        
        run_dino_now = (
            frame_idx == 1 or
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

        preds.extend(build_held_dino_predictions(held_dino_detections, held_dino_age, yolo_preds))
                
        final_dets = apply_global_nms(preds, IOU_THRESH)
        
        # Immediate alert on any detected spillage (no dwell timer)
        _process_instant_alert_logic(final_dets)
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