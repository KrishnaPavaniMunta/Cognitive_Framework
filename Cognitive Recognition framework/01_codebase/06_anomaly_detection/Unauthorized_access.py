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

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()
V1_PATH = Path(r"D:\Object Detection Model\yolo_tr\yolo_tr\Cognitive Recognition framework\03_models_and_weights\models\yolo_trained_v1.pt")
OUT_DIR = Path(r"D:\Object Detection Model\yolo_tr\yolo_tr\Cognitive Recognition framework\04_outputs_runs_and_logs\OD_Outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────
DEFAULT_CONF_THRESH = 0.30
IOU_THRESH = 0.40
DINO_VIDEO_INTERVAL_FRAMES = 15  # Re-verify roles via DINO every 15 frames

TARGET_CLASSES = {"doctor", "healthcare_worker", "patient", "person"}

# DINO only searches for these two specific roles within the cropped YOLO box.
DINO_MODEL_ID = "IDEA-Research/grounding-dino-base"
DINO_FALLBACK = {
    "doctor": ("person wearing white or blue coat. stethoscope on person. white or blue medical attire.", 0.30),
    "healthcare_worker": ("healthcare worker wearing blue shade scrubs. blue surgical scrubs only.", 0.30),
    "patient": ("hospital patient gown. person in hospital bed. person in stretcher.", 0.30),
    "person": ("person wearing non medical clothes. casual clothes. not wearing blue scrubs. not wearing white medical coat.", 0.30),
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

_dino_processor = None
_dino_model = None
_unauthorized_present_last_frame = False

ROLE_COLORS = {
    "doctor": (0, 255, 0),             # Green (Authorized)
    "healthcare_worker": (255, 191, 0), # Cyan/Teal (Authorized)
    "patient": (255, 0, 128),          # Pink/Purple (Patient)
    "person": (0, 0, 255)              # Pure Red (Unauthorized Civilian)
}

def get_role_color(class_name):
    # Strip the model tags before assigning colors
    clean_name = (
        class_name.lower()
        .replace("[yolo] ", "")
        .replace("[dino] ", "")
        .replace("[tracker] ", "")
        .replace("[yolo-fallback] ", "")
    )
    return ROLE_COLORS.get(clean_name, (200, 200, 200))

def _load_dino():
    global _dino_processor, _dino_model
    if _dino_model is None:
        print(f"  [DINO] Loading {DINO_MODEL_ID} for internal crop verification...")
        _dino_processor = AutoProcessor.from_pretrained(DINO_MODEL_ID)
        _dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL_ID).to("cuda")
        _dino_model.eval()

# ── Ensemble Layer ─────────────────────────────────────────────────────────────
def run_yolo_ensemble(v1, frame):
    all_boxes = []
    
    r1 = v1(frame, conf=DEFAULT_CONF_THRESH, iou=IOU_THRESH, verbose=False)[0]

    for model, result in [(v1, r1)]:
        if result.boxes is not None:
            for box in result.boxes:
                name = model.names[int(box.cls)].lower()
                
                
                if name not in TARGET_CLASSES:
                    continue
                
                conf = float(box.conf)
                xyxy = box.xyxy[0].cpu().tolist()
                all_boxes.append((*xyxy, conf, name))
                    
    return all_boxes


def compute_iou_xyxy(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)

    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    
    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

# ── Dynamic Crop Verification (DINO) ───────────────────────────────────────────
def check_crop_with_dino(frame, box, target_classes):
    """Runs DINO on the cropped bounding box and returns the strongest role match."""
    _load_dino()
    x1, y1, x2, y2 = map(int, box)
    h, w = frame.shape[:2]
    
    # Pad the crop by 5% to ensure edges of lab coats or stethoscopes aren't cut off
    pad_x = int((x2 - x1) * 0.05)
    pad_y = int((y2 - y1) * 0.05)
    x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
    
    if x2 - x1 < 20 or y2 - y1 < 20:
        # Crop too small to accurately judge. Return all 3 values so callers
        # can always unpack safely (this used to return a bare None and crash
        # with "cannot unpack non-iterable NoneType" at the call site).
        return None, 0.0, {cls: 0.0 for cls in target_classes}
        
    crop = frame[y1:y2, x1:x2]
    pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    
    best_margin = float("-inf")
    best_conf = 0.0
    best_label = None
    class_scores = {cls: 0.0 for cls in target_classes}
    
    for cls in ("doctor", "healthcare_worker", "patient", "person"):
        if cls not in target_classes or cls not in DINO_FALLBACK:
            continue
        prompt, target_thresh = DINO_FALLBACK[cls]
        inputs = _dino_processor(images=pil_img, text=prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = _dino_model(**inputs)
            
        results = _dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=target_thresh,
            text_threshold=target_thresh,  # was hardcoded 0.25, looser than patient's old 0.24 bar
            target_sizes=[pil_img.size[::-1]]
        )[0]
        
        scores = results["scores"].cpu().numpy()
        if len(scores) > 0:
            max_score = float(max(scores))
            class_scores[cls] = max_score
            # Compare MARGIN above each class's own threshold, not raw score,
            # so classes with different thresholds/prompt quality aren't
            # compared on an uneven footing.
            margin = max_score - target_thresh
            if margin > best_margin:
                best_margin = margin
                best_conf = max_score
                best_label = cls
                
    if best_label is not None and best_margin >= 0:
        return best_label, best_conf, class_scores
    # Nothing cleared its own bar -> default to "person" (unauthorized),
    # rather than defaulting to None and letting the caller fall back to
    # whatever YOLO happened to guess.
    if "person" in target_classes:
        return "person", 0.0, class_scores
    return None, 0.0, class_scores

# ── Role Tracking System ───────────────────────────────────────────────────────
class RoleTracker:
    def __init__(self):
        self.tracked_roles = [] 

    def process_detections(self, frame, yolo_preds, frame_idx):
        final_preds = []
        new_tracked_roles = []
        dino_roles = ["doctor", "healthcare_worker", "patient", "person"]
        
        # Verify specific roles instantly on frame 1, or periodically to catch changes
        refresh_all = (frame_idx == 1 or frame_idx % DINO_VIDEO_INTERVAL_FRAMES == 0)
        
        if yolo_preds:
            print(f"\n--- Frame {frame_idx} Analysis ---")
            
        for y_box in yolo_preds:
            xyxy, conf, label = y_box[:4], y_box[4], y_box[5]

            matched_role = None
            best_iou = 0.0
            best_idx = -1
            
            # Check if we already verified this person's role in a previous frame
            for idx, r in enumerate(self.tracked_roles):
                iou = compute_iou_xyxy(xyxy, r['box'])
                if iou > 0.50 and iou > best_iou:
                    best_iou = iou
                    matched_role = r
                    best_idx = idx

            # Remove the matched role from the pool so a second YOLO box in
            # this same frame can't also claim it (previously two boxes
            # could both inherit the same tracked identity).
            if best_idx >= 0:
                del self.tracked_roles[best_idx]

            # Run the heavy DINO model if this is a new person, or if it's time to refresh,
            # or if YOLO now strongly disagrees with what we're tracking (see below).
            run_dino = refresh_all or (matched_role is None)
            if (
                matched_role is not None
                and not run_dino
                and label == "person"
                and matched_role['label'] != "person"
            ):
                # Stale memory guard: if YOLO now sees plain "person" but we're
                # still tracking e.g. "patient" from an old DINO call, don't
                # blindly trust the stale label for up to 15 more frames —
                # force an immediate re-check instead.
                run_dino = True
            
            if run_dino:
                d_label, d_conf, d_scores = check_crop_with_dino(frame, xyxy, dino_roles)
                if d_label is not None:
                    print(
                        f"  [DINO] Cross-Check: YOLO={label.upper()} -> "
                        f"DINO={d_label.upper()} (YOLO: {conf:.2f}, DINO: {d_conf:.2f})"
                    )
                    print(
                        "         Scores: "
                        + ", ".join(f"{k}:{v:.2f}" for k, v in d_scores.items())
                    )
                    matched_role = {
                        'box': xyxy,
                        'label': d_label,
                        'conf': max(conf, d_conf),
                        'source': '[DINO]'
                    }
                else:
                    print(
                        f"  [DINO] Cross-Check: No role match for YOLO={label.upper()} "
                        f"(YOLO: {conf:.2f}). Falling back to YOLO label."
                    )
                    matched_role = {
                        'box': xyxy,
                        'label': label,
                        'conf': conf,
                        'source': '[YOLO-FALLBACK]'
                    }
            else:
                print(f"  [TRACKER] Memory: Maintained {matched_role['label'].upper()} from previous frames")
                matched_role['box'] = xyxy
                matched_role['source'] = '[TRACKER]'
            
            if matched_role:
                new_tracked_roles.append({
                    'box': matched_role['box'], 
                    'label': matched_role['label'], 
                    'conf': matched_role['conf'],
                    'source': matched_role['source']
                })
                # Append with the model source prefixed to the name for rendering
                final_preds.append((*xyxy, matched_role['conf'], f"{matched_role['source']} {matched_role['label']}"))
                
        self.tracked_roles = new_tracked_roles
        return final_preds

# ── Access Security Control Logic ──────────────────────────────────────────────
def _process_unauthorized_access_alerts(final_detections):
    global _unauthorized_present_last_frame
    unauthorized_present = any("person" in d[5].lower() for d in final_detections)

    if unauthorized_present and not _unauthorized_present_last_frame:
        print(f"\n[SECURITY ALERT] !!! UNAUTHORIZED AREA ACCESS DETECTED !!!")
        print(" -> Unverified civilian 'Person' identified in restricted medical zone.")
        print(f" -> System Timestamp: {datetime.now().strftime('%H:%M:%S')} | Verification Requested.\n")

    _unauthorized_present_last_frame = unauthorized_present

# ── Rendering Pipeline ─────────────────────────────────────────────────────────
def draw_predictions(frame, final_detections, fps=25.0):
    for x1, y1, x2, y2, conf, name in final_detections:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        
        box_color = get_role_color(name)
        clean_name = name.upper()
        
        if "PERSON" in clean_name:
            label = f"{clean_name} (UNAUTHORIZED): {conf:.1%}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
        else:
            label = f"{clean_name}: {conf:.1%}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), box_color, -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    return frame

# ── Processing Execution Loops ──────────────────────────────────────────────────
def process_image(v1, img_path):
    frame = cv2.imread(str(img_path))
    
    preds = run_yolo_ensemble(v1, frame)
    
    tracker = RoleTracker()
    final_dets = tracker.process_detections(frame, preds, frame_idx=1)
    _process_unauthorized_access_alerts(final_dets)  # was missing in image mode
    
    annotated = draw_predictions(frame, final_dets, fps=1.0)
    
    out_path = OUT_DIR / f"security_out_{datetime.now().strftime('%M%S')}_{img_path.name}"
    cv2.imwrite(str(out_path), annotated)
    print(f"[SUCCESS] Security image framework executed: {out_path.resolve()}")

def process_video(v1, vid_path):
    global _unauthorized_present_last_frame
    cap = cv2.VideoCapture(str(vid_path))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    
    out_path = OUT_DIR / f"security_out_{datetime.now().strftime('%H%M%S')}_{vid_path.stem}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    
    cv2.namedWindow("Access Control Stream (Resizable)", cv2.WINDOW_NORMAL)
    frame_idx = 0
    _unauthorized_present_last_frame = False
    
    tracker = RoleTracker()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        
        # 1. Broad Net: Find Humans
        preds = run_yolo_ensemble(v1, frame)
        
        # 2. Targeted Verification: Crop & Track specific identities
        final_dets = tracker.process_detections(frame, preds, frame_idx)
        
        # 3. Security Checks & Rendering
        _process_unauthorized_access_alerts(final_dets)
        annotated = draw_predictions(frame, final_dets, fps=fps)
            
        writer.write(annotated)
        cv2.imshow("Access Control Stream (Resizable)", annotated)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"[SUCCESS] Video parsing stream finalized: {out_path.resolve()}")

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
        
    print("Initializing Core Models on CUDA (CROP-AND-TRACK MEDICAL ACCESS FRAMEWORK)...")
    v1 = YOLO(str(V1_PATH)).to("cuda")

    while True:
        url_input = input("\nEnter Media Path / URL String (or 'quit'): ").strip().strip('"' + "'")
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
                print(f"  [ERROR] Input read failed: {e}")
                continue
            
        suffix = media_path.suffix.lower()
        if suffix in VIDEO_EXTS:
            process_video(v1, media_path)
        else:
            process_image(v1, media_path)
            
        if is_temp and media_path.exists():
            os.unlink(media_path)

if __name__ == "__main__":
    main()