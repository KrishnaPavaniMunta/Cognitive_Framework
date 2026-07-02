import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import tempfile
import argparse
import importlib.util
import sys
import re
import requests
import yaml
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

# ── Grounding DINO Combined Tuning Map ──────────────────────────────────────────
DINO_MODEL_ID = "IDEA-Research/grounding-dino-base"

# Formatted as: "class_key": ("custom phrase detection prompt", individual_confidence_threshold)
DINO_FALLBACK = {
    "surgical_scissor": ("surgical scissors. stainless steel scissors. metal surgical scissors with pointed blades.", 0.45),
    "surgical_light": ("round surgical operating light. operating room light on articulated arm. large circular surgical light.", 0.40),
    "glove": ("blue surgical glove on hand. purple nitrile glove on hand. white latex medical glove on hand.", 0.45),
    "mask": ("blue surgical face mask. white medical face mask. blue medical face mask worn by person.", 0.35),
    "hair_net": ("surgical hair net on head.blue mesh hair cover on head. disposable bouffant cap.", 0.45),
    "radiator": ("radiator heater panel mounted on wall. ribbed slotted steel heating element.", 0.50),
    "exit_sign": ("green exit sign. illuminated green exit sign with arrow. green rectangular sign mounted above doorway.", 0.45),
    "door": ("hospital corridor door. fire door, emergency exit door.", 0.45),
    "medical_tray": ("steel silver medical tray. flat rectangular plastic medical tray. metal instrument tray.", 0.40),
    "hand_sanitizer": ("wall-mounted soap sanitizer. dispenser. handwash. sink wall dispenser", 0.40),
    "bin": ("hospital clinical waste bin with pedal lid. yellow medical waste bin. hospital trash bin with swing lid or pedal lid. wheeled clinical waste container.", 0.52),
    "hazmat_sign": ("hazardous materials sign. fire Symbol.  kite/triangle shaped yellow warning placard. Danger.", 0.30),
    "utility_trolley": ("trolley with multiple shelves. stand with wheels. rolling cart with shelves and push handle. wheeled medical supply trolley.", 0.42),
    "oxygen_pump": ("upright oxygen concentrator machine. oxygen concentrator tower with front control panel, vents, wheels, and oxygen tubing. hospital oxygen concentrator unit plugged into wall.", 0.58),
    "switch_board": ("wall mounted switch board. electrical panel with power sockets. wall mounted switch plate with sockets.", 0.40),
    "iv_stand": ("hospital iv stand pole with wheeled base and hanging hooks. intravenous drip stand.", 0.48),
    "infusion_pump": ("infusion pump device mounted on iv pole with display and control buttons.", 0.50),
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

_dino_processor = None
_dino_model = None
CLASS_COLORS = {}
_dimensions_config_cache = None

# Map detector output names to YAML object keys used by depth-size filtering.
# Generic classes map to multiple candidates and pass if any candidate fits.
CLASS_NAME_ALIAS_CANDIDATES = {
    "wheelchair": ["wheelchair_manual", "wheelchair_powered"],
    "manual_wheelchair": ["wheelchair_manual"],
    "powered_wheelchair": ["wheelchair_powered"],
    "electric_wheelchair": ["wheelchair_powered"],
    "security_camera": ["security_camera_dome", "security_camera_bullet"],
    "camera_dome": ["security_camera_dome"],
    "dome_camera": ["security_camera_dome"],
    "camera_bullet": ["security_camera_bullet"],
    "bullet_camera": ["security_camera_bullet"],
}


def _load_dimensions_config():
    global _dimensions_config_cache
    if _dimensions_config_cache is not None:
        return _dimensions_config_cache

    if not DIMENSIONS_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Dimensions config not found: {DIMENSIONS_CONFIG_PATH}")

    with DIMENSIONS_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    objects = raw_config.get("objects")
    if not isinstance(objects, dict):
        raise ValueError(f"Invalid dimensions config format in {DIMENSIONS_CONFIG_PATH}: missing 'objects' mapping")

    parsed = {}
    for class_name, spec in objects.items():
        if not isinstance(spec, dict):
            continue
        range_spec = spec.get("range") or {}
        width_range = range_spec.get("width")
        height_range = range_spec.get("height")
        if not width_range or not height_range or len(width_range) != 2 or len(height_range) != 2:
            continue

        parsed[str(class_name)] = {
            "min_w": float(width_range[0]),
            "max_w": float(width_range[1]),
            "min_h": float(height_range[0]),
            "max_h": float(height_range[1]),
        }

    _dimensions_config_cache = parsed
    print(f"[DEPTH FILTER] Loaded physical-size limits for {len(parsed)} classes from {DIMENSIONS_CONFIG_PATH.name}")
    return _dimensions_config_cache


def _load_rgbd_reader():
    reader_path = PROJECT_ROOT / "01_codebase" / "06_anomaly_detection" / "Blocked_exit_detection" / "RGBD_Reader.py"
    if not reader_path.exists():
        raise FileNotFoundError(f"RGBD reader not found: {reader_path}")

    spec = importlib.util.spec_from_file_location("rgbd_reader_module", str(reader_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load RGBD reader module from {reader_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _depth_to_colormap(depth_mm, max_mm=5000):
    clipped = np.clip(depth_mm, 0, max_mm).astype(np.float32)
    norm = (clipped / float(max_mm) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_JET)


def get_oriented_3d_dimensions(depth_mm, x1, y1, x2, y2, intrinsics):
    """
    Converts 2D bounding box depth pixels into a 3D point cloud cluster.
    Uses raw Y-span for vertical height and 2D PCA in XZ for rotation-robust width.
    """
    h_img, w_img = depth_mm.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w_img, int(x2)), min(h_img, int(y2))

    if x2 <= x1 or y2 <= y1:
        return None

    depth_roi = depth_mm[y1:y2, x1:x2].astype(np.float32)
    x_indices, y_indices = np.meshgrid(np.arange(x1, x2), np.arange(y1, y2))

    valid_mask = np.isfinite(depth_roi) & (depth_roi > 100) & (depth_roi < 6000)
    if np.sum(valid_mask) < 20:
        return None

    z_m = depth_roi[valid_mask] / 1000.0
    x_px = x_indices[valid_mask]
    y_px = y_indices[valid_mask]

    X = (x_px - float(intrinsics.cx)) * z_m / float(intrinsics.fx)
    Y = (y_px - float(intrinsics.cy)) * z_m / float(intrinsics.fy)
    Z = z_m

    # Vertical size is directly measured from camera-space Y span.
    extent_height = float(np.max(Y) - np.min(Y))

    median_z = np.median(Z)
    std_z = np.std(Z)
    z_threshold = max(1.5 * std_z, 0.5)
    inlier_mask = np.abs(Z - median_z) < z_threshold

    X_filtered = X[inlier_mask]
    Z_filtered = Z[inlier_mask]

    if len(X_filtered) < 20:
        return None

    # Compute horizontal width using PCA only in floor plane (X, Z).
    pts_2d = np.column_stack((X_filtered, Z_filtered))
    pts_centered = pts_2d - np.mean(pts_2d, axis=0)

    cov = np.cov(pts_centered, rowvar=False)
    _, eigenvectors = np.linalg.eigh(cov)

    pts_projected = pts_centered @ eigenvectors
    horizontal_dimensions = np.max(pts_projected, axis=0) - np.min(pts_projected, axis=0)
    extent_width = float(np.max(horizontal_dimensions))

    if extent_width <= 0 or extent_height <= 0:
        return float(np.max(X) - np.min(X)), extent_height

    return extent_width, extent_height


def apply_depth_size_filter(predictions, depth_mm, intrinsics):
    if not predictions or depth_mm is None:
        return predictions

    size_limits = _load_dimensions_config()
    filtered = []
    removed = 0

    for x1, y1, x2, y2, conf, name in predictions:
        base_name = str(name).replace("[DINO] ", "")
        candidate_names = CLASS_NAME_ALIAS_CANDIDATES.get(base_name, [base_name])
        candidate_limits = [
            (candidate, size_limits[candidate])
            for candidate in candidate_names
            if candidate in size_limits
        ]

        if not candidate_limits:
            filtered.append((x1, y1, x2, y2, conf, name))
            continue

        dims_3d = get_oriented_3d_dimensions(depth_mm, x1, y1, x2, y2, intrinsics)
        if dims_3d is None:
            # Keep detection when point cloud extraction fails due to sensor gaps.
            filtered.append((x1, y1, x2, y2, conf, name))
            continue

        width_m, height_m = dims_3d

        fits_any = any(
            (
                width_m >= limits["min_w"] and width_m <= limits["max_w"] and
                height_m >= limits["min_h"] and height_m <= limits["max_h"]
            )
            for _, limits in candidate_limits
        )

        if not fits_any:
            removed += 1
            continue

        filtered.append((x1, y1, x2, y2, conf, name))

    if removed > 0:
        print(f"  [3D POINT CLOUD FILTER] Rotational PCA filter removed {removed} false-positive detections.")
    return filtered

def get_yolo_color(class_name):
    if class_name not in CLASS_COLORS:
        CLASS_COLORS[class_name] = (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255)
        )
    return CLASS_COLORS[class_name]

def _load_dino():
    global _dino_processor, _dino_model
    if _dino_model is None:
        print(f"  [DINO] Loading {DINO_MODEL_ID} directly on CUDA GPU...")
        _dino_processor = AutoProcessor.from_pretrained(DINO_MODEL_ID)
        _dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL_ID).to("cuda")
        _dino_model.eval()


def _normalize_text(s):
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9\s_-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _prompt_anchor(prompt):
    # First sentence is treated as the canonical anchor phrase for matching.
    return _normalize_text(str(prompt).split(".")[0])

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

# ── Explicit Batch Groups ────────────────────────────────────────────────────
# Classes within a group must NOT share overlapping vocabulary in their prompts,
# otherwise label-matching after batched DINO inference becomes ambiguous.
# Pairs with genuine conceptual/vocabulary overlap are kept solo.
DINO_BATCH_GROUPS = [
    ["surgical_light", "radiator", "hand_sanitizer", "hazmat_sign", "switch_board"],  # distinct vocab, safe to batch
    ["glove", "mask", "hair_net"],                                   # PPE, distinct body locations, safe to batch
    ["bin"],                  # standalone, no clean overlap partner left in this set
    ["door"],                 # overlaps with exit_sign vocabulary — solo
    ["exit_sign"],            # overlaps with door vocabulary — solo
    ["surgical_scissor"],     # overlaps with medical_tray ("instrument tray") — solo
    ["medical_tray"],         # overlaps with surgical_scissor — solo
    ["utility_trolley"],      # overlaps with medical_tray/trolley-cart vocabulary — solo
    ["oxygen_pump"],          # overlaps with iv_stand/infusion contexts — solo
    ["iv_stand"],             # overlaps with infusion_pump ("pole"/"IV") — solo
    ["infusion_pump"],        # overlaps with iv_stand — solo
]


def run_dino_fallback(pil_image, target_classes):
    _load_dino()

    if not target_classes:
        return []

    target_set = set(target_classes)
    dino_boxes = []

    for group in DINO_BATCH_GROUPS:
        active = [cls for cls in group if cls in target_set and cls in DINO_FALLBACK]
        if not active:
            continue

        cls_prompt_thresh = [(cls, *DINO_FALLBACK[cls]) for cls in active]
        combined_prompt = " ".join(cpt[1] for cpt in cls_prompt_thresh)
        min_thresh = min(cpt[2] for cpt in cls_prompt_thresh)

        inputs = _dino_processor(
            images=pil_image,
            text=combined_prompt,
            return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            outputs = _dino_model(**inputs)

        results = _dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=min_thresh,
            text_threshold=0.25,
            target_sizes=[pil_image.size[::-1]]
        )[0]

        labels_out = results.get("text_labels", results.get("labels", []))

        for box, score, label in zip(
            results["boxes"].cpu().numpy(),
            results["scores"].cpu().numpy(),
            labels_out
        ):
            label_norm = _normalize_text(label)
            if not label_norm:
                continue

            matched = None
            ambiguous = False
            for cls, prompt, thresh in cls_prompt_thresh:
                first_term = _prompt_anchor(prompt)
                if not first_term:
                    continue

                # Compare only against canonical anchor phrases to avoid cross-class collisions.
                if label_norm == first_term or label_norm in first_term or first_term in label_norm:
                    if matched is not None and matched[0] != cls:
                        ambiguous = True
                        break
                    matched = (cls, thresh)
            if matched is None or ambiguous:
                continue  # ambiguous match — drop rather than mislabel

            cls, own_thresh = matched
            if float(score) < own_thresh:
                continue  # didn't clear this specific class's own threshold

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


def process_rgbd_bag(v1, v2, v3, bag_path, max_frames=RGBD_MAX_FRAMES_DEFAULT):
    rgbd_reader = _load_rgbd_reader()

    bag_input = Path(bag_path)
    if bag_input.is_file():
        bag_source = bag_input
        bag_label = bag_input.stem
        fallback_source = None
    else:
        bag_dir = bag_input
        if not (bag_dir / "metadata.yaml").exists():
            raise FileNotFoundError(f"metadata.yaml not found in bag dir: {bag_dir}")

        fallback_source = None
        recovered_default = bag_dir / f"{bag_dir.name}_recovered.db3"
        if recovered_default.exists():
            bag_source = recovered_default
            fallback_source = bag_dir
            print(f"[RGBD] Using recovered storage file: {recovered_default.name}")
        else:
            recovered_candidates = sorted(bag_dir.glob("*_recovered.db3"))
            if recovered_candidates:
                bag_source = recovered_candidates[0]
                fallback_source = bag_dir
                print(f"[RGBD] Using recovered storage file: {bag_source.name}")
            else:
                bag_source = bag_dir
                if recovered_default.exists():
                    fallback_source = recovered_default

        bag_label = bag_dir.name

    def _open_intrinsics_with_fallback(primary, secondary):
        try:
            return rgbd_reader.read_intrinsics(primary), primary
        except Exception as exc:
            if secondary is None:
                raise
            print(f"[RGBD] Warning: failed reading intrinsics from {Path(primary).name}: {exc}")
            print(f"[RGBD] Retrying with alternate source: {Path(secondary).name}")
            return rgbd_reader.read_intrinsics(secondary), secondary

    intr, active_source = _open_intrinsics_with_fallback(bag_source, fallback_source)
    print(f"[RGBD] Intrinsics: fx={intr.fx:.2f}, fy={intr.fy:.2f}, cx={intr.cx:.2f}, cy={intr.cy:.2f}")

    out_path = OUT_DIR / f"out_rgbd_{datetime.now().strftime('%H%M%S')}_{bag_label}.mp4"
    writer = None
    cv2.namedWindow("RGBD Inference (Resizable)", cv2.WINDOW_NORMAL)

    frame_idx = 0
    held_dino_detections = []
    held_dino_age = DINO_HOLD_FRAMES + 1

    def _iter_frames_with_fallback(primary, secondary):
        try:
            for frm in rgbd_reader.iter_rgbd_frames(
                primary,
                max_time_diff=rgbd_reader.MAX_TIME_DIFF,
                max_frames=max_frames,
            ):
                yield frm
        except Exception as exc:
            if secondary is None:
                raise
            print(f"[RGBD] Warning: frame stream failed from {Path(primary).name}: {exc}")
            print(f"[RGBD] Retrying stream with alternate source: {Path(secondary).name}")
            for frm in rgbd_reader.iter_rgbd_frames(
                secondary,
                max_time_diff=rgbd_reader.MAX_TIME_DIFF,
                max_frames=max_frames,
            ):
                yield frm

    secondary_source = fallback_source if active_source == bag_source else bag_source

    for frame in _iter_frames_with_fallback(
        active_source,
        secondary_source,
    ):
        frame_idx += 1
        rgb_frame = frame.rgb.copy()
        depth_mm = frame.depth_mm

        preds = run_yolo_ensemble(v1, v2, v3, rgb_frame)

        run_dino_now = (
            frame_idx > 1 and
            ((frame_idx - 1) % DINO_VIDEO_INTERVAL_FRAMES == 0)
        )

        if run_dino_now:
            pil_img = Image.fromarray(cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB))
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
        final_dets = apply_depth_size_filter(final_dets, depth_mm, intr)

        annotated_rgb = draw_predictions(rgb_frame, final_dets)
        cv2.putText(
            annotated_rgb,
            f"Frame: {frame_idx} | Timestamp: {frame.timestamp:.2f}s",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if writer is None:
            height, width = annotated_rgb.shape[:2]
            writer = cv2.VideoWriter(
                str(out_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                20.0,
                (width, height),
            )

        writer.write(annotated_rgb)
        cv2.imshow("RGBD Inference (Resizable)", annotated_rgb)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n[INFO] RGBD bag processing interrupted early by user input key.")
            break

        if frame_idx % 100 == 0:
            print(f"[RGBD] Processed synced frames: {frame_idx}")

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    print("\n" + "═" * 70)
    print("[SUCCESS] RGBD bag rendering complete!")
    print(f" -> Processed synced frames: {frame_idx}")
    print(f" -> Saved to: {out_path.resolve()}")
    print("═" * 70 + "\n")

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
        process_rgbd_bag(v1, v2, v3, args.bag, max_frames=args.max_frames)
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