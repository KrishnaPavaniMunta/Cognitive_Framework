"""
Door + Exit Sign Detection — YOLO + Grounding DINO + SAM  (GPU only)

Detection strategy:
  - Doors  : YOLO every frame  →  DINO+SAM fallback every DINO_INTERVAL frames
  - Signs  : YOLO every frame  →  DINO-only fallback every DINO_INTERVAL frames (no SAM)

Usage:
    python door_sign_detect.py
    python door_sign_detect.py --video /path/to/video.mp4
    python door_sign_detect.py --video /path/to/video.mp4 --yolo-only
    python door_sign_detect.py --video /path/to/video.mp4 --max-frames 300
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from ultralytics import YOLO

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()
REPO_ROOT = BASE_DIR.parents[2]

YOLO_MODEL_PATH = REPO_ROOT / "03_models_and_weights/models/yolo26m.pt"
SAM_CKPT_PATH   = BASE_DIR / "sam_masker" / "sam_vit_h_4b8939.pth"
DEFAULT_VIDEO   = REPO_ROOT / "10_Testing/Rules for AD/Person_blocking_hospital_exit_202606091138.mp4"
OUT_DIR         = REPO_ROOT / "04_outputs_runs_and_logs/AD_Rules_Outputs"

# ── DINO settings ─────────────────────────────────────────────────────────────
DINO_MODEL_ID   = "IDEA-Research/grounding-dino-base"

# Separate prompts — mixing them in one string dilutes confidence scores
DINO_TEXT_DOOR  = "hospital corridor door."
DINO_TEXT_SIGN  = "green exit sign. fire exit sign. emergency exit sign."

DINO_BOX_THR    = 0.30
DINO_TEXT_THR   = 0.35
DINO_INTERVAL   = 15   # run DINO fallback every N frames
HOLD_FRAMES     = 15   # hold last SAM/DINO result for N frames

# ── YOLO settings ─────────────────────────────────────────────────────────────
YOLO_CONF = 0.45
YOLO_IOU  = 0.45

# ── Label sets ────────────────────────────────────────────────────────────────
DOOR_LABELS: set[str] = {
    "door", "hospital door", "corridor door", "fire door", "exit door",
}
SIGN_LABELS: set[str] = {
    "exit sign", "fire exit sign", "emergency exit sign",
    "exit", "fire exit", "emergency exit",
    "green exit sign", "exit sign above door",
}

# ── Drawing colours (BGR) ─────────────────────────────────────────────────────
DOOR_BOX_COLOR = (255,  80,   0)   # blue-orange for doors
SIGN_BOX_COLOR = (0,   200,   0)   # green for exit signs
TXT_COLOR      = (255, 255, 255)

# ─────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(BASE_DIR / "sam_masker"))
from grounded_sam import GroundedSAMRefiner  # noqa: E402


# ── Helpers ───────────────────────────────────────────────────────────────────

def _assert_gpu() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"[INIT] GPU: {name}")
        return "cuda"
    else:
        print("[WARN] No CUDA GPU detected, falling back to CPU (slower)")
        return "cpu"


def _is_door_label(label: str) -> bool:
    clean = label.strip().lower()
    if clean in DOOR_LABELS:
        return True
    parts = clean.split()
    return "door" in parts and "exit" not in parts  # don't confuse "exit door" sign


def _is_sign_label(label: str) -> bool:
    clean = label.strip().lower()
    if clean in SIGN_LABELS:
        return True
    # must contain "exit" and at least one of "sign","fire","emergency","green"
    has_exit = "exit" in clean.split()
    has_qualifier = any(w in clean for w in ("sign", "fire", "emergency", "green"))
    return has_exit and has_qualifier


def _iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1);  iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2);  iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter  = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1e-6, (bx2 - bx1) * (by2 - by1))
    return inter / (area_a + area_b - inter + 1e-6)


def _nms_merge(
    dets: list[tuple[float, float, float, float, float]],
    iou_thr: float = 0.30,
) -> list[tuple[float, float, float, float, float]]:
    if not dets:
        return []
    dets_sorted = sorted(dets, key=lambda d: d[4], reverse=True)
    keep: list[tuple[float, float, float, float, float]] = []
    while dets_sorted:
        best = dets_sorted.pop(0)
        keep.append(best)
        dets_sorted = [d for d in dets_sorted if _iou(best[:4], d[:4]) < iou_thr]
    return keep


def _names_map(model: YOLO) -> dict[int, str]:
    names = model.names
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(v) for i, v in enumerate(names)}


def _warmup(model: YOLO, height: int, width: int, device: str) -> None:
    print("[INIT] Warming up YOLO ...")
    dummy = np.zeros((height, width, 3), dtype=np.uint8)
    model.predict(source=dummy, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False, device=device)
    print("[INIT] Warmup done.")


# ── YOLO detection ────────────────────────────────────────────────────────────

def yolo_detect(
    model: YOLO,
    frame: np.ndarray,
    device: str,
) -> tuple[
    list[tuple[float, float, float, float, float]],
    list[tuple[float, float, float, float, float]],
]:
    """Return (door_boxes, sign_boxes) from a single YOLO pass."""
    out = model.predict(
        source=frame, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False, device=device
    )[0]

    doors: list[tuple[float, float, float, float, float]] = []
    signs: list[tuple[float, float, float, float, float]] = []

    if out.boxes is None or len(out.boxes) == 0:
        return doors, signs

    nmap = _names_map(model)
    xyxy = out.boxes.xyxy.detach().cpu().numpy()
    conf = out.boxes.conf.detach().cpu().numpy()
    cls  = out.boxes.cls.detach().cpu().numpy().astype(int)

    for i in range(len(xyxy)):
        label = nmap.get(int(cls[i]), "")
        x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
        c = float(conf[i])
        if _is_door_label(label):
            doors.append((x1, y1, x2, y2, c))
        elif _is_sign_label(label):
            signs.append((x1, y1, x2, y2, c))

    return doors, signs


# ── DINO detection ────────────────────────────────────────────────────────────

def _dino_query(
    processor: AutoProcessor,
    model: AutoModelForZeroShotObjectDetection,
    frame: np.ndarray,
    device: str,
    text_prompt: str,
    label_filter,          # callable(str) -> bool
) -> list[tuple[float, float, float, float, float]]:
    """Run one DINO query with a given text prompt and label filter."""
    pil    = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(images=pil, text=text_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([pil.size[::-1]], device=device)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=DINO_BOX_THR,
        text_threshold=DINO_TEXT_THR,
        target_sizes=target_sizes,
    )[0]

    hits: list[tuple[float, float, float, float, float]] = []
    for b, s, lab in zip(
        results.get("boxes",  []),
        results.get("scores", []),
        results.get("labels", []),
    ):
        if not label_filter(str(lab)):
            continue
        x1, y1, x2, y2 = [float(v) for v in b.detach().cpu().tolist()]
        score = float(min(max(s.detach().cpu().item(), 0.0), 1.0))
        hits.append((x1, y1, x2, y2, score))
    return hits


def dino_detect(
    processor: AutoProcessor,
    dino_model: AutoModelForZeroShotObjectDetection,
    frame: np.ndarray,
    device: str,
) -> tuple[
    list[tuple[float, float, float, float, float]],
    list[tuple[float, float, float, float, float]],
]:
    """Return (door_boxes, sign_boxes) from two separate DINO queries."""
    door_boxes = _dino_query(
        processor, dino_model, frame, device,
        text_prompt=DINO_TEXT_DOOR,
        label_filter=_is_door_label,
    )
    sign_boxes = _dino_query(
        processor, dino_model, frame, device,
        text_prompt=DINO_TEXT_SIGN,
        label_filter=_is_sign_label,
    )
    return door_boxes, sign_boxes


# ── Drawing ───────────────────────────────────────────────────────────────────

def _draw_boxes(
    frame: np.ndarray,
    boxes: list[tuple],
    label: str,
    color: tuple[int, int, int],
    source_tag: str,
) -> np.ndarray:
    for det in boxes:
        x1, y1, x2, y2, conf = det[0], det[1], det[2], det[3], det[4]
        poly = det[5] if len(det) > 5 else None

        if poly is not None:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [poly], color)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
            cv2.polylines(frame, [poly], isClosed=True, color=color, thickness=2)

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(
            frame,
            f"{label} {conf:.2f} [{source_tag}]",
            (int(x1), max(16, int(y1) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            TXT_COLOR,
            2,
            cv2.LINE_AA,
        )
    return frame


def draw_all(
    frame: np.ndarray,
    door_dets: list[tuple],
    sign_dets: list[tuple],
    door_tag: str,
    sign_tag: str,
) -> np.ndarray:
    frame = _draw_boxes(frame, door_dets, "door",      DOOR_BOX_COLOR, door_tag)
    frame = _draw_boxes(frame, sign_dets, "exit sign", SIGN_BOX_COLOR, sign_tag)
    return frame


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_demo(
    video_path: Path,
    max_frames: int = 0,
    yolo_only: bool = False,
) -> Path:
    if not YOLO_MODEL_PATH.exists():
        raise FileNotFoundError(f"YOLO weights not found: {YOLO_MODEL_PATH}")
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")
    if not yolo_only and not SAM_CKPT_PATH.exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {SAM_CKPT_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = _assert_gpu()
    print(f"[INIT] video={video_path}")

    # ── Load models ───────────────────────────────────────────────────────────
    print("[INIT] Loading YOLO ...")
    yolo_model = YOLO(str(YOLO_MODEL_PATH))
    yolo_model.to(device)

    processor   = None
    dino_model  = None
    sam_refiner = None

    if not yolo_only:
        print("[INIT] Loading Grounding DINO ...")
        processor  = AutoProcessor.from_pretrained(DINO_MODEL_ID)
        dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL_ID).to(device)
        dino_model.eval()

        # SAM only needed for doors (pixel-precise polygon)
        print("[INIT] Loading SAM refiner (doors only) ...")
        sam_refiner = GroundedSAMRefiner(
            ckpt_path=SAM_CKPT_PATH, model_type="vit_h", device=device
        )

    # ── Open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] {width}x{height} @ {fps:.1f}fps  total_frames={total}")

    _warmup(yolo_model, height, width, device)

    # ── Output writer ─────────────────────────────────────────────────────────
    stamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"{video_path.stem}_door_sign_{stamp}.mp4"
    writer   = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    # ── State ─────────────────────────────────────────────────────────────────
    frame_idx = 0

    # Doors — held SAM-refined detections
    held_doors:     list[tuple] = []
    held_doors_age: int         = HOLD_FRAMES + 1

    # Signs — held DINO detections (no SAM)
    held_signs:     list[tuple] = []
    held_signs_age: int         = HOLD_FRAMES + 1

    yolo_door_hits = 0
    yolo_sign_hits = 0
    dino_door_hits = 0
    dino_sign_hits = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if max_frames > 0 and frame_idx > max_frames:
                break

            # ── YOLO — single pass, split into doors + signs ──────────────────
            yolo_door_boxes, yolo_sign_boxes = yolo_detect(yolo_model, frame, device)
            yolo_door_hits += len(yolo_door_boxes)
            yolo_sign_hits += len(yolo_sign_boxes)

            if yolo_only:
                frame = draw_all(
                    frame,
                    yolo_door_boxes, yolo_sign_boxes,
                    door_tag="yolo", sign_tag="yolo",
                )

            else:
                run_dino_now = (frame_idx > 1) and ((frame_idx - 1) % DINO_INTERVAL == 0)

                if run_dino_now:
                    # ── DINO fallback ─────────────────────────────────────────
                    dino_door_boxes, dino_sign_boxes = dino_detect(
                        processor, dino_model, frame, device
                    )
                    dino_door_hits += len(dino_door_boxes)
                    dino_sign_hits += len(dino_sign_boxes)

                    # ── Doors: merge YOLO + DINO → SAM refine ────────────────
                    door_proposals = _nms_merge(yolo_door_boxes + dino_door_boxes, iou_thr=0.30)
                    door_proposals = sorted(door_proposals, key=lambda d: d[4], reverse=True)[:2]
                    held_doors = (
                        sam_refiner.refine_boxes(frame, door_proposals, frame_id=frame_idx)
                        if door_proposals else []
                    )
                    held_doors_age = 0

                    # ── Signs: merge YOLO + DINO — NO SAM ────────────────────
                    sign_proposals = _nms_merge(yolo_sign_boxes + dino_sign_boxes, iou_thr=0.30)
                    held_signs     = sorted(sign_proposals, key=lambda d: d[4], reverse=True)[:4]
                    held_signs_age = 0

                    door_tag = "yolo+dino+sam"
                    sign_tag = "yolo+dino"

                else:
                    held_doors_age += 1
                    held_signs_age += 1
                    door_tag = "held"
                    sign_tag = "held"

                active_doors = held_doors if held_doors_age <= HOLD_FRAMES else []
                active_signs = held_signs if held_signs_age <= HOLD_FRAMES else []

                frame = draw_all(frame, active_doors, active_signs, door_tag, sign_tag)

            # ── HUD ───────────────────────────────────────────────────────────
            mode = "yolo-only" if yolo_only else f"dino_every={DINO_INTERVAL}"
            cv2.putText(
                frame,
                (f"f={frame_idx}/{total}  "
                 f"doors(y={len(yolo_door_boxes)})  "
                 f"signs(y={len(yolo_sign_boxes)})  "
                 f"{mode}"),
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            writer.write(frame)

            if frame_idx % 50 == 0:
                print(
                    f"[RUN] f={frame_idx}/{total}  "
                    f"yolo_doors={yolo_door_hits} yolo_signs={yolo_sign_hits}  "
                    f"dino_doors={dino_door_hits} dino_signs={dino_sign_hits}"
                )

    finally:
        cap.release()
        writer.release()

    print(f"\n[DONE] output       : {out_path}")
    print(f"[DONE] frames       : {frame_idx}")
    print(f"[DONE] yolo_doors   : {yolo_door_hits}")
    print(f"[DONE] yolo_signs   : {yolo_sign_hits}")
    print(f"[DONE] dino_doors   : {dino_door_hits}")
    print(f"[DONE] dino_signs   : {dino_sign_hits}")
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Door + exit sign detection (YOLO + DINO + SAM, GPU only)"
    )
    parser.add_argument(
        "--video", type=str, default=str(DEFAULT_VIDEO), help="Input video path"
    )
    parser.add_argument(
        "--max-frames", type=int, default=0,
        help="Process only N frames (0 = full video)"
    )
    parser.add_argument(
        "--yolo-only", action="store_true",
        help="Skip DINO + SAM, run YOLO alone"
    )
    args = parser.parse_args()

    run_demo(
        video_path = Path(args.video),
        max_frames = max(0, int(args.max_frames)),
        yolo_only  = bool(args.yolo_only),
    )


if __name__ == "__main__":
    main()