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

BASE_DIR = Path(__file__).parent.resolve()
REPO_ROOT = BASE_DIR.parents[2]

V1_PATH = REPO_ROOT / "03_models_and_weights/models/yolo_trained_v1.pt"
V3_PATH = REPO_ROOT / "03_models_and_weights/models/yolo_trained_v3.pt"
SAM_CKPT_PATH = BASE_DIR / "sam_masker" / "sam_vit_h_4b8939.pth"

DEFAULT_VIDEO = REPO_ROOT / "10_Testing/Rules for AD/Person_blocking_hospital_exit_202606091138.mp4"
OUT_DIR = REPO_ROOT / "04_outputs_runs_and_logs/AD_Rules_Outputs"

DINO_MODEL_ID = "IDEA-Research/grounding-dino-base"
DINO_TEXT = "hospital corridor door."
DINO_BOX_THR = 0.18
DINO_TEXT_THR = 0.25
DINO_INTERVAL = 15
HOLD_FRAMES = 15

YOLO_CONF = 0.25
YOLO_IOU = 0.45


sys.path.insert(0, str(BASE_DIR / "sam_masker"))
from grounded_sam import GroundedSAMRefiner  # noqa: E402


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1e-6, (bx2 - bx1) * (by2 - by1))
    return inter / (area_a + area_b - inter + 1e-6)


def _nms_merge(dets: list[tuple[float, float, float, float, float]], iou_thr: float = 0.30) -> list[tuple[float, float, float, float, float]]:
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


def yolo_doors(model: YOLO, frame: np.ndarray, device: str) -> list[tuple[float, float, float, float, float]]:
    out = model.predict(source=frame, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False, device=device)[0]
    nmap = _names_map(model)
    doors: list[tuple[float, float, float, float, float]] = []
    if out.boxes is None or len(out.boxes) == 0:
        return doors

    xyxy = out.boxes.xyxy.detach().cpu().numpy()
    conf = out.boxes.conf.detach().cpu().numpy()
    cls = out.boxes.cls.detach().cpu().numpy().astype(int)

    for i in range(len(xyxy)):
        label = nmap.get(int(cls[i]), "")
        if "door" not in label.lower():
            continue
        x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
        doors.append((x1, y1, x2, y2, float(conf[i])))
    return doors


def dino_doors(
    processor: AutoProcessor,
    model: AutoModelForZeroShotObjectDetection,
    frame: np.ndarray,
    device: str,
) -> list[tuple[float, float, float, float, float]]:
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(images=pil, text=DINO_TEXT, return_tensors="pt").to(device)
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

    doors: list[tuple[float, float, float, float, float]] = []
    boxes = results.get("boxes", [])
    scores = results.get("scores", [])
    labels = results.get("labels", [])
    for b, s, lab in zip(boxes, scores, labels):
        lab_txt = str(lab).lower()
        if "door" not in lab_txt:
            continue
        x1, y1, x2, y2 = [float(v) for v in b.detach().cpu().tolist()]
        doors.append((x1, y1, x2, y2, float(s.detach().cpu().item())))
    return doors


def draw_refined(frame: np.ndarray, refined: list[tuple], source_tag: str) -> np.ndarray:
    for det in refined:
        x1, y1, x2, y2, conf = det[0], det[1], det[2], det[3], det[4]
        poly = det[5] if len(det) > 5 else None

        if poly is not None:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [poly], (255, 0, 0))
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
            cv2.polylines(frame, [poly], isClosed=True, color=(255, 0, 0), thickness=2)

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(
            frame,
            f"door {conf:.2f} {source_tag}",
            (int(x1), max(16, int(y1) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return frame


def run_demo(video_path: Path, max_frames: int = 0) -> Path:
    if not V1_PATH.exists() or not V3_PATH.exists():
        raise FileNotFoundError("YOLO weights not found in 03_models_and_weights/models")
    if not SAM_CKPT_PATH.exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {SAM_CKPT_PATH}")
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INIT] device={device}")
    print(f"[INIT] video={video_path}")

    print("[INIT] Loading YOLO v1 + v3...")
    yolo_v1 = YOLO(str(V1_PATH))
    yolo_v3 = YOLO(str(V3_PATH))

    print("[INIT] Loading Grounding DINO...")
    processor = AutoProcessor.from_pretrained(DINO_MODEL_ID)
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL_ID).to(device)
    dino_model.eval()

    print("[INIT] Loading SAM refiner...")
    sam_refiner = GroundedSAMRefiner(ckpt_path=SAM_CKPT_PATH, model_type="vit_h", device=device)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"{video_path.stem}_door_groundedsam_{stamp}.mp4"

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_idx = 0
    dino_hits = 0
    yolo_hits = 0
    held_refined: list[tuple] = []
    held_age = HOLD_FRAMES + 1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if max_frames > 0 and frame_idx > max_frames:
            break

        # YOLO still runs each frame, but SAM is only triggered at DINO cadence.
        yolo_boxes = yolo_doors(yolo_v1, frame, device=device) + yolo_doors(yolo_v3, frame, device=device)
        if yolo_boxes:
            yolo_hits += len(yolo_boxes)

        run_grounded_sam_now = ((frame_idx - 1) % DINO_INTERVAL == 0)
        source_tag = "held"

        if run_grounded_sam_now:
            dino_boxes = dino_doors(processor, dino_model, frame, device=device)
            if dino_boxes:
                dino_hits += len(dino_boxes)

            proposals = _nms_merge(yolo_boxes + dino_boxes, iou_thr=0.30)
            proposals = sorted(proposals, key=lambda d: d[4], reverse=True)[:2]

            held_refined = sam_refiner.refine_boxes(frame, proposals, frame_id=frame_idx) if proposals else []
            held_age = 0
            source_tag = "yolo+dino+sam"
        else:
            held_age += 1

        refined = held_refined if held_age <= HOLD_FRAMES else []
        frame = draw_refined(frame, refined, source_tag=source_tag)

        cv2.putText(
            frame,
            f"f={frame_idx}/{total} held={len(refined)} dino+sam_every={DINO_INTERVAL}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)

        if frame_idx % 50 == 0:
            print(f"[RUN] frame {frame_idx}/{total}")

    cap.release()
    writer.release()

    print(f"[DONE] output={out_path}")
    print(f"[DONE] frames={frame_idx} yolo_hits={yolo_hits} dino_hits={dino_hits}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Door-only Grounded-SAM demo (YOLO v1+v3 + DINO + SAM)")
    parser.add_argument("--video", type=str, default=str(DEFAULT_VIDEO), help="Input video path")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit frames for quick test (0 = full video)")
    args = parser.parse_args()

    run_demo(Path(args.video), max_frames=max(0, int(args.max_frames)))


if __name__ == "__main__":
    main()
