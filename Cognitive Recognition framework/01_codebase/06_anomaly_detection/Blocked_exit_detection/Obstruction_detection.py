"""
Obstruction detection (RGB-D)
YOLO v1 + YOLO v3 + DINO + SAM -> reuse door zone geometry -> flag blocked exit.

Pipeline:
  1) Detect door with the same cadence/hold logic as Door-exit_Detect (via door_zone_rgbd).
  2) Detect non-door objects with YOLO v1+v3 every frame.
  3) Refresh object proposals with DINO every DINO_INTERVAL frames.
  4) Refine object masks with SAM and hold them for HOLD_FRAMES.
    5) Back-project object mask pixels to 3D and mark obstruction when any part of an
         object intersects the door half-cylinder keep-clear zone.

Usage:
  python Obstruction_detection.py --bag "02_datasets/saxon/hallway 1"
  python Obstruction_detection.py --bag "02_datasets/saxon/hallway 1" --max-frames 300
  python Obstruction_detection.py --bag "02_datasets/saxon/hallway 1" --radius 1.2
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys

import cv2
import numpy as np

# Local modules in the same folder.
BASE_DIR = Path(__file__).parent.resolve()
REPO_ROOT = BASE_DIR.parents[2]
OUT_DIR = REPO_ROOT / "04_outputs_runs_and_logs/AD_Rules_Outputs"
sys.path.insert(0, str(BASE_DIR))

import RGBD_Reader as rd  # noqa: E402
import door_zone_rgbd as dz  # noqa: E402


det = dz.det
DINO_INTERVAL = det.DINO_INTERVAL
HOLD_FRAMES = det.HOLD_FRAMES

V1_PATH = REPO_ROOT / "03_models_and_weights/models/yolo_trained_v1.pt"
V3_PATH = REPO_ROOT / "03_models_and_weights/models/yolo_trained_v3.pt"

YOLO_CONF = 0.40
YOLO_IOU = 0.45
DINO_TEXT_OBSTRUCTION = (
    "object. obstacle. item. thing."
)
MAX_OBJ_PROPOSALS = 12
DOOR_MASK_DISTORTION_THRESHOLD = 0.05

OBJ_SAFE_COLOR = (0, 180, 255)   # orange
OBJ_BLOCK_COLOR = (0, 0, 255)    # red
TXT_COLOR = (255, 255, 255)

HUD_FS = 0.44
HUD_TH = 1
LABEL_FS = 0.40
LABEL_TH = 1
ALERT_FS = 1.20
ALERT_TH = 3


def _normalize_label(label: str) -> str:
    return str(label).strip().lower().replace("-", " ").replace("_", " ")


def _is_door_label(label: str) -> bool:
    clean = _normalize_label(label)
    return det._is_door_label(clean)


def _is_sign_label(label: str) -> bool:
    clean = _normalize_label(label)
    return det._is_sign_label(clean)


def _is_obstruction_label(label: str) -> bool:
    clean = _normalize_label(label)
    if not clean:
        return False
    if _is_door_label(clean):
        return False
    if _is_sign_label(clean):
        return False
    return True


def _names_map(model) -> dict[int, str]:
    names = model.names
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(v) for i, v in enumerate(names)}


def _yolo_non_door_boxes(v1, v3, frame: np.ndarray, device: str) -> list[tuple[float, float, float, float, float]]:
    """Run YOLO v1+v3 and return merged non-door proposals (class-wise NMS)."""
    r1 = v1.predict(source=frame, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False, device=device)[0]
    r3 = v3.predict(source=frame, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False, device=device)[0]

    grouped: dict[str, list[tuple[float, float, float, float, float]]] = {}

    for model, result in ((v1, r1), (v3, r3)):
        if result.boxes is None or len(result.boxes) == 0:
            continue
        nmap = _names_map(model)
        xyxy = result.boxes.xyxy.detach().cpu().numpy()
        conf = result.boxes.conf.detach().cpu().numpy()
        cls = result.boxes.cls.detach().cpu().numpy().astype(int)

        for i in range(len(xyxy)):
            label = _normalize_label(nmap.get(int(cls[i]), ""))
            if not _is_obstruction_label(label):
                continue
            x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
            c = float(conf[i])
            grouped.setdefault(label, []).append((x1, y1, x2, y2, c))

    merged: list[tuple[float, float, float, float, float]] = []
    for _, dets in grouped.items():
        merged.extend(det._nms_merge(dets, iou_thr=0.45))

    merged = sorted(merged, key=lambda d: d[4], reverse=True)
    return merged[:MAX_OBJ_PROPOSALS]


def _dino_non_door_boxes(processor, dino_model, frame: np.ndarray, device: str) -> list[tuple[float, float, float, float, float]]:
    """Run DINO with a generic obstruction prompt and keep labels that are not doors/signs."""
    return det._dino_query(
        processor,
        dino_model,
        frame,
        device,
        text_prompt=DINO_TEXT_OBSTRUCTION,
        label_filter=_is_obstruction_label,
    )


def _detect_obstruction_masks(
    frame: np.ndarray,
    frame_idx: int,
    models: dict,
    state: dict,
    device: str,
) -> list[tuple]:
    """
    Same hold/cadence logic pattern as Door-exit_Detect, applied to non-door objects.
    Returns held SAM-refined object detections as tuples (x1,y1,x2,y2,conf,polygon|None).
    """
    yolo_obj_boxes = _yolo_non_door_boxes(models["v1"], models["v3"], frame, device)

    run_dino_now = (frame_idx > 1) and ((frame_idx - 1) % DINO_INTERVAL == 0)
    if run_dino_now:
        dino_obj_boxes = _dino_non_door_boxes(models["proc"], models["dino"], frame, device)
        proposals = det._nms_merge(yolo_obj_boxes + dino_obj_boxes, iou_thr=0.30)
        proposals = sorted(proposals, key=lambda d: d[4], reverse=True)[:MAX_OBJ_PROPOSALS]
        state["held_objs"] = (
            models["sam"].refine_boxes(frame, proposals, frame_id=frame_idx)
            if proposals else []
        )
        state["held_age"] = 0
    else:
        state["held_age"] += 1

    return state["held_objs"] if state["held_age"] <= HOLD_FRAMES else []


def _point_inside_half_cylinder(
    obj_xyz: tuple[float, float, float],
    door_xyz: tuple[float, float, float],
    radius: float,
    door_bottom_y: float,
    door_top_y: float,
) -> bool:
    """Check if a 3D point is inside the front half-cylinder zone anchored at the door."""
    x_o, y_o, z_o = obj_xyz
    x_d, _, z_d = door_xyz

    y_min = min(door_top_y, door_bottom_y)
    y_max = max(door_top_y, door_bottom_y)
    if y_o < y_min or y_o > y_max:
        return False

    dx = x_o - x_d
    dz = z_o - z_d

    # Only keep-clear space in front of the door (toward camera): z <= z_d.
    if dz > 0:
        return False

    return (dx * dx + dz * dz) <= (radius * radius)


def _mask_intersects_half_cylinder(
    obj_mask: np.ndarray,
    depth_mm: np.ndarray,
    intr,
    door_xyz: tuple[float, float, float],
    radius: float,
    door_bottom_y: float,
    door_top_y: float,
) -> bool:
    """True when any valid-depth pixel of the object mask enters the 3D keep-clear zone."""
    valid_mask = (obj_mask > 0) & (depth_mm > 0)
    ys, xs = np.where(valid_mask)
    if ys.size == 0:
        return False

    z = depth_mm[ys, xs].astype(np.float32) / 1000.0
    u = xs.astype(np.float32)
    v = ys.astype(np.float32)

    x = (u - float(intr.cx)) * z / float(intr.fx)
    y = (v - float(intr.cy)) * z / float(intr.fy)

    x_d, _, z_d = door_xyz
    dx = x - float(x_d)
    dz = z - float(z_d)

    y_min = min(door_top_y, door_bottom_y)
    y_max = max(door_top_y, door_bottom_y)

    inside = (
        (y >= y_min)
        & (y <= y_max)
        & (dz <= 0.0)
        & ((dx * dx + dz * dz) <= (radius * radius))
    )
    return bool(np.any(inside))


def _door_mask_distortion(mask: np.ndarray) -> float | None:
    """
    Compute how non-rectangular the door SAM mask is.
    Distortion is defined as 1 - IoU(mask, best-fit rotated rectangle).
    """
    if mask is None or mask.size == 0:
        return None

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) <= 1.0:
        return None

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect).astype(np.int32)

    rect_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillPoly(rect_mask, [box], 255)

    a = mask > 0
    b = rect_mask > 0
    inter = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    if union <= 0:
        return None

    iou = float(inter) / float(union)
    return float(max(0.0, min(1.0, 1.0 - iou)))


def run(
    bag_dir: Path,
    radius: float,
    max_frames: int,
    fps: float,
) -> Path:
    if not (bag_dir / "metadata.yaml").exists():
        raise FileNotFoundError(f"metadata.yaml not found in bag dir: {bag_dir}")
    if not V1_PATH.exists():
        raise FileNotFoundError(f"YOLO v1 weights not found: {V1_PATH}")
    if not V3_PATH.exists():
        raise FileNotFoundError(f"YOLO v3 weights not found: {V3_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = det._assert_gpu()

    intr = rd.read_intrinsics(bag_dir)
    print(f"[INTRINSICS] {intr}")
    print(f"[ZONE] radius={radius:.2f}m  (door height from detected mask)")

    print("[INIT] Loading YOLO v1 ...")
    v1 = det.YOLO(str(V1_PATH))
    v1.to(device)

    print("[INIT] Loading YOLO v3 ...")
    v3 = det.YOLO(str(V3_PATH))
    v3.to(device)

    print("[INIT] Loading Grounding DINO ...")
    processor = det.AutoProcessor.from_pretrained(det.DINO_MODEL_ID)
    dino_model = det.AutoModelForZeroShotObjectDetection.from_pretrained(det.DINO_MODEL_ID).to(device)
    dino_model.eval()

    print("[INIT] Loading SAM refiner ...")
    sam_refiner = det.GroundedSAMRefiner(
        ckpt_path=det.SAM_CKPT_PATH,
        model_type="vit_h",
        device=device,
    )

    models = {
        "v1": v1,
        "v3": v3,
        "proc": processor,
        "dino": dino_model,
        "sam": sam_refiner,
    }

    door_state = {"held_doors": [], "held_age": HOLD_FRAMES + 1}
    obj_state = {"held_objs": [], "held_age": HOLD_FRAMES + 1}

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_video = OUT_DIR / f"{bag_dir.name}_obstruction_{stamp}.mp4"
    out_csv = OUT_DIR / f"{bag_dir.name}_obstruction_{stamp}.csv"

    writer: cv2.VideoWriter | None = None
    csv_f = out_csv.open("w", newline="", encoding="utf-8")
    csv_w = csv.writer(csv_f)
    csv_w.writerow([
        "frame",
        "timestamp",
        "door_Z_m",
        "door_mask_distortion",
        "door_shape_obstruction",
        "objects_active",
        "objects_blocking",
        "obstruction_flag",
    ])

    frame_idx = 0
    obstruction_frames = 0

    try:
        for f in rd.iter_rgbd_frames(bag_dir, max_frames=max_frames):
            frame_idx += 1
            frame = f.rgb
            h, w = frame.shape[:2]

            if writer is None:
                writer = cv2.VideoWriter(
                    str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                )

            doors = dz._detect_door_mask(frame, frame_idx, {
                "yolo": models["v3"],
                "proc": models["proc"],
                "dino": models["dino"],
                "sam": models["sam"],
            }, door_state, device)
            objects = _detect_obstruction_masks(frame, frame_idx, models, obj_state, device)

            door_valid = False
            door_Z = ""
            door_distortion = ""
            door_shape_obstruction = 0
            door_xyz: tuple[float, float, float] | None = None
            door_top_y: float | None = None
            door_bottom_y: float | None = None

            if doors:
                best = max(doors, key=lambda d: d[4])
                mask = dz._mask_from_det(best, (h, w))
                if mask is not None:
                    distortion = _door_mask_distortion(mask)
                    if distortion is not None:
                        door_distortion = round(distortion, 4)
                        door_shape_obstruction = int(distortion > DOOR_MASK_DISTORTION_THRESHOLD)

                    centroid = dz._mask_centroid(mask)
                    z_m = dz._median_depth_m(f.depth_mm, mask)
                    if centroid is not None and z_m is not None and z_m > 0:
                        u, v = centroid
                        X, Y, Z = dz._backproject(u, v, z_m, intr)

                        y1, y2 = float(best[1]), float(best[3])
                        _, Y_top, _ = dz._backproject(u, y1, z_m, intr)
                        _, Y_bottom, _ = dz._backproject(u, y2, z_m, intr)
                        door_h = abs(Y_bottom - Y_top)

                        door_top_y = Y - door_h / 2.0
                        door_bottom_y = Y + door_h / 2.0
                        door_xyz = (X, Y, Z)
                        door_Z = round(Z, 3)
                        door_valid = True

                        poly = best[5] if len(best) > 5 else None
                        if poly is not None:
                            cv2.polylines(frame, [poly.astype(np.int32)], True, dz.DOOR_POLY_COLOR, 2)

                        frame = dz._draw_zone(frame, door_xyz, radius, door_bottom_y, door_top_y, intr)

                        if distortion is not None:
                            cv2.putText(
                                frame,
                                f"door-shape distortion={distortion:.2f} ({'BLOCK' if door_shape_obstruction else 'ok'})",
                                (12, 80),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                HUD_FS,
                                (0, 0, 255) if door_shape_obstruction else (0, 220, 0),
                                HUD_TH,
                                cv2.LINE_AA,
                            )

            blocking_count = 0
            for obj in objects:
                obj_mask = dz._mask_from_det(obj, (h, w))
                if obj_mask is None:
                    continue

                z_m = dz._median_depth_m(f.depth_mm, obj_mask)
                Zo_txt = f"{z_m:.2f}m" if (z_m is not None and z_m > 0) else "n/a"
                blocked = False
                if door_valid and door_xyz is not None and door_top_y is not None and door_bottom_y is not None:
                    blocked = _mask_intersects_half_cylinder(
                        obj_mask,
                        f.depth_mm,
                        intr,
                        door_xyz,
                        radius,
                        door_bottom_y,
                        door_top_y,
                    )

                if blocked:
                    blocking_count += 1

                poly = obj[5] if len(obj) > 5 else None
                color = OBJ_BLOCK_COLOR if blocked else OBJ_SAFE_COLOR
                if poly is not None:
                    cv2.polylines(frame, [poly.astype(np.int32)], True, color, 2)
                x1, y1, x2, y2 = [int(round(v)) for v in obj[:4]]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"obj {Zo_txt} {'BLOCK' if blocked else 'clear'}",
                    (x1, max(16, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    LABEL_FS,
                    TXT_COLOR,
                    LABEL_TH,
                    cv2.LINE_AA,
                )

            obstruction_flag = 1 if (blocking_count > 0 or door_shape_obstruction == 1) else 0
            if obstruction_flag:
                obstruction_frames += 1

            csv_w.writerow([
                frame_idx,
                round(f.timestamp, 6),
                door_Z,
                door_distortion,
                door_shape_obstruction,
                len(objects),
                blocking_count,
                obstruction_flag,
            ])

            cv2.putText(
                frame,
                f"f={frame_idx} doors={1 if door_valid else 0} objs={len(objects)} block={blocking_count}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                HUD_FS,
                (0, 255, 255),
                HUD_TH,
                cv2.LINE_AA,
            )
            mode = f"dino_every={DINO_INTERVAL} hold={HOLD_FRAMES}"
            cv2.putText(
                frame,
                mode,
                (12, 54),
                cv2.FONT_HERSHEY_SIMPLEX,
                HUD_FS,
                (240, 240, 240),
                HUD_TH,
                cv2.LINE_AA,
            )

            if obstruction_flag:
                cv2.putText(
                    frame,
                    "BLOCK ALERT",
                    (int(w * 0.33), 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    ALERT_FS,
                    (0, 0, 255),
                    ALERT_TH,
                    cv2.LINE_AA,
                )

            writer.write(frame)

            if frame_idx % 50 == 0:
                print(
                    f"[RUN] frame={frame_idx} active_objs={len(objects)} "
                    f"blocking={blocking_count} obstruction_frames={obstruction_frames}"
                )

    finally:
        if writer is not None:
            writer.release()
        csv_f.close()

    print(f"\n[DONE] video : {out_video}")
    print(f"[DONE] csv   : {out_csv}")
    print(f"[DONE] frames: {frame_idx}  obstruction_frames: {obstruction_frames}")
    return out_video


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Obstruction detection with YOLO v1+v3 + DINO + SAM, "
            "using RGB-D door keep-clear zone checking."
        )
    )
    parser.add_argument("--bag", type=str, required=True, help="Path to bag folder (contains metadata.yaml)")
    parser.add_argument("--radius", type=float, default=1.0, help="Door zone radius in metres")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N synced frames (0 = all)")
    parser.add_argument("--fps", type=float, default=25.0, help="Output video FPS")
    args = parser.parse_args()

    run(
        bag_dir=Path(args.bag),
        radius=float(args.radius),
        max_frames=max(0, int(args.max_frames)),
        fps=float(args.fps),
    )


if __name__ == "__main__":
    main()
