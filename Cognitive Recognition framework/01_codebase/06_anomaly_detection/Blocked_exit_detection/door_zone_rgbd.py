"""
Door keep-clear zone (RGB-D)  —  YOLO + DINO + SAM  →  3D back-projection  →  half-cylinder zone
────────────────────────────────────────────────────────────────────────────────────────────────
Combines:
  • RGBD_Reader.py        → reads synced RGB + depth + aligned intrinsics from a ROS2 bag
  • Door-exit_Detect.py   → YOLO + Grounding DINO + SAM door segmentation

Then:
  1. Back-projects the SAM door mask centroid into metric 3D (X, Y, Z) using depth + intrinsics.
  2. Calculates the detected door height from bounding box corners.
  3. Builds a semi-circular keep-clear zone (~1 m radius) at the door center,
     extruded upward and downward by half the detected door height, and renders it onto the RGB frame.

Geometry / camera frame conventions
  • Camera optical frame: +X right, +Y down, +Z forward (standard pinhole).
  • "Vertical" = camera Y axis (assumes the camera is roughly level).
  • "In front of the door" = toward the camera (−Z from the door), the only robust choice
    for a single RGB-D view without odometry.
  • Door center Y is the camera Y coordinate at the door centroid depth.
  • Zone extrudes from (door_center_Y - door_height/2) to (door_center_Y + door_height/2).

Usage:
    python door_zone_rgbd.py --bag "02_datasets/saxon/hallway 1"
    python door_zone_rgbd.py --bag "02_datasets/saxon/hallway 1" --max-frames 300
    python door_zone_rgbd.py --bag "02_datasets/saxon/hallway 1" --radius 1.5
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
from datetime import datetime
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent.resolve()
REPO_ROOT = BASE_DIR.parents[2]
OUT_DIR   = REPO_ROOT / "04_outputs_runs_and_logs/AD_Rules_Outputs"

# ── Import the step-1 reader (clean module name) ──────────────────────────────
sys.path.insert(0, str(BASE_DIR))
import RGBD_Reader as rd  # noqa: E402


# ── Import the hyphenated detector module via importlib ───────────────────────
def _load_detector():
    det_path = BASE_DIR / "Door-exit_Detect.py"
    if not det_path.exists():
        raise FileNotFoundError(f"Detector not found: {det_path}")
    spec = importlib.util.spec_from_file_location("door_exit_detect", det_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


det = _load_detector()

# Pull cadence constants from the detector so behaviour stays in sync.
DINO_INTERVAL = det.DINO_INTERVAL
HOLD_FRAMES   = det.HOLD_FRAMES

# ── Zone drawing colours (BGR) ────────────────────────────────────────────────
ZONE_FACE_COLOR   = (60,  60, 220)    # light red fill (BGR)
ZONE_SHADE_COLOR  = (30,  30, 160)    # darker red for shaded curved wall
ZONE_EDGE_COLOR   = (0,   0, 255)     # bright red edges
ZONE_FILL_ALPHA   = 0.35              # translucency of faces
DOOR_POLY_COLOR   = (0, 200, 255)     # door mask outline (yellow)
TXT_COLOR         = (255, 255, 255)

ARC_SAMPLES = 32   # number of samples along the semicircle arc (higher = smoother)


# ── 3D geometry helpers ───────────────────────────────────────────────────────

def _mask_from_det(det_tuple: tuple, shape: tuple[int, int]) -> np.ndarray | None:
    """Build a binary mask from a refined detection (uses polygon if present, else bbox)."""
    h, w = shape
    poly = det_tuple[5] if len(det_tuple) > 5 else None
    mask = np.zeros((h, w), dtype=np.uint8)
    if poly is not None:
        cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
    else:
        x1, y1, x2, y2 = [int(round(v)) for v in det_tuple[:4]]
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return mask if mask.any() else None


def _mask_centroid(mask: np.ndarray) -> tuple[int, int] | None:
    m = cv2.moments(mask, binaryImage=True)
    if m["m00"] <= 0:
        return None
    return int(round(m["m10"] / m["m00"])), int(round(m["m01"] / m["m00"]))


def _median_depth_m(depth_mm: np.ndarray, mask: np.ndarray) -> float | None:
    """Median valid depth (metres) over the mask region."""
    vals = depth_mm[mask > 0]
    vals = vals[vals > 0]
    if vals.size == 0:
        return None
    return float(np.median(vals)) / 1000.0


def _backproject(u: float, v: float, z: float, intr) -> tuple[float, float, float]:
    """Pixel + depth → metric 3D in the camera frame."""
    x = (u - intr.cx) * z / intr.fx
    y = (v - intr.cy) * z / intr.fy
    return x, y, z


def _project(x: float, y: float, z: float, intr) -> tuple[int, int] | None:
    """Metric 3D (camera frame) → pixel. Returns None if behind the camera."""
    if z <= 1e-6:
        return None
    u = intr.fx * x / z + intr.cx
    v = intr.fy * y / z + intr.cy
    if not (np.isfinite(u) and np.isfinite(v)):
        return None
    return int(round(u)), int(round(v))


def _half_cylinder_points(
    door_xyz: tuple[float, float, float],
    radius: float,
    door_bottom_y: float,
    door_top_y: float,
    intr,
) -> tuple[list, list]:
    """
    Sample the half-cylinder in 3D and project all points to pixels.

    Returns (bottom_pts, top_pts) — two parallel arcs in pixel space,
    ordered left-to-right (theta 0→π).  Each entry is (px, py) or None.
    """
    x_door, _, z_door = door_xyz
    bottom_pts = []
    top_pts = []

    for i in range(ARC_SAMPLES + 1):
        theta = np.pi * i / ARC_SAMPLES         # 0 → π
        x = x_door + radius * np.cos(theta)     # left → right along door
        z = z_door - radius * np.sin(theta)     # bulge toward camera (−Z)

        bottom_pts.append(_project(x, door_bottom_y, z, intr))
        top_pts.append(_project(x, door_top_y, z, intr))

    return bottom_pts, top_pts


def _draw_zone(
    frame: np.ndarray,
    door_xyz: tuple[float, float, float],
    radius: float,
    door_bottom_y: float,
    door_top_y: float,
    intr,
) -> np.ndarray:
    """
    Render a 3D-looking translucent half-cylinder keep-clear zone.

    Faces drawn (back-to-front for correct painter's order):
      1. Curved wall strip-quads  (shaded by angle — darker toward sides)
      2. Top disc cap             (lighter fill)
      3. Bottom disc cap          (slightly darker, floor-facing)
      4. Flat back wall           (two vertical edges of the opening)
      5. All edges in bright red
    """
    bottom_pts, top_pts = _half_cylinder_points(
        door_xyz, radius, door_bottom_y, door_top_y, intr
    )

    # Filter paired points where both top & bottom projected successfully.
    valid = [
        (b, t) for b, t in zip(bottom_pts, top_pts)
        if b is not None and t is not None
    ]
    if len(valid) < 3:
        return frame

    bot_ok = [p[0] for p in valid]
    top_ok = [p[1] for p in valid]
    n = len(valid)

    overlay = frame.copy()

    # ── 1. Curved wall: strip quads, each quad between sample i and i+1 ────
    for i in range(n - 1):
        quad = np.array([
            bot_ok[i], bot_ok[i + 1],
            top_ok[i + 1], top_ok[i],
        ], dtype=np.int32)

        # Shade by angle: mid-point angle gives lighting cue (0 = left/right edge, π/2 = front-centre)
        theta_mid = np.pi * (i + 0.5) / ARC_SAMPLES
        # sin(theta) → 1.0 at centre, 0.0 at edges.  Brighten centre, darken edges.
        t_shade = np.sin(theta_mid)               # 0..1
        r = int(ZONE_SHADE_COLOR[2] + t_shade * (ZONE_FACE_COLOR[2] - ZONE_SHADE_COLOR[2]))
        g = int(ZONE_SHADE_COLOR[1] + t_shade * (ZONE_FACE_COLOR[1] - ZONE_SHADE_COLOR[1]))
        b = int(ZONE_SHADE_COLOR[0] + t_shade * (ZONE_FACE_COLOR[0] - ZONE_SHADE_COLOR[0]))
        cv2.fillPoly(overlay, [quad], (b, g, r))

    # ── 2. Top cap disc ─────────────────────────────────────────────────────
    top_cap = np.array(top_ok, dtype=np.int32)
    cap_color = tuple(min(255, int(c * 1.15)) for c in ZONE_FACE_COLOR)  # slightly lighter
    cv2.fillPoly(overlay, [top_cap], cap_color)

    # ── 3. Bottom cap disc ──────────────────────────────────────────────────
    bot_cap = np.array(bot_ok, dtype=np.int32)
    floor_color = tuple(max(0, int(c * 0.75)) for c in ZONE_FACE_COLOR)  # slightly darker
    cv2.fillPoly(overlay, [bot_cap], floor_color)

    # ── 4. Flat back wall (the open rectangle at the door face) ─────────────
    if top_ok and bot_ok:
        back_wall = np.array([
            bot_ok[0], bot_ok[-1],
            top_ok[-1], top_ok[0],
        ], dtype=np.int32)
        back_color = tuple(max(0, int(c * 0.55)) for c in ZONE_FACE_COLOR)  # darkest face
        cv2.fillPoly(overlay, [back_wall], back_color)

    # Blend translucent fill.
    cv2.addWeighted(overlay, ZONE_FILL_ALPHA, frame, 1.0 - ZONE_FILL_ALPHA, 0, frame)

    # ── 5. Edges in bright red ───────────────────────────────────────────────
    edge_t = 2
    # Bottom arc
    cv2.polylines(frame, [np.array(bot_ok, dtype=np.int32)], isClosed=False,
                  color=ZONE_EDGE_COLOR, thickness=edge_t)
    # Top arc
    cv2.polylines(frame, [np.array(top_ok, dtype=np.int32)], isClosed=False,
                  color=ZONE_EDGE_COLOR, thickness=edge_t)
    # Left vertical edge
    if bot_ok and top_ok:
        cv2.line(frame, bot_ok[0],  top_ok[0],  ZONE_EDGE_COLOR, edge_t)
        cv2.line(frame, bot_ok[-1], top_ok[-1], ZONE_EDGE_COLOR, edge_t)
    # Vertical stripes along curved wall every ~4 samples for 3D grid cue
    for i in range(0, n, max(1, n // 8)):
        cv2.line(frame, bot_ok[i], top_ok[i], ZONE_EDGE_COLOR, 1)

    return frame


# ── Detection wrapper (YOLO + DINO + SAM door mask) ───────────────────────────

def _detect_door_mask(
    frame: np.ndarray,
    frame_idx: int,
    models: dict,
    state: dict,
    device: str,
) -> list[tuple]:
    """
    Run the same YOLO→DINO→SAM cadence as Door-exit_Detect, but only for doors.
    Returns the held SAM-refined door detections (each: x1,y1,x2,y2,conf,polygon|None).
    """
    yolo_door_boxes, _ = det.yolo_detect(models["yolo"], frame, device)

    run_dino_now = (frame_idx > 1) and ((frame_idx - 1) % DINO_INTERVAL == 0)
    if run_dino_now:
        dino_door_boxes, _ = det.dino_detect(models["proc"], models["dino"], frame, device)
        proposals = det._nms_merge(yolo_door_boxes + dino_door_boxes, iou_thr=0.30)
        proposals = sorted(proposals, key=lambda d: d[4], reverse=True)[:2]
        state["held_doors"] = (
            models["sam"].refine_boxes(frame, proposals, frame_id=frame_idx)
            if proposals else []
        )
        state["held_age"] = 0
    else:
        state["held_age"] += 1

    return state["held_doors"] if state["held_age"] <= HOLD_FRAMES else []


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(
    bag_dir: Path,
    radius: float,
    camera_height: float,
    max_frames: int,
    fps: float,
) -> Path:
    if not (bag_dir / "metadata.yaml").exists():
        raise FileNotFoundError(f"metadata.yaml not found in bag dir: {bag_dir}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = det._assert_gpu()

    # Floor / ceiling planes in camera Y (down = positive).
    # Double check this assignment in your main `run` function:
    floor_y = +abs(camera_height)

    # ── Aligned intrinsics ────────────────────────────────────────────────────
    intr = rd.read_intrinsics(bag_dir)
    print(f"[INTRINSICS] {intr}")
    print(f"[ZONE] radius={radius:.2f}m  (height from detected door)")

    # ── Load detection models ─────────────────────────────────────────────────
    print("[INIT] Loading YOLO ...")
    yolo_model = det.YOLO(str(det.YOLO_MODEL_PATH))
    yolo_model.to(device)

    print("[INIT] Loading Grounding DINO ...")
    processor = det.AutoProcessor.from_pretrained(det.DINO_MODEL_ID)
    dino_model = det.AutoModelForZeroShotObjectDetection.from_pretrained(det.DINO_MODEL_ID).to(device)
    dino_model.eval()

    print("[INIT] Loading SAM refiner ...")
    sam_refiner = det.GroundedSAMRefiner(
        ckpt_path=det.SAM_CKPT_PATH, model_type="vit_h", device=device
    )

    models = {"yolo": yolo_model, "proc": processor, "dino": dino_model, "sam": sam_refiner}
    state = {"held_doors": [], "held_age": HOLD_FRAMES + 1}

    # ── Outputs ───────────────────────────────────────────────────────────────
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_video = OUT_DIR / f"{bag_dir.name}_door_zone_{stamp}.mp4"
    out_csv = OUT_DIR / f"{bag_dir.name}_door_zone_{stamp}.csv"

    writer: cv2.VideoWriter | None = None
    csv_f = out_csv.open("w", newline="", encoding="utf-8")
    csv_w = csv.writer(csv_f)
    csv_w.writerow([
        "frame", "timestamp", "door_u", "door_v", "door_Z_m",
        "door_X_m", "door_Y_m", "radius_m", "door_height_m",
    ])

    frame_idx = 0
    door_frames = 0
    try:
        for f in rd.iter_rgbd_frames(bag_dir, max_frames=max_frames):
            frame_idx += 1
            frame = f.rgb
            h, w = frame.shape[:2]

            if writer is None:
                writer = cv2.VideoWriter(
                    str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                )

            # ── Door segmentation ────────────────────────────────────────────
            doors = _detect_door_mask(frame, frame_idx, models, state, device)

            door_u = door_v = door_Z = door_X = door_Y = ""
            door_height = ""
            if doors:
                # Use the highest-confidence door.
                best = max(doors, key=lambda d: d[4])
                mask = _mask_from_det(best, (h, w))
                if mask is not None:
                    centroid = _mask_centroid(mask)
                    z_m = _median_depth_m(f.depth_mm, mask)
                    if centroid is not None and z_m is not None and z_m > 0:
                        u, v = centroid
                        X, Y, Z = _backproject(u, v, z_m, intr)

                        # Extract door bounding box and compute door height.
                        x1, y1, x2, y2 = best[0], best[1], best[2], best[3]
                        door_pixel_height = abs(y2 - y1)
                        
                        # Back-project door top and bottom to get 3D height.
                        _, Y_top, _ = _backproject(u, y1, z_m, intr)      # top of door
                        _, Y_bottom, _ = _backproject(u, y2, z_m, intr)   # bottom of door
                        door_height_3d = abs(Y_bottom - Y_top)
                        
                        # Extrude from door center upward and downward by half the door height.
                        door_center_y = Y  # camera frame Y at door centroid
                        door_bottom_y = door_center_y + door_height_3d / 2   # downward (positive Y in camera frame)
                        door_top_y = door_center_y - door_height_3d / 2       # upward (negative Y in camera frame)

                        # Draw door mask outline.
                        poly = best[5] if len(best) > 5 else None
                        if poly is not None:
                            cv2.polylines(frame, [poly.astype(np.int32)], True, DOOR_POLY_COLOR, 2)

                        # Build + draw the 3D keep-clear half-cylinder zone.
                        door_xyz = (X, door_center_y, Z)   # footprint X/Z at door center
                        frame = _draw_zone(
                            frame, door_xyz, radius, door_bottom_y, door_top_y, intr
                        )

                        cv2.putText(
                            frame, f"door Z={Z:.2f}m  zone r={radius:.1f}m  h={door_height_3d:.2f}m",
                            (int(best[0]), max(20, int(best[1]) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, TXT_COLOR, 2, cv2.LINE_AA,
                        )

                        door_u, door_v, door_Z = u, v, round(Z, 3)
                        door_X, door_Y = round(X, 3), round(Y, 3)
                        door_height = round(door_height_3d, 3)
                        door_frames += 1

            csv_w.writerow([
                frame_idx, round(f.timestamp, 6), door_u, door_v, door_Z,
                door_X, door_Y, radius, door_height,
            ])

            cv2.putText(
                frame, f"f={frame_idx}  doors_with_zone={door_frames}",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA,
            )
            writer.write(frame)

            if frame_idx % 50 == 0:
                print(f"[RUN] frame {frame_idx}  doors_with_zone={door_frames}")

    finally:
        if writer is not None:
            writer.release()
        csv_f.close()

    print(f"\n[DONE] video : {out_video}")
    print(f"[DONE] csv   : {out_csv}")
    print(f"[DONE] frames: {frame_idx}  frames_with_zone: {door_frames}")
    return out_video


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Door keep-clear half-cylinder zone from an RGB-D ROS bag (YOLO+DINO+SAM)."
    )
    parser.add_argument("--bag", type=str, required=True, help="Path to bag folder (contains metadata.yaml)")
    parser.add_argument("--radius", type=float, default=1.0, help="Zone radius in metres (default 1.0)")
    parser.add_argument("--camera-height", type=float, default=1.2, help="Camera height above floor (m)")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N synced frames (0 = all)")
    parser.add_argument("--fps", type=float, default=25.0, help="Output video FPS")
    args = parser.parse_args()

    run(
        bag_dir=Path(args.bag),
        radius=float(args.radius),
        camera_height=float(args.camera_height),
        max_frames=max(0, int(args.max_frames)),
        fps=float(args.fps),
    )


if __name__ == "__main__":
    main()
