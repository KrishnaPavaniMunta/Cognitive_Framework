from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _candidate_bag_sources(bag_input: Path):
    candidates = []

    if bag_input.is_file():
        candidates.append(bag_input)
        bag_dir = bag_input.parent
    else:
        bag_dir = bag_input

    if bag_dir.exists():
        if bag_dir not in candidates:
            candidates.append(bag_dir)

        recovered_default = bag_dir / f"{bag_dir.name}_recovered.db3"
        if recovered_default.exists() and recovered_default not in candidates:
            candidates.insert(0, recovered_default)

        for recovered_candidate in sorted(bag_dir.glob("*_recovered.db3")):
            if recovered_candidate not in candidates:
                candidates.append(recovered_candidate)

    return candidates


def _choose_source_and_intrinsics(rgbd_reader, bag_input: Path):
    candidates = _candidate_bag_sources(bag_input)
    if not candidates:
        raise FileNotFoundError(f"No readable bag sources found for: {bag_input}")

    source_errors = []
    for candidate in candidates:
        try:
            intr = rgbd_reader.read_intrinsics(candidate)
            return candidate, intr, candidates
        except Exception as exc:
            source_errors.append(f"{candidate.name}: {exc}")

    msg = "\n".join(f" - {err}" for err in source_errors)
    raise RuntimeError(f"Unable to read intrinsics from any source:\n{msg}")


def _get_frame_by_index(rgbd_reader, source, frame_index: int):
    for i, frame in enumerate(
        rgbd_reader.iter_rgbd_frames(
            source,
            max_time_diff=rgbd_reader.MAX_TIME_DIFF,
            max_frames=frame_index + 1,
        )
    ):
        if i == frame_index:
            return frame
    return None


def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_yolo_models(detector_module, device: str):
    v1 = YOLO(str(detector_module.V1_PATH)).to(device)
    v2 = YOLO(str(detector_module.V2_PATH)).to(device)
    v3 = YOLO(str(detector_module.V3_PATH)).to(device)
    return v1, v2, v3


def _estimate_box_dimensions_3d(filter3d, depth_mm, x1, y1, x2, y2, intrinsics):
    points = filter3d._backproject_depth_points(depth_mm, x1, y1, x2, y2, intrinsics, min_points=8)
    if points is None:
        return None

    X, Y, Z = points
    points3d = np.column_stack((X, Y, Z))
    if len(points3d) < 8:
        return None

    median_z = float(np.median(Z))
    std_z = float(np.std(Z))
    z_threshold = max(1.5 * std_z, 0.5)
    inlier_mask = np.abs(Z - median_z) < z_threshold

    Xf = X[inlier_mask]
    Yf = Y[inlier_mask]
    Zf = Z[inlier_mask]
    if len(Xf) < 8:
        Xf, Yf, Zf = X, Y, Z

    height_m = float(max(0.0, float(np.max(Yf) - np.min(Yf))))

    pts_floor = np.column_stack((Xf, Zf))
    if len(pts_floor) >= 2:
        pts_centered = pts_floor - np.mean(pts_floor, axis=0)
        cov = np.cov(pts_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, order]
        pts_projected = pts_centered @ eigenvectors
        extents = np.max(pts_projected, axis=0) - np.min(pts_projected, axis=0)
        length_m = float(max(0.0, float(np.max(extents))))
        width_m = float(max(0.0, float(np.min(extents))))
    else:
        length_m = float(max(0.0, float(np.max(Zf) - np.min(Zf))))
        width_m = float(max(0.0, float(np.max(Xf) - np.min(Xf))))

    depth_span_m = float(max(0.0, float(np.max(Zf) - np.min(Zf))))
    center_point = {
        "x_m": float(np.median(Xf)),
        "y_m": float(np.median(Yf)),
        "z_m": float(np.median(Zf)),
    }

    return {
        "length_m": length_m,
        "width_m": width_m,
        "height_m": height_m,
        "depth_span_m": depth_span_m,
        "n_valid_points": int(len(Xf)),
        "center_point": center_point,
    }


def _annotate(frame_bgr, detections_with_dims):
    annotated = frame_bgr.copy()
    for det in detections_with_dims:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(annotated, p1, p2, (0, 255, 255), 2)

        label = f"{det['class_name']} {det['confidence']:.1%}"
        dims = det.get("dimensions_3d")
        dim_text = "no depth"
        if dims is not None:
            dim_text = (
                f"L={dims['length_m']:.2f}m W={dims['width_m']:.2f}m H={dims['height_m']:.2f}m"
            )

        lines = [label, dim_text]
        y_text = max(18, p1[1] - 22)
        for idx, text in enumerate(lines):
            cv2.putText(
                annotated,
                text,
                (p1[0], y_text + idx * 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
    return annotated


def main():
    parser = argparse.ArgumentParser(
        description="Extract bounding boxes and RGBD object dimensions for all detections in one frame."
    )
    parser.add_argument("--bag", type=str, required=True, help="Path to ROS2 bag directory or .db3 file")
    parser.add_argument("--frame-index", type=int, default=230, help="Zero-based synced RGBD frame index")
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional JSON output path. Defaults to 09_misc/extractobject_dimensions_frame<idx>.json",
    )
    parser.add_argument(
        "--output-image",
        type=str,
        default="",
        help="Optional annotated image output path. Defaults to 09_misc/extractobject_dimensions_frame<idx>.jpg",
    )
    parser.add_argument(
        "--apply-depth-filter",
        action="store_true",
        help="Also apply the RGBD physical-size filter before reporting detections.",
    )
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]

    rgbd_reader_path = project_root / "01_codebase" / "06_anomaly_detection" / "Blocked_exit_detection" / "RGBD_Reader.py"
    filter3d_path = project_root / "01_codebase" / "07_object_detection" / "rgbd_3d_filter.py"
    detector_path = project_root / "01_codebase" / "07_object_detection" / "YOLO_ensemble+DINO.py"

    rgbd_reader = _load_module("rgbd_reader_module_for_object_dims", rgbd_reader_path)
    filter3d = _load_module("rgbd_3d_filter_module_for_object_dims", filter3d_path)
    detector = _load_module("yolo_ensemble_module_for_object_dims", detector_path)

    bag_input = Path(args.bag)
    active_source, intr, all_sources = _choose_source_and_intrinsics(rgbd_reader, bag_input)

    frame = None
    source_used = None
    source_errors = []
    for source in [active_source] + [s for s in all_sources if s != active_source]:
        try:
            frame = _get_frame_by_index(rgbd_reader, source, args.frame_index)
            if frame is not None:
                source_used = source
                break
        except Exception as exc:
            source_errors.append(f"{source.name}: {exc}")

    if frame is None:
        detail = "\n".join(f" - {err}" for err in source_errors)
        raise RuntimeError(f"Unable to fetch frame index {args.frame_index} from any source.\n{detail}")

    device = _get_device()
    print(f"[INFO] Loading YOLO ensemble on {device}...")
    v1, v2, v3 = _load_yolo_models(detector, device)

    preds = detector.run_yolo_ensemble(v1, v2, v3, frame.rgb)
    preds = detector.apply_global_nms(preds, detector.IOU_THRESH)
    frame_height, frame_width = frame.rgb.shape[:2]
    preds = detector.apply_common_sense_rules(preds, frame_height, frame_width)
    preds = detector.refine_bin_detections(frame.rgb, preds)

    if args.apply_depth_filter:
        preds = detector.apply_depth_size_filter(preds, frame.depth_mm, intr)

    results = []
    for x1, y1, x2, y2, conf, name in preds:
        dims = _estimate_box_dimensions_3d(filter3d, frame.depth_mm, x1, y1, x2, y2, intr)
        results.append(
            {
                "class_name": str(name),
                "confidence": float(conf),
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "dimensions_3d": dims,
            }
        )

    output_json = Path(args.output_json) if args.output_json else script_path.parent / f"extractobject_dimensions_frame{args.frame_index}.json"
    output_image = Path(args.output_image) if args.output_image else script_path.parent / f"extractobject_dimensions_frame{args.frame_index}.jpg"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_image.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "bag_source": str(source_used),
        "frame_index": int(args.frame_index),
        "timestamp_s": float(frame.timestamp),
        "image_width": int(frame.rgb.shape[1]),
        "image_height": int(frame.rgb.shape[0]),
        "detections": results,
    }

    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    annotated = _annotate(frame.rgb, results)
    cv2.imwrite(str(output_image), annotated)

    print("\n" + "=" * 72)
    print("[EXTRACT OBJECT DIMENSIONS]")
    print(f"Source used      : {source_used}")
    print(f"Frame index      : {args.frame_index}")
    print(f"Timestamp (s)    : {frame.timestamp:.3f}")
    print(f"Detections found : {len(results)}")
    for idx, det in enumerate(results, start=1):
        dims = det.get("dimensions_3d")
        bbox = det["bbox_xyxy"]
        print(f"{idx:02d}. {det['class_name']}  conf={det['confidence']:.1%}  bbox={bbox}")
        if dims is None:
            print("    dims: unavailable (insufficient valid depth)")
        else:
            print(
                "    dims: "
                f"L={dims['length_m']:.3f}m "
                f"W={dims['width_m']:.3f}m "
                f"H={dims['height_m']:.3f}m "
                f"depth_span={dims['depth_span_m']:.3f}m "
                f"points={dims['n_valid_points']}"
            )
    print(f"JSON saved       : {output_json}")
    print(f"Image saved      : {output_image}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
