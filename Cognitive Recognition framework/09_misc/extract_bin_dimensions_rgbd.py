from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import cv2


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


def _parse_bbox(bbox_text: str):
    parts = [p.strip() for p in bbox_text.split(",")]
    if len(parts) != 4:
        raise ValueError("--bbox must be in format: x1,y1,x2,y2")
    x1, y1, x2, y2 = [float(v) for v in parts]
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid bbox: x2/y2 must be greater than x1/y1")
    return x1, y1, x2, y2


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


def _select_roi(frame_bgr):
    show = frame_bgr.copy()
    cv2.namedWindow("Select Bin ROI", cv2.WINDOW_NORMAL)
    x, y, w, h = cv2.selectROI("Select Bin ROI", show, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Bin ROI")
    if w <= 0 or h <= 0:
        raise RuntimeError("No ROI selected. Aborted.")
    return float(x), float(y), float(x + w), float(y + h)


def main():
    parser = argparse.ArgumentParser(
        description="Extract real-world bin dimensions from a specific RGBD frame."
    )
    parser.add_argument(
        "--bag",
        type=str,
        required=True,
        help="Path to ROS2 bag directory or .db3 file",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=230,
        help="Zero-based synced RGBD frame index (default: 230)",
    )
    parser.add_argument(
        "--bbox",
        type=str,
        default="",
        help="Optional bbox in pixels: x1,y1,x2,y2. If omitted, interactive ROI selection is used.",
    )
    parser.add_argument(
        "--save-debug-image",
        type=str,
        default="",
        help="Optional output image path to save the selected ROI visualization.",
    )
    parser.add_argument(
        "--save-frame-image",
        type=str,
        default="",
        help="Optional output image path to save the raw RGB frame before ROI selection.",
    )
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]

    rgbd_reader_path = project_root / "01_codebase" / "06_anomaly_detection" / "Blocked_exit_detection" / "RGBD_Reader.py"
    filter3d_path = project_root / "01_codebase" / "07_object_detection" / "rgbd_3d_filter.py"

    rgbd_reader = _load_module("rgbd_reader_module_for_bin_dims", rgbd_reader_path)
    filter3d = _load_module("rgbd_3d_filter_module_for_bin_dims", filter3d_path)

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
        raise RuntimeError(
            f"Unable to fetch frame index {args.frame_index} from any source.\n{detail}"
        )

    if args.save_frame_image:
        frame_out = Path(args.save_frame_image)
        frame_out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(frame_out), frame.rgb)
        print(f"Saved frame image     : {frame_out}")

    if args.bbox:
        x1, y1, x2, y2 = _parse_bbox(args.bbox)
    else:
        x1, y1, x2, y2 = _select_roi(frame.rgb)

    dims = filter3d.get_oriented_3d_dimensions(frame.depth_mm, x1, y1, x2, y2, intr)
    estimate_mode = "robust_pca"
    if dims is None:
        # Fallback: provide a rough estimate from sparse valid points.
        sparse_pts = filter3d._backproject_depth_points(
            frame.depth_mm,
            x1,
            y1,
            x2,
            y2,
            intr,
            min_points=5,
        )
        if sparse_pts is None:
            raise RuntimeError("Could not estimate dimensions from the selected ROI (insufficient valid depth).")
        X, Y, _ = sparse_pts
        width_m = float(max(0.0, float(X.max() - X.min())))
        height_m = float(max(0.0, float(Y.max() - Y.min())))
        dims = (width_m, height_m)
        estimate_mode = "rough_sparse"

    width_m, height_m = dims

    floor_plane = filter3d.estimate_floor_plane(frame.depth_mm, intr)
    floor_clearance = None
    if floor_plane is not None:
        floor_clearance = filter3d.estimate_box_floor_clearance(
            frame.depth_mm,
            x1,
            y1,
            x2,
            y2,
            intr,
            floor_plane,
        )

    print("\n" + "=" * 68)
    print("[RGBD BIN DIMENSIONS]")
    print(f"Source used           : {source_used}")
    print(f"Synced frame index    : {args.frame_index}")
    print(f"Frame timestamp (s)   : {frame.timestamp:.3f}")
    print(f"ROI (x1,y1,x2,y2)     : ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
    print(f"Estimate mode         : {estimate_mode}")
    print(f"Estimated width  (m)  : {width_m:.3f}")
    print(f"Estimated height (m)  : {height_m:.3f}")
    if floor_clearance is not None:
        print(f"Floor clearance  (m)  : {floor_clearance:.3f}")
    else:
        print("Floor clearance  (m)  : n/a (floor plane unavailable)")
    print("=" * 68 + "\n")

    if args.save_debug_image:
        dbg = frame.rgb.copy()
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(dbg, p1, p2, (0, 255, 255), 2)
        cv2.putText(
            dbg,
            f"W={width_m:.2f}m H={height_m:.2f}m",
            (p1[0], max(15, p1[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        out_path = Path(args.save_debug_image)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), dbg)
        print(f"Saved debug image     : {out_path}")


if __name__ == "__main__":
    main()
