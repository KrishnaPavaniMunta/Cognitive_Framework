import importlib.util
import sys
from datetime import datetime
from pathlib import Path

import cv2
from PIL import Image

from bin_classifier import refine_bin_detections


def _load_rgbd_reader(project_root):
    reader_path = (
        Path(project_root)
        / "01_codebase"
        / "06_anomaly_detection"
        / "Blocked_exit_detection"
        / "RGBD_Reader.py"
    )
    if not reader_path.exists():
        raise FileNotFoundError(f"RGBD reader not found: {reader_path}")

    spec = importlib.util.spec_from_file_location("rgbd_reader_module", str(reader_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load RGBD reader module from {reader_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
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


def process_rgbd_bag(
    v1,
    v2,
    v3,
    bag_path,
    max_frames,
    *,
    project_root,
    out_dir,
    dino_video_interval_frames,
    dino_hold_frames,
    dino_seen_conf_thresh,
    iou_thresh,
    dino_fallback,
    run_yolo_ensemble,
    run_dino_fallback,
    apply_global_nms,
    apply_common_sense_rules,
    apply_depth_size_filter,
    draw_predictions,
):
    rgbd_reader = _load_rgbd_reader(project_root)

    bag_input = Path(bag_path)
    if bag_input.is_file():
        bag_label = bag_input.stem
    else:
        bag_label = bag_input.name

    if bag_input.is_dir() and not (bag_input / "metadata.yaml").exists():
        raise FileNotFoundError(f"metadata.yaml not found in bag dir: {bag_input}")

    candidate_sources = _candidate_bag_sources(bag_input)
    if not candidate_sources:
        raise FileNotFoundError(f"No readable bag sources found for: {bag_input}")

    intr = None
    active_source = None
    source_errors = []
    for candidate in candidate_sources:
        try:
            intr = rgbd_reader.read_intrinsics(candidate)
            active_source = candidate
            break
        except Exception as exc:
            source_errors.append(f"{candidate.name}: {exc}")

    if intr is None or active_source is None:
        error_lines = "\n".join(f" - {msg}" for msg in source_errors)
        raise RuntimeError(
            "Unable to read RGBD intrinsics from any candidate bag source:\n"
            f"{error_lines}"
        )

    print(f"[RGBD] Intrinsics: fx={intr.fx:.2f}, fy={intr.fy:.2f}, cx={intr.cx:.2f}, cy={intr.cy:.2f}")

    out_path = Path(out_dir) / f"out_rgbd_{datetime.now().strftime('%H%M%S')}_{bag_label}.mp4"
    writer = None
    cv2.namedWindow("RGBD Inference (Resizable)", cv2.WINDOW_NORMAL)

    frame_idx = 0
    held_dino_detections = []
    held_dino_age = dino_hold_frames + 1

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
                raise RuntimeError(f"RGBD frame stream failed for {Path(primary).name}: {exc}") from exc
            print(f"[RGBD] Warning: frame stream failed from {Path(primary).name}: {exc}")
            print(f"[RGBD] Retrying stream with alternate source: {Path(secondary).name}")
            try:
                for frm in rgbd_reader.iter_rgbd_frames(
                    secondary,
                    max_time_diff=rgbd_reader.MAX_TIME_DIFF,
                    max_frames=max_frames,
                ):
                    yield frm
            except Exception as retry_exc:
                raise RuntimeError(
                    f"RGBD frame stream failed for both {Path(primary).name} and {Path(secondary).name}: {retry_exc}"
                ) from retry_exc

    secondary_source = next((src for src in candidate_sources if src != active_source), None)

    for frame in _iter_frames_with_fallback(active_source, secondary_source):
        frame_idx += 1
        rgb_frame = frame.rgb.copy()
        depth_mm = frame.depth_mm

        preds = run_yolo_ensemble(v1, v2, v3, rgb_frame)

        run_dino_now = frame_idx > 1 and ((frame_idx - 1) % dino_video_interval_frames == 0)

        if run_dino_now:
            pil_img = Image.fromarray(cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB))
            seen_classes = {p[5].replace("[DINO] ", "") for p in preds if p[4] >= dino_seen_conf_thresh}
            missing_targets = [c for c in dino_fallback.keys() if c not in seen_classes]
            dino_preds = run_dino_fallback(pil_img, missing_targets)
            held_dino_detections = dino_preds
            held_dino_age = 0
        else:
            held_dino_age += 1

        if held_dino_age <= dino_hold_frames:
            preds.extend(held_dino_detections)

        final_dets = apply_global_nms(preds, iou_thresh)
        frame_height = rgb_frame.shape[0]
        frame_width = rgb_frame.shape[1]
        final_dets = apply_common_sense_rules(final_dets, frame_height, frame_width)
        final_dets = refine_bin_detections(rgb_frame, final_dets)
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
    if frame_idx == 0:
        print("[WARNING] RGBD bag processing finished with 0 synchronized frames.")
        print(" -> Bag stream may be heavily corrupted or topics/timestamps are mismatched.")
    else:
        print("[SUCCESS] RGBD bag rendering complete!")
        print(f" -> Processed synced frames: {frame_idx}")
        print(f" -> Saved to: {out_path.resolve()}")
    print("═" * 70 + "\n")
