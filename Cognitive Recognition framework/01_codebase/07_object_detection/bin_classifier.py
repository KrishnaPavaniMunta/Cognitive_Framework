from __future__ import annotations

import cv2
import numpy as np

PRESERVE_CANONICAL_BIN_LABEL = False


def _clip_box(frame, box_coords):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in box_coords]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def classify_bin_type(frame, box_coords):
    """Classify a detected bin ROI into general_bin, yellow_bin, or bin_tiger_stripe."""
    clipped = _clip_box(frame, box_coords)
    if clipped is None:
        return "general_bin"

    x1, y1, x2, y2 = clipped
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return "general_bin"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    yellow_lower = np.array([15, 70, 70], dtype=np.uint8)
    yellow_upper = np.array([45, 255, 255], dtype=np.uint8)
    black_lower = np.array([0, 0, 0], dtype=np.uint8)
    black_upper = np.array([180, 255, 80], dtype=np.uint8)

    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    black_mask = cv2.inRange(hsv, black_lower, black_upper)

    kernel = np.ones((3, 3), np.uint8)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)

    crop_area = float(crop.shape[0] * crop.shape[1])
    if crop_area <= 0:
        return "general_bin"

    yellow_percentage = float(cv2.countNonZero(yellow_mask)) / crop_area
    print(f"[DEBUG] yellow_percentage = {yellow_percentage:.2f}")
    if yellow_percentage < 0.10:
        return "general_bin"

    contours_info = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
    if not contours:
        return "yellow_bin" if yellow_percentage > 0.40 else "general_bin"

    height, width = crop.shape[:2]
    min_stripe_area = max(12.0, crop_area * 0.0015)
    max_stripe_area = crop_area * 0.12
    top_lid_area = crop_area * 0.08
    stripe_candidates = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_stripe_area or area > max_stripe_area:
            continue

        x, y, contour_width, contour_height = cv2.boundingRect(contour)
        center_y = y + (contour_height / 2.0)

        # Ignore the big black lid contour near the top of the bin.
        if y < int(height * 0.30) and area > top_lid_area:
            continue

        if center_y < height * 0.22 or center_y > height * 0.90:
            continue

        stripe_candidates.append((x, y, contour_width, contour_height, area, center_y))
        print(f"[DEBUG] stripe_candidates found: {len(stripe_candidates)}")
    if not stripe_candidates:
        return "yellow_bin" if yellow_percentage > 0.40 else "general_bin"

    mid_band_candidates = [item for item in stripe_candidates if height * 0.28 <= item[5] <= height * 0.82]
    combined_area = sum(item[4] for item in mid_band_candidates)
    xs = [item[0] + (item[2] / 2.0) for item in mid_band_candidates]

    if len(mid_band_candidates) >= 2 and combined_area / crop_area >= 0.006:
        if max(xs) - min(xs) >= width * 0.12:
            return "bin_tiger_stripe"
        return "bin_tiger_stripe"

    if len(stripe_candidates) >= 3 and combined_area / crop_area >= 0.01:
        return "bin_tiger_stripe"

    return "yellow_bin" if yellow_percentage > 0.40 else "general_bin"


def refine_bin_detections(frame, detections):
    refined = []
    for x1, y1, x2, y2, conf, name in detections:
        base_name = str(name).replace("[DINO] ", "")
        if base_name == "bin":
            refined_name = classify_bin_type(frame, (x1, y1, x2, y2))
            if PRESERVE_CANONICAL_BIN_LABEL:
                refined_name = "bin"
            refined.append((x1, y1, x2, y2, conf, refined_name))
            continue
        refined.append((x1, y1, x2, y2, conf, name))
    return refined
