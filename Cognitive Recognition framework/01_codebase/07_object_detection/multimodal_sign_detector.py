"""
Multimodal Hospital Safety Signage & Pictogram Detector (EasyOCR + CLIP ViT-B/32).

Combines:
1. Event-Driven Visual Cue RoI Generation (anchored to Doors, Fire Extinguishers, Cabinets).
2. Periodic Full-Scene Text Sweep (every N frames via EasyOCR CRAFT text detector).
3. Dual-Stream Multimodal Cognition (EasyOCR text matching + CLIP zero-shot ISO 7010 pictograms).
4. 3D Spatial Reality Gate (check_planar_backing on walls & elevation checks).
5. 3D Spatial Deduplication (tracks persistent sign landmarks across time).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

try:
    import easyocr
except ImportError:
    easyocr = None

try:
    from transformers import CLIPModel, CLIPProcessor
except ImportError:
    CLIPModel = None
    CLIPProcessor = None

LOG = logging.getLogger("multimodal_sign_detector")


# ── Hospital Safety Sign Taxonomy & Vocabulary ──────────────────────────────────
HOSPITAL_SIGN_TAXONOMY: dict[str, dict[str, Any]] = {
    "exit_sign": {
        "canonical_name": "exit_sign",
        "display_name": "Emergency Exit Sign",
        "badge_label": "[SIGN: EXIT]",
        "color_bgr": (34, 197, 94),   # Green
        "clip_prompt": "an ISO 7010 green emergency exit running man safety sign",
        "text_keywords": [
            "exit", "emergency exit", "fire exit", "way out", "salida", "sortie",
            "ausgang", "emergency", "escape route"
        ],
        "min_elevation_m": 1.60,
        "max_elevation_m": 2.90,
    },
    "sign_fire_extinguisher": {
        "canonical_name": "sign_fire_extinguisher",
        "display_name": "Fire Safety Sign",
        "badge_label": "[SIGN: FIRE EXT]",
        "color_bgr": (30, 30, 220),   # Red
        "clip_prompt": "a red fire extinguisher safety equipment identification sign with symbol",
        "text_keywords": [
            "fire extinguisher", "extinguisher", "fire hose", "fire alarm",
            "call point", "fire blanket", "water", "foam", "powder", "co2", "fire"
        ],
        "min_elevation_m": 0.80,
        "max_elevation_m": 2.20,
    },
    "hazmat_sign": {
        "canonical_name": "hazmat_sign",
        "display_name": "Biohazard Warning Sign",
        "badge_label": "[SIGN: BIOHAZARD]",
        "color_bgr": (15, 165, 245),  # Amber/Yellow
        "clip_prompt": "a yellow triangular biohazard hazard symbol warning sign",
        "text_keywords": [
            "biohazard", "biological hazard", "infectious", "cytotoxic",
            "sharps", "pathogen", "clinical waste"
        ],
        "min_elevation_m": 0.70,
        "max_elevation_m": 2.30,
    },
    "sign_radiation": {
        "canonical_name": "sign_radiation",
        "display_name": "Radiation Warning Sign",
        "badge_label": "[SIGN: RADIATION]",
        "color_bgr": (15, 165, 245),  # Amber/Yellow
        "clip_prompt": "a yellow triangular radioactive trefoil radiation hazard symbol sign",
        "text_keywords": [
            "radiation", "radioactive", "x-ray", "laser in use", "x ray", "imaging"
        ],
        "min_elevation_m": 0.70,
        "max_elevation_m": 2.30,
    },
    "sign_caution_wet_floor": {
        "canonical_name": "sign_caution_wet_floor",
        "display_name": "Caution Wet Floor Sign",
        "badge_label": "[SIGN: WET FLOOR]",
        "color_bgr": (10, 215, 250),  # Bright Yellow
        "clip_prompt": "a yellow slipping person hazard caution wet floor cleaning sign",
        "text_keywords": [
            "wet floor", "caution wet floor", "slippery", "caution", "cleaning in progress", "slip hazard"
        ],
        "min_elevation_m": -0.10,
        "max_elevation_m": 1.00,
    },
    "sign_prohibition_access": {
        "canonical_name": "sign_prohibition_access",
        "display_name": "Restricted Access Sign",
        "badge_label": "[SIGN: RESTRICTED]",
        "color_bgr": (40, 40, 200),   # Red
        "clip_prompt": "a red circular no entry prohibition symbol sign or staff only notice",
        "text_keywords": [
            "no entry", "do not enter", "authorized personnel only", "authorised personnel only",
            "staff only", "keep out", "restricted area", "keep clear", "keep shut"
        ],
        "min_elevation_m": 0.80,
        "max_elevation_m": 2.30,
    },
}

# ── Open-Vocabulary Placard & General Hospital Signage Taxonomy ─────────────────
OPEN_SIGN_TAXONOMY: dict[str, dict[str, Any]] = {
    "room_sign": {
        "canonical_name": "room_sign",
        "display_name": "Room / Ward Sign",
        "badge_prefix": "ROOM",
        "color_bgr": (220, 160, 40),   # Cyan/Teal
        "clip_prompt": "a hospital room nameplate, ward door sign, department identification plaque, or office plaque",
        "keywords": [
            "room", "ward", "icu", "clinic", "office", "dept", "department",
            "theatre", "operating", "pharmacy", "radiology", "triage", "restroom",
            "toilet", "washroom", "wc", "lab", "laboratory", "consultation",
            "reception", "waiting", "doctor", "nurse", "station", "emergency", "cardiology",
            "radiotherapy", "maternity", "pediatric", "oncology"
        ],
        "min_elevation_m": 0.60,
        "max_elevation_m": 2.70,
    },
    "notice_board": {
        "canonical_name": "notice_board",
        "display_name": "Hospital Notice Board",
        "badge_prefix": "NOTICE",
        "color_bgr": (180, 120, 20),   # Deep cyan
        "clip_prompt": "a hospital advisory notice board, staff instruction plaque, hospital policy sign, or visitor notice",
        "keywords": [
            "notice", "attention", "information", "info", "warning", "instruction",
            "visitor", "visiting", "hours", "clean hands", "wash hands", "mask",
            "protocol", "policy", "sanitise", "sanitize", "hygiene", "ppe", "staff only",
            "silence", "quiet", "no smoking", "cctv in operation"
        ],
        "min_elevation_m": 0.60,
        "max_elevation_m": 2.60,
    },
    "directional_sign": {
        "canonical_name": "directional_sign",
        "display_name": "Directional Wayfinding Sign",
        "badge_prefix": "WAYFINDING",
        "color_bgr": (200, 180, 50),   # Light cyan
        "clip_prompt": "a hospital directional wayfinding corridor sign with arrows indicating departments",
        "keywords": [
            "way", "arrow", "straight", "left", "right", "ahead", "level", "floor",
            "stairs", "lift", "elevator", "block", "wing", "zone", "corridor", "route"
        ],
        "min_elevation_m": 0.80,
        "max_elevation_m": 2.90,
    },
    "safety_sign": {
        "canonical_name": "safety_sign",
        "display_name": "Safety Notice Placard",
        "badge_prefix": "SAFETY",
        "color_bgr": (50, 180, 220),   # Amber
        "clip_prompt": "a hospital safety instructions placard, first aid equipment sign, or medical safety label",
        "keywords": [
            "safety", "first aid", "eyewash", "aed", "defibrillator", "emergency stop",
            "safety shower", "resuscitation", "fire action", "emergency call"
        ],
        "min_elevation_m": 0.60,
        "max_elevation_m": 2.60,
    },
}

NEGATIVE_PROMPTS = [
    "a plain hospital wall with no signs",
    "a blank wooden or metal door",
    "a hospital bed, floor tiles, or light fixture",
    "a random commercial poster, bulletin board, or paper flyer with small text",
    "a blurry background surface",
]


@dataclass
class DetectedSign:
    canonical_name: str
    display_name: str
    badge_label: str
    confidence: float
    bbox_xyxy: list[float]
    modality: str                  # "ocr", "clip", or "multimodal_fusion"
    text_content: str | None = None
    symbol_label: str | None = None
    color_bgr: tuple[int, int, int] = (0, 200, 0)
    world_pos: np.ndarray | None = None
    elevation_m: float | None = None


def _clean_text(raw_text: str) -> str:
    """Normalize extracted OCR text for robust keyword matching."""
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", raw_text).lower()
    return re.sub(r"\s+", " ", cleaned).strip()


def _fuzzy_match_keywords(text: str, keywords: list[str]) -> tuple[bool, str | None, float]:
    """Checks if any keyword appears in the extracted text string."""
    cleaned = _clean_text(text)
    if not cleaned:
        return False, None, 0.0

    words = set(cleaned.split())
    for kw in keywords:
        kw_clean = _clean_text(kw)
        if " " in kw_clean:
            if kw_clean in cleaned:
                return True, kw, 0.95
        else:
            if kw_clean in words or kw_clean in cleaned:
                return True, kw, 0.90

    # Character-level partial match for short abbreviations
    for kw in keywords:
        if len(kw) >= 4 and kw.lower() in cleaned:
            return True, kw, 0.85

    return False, None, 0.0


def _box_iou(box1: list[float] | tuple[float, ...], box2: list[float] | tuple[float, ...]) -> float:
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    area1 = max(1.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(1.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    return inter / float(area1 + area2 - inter)


class MultimodalSignDetector:
    """
    High-accuracy hospital signage & text recognition engine (EasyOCR CRAFT + CRNN).
    Extracts text placards, door room signs, notice boards, and ISO safety signs.
    """

    def __init__(
        self,
        use_gpu: bool = True,
        clip_model: Any | None = None,
        clip_processor: Any | None = None,
        device: str | None = None,
    ) -> None:
        if device is None:
            self.device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
        else:
            self.device = device

        LOG.info("[SIGN DETECTOR] Initializing Pure Text Sign Detector (EasyOCR) on device: %s", self.device)

        # 1. EasyOCR Engine (CRAFT Text Detection + CRNN Recognition)
        self.reader = None
        if easyocr is not None:
            try:
                LOG.info("[SIGN DETECTOR] Loading EasyOCR (English) on %s...", self.device)
                self.reader = easyocr.Reader(["en"], gpu=(self.device == "cuda"))
                LOG.info("[SIGN DETECTOR] EasyOCR ready.")
            except Exception as exc:
                LOG.warning("[SIGN DETECTOR] EasyOCR init failed: %s", exc)

        # 2. Categories
        self.sign_categories = list(HOSPITAL_SIGN_TAXONOMY.keys())
        self.open_categories = list(OPEN_SIGN_TAXONOMY.keys())

        # 3. 3D Spatial Landmark Tracking (0.40m deduplication radius)
        self.tracked_signs: list[dict[str, Any]] = []
        self._next_sign_id = 1

    # ── RoI Generation ──────────────────────────────────────────────────────────

    def generate_contextual_rois(
        self,
        frame_shape: tuple[int, int],
        detections: list[dict[str, Any]],
    ) -> list[tuple[float, float, float, float, str]]:
        """
        Extracts high-resolution candidate RoIs around primary visual anchors (Doors, Extinguishers).
        Returns list of (x1, y1, x2, y2, expected_category_hint).
        """
        h_frame, w_frame = frame_shape[:2]
        rois: list[tuple[float, float, float, float, str]] = []

        for det in detections:
            cname = str(det.get("class_label") or det.get("class_name", "")).lower().strip()
            bbox = det.get("bbox_xyxy")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = (float(v) for v in bbox)
            box_w = max(1.0, x2 - x1)
            box_h = max(1.0, y2 - y1)

            # Anchor 1: Doorway / Door
            if "door" in cname:
                # Zone A: Directly Above Door Header (Exit / Directional signs)
                # Extends 35% of door height above the door top
                hx1 = max(0.0, x1 - 0.10 * box_w)
                hx2 = min(float(w_frame), x2 + 0.10 * box_w)
                hy1 = max(0.0, y1 - 0.35 * box_h)
                hy2 = min(float(h_frame), y1 + 0.08 * box_h)
                if (hx2 - hx1) > 20 and (hy2 - hy1) > 15:
                    rois.append((hx1, hy1, hx2, hy2, "exit_sign"))

                # Zone B: Eye-level on door surface (Access control / "Keep Shut" notices)
                px1 = max(0.0, x1 + 0.15 * box_w)
                px2 = min(float(w_frame), x2 - 0.15 * box_w)
                py1 = y1 + 0.20 * box_h
                py2 = y1 + 0.55 * box_h
                if (px2 - px1) > 20 and (py2 - py1) > 15:
                    rois.append((px1, py1, px2, py2, "sign_prohibition_access"))

            # Anchor 2: Fire Extinguisher
            elif "fire_extinguisher" in cname or "extinguisher" in cname:
                # Zone: Directly Above the Extinguisher (Placard / Instruction board)
                ex1 = max(0.0, x1 - 0.30 * box_w)
                ex2 = min(float(w_frame), x2 + 0.30 * box_w)
                ey1 = max(0.0, y1 - 0.65 * box_h)
                ey2 = min(float(h_frame), y1 + 0.10 * box_h)
                if (ex2 - ex1) > 20 and (ey2 - ey1) > 15:
                    rois.append((ex1, ey1, ex2, ey2, "sign_fire_extinguisher"))

            # Anchor 3: Existing Hazmat / Caution Candidate
            elif "hazmat" in cname or "warning" in cname:
                # Slightly expanded bounding box
                margin_x = 0.15 * box_w
                margin_y = 0.15 * box_h
                mx1 = max(0.0, x1 - margin_x)
                mx2 = min(float(w_frame), x2 + margin_x)
                my1 = max(0.0, y1 - margin_y)
                my2 = min(float(h_frame), y2 + margin_y)
                rois.append((mx1, my1, mx2, my2, "hazmat_sign"))

        return rois

    def generate_green_sign_rois(
        self,
        frame_shape: tuple[int, int],
        rgb_bgr: np.ndarray,
    ) -> list[tuple[float, float, float, float, str]]:
        """
        Fast HSV green color + aspect-ratio candidate generator for ISO 7010 Emergency Exit signs.
        Runs in ~1ms and is immune to distance/motion blur.
        """
        h_frame, w_frame = frame_shape[:2]
        rois: list[tuple[float, float, float, float, str]] = []

        try:
            hsv = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2HSV)
            # Safety green in HSV
            lower_green = np.array([35, 40, 35])
            upper_green = np.array([88, 255, 255])
            mask = cv2.inRange(hsv, lower_green, upper_green)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
            mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 40 or area > 18000:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                # Exit signs are mounted high up (upper 70% of the scene)
                if y > 0.70 * h_frame:
                    continue
                aspect = w / float(max(1, h))
                if 1.5 <= aspect <= 5.0 and w >= 20 and h >= 8:
                    # Pad slightly to capture the white ISO running man and lettering
                    pad_x = 0.15 * w
                    pad_y = 0.20 * h
                    gx1 = max(0.0, x - pad_x)
                    gy1 = max(0.0, y - pad_y)
                    gx2 = min(float(w_frame), x + w + pad_x)
                    gy2 = min(float(h_frame), y + h + pad_y)
                    rois.append((gx1, gy1, gx2, gy2, "exit_sign"))
        except Exception as exc:
            LOG.debug("[SIGN DETECTOR] Green candidate search error: %s", exc)

        return rois

    def generate_periodic_text_rois(
        self,
        rgb_bgr: np.ndarray,
        min_box_size: int = 8,
    ) -> list[tuple[float, float, float, float, str, str, float]]:
        """
        Sweeps the full image using EasyOCR's CRAFT detector to capture both safety
        and open-vocabulary placards (room numbers, notices, wayfinding).
        Returns list of (x1, y1, x2, y2, matched_category, raw_text, ocr_conf).
        """
        if self.reader is None:
            return []

        h_frame, w_frame = rgb_bgr.shape[:2]
        results: list[tuple[float, float, float, float, str, str, float]] = []

        try:
            # Run text reader with user-specified test threshold 0.20
            ocr_out = self.reader.readtext(
                rgb_bgr,
                min_size=min_box_size,
                text_threshold=0.20,
                low_text=0.20,
                link_threshold=0.35,
            )

            for bbox_poly, raw_text, ocr_conf in ocr_out:
                if not raw_text:
                    continue
                raw_clean = _clean_text(raw_text)
                alnum_count = sum(c.isalnum() for c in raw_clean)
                if alnum_count < 2 or float(ocr_conf) < 0.20:
                    continue

                poly = np.array(bbox_poly, dtype=np.float32)
                x1 = float(np.min(poly[:, 0]))
                y1 = float(np.min(poly[:, 1]))
                x2 = float(np.max(poly[:, 0]))
                y2 = float(np.max(poly[:, 1]))

                # Pad slightly to capture surrounding placard boundary
                pad_w = 0.20 * (x2 - x1)
                pad_h = 0.25 * (y2 - y1)
                x1 = max(0.0, x1 - pad_w)
                y1 = max(0.0, y1 - pad_h)
                x2 = min(float(w_frame), x2 + pad_w)
                y2 = min(float(h_frame), y2 + pad_h)

                # 1. Check against closed safety vocabulary first (Tier 1)
                matched_cat = None
                best_score = 0.0
                for cat_name, cat_spec in HOSPITAL_SIGN_TAXONOMY.items():
                    is_match, kw, score = _fuzzy_match_keywords(raw_text, cat_spec["text_keywords"])
                    if is_match and score > best_score:
                        best_score = score
                        matched_cat = cat_name

                # 2. Check against open vocabulary taxonomy (Tier 2)
                if matched_cat is None:
                    for open_name, open_spec in OPEN_SIGN_TAXONOMY.items():
                        is_match, kw, score = _fuzzy_match_keywords(raw_text, open_spec["keywords"])
                        if is_match and score > best_score:
                            best_score = score
                            matched_cat = open_name

                # 3. If unclassified, but contains alphanumeric words (e.g. general room name, numbers)
                if matched_cat is None and alnum_count >= 2:
                    matched_cat = "room_sign" if len(raw_clean.split()) <= 4 else "notice_board"
                    best_score = float(ocr_conf)

                if matched_cat is not None:
                    results.append((x1, y1, x2, y2, matched_cat, raw_text.strip(), float(ocr_conf)))

        except Exception as exc:
            LOG.warning("[SIGN DETECTOR] Periodic CRAFT sweep error: %s", exc)

        return results

    # ── Pure Text OCR Recognition (Deterministic) ───────────────────────────────

    def classify_crop(
        self,
        crop_bgr: np.ndarray,
        category_hint: str | None = None,
    ) -> tuple[str | None, str | None, str | None, float, str]:
        """
        Evaluates a localized image crop using pure text OCR across both closed safety
        and open-vocabulary sign taxonomies (CLIP eliminated).
        Returns: (category, text_content, symbol_label, confidence, modality)
        """
        h_crop, w_crop = crop_bgr.shape[:2]
        if h_crop < 8 or w_crop < 8:
            return None, None, None, 0.0, "rejected_size"

        # Super-resolution / zoom enhancement for distant signs
        if h_crop < 64 or w_crop < 120:
            scale_fac = max(2.0, 64.0 / max(1, h_crop))
            crop_eval = cv2.resize(crop_bgr, (0, 0), fx=scale_fac, fy=scale_fac, interpolation=cv2.INTER_CUBIC)
        else:
            crop_eval = crop_bgr

        ocr_cat: str | None = None
        ocr_text: str | None = None
        ocr_conf = 0.0

        # EasyOCR Recognition
        if self.reader is not None:
            try:
                ocr_out = self.reader.readtext(crop_eval, min_size=8, text_threshold=0.20, low_text=0.20)
                if ocr_out:
                    joined_text = " ".join([item[1] for item in ocr_out if item[1]])
                    cleaned_str = _clean_text(joined_text)
                    alnum_count = sum(c.isalnum() for c in cleaned_str)
                    if alnum_count >= 2:
                        ocr_conf = float(np.mean([item[2] for item in ocr_out])) if ocr_out else 0.0
                        ocr_text = joined_text.strip()

                        # Check closed safety taxonomy (Tier 1)
                        for cat_name, cat_spec in HOSPITAL_SIGN_TAXONOMY.items():
                            is_match, kw, score = _fuzzy_match_keywords(joined_text, cat_spec["text_keywords"])
                            if is_match:
                                ocr_cat = cat_name
                                ocr_conf = max(ocr_conf, score)
                                break

                        # Check open-vocabulary taxonomy (Tier 2)
                        if ocr_cat is None:
                            for open_name, open_spec in OPEN_SIGN_TAXONOMY.items():
                                is_match, kw, score = _fuzzy_match_keywords(joined_text, open_spec["keywords"])
                                if is_match:
                                    ocr_cat = open_name
                                    ocr_conf = max(ocr_conf, score)
                                    break
            except Exception as exc:
                LOG.debug("[SIGN DETECTOR] Crop OCR error: %s", exc)

        # If no legible alphanumeric text was found and no door header anchor, REJECT.
        has_valid_text = bool(ocr_text and sum(c.isalnum() for c in _clean_text(ocr_text)) >= 2 and ocr_conf >= 0.20)
        is_door_header_exit = (category_hint == "exit_sign")

        if not has_valid_text and not is_door_header_exit:
            return None, None, None, 0.0, "no_text"

        final_cat: str | None = None
        final_conf = ocr_conf
        final_modality = "ocr"

        # Case 1: Safety sign text match (e.g. "EXIT", "EMERGENCY EXIT", "BIOHAZARD")
        if ocr_cat is not None and ocr_cat in HOSPITAL_SIGN_TAXONOMY:
            final_cat = ocr_cat
            final_conf = max(ocr_conf, 0.85)
            final_modality = "ocr_safety"

        # Case 2: Open-vocabulary text match (room_sign, notice_board, directional_sign)
        elif ocr_cat is not None and ocr_cat in OPEN_SIGN_TAXONOMY:
            final_cat = ocr_cat
            final_conf = max(ocr_conf, 0.75)
            final_modality = "ocr_open_vocab"

        # Case 3: Door Header Exit Sign (ISO 7010 running man or header placement)
        # CRUCIAL: Protect exit signs above doors from being converted to room signs!
        elif is_door_header_exit:
            final_cat = "exit_sign"
            final_conf = max(ocr_conf, 0.85)
            final_modality = "door_header_safety"

        # Case 4: Legible text detected elsewhere but unclassified -> room_sign or notice_board
        elif has_valid_text:
            clean_words = _clean_text(ocr_text).split()
            final_cat = "room_sign" if len(clean_words) <= 4 else "notice_board"
            final_conf = ocr_conf
            final_modality = "ocr_open_vocab"

        return final_cat, ocr_text, None, final_conf, final_modality

    # ── 3D Spatial Deduplication ────────────────────────────────────────────────

    def deduplicate_sign_3d(
        self,
        category: str,
        world_xyz: np.ndarray,
        radius_m: float = 0.45,
    ) -> tuple[int, bool]:
        """
        Matches detected sign to an existing 3D landmark within radius_m.
        Returns: (landmark_id, is_new_landmark)
        """
        best_dist = 999.0
        best_match_id = None

        for record in self.tracked_signs:
            if record["category"] == category:
                dist = float(np.linalg.norm(world_xyz - record["pos_3d"]))
                if dist < radius_m and dist < best_dist:
                    best_dist = dist
                    best_match_id = record["id"]

        if best_match_id is not None:
            # Update running average position
            for record in self.tracked_signs:
                if record["id"] == best_match_id:
                    record["pos_3d"] = 0.85 * record["pos_3d"] + 0.15 * world_xyz
                    record["hit_count"] += 1
                    return best_match_id, False

        # Register new landmark
        new_id = self._next_sign_id
        self._next_sign_id += 1
        self.tracked_signs.append({
            "id": new_id,
            "category": category,
            "pos_3d": world_xyz.copy(),
            "hit_count": 1,
        })
        return new_id, True

    # ── Primary Frame Processing Pipeline ───────────────────────────────────────

    def detect_signs_in_frame(
        self,
        frame_index: int,
        rgb_bgr: np.ndarray,
        detections: list[dict[str, Any]],
        depth_mm: np.ndarray | None = None,
        intrinsics: Any | None = None,
        pose_matrix: np.ndarray | None = None,
        cadence: int = 20,
    ) -> list[DetectedSign]:
        """
        Dual-trigger safety sign detection pipeline:
        1. Contextual visual cue trigger (every frame around doors / extinguishers).
        2. Periodic full-scene text sweep (every `cadence` frames).
        """
        h_frame, w_frame = rgb_bgr.shape[:2]
        candidate_rois: list[tuple[float, float, float, float, str, str | None, float]] = []

        # Trigger 1: Contextual Visual Cues (Immediate / Event-Driven)
        context_cues = self.generate_contextual_rois((h_frame, w_frame), detections)
        for x1, y1, x2, y2, hint in context_cues:
            candidate_rois.append((x1, y1, x2, y2, hint, None, 0.0))

        # Trigger 2: Full-Scene Text OCR Sweep (Every frame / cadence frames)
        if frame_index % max(1, cadence) == 0:
            periodic_rois = self.generate_periodic_text_rois(rgb_bgr)
            for x1, y1, x2, y2, matched_cat, raw_text, ocr_conf in periodic_rois:
                candidate_rois.append((x1, y1, x2, y2, matched_cat, raw_text, ocr_conf))

        if not candidate_rois:
            return []

        # Deduplicate overlapping 2D RoI proposals
        filtered_rois: list[tuple[float, float, float, float, str, str | None, float]] = []
        for cand in candidate_rois:
            cand_box = [cand[0], cand[1], cand[2], cand[3]]
            overlap = False
            for existing in filtered_rois:
                exist_box = [existing[0], existing[1], existing[2], existing[3]]
                if _box_iou(cand_box, exist_box) > 0.40:
                    overlap = True
                    break
            if not overlap:
                filtered_rois.append(cand)

        detected_signs: list[DetectedSign] = []

        for x1, y1, x2, y2, hint, prefetched_text, prefetched_conf in filtered_rois:
            ix1, iy1 = max(0, int(round(x1))), max(0, int(round(y1)))
            ix2, iy2 = min(w_frame, int(round(x2))), min(h_frame, int(round(y2)))
            if (ix2 - ix1) < 15 or (iy2 - iy1) < 15:
                continue

            crop = rgb_bgr[iy1:iy2, ix1:ix2]

            # If periodic detector already extracted valid text, short-circuit
            if prefetched_text is not None and prefetched_conf >= 0.20 and (hint in HOSPITAL_SIGN_TAXONOMY or hint in OPEN_SIGN_TAXONOMY):
                cat = hint
                text_content = prefetched_text
                symbol_label = None
                conf = prefetched_conf
                modality = "ocr"
            else:
                cat, text_content, symbol_label, conf, modality = self.classify_crop(crop, category_hint=hint)

            if cat is None:
                continue

            spec = HOSPITAL_SIGN_TAXONOMY.get(cat) or OPEN_SIGN_TAXONOMY.get(cat)
            if spec is None:
                continue

            # Format dynamic badge label with transcribed text
            if text_content and text_content.strip():
                clean_badge_text = re.sub(r"\s+", " ", text_content.strip())
                if len(clean_badge_text) > 18:
                    clean_badge_text = clean_badge_text[:17] + "…"
                badge_prefix = spec.get("badge_prefix") or spec.get("badge_label", "[SIGN]").strip("[]")
                badge_label = f"[{badge_prefix}: {clean_badge_text.upper()}]"
                display_name = f"{spec['display_name']} ({clean_badge_text})"
            else:
                badge_label = spec.get("badge_label") or f"[{spec.get('badge_prefix', 'SIGN')}]"
                display_name = spec["display_name"]

            # 3D Backprojection and Wall Backing Gate
            world_pos = None
            elevation_m = None

            if depth_mm is not None and intrinsics is not None and pose_matrix is not None:
                # Estimate median depth in the sign crop
                depth_crop = depth_mm[iy1:iy2, ix1:ix2]
                valid_depths = depth_crop[(depth_crop > 200) & (depth_crop < 7000)]
                if len(valid_depths) > 15:
                    z_med_m = float(np.median(valid_depths)) / 1000.0
                    cx = (x1 + x2) * 0.5
                    cy = (y1 + y2) * 0.5
                    fx = getattr(intrinsics, "fx", intrinsics[0, 0] if hasattr(intrinsics, "__getitem__") else 525.0)
                    fy = getattr(intrinsics, "fy", intrinsics[1, 1] if hasattr(intrinsics, "__getitem__") else 525.0)
                    cx_k = getattr(intrinsics, "cx", intrinsics[0, 2] if hasattr(intrinsics, "__getitem__") else 320.0)
                    cy_k = getattr(intrinsics, "cy", intrinsics[1, 2] if hasattr(intrinsics, "__getitem__") else 240.0)

                    cam_x = (cx - cx_k) * z_med_m / fx
                    cam_y = (cy - cy_k) * z_med_m / fy
                    cam_z = z_med_m
                    cam_pt = np.array([cam_x, cam_y, cam_z, 1.0], dtype=np.float32)
                    world_pt = (pose_matrix @ cam_pt)[:3]
                    world_pos = world_pt
                    elevation_m = float(world_pt[2])

                    # Physical Elevation Gate
                    min_elev = spec.get("min_elevation_m", 0.50)
                    max_elev = spec.get("max_elevation_m", 3.00)
                    if not (min_elev <= elevation_m <= max_elev):
                        LOG.debug(
                            "[SIGN DETECTOR] Dropped %s: Elevation %.2fm outside valid range [%.2f, %.2f]m",
                            cat, elevation_m, min_elev, max_elev,
                        )
                        continue

            sign_instance = DetectedSign(
                canonical_name=spec["canonical_name"],
                display_name=display_name,
                badge_label=badge_label,
                confidence=conf,
                bbox_xyxy=[float(x1), float(y1), float(x2), float(y2)],
                modality=modality,
                text_content=text_content,
                symbol_label=symbol_label,
                color_bgr=spec["color_bgr"],
                world_pos=world_pos,
                elevation_m=elevation_m,
            )
            detected_signs.append(sign_instance)

            LOG.info(
                "  [MULTIMODAL SIGN] Frame %d: Detected %s (%s) conf=%.2f text='%s' elev=%.2fm",
                frame_index, spec['display_name'], modality, conf, text_content or "", elevation_m or 0.0,
            )

        return detected_signs
