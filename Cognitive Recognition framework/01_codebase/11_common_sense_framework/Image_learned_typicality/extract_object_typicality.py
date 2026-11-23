"""
OneThinker Object Typicality & Common-Sense Reasoning Extractor
================================================================

Uses the OneThinker Vision-Language Reasoning Model (Default: OneThink/OneThinker-8B)
to inspect either:
  1. A single image for a given --target-class
  2. A folder of images for a given --target-class
  3. A root dataset folder containing multiple class folders (processes one or all classes)

For each target class, all outputs are consolidated into a dedicated class folder:
    <output_dir>/<target_class>/
        ├── <target_class>_typicality.json       (Machine-readable structured knowledge & typicality scores)
        ├── <target_class>_interpretation.md     (Human-readable interpretation report & per-image reasoning)
        ├── <target_class>_interpretation.txt    (Plain-text human-readable summary)
        └── annotated_images/                    (Visual grounding overlays of target + anchor objects)

Usage Examples:
---------------
1) Run on a folder of images for a specific class:
   python extract_object_typicality.py --input "path/to/security_camera_imgs" --target-class "security_camera"

2) Run on a root folder containing multiple class subfolders (e.g. security_camera, exit_sign, etc.):
   python extract_object_typicality.py --input "02_datasets/datasets/Hospital_Updated/datasets" --target-class "security_camera" --max-images 40

3) Run on ALL class subfolders inside a dataset root folder:
   python extract_object_typicality.py --input "02_datasets/datasets/Hospital_Updated/datasets" --max-images 40

4) Run on a single image:
   python extract_object_typicality.py --input "02_datasets/Testing Images/consultation_room.jpg" --target-class "security_camera"
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import sys
from typing import Any

import cv2
import numpy as np
from PIL import Image
import torch

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent.parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "04_outputs_runs_and_logs" / "outputs" / "common_sense_typicality"
DEFAULT_MODEL_ID = "OneThink/OneThinker-8B"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

LOG = logging.getLogger("onethinker_typicality")


# ── Canonical Predicate Normalization ─────────────────────────────────────────
PREDICATE_SYNONYMS: dict[str, str] = {
    "affixed to": "affixed_to",
    "affixed_to": "affixed_to",
    "attached to": "affixed_to",
    "attached_to": "affixed_to",
    "mounted on": "mounted_on",
    "mounted_on": "mounted_on",
    "mounted to": "mounted_on",
    "bolted to": "affixed_to",
    "fixed to": "affixed_to",
    "fixed on": "affixed_to",
    "hanging from": "hanging_from",
    "hanging_from": "hanging_from",
    "suspended from": "hanging_from",
    "on top of": "on_top_of",
    "on_top_of": "on_top_of",
    "resting on": "resting_on",
    "resting_on": "resting_on",
    "sitting on": "resting_on",
    "placed on": "resting_on",
    "standing on": "on_floor",
    "on floor": "on_floor",
    "on_floor": "on_floor",
    "near": "near",
    "next to": "adjacent_to",
    "beside": "adjacent_to",
    "adjacent to": "adjacent_to",
    "adjacent_to": "adjacent_to",
    "above": "above",
    "over": "above",
    "below": "below",
    "under": "below",
    "beneath": "below",
    "inside": "inside",
    "in": "inside",
    "in corner of": "in_corner_of",
    "in_corner_of": "in_corner_of",
    "along": "along",
}


def normalize_predicate(raw_pred: str) -> str:
    if not raw_pred:
        return "near"
    cleaned = raw_pred.strip().lower().replace("-", " ")
    cleaned_underscored = re.sub(r"\s+", " ", cleaned)
    if cleaned_underscored in PREDICATE_SYNONYMS:
        return PREDICATE_SYNONYMS[cleaned_underscored]
    return re.sub(r"[^a-z0-9_]", "_", cleaned_underscored.replace(" ", "_")).strip("_") or "near"


def normalize_entity_name(raw_name: str) -> str:
    if not raw_name:
        return "unknown"
    cleaned = raw_name.strip().lower()
    cleaned = re.sub(r"^(the|a|an)\s+", "", cleaned)
    cleaned = re.sub(r"[^a-z0-9\s_-]", "", cleaned)
    return re.sub(r"\s+", "_", cleaned).strip("_") or "unknown"


# ── Prompt Construction ───────────────────────────────────────────────────────
def build_onethinker_prompt(target_class: str) -> str:
    readable_class = target_class.replace("_", " ")
    return f"""You are a visual common-sense and spatial reasoning expert analyzing an image for the target object class: "{readable_class}".

Carefully inspect the image, reason step-by-step inside <think>...</think> tags, and then output your final structured analysis inside <answer>...</answer> tags as valid JSON.

In your <think> reasoning, briefly and concisely (in 5 short bullet points, under 150 words total) analyze:
1. Whether "{readable_class}" is visible in the image, and where it is located.
2. What the "{readable_class}" looks like visually (its color, shape, material, orientation/posture, and distinctive features).
3. What primary physical surface or anchor object the "{readable_class}" is physically interacting with (e.g., affixed to a wall, mounted on a ceiling, resting on a desk, standing on a floor, mounted above a door).
4. What other surrounding objects or architectural features are near or around it (e.g., keyboard near mouse and monitor, security camera near ceiling light or corridor corner).
5. What typical indoor/outdoor environment or room zone this represents.

Inside <answer>...</answer>, output ONLY a valid JSON object matching this exact schema (use normalized coordinates in [0, 1000] for bounding boxes [x1, y1, x2, y2] and points [x, y]):
{{
  "target_class": "{target_class}",
  "target_present": true,
  "confidence": 0.95,
  "visual_appearance": {{
    "colors": ["primary color", "secondary color"],
    "shape": "concise shape description (e.g., dome, rectangular box, cylindrical)",
    "material": "apparent material (e.g., white plastic, metal, glass)",
    "orientation_or_posture": "how it is positioned (e.g., angled downward from upper corner, flat on desk)",
    "description": "A vivid 1-2 sentence human-readable description of how the {readable_class} looks in this image."
  }},
  "target_grounding": {{
    "bbox_1000": [x1, y1, x2, y2],
    "center_point_1000": [cx, cy],
    "physical_elevation_3d": "resting_on_ground | waist_or_table_height | eye_or_upper_wall_height | ceiling_height"
  }},
  "primary_anchor_interaction": {{
    "predicate": "affixed_to | mounted_on | on_top_of | resting_on | hanging_from | adjacent_to | near | above | below | inside",
    "anchor_object": "specific surface or object it interacts with (e.g., wall, ceiling, desk, floor, door_frame, persons_back)",
    "anchor_bbox_1000": [ax1, ay1, ax2, ay2],
    "contact_point_1000": [px, py],
    "interaction_explanation": "Clear explanation of how the {readable_class} physically interacts with this surface/anchor."
  }},
  "surrounding_context": {{
    "environment_type": "e.g., hospital corridor, office desk, hallway junction, room entrance, outdoor sidewalk",
    "nearby_objects": [
      {{
        "object": "name of nearby object (e.g., mouse, monitor, ceiling_light, door, sidewalk)",
        "target_relation_to_nearby_object": "how the TARGET ({readable_class}) is positioned relative to this nearby object (so '{readable_class} is <relation> <object>'; e.g., if {readable_class} is above the sidewalk, write 'above', NEVER 'below')",
        "bbox_1000": [nx1, ny1, nx2, ny2],
        "reason": "how the {readable_class} relates to this nearby object"
      }}
    ]
  }},
  "human_readable_summary": "A rich, natural-language paragraph (3-4 sentences) describing the {readable_class}, its appearance (color, shape), the exact surface/anchor it is attached to or resting on, what objects surround it, and why this placement is typical or notable."
}}

If "{readable_class}" is NOT visible in the image, set "target_present": false and explain what is visible instead in "human_readable_summary"."""


# ── OneThinker Model Wrapper ──────────────────────────────────────────────────
class OneThinkerReasoner:
    """Loads OneThinker (Qwen2-VL / Qwen3-VL / AutoModelForImageTextToText) and runs visual reasoning."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "auto",
        max_new_tokens: int = 1536,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.processor: Any = None
        self.model: Any = None
        self._load_model()

    def _load_model(self) -> None:
        import importlib.util
        import transformers
        from transformers import AutoProcessor

        LOG.info("Loading OneThinker processor and model from '%s'...", self.model_id)
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_id, trust_remote_code=True, local_files_only=True
            )
            local_only = True
        except Exception:
            self.processor = AutoProcessor.from_pretrained(
                self.model_id, trust_remote_code=True
            )
            local_only = False

        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
            torch.float16 if torch.cuda.is_available() else torch.float32
        )

        # Try modern Transformers multimodal auto classes in order of compatibility
        loader_classes = []
        for cls_name in (
            "Qwen3VLForConditionalGeneration",
            "AutoModelForImageTextToText",
            "Qwen2_5_VLForConditionalGeneration",
            "Qwen2VLForConditionalGeneration",
        ):
            if hasattr(transformers, cls_name):
                loader_classes.append(getattr(transformers, cls_name))

        errors: list[str] = []
        for model_cls in loader_classes:
            try:
                kwargs: dict[str, Any] = {
                    "dtype": dtype,
                    "trust_remote_code": True,
                    "local_files_only": local_only,
                    "low_cpu_mem_usage": True,
                }
                self.model = model_cls.from_pretrained(self.model_id, **kwargs)
                if torch.cuda.is_available() and self.device != "cpu":
                    self.model = self.model.to("cuda")
                self.model.eval()
                if hasattr(self.model, "generation_config") and self.model.generation_config is not None:
                    self.model.generation_config.use_cache = True
                LOG.info("Loaded '%s' on %s using %s (dtype=%s, local_cache=%s)", self.model_id, next(self.model.parameters()).device, model_cls.__name__, dtype, local_only)
                return
            except Exception as exc:
                errors.append(f"{model_cls.__name__}: {exc}")
                continue

        raise RuntimeError(f"Could not load model '{self.model_id}'. Errors:\n" + "\n".join(errors))

    @torch.inference_mode()
    def analyze_image(self, image_path: Path, target_class: str) -> dict[str, Any]:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        pil_img = Image.open(image_path).convert("RGB")
        width, height = pil_img.size

        # Cap vision encoder input size to prevent quadratic ViT attention VRAM spikes
        # (Normalized 0-1000 coordinates still map back to original width x height)
        max_side = 512
        if max(width, height) > max_side:
            scale = max_side / float(max(width, height))
            infer_img = pil_img.resize(
                (max(28, int(round(width * scale))), max(28, int(round(height * scale)))),
                Image.Resampling.BICUBIC,
            )
        else:
            infer_img = pil_img

        prompt_text = build_onethinker_prompt(target_class)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": infer_img},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        if hasattr(self.processor, "apply_chat_template"):
            text_input = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text_input = prompt_text

        inputs = self.processor(
            text=[text_input],
            images=[infer_img],
            padding=True,
            return_tensors="pt",
        )

        model_device = next(self.model.parameters()).device
        inputs = {
            k: v.to(model_device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

        if "input_ids" in inputs:
            trimmed_ids = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
        else:
            trimmed_ids = generated_ids

        raw_output = self.processor.batch_decode(
            trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return parse_onethinker_response(
            raw_output=raw_output,
            image_path=image_path,
            image_size=(width, height),
            target_class=target_class,
        )


# ── Output Parser & Coordinate Conversion ─────────────────────────────────────
def _scale_box_1000_to_px(
    box: list[float] | None, width: int, height: int
) -> tuple[list[float] | None, list[int] | None]:
    if not box or not isinstance(box, (list, tuple)) or len(box) != 4:
        return None, None
    try:
        vals = [float(x) for x in box]
    except (TypeError, ValueError):
        return None, None

    # Handle either [0, 1] or [0, 1000] scale
    scale = 1.0 if max(vals) <= 1.05 else 1000.0
    norm_box = [round(max(0.0, min(1.0, v / scale)), 4) for v in vals]
    px_box = [
        int(round(norm_box[0] * width)),
        int(round(norm_box[1] * height)),
        int(round(norm_box[2] * width)),
        int(round(norm_box[3] * height)),
    ]
    return norm_box, px_box


def _scale_point_1000_to_px(
    pt: list[float] | None, width: int, height: int
) -> tuple[list[float] | None, list[int] | None]:
    if not pt or not isinstance(pt, (list, tuple)) or len(pt) != 2:
        return None, None
    try:
        vals = [float(x) for x in pt]
    except (TypeError, ValueError):
        return None, None

    scale = 1.0 if max(vals) <= 1.05 else 1000.0
    norm_pt = [round(max(0.0, min(1.0, v / scale)), 4) for v in vals]
    px_pt = [int(round(norm_pt[0] * width)), int(round(norm_pt[1] * height))]
    return norm_pt, px_pt


def _extract_json_dict(text: str) -> dict[str, Any] | None:
    # 1. Look inside <answer>...</answer> (or after <answer> if closing tag was truncated)
    ans_match = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if ans_match:
        candidate_str = ans_match.group(1).strip()
    elif "<answer>" in text.lower():
        idx = text.lower().find("<answer>") + len("<answer>")
        candidate_str = text[idx:].strip()
    else:
        candidate_str = text.strip()

    # 2. Strip markdown code fences if present
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate_str, flags=re.DOTALL)
    if fence_match:
        candidate_str = fence_match.group(1).strip()
    else:
        brace_start = candidate_str.find("{")
        if brace_start != -1:
            candidate_str = candidate_str[brace_start:].strip()

    for suffix in ("", '"\n}', '"}', "\n}", "}", "]}\n}", '"]}\n}'):
        try:
            parsed = json.loads(candidate_str + suffix)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return None


GROUND_PLANE_TERMS = {
    "sidewalk", "floor", "ground", "pavement", "road", "carpet",
    "corridor_floor", "tile_floor", "walkway", "pathway",
}

WAIST_OR_TORSO_ANCHOR_TERMS = {
    "back", "shoulder", "torso", "waist", "shirt", "person", "child",
    "desk", "table", "counter", "chair", "bench", "trolley", "bed", "stretcher",
}


def _compute_2d_frame_vertical_zone(
    center_pt_norm: list[float] | None,
    bbox_norm: list[float] | None,
) -> str:
    """
    Derive the 2D image vertical zone mathematically from the grounded bounding box
    center y_norm in [0, 1] (0 = top of frame, 1 = bottom of frame).
    """
    y_norm: float | None = None
    if isinstance(bbox_norm, list) and len(bbox_norm) == 4:
        y_norm = (float(bbox_norm[1]) + float(bbox_norm[3])) * 0.5
    elif isinstance(center_pt_norm, list) and len(center_pt_norm) == 2:
        y_norm = float(center_pt_norm[1])

    if y_norm is None:
        return "unknown"
    if y_norm <= 0.3333:
        return "upper_frame"
    if y_norm <= 0.6667:
        return "mid_frame"
    return "lower_frame"


def _resolve_3d_physical_elevation(
    raw_zone: str,
    anchor_object: str,
    predicate: str,
    orientation_text: str,
    interaction_text: str,
) -> str:
    """
    Resolve true 3D physical elevation relative to the ground plane:
      - resting_on_ground
      - waist_or_table_height
      - eye_or_upper_wall_height
      - ceiling_height
    Corrects camera-tilt confusion (e.g. a camera looking down at a backpack worn on
    a person's back or resting on a table).
    """
    z = normalize_entity_name(raw_zone)
    context_blob = f"{anchor_object} {orientation_text} {interaction_text}".lower()

    if "ceiling" in z or "ceiling" in anchor_object:
        return "ceiling_height"
    if any(k in z for k in ("eye", "upper_wall", "high_wall")):
        return "eye_or_upper_wall_height"
    if any(k in z for k in ("waist", "table", "counter", "mid_wall", "torso")):
        return "waist_or_table_height"

    # If model said floor_level / resting_on_ground, check if it is actually carried on a person or on furniture
    if any(term in anchor_object for term in WAIST_OR_TORSO_ANCHOR_TERMS) or any(
        phrase in context_blob for phrase in ("worn", "back", "shoulder", "desk", "table", "counter", "chair")
    ):
        return "waist_or_table_height"

    if any(k in anchor_object for k in ("upper_wall", "door_frame", "high_wall")):
        return "eye_or_upper_wall_height"

    if any(k in z for k in ("floor", "ground")):
        return "resting_on_ground"

    return z if z != "unknown" else "waist_or_table_height"


def _verify_and_fix_relation_direction(
    target_class: str,
    raw_relation: str,
    nearby_object: str,
    reason: str,
) -> str:
    """
    Ensures the spatial relation describes how the TARGET is positioned relative to
    the NEARBY OBJECT (i.e. '<target_class> is <relation> <nearby_object>').
    Fixes inverted VLM tags where the prose reason says 'target is above sidewalk'
    but the tag says 'below' (meaning 'sidewalk is below target').
    """
    rel = normalize_predicate(raw_relation)
    reason_low = reason.lower()
    target_readable = target_class.replace("_", " ").lower()
    nearby_readable = nearby_object.replace("_", " ").lower()

    # 1. Physical ground-plane invariant: above-ground objects are 'above' or 'on_floor' relative to sidewalk/floor, never 'below' it
    if any(g in nearby_object for g in GROUND_PLANE_TERMS) and rel == "below":
        return "above"

    # 2. Explicit prose self-consistency check ('<target> is above <nearby>' vs 'below')
    if rel == "below":
        if (
            f"{target_readable} is above" in reason_low
            or f"above the {nearby_readable}" in reason_low
            or "above the sidewalk" in reason_low
            or "above the floor" in reason_low
            or "above the ground" in reason_low
        ):
            return "above"
    elif rel == "above":
        if (
            f"{target_readable} is below" in reason_low
            or f"{target_readable} is under" in reason_low
            or f"below the {nearby_readable}" in reason_low
            or f"under the {nearby_readable}" in reason_low
        ):
            return "below"
        # Perspective background check: background objects near the top of a tilted 2D image
        # are behind/near in 3D, not physically below the target
        if "in the background" in reason_low and ("top of the image" in reason_low or "legs" in nearby_object):
            return "near_in_background"

    return rel


def parse_onethinker_response(
    raw_output: str,
    image_path: Path,
    image_size: tuple[int, int],
    target_class: str,
) -> dict[str, Any]:
    width, height = image_size

    think_match = re.search(r"<think>(.*?)</think>", raw_output, flags=re.DOTALL | re.IGNORECASE)
    chain_of_thought = think_match.group(1).strip() if think_match else ""

    parsed_json = _extract_json_dict(raw_output) or {}

    target_present = bool(parsed_json.get("target_present", True))
    confidence = float(parsed_json.get("confidence", 0.85 if parsed_json else 0.5))

    vis = parsed_json.get("visual_appearance") or {}
    if not isinstance(vis, dict):
        vis = {}

    tg = parsed_json.get("target_grounding") or {}
    if not isinstance(tg, dict):
        tg = {}
    target_bbox_norm, target_bbox_px = _scale_box_1000_to_px(tg.get("bbox_1000"), width, height)
    target_pt_norm, target_pt_px = _scale_point_1000_to_px(tg.get("center_point_1000"), width, height)
    if target_bbox_norm is not None:
        # Derive center point mathematically from grounded bbox center for exact consistency
        target_pt_norm = [
            round((target_bbox_norm[0] + target_bbox_norm[2]) * 0.5, 4),
            round((target_bbox_norm[1] + target_bbox_norm[3]) * 0.5, 4),
        ]
        target_pt_px = [
            int(round(target_pt_norm[0] * width)),
            int(round(target_pt_norm[1] * height)),
        ]

    pa = parsed_json.get("primary_anchor_interaction") or {}
    if not isinstance(pa, dict):
        pa = {}
    anchor_bbox_norm, anchor_bbox_px = _scale_box_1000_to_px(pa.get("anchor_bbox_1000"), width, height)
    contact_pt_norm, contact_pt_px = _scale_point_1000_to_px(pa.get("contact_point_1000"), width, height)

    predicate = normalize_predicate(str(pa.get("predicate") or "near"))
    anchor_object = normalize_entity_name(str(pa.get("anchor_object") or "unknown"))
    interaction_explanation = str(pa.get("interaction_explanation") or "").strip()
    orientation_text = str(vis.get("orientation_or_posture") or "").strip()

    # 1. Mathematically derive 2D vertical frame zone from grounded bounding box center
    image_vertical_zone_2d = _compute_2d_frame_vertical_zone(target_pt_norm, target_bbox_norm)

    # 2. Resolve 3D physical elevation relative to ground plane
    raw_phys_elev = str(tg.get("physical_elevation_3d") or tg.get("vertical_zone") or "unknown")
    physical_elevation_3d = _resolve_3d_physical_elevation(
        raw_zone=raw_phys_elev,
        anchor_object=anchor_object,
        predicate=predicate,
        orientation_text=orientation_text,
        interaction_text=interaction_explanation,
    )

    sc = parsed_json.get("surrounding_context") or {}
    if not isinstance(sc, dict):
        sc = {}
    raw_nearby = sc.get("nearby_objects") if isinstance(sc.get("nearby_objects"), list) else []
    nearby_objects = []
    for item in raw_nearby:
        if not isinstance(item, dict):
            continue
        nb_norm, nb_px = _scale_box_1000_to_px(item.get("bbox_1000"), width, height)
        nb_obj_name = normalize_entity_name(str(item.get("object") or "object"))
        raw_rel = str(
            item.get("target_relation_to_nearby_object")
            or item.get("spatial_relation")
            or "near"
        )
        reason_str = str(item.get("reason") or "").strip()
        verified_rel = _verify_and_fix_relation_direction(
            target_class=target_class,
            raw_relation=raw_rel,
            nearby_object=nb_obj_name,
            reason=reason_str,
        )
        nearby_objects.append({
            "object": nb_obj_name,
            "target_relation_to_nearby_object": verified_rel,
            "spatial_relation": verified_rel,
            "bbox_norm": nb_norm,
            "bbox_px": nb_px,
            "reason": reason_str,
        })

    human_summary = str(parsed_json.get("human_readable_summary") or "").strip()
    if not human_summary:
        # Fallback if model answered in free text
        clean_text = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()
        clean_text = re.sub(r"</?answer>", "", clean_text).strip()
        human_summary = clean_text

    return {
        "image_name": image_path.name,
        "image_path": str(image_path.resolve()),
        "image_dimensions": {"width": width, "height": height},
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "target_class": target_class,
        "target_present": target_present,
        "confidence": round(confidence, 4),
        "chain_of_thought_reasoning": chain_of_thought,
        "visual_appearance": {
            "colors": [str(c).lower().strip() for c in (vis.get("colors") or []) if str(c).strip()],
            "shape": str(vis.get("shape") or "").strip(),
            "material": str(vis.get("material") or "").strip(),
            "orientation_or_posture": orientation_text,
            "description": str(vis.get("description") or "").strip(),
        },
        "target_grounding": {
            "bbox_norm": target_bbox_norm,
            "bbox_px": target_bbox_px,
            "center_point_norm": target_pt_norm,
            "center_point_px": target_pt_px,
            "physical_elevation_3d": physical_elevation_3d,
            "image_vertical_zone_2d": image_vertical_zone_2d,
            "vertical_zone": physical_elevation_3d,
        },
        "primary_anchor_interaction": {
            "predicate": predicate,
            "anchor_object": anchor_object,
            "triplet": f"{target_class} --[{predicate}]--> {anchor_object}",
            "anchor_bbox_norm": anchor_bbox_norm,
            "anchor_bbox_px": anchor_bbox_px,
            "contact_point_norm": contact_pt_norm,
            "contact_point_px": contact_pt_px,
            "interaction_explanation": interaction_explanation,
        },
        "surrounding_context": {
            "environment_type": str(sc.get("environment_type") or "").strip(),
            "nearby_objects": nearby_objects,
        },
        "human_readable_summary": human_summary,
        "raw_model_output": raw_output,
    }


# ── Aggregation & Typicality Calculation ──────────────────────────────────────
def aggregate_class_typicality(
    target_class: str,
    model_id: str,
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Computes empirical Typicality Scores and distributions across all processed
    images for a single target class.
    """
    total_images = len(observations)
    positive_obs = [obs for obs in observations if obs.get("target_present", False)]
    pos_count = len(positive_obs)
    denom = max(1, pos_count)

    triplet_counter: Counter[tuple[str, str]] = Counter()
    anchor_counter: Counter[str] = Counter()
    predicate_counter: Counter[str] = Counter()
    physical_elevation_counter: Counter[str] = Counter()
    frame_zone_2d_counter: Counter[str] = Counter()
    nearby_obj_counter: Counter[str] = Counter()
    nearby_rel_counter: Counter[tuple[str, str]] = Counter()
    color_counter: Counter[str] = Counter()
    shape_counter: Counter[str] = Counter()
    material_counter: Counter[str] = Counter()
    env_counter: Counter[str] = Counter()

    y_centers: list[float] = []
    x_centers: list[float] = []
    aspect_ratios: list[float] = []

    for obs in positive_obs:
        pa = obs.get("primary_anchor_interaction") or {}
        pred = pa.get("predicate") or "near"
        anchor = pa.get("anchor_object") or "unknown"
        if anchor != "unknown":
            triplet_counter[(pred, anchor)] += 1
            anchor_counter[anchor] += 1
            predicate_counter[pred] += 1

        tg = obs.get("target_grounding") or {}
        phys_elev = tg.get("physical_elevation_3d") or tg.get("vertical_zone") or "unknown"
        if phys_elev != "unknown":
            physical_elevation_counter[phys_elev] += 1

        fzone_2d = tg.get("image_vertical_zone_2d") or _compute_2d_frame_vertical_zone(
            tg.get("center_point_norm"), tg.get("bbox_norm")
        )
        if fzone_2d != "unknown":
            frame_zone_2d_counter[fzone_2d] += 1

        cpt = tg.get("center_point_norm")
        if isinstance(cpt, list) and len(cpt) == 2:
            x_centers.append(float(cpt[0]))
            y_centers.append(float(cpt[1]))

        bbox = tg.get("bbox_norm")
        if isinstance(bbox, list) and len(bbox) == 4:
            bw = max(1e-4, float(bbox[2]) - float(bbox[0]))
            bh = max(1e-4, float(bbox[3]) - float(bbox[1]))
            aspect_ratios.append(bw / bh)

        sc = obs.get("surrounding_context") or {}
        env = str(sc.get("environment_type") or "").strip().lower()
        if env:
            env_counter[env] += 1

        for nb in sc.get("nearby_objects") or []:
            nb_obj = nb.get("object") or "unknown"
            nb_rel = nb.get("target_relation_to_nearby_object") or nb.get("spatial_relation") or "near"
            if nb_obj != "unknown":
                nearby_obj_counter[nb_obj] += 1
                nearby_rel_counter[(nb_rel, nb_obj)] += 1

        vis = obs.get("visual_appearance") or {}
        for col in vis.get("colors") or []:
            if col:
                color_counter[col.lower().strip()] += 1
        if vis.get("shape"):
            shape_counter[vis["shape"].lower().strip()] += 1
        if vis.get("material"):
            material_counter[vis["material"].lower().strip()] += 1

    # Build ranked lists with Typicality Scores in [0, 1]
    anchor_interactions_ranked = [
        {
            "rank": idx + 1,
            "predicate": pred,
            "anchor_object": anchor,
            "statement": f"{target_class} is {pred.replace('_', ' ')} {anchor.replace('_', ' ')}",
            "count": count,
            "typicality_score": round(count / denom, 4),
        }
        for idx, ((pred, anchor), count) in enumerate(triplet_counter.most_common())
    ]

    surrounding_objects_ranked = [
        {
            "rank": idx + 1,
            "nearby_object": obj,
            "count": count,
            "co_occurrence_typicality": round(count / denom, 4),
        }
        for idx, (obj, count) in enumerate(nearby_obj_counter.most_common())
    ]

    surrounding_relations_ranked = [
        {
            "rank": idx + 1,
            "target_relation_to_nearby_object": rel,
            "spatial_relation": rel,
            "nearby_object": obj,
            "statement": f"{target_class} is {rel.replace('_', ' ')} {obj.replace('_', ' ')}",
            "count": count,
            "typicality_score": round(count / denom, 4),
        }
        for idx, ((rel, obj), count) in enumerate(nearby_rel_counter.most_common())
    ]

    physical_elevations_ranked = [
        {
            "physical_elevation_3d": zone,
            "vertical_zone": zone,
            "count": count,
            "typicality_score": round(count / denom, 4),
        }
        for zone, count in physical_elevation_counter.most_common()
    ]

    frame_zones_2d_ranked = [
        {
            "image_vertical_zone_2d": zone,
            "count": count,
            "typicality_score": round(count / denom, 4),
        }
        for zone, count in frame_zone_2d_counter.most_common()
    ]

    spatial_stats: dict[str, Any] = {}
    if y_centers:
        y_arr = np.asarray(y_centers, dtype=np.float32)
        x_arr = np.asarray(x_centers, dtype=np.float32)
        spatial_stats["normalized_image_position"] = {
            "mean_x": round(float(np.mean(x_arr)), 4),
            "mean_y": round(float(np.mean(y_arr)), 4),
            "std_y": round(float(np.std(y_arr)), 4),
            "y_percentile_5": round(float(np.percentile(y_arr, 5)), 4),
            "y_percentile_95": round(float(np.percentile(y_arr, 95)), 4),
        }
    if aspect_ratios:
        ar_arr = np.asarray(aspect_ratios, dtype=np.float32)
        spatial_stats["bbox_aspect_ratio"] = {
            "mean": round(float(np.mean(ar_arr)), 4),
            "std": round(float(np.std(ar_arr)), 4),
            "min": round(float(np.min(ar_arr)), 4),
            "max": round(float(np.max(ar_arr)), 4),
        }

    # Synthesize a natural-language Class Typicality Synthesis paragraph
    synthesized_rules = _synthesize_human_class_summary(
        target_class=target_class,
        pos_count=pos_count,
        anchor_interactions=anchor_interactions_ranked,
        surrounding_relations=surrounding_relations_ranked,
        physical_elevations=physical_elevations_ranked,
        frame_zones_2d=frame_zones_2d_ranked,
        colors=color_counter.most_common(5),
        shapes=shape_counter.most_common(4),
        materials=material_counter.most_common(4),
        environments=env_counter.most_common(5),
    )

    return {
        "schema_version": "1.1.0",
        "target_class": target_class,
        "model_id": model_id,
        "last_updated_utc": datetime.now(timezone.utc).isoformat(),
        "total_images_analyzed": total_images,
        "images_with_target_present": pos_count,
        "human_readable_class_synthesis": synthesized_rules,
        "typicality_summary": {
            "primary_anchor_interactions": anchor_interactions_ranked,
            "surrounding_object_relations": surrounding_relations_ranked,
            "surrounding_objects_frequency": surrounding_objects_ranked,
            "physical_elevation_zones_3d": physical_elevations_ranked,
            "image_vertical_zones_2d": frame_zones_2d_ranked,
            "vertical_elevation_zones": physical_elevations_ranked,
            "typical_environments": [
                {"environment": env, "count": c, "typicality_score": round(c / denom, 4)}
                for env, c in env_counter.most_common(10)
            ],
            "visual_appearance_priors": {
                "typical_colors": [{"color": col, "count": c} for col, c in color_counter.most_common(8)],
                "typical_shapes": [{"shape": shp, "count": c} for shp, c in shape_counter.most_common(8)],
                "typical_materials": [{"material": mat, "count": c} for mat, c in material_counter.most_common(8)],
            },
            "spatial_2d_statistics": spatial_stats,
        },
        "per_image_observations": observations,
    }


def _synthesize_human_class_summary(
    target_class: str,
    pos_count: int,
    anchor_interactions: list[dict[str, Any]],
    surrounding_relations: list[dict[str, Any]],
    physical_elevations: list[dict[str, Any]],
    frame_zones_2d: list[dict[str, Any]],
    colors: list[tuple[str, int]],
    shapes: list[tuple[str, int]],
    materials: list[tuple[str, int]],
    environments: list[tuple[str, int]],
) -> str:
    readable = target_class.replace("_", " ")
    if pos_count == 0:
        return f"No positive observations of '{readable}' were found in the analyzed images."

    parts: list[str] = []

    # 1. Appearance
    col_str = ", ".join(f"{c} ({n})" for c, n in colors[:3]) if colors else "varied colors"
    shp_str = ", ".join(f"{s}" for s, _ in shapes[:2]) if shapes else "standard form"
    mat_str = ", ".join(f"{m}" for m, _ in materials[:2]) if materials else "mixed material"
    parts.append(
        f"Across {pos_count} analyzed image(s), the {readable} typically appears in {col_str} "
        f"with a {shp_str} profile ({mat_str})."
    )

    # 2. Primary surface / anchor typicality
    if anchor_interactions:
        top_anchors = [
            f"{item['predicate'].replace('_', ' ')} {item['anchor_object'].replace('_', ' ')} "
            f"(typicality={item['typicality_score']:.2f}, {item['count']}/{pos_count})"
            for item in anchor_interactions[:3]
        ]
        parts.append(f"Physically, it is most typically found: {'; '.join(top_anchors)}.")

    # 3. 3D Physical Elevation vs 2D Frame Zone
    if physical_elevations:
        top_phys = [
            f"{z['physical_elevation_3d'].replace('_', ' ')} ({z['typicality_score']*100:.0f}%)"
            for z in physical_elevations[:2]
        ]
        f2d_str = ""
        if frame_zones_2d:
            top_2d = [
                f"{z['image_vertical_zone_2d'].replace('_', ' ')} ({z['typicality_score']*100:.0f}%)"
                for z in frame_zones_2d[:2]
            ]
            f2d_str = f" and 2D image vertical zone is {', '.join(top_2d)}"
        parts.append(f"Its 3D physical elevation relative to the ground is {', '.join(top_phys)}{f2d_str}.")

    # 4. Surrounding context
    if surrounding_relations:
        top_surr = [
            f"{s['spatial_relation'].replace('_', ' ')} {s['nearby_object'].replace('_', ' ')} "
            f"(score={s['typicality_score']:.2f})"
            for s in surrounding_relations[:4]
        ]
        parts.append(f"It is commonly surrounded by or co-located with: {', '.join(top_surr)}.")

    # 5. Environment
    if environments:
        env_list = ", ".join(f"{e}" for e, _ in environments[:3])
        parts.append(f"Typical scene environments include: {env_list}.")

    return " ".join(parts)


# ── Human-Readable Report Writers (.md and .txt) ──────────────────────────────
def write_human_readable_reports(
    class_dir: Path,
    knowledge: dict[str, Any],
) -> tuple[Path, Path]:
    target_class = knowledge["target_class"]
    readable_title = target_class.replace("_", " ").title()
    summary = knowledge["typicality_summary"]
    observations = knowledge["per_image_observations"]

    md_lines: list[str] = [
        f"# 🧠 Object Typicality & Common-Sense Report: {readable_title} (`{target_class}`)",
        "",
        f"- **Model**: `{knowledge['model_id']}`",
        f"- **Last Updated (UTC)**: `{knowledge['last_updated_utc']}`",
        f"- **Images Analyzed**: `{knowledge['total_images_analyzed']}` "
        f"(`{knowledge['images_with_target_present']}` with target present)",
        "",
        "---",
        "",
        "## 1. Executive Human-Readable Synthesis",
        "",
        f"> {knowledge['human_readable_class_synthesis']}",
        "",
        "---",
        "",
        "## 2. Learned Surface & Anchor Typicality (`target --[predicate]--> anchor`)",
        "",
    ]

    anchors = summary.get("primary_anchor_interactions") or []
    if anchors:
        md_lines.extend([
            "| Rank | Spatial Statement | Predicate | Anchor Surface / Object | Count | Typicality Score |",
            "| :--- | :--- | :--- | :--- | :---: | :---: |",
        ])
        for item in anchors:
            md_lines.append(
                f"| {item['rank']} | **{item['statement']}** | `{item['predicate']}` | "
                f"`{item['anchor_object']}` | {item['count']} | **{item['typicality_score']:.2f}** |"
            )
    else:
        md_lines.append("_No anchor interactions recorded yet._")

    md_lines.extend([
        "",
        "---",
        "",
        "## 3. What Surrounds It? (Co-Occurring Objects & Spatial Context)",
        "",
    ])

    surr_rels = summary.get("surrounding_object_relations") or []
    if surr_rels:
        md_lines.extend([
            "| Rank | Context Statement (`target -> nearby`) | Target Relation to Object | Surrounding Object | Count | Typicality Score |",
            "| :--- | :--- | :--- | :--- | :---: | :---: |",
        ])
        for item in surr_rels[:15]:
            rel_val = item.get("target_relation_to_nearby_object") or item.get("spatial_relation")
            md_lines.append(
                f"| {item['rank']} | {item['statement']} | `{rel_val}` | "
                f"`{item['nearby_object']}` | {item['count']} | **{item['typicality_score']:.2f}** |"
            )
    else:
        md_lines.append("_No surrounding context objects recorded yet._")

    md_lines.extend([
        "",
        "---",
        "",
        "## 4. Visual Appearance & Elevation Priors",
        "",
    ])

    vp = summary.get("visual_appearance_priors") or {}
    colors_str = ", ".join(f"`{c['color']}` ({c['count']})" for c in (vp.get("typical_colors") or [])) or "N/A"
    shapes_str = ", ".join(f"`{s['shape']}` ({s['count']})" for s in (vp.get("typical_shapes") or [])) or "N/A"
    mats_str = ", ".join(f"`{m['material']}` ({m['count']})" for m in (vp.get("typical_materials") or [])) or "N/A"
    phys_elev_str = ", ".join(
        f"`{z['physical_elevation_3d']}` ({z['typicality_score']*100:.0f}%)"
        for z in (summary.get("physical_elevation_zones_3d") or summary.get("vertical_elevation_zones") or [])
    ) or "N/A"
    frame_2d_str = ", ".join(
        f"`{z['image_vertical_zone_2d']}` ({z['typicality_score']*100:.0f}%)"
        for z in (summary.get("image_vertical_zones_2d") or [])
    ) or "N/A"

    md_lines.extend([
        f"- **Typical Colors**: {colors_str}",
        f"- **Typical Shapes**: {shapes_str}",
        f"- **Typical Materials**: {mats_str}",
        f"- **3D Physical Elevation (Ground-Relative)**: {phys_elev_str}",
        f"- **2D Image Vertical Zone (Derived from BBox $y_{{\\text{{center}}}}$)**: {frame_2d_str}",
        "",
        "---",
        "",
        "## 5. Per-Image Human-Readable Reasoning & Observations",
        "",
    ])

    for idx, obs in enumerate(observations, start=1):
        img_name = obs.get("image_name", f"image_{idx}")
        present = obs.get("target_present", False)
        status_badge = "DETECTED" if present else "NOT PRESENT"
        pa = obs.get("primary_anchor_interaction") or {}
        vis = obs.get("visual_appearance") or {}
        tg = obs.get("target_grounding") or {}
        sc = obs.get("surrounding_context") or {}

        md_lines.extend([
            f"### [{idx}] `{img_name}` ({status_badge})",
            f"- **Image Path**: `{obs.get('image_path')}`",
            f"- **Human Interpretation**: {obs.get('human_readable_summary')}",
        ])
        if present:
            cpt_norm = tg.get("center_point_norm") or [None, None]
            y_norm_str = f"{cpt_norm[1]:.3f}" if isinstance(cpt_norm, list) and len(cpt_norm) == 2 and cpt_norm[1] is not None else "N/A"
            md_lines.extend([
                f"- **Visual Appearance**: Colors={vis.get('colors')}, Shape=`{vis.get('shape')}`, "
                f"Material=`{vis.get('material')}` — {vis.get('description')}",
                f"- **Primary Anchor Interaction**: `{pa.get('triplet')}` — {pa.get('interaction_explanation')}",
                f"- **Grounding & Elevation**: Target Box (px)=`{tg.get('bbox_px')}`, "
                f"Anchor Box (px)=`{pa.get('anchor_bbox_px')}`, "
                f"3D Physical Elevation=`{tg.get('physical_elevation_3d')}`, "
                f"2D Frame Zone=`{tg.get('image_vertical_zone_2d')}` ($y_{{\\text{{norm}}}}={y_norm_str}$)",
            ])
            nearby_list = sc.get("nearby_objects") or []
            if nearby_list:
                nb_formatted = "; ".join(
                    f"`{n.get('target_relation_to_nearby_object') or n.get('spatial_relation')}` **{n['object']}** ({n['reason']})"
                    for n in nearby_list
                )
                md_lines.append(f"- **Surrounding Objects (`target -> object`)**: {nb_formatted}")

        cot = obs.get("chain_of_thought_reasoning")
        if cot:
            md_lines.extend([
                "- **OneThinker Step-by-Step Reasoning (`<think>`)**:",
                f"  > {cot.replace(chr(10), chr(10) + '  > ')}",
            ])
        md_lines.append("")

    md_path = class_dir / f"{target_class}_interpretation.md"
    txt_path = class_dir / f"{target_class}_interpretation.txt"

    md_content = "\n".join(md_lines)
    md_path.write_text(md_content, encoding="utf-8")
    txt_path.write_text(md_content, encoding="utf-8")
    return md_path, txt_path


# ── Annotated Grounding Image Saver ───────────────────────────────────────────
def save_annotated_grounding_image(
    image_path: Path,
    observation: dict[str, Any],
    annotated_dir: Path,
) -> Path | None:
    if not observation.get("target_present", False):
        return None
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    tg = observation.get("target_grounding") or {}
    pa = observation.get("primary_anchor_interaction") or {}

    t_box = tg.get("bbox_px")
    a_box = pa.get("anchor_bbox_px")

    # Draw Anchor Box (Blue/Cyan)
    if isinstance(a_box, list) and len(a_box) == 4:
        ax1, ay1, ax2, ay2 = a_box
        cv2.rectangle(img, (ax1, ay1), (ax2, ay2), (255, 180, 0), 2)
        anchor_lbl = f"Anchor: {pa.get('anchor_object', 'surface')}"
        cv2.putText(
            img, anchor_lbl, (ax1, max(20, ay1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 0), 2, cv2.LINE_AA
        )

    # Draw Target Box (Green)
    if isinstance(t_box, list) and len(t_box) == 4:
        tx1, ty1, tx2, ty2 = t_box
        cv2.rectangle(img, (tx1, ty1), (tx2, ty2), (40, 220, 40), 3)
        triplet_lbl = f"{observation['target_class']} [{pa.get('predicate', 'near')} -> {pa.get('anchor_object', '')}]"
        cv2.putText(
            img, triplet_lbl, (tx1, max(22, ty1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 220, 40), 2, cv2.LINE_AA
        )

    annotated_dir.mkdir(parents=True, exist_ok=True)
    out_file = annotated_dir / f"grounded_{image_path.name}"
    cv2.imwrite(str(out_file), img)
    return out_file


# ── Input Folder Discovery (Single Image, Class Folder, or Multi-Class Root) ──
def collect_images_in_folder(folder: Path, max_images: int | None = None) -> list[Path]:
    images = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        and "annotated_images" not in p.parts
    )
    if max_images is not None and max_images > 0:
        images = images[:max_images]
    return images


def resolve_class_image_jobs(
    input_path: Path,
    target_class: str | None = None,
    max_images_per_class: int | None = None,
) -> dict[str, list[Path]]:
    """
    Resolves input_path into {class_name: [image_paths, ...]}.
    Supports:
      1. Single image file (--target-class required)
      2. Folder of images for a single class (--target-class given, or inferred from folder name)
      3. Root dataset folder containing class subfolders (processes matching --target-class or all subfolders)
    """
    input_path = input_path.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        if not target_class:
            raise ValueError("When --input is a single image file, please provide --target-class.")
        cname = normalize_entity_name(target_class)
        return {cname: [input_path]}

    # Input is a directory. Check if it has immediate class subdirectories
    subdirs = sorted(
        d for d in input_path.iterdir()
        if d.is_dir() and not d.name.startswith(".") and d.name not in ("__pycache__", "annotated_images")
    )

    if target_class:
        cname = normalize_entity_name(target_class)
        # Check if there is a matching subfolder for target_class inside input_path
        for d in subdirs:
            if normalize_entity_name(d.name) == cname:
                imgs = collect_images_in_folder(d, max_images=max_images_per_class)
                return {cname: imgs}
        # Otherwise treat input_path itself as the folder of images for target_class
        imgs = collect_images_in_folder(input_path, max_images=max_images_per_class)
        return {cname: imgs}

    # No target_class specified: check if subdirs represent distinct object classes
    # (Skip treating train/val/test/images/labels as class names)
    split_folder_names = {"train", "val", "valid", "test", "images", "labels"}
    if subdirs and not any(d.name.lower() in split_folder_names for d in subdirs):
        jobs: dict[str, list[Path]] = {}
        for d in subdirs:
            cname = normalize_entity_name(d.name)
            imgs = collect_images_in_folder(d, max_images=max_images_per_class)
            if imgs:
                jobs[cname] = imgs
        if jobs:
            return jobs

    # Fallback: infer target class from the folder name itself
    inferred_class = normalize_entity_name(input_path.name)
    imgs = collect_images_in_folder(input_path, max_images=max_images_per_class)
    return {inferred_class: imgs}


# ── Main Class Pipeline Execution ─────────────────────────────────────────────
def process_class_images(
    reasoner: OneThinkerReasoner,
    target_class: str,
    image_paths: list[Path],
    output_root: Path,
    save_annotated: bool = True,
    overwrite_existing: bool = False,
) -> Path:
    class_dir = output_root / target_class
    class_dir.mkdir(parents=True, exist_ok=True)
    json_path = class_dir / f"{target_class}_typicality.json"
    annotated_dir = class_dir / "annotated_images"

    # Load existing observations for this class if appending/updating
    existing_by_path: dict[str, dict[str, Any]] = {}
    if json_path.exists() and not overwrite_existing:
        try:
            prev_data = json.loads(json_path.read_text(encoding="utf-8"))
            for obs in prev_data.get("per_image_observations") or []:
                if obs.get("image_path"):
                    existing_by_path[obs["image_path"]] = obs
        except Exception as exc:
            LOG.warning("Could not load existing JSON at %s: %s", json_path, exc)

    total = len(image_paths)
    LOG.info("=== Processing Class '%s' (%d images) -> %s ===", target_class, total, class_dir)

    for idx, img_path in enumerate(image_paths, start=1):
        resolved_key = str(img_path.resolve())
        if resolved_key in existing_by_path and not overwrite_existing:
            LOG.info("  [%d/%d] Skipping already-analyzed image: %s", idx, total, img_path.name)
            continue

        LOG.info("  [%d/%d] Reasoning with OneThinker on: %s", idx, total, img_path.name)
        try:
            obs = reasoner.analyze_image(img_path, target_class=target_class)
            existing_by_path[resolved_key] = obs

            if save_annotated:
                save_annotated_grounding_image(img_path, obs, annotated_dir)

            # Save incrementally after each image so progress is never lost
            all_obs = list(existing_by_path.values())
            knowledge = aggregate_class_typicality(
                target_class=target_class,
                model_id=reasoner.model_id,
                observations=all_obs,
            )
            json_path.write_text(json.dumps(knowledge, indent=2, ensure_ascii=False), encoding="utf-8")
            write_human_readable_reports(class_dir, knowledge)

            pa = obs.get("primary_anchor_interaction") or {}
            LOG.info(
                "         -> Result: %s | Summary: %s",
                pa.get("triplet", "N/A"),
                (obs.get("human_readable_summary") or "")[:110] + "...",
            )
        except Exception as exc:
            LOG.error("  [%d/%d] Failed on image %s: %s", idx, total, img_path, exc)

    # Final aggregation & report write even if all images were cached
    all_obs = list(existing_by_path.values())
    knowledge = aggregate_class_typicality(
        target_class=target_class,
        model_id=reasoner.model_id,
        observations=all_obs,
    )
    json_path.write_text(json.dumps(knowledge, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path, txt_path = write_human_readable_reports(class_dir, knowledge)

    LOG.info("Completed class '%s':", target_class)
    LOG.info("  Machine-readable JSON : %s", json_path)
    LOG.info("  Human-readable Report : %s", md_path)
    LOG.info("  Plain-text Report     : %s", txt_path)
    return class_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract machine-readable and human-readable object typicality & spatial reasoning using OneThinker."
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Path to an image file, a folder of images for a class, or a root dataset folder containing multiple class folders.",
    )
    parser.add_argument(
        "--target-class", "-c",
        type=str,
        default=None,
        help="Target object class name (e.g. 'security_camera', 'keyboard', 'exit_sign'). "
             "If omitted on a multi-class folder, all class subfolders are processed.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Root directory where per-class folders will be stored (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument(
        "--model-id", "-m",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"HuggingFace model ID or local path for OneThinker (default: {DEFAULT_MODEL_ID}).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=50,
        help="Maximum number of images to process per class (default: 50; set 0 for unlimited).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1536,
        help="Maximum tokens generated per image (default: 1536).",
    )
    parser.add_argument(
        "--no-annotated-images",
        action="store_true",
        help="Disable saving visual bounding-box overlay images inside each class folder.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run and overwrite existing per-image records instead of resuming/appending.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args(argv)

    max_imgs = args.max_images if (args.max_images and args.max_images > 0) else None
    jobs = resolve_class_image_jobs(
        input_path=args.input,
        target_class=args.target_class,
        max_images_per_class=max_imgs,
    )

    if not jobs or all(len(v) == 0 for v in jobs.values()):
        LOG.error("No images found to process under: %s", args.input)
        return 1

    LOG.info("Resolved %d target class(es): %s", len(jobs), ", ".join(f"{k} ({len(v)} imgs)" for k, v in jobs.items()))

    reasoner = OneThinkerReasoner(
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
    )

    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    for cname, img_list in jobs.items():
        if not img_list:
            continue
        process_class_images(
            reasoner=reasoner,
            target_class=cname,
            image_paths=img_list,
            output_root=output_root,
            save_annotated=not args.no_annotated_images,
            overwrite_existing=args.overwrite,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
