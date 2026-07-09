import re

import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from dino_prompts import DINO_BATCH_GROUPS, DINO_FALLBACK, DINO_MODEL_ID

_dino_processor = None
_dino_model = None


def _load_dino():
    global _dino_processor, _dino_model
    if _dino_model is None:
        print(f"  [DINO] Loading {DINO_MODEL_ID} directly on CUDA GPU...")
        _dino_processor = AutoProcessor.from_pretrained(DINO_MODEL_ID)
        _dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL_ID).to("cuda")
        _dino_model.eval()


def _normalize_text(s):
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9\s_-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _prompt_anchor(prompt):
    # First sentence is treated as the canonical anchor phrase for matching.
    return _normalize_text(str(prompt).split(".")[0])


def run_dino_fallback(pil_image, target_classes):
    _load_dino()

    if not target_classes:
        return []

    target_set = set(target_classes)
    dino_boxes = []

    for group in DINO_BATCH_GROUPS:
        active = [cls for cls in group if cls in target_set and cls in DINO_FALLBACK]
        if not active:
            continue

        cls_prompt_thresh = [(cls, *DINO_FALLBACK[cls]) for cls in active]
        combined_prompt = " ".join(cpt[1] for cpt in cls_prompt_thresh)
        min_thresh = min(cpt[2] for cpt in cls_prompt_thresh)

        inputs = _dino_processor(
            images=pil_image,
            text=combined_prompt,
            return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            outputs = _dino_model(**inputs)

        results = _dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=min_thresh,
            text_threshold=0.25,
            target_sizes=[pil_image.size[::-1]]
        )[0]

        labels_out = results.get("text_labels", results.get("labels", []))

        for box, score, label in zip(
            results["boxes"].cpu().numpy(),
            results["scores"].cpu().numpy(),
            labels_out
        ):
            label_norm = _normalize_text(label)
            if not label_norm:
                continue

            matched = None
            ambiguous = False
            for cls, prompt, thresh in cls_prompt_thresh:
                first_term = _prompt_anchor(prompt)
                if not first_term:
                    continue

                # Compare only against canonical anchor phrases to avoid cross-class collisions.
                if label_norm == first_term or label_norm in first_term or first_term in label_norm:
                    if matched is not None and matched[0] != cls:
                        ambiguous = True
                        break
                    matched = (cls, thresh)
            if matched is None or ambiguous:
                continue  # ambiguous match - drop rather than mislabel

            cls, own_thresh = matched
            if float(score) < own_thresh:
                continue  # didn't clear this specific class's own threshold

            dino_boxes.append(
                (*box.tolist(), float(score), f"[DINO] {cls}")
            )

    return dino_boxes
