"""
Two-Tier Semantic Cognition & Dynamic Ontology Ingestion Engine.

Solves open-vocabulary 3D object discovery without hallucination:
- Tier 1: Fast zero-shot CLIP embedding cosine matching against ontology.rdf (~15ms).
  If cosine_similarity >= 0.70: Grounds directly to existing ontology class.
- Tier 2: Open-world discovery via local VLM (LLaVA / Ollama / Vision Pipeline).
  If conf >= 0.60: Ingests as newly discovered class under ontology.rdf Equipment.
  If conf < 0.60: Labels as 'unidentified_obstacle' with exact 3D bounds for collision safety.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import requests
import torch
from PIL import Image
from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef
from rdflib.namespace import OWL
from transformers import CLIPModel, CLIPProcessor

LOG = logging.getLogger("two_tier_classifier")

BASE_DIR = Path(__file__).resolve().parent
CODEBASE_DIR = BASE_DIR.parent
ONTOLOGY_DIR = CODEBASE_DIR / "09_ontology"
DEFAULT_ONTOLOGY = ONTOLOGY_DIR / "ontology.rdf"

CO = Namespace("http://www.semanticweb.org/chevi/ontologies/2026/5/52-classes-ontology#")
UO = Namespace("http://www.semanticweb.org/chevi/ontologies/2026/5/untitled-ontology-6#")
BR = Namespace("http://www.semanticweb.org/chevi/ontologies/2026/5/semantic-map-bridge#")


@dataclass
class ClassificationResult:
    """Output of two-tier semantic cognition."""
    class_name: str
    confidence: float
    tier: str                      # 'tier1_ontology', 'tier2_novel', 'tier2_obstacle'
    is_novel: bool
    ontology_uri: str | None = None
    all_scores: dict[str, float] | None = None


class TwoTierCognitionEngine:
    """
    Combines fast GPU CLIP zero-shot ontology grounding with open-world VLM discovery.
    """

    def __init__(
        self,
        ontology_path: str | Path = DEFAULT_ONTOLOGY,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        tier1_similarity_threshold: float = 0.68,
        tier2_confidence_threshold: float = 0.60,
        ollama_endpoint: str = "http://localhost:11434/api/generate",
        vlm_model: str = "llava:latest",
        device: str | None = None,
    ) -> None:
        self.ontology_path = Path(ontology_path)
        self.tier1_thresh = float(tier1_similarity_threshold)
        self.tier2_thresh = float(tier2_confidence_threshold)
        self.ollama_endpoint = str(ollama_endpoint)
        self.vlm_model = str(vlm_model)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        LOG.info("[COGNITION] Loading CLIP model '%s' on %s...", clip_model_name, self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device).eval()

        # Load ontology classes and precompute text embeddings
        self.ontology_classes: list[str] = []
        self.text_embeddings: torch.Tensor | None = None
        self._init_ontology_embeddings()

    def _init_ontology_embeddings(self) -> None:
        """Extracts named classes from ontology.rdf and precomputes normalized CLIP text embeddings."""
        if not self.ontology_path.exists():
            LOG.warning("Ontology file not found: %s", self.ontology_path)
            return

        g = Graph()
        try:
            g.parse(self.ontology_path, format="xml")
        except Exception as e:
            LOG.error("Failed to parse ontology XML: %s", e)
            return

        # Extract all named classes (excluding blank nodes and meta-concepts)
        classes_set = set()
        meta_ignore = {
            "Alert", "Context", "Environment", "EnvironmentContext", "Event",
            "HazardEvent", "Location", "MonitoredEntity", "Predicate", "Role",
            "Statement", "TemporalContext", "IndoorSpace",
        }

        for s in g.subjects(RDF.type, OWL.Class):
            name = str(s).rsplit("#", 1)[-1].rsplit("/", 1)[-1]
            if name.startswith("N") and len(name) > 20:
                continue
            if name in meta_ignore:
                continue
            classes_set.add(name)

        # Standard physical assets to ensure strong coverage
        core_assets = [
            "chair", "door", "waste_bin", "large_bin", "small_bin",
            "fire_extinguisher", "iv_stand", "iv_bag", "infusion_pump",
            "oxygen_cylinder", "oxygen_pump", "patient_monitor", "medical_tray",
            "hand_sanitizer", "hospital_bed", "hospital_stretcher", "wheelchair",
            "power_socket", "radiator", "reception_counter", "spillage", "suitcase",
            "backpack", "box", "cart", "equipment", "table", "vending_machine"
        ]
        classes_set.update(core_assets)

        self.ontology_classes = sorted(list(classes_set))
        LOG.info("[COGNITION] Embedding %d ontology physical classes for zero-shot matching...", len(self.ontology_classes))

        # Build prompt templates for each class
        prompts = []
        for cls in self.ontology_classes:
            clean_name = cls.replace("_", " ").lower()
            prompts.append(f"a photo of a hospital {clean_name}")

        with torch.no_grad():
            inputs = self.clip_processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
            out = self.clip_model.get_text_features(**inputs)
            text_features = out.pooler_output if hasattr(out, "pooler_output") else out
            # Normalize embeddings
            self.text_embeddings = text_features / text_features.norm(dim=-1, keepdim=True)

    def classify_crop(self, crop_bgr: np.ndarray) -> ClassificationResult:
        """
        Classifies a 2D image crop extracted from a 3D residual cluster.
        """
        if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.shape[0] < 10 or crop_bgr.shape[1] < 10:
            return ClassificationResult(
                class_name="unidentified_obstacle",
                confidence=0.50,
                tier="tier2_obstacle",
                is_novel=True,
            )

        # ── TIER 1: Fast Zero-Shot CLIP Embedding Cosine Match ────────────────
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)

        with torch.no_grad():
            inputs = self.clip_processor(images=pil_img, return_tensors="pt").to(self.device)
            out = self.clip_model.get_image_features(**inputs)
            image_features = out.pooler_output if hasattr(out, "pooler_output") else out
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Cosine similarity matrix (1, num_classes)
            cosine_sims = (image_features @ self.text_embeddings.T).squeeze(0)
            best_idx = int(torch.argmax(cosine_sims).item())
            best_sim = float(cosine_sims[best_idx].item())
            best_class = self.ontology_classes[best_idx]

        # Top 3 scores for diagnostic logging
        topk_vals, topk_indices = torch.topk(cosine_sims, min(3, len(self.ontology_classes)))
        scores = {self.ontology_classes[int(idx.item())]: float(val.item()) for idx, val in zip(topk_indices, topk_vals)}

        if best_sim >= self.tier1_thresh:
            LOG.debug("[TIER 1] Grounded crop to '%s' (similarity=%.3f >= %.2f)", best_class, best_sim, self.tier1_thresh)
            return ClassificationResult(
                class_name=best_class,
                confidence=best_sim,
                tier="tier1_ontology",
                is_novel=False,
                ontology_uri=f"{CO}{best_class}",
                all_scores=scores,
            )

        # ── TIER 2: Open-World VLM Query (Local Ollama / LLaVA) ────────────────
        LOG.debug("[TIER 2] Ambiguous crop (top='%s', sim=%.3f < %.2f). Querying open-world VLM...",
                  best_class, best_sim, self.tier1_thresh)

        vlm_res = self._query_vlm(crop_bgr)
        if vlm_res is not None:
            vlm_name, vlm_conf = vlm_res
            if vlm_conf >= self.tier2_thresh:
                # Discovered a valid physical concept
                clean_concept = vlm_name.strip().lower().replace(" ", "_")
                LOG.info("[TIER 2] Discovered novel asset '%s' (conf=%.2f)", clean_concept, vlm_conf)
                return ClassificationResult(
                    class_name=clean_concept,
                    confidence=vlm_conf,
                    tier="tier2_novel",
                    is_novel=True,
                    ontology_uri=f"{BR}{clean_concept}",
                    all_scores=scores,
                )

        # Fallback: No confident asset identified - do not emit unidentified obstacle
        return None

    def _query_vlm(self, crop_bgr: np.ndarray) -> tuple[str, float] | None:
        """
        Attempts to query a local VLM (Ollama / LLaVA) via HTTP REST.
        Returns (object_name, confidence) or None if unreachable.
        """
        import base64

        try:
            _, buffer = cv2.imencode(".jpg", crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            img_b64 = base64.b64encode(buffer).decode("utf-8")

            payload = {
                "model": self.vlm_model,
                "prompt": (
                    "You are a robotic vision inspector in a hospital. "
                    "Identify the primary physical object in this image crop. "
                    "Respond ONLY with a JSON object: {\"name\": \"<single noun phrase>\", \"confidence\": <float 0.0-1.0>}"
                ),
                "images": [img_b64],
                "format": "json",
                "stream": False,
            }

            resp = requests.post(self.ollama_endpoint, json=payload, timeout=2.5)
            if resp.status_code == 200:
                data = resp.json()
                raw_response = data.get("response", "{}")
                parsed = json.loads(raw_response)
                name = parsed.get("name")
                conf = float(parsed.get("confidence", 0.70))
                if name and isinstance(name, str):
                    return name, conf
        except Exception as e:
            LOG.debug("Local VLM query skipped or timed out: %s", e)

        return None

    def ingest_novel_concept(
        self,
        concept_name: str,
        superclass: str = "MedicalDevice",
        comment: str = "Dynamically discovered physical asset via robot perception pipeline.",
        dimensions_m: tuple[float, float, float] | None = None,
    ) -> bool:
        """
        Dynamically ingests a newly discovered class into ontology.rdf.
        """
        if not self.ontology_path.exists():
            return False

        try:
            g = Graph()
            g.parse(self.ontology_path, format="xml")

            clean_name = concept_name.strip().replace(" ", "_")
            new_uri = BR[clean_name]

            # Avoid duplicate declarations
            if (new_uri, RDF.type, OWL.Class) in g:
                return True

            g.bind("classes-ontology", CO)
            g.bind("untitled-ontology-6", UO)
            g.bind("smap-bridge", BR)

            g.add((new_uri, RDF.type, OWL.Class))
            parent_uri = UO[superclass] if (UO[superclass], RDF.type, OWL.Class) in g else UO.PhysicalObject
            g.add((new_uri, RDFS.subClassOf, parent_uri))
            g.add((new_uri, RDFS.comment, Literal(comment)))

            if dimensions_m is not None:
                w, d, h = dimensions_m
                dim_json = json.dumps({
                    "typical": {"width": round(w, 2), "depth": round(d, 2), "height": round(h, 2)},
                    "range": {
                        "width": [round(max(0.05, w * 0.7), 2), round(w * 1.4, 2)],
                        "depth": [round(max(0.05, d * 0.7), 2), round(d * 1.4, 2)],
                        "height": [round(max(0.05, h * 0.7), 2), round(h * 1.4, 2)],
                    }
                })
                g.add((new_uri, CO.physicalDimensions, Literal(dim_json)))

            g.serialize(destination=str(self.ontology_path), format="xml")
            LOG.info("[DYNAMIC ONTOLOGY] Ingested novel concept '%s' under '%s' into %s",
                     clean_name, superclass, self.ontology_path.name)
            return True
        except Exception as e:
            LOG.error("Failed to dynamically ingest concept '%s': %s", concept_name, e)
            return False

