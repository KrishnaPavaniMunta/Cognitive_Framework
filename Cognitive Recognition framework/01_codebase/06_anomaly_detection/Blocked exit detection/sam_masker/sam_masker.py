from __future__ import annotations
import gc
import numpy as np
import cv2
import torch
from segment_anything import sam_model_registry, SamPredictor
from typing import Any, Dict, List, Optional, Sequence, Tuple


class SAMMasker:
    """
    SAM per mask da bbox con:
    - multi-scale crop
    - prompt positivi guidati dalla depth: seed sui punti più vicini
    - anti-wall: i seed sono scelti solo nella parte interna della bbox
    - connected component depth-front selezionata con prior centrale
    - negativi da punti più lontani + contesto esterno
    - custom scoring delle candidate masks
    - connected-components finali filtrate dai punti positivi
    """

    def __init__(
        self,
        predictor: SamPredictor | None = None,
        *,
        ckpt_path: str | None = None,
        model_type: str = "vit_l",
        device: str | None = None,
        # crop/preprocess params
        pad: int = 40,
        pad_scales=(1.0, 0.55),
        gamma: float = 1.8,
        clahe_clip: float = 2.0,
        clahe_grid=(8, 8),
        unsharp_sigma: float = 1.0,
        unsharp_amount: float = 0.5,
        # prompt geometry
        points_margin: int = 20,
        inner_box_frac_x: float = 0.12,
        inner_box_frac_y: float = 0.12,
        # sanity
        min_pixels: int = 80,
        # quality gating
        min_area_ratio: float = 0.05,
        max_area_ratio: float = 0.98,
        min_iou: float = 0.10,
        min_inside_box_ratio: float = 0.72,
        # depth-guided seed params
        enable_depth_prompt: bool = True,
        depth_seed_percentile: float = 10.0,
        depth_seed_band_abs_m: float = 0.08,
        depth_bg_percentile: float = 80.0,
        depth_bg_band_abs_m: float = 0.10,
        depth_min_valid_pixels: int = 25,
        depth_max_pos_points: int = 4,
        depth_max_neg_points: int = 5,
        depth_use_two_front_packs: bool = True,
        # front component filtering
        front_border_margin_px: int = 6,
        front_prefer_center: float = 10.0,
        front_penalize_border_touch: float = 25.0,
        front_min_comp_area: int = 25,
        front_thin_fill_ratio: float = 0.12,
        # legacy fallback
        enable_retry: bool = True,
        retry_pos_cy_frac: float = 0.30,
        retry_add_bottom_negative: bool = True,
    ):
        self.pad = int(pad)
        self.pad_scales = tuple(float(s) for s in pad_scales)

        self.gamma = float(gamma)
        self.clahe_clip = float(clahe_clip)
        self.clahe_grid = tuple(clahe_grid)
        self.unsharp_sigma = float(unsharp_sigma)
        self.unsharp_amount = float(unsharp_amount)

        self.points_margin = int(points_margin)
        self.inner_box_frac_x = float(inner_box_frac_x)
        self.inner_box_frac_y = float(inner_box_frac_y)

        self.min_pixels = int(min_pixels)

        self.min_area_ratio = float(min_area_ratio)
        self.max_area_ratio = float(max_area_ratio)
        self.min_iou = float(min_iou)
        self.min_inside_box_ratio = float(min_inside_box_ratio)

        self.enable_depth_prompt = bool(enable_depth_prompt)
        self.depth_seed_percentile = float(depth_seed_percentile)
        self.depth_seed_band_abs_m = float(depth_seed_band_abs_m)
        self.depth_bg_percentile = float(depth_bg_percentile)
        self.depth_bg_band_abs_m = float(depth_bg_band_abs_m)
        self.depth_min_valid_pixels = int(depth_min_valid_pixels)
        self.depth_max_pos_points = int(depth_max_pos_points)
        self.depth_max_neg_points = int(depth_max_neg_points)
        self.depth_use_two_front_packs = bool(depth_use_two_front_packs)

        self.front_border_margin_px = int(front_border_margin_px)
        self.front_prefer_center = float(front_prefer_center)
        self.front_penalize_border_touch = float(front_penalize_border_touch)
        self.front_min_comp_area = int(front_min_comp_area)
        self.front_thin_fill_ratio = float(front_thin_fill_ratio)

        self.enable_retry = bool(enable_retry)
        self.retry_pos_cy_frac = float(retry_pos_cy_frac)
        self.retry_add_bottom_negative = bool(retry_add_bottom_negative)

        self.ckpt_path = ckpt_path
        self.model_type = model_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._external_predictor = predictor is not None
        self.predictor: SamPredictor | None = predictor
        self._sam = None  # type: ignore

        if self.predictor is None and self.ckpt_path is None:
            raise ValueError("Devi passare predictor oppure ckpt_path (per lazy load).")

    # ----------------------------
    # Loading / Unloading
    # ----------------------------
    @property
    def is_loaded(self) -> bool:
        return self.predictor is not None

    def ensure_loaded(self) -> None:
        if self.predictor is not None:
            return
        if self.ckpt_path is None:
            raise RuntimeError("ckpt_path non impostato: impossibile lazy-load SAM.")

        sam = sam_model_registry[self.model_type](checkpoint=self.ckpt_path)
        sam.to(device=self.device)
        self._sam = sam
        self.predictor = SamPredictor(sam)

    def unload(self) -> None:
        if self._external_predictor:
            return

        self.predictor = None
        if self._sam is not None:
            try:
                self._sam.to("cpu")
            except Exception:
                pass
        self._sam = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

    # ----------------------------
    # Public API
    # ----------------------------
    def segment(
        self,
        image_rgb: np.ndarray,
        box_xyxy,
        label: str | None = None,
        depth_map: np.ndarray | None = None,
    ):
        """
        Ritorna (mask_bool_fullres or None, info_dict).
        La selezione dei seed positivi è guidata dalla depth:
        punti con depth minore = punti più vicini alla camera,
        ma solo nella parte interna della bbox per ridurre l'aggancio al muro.
        """
        self.ensure_loaded()
        assert self.predictor is not None, "predictor non inizializzato"

        if image_rgb is None or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            return None, {"reason": "bad_image"}

        H, W = image_rgb.shape[:2]
        box = self.clip_box(box_xyxy, W, H)

        if depth_map is not None:
            if depth_map.ndim != 2 or depth_map.shape[0] != H or depth_map.shape[1] != W:
                depth_map = None

        all_candidates = []

        for pad_scale in self.pad_scales:
            try:
                cands = self._segment_candidates_for_pad(
                    image_rgb=image_rgb,
                    box_xyxy=box,
                    pad_scale=pad_scale,
                    depth_map=depth_map,
                )
                all_candidates.extend(cands)
            except Exception as e:
                all_candidates.append({
                    "mask": None,
                    "info": {
                        "tag": f"pad_{pad_scale:.2f}",
                        "reason": f"candidate_error:{type(e).__name__}",
                        "passes_gate": False,
                        "combined_score": -1.0,
                    }
                })

        if not all_candidates:
            return None, {"reason": "no_candidates", "passes_gate": False}

        best = self._pick_best_candidate(all_candidates)

        if best["mask"] is None or not best["info"].get("passes_gate", False):
            return None, best["info"]

        return best["mask"], best["info"]

    # ----------------------------
    # Candidate generation
    # ----------------------------
    def _segment_candidates_for_pad(
        self,
        *,
        image_rgb: np.ndarray,
        box_xyxy: np.ndarray,
        pad_scale: float,
        depth_map: np.ndarray | None = None,
    ):
        if self.predictor is None:
            return [{
                "mask": None,
                "info": {
                    "tag": f"pad_{pad_scale:.2f}",
                    "reason": "predictor_not_loaded",
                    "passes_gate": False,
                    "combined_score": -1.0,
                }
            }]

        H, W = image_rgb.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box_xyxy]

        pad_cur = max(4, int(round(self.pad * pad_scale)))

        cx1 = max(0, x1 - pad_cur)
        cy1 = max(0, y1 - pad_cur)
        cx2 = min(W, x2 + pad_cur)
        cy2 = min(H, y2 + pad_cur)

        crop = image_rgb[cy1:cy2, cx1:cx2].copy()
        crop = self._preprocess_for_sam(crop)
        crop = self._unsharp_mask(crop)

        self.predictor.set_image(crop)

        crop_box = np.array([x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1], dtype=np.int32)
        cH, cW = crop.shape[:2]

        prompt_packs = []

        if self.enable_depth_prompt and depth_map is not None:
            depth_crop = depth_map[cy1:cy2, cx1:cx2]
            depth_packs = self._make_depth_seed_prompt_packs(
                depth_crop=depth_crop,
                crop_box=crop_box,
                W=cW,
                H=cH,
                margin=self.points_margin,
            )
            prompt_packs.extend(depth_packs)

        if not prompt_packs:
            prompt_packs.append(
                self._make_fallback_center_pack(
                    crop_box=crop_box,
                    W=cW,
                    H=cH,
                    margin=self.points_margin,
                )
            )

        candidates = []

        for pack in prompt_packs:
            pts = pack["points"]
            lbs = pack["labels"]
            tag = f"pad_{pad_scale:.2f}|{pack['tag']}"

            try:
                masks, scores, _ = self.predictor.predict(
                    box=crop_box[None, :],
                    point_coords=pts,
                    point_labels=lbs,
                    multimask_output=True,
                )
            except Exception as e:
                candidates.append({
                    "mask": None,
                    "info": {
                        "tag": tag,
                        "reason": f"predict_error:{type(e).__name__}",
                        "passes_gate": False,
                        "combined_score": -1.0,
                    }
                })
                continue

            pos_pts = pts[lbs == 1] if np.any(lbs == 1) else np.empty((0, 2), dtype=np.float32)

            for mi in range(len(masks)):
                mask_local = masks[mi].astype(np.uint8)
                mask_local = self._keep_components_supported_by_positive_points(mask_local, pos_pts)

                full = np.zeros((H, W), dtype=bool)
                full[cy1:cy2, cx1:cx2] = mask_local.astype(bool)

                info = self._compute_quality(
                    mask_bool=full,
                    box_xyxy=box_xyxy,
                    sam_score=float(scores[mi]),
                    tag=tag,
                    prompt_tag=pack["tag"],
                )
                info["pad_scale"] = float(pad_scale)
                info["prompt_tag"] = pack["tag"]

                if pack["tag"].startswith("depth_"):
                    info["combined_score"] += 0.06

                if info["area"] < self.min_pixels:
                    info["passes_gate"] = False
                    info["reason"] = "too_small"

                candidates.append({
                    "mask": full if info["area"] >= self.min_pixels else None,
                    "info": info,
                })

        return candidates

    # ----------------------------
    # Prompt packs
    # ----------------------------
    def _make_depth_seed_prompt_packs(
        self,
        *,
        depth_crop: np.ndarray,
        crop_box: np.ndarray,
        W: int,
        H: int,
        margin: int,
    ):
        x1, y1, x2, y2 = [int(v) for v in crop_box]
        if x2 <= x1 or y2 <= y1:
            return []

        roi = depth_crop[y1:y2, x1:x2]
        if roi.size == 0:
            return []

        inner_margin_x = max(3, int(self.inner_box_frac_x * max(1, x2 - x1)))
        inner_margin_y = max(3, int(self.inner_box_frac_y * max(1, y2 - y1)))

        inner = np.zeros_like(roi, dtype=bool)
        iy1 = inner_margin_y
        iy2 = max(iy1 + 1, roi.shape[0] - inner_margin_y)
        ix1 = inner_margin_x
        ix2 = max(ix1 + 1, roi.shape[1] - inner_margin_x)
        inner[iy1:iy2, ix1:ix2] = True

        valid = np.isfinite(roi) & (roi > 0) & inner
        if int(valid.sum()) < self.depth_min_valid_pixels:
            return []

        zvals = roi[valid].astype(np.float32)
        z_front = float(np.percentile(zvals, self.depth_seed_percentile))

        fg_tight = valid & (roi <= (z_front + self.depth_seed_band_abs_m))
        fg_tight = self._largest_or_central_component(
            fg_tight,
            border_margin_px=self.front_border_margin_px,
            prefer_center=self.front_prefer_center,
            penalize_border_touch=self.front_penalize_border_touch,
            min_comp_area=self.front_min_comp_area,
            thin_fill_ratio=self.front_thin_fill_ratio,
        )

        packs = []

        pack1 = self._build_prompt_pack_from_fg_bg(
            fg_mask=fg_tight,
            valid_mask=valid,
            roi_depth=roi,
            crop_box=crop_box,
            W=W,
            H=H,
            margin=margin,
            tag="depth_front_tight",
        )
        if pack1 is not None:
            packs.append(pack1)

        if self.depth_use_two_front_packs:
            fg_loose = valid & (roi <= (z_front + 1.8 * self.depth_seed_band_abs_m))
            fg_loose = self._largest_or_central_component(
                fg_loose,
                border_margin_px=self.front_border_margin_px,
                prefer_center=self.front_prefer_center,
                penalize_border_touch=self.front_penalize_border_touch,
                min_comp_area=self.front_min_comp_area,
                thin_fill_ratio=self.front_thin_fill_ratio,
            )

            pack2 = self._build_prompt_pack_from_fg_bg(
                fg_mask=fg_loose,
                valid_mask=valid,
                roi_depth=roi,
                crop_box=crop_box,
                W=W,
                H=H,
                margin=margin,
                tag="depth_front_loose",
            )
            if pack2 is not None:
                packs.append(pack2)

        return packs

    def _build_prompt_pack_from_fg_bg(
        self,
        *,
        fg_mask: np.ndarray,
        valid_mask: np.ndarray,
        roi_depth: np.ndarray,
        crop_box: np.ndarray,
        W: int,
        H: int,
        margin: int,
        tag: str,
    ):
        x1, y1, x2, y2 = [int(v) for v in crop_box]
        if fg_mask is None or int(fg_mask.sum()) == 0:
            return None

        pos_pts = self._sample_points_from_depth_front(
            fg_mask=fg_mask,
            roi_depth=roi_depth,
            x_offset=x1,
            y_offset=y1,
            max_points=self.depth_max_pos_points,
        )
        if len(pos_pts) == 0:
            return None

        far_valid = valid_mask & (~fg_mask)
        neg_pts = self._sample_points_from_depth_back(
            candidate_mask=far_valid,
            roi_depth=roi_depth,
            x_offset=x1,
            y_offset=y1,
            max_points=self.depth_max_neg_points,
            back_percentile=self.depth_bg_percentile,
            back_band_abs_m=self.depth_bg_band_abs_m,
        )

        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        def clamp_pt(x, y):
            return [int(np.clip(x, 0, W - 1)), int(np.clip(y, 0, H - 1))]

        ext_negs = [
            clamp_pt(x1 - margin, y1 + 0.5 * bh),
            clamp_pt(x2 + margin, y1 + 0.5 * bh),
            clamp_pt(x1 + 0.5 * bw, y1 - margin),
            clamp_pt(x1 + 0.5 * bw, y2 + margin),
        ]

        pts = []
        lbs = []

        for p in pos_pts:
            pts.append(clamp_pt(p[0], p[1]))
            lbs.append(1)

        for p in neg_pts:
            pts.append(clamp_pt(p[0], p[1]))
            lbs.append(0)

        for p in ext_negs:
            pts.append(p)
            lbs.append(0)

        return {
            "tag": tag,
            "points": np.array(pts, dtype=np.float32),
            "labels": np.array(lbs, dtype=np.int32),
        }

    @staticmethod
    def _make_fallback_center_pack(
        *,
        crop_box: np.ndarray,
        W: int,
        H: int,
        margin: int,
    ):
        x1, y1, x2, y2 = [int(v) for v in crop_box]
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        def clamp_pt(x, y):
            return [int(np.clip(x, 0, W - 1)), int(np.clip(y, 0, H - 1))]

        ext_negs = [
            clamp_pt(x1 - margin, y1 + 0.5 * bh),
            clamp_pt(x2 + margin, y1 + 0.5 * bh),
            clamp_pt(x1 + 0.5 * bw, y1 - margin),
            clamp_pt(x1 + 0.5 * bw, y2 + margin),
        ]

        pts = [clamp_pt(x1 + 0.5 * bw, y1 + 0.5 * bh)] + ext_negs
        lbs = [1, 0, 0, 0, 0]

        return {
            "tag": "fallback_center",
            "points": np.array(pts, dtype=np.float32),
            "labels": np.array(lbs, dtype=np.int32),
        }

    @staticmethod
    def _largest_or_central_component(
        mask_bool: np.ndarray,
        *,
        border_margin_px: int = 6,
        prefer_center: float = 8.0,
        penalize_border_touch: float = 20.0,
        min_comp_area: int = 20,
        thin_fill_ratio: float = 0.12,
    ) -> np.ndarray:
        if mask_bool is None or mask_bool.size == 0:
            return mask_bool

        mask_u8 = mask_bool.astype(np.uint8)
        num, cc = cv2.connectedComponents(mask_u8)
        if num <= 1:
            return mask_bool

        H, W = mask_bool.shape[:2]
        best_score = None
        best_cid = None

        for cid in range(1, num):
            comp = (cc == cid)
            area = int(comp.sum())
            if area < min_comp_area:
                continue

            ys, xs = np.where(comp)
            if len(xs) == 0:
                continue

            x1 = int(xs.min())
            x2 = int(xs.max())
            y1 = int(ys.min())
            y2 = int(ys.max())

            cx = float(xs.mean())
            cy = float(ys.mean())

            dx = abs(cx - (W - 1) / 2.0) / max(1.0, W / 2.0)
            dy = abs(cy - (H - 1) / 2.0) / max(1.0, H / 2.0)
            center_penalty = prefer_center * (dx + dy)

            touches_border = (
                (x1 <= border_margin_px)
                or (y1 <= border_margin_px)
                or (x2 >= W - 1 - border_margin_px)
                or (y2 >= H - 1 - border_margin_px)
            )
            border_penalty = penalize_border_touch if touches_border else 0.0

            bw = max(1, x2 - x1 + 1)
            bh = max(1, y2 - y1 + 1)
            fill_ratio = area / float(bw * bh)
            thin_penalty = 8.0 if fill_ratio < thin_fill_ratio else 0.0

            score = area - center_penalty - border_penalty - thin_penalty

            if best_score is None or score > best_score:
                best_score = score
                best_cid = cid

        if best_cid is None:
            return np.zeros_like(mask_bool, dtype=bool)

        return (cc == best_cid)

    @staticmethod
    def _sample_points_from_depth_front(
        fg_mask: np.ndarray,
        roi_depth: np.ndarray,
        *,
        x_offset: int,
        y_offset: int,
        max_points: int,
    ):
        ys, xs = np.where(fg_mask)
        if len(xs) == 0:
            return []

        zs = roi_depth[ys, xs].astype(np.float32)
        order = np.argsort(zs)
        xs = xs[order]
        ys = ys[order]

        n_front = min(len(xs), max(max_points * 8, 40))
        xs = xs[:n_front]
        ys = ys[:n_front]

        cx = float(xs.mean())
        cy = float(ys.mean())
        d2 = (xs - cx) ** 2 + (ys - cy) ** 2
        order_center = np.argsort(d2)

        picks = []
        used = set()

        for i in order_center:
            px = int(xs[i])
            py = int(ys[i])
            key = (px, py)
            if key in used:
                continue
            picks.append([px + x_offset, py + y_offset])
            used.add(key)
            if len(picks) >= max_points:
                break

        return picks

    @staticmethod
    def _sample_points_from_depth_back(
        candidate_mask: np.ndarray,
        roi_depth: np.ndarray,
        *,
        x_offset: int,
        y_offset: int,
        max_points: int,
        back_percentile: float,
        back_band_abs_m: float,
    ):
        ys, xs = np.where(candidate_mask)
        if len(xs) == 0:
            return []

        zs = roi_depth[ys, xs].astype(np.float32)
        z_back = float(np.percentile(zs, back_percentile))
        keep = zs >= (z_back - back_band_abs_m)

        xs = xs[keep]
        ys = ys[keep]
        zs = zs[keep]

        if len(xs) == 0:
            return []

        order = np.argsort(-zs)
        xs = xs[order]
        ys = ys[order]

        n_back = min(len(xs), max(max_points * 8, 40))
        xs = xs[:n_back]
        ys = ys[:n_back]

        return SAMMasker._pick_spread_points(
            xs=xs,
            ys=ys,
            x_offset=x_offset,
            y_offset=y_offset,
            max_points=max_points,
        )

    @staticmethod
    def _pick_spread_points(
        *,
        xs: np.ndarray,
        ys: np.ndarray,
        x_offset: int,
        y_offset: int,
        max_points: int,
    ):
        if len(xs) == 0:
            return []

        xs = xs.astype(np.float32)
        ys = ys.astype(np.float32)

        idxs = []

        cx = float(xs.mean())
        cy = float(ys.mean())
        d2 = (xs - cx) ** 2 + (ys - cy) ** 2
        idxs.append(int(np.argmin(d2)))

        idxs.append(int(np.argmin(xs)))
        idxs.append(int(np.argmax(xs)))
        idxs.append(int(np.argmin(ys)))
        idxs.append(int(np.argmax(ys)))

        uniq = []
        seen = set()
        for i in idxs:
            if i not in seen:
                uniq.append(i)
                seen.add(i)

        uniq = uniq[:max_points]

        pts = []
        for i in uniq:
            pts.append([int(xs[i] + x_offset), int(ys[i] + y_offset)])

        return pts

    # ----------------------------
    # Quality / ranking
    # ----------------------------
    def _compute_quality(
        self,
        mask_bool: np.ndarray,
        box_xyxy: np.ndarray,
        sam_score: float,
        *,
        tag: str,
        prompt_tag: str = "",
    ):
        x1, y1, x2, y2 = [int(v) for v in box_xyxy]

        area = int(mask_bool.sum())
        bbox_area = int(max(1, (x2 - x1) * (y2 - y1)))
        area_ratio = float(area / bbox_area)

        iou = float(self.mask_box_iou(mask_bool, box_xyxy))
        inside_ratio = float(self.mask_inside_box_ratio(mask_bool, box_xyxy))
        n_comp = int(self.count_components(mask_bool))
        touches_crop_like = int(self.touches_box_border_heavily(mask_bool, box_xyxy))

        area_term = self._triangular_score(area_ratio, center=0.35, halfwidth=0.35)
        iou_term = np.clip(iou, 0.0, 1.0)
        inside_term = np.clip(inside_ratio, 0.0, 1.0)
        comp_term = 1.0 / max(1, n_comp)
        border_penalty = 0.15 if touches_crop_like else 0.0

        oversize_penalty = 0.0
        if area_ratio > 0.75:
            oversize_penalty = 0.20 * min(1.0, (area_ratio - 0.75) / 0.25)

        outside_penalty = 0.0
        if inside_ratio < 0.75:
            outside_penalty = 0.25 * (0.75 - inside_ratio) / 0.75

        combined = (
            0.34 * float(sam_score)
            + 0.22 * float(iou_term)
            + 0.14 * float(area_term)
            + 0.22 * float(inside_term)
            + 0.05 * float(comp_term)
            - float(border_penalty)
            - float(oversize_penalty)
            - float(outside_penalty)
        )

        passes = (
            (area >= self.min_pixels)
            and (self.min_area_ratio <= area_ratio <= self.max_area_ratio)
            and (iou >= self.min_iou)
            and (inside_ratio >= self.min_inside_box_ratio)
        )

        return {
            "tag": tag,
            "prompt_tag": prompt_tag,
            "area": area,
            "bbox_area": bbox_area,
            "area_ratio": area_ratio,
            "mask_box_iou": iou,
            "inside_box_ratio": inside_ratio,
            "n_components": n_comp,
            "sam_score": float(sam_score),
            "combined_score": float(combined),
            "passes_gate": bool(passes),
        }

    @staticmethod
    def _triangular_score(x: float, center: float, halfwidth: float) -> float:
        if halfwidth <= 1e-9:
            return 0.0
        v = 1.0 - abs(float(x) - float(center)) / float(halfwidth)
        return float(np.clip(v, 0.0, 1.0))

    def _pick_best_candidate(self, candidates):
        best = None
        best_key = None

        for c in candidates:
            info = c["info"]
            key = (
                1 if info.get("passes_gate", False) else 0,
                float(info.get("combined_score", -1e9)),
                float(info.get("sam_score", -1e9)),
            )
            if best is None or key > best_key:
                best = c
                best_key = key

        return best

    # ----------------------------
    # Utilities
    # ----------------------------
    @staticmethod
    def clip_box(box_xyxy, W: int, H: int) -> np.ndarray:
        x1, y1, x2, y2 = box_xyxy
        x1 = int(max(0, min(W - 1, round(float(x1)))))
        y1 = int(max(0, min(H - 1, round(float(y1)))))
        x2 = int(max(0, min(W - 1, round(float(x2)))))
        y2 = int(max(0, min(H - 1, round(float(y2)))))
        if x2 <= x1:
            x2 = min(W - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(H - 1, y1 + 1)
        return np.array([x1, y1, x2, y2], dtype=np.int32)

    @staticmethod
    def mask_box_iou(mask_bool: np.ndarray, box_xyxy: np.ndarray) -> float:
        x1, y1, x2, y2 = [int(v) for v in box_xyxy]
        box_mask = np.zeros_like(mask_bool, dtype=bool)
        box_mask[y1:y2, x1:x2] = True
        inter = np.logical_and(mask_bool, box_mask).sum()
        union = np.logical_or(mask_bool, box_mask).sum()
        return float(inter / (union + 1e-6))

    @staticmethod
    def mask_inside_box_ratio(mask_bool: np.ndarray, box_xyxy: np.ndarray) -> float:
        x1, y1, x2, y2 = [int(v) for v in box_xyxy]
        ys, xs = np.where(mask_bool)
        if len(xs) == 0:
            return 0.0
        inside = ((xs >= x1) & (xs < x2) & (ys >= y1) & (ys < y2)).sum()
        return float(inside / max(1, len(xs)))

    @staticmethod
    def count_components(mask_bool: np.ndarray) -> int:
        mask_u8 = mask_bool.astype(np.uint8)
        num, _ = cv2.connectedComponents(mask_u8)
        return max(0, num - 1)

    @staticmethod
    def touches_box_border_heavily(mask_bool: np.ndarray, box_xyxy: np.ndarray) -> bool:
        x1, y1, x2, y2 = [int(v) for v in box_xyxy]
        if x2 <= x1 or y2 <= y1:
            return False

        crop = mask_bool[y1:y2, x1:x2]
        if crop.size == 0:
            return False

        top = crop[0, :].sum()
        bottom = crop[-1, :].sum()
        left = crop[:, 0].sum()
        right = crop[:, -1].sum()

        border_sum = top + bottom + left + right
        perim = max(1, 2 * crop.shape[0] + 2 * crop.shape[1] - 4)
        return (border_sum / perim) > 0.6

    def _preprocess_for_sam(self, img_rgb: np.ndarray) -> np.ndarray:
        img = img_rgb.astype(np.float32) / 255.0
        img = np.power(img, 1.0 / max(1e-6, self.gamma))
        img = (img * 255).clip(0, 255).astype(np.uint8)

        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        L, A, B = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=self.clahe_grid)
        L2 = clahe.apply(L)
        lab2 = cv2.merge([L2, A, B])
        out = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
        return out

    def _unsharp_mask(self, img_rgb: np.ndarray) -> np.ndarray:
        blur = cv2.GaussianBlur(img_rgb, (0, 0), self.unsharp_sigma)
        sharp = cv2.addWeighted(img_rgb, 1 + self.unsharp_amount, blur, -self.unsharp_amount, 0)
        return sharp

    @staticmethod
    def _keep_components_supported_by_positive_points(mask_u8: np.ndarray, pos_pts: np.ndarray) -> np.ndarray:
        if mask_u8 is None or mask_u8.size == 0:
            return mask_u8

        num, cc = cv2.connectedComponents(mask_u8.astype(np.uint8))
        if num <= 1:
            return mask_u8

        keep_ids = set()
        H, W = cc.shape[:2]

        for p in pos_pts:
            px = int(np.clip(round(float(p[0])), 0, W - 1))
            py = int(np.clip(round(float(p[1])), 0, H - 1))
            cid = int(cc[py, px])
            if cid != 0:
                keep_ids.add(cid)

        if not keep_ids:
            areas = []
            for cid in range(1, num):
                areas.append((int((cc == cid).sum()), cid))
            if not areas:
                return mask_u8
            _, cid_best = max(areas)
            return (cc == cid_best).astype(np.uint8)

        out = np.zeros_like(mask_u8, dtype=np.uint8)
        for cid in keep_ids:
            out[cc == cid] = 1
        return out





class DetectionsRefinements:
    """
    Utility helpers per raffinamento detection/segmentazione.

    Use cases principali:
    - capire se un bbox è contenuto in un altro
    - ordinare detection per profondità
    - costruire exclusion masks per togliere oggetti in foreground
      dalla segmentazione di oggetti background-like (es. wall)
    - applicare sottrazione di bbox o mask già calcolate
    """

    DEFAULT_BACKGROUND_LIKE = {
        "wall",
        "floor",
        "door",
        "window",
        "ceiling",
    }

    # ------------------------------------------------------------------
    # Basic geometry
    # ------------------------------------------------------------------
    @staticmethod
    def clip_box_xyxy(box: Sequence[float], W: int, H: int) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = box
        x1 = int(np.clip(np.floor(x1), 0, W - 1))
        y1 = int(np.clip(np.floor(y1), 0, H - 1))
        x2 = int(np.clip(np.ceil(x2),  0, W - 1))
        y2 = int(np.clip(np.ceil(y2),  0, H - 1))

        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        return x1, y1, x2, y2

    @staticmethod
    def box_area_xyxy(box: Sequence[float]) -> float:
        x1, y1, x2, y2 = box
        return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))

    @staticmethod
    def intersection_area(boxA: Sequence[float], boxB: Sequence[float]) -> float:
        ax1, ay1, ax2, ay2 = boxA
        bx1, by1, bx2, by2 = boxB

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        return float(max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1))

    @staticmethod
    def containment_ratio(inner: Sequence[float], outer: Sequence[float]) -> float:
        """
        Quanto del box 'inner' è contenuto in 'outer'.
        1.0 => inner completamente dentro outer.
        """
        a_inner = DetectionsRefinements.box_area_xyxy(inner)
        if a_inner <= 0.0:
            return 0.0
        inter = DetectionsRefinements.intersection_area(inner, outer)
        return float(inter / a_inner)

    @staticmethod
    def iou_xyxy(boxA: Sequence[float], boxB: Sequence[float]) -> float:
        inter = DetectionsRefinements.intersection_area(boxA, boxB)
        areaA = DetectionsRefinements.box_area_xyxy(boxA)
        areaB = DetectionsRefinements.box_area_xyxy(boxB)
        union = areaA + areaB - inter
        if union <= 0.0:
            return 0.0
        return float(inter / union)

    # ------------------------------------------------------------------
    # Depth helpers
    # ------------------------------------------------------------------
    @staticmethod
    def get_detection_depth_median(det: Dict[str, Any]) -> Optional[float]:
        ds = det.get("depth", None)
        if ds is None:
            return None
        z = ds.get("median", None)
        if z is None:
            return None
        try:
            z = float(z)
        except Exception:
            return None
        if not np.isfinite(z):
            return None
        return z

    @staticmethod
    def sort_detection_indices_by_depth(
        detections_depth: Sequence[Dict[str, Any]],
        near_to_far: bool = True,
    ) -> List[int]:
        """
        Ritorna gli indici ordinati per depth mediana.
        Detection senza depth vanno in fondo.
        """
        def key_fn(i: int):
            z = DetectionsRefinements.get_detection_depth_median(detections_depth[i])
            return float("inf") if z is None else z

        order = sorted(range(len(detections_depth)), key=key_fn)
        if not near_to_far:
            valid = [i for i in order if DetectionsRefinements.get_detection_depth_median(detections_depth[i]) is not None]
            invalid = [i for i in order if DetectionsRefinements.get_detection_depth_median(detections_depth[i]) is None]
            order = valid[::-1] + invalid
        return order

    # ------------------------------------------------------------------
    # Label policy
    # ------------------------------------------------------------------
    @staticmethod
    def normalize_label(label: Optional[str]) -> str:
        if label is None:
            return "unknown"
        return str(label).strip().lower()

    @staticmethod
    def is_background_like(
        label: Optional[str],
        background_like_labels: Optional[set[str]] = None,
    ) -> bool:
        bg = background_like_labels or DetectionsRefinements.DEFAULT_BACKGROUND_LIKE
        return DetectionsRefinements.normalize_label(label) in bg

    # ------------------------------------------------------------------
    # Box relation predicates
    # ------------------------------------------------------------------
    @staticmethod
    def is_box_inside(
        inner_box: Sequence[float],
        outer_box: Sequence[float],
        contain_thr: float = 0.85,
    ) -> bool:
        return DetectionsRefinements.containment_ratio(inner_box, outer_box) >= contain_thr

    @staticmethod
    def is_in_front(
        z_front: Optional[float],
        z_back: Optional[float],
        front_margin_m: float = 0.20,
    ) -> bool:
        """
        True se l'oggetto 'front' è significativamente più vicino del 'back'.
        """
        if z_front is None or z_back is None:
            return False
        return z_front < (z_back - front_margin_m)

    # ------------------------------------------------------------------
    # Exclusion masks from boxes
    # ------------------------------------------------------------------
    @staticmethod
    def build_box_exclusion_masks(
        detections_depth: Sequence[Dict[str, Any]],
        H: int,
        W: int,
        *,
        contain_thr: float = 0.85,
        front_margin_m: float = 0.20,
        only_background_like: bool = True,
        background_like_labels: Optional[set[str]] = None,
    ) -> List[np.ndarray]:
        """
        Per ogni detection i costruisce una mask booleana HxW dei pixel da escludere
        usando i bbox di detection interne e più vicine.

        Tipicamente utile per:
        - wall dietro
        - chair davanti
        => i pixel della chair vengono esclusi dalla regione del wall
        """
        bg_labels = background_like_labels or DetectionsRefinements.DEFAULT_BACKGROUND_LIKE

        clipped_boxes: List[Tuple[int, int, int, int]] = [
            DetectionsRefinements.clip_box_xyxy(d["box"], W, H)
            for d in detections_depth
        ]
        depths: List[Optional[float]] = [
            DetectionsRefinements.get_detection_depth_median(d)
            for d in detections_depth
        ]
        labels: List[str] = [
            DetectionsRefinements.normalize_label(d.get("label"))
            for d in detections_depth
        ]

        excl = [np.zeros((H, W), dtype=bool) for _ in range(len(detections_depth))]

        for i in range(len(detections_depth)):
            if only_background_like and labels[i] not in bg_labels:
                continue

            zi = depths[i]
            if zi is None:
                continue

            box_i = clipped_boxes[i]

            for j in range(len(detections_depth)):
                if i == j:
                    continue

                zj = depths[j]
                if zj is None:
                    continue

                box_j = clipped_boxes[j]

                inside = DetectionsRefinements.is_box_inside(
                    inner_box=box_j,
                    outer_box=box_i,
                    contain_thr=contain_thr,
                )
                if not inside:
                    continue

                if not DetectionsRefinements.is_in_front(
                    z_front=zj,
                    z_back=zi,
                    front_margin_m=front_margin_m,
                ):
                    continue

                x1, y1, x2, y2 = box_j
                excl[i][y1:y2, x1:x2] = True

        return excl

    # ------------------------------------------------------------------
    # Exclusion masks from already-computed masks
    # ------------------------------------------------------------------
    @staticmethod
    def build_dynamic_exclusion_from_computed_masks(
        target_idx: int,
        detections_depth: Sequence[Dict[str, Any]],
        computed_masks: Sequence[Optional[np.ndarray]],
        H: int,
        W: int,
        *,
        contain_thr: float = 0.85,
        front_margin_m: float = 0.20,
        only_background_like: bool = True,
        background_like_labels: Optional[set[str]] = None,
    ) -> np.ndarray:
        """
        Costruisce una exclusion mask per la detection `target_idx`
        unendo le mask già calcolate degli oggetti davanti e contenuti nel suo bbox.

        Questa è preferibile rispetto alla sottrazione tramite bbox
        perché rimuove la forma reale dell'oggetto foreground e non un rettangolo.
        """
        bg_labels = background_like_labels or DetectionsRefinements.DEFAULT_BACKGROUND_LIKE

        target_det = detections_depth[target_idx]
        target_label = DetectionsRefinements.normalize_label(target_det.get("label"))
        if only_background_like and target_label not in bg_labels:
            return np.zeros((H, W), dtype=bool)

        z_target = DetectionsRefinements.get_detection_depth_median(target_det)
        if z_target is None:
            return np.zeros((H, W), dtype=bool)

        target_box = DetectionsRefinements.clip_box_xyxy(target_det["box"], W, H)
        dynamic_excl = np.zeros((H, W), dtype=bool)

        for j, mj in enumerate(computed_masks):
            if j == target_idx or mj is None:
                continue

            det_j = detections_depth[j]
            z_j = DetectionsRefinements.get_detection_depth_median(det_j)
            if not DetectionsRefinements.is_in_front(
                z_front=z_j,
                z_back=z_target,
                front_margin_m=front_margin_m,
            ):
                continue

            box_j = DetectionsRefinements.clip_box_xyxy(det_j["box"], W, H)
            inside = DetectionsRefinements.is_box_inside(
                inner_box=box_j,
                outer_box=target_box,
                contain_thr=contain_thr,
            )
            if not inside:
                continue

            dynamic_excl |= mj.astype(bool)

        return dynamic_excl

    # ------------------------------------------------------------------
    # Mask post-processing
    # ------------------------------------------------------------------
    @staticmethod
    def apply_exclusion_to_mask(
        mask: Optional[np.ndarray],
        exclusion_mask: Optional[np.ndarray],
        *,
        min_remaining_pixels: int = 1,
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Sottrae exclusion_mask da mask e ritorna:
        - new_mask oppure None se resta troppo poco
        - info dict con debug
        """
        if mask is None:
            return None, {
                "mask_in": False,
                "excluded_pixels": 0,
                "remaining_pixels": 0,
                "reason": "mask_none",
            }

        out = mask.astype(bool).copy()
        excluded_pixels = 0

        if exclusion_mask is not None:
            exclusion_mask = exclusion_mask.astype(bool)
            excluded_pixels = int(np.count_nonzero(out & exclusion_mask))
            out &= ~exclusion_mask

        remaining = int(np.count_nonzero(out))
        if remaining < int(min_remaining_pixels):
            return None, {
                "mask_in": True,
                "excluded_pixels": excluded_pixels,
                "remaining_pixels": remaining,
                "reason": "mask_fully_removed_by_exclusion",
            }

        return out, {
            "mask_in": True,
            "excluded_pixels": excluded_pixels,
            "remaining_pixels": remaining,
            "reason": "ok",
        }

    @staticmethod
    def mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            return None
        x1 = int(xs.min())
        y1 = int(ys.min())
        x2 = int(xs.max()) + 1
        y2 = int(ys.max()) + 1
        return x1, y1, x2, y2

    @staticmethod
    def subtract_box_from_mask(
        mask: Optional[np.ndarray],
        box: Sequence[float],
        H: int,
        W: int,
        *,
        min_remaining_pixels: int = 1,
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Utility veloce: crea exclusion mask rettangolare dal bbox e la sottrae.
        """
        if mask is None:
            return None, {
                "mask_in": False,
                "excluded_pixels": 0,
                "remaining_pixels": 0,
                "reason": "mask_none",
            }

        excl = np.zeros((H, W), dtype=bool)
        x1, y1, x2, y2 = DetectionsRefinements.clip_box_xyxy(box, W, H)
        excl[y1:y2, x1:x2] = True

        return DetectionsRefinements.apply_exclusion_to_mask(
            mask=mask,
            exclusion_mask=excl,
            min_remaining_pixels=min_remaining_pixels,
        )

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------
    @staticmethod
    def describe_nested_relations(
        detections_depth: Sequence[Dict[str, Any]],
        H: int,
        W: int,
        *,
        contain_thr: float = 0.85,
        front_margin_m: float = 0.20,
    ) -> List[Dict[str, Any]]:
        """
        Restituisce una lista di relazioni del tipo:
        j è dentro i ed è davanti a i
        """
        out: List[Dict[str, Any]] = []

        clipped_boxes = [
            DetectionsRefinements.clip_box_xyxy(d["box"], W, H)
            for d in detections_depth
        ]
        depths = [
            DetectionsRefinements.get_detection_depth_median(d)
            for d in detections_depth
        ]

        for i, di in enumerate(detections_depth):
            for j, dj in enumerate(detections_depth):
                if i == j:
                    continue

                inside = DetectionsRefinements.is_box_inside(
                    inner_box=clipped_boxes[j],
                    outer_box=clipped_boxes[i],
                    contain_thr=contain_thr,
                )
                if not inside:
                    continue

                front = DetectionsRefinements.is_in_front(
                    z_front=depths[j],
                    z_back=depths[i],
                    front_margin_m=front_margin_m,
                )

                out.append({
                    "outer_idx": i,
                    "outer_label": di.get("label"),
                    "outer_depth": depths[i],
                    "inner_idx": j,
                    "inner_label": dj.get("label"),
                    "inner_depth": depths[j],
                    "containment_ratio": DetectionsRefinements.containment_ratio(
                        clipped_boxes[j], clipped_boxes[i]
                    ),
                    "is_front": front,
                })

        return out

if __name__ == "__main__":
    ckpt_path = "sam_vit_l_0b3195.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    masker = SAMMasker(
        ckpt_path=ckpt_path,
        model_type="vit_l",
        device=device,
        pad=25,
        pad_scales=(1.0, 0.75),
        min_pixels=80,
        min_area_ratio=0.05,
        max_area_ratio=0.90,
        min_iou=0.12,
        min_inside_box_ratio=0.72,
        enable_depth_prompt=True,
        depth_seed_percentile=8.0,
        depth_seed_band_abs_m=0.06,
        depth_bg_percentile=90.0,
        depth_bg_band_abs_m=0.08,
        depth_max_pos_points=4,
        depth_max_neg_points=5,
        inner_box_frac_x=0.12,
        inner_box_frac_y=0.12,
        front_border_margin_px=6,
        front_prefer_center=10.0,
        front_penalize_border_touch=25.0,
        front_min_comp_area=25,
    )