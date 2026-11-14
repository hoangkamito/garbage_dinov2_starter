"""Stage 7 — Crop Preprocessing Ablation on Clean Validation (1,743 images / 3,932 GT boxes).

Tests the Crop Domain Shift hypothesis without retraining any model:
- Uses the locked Detector checkpoint (`detection/checkpoints_wastedetect/best_model.pt`)
  and locked Classifier checkpoint (`checkpoints/dinov2_garbage_final.pth`).
- Reuses cached Detector proposals (`evaluation_runs/val_grid_search_01/cache/predictions.sqlite3`)
  so Detector inference is NEVER repeated across variants.
- Caches Classifier probabilities per preprocessing variant in SQLite (`variant_crops` + `gt_variant_crops`)
  for instant resume and zero wasted GPU Compute Units.

Variants:
- `A_original_crop`: Tight bbox crop -> Resize(256) -> CenterCrop(224) -> Normalize
- `B_direct_resize`: Tight bbox crop -> Resize((224, 224)) -> Normalize
- `C_context_pad_15`: Expand bbox 15% each side (clipped to image bounds) -> Resize((224, 224)) -> Normalize
- `D_letterbox_224`: Tight bbox crop -> Aspect-ratio preserving square pad RGB(114,114,114) -> Resize((224, 224)) -> Normalize
- `E_context_pad_25_letterbox`: Expand bbox 25% each side (clipped) -> Square letterbox RGB(114,114,114) -> Resize((224, 224)) -> Normalize

Evaluates:
1. End-to-end Cascade detection via `COCOeval` (`mAP@50`, `mAP@50:95`, `AR@100`, per-class `AP50`/`AP50:95`, `F1@0.5`)
   at locked `nms_iou=0.70`, `proposal_threshold=0.0`, `cascade_pr_threshold=0.25`.
2. GT-aligned crop classification on:
   - Exact Ground-Truth bounding boxes (3,932 boxes)
   - Matched Detector proposals at IoU >= 0.50 (3,663 boxes at `nms_iou=0.70`)
3. Specific transition diagnostics: `glass -> plastic`, `paper -> cardboard`, `paper -> trash`, and Out-of-Scope (OOS) rates.
4. Strictly restricted to Validation (`valid_clean_candidate`); Test paths are forbidden.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import shutil
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision import transforms as T

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "detection") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "detection"))
if str(PROJECT_ROOT / "evaluation_runs") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "evaluation_runs"))

from evaluate_cascade import (
    as_coco_detection,
    compile_label_policy,
    detector_categories,
    detector_index_map,
    read_json,
    sha256_file,
    validate_coco,
)
from grid_metrics import metric_sweep
from threshold_cache import cascade_score, select_raw, selection_key
from waste_cascade import NORM_MEAN, NORM_STD, WasteCascade, canonical_name

from cu_budget_tracker import CUBudgetTracker, PilotTimingMeasurement
from gpu_runtime_manager import inspect_runtime


LETTERBOX_FILL_RGB = (114, 114, 114)

VARIANT_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "A_original_crop": {
        "title": "A. Original Crop (Resize 256 -> CenterCrop 224)",
        "pad_ratio": 0.0,
        "mode": "resize256_centercrop224",
        "description_vi": "Cắt sát bbox -> Resize cạnh ngắn về 256 -> CenterCrop(224) -> Normalize (cấu hình gốc).",
    },
    "B_direct_resize": {
        "title": "B. Direct Resize (224x224)",
        "pad_ratio": 0.0,
        "mode": "direct_resize224",
        "description_vi": "Cắt sát bbox -> Resize trực tiếp về (224, 224) không cắt viền CenterCrop -> Normalize.",
    },
    "C_context_pad_15": {
        "title": "C. Context Padding 15% + Resize (224x224)",
        "pad_ratio": 0.15,
        "mode": "direct_resize224",
        "description_vi": "Mở rộng bbox 15% mỗi chiều (clip theo biên ảnh) -> Resize((224, 224)) -> Normalize.",
    },
    "D_letterbox_224": {
        "title": "D. Letterbox Square Pad + Resize (224x224)",
        "pad_ratio": 0.0,
        "mode": "letterbox224",
        "description_vi": "Cắt sát bbox -> Giữ nguyên tỷ lệ khung hình, đệm viền vuông RGB(114,114,114) -> Resize(224) -> Normalize.",
    },
    "E_context_pad_25_letterbox": {
        "title": "E. Context Padding 25% + Letterbox (224x224)",
        "pad_ratio": 0.25,
        "mode": "letterbox224",
        "description_vi": "Mở rộng bbox 25% mỗi chiều (clip theo biên ảnh) -> Đệm vuông Letterbox RGB(114,114,114) -> Resize(224) -> Normalize.",
    },
}


_TF_ORIGINAL = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(NORM_MEAN, NORM_STD),
])

_TF_DIRECT_224 = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(NORM_MEAN, NORM_STD),
])

_TF_TO_TENSOR_NORM = T.Compose([
    T.ToTensor(),
    T.Normalize(NORM_MEAN, NORM_STD),
])


def expand_and_clip_xyxy(
    coords: Tuple[int, int, int, int] | List[int],
    image_size: Tuple[int, int],
    pad_ratio: float,
) -> Tuple[int, int, int, int]:
    """Expand `[x1, y1, x2, y2]` by `pad_ratio` on each side and clip to `[0, W] x [0, H]`."""
    x1, y1, x2, y2 = [float(v) for v in coords]
    W, H = float(image_size[0]), float(image_size[1])
    if pad_ratio <= 0.0:
        cx1 = max(0, min(int(W), int(round(x1))))
        cy1 = max(0, min(int(H), int(round(y1))))
        cx2 = max(cx1 + 1, min(int(W), int(round(x2))))
        cy2 = max(cy1 + 1, min(int(H), int(round(y2))))
        return (cx1, cy1, cx2, cy2)

    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    dx = bw * pad_ratio
    dy = bh * pad_ratio
    nx1 = max(0, min(int(W) - 1, int(round(x1 - dx))))
    ny1 = max(0, min(int(H) - 1, int(round(y1 - dy))))
    nx2 = max(nx1 + 1, min(int(W), int(round(x2 + dx))))
    ny2 = max(ny1 + 1, min(int(H), int(round(y2 + dy))))
    return (nx1, ny1, nx2, ny2)


def letterbox_pil_to_square_224(crop_img: Image.Image, target_size: int = 224) -> Image.Image:
    """Preserve aspect ratio, resize longest side to `target_size`, and pad shorter side symmetrically."""
    w, h = crop_img.size
    if w <= 0 or h <= 0:
        return Image.new("RGB", (target_size, target_size), LETTERBOX_FILL_RGB)
    scale = target_size / float(max(w, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = crop_img.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (target_size, target_size), LETTERBOX_FILL_RGB)
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))
    return canvas


def preprocess_crop_for_variant(
    full_image: Image.Image,
    coords: Tuple[int, int, int, int] | List[int],
    variant_name: str,
) -> torch.Tensor:
    """Apply the specified crop preprocessing variant to a single bounding box on `full_image`."""
    if variant_name not in VARIANT_DEFINITIONS:
        raise KeyError(f"Unknown crop variant: {variant_name}")
    vdef = VARIANT_DEFINITIONS[variant_name]
    pad_ratio = float(vdef["pad_ratio"])
    mode = vdef["mode"]

    if pad_ratio > 0.0:
        crop_coords = expand_and_clip_xyxy(coords, full_image.size, pad_ratio)
    else:
        crop_coords = (int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3]))

    patch = full_image.crop(crop_coords)
    if mode == "resize256_centercrop224":
        return _TF_ORIGINAL(patch)
    if mode == "direct_resize224":
        return _TF_DIRECT_224(patch)
    if mode == "letterbox224":
        lb = letterbox_pil_to_square_224(patch, target_size=224)
        return _TF_TO_TENSOR_NORM(lb)
    raise ValueError(f"Unsupported mode: {mode}")


class CropAblationCacheDB:
    """SQLite cache storing detector proposals once and classifier probabilities per variant."""

    def __init__(self, db_path: str | Path, base_val_sqlite: Optional[str | Path] = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.db_path.exists() and base_val_sqlite is not None and Path(base_val_sqlite).exists():
            shutil.copy2(base_val_sqlite, self.db_path)

        self.db = sqlite3.connect(self.db_path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS images (id INTEGER PRIMARY KEY, sha TEXT NOT NULL, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS selections (image_id INTEGER, key TEXT, data TEXT NOT NULL, PRIMARY KEY(image_id,key));
            CREATE TABLE IF NOT EXISTS crops (image_id INTEGER, coords TEXT, probs TEXT NOT NULL, PRIMARY KEY(image_id,coords));
            CREATE TABLE IF NOT EXISTS variant_crops (
                variant TEXT NOT NULL,
                image_id INTEGER NOT NULL,
                coords TEXT NOT NULL,
                probs TEXT NOT NULL,
                PRIMARY KEY(variant, image_id, coords)
            );
        """)
        # Seed A_original_crop from existing `crops` table if present
        count_a = self.db.execute(
            "SELECT COUNT(*) FROM variant_crops WHERE variant='A_original_crop'"
        ).fetchone()[0]
        count_base = self.db.execute("SELECT COUNT(*) FROM crops").fetchone()[0]
        if count_a == 0 and count_base > 0:
            with self.db:
                self.db.execute("""
                    INSERT OR IGNORE INTO variant_crops (variant, image_id, coords, probs)
                    SELECT 'A_original_crop', image_id, coords, probs FROM crops
                """)

    def close(self) -> None:
        self.db.close()

    def get_image_raw(self, image_id: int) -> Optional[Tuple[str, Dict[str, Any]]]:
        row = self.db.execute("SELECT sha, data FROM images WHERE id=?", (image_id,)).fetchone()
        return (row[0], json.loads(row[1])) if row else None

    def save_image_raw(self, image_id: int, sha: str, raw: Dict[str, Any]) -> None:
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO images VALUES (?,?,?)", (image_id, sha, json.dumps(raw)))

    def get_selection(self, image_id: int, key: str) -> Optional[List[Dict[str, Any]]]:
        row = self.db.execute("SELECT data FROM selections WHERE image_id=? AND key=?", (image_id, key)).fetchone()
        return json.loads(row[0]) if row else None

    def save_selection(self, image_id: int, key: str, rows: List[Dict[str, Any]]) -> None:
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO selections VALUES (?,?,?)", (image_id, key, json.dumps(rows)))

    def get_variant_crops(self, variant: str, image_id: int) -> Dict[Tuple[int, int, int, int], List[float]]:
        cursor = self.db.execute(
            "SELECT coords, probs FROM variant_crops WHERE variant=? AND image_id=?",
            (variant, image_id),
        )
        return {tuple(json.loads(c)): json.loads(p) for c, p in cursor}

    def save_variant_crops(
        self,
        variant: str,
        image_id: int,
        predictions: Dict[Tuple[int, int, int, int], List[float]],
    ) -> None:
        if not predictions:
            return
        with self.db:
            self.db.executemany(
                "INSERT OR REPLACE INTO variant_crops VALUES (?,?,?,?)",
                [
                    (variant, image_id, json.dumps(list(coords)), json.dumps(probs))
                    for coords, probs in predictions.items()
                ],
            )


def box_iou_matrix_xyxy(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    area1 = np.clip(boxes1[:, 2] - boxes1[:, 0], 0.0, None) * np.clip(boxes1[:, 3] - boxes1[:, 1], 0.0, None)
    area2 = np.clip(boxes2[:, 2] - boxes2[:, 0], 0.0, None) * np.clip(boxes2[:, 3] - boxes2[:, 1], 0.0, None)
    union = area1[:, None] + area2[None, :] - inter
    return inter / np.clip(union, 1e-8, None)


def verify_base_cache_fingerprint(
    fingerprint_path: str | Path,
    valid_coco_path: str | Path,
    det_ckpt_path: str | Path,
    cls_ckpt_path: str | Path,
    class_mapping_path: str | Path,
) -> Dict[str, Any]:
    """Verify SHA-256 fingerprint of the base Validation SQLite cache against current inputs."""
    fp_file = Path(fingerprint_path)
    if not fp_file.exists():
        raise FileNotFoundError(f"Không tìm thấy fingerprint.json của cache: {fp_file}")
    fp = json.loads(fp_file.read_text(encoding="utf-8"))

    checks = {
        "validation_coco_sha256": (fp.get("validation_coco_sha256"), sha256_file(valid_coco_path)),
        "det_ckpt_sha256": (fp.get("checkpoints", {}).get("det_ckpt"), sha256_file(det_ckpt_path)),
        "cls_ckpt_sha256": (fp.get("checkpoints", {}).get("cls_ckpt"), sha256_file(cls_ckpt_path)),
        "class_mapping_sha256": (fp.get("class_mapping_sha256"), sha256_file(class_mapping_path)),
    }
    mismatches = [
        f"{k}: expected={exp} vs actual={act}"
        for k, (exp, act) in checks.items()
        if exp != act
    ]
    if mismatches:
        raise ValueError(f"Fingerprint SQLite cache không khớp với đầu vào hiện tại: {'; '.join(mismatches)}")
    return {
        "verified": True,
        "fingerprint_path": str(fp_file),
        "checks": {k: act for k, (_, act) in checks.items()},
    }


@torch.inference_mode()
def ensure_detector_and_variant_crops_for_image(
    cache: CropAblationCacheDB,
    model: Optional[WasteCascade],
    image_meta: Dict[str, Any],
    image_path: Path,
    gt_boxes_xyxy: List[Tuple[int, int, int, int]],
    variants: List[str],
    nms_iou: float = 0.70,
    proposal_threshold: float = 0.0,
    max_proposals: int = 100,
    cls_batch_size: int = 32,
    include_pure_gt_boxes: bool = False,
    include_no_nms_proposals: bool = False,
) -> Dict[str, int]:
    """Ensure detector proposals and all requested variant crop probabilities are cached for `image_id`."""
    image_id = int(image_meta["id"])
    stats = {"detector_runs": 0, "new_crop_evals": 0, "reused_crop_evals": 0}

    cached_img = cache.get_image_raw(image_id)
    pil_image: Optional[Image.Image] = None
    if cached_img is not None:
        if image_path.exists():
            digest = sha256_file(image_path)
            if cached_img[0] != digest:
                raise ValueError(f"SHA-256 của ảnh Validation thay đổi so với cache: {image_path}")
        raw = cached_img[1]
    else:
        if model is None:
            raise RuntimeError(f"Thiếu detector cache cho image_id={image_id} và model chưa được nạp.")
        digest = sha256_file(image_path)
        with Image.open(image_path) as pil:
            pil_image = pil.convert("RGB")
        tensor = model.det_tf(pil_image).unsqueeze(0).to(model.device)
        output = model.detector(tensor)
        scores, det_idx = output["pred_logits"].sigmoid()[0].max(-1)
        raw = {
            "scores": scores.cpu().tolist(),
            "det_idx": det_idx.cpu().tolist(),
            "norm_boxes": output["pred_boxes"][0].float().cpu().tolist(),
            "size": list(pil_image.size),
        }
        cache.save_image_raw(image_id, digest, raw)
        stats["detector_runs"] += 1

    # Collect proposal coordinates from NMS selection (and optionally non-NMS or pure GT boxes)
    required_coords_set = set(gt_boxes_xyxy) if include_pure_gt_boxes else set()
    iou_modes = (None, nms_iou) if include_no_nms_proposals else (nms_iou,)
    for iou_val in iou_modes:
        s_key = selection_key(proposal_threshold, max_proposals, iou_val)
        sel = cache.get_selection(image_id, s_key)
        if sel is None:
            dev = model.device if model is not None else "cpu"
            sel = select_raw(raw, dev, proposal_threshold, max_proposals, iou_val)
            cache.save_selection(image_id, s_key, sel)
        for item in sel:
            required_coords_set.add(tuple(item["coords"]))

    all_coords = sorted(required_coords_set)

    for variant in variants:
        existing = cache.get_variant_crops(variant, image_id)
        missing = [c for c in all_coords if c not in existing]
        stats["reused_crop_evals"] += len(all_coords) - len(missing)
        if not missing:
            continue

        if model is None:
            raise RuntimeError(f"Variant {variant} thiếu {len(missing)} crops tại image_id={image_id} nhưng model=None.")
        if pil_image is None:
            with Image.open(image_path) as pil:
                pil_image = pil.convert("RGB")

        new_preds: Dict[Tuple[int, int, int, int], List[float]] = {}
        for k in range(0, len(missing), cls_batch_size):
            batch_coords = missing[k : k + cls_batch_size]
            batch_tensors = torch.stack(
                [preprocess_crop_for_variant(pil_image, c, variant) for c in batch_coords]
            ).to(model.device)
            probs = model.classifier(batch_tensors).softmax(dim=-1).cpu().tolist()
            new_preds.update(zip(batch_coords, probs))

        cache.save_variant_crops(variant, image_id, new_preds)
        stats["new_crop_evals"] += len(new_preds)

    return stats


def evaluate_variant_on_validation(
    cache: CropAblationCacheDB,
    variant: str,
    metadata: Dict[str, Any],
    cls_names: List[str],
    forward_map: Dict[str, Optional[int]],
    index_map: Dict[int, int],
    categories: Dict[int, str],
    nms_iou: float = 0.70,
    proposal_threshold: float = 0.0,
    max_proposals: int = 100,
    locked_cascade_pr_threshold: float = 0.25,
) -> Dict[str, Any]:
    """Compute COCOeval mAP@50, mAP@50:95, GT-aligned classification accuracy, and confusion diagnostics."""
    from dg38_det_trainer import evaluate_coco_predictions_detailed

    image_ids = [int(im["id"]) for im in metadata["images"]]
    img_sizes = {int(im["id"]): (int(im["width"]), int(im["height"])) for im in metadata["images"]}

    anns_by_img: Dict[int, List[Dict[str, Any]]] = {img_id: [] for img_id in image_ids}
    for ann in metadata["annotations"]:
        anns_by_img[int(ann["image_id"])].append(ann)

    exported_nms: List[Dict[str, Any]] = []
    outside_nms: List[Dict[str, Any]] = []

    gt_direct_total = 0
    gt_direct_correct = 0
    gt_direct_inscope_wrong = 0
    gt_direct_oos = 0
    gt_direct_cm: Dict[str, Counter] = {c: Counter() for c in categories.values()}

    matched_prop_total = 0
    matched_prop_correct = 0
    matched_prop_inscope_wrong = 0
    matched_prop_oos = 0
    matched_prop_cm: Dict[str, Counter] = {c: Counter() for c in categories.values()}

    s_key_nms = selection_key(proposal_threshold, max_proposals, nms_iou)

    for img_id in image_ids:
        raw = cache.get_image_raw(img_id)[1]
        sel_nms = cache.get_selection(img_id, s_key_nms)
        crop_probs = cache.get_variant_crops(variant, img_id)
        W, H = img_sizes[img_id]

        # 1. Build Cascade + NMS predictions
        for item in sel_nms:
            q, coords = int(item["query"]), item["coords"]
            det_score = float(raw["scores"][q])
            probs = crop_probs[tuple(coords)]
            cls_idx = int(np.argmax(probs))
            conf = float(probs[cls_idx])
            cname = cls_names[cls_idx]
            internal_cid = forward_map.get(canonical_name(cname))
            mapped_cid = None if internal_cid is None else index_map[internal_cid]
            row = {
                "bbox_xyxy": coords,
                "class_name": cname,
                "category_id": mapped_cid,
                "score": cascade_score(det_score, conf),
                "det_score": det_score,
                "cls_score": conf,
            }
            coco_row = as_coco_detection(row, img_id)
            if coco_row is None:
                outside_nms.append(dict(row, image_id=img_id))
            else:
                exported_nms.append(coco_row)

        # 2. Prepare GT boxes (exact float for 1-to-1 IoU matching, integer rounded if pure GT crops exist)
        gt_anns = anns_by_img[img_id]
        gt_float_xyxy_list: List[List[float]] = []
        gt_cids: List[int] = []
        for ann in gt_anns:
            x, y, w, h = [float(v) for v in ann["bbox"]]
            gt_float_xyxy_list.append([x, y, x + w, y + h])
            gt_cid = int(ann["category_id"])
            gt_cids.append(gt_cid)
            gt_cname = categories[gt_cid]

            gx1 = max(0, min(W - 1, int(round(x))))
            gy1 = max(0, min(H - 1, int(round(y))))
            gx2 = max(gx1 + 1, min(W, int(round(x + w))))
            gy2 = max(gy1 + 1, min(H, int(round(y + h))))
            gcoords = (gx1, gy1, gx2, gy2)
            if gcoords in crop_probs:
                probs = crop_probs[gcoords]
                pred_cname = cls_names[int(np.argmax(probs))]
                gt_direct_cm[gt_cname][pred_cname] += 1
                gt_direct_total += 1
                mapped_internal = forward_map.get(canonical_name(pred_cname))
                mapped_cid = None if mapped_internal is None else index_map[mapped_internal]
                if mapped_cid is None:
                    gt_direct_oos += 1
                elif mapped_cid == gt_cid:
                    gt_direct_correct += 1
                else:
                    gt_direct_inscope_wrong += 1

        # 3. 1-to-1 greedy matching between proposals (sorted by det_score desc) and GT boxes at IoU >= 0.50
        if sel_nms and gt_float_xyxy_list:
            sel_sorted = sorted(sel_nms, key=lambda r: float(raw["scores"][int(r["query"])]), reverse=True)
            prop_boxes_sorted = [r["coords"] for r in sel_sorted]
            prop_cls_sorted = [
                cls_names[int(np.argmax(crop_probs[tuple(r["coords"])]))]
                for r in sel_sorted
            ]
            p_arr = np.asarray(prop_boxes_sorted, dtype=np.float32)
            g_arr = np.asarray(gt_float_xyxy_list, dtype=np.float32)
            ious = box_iou_matrix_xyxy(p_arr, g_arr)
            gt_matched = [False] * len(gt_float_xyxy_list)
            for p_idx in range(len(prop_boxes_sorted)):
                best_g = -1
                best_iou = 0.50 - 1e-9
                for g_idx in range(len(gt_float_xyxy_list)):
                    if not gt_matched[g_idx] and ious[p_idx, g_idx] >= 0.50 and ious[p_idx, g_idx] > best_iou:
                        best_iou = float(ious[p_idx, g_idx])
                        best_g = g_idx
                if best_g >= 0:
                    gt_matched[best_g] = True
                    gt_cid = gt_cids[best_g]
                    gt_cname = categories[gt_cid]
                    pred_cname = prop_cls_sorted[p_idx]
                    matched_prop_cm[gt_cname][pred_cname] += 1
                    matched_prop_total += 1

                    mapped_internal = forward_map.get(canonical_name(pred_cname))
                    mapped_cid = None if mapped_internal is None else index_map[mapped_internal]
                    if mapped_cid is None:
                        matched_prop_oos += 1
                    elif mapped_cid == gt_cid:
                        matched_prop_correct += 1
                    else:
                        matched_prop_inscope_wrong += 1

    # Official COCOeval on active metadata subset
    tmp_coco = cache.db_path.parent / f"_tmp_eval_{variant}.json"
    tmp_coco.write_text(json.dumps(metadata), encoding="utf-8")
    try:
        coco_eval_res = evaluate_coco_predictions_detailed(
            tmp_coco, exported_nms, locked_pr_threshold=locked_cascade_pr_threshold
        )
    finally:
        tmp_coco.unlink(missing_ok=True)

    def extract_transition_highlights(cm_dict: Dict[str, Counter], total_count: int) -> Dict[str, Any]:
        glass_tot = sum(cm_dict["glass"].values())
        paper_tot = sum(cm_dict["paper"].values())
        plastic_tot = sum(cm_dict["plastic"].values())
        return {
            "total_evaluated_boxes": total_count,
            "glass_total": glass_tot,
            "glass_to_glass_correct": cm_dict["glass"]["glass"],
            "glass_to_glass_pct": round(100.0 * cm_dict["glass"]["glass"] / max(glass_tot, 1), 2),
            "glass_to_plastic_count": cm_dict["glass"]["plastic"],
            "glass_to_plastic_pct": round(100.0 * cm_dict["glass"]["plastic"] / max(glass_tot, 1), 2),
            "paper_total": paper_tot,
            "paper_to_paper_correct": cm_dict["paper"]["paper"],
            "paper_to_paper_pct": round(100.0 * cm_dict["paper"]["paper"] / max(paper_tot, 1), 2),
            "paper_to_cardboard_oos_count": cm_dict["paper"]["cardboard"],
            "paper_to_cardboard_oos_pct": round(100.0 * cm_dict["paper"]["cardboard"] / max(paper_tot, 1), 2),
            "paper_to_trash_oos_count": cm_dict["paper"]["trash"],
            "paper_to_trash_oos_pct": round(100.0 * cm_dict["paper"]["trash"] / max(paper_tot, 1), 2),
            "plastic_total": plastic_tot,
            "plastic_to_plastic_correct": cm_dict["plastic"]["plastic"],
            "plastic_to_plastic_pct": round(100.0 * cm_dict["plastic"]["plastic"] / max(plastic_tot, 1), 2),
        }

    return {
        "variant": variant,
        "variant_meta": VARIANT_DEFINITIONS[variant],
        "cascade_nms_coco_metrics": {
            "mAP50": coco_eval_res["mAP50"],
            "mAP50_95": coco_eval_res["mAP50_95"],
            "AR100": coco_eval_res["AR100"],
            "per_class_ap": coco_eval_res["per_class_ap"],
            "locked_operating_point_thr025": coco_eval_res["operating_point"],
            "best_val_operating_point": coco_eval_res["validation_best_pr_point"],
            "exported_inscope_predictions": len(exported_nms),
            "outside_scope_dropped": len(outside_nms),
            "outside_scope_by_class": dict(Counter(p["class_name"] for p in outside_nms)),
        },
        "matched_proposals_iou50_eval": {
            "total_matched_gt": matched_prop_total,
            "correct_inscope_count": matched_prop_correct,
            "correct_inscope_pct": round(100.0 * matched_prop_correct / max(matched_prop_total, 1), 2),
            "wrong_inscope_count": matched_prop_inscope_wrong,
            "wrong_inscope_pct": round(100.0 * matched_prop_inscope_wrong / max(matched_prop_total, 1), 2),
            "oos_dropped_count": matched_prop_oos,
            "oos_dropped_pct": round(100.0 * matched_prop_oos / max(matched_prop_total, 1), 2),
            "transitions": extract_transition_highlights(matched_prop_cm, matched_prop_total),
            "confusion_matrix_5x10": {k: dict(v) for k, v in matched_prop_cm.items()},
        },
        "pure_gt_boxes_eval": {
            "total_gt_boxes": gt_direct_total,
            "correct_inscope_count": gt_direct_correct,
            "correct_inscope_pct": round(100.0 * gt_direct_correct / max(gt_direct_total, 1), 2),
            "wrong_inscope_count": gt_direct_inscope_wrong,
            "wrong_inscope_pct": round(100.0 * gt_direct_inscope_wrong / max(gt_direct_total, 1), 2),
            "oos_dropped_count": gt_direct_oos,
            "oos_dropped_pct": round(100.0 * gt_direct_oos / max(gt_direct_total, 1), 2),
            "transitions": extract_transition_highlights(gt_direct_cm, gt_direct_total),
            "confusion_matrix_5x10": {k: dict(v) for k, v in gt_direct_cm.items()},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Giai đoạn 7 — Ablation Tiền xử lý Vùng Cắt (Crop Preprocessing) trên Validation sạch.")
    parser.add_argument("--mode", choices=["smoke", "pilot", "run"], default="run")
    parser.add_argument(
        "--variants",
        default="A_original_crop,B_direct_resize,C_context_pad_15,D_letterbox_224,E_context_pad_25_letterbox",
        help="Danh sách các biến thể crop cần chạy, phân cách bằng dấu phẩy",
    )
    parser.add_argument("--det-ckpt", default=str(PROJECT_ROOT / "detection/checkpoints_wastedetect/best_model.pt"))
    parser.add_argument("--cls-ckpt", default=str(PROJECT_ROOT / "checkpoints/dinov2_garbage_final.pth"))
    parser.add_argument(
        "--valid-coco",
        default=str(PROJECT_ROOT / "WasteDetect_Dataset/valid_clean_candidate/annotations_coco.json"),
    )
    parser.add_argument(
        "--valid-images",
        default=str(PROJECT_ROOT / "WasteDetect_Dataset/valid_clean_candidate/images"),
    )
    parser.add_argument("--class-mapping", default=str(PROJECT_ROOT / "detection/class_mapping.eval.json"))
    parser.add_argument(
        "--base-sqlite",
        default=str(PROJECT_ROOT / "evaluation_runs/val_grid_search_01/cache/predictions.sqlite3"),
        help="SQLite cache hiện có của Validation để tái sử dụng 100%% detector proposals và Variant A",
    )
    parser.add_argument(
        "--fingerprint",
        default=str(PROJECT_ROOT / "evaluation_runs/val_grid_search_01/cache/fingerprint.json"),
        help="Đường dẫn fingerprint.json của cache Validation",
    )
    parser.add_argument(
        "--work-sqlite",
        default=None,
        help="Đường dẫn SQLite làm việc (mặc định nằm trong --output-dir/crop_ablation_cache.sqlite3)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "crop_ablation_results"),
    )
    parser.add_argument("--cls-batch-size", type=int, default=32)
    parser.add_argument("--smoke-images", type=int, default=5)
    parser.add_argument("--pilot-images", type=int, default=25)
    parser.add_argument("--eval-pure-gt", action="store_true", help="Chạy thêm DINOv2 trực tiếp trên các GT boxes nguyên bản")
    parser.add_argument("--cu-per-hour", type=float, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if any(p.lower().startswith("test") for p in Path(args.valid_coco).parts + Path(args.valid_images).parts):
        raise PermissionError("Giai đoạn 7 chỉ được phép chạy trên tập Validation sạch; nghiêm cấm dùng tập Test!")

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        if v not in VARIANT_DEFINITIONS:
            raise ValueError(f"Biến thể không hợp lệ: {v}. Chọn trong {list(VARIANT_DEFINITIONS)}")

    # Verify fingerprint of base SQLite cache against all 4 authoritative inputs
    fp_path = Path(args.fingerprint)
    if not fp_path.exists() and args.base_sqlite:
        candidate_fp = Path(args.base_sqlite).parent / "fingerprint.json"
        if candidate_fp.exists():
            fp_path = candidate_fp
    if fp_path.exists():
        fp_check = verify_base_cache_fingerprint(
            fingerprint_path=fp_path,
            valid_coco_path=args.valid_coco,
            det_ckpt_path=args.det_ckpt,
            cls_ckpt_path=args.cls_ckpt,
            class_mapping_path=args.class_mapping,
        )
        print(f"[CACHE FINGERPRINT VERIFIED] {json.dumps(fp_check['checks'])}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = Path(args.work_sqlite) if args.work_sqlite else (out_dir / "crop_ablation_cache.sqlite3")
    if db_path.resolve() == Path(args.base_sqlite).resolve():
        raise PermissionError("Không được ghi trực tiếp vào base SQLite cache gốc; phải dùng bản sao làm việc!")

    metadata = read_json(args.valid_coco)
    categories = detector_categories(metadata)
    validate_coco(metadata, categories)
    index_map = detector_index_map(categories)

    cls_meta = torch.load(args.cls_ckpt, map_location="cpu", weights_only=True)
    class_names = list(cls_meta["classes"])
    del cls_meta
    resolved_policy = compile_label_policy(read_json(args.class_mapping), class_names, categories)
    reverse_idx = {v: k for k, v in index_map.items()}
    internal_policy = {k: None if v is None else reverse_idx[v] for k, v in resolved_policy.items()}

    need_model = bool(args.eval_pure_gt or any(v != "A_original_crop" for v in variants))
    model: Optional[WasteCascade] = None
    if need_model:
        model = WasteCascade(
            args.det_ckpt,
            args.cls_ckpt,
            {i: categories[cid] for i, cid in index_map.items()},
            args.device,
            classifier_to_detector=internal_policy,
        )

    cache = CropAblationCacheDB(db_path, base_val_sqlite=args.base_sqlite)
    try:
        all_images = metadata["images"]
        if args.mode == "smoke":
            active_images = all_images[: args.smoke_images]
        elif args.mode == "pilot":
            active_images = all_images[: args.pilot_images]
        else:
            active_images = all_images

        active_ids = {int(im["id"]) for im in active_images}
        sub_meta = {
            **metadata,
            "images": active_images,
            "annotations": [ann for ann in metadata["annotations"] if int(ann["image_id"]) in active_ids],
        }

        anns_by_img: Dict[int, List[Tuple[int, int, int, int]]] = {int(im["id"]): [] for im in active_images}
        img_sizes = {int(im["id"]): (int(im["width"]), int(im["height"])) for im in active_images}
        for ann in sub_meta["annotations"]:
            iid = int(ann["image_id"])
            W, H = img_sizes[iid]
            x, y, w, h = ann["bbox"]
            gx1 = max(0, min(W - 1, int(round(x))))
            gy1 = max(0, min(H - 1, int(round(y))))
            gx2 = max(gx1 + 1, min(W, int(round(x + w))))
            gy2 = max(gy1 + 1, min(H, int(round(y + h))))
            anns_by_img[iid].append((gx1, gy1, gx2, gy2))

        t0 = time.perf_counter()
        agg_stats = Counter()
        for idx, im in enumerate(active_images, 1):
            iid = int(im["id"])
            st = ensure_detector_and_variant_crops_for_image(
                cache=cache,
                model=model,
                image_meta=im,
                image_path=Path(args.valid_images) / im["file_name"],
                gt_boxes_xyxy=anns_by_img[iid],
                variants=variants,
                cls_batch_size=args.cls_batch_size,
                include_pure_gt_boxes=args.eval_pure_gt,
            )
            agg_stats.update(st)
            if idx % 100 == 0 or idx == len(active_images):
                print(
                    f"[{args.mode.upper()}] Processed {idx}/{len(active_images)} images "
                    f"({time.perf_counter() - t0:.1f}s) | {dict(agg_stats)}",
                    flush=True,
                )

        elapsed_sec = time.perf_counter() - t0

        if args.mode == "pilot":
            if not args.cu_per_hour or args.cu_per_hour <= 0:
                raise ValueError("Chế độ pilot yêu cầu --cu-per-hour <Usage Rate từ Colab UI>.")
            rt = inspect_runtime(model.device if model is not None else args.device)
            pilot = PilotTimingMeasurement(
                gpu_name=rt["gpu_name"],
                observed_cu_per_hour=args.cu_per_hour,
                train_batches_measured=len(active_images),
                train_time_sec=max(elapsed_sec, 0.1),
                total_train_batches_per_epoch=len(all_images),
                checkpoint_save_sec=2.0,
                startup_overhead_sec=15.0,
            )
            fc = pilot.forecast_run(epochs=1, num_seeds=1, extra_eval_sec_per_seed=20.0)
            tracker = CUBudgetTracker(Path(__file__).resolve().parent / "cu_budget_ledger.json")
            gate = tracker.check_Experiment_gate(
                run_id="crop_ablation_5variants_val",
                category="crop_preprocessing_ablation",
                forecast=fc,
            )
            print(tracker.format_budget_table_markdown())
            print(json.dumps({"forecast": fc, "gate": gate}, indent=2, ensure_ascii=False))
            return

        # Evaluate each variant
        variant_reports: Dict[str, Any] = {}
        summary_rows: List[Dict[str, Any]] = []
        for v in variants:
            v_rep = evaluate_variant_on_validation(
                cache=cache,
                variant=v,
                metadata=sub_meta,
                cls_names=class_names,
                forward_map=internal_policy,
                index_map=index_map,
                categories=categories,
            )
            variant_reports[v] = v_rep
            c_metrics = v_rep["cascade_nms_coco_metrics"]
            m_eval = v_rep["matched_proposals_iou50_eval"]
            g_eval = v_rep["pure_gt_boxes_eval"]
            summary_rows.append({
                "variant": v,
                "title": v_rep["variant_meta"]["title"],
                "cascade_nms_mAP50": c_metrics["mAP50"],
                "cascade_nms_mAP50_95": c_metrics["mAP50_95"],
                "cascade_nms_AR100": c_metrics["AR100"],
                "cascade_nms_F1_thr025": c_metrics["locked_operating_point_thr025"]["f1_iou50"],
                "glass_AP50": c_metrics["per_class_ap"]["glass"]["AP50"],
                "plastic_AP50": c_metrics["per_class_ap"]["plastic"]["AP50"],
                "paper_AP50": c_metrics["per_class_ap"]["paper"]["AP50"],
                "metal_AP50": c_metrics["per_class_ap"]["metal"]["AP50"],
                "organic_AP50": c_metrics["per_class_ap"]["organic"]["AP50"],
                "matched_gt_acc_pct": m_eval["correct_inscope_pct"],
                "matched_gt_wrong_inscope_pct": m_eval["wrong_inscope_pct"],
                "matched_gt_oos_pct": m_eval["oos_dropped_pct"],
                "matched_glass_to_plastic_pct": m_eval["transitions"]["glass_to_plastic_pct"],
                "matched_paper_to_cardboard_pct": m_eval["transitions"]["paper_to_cardboard_oos_pct"],
                "pure_gt_acc_pct": g_eval["correct_inscope_pct"],
                "pure_gt_glass_to_plastic_pct": g_eval["transitions"]["glass_to_plastic_pct"],
                "pure_gt_paper_to_cardboard_pct": g_eval["transitions"]["paper_to_cardboard_oos_pct"],
            })

        # Select best variant strictly on Validation mAP50:95 (tie-breaker: mAP50)
        best_variant = max(
            summary_rows,
            key=lambda r: (float(r["cascade_nms_mAP50_95"] or 0.0), float(r["cascade_nms_mAP50"] or 0.0)),
        )

        full_report = {
            "schema_version": 1,
            "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "mode": args.mode,
            "split": "valid_clean_candidate",
            "images_evaluated": len(active_images),
            "gt_boxes_evaluated": len(sub_meta["annotations"]),
            "elapsed_seconds": round(elapsed_sec, 2),
            "cache_work_stats": dict(agg_stats),
            "best_validation_variant": best_variant,
            "summary_table": summary_rows,
            "variants": variant_reports,
            "scientific_note_vi": (
                "Thí nghiệm kiểm tra giả thuyết Crop Domain Shift trên tập Validation sạch (1.743 ảnh). "
                "Chỉ thay đổi phép tiền xử lý vùng cắt (crop preprocessing), giữ cố định hoàn toàn Detector proposals, "
                "Classifier checkpoint, ánh xạ lớp và NMS (IoU=0.70). Không sử dụng tập Test để chọn biến thể."
            ),
        }

        out_json = out_dir / ("smoke_crop_ablation.json" if args.mode == "smoke" else "crop_ablation_report.json")
        out_json.write_text(json.dumps(full_report, indent=2, ensure_ascii=False), encoding="utf-8")

        if summary_rows:
            csv_path = out_dir / ("smoke_crop_ablation.csv" if args.mode == "smoke" else "crop_ablation_summary.csv")
            with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
                writer.writeheader()
                writer.writerows(summary_rows)

        if args.mode == "run" and args.cu_per_hour and args.cu_per_hour > 0:
            rt = inspect_runtime(model.device if model is not None else args.device)
            tracker = CUBudgetTracker(Path(__file__).resolve().parent / "cu_budget_ledger.json")
            tracker.record_actual_usage(
                run_id="crop_ablation_validation_full",
                category="crop_preprocessing_ablation",
                gpu_name=rt["gpu_name"],
                observed_cu_per_hour=args.cu_per_hour,
                elapsed_seconds=elapsed_sec,
                status="COMPLETED",
                epochs_completed=1,
                best_val_metric=best_variant["cascade_nms_mAP50"],
                checkpoint_path=str(out_json),
            )

        print(json.dumps({"best_validation_variant": best_variant, "summary_table": summary_rows}, indent=2, ensure_ascii=False))
    finally:
        cache.close()


if __name__ == "__main__":
    main()
