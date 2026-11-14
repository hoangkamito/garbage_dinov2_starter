"""Validation evaluation and threshold selection on WasteDetect_Dataset/valid_clean_candidate.

Uses detection/evaluate_cascade.py and detection/waste_cascade.py without modifying any
checkpoint, inference module, or guarded test set.
"""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import copy
import io
import json
from pathlib import Path
import platform
import sys
import time
from importlib.metadata import version

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "detection"))

from evaluate_cascade import (
    EXPERIMENTS,
    as_coco_detection,
    coco_metrics,
    compile_label_policy,
    detector_categories,
    detector_index_map,
    image_path,
    read_json,
    run_experiments,
    sha256_file,
    validate_coco,
    write_results,
)
from waste_cascade import WasteCascade, canonical_name, select_proposals


def compute_f1(p: float | None, r: float | None) -> float:
    if not p or not r or (p + r) <= 0:
        return 0.0
    return 2.0 * p * r / (p + r)


@torch.inference_mode()
def run_cached_grid_search(
    model: WasteCascade,
    coco_gt: COCO,
    image_root: Path,
    index_map: dict[int, int],
    *,
    proposal_threshold: float,
    max_proposals: int,
    cls_batch_size: int,
    nms_grid: list[float],
    pr_grid: list[float],
) -> dict:
    images = coco_gt.dataset["images"]
    image_ids = [im["id"] for im in images]

    # Step 1: Run detector once per image and cache query scores/boxes + classifier crop probs
    cached_items = []
    t0 = time.perf_counter()
    for idx, im in enumerate(images):
        with Image.open(image_path(image_root, im["file_name"])) as pil:
            image = pil.convert("RGB")
        W, H = image.size
        image_tensor = model.det_tf(image).unsqueeze(0).to(model.device)
        output = model.detector(image_tensor)
        scores, det_idx = output["pred_logits"].sigmoid()[0].max(-1)
        norm_boxes = output["pred_boxes"][0]

        # Collect union of query indices needed across no-NMS and all nms_iou in nms_grid
        union_indices_set = set()
        ind_no_nms, boxes_no_nms = select_proposals(
            norm_boxes, scores, (W, H), proposal_threshold, max_proposals, nms_iou=None
        )
        for q, box in zip(ind_no_nms.tolist(), boxes_no_nms.tolist()):
            coords = [round(v) for v in box]
            if coords[2] > coords[0] and coords[3] > coords[1]:
                union_indices_set.add((q, tuple(coords)))

        per_nms_selected = {None: []}
        for q, box in zip(ind_no_nms.tolist(), boxes_no_nms.tolist()):
            coords = [round(v) for v in box]
            if coords[2] > coords[0] and coords[3] > coords[1]:
                per_nms_selected[None].append((q, coords))

        for iou_val in nms_grid:
            ind_nms, boxes_nms = select_proposals(
                norm_boxes, scores, (W, H), proposal_threshold, max_proposals, nms_iou=iou_val
            )
            sel_list = []
            for q, box in zip(ind_nms.tolist(), boxes_nms.tolist()):
                coords = [round(v) for v in box]
                if coords[2] > coords[0] and coords[3] > coords[1]:
                    sel_list.append((q, coords))
                    union_indices_set.add((q, tuple(coords)))
            per_nms_selected[iou_val] = sel_list

        # Classify unique (q, coords) crops once per image
        union_list = sorted(union_indices_set)
        crop_map = {}
        if union_list:
            crops = [model.cls_tf(image.crop(coords)) for _, coords in union_list]
            batch_probs = []
            for k in range(0, len(crops), cls_batch_size):
                x = torch.stack(crops[k : k + cls_batch_size]).to(model.device)
                batch_probs.append(model.classifier(x).softmax(-1).cpu())
            probs = torch.cat(batch_probs, dim=0)
            for (q, coords), prob_vec in zip(union_list, probs):
                conf, cls_idx = prob_vec.max(-1)
                cname = model.cls_names[int(cls_idx)]
                mapped_id = model.det_name_to_id.get(canonical_name(cname))
                crop_map[(q, coords)] = (cname, mapped_id, float(conf))

        cached_items.append({
            "image_id": im["id"],
            "scores": scores.cpu().tolist(),
            "det_idx": det_idx.cpu().tolist(),
            "per_nms_selected": per_nms_selected,
            "crop_map": crop_map,
        })
        if (idx + 1) % 50 == 0 or idx + 1 == len(images):
            print(f"Cached detector + classifier outputs for {idx + 1}/{len(images)} validation images ({time.perf_counter() - t0:.1f}s)")

    # Step 2: Evaluate COCO metrics across NMS grid and PR threshold grid
    def build_preds(use_classify: bool, iou_key: float | None):
        exported = []
        out_of_scope = []
        cls_counts = Counter()
        for item in cached_items:
            im_id = item["image_id"]
            scores = item["scores"]
            det_idx = item["det_idx"]
            crop_map = item["crop_map"]
            for q, coords in item["per_nms_selected"][iou_key]:
                det_s = float(scores[q])
                if not use_classify:
                    cid = index_map[int(det_idx[q])]
                    pred = {
                        "image_id": im_id,
                        "bbox_xyxy": list(coords),
                        "category_id": cid,
                        "score": det_s,
                    }
                    exported.append(as_coco_detection(pred, im_id))
                else:
                    cname, mapped_internal_id, conf = crop_map[(q, tuple(coords))]
                    cls_counts[cname] += 1
                    cid = None if mapped_internal_id is None else index_map[mapped_internal_id]
                    pred = {
                        "image_id": im_id,
                        "bbox_xyxy": list(coords),
                        "class_name": cname,
                        "category_id": cid,
                        "score": float(det_s * conf),
                        "det_score": det_s,
                        "cls_score": conf,
                    }
                    conv = as_coco_detection(pred, im_id)
                    if conv is None:
                        out_of_scope.append(pred)
                    else:
                        exported.append(conv)
        return exported, out_of_scope, dict(cls_counts)

    grid_results = {"no_nms": {}, "nms_sweep": {}}

    # Evaluate no-NMS baselines (detector_original and cascade_original) across pr_grid
    for exp_name, use_cls in [("detector_original", False), ("cascade_original", True)]:
        exported, oos, cls_counts = build_preds(use_cls, None)
        base_m = coco_metrics(coco_gt, exported, image_ids, score_threshold=pr_grid[0])
        pr_sweep = []
        for thr in pr_grid:
            m = coco_metrics(coco_gt, exported, image_ids, score_threshold=thr)
            f1 = compute_f1(m["precision_iou50"], m["recall_iou50"])
            pr_sweep.append({
                "pr_score_threshold": thr,
                "precision_iou50": m["precision_iou50"],
                "recall_iou50": m["recall_iou50"],
                "f1_iou50": f1,
                "TP": m["TP"],
                "FP": m["FP"],
                "FN": m["FN"],
                "outside_scope_dropped_at_pr_threshold": sum(p["score"] >= thr for p in oos),
            })
        best_pr = max(pr_sweep, key=lambda x: (x["f1_iou50"], x["precision_iou50"]))
        grid_results["no_nms"][exp_name] = {
            "mAP50_95": base_m["mAP50_95"],
            "mAP50": base_m["mAP50"],
            "AR100": base_m["AR100"],
            "exported_prediction_count": len(exported),
            "outside_scope_dropped": len(oos),
            "outside_scope_by_class": dict(Counter(p["class_name"] for p in oos)),
            "classifier_predictions_by_class": cls_counts,
            "best_pr_operating_point": best_pr,
            "pr_threshold_sweep": pr_sweep,
        }

    # Evaluate NMS variants across nms_grid and pr_grid
    for iou_val in nms_grid:
        iou_key_str = f"{iou_val:.2f}"
        grid_results["nms_sweep"][iou_key_str] = {}
        for exp_name, use_cls in [("detector_nms", False), ("cascade_nms", True)]:
            exported, oos, cls_counts = build_preds(use_cls, iou_val)
            base_m = coco_metrics(coco_gt, exported, image_ids, score_threshold=pr_grid[0])
            pr_sweep = []
            for thr in pr_grid:
                m = coco_metrics(coco_gt, exported, image_ids, score_threshold=thr)
                f1 = compute_f1(m["precision_iou50"], m["recall_iou50"])
                pr_sweep.append({
                    "pr_score_threshold": thr,
                    "precision_iou50": m["precision_iou50"],
                    "recall_iou50": m["recall_iou50"],
                    "f1_iou50": f1,
                    "TP": m["TP"],
                    "FP": m["FP"],
                    "FN": m["FN"],
                    "outside_scope_dropped_at_pr_threshold": sum(p["score"] >= thr for p in oos),
                })
            best_pr = max(pr_sweep, key=lambda x: (x["f1_iou50"], x["precision_iou50"]))
            grid_results["nms_sweep"][iou_key_str][exp_name] = {
                "nms_iou": iou_val,
                "mAP50_95": base_m["mAP50_95"],
                "mAP50": base_m["mAP50"],
                "AR100": base_m["AR100"],
                "exported_prediction_count": len(exported),
                "outside_scope_dropped": len(oos),
                "outside_scope_by_class": dict(Counter(p["class_name"] for p in oos)),
                "classifier_predictions_by_class": cls_counts,
                "best_pr_operating_point": best_pr,
                "pr_threshold_sweep": pr_sweep,
            }

    # Recommend locked NMS IoU (maximizing mean mAP50_95 across detector_nms & cascade_nms, tie-break cascade_nms mAP50_95)
    best_iou_str = max(
        grid_results["nms_sweep"].keys(),
        key=lambda k: (
            (grid_results["nms_sweep"][k]["detector_nms"]["mAP50_95"] or 0.0)
            + (grid_results["nms_sweep"][k]["cascade_nms"]["mAP50_95"] or 0.0),
            grid_results["nms_sweep"][k]["cascade_nms"]["mAP50_95"] or 0.0,
        ),
    )
    rec_nms_iou = grid_results["nms_sweep"][best_iou_str]["cascade_nms"]["nms_iou"]
    rec_baseline_pr = grid_results["nms_sweep"][best_iou_str]["detector_nms"]["best_pr_operating_point"]["pr_score_threshold"]
    rec_cascade_pr = grid_results["nms_sweep"][best_iou_str]["cascade_nms"]["best_pr_operating_point"]["pr_score_threshold"]

    grid_results["recommended_locked_parameters"] = {
        "proposal_threshold": proposal_threshold,
        "nms_iou": rec_nms_iou,
        "baseline_pr_threshold": rec_baseline_pr,
        "cascade_pr_threshold": rec_cascade_pr,
        "selection_rule": "Locked on valid_clean_candidate: nms_iou maximizes combined mAP50_95 (detector_nms + cascade_nms); baseline_pr_threshold and cascade_pr_threshold maximize micro-F1 at IoU=0.50 for their respective score families.",
    }
    return grid_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["single_run", "grid_search"], default="single_run")
    parser.add_argument("--det_ckpt", default="detection/checkpoints_wastedetect/best_model.pt")
    parser.add_argument("--cls_ckpt", default="checkpoints/dinov2_garbage_final.pth")
    parser.add_argument("--valid_coco", default="WasteDetect_Dataset/valid_clean_candidate/annotations_coco.json")
    parser.add_argument("--valid_images", default="WasteDetect_Dataset/valid_clean_candidate/images")
    parser.add_argument("--class_mapping", default="detection/class_mapping.eval.json")
    parser.add_argument("--detector_class_map", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--proposal_threshold", type=float, default=0.0)
    parser.add_argument("--baseline_pr_threshold", type=float, default=0.05)
    parser.add_argument("--cascade_pr_threshold", type=float, default=0.05)
    parser.add_argument("--nms_iou", type=float, default=0.5)
    parser.add_argument("--nms_grid", default="0.3,0.4,0.5,0.6,0.7")
    parser.add_argument("--pr_threshold_grid", default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50")
    parser.add_argument("--max_proposals", type=int, default=100)
    parser.add_argument("--cls_batch_size", type=int, default=8)
    parser.add_argument("--warmup_images", type=int, default=1)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    out_path = Path(args.out_dir)
    if out_path.exists():
        parser.error(f"--out_dir already exists: {out_path}")

    metadata = read_json(args.valid_coco)
    categories = detector_categories(metadata)
    validate_coco(metadata, categories)
    index_map = detector_index_map(
        categories, read_json(args.detector_class_map) if args.detector_class_map else None
    )
    policy = read_json(args.class_mapping)
    cls_metadata = torch.load(args.cls_ckpt, map_location="cpu", weights_only=True)
    class_names = list(cls_metadata["classes"])
    del cls_metadata
    resolved = compile_label_policy(policy, class_names, categories)
    reverse_indices = {external: internal for internal, external in index_map.items()}
    internal_policy = {name: None if cid is None else reverse_indices[cid] for name, cid in resolved.items()}

    model = WasteCascade(
        args.det_ckpt,
        args.cls_ckpt,
        {i: categories[cid] for i, cid in index_map.items()},
        args.device,
        classifier_to_detector=internal_policy,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = metadata
        coco_gt.createIndex()

    if args.mode == "single_run":
        settings = {
            "nms_iou": args.nms_iou,
            "no_nms": False,
            "proposal_threshold": args.proposal_threshold,
            "baseline_pr_threshold": args.baseline_pr_threshold,
            "cascade_pr_threshold": args.cascade_pr_threshold,
            "max_proposals": args.max_proposals,
            "cls_batch_size": args.cls_batch_size,
            "warmup_images": args.warmup_images,
        }
        records, summaries = run_experiments(model, coco_gt, args.valid_images, index_map, **settings)
        report = {
            "split": "valid_clean_candidate",
            "valid_images": len(metadata["images"]),
            "valid_gt_boxes": len(metadata.get("annotations", [])),
            "categories": categories,
            "detector_index_to_coco_id": index_map,
            "settings": settings,
            "label_policy": {
                "config": policy,
                "resolved_classifier_to_coco_id": resolved,
                "sha256": sha256_file(args.class_mapping),
            },
            "inputs": {
                name: {"path": str(Path(getattr(args, name)).resolve()), "sha256": sha256_file(getattr(args, name))}
                for name in ("valid_coco", "det_ckpt", "cls_ckpt")
            },
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torchvision": version("torchvision"),
                "pycocotools": version("pycocotools"),
                "device": str(model.device),
            },
            "experiments": summaries,
        }
        write_results(out_path, report, records)
        print(json.dumps(report["experiments"], indent=2, ensure_ascii=False))
    else:
        nms_grid = [float(x.strip()) for x in args.nms_grid.split(",") if x.strip()]
        pr_grid = [float(x.strip()) for x in args.pr_threshold_grid.split(",") if x.strip()]
        grid_report = run_cached_grid_search(
            model,
            coco_gt,
            Path(args.valid_images),
            index_map,
            proposal_threshold=args.proposal_threshold,
            max_proposals=args.max_proposals,
            cls_batch_size=args.cls_batch_size,
            nms_grid=nms_grid,
            pr_grid=pr_grid,
        )
        out_path.mkdir(parents=True, exist_ok=False)
        report_file = out_path / "validation_grid_search_report.json"
        report_file.write_text(json.dumps(grid_report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(grid_report["recommended_locked_parameters"], indent=2, ensure_ascii=False))
        print(f"Saved full grid search report to {report_file}")


if __name__ == "__main__":
    main()
