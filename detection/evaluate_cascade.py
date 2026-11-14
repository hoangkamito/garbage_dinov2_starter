"""Four controlled detector/cascade ablations on a declared independent COCO test set.

AP/AR and fixed-threshold micro P/R use COCOeval, including crowd-ignore rules.
No classifier label is implicitly merged into a detector category.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import hashlib
import io
import json
import math
import platform
import statistics
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from waste_cascade import WasteCascade, canonical_name, synchronize

EXPERIMENTS = {
    "detector_original": (False, False),
    "detector_nms": (False, True),
    "cascade_original": (True, False),
    "cascade_nms": (True, True),
}
INDEPENDENT_OF = {"detector_train", "detector_validation", "classifier_train", "classifier_validation"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detector_categories(metadata):
    """Read external COCO IDs as-is; do not silently renumber categories."""
    rows = metadata.get("categories", [])
    if len(rows) != 5:
        raise ValueError("Test COCO must declare exactly five detector categories.")
    categories = {}
    names = set()
    for row in rows:
        cid, name = row["id"], row["name"]
        if type(cid) is not int or not isinstance(name, str) or not canonical_name(name):
            raise ValueError("Invalid COCO category ID/name.")
        if cid in categories or canonical_name(name) in names:
            raise ValueError("Duplicate COCO category ID/name.")
        categories[cid] = name
        names.add(canonical_name(name))
    return categories


def resolve_target(target, categories):
    if type(target) is int and target in categories:
        return target
    by_name = {canonical_name(name): cid for cid, name in categories.items()}
    if isinstance(target, str) and canonical_name(target) in by_name:
        return by_name[canonical_name(target)]
    raise ValueError(f"Unknown detector category: {target!r}")


def detector_index_map(categories, config=None):
    """The current trained detector uses COCO IDs 0..4 directly.

    Other test ID schemes require an explicit query-class-index mapping.
    """
    if config is None:
        if set(categories) != set(range(5)):
            raise ValueError("Non-0..4 COCO IDs require --detector_class_map; no automatic remapping.")
        return {i: i for i in range(5)}
    if set(config) != {str(i) for i in range(5)}:
        raise ValueError("detector_class_map must contain exactly index keys 0..4.")
    result = {int(i): resolve_target(target, categories) for i, target in config.items()}
    if len(set(result.values())) != 5:
        raise ValueError("Detector index mapping must be one-to-one onto all five COCO categories.")
    return result


def compile_label_policy(config, class_names, categories):
    """Require explicit decisions for all classifier classes; null means out of scope.

    Multiple classifier classes may map to one detector category, but each
    non-identity mapping must have its own non-empty evidence entry.
    """
    if config.get("outside_scope_action") != "drop_and_count":
        raise ValueError("outside_scope_action must be drop_and_count.")
    mapping = config.get("mapping", {})
    evidence = config.get("evidence", {})
    names = [canonical_name(name) for name in class_names]
    normalized = {canonical_name(k): v for k, v in mapping.items()}
    normalized_evidence = {canonical_name(k): v for k, v in evidence.items()}
    if len(normalized) != len(mapping) or len(set(names)) != len(names):
        raise ValueError("Duplicate canonical classifier names in mapping/checkpoint.")
    if set(normalized) != set(names):
        raise ValueError(f"Policy must explicitly cover checkpoint classes. Missing={sorted(set(names)-set(normalized))}; "
                         f"unknown={sorted(set(normalized)-set(names))}")
    result = {}
    for name, target in normalized.items():
        cid = None if target is None else resolve_target(target, categories)
        if cid is not None and name != canonical_name(categories[cid]):
            proof = normalized_evidence.get(name)
            if not isinstance(proof, str) or not proof.strip():
                raise ValueError(f"Non-identity mapping {name!r} needs an evidence entry.")
        result[name] = cid
    return result


def image_path(root, file_name):
    root = Path(root).resolve()
    path = (root / str(file_name).replace("\\", "/")).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Image path escapes dataset root: {file_name}")
    return path


def validate_coco(metadata, categories):
    images = metadata.get("images", [])
    ids = [im["id"] for im in images]
    if not images or len(ids) != len(set(ids)):
        raise ValueError("Test images must be non-empty with unique IDs.")
    ids = set(ids)
    annotation_ids = set()
    for ann in metadata.get("annotations", []):
        if (type(ann["id"]) is not int or ann["id"] <= 0 or ann["id"] in annotation_ids
                or ann["image_id"] not in ids or ann["category_id"] not in categories):
            raise ValueError("Invalid annotation ID/image_id/category_id.")
        annotation_ids.add(ann["id"])
        box = ann["bbox"]
        if (len(box) != 4 or not all(math.isfinite(v) for v in box)
                or box[2] <= 0 or box[3] <= 0):
            raise ValueError("Ground-truth boxes must be finite with positive width/height.")
        # COCO bbox-only exports sometimes omit area/iscrowd.
        ann.setdefault("area", box[2] * box[3])
        ann.setdefault("iscrowd", 0)
    metadata.setdefault("info", {})
    metadata.setdefault("licenses", [])


def validate_test_split(manifest_path, test_coco, test_images, metadata):
    """Require a bound provenance declaration; audit supplied exclusion references.

    This cannot prove subject/session independence. Provenance of BOTH models'
    training and validation data remains a dataset-curation responsibility.
    """
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    base = manifest_path.parent
    if (manifest.get("split") != "test" or manifest.get("independent") is not True
            or not isinstance(manifest.get("description"), str) or not manifest["description"].strip()
            or not INDEPENDENT_OF.issubset(set(manifest.get("independent_of", [])))):
        raise ValueError("An independent test manifest for BOTH models' train/validation sets is required.")
    test_coco, test_images = Path(test_coco).resolve(), Path(test_images).resolve()
    if ((base / manifest.get("test_coco", "")).resolve() != test_coco
            or (base / manifest.get("test_images", "")).resolve() != test_images):
        raise ValueError("Manifest is not bound to the requested test annotations/image root.")
    split = str(metadata.get("info", {}).get("split", metadata.get("split", ""))).lower()
    path_parts = {p.lower() for p in test_images.parts + test_coco.parts}
    if split in {"train", "training", "val", "valid", "validation"} or path_parts & {"train", "training"}:
        raise ValueError("Training/validation data cannot be used as the independent test.")
    test_names = {str(im["file_name"]).replace("\\", "/").casefold() for im in metadata["images"]}
    test_hashes = set()
    for im in metadata["images"]:
        file = image_path(test_images, im["file_name"])
        if not file.is_file():
            raise FileNotFoundError(f"Missing test image: {file}")
        test_hashes.add(sha256_file(file))
    audits = []
    for reference in manifest.get("references", []):
        ref_path = (base / reference["coco"]).resolve()
        if ref_path == test_coco or sha256_file(ref_path) == sha256_file(test_coco):
            raise ValueError("Test annotations equal an excluded train/validation reference.")
        ref = read_json(ref_path)
        ref_names = {str(im["file_name"]).replace("\\", "/").casefold() for im in ref.get("images", [])}
        if reference.get("images"):
            ref_root = (base / reference["images"]).resolve()
            for im in ref.get("images", []):
                if sha256_file(image_path(ref_root, im["file_name"])) in test_hashes:
                    raise ValueError("Image content overlaps an excluded train/validation reference.")
        elif test_names & ref_names:
            raise ValueError("Overlapping filenames in excluded reference; supply its image root for content audit.")
        audits.append({"coco": str(ref_path), "sha256": sha256_file(ref_path),
                       "content_checked": bool(reference.get("images"))})
    return {"manifest": manifest, "manifest_sha256": sha256_file(manifest_path),
            "reference_audits": audits, "test_unique_image_sha256_count": len(test_hashes),
            "independence_basis": "User-declared provenance for both models; supplied references audited. "
                                  "No claim of subject/session independence beyond the declaration."}


def as_coco_detection(prediction, image_id):
    if prediction["category_id"] is None:
        return None
    x1, y1, x2, y2 = prediction["bbox_xyxy"]
    return {"image_id": image_id, "category_id": int(prediction["category_id"]),
            "bbox": [x1, y1, x2-x1, y2-y1], "score": float(prediction["score"])}


def coco_metrics(coco_gt, predictions, image_ids, score_threshold=0.05):
    """AP/AR over all exported scores, micro P/R at IoU .50 and a fixed score.

    P/R uses COCOeval's un-interpolated matches at area=all, maxDets=100
    per image/category. It is NOT the mean of COCO's interpolated AP array.
    """
    if not image_ids:
        return {"mAP50_95": None, "mAP50": None, "AR100": None,
                "precision_iou50": 0., "recall_iou50": None,
                "TP": 0, "FP": 0, "FN": 0, "ignored_predictions": 0,
                "pr_score_threshold": score_threshold, "status": "no_images"}
    with contextlib.redirect_stdout(io.StringIO()):
        if predictions:
            coco_dt = coco_gt.loadRes(copy.deepcopy(predictions))
        else:
            # COCO.loadRes([]) indexes anns[0]; construct an empty result set instead.
            coco_dt = COCO()
            coco_dt.dataset = {"images": copy.deepcopy(coco_gt.dataset["images"]),
                               "categories": copy.deepcopy(coco_gt.dataset["categories"]),
                               "annotations": []}
            coco_dt.createIndex()
        evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
        evaluator.params.imgIds = list(image_ids)
        evaluator.params.catIds = sorted(coco_gt.getCatIds())
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    tp = fp = gt_count = ignored = 0
    iou_index = int(np.flatnonzero(np.isclose(evaluator.params.iouThrs, .5))[0])
    all_area = list(evaluator.params.areaRng[0])
    for row in evaluator.evalImgs:
        if row is None or list(row["aRng"]) != all_area or row["maxDet"] != 100:
            continue
        selected = np.asarray(row["dtScores"]) >= score_threshold
        dt_ignore = row["dtIgnore"][iou_index].astype(bool)
        matched = row["dtMatches"][iou_index] > 0
        tp += int(np.count_nonzero(selected & ~dt_ignore & matched))
        fp += int(np.count_nonzero(selected & ~dt_ignore & ~matched))
        ignored += int(np.count_nonzero(selected & dt_ignore))
        gt_count += int(np.count_nonzero(row["gtIgnore"] == 0))
    def defined_stat(index):
        value = float(evaluator.stats[index])
        return value if value >= 0 else None
    return {"mAP50_95": defined_stat(0), "mAP50": defined_stat(1), "AR100": defined_stat(8),
            "precision_iou50": tp / max(tp + fp, 1),
            "recall_iou50": tp / gt_count if gt_count else None,
            "TP": tp, "FP": fp, "FN": gt_count-tp, "ignored_predictions": ignored,
            "pr_score_threshold": score_threshold, "status": "ok" if gt_count else "no_valid_ground_truth"}


def latency_summary(samples):
    if not samples:
        return None
    wall = [row[0] for row in samples]
    return {"end_to_end_mean": statistics.mean(wall), "end_to_end_p50": float(np.percentile(wall, 50)),
            "end_to_end_p95": float(np.percentile(wall, 95)),
            "detector_model_mean": statistics.mean(row[1] for row in samples),
            "classifier_model_mean": statistics.mean(row[2] for row in samples)}


def run_experiments(model, coco_gt, image_root, index_map, *, nms_iou=.5, no_nms=False,
                    proposal_threshold=0., baseline_pr_threshold=.05, cascade_pr_threshold=.05,
                    max_proposals=100, cls_batch_size=8, warmup_images=1):
    images = coco_gt.dataset["images"]
    if not images:
        raise ValueError("Independent test set has no images.")
    records = {name: {"raw": [], "predictions": [], "out_of_scope": [], "timings": [],
                      "proposal_counts": Counter(), "classifier_counts": Counter()} for name in EXPERIMENTS}
    active = [name for name, (_, use_nms) in EXPERIMENTS.items() if not (no_nms and use_nms)]

    def predict(image, name):
        classify, use_nms = EXPERIMENTS[name]
        return model.predict(image, proposal_threshold=proposal_threshold, max_proposals=max_proposals,
                             cls_batch_size=cls_batch_size, nms_iou=nms_iou if use_nms else None,
                             classify=classify)

    for im in images[:warmup_images]:
        with Image.open(image_path(image_root, im["file_name"])) as pil:
            image = pil.convert("RGB")
        for name in active:
            predict(image, name)
    for image_index, im in enumerate(images):
        with Image.open(image_path(image_root, im["file_name"])) as pil:
            image = pil.convert("RGB")
        if image.size != (im["width"], im["height"]):
            raise ValueError(f"Image dimensions disagree with COCO: {im['file_name']}")
        # Rotate order to reduce systematic first/last-run latency bias.
        offset = image_index % len(active)
        for name in active[offset:] + active[:offset]:
            bucket = records[name]
            synchronize(model.device)
            start = time.perf_counter()
            result = predict(image, name)
            synchronize(model.device)
            elapsed = (time.perf_counter() - start) * 1000
            bucket["timings"].append((elapsed, result["detector_ms"], result["classifier_ms"]))
            bucket["proposal_counts"].update(result["proposal_stats"])
            classify, _ = EXPERIMENTS[name]
            predictions = result["cascade" if classify else "baseline"]
            for prediction in predictions:
                prediction = dict(prediction, image_id=im["id"])
                cid = prediction["category_id"]
                prediction["category_id"] = None if cid is None else index_map[cid]
                bucket["raw"].append(prediction)
                if classify:
                    bucket["classifier_counts"][prediction["class_name"]] += 1
                converted = as_coco_detection(prediction, im["id"])
                if converted is None:
                    bucket["out_of_scope"].append(prediction)
                else:
                    bucket["predictions"].append(converted)
        if (image_index + 1) % 20 == 0 or image_index + 1 == len(images):
            print(f"Evaluated {image_index + 1}/{len(images)} images")
    summaries = {}
    for name, (classify, use_nms) in EXPERIMENTS.items():
        bucket = records[name]
        threshold = cascade_pr_threshold if classify else baseline_pr_threshold
        if name not in active:
            summaries[name] = {"status": "skipped_nms_disabled", "metrics": None, "latency_ms": None,
                               "nms_iou": nms_iou, "pr_score_threshold": threshold}
            continue
        metrics = coco_metrics(coco_gt, bucket["predictions"], [im["id"] for im in images], threshold)
        out_of_scope = bucket["out_of_scope"]
        summaries[name] = {
            "status": "completed", "metrics": metrics,
            "nms_iou": nms_iou if use_nms else None, "pr_score_threshold": threshold,
            "proposal_counts": dict(bucket["proposal_counts"]),
            "raw_prediction_count": len(bucket["raw"]), "exported_prediction_count": len(bucket["predictions"]),
            "outside_scope_dropped": len(out_of_scope),
            "outside_scope_dropped_at_pr_threshold": sum(p["score"] >= threshold for p in out_of_scope),
            "outside_scope_by_class": dict(Counter(p["class_name"] for p in out_of_scope)),
            "classifier_predictions_by_class": dict(bucket["classifier_counts"]),
            "below_pr_threshold": sum(p["score"] < threshold for p in bucket["predictions"]),
            "latency_ms": latency_summary(bucket["timings"]),
        }
    return records, summaries


def write_results(output, report, records):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)  # Never overwrite a previous scientific run.
    def save(path, value):
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    save(output / "metrics.json", report)
    rows = []
    for name, summary in report["experiments"].items():
        folder = output / name
        folder.mkdir()
        bucket = records[name]
        save(folder / "predictions.json", bucket["predictions"])
        save(folder / "raw_predictions.json", bucket["raw"])
        save(folder / "out_of_scope_predictions.json", bucket["out_of_scope"])
        save(folder / "metrics.json", summary)
        metrics = summary.get("metrics") or {}
        counts = summary.get("proposal_counts", {})
        latency = summary.get("latency_ms") or {}
        rows.append({"experiment": name, "status": summary["status"],
                     **{k: metrics.get(k) for k in ("mAP50", "mAP50_95", "AR100", "precision_iou50", "recall_iou50", "TP", "FP", "FN")},
                     "proposal_threshold": report["settings"]["proposal_threshold"],
                     "pr_score_threshold": summary["pr_score_threshold"], "nms_iou": summary["nms_iou"],
                     "label_policy": report["label_policy"]["config"].get("name", "unnamed"),
                     "exported_predictions": summary.get("exported_prediction_count"),
                     "nms_removed": counts.get("nms_removed"),
                     "invalid_boxes_removed": counts.get("invalid_boxes", 0) + counts.get("rounded_invalid", 0),
                     "proposal_limit_removed": counts.get("pre_limit_removed", 0) + counts.get("post_limit_removed", 0),
                     "outside_scope_dropped": summary.get("outside_scope_dropped"),
                     **{f"latency_{k}_ms": v for k, v in latency.items()}})
    # All rows share columns, even when NMS experiments were skipped.
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--det_ckpt", required=True)
    parser.add_argument("--cls_ckpt", required=True)
    parser.add_argument("--test_coco", required=True)
    parser.add_argument("--test_images", required=True)
    parser.add_argument("--test_manifest", required=True, help="Bound independent-test provenance JSON")
    parser.add_argument("--class_mapping", required=True, help="Explicit classifier -> detector label policy JSON")
    parser.add_argument("--detector_class_map", help="Optional index 0..4 -> COCO name/ID JSON")
    parser.add_argument("--out_dir", default="cascade_eval")
    parser.add_argument("--proposal_threshold", type=float, default=0., help="Use low validation-fixed threshold for AP/AR")
    parser.add_argument("--baseline_pr_threshold", type=float, default=.05)
    parser.add_argument("--cascade_pr_threshold", type=float, default=.05)
    parser.add_argument("--nms_iou", type=float, default=.5)
    parser.add_argument("--no_nms", action="store_true", help="Skip NMS variants; keep both original baselines")
    parser.add_argument("--max_proposals", type=int, default=100)
    parser.add_argument("--cls_batch_size", type=int, default=8)
    parser.add_argument("--warmup_images", type=int, default=1)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    for name in ("proposal_threshold", "baseline_pr_threshold", "cascade_pr_threshold", "nms_iou"):
        if not 0 <= getattr(args, name) <= 1:
            parser.error(f"--{name} must be in [0, 1]")
    if args.max_proposals < 0 or args.cls_batch_size < 1 or args.warmup_images < 0:
        parser.error("max_proposals/warmup_images must be >= 0; cls_batch_size must be >= 1")
    if Path(args.out_dir).exists():
        parser.error("--out_dir already exists; choose a fresh run directory")
    metadata = read_json(args.test_coco)
    categories = detector_categories(metadata)
    validate_coco(metadata, categories)
    provenance = validate_test_split(args.test_manifest, args.test_coco, args.test_images, metadata)
    index_map = detector_index_map(categories, read_json(args.detector_class_map) if args.detector_class_map else None)
    policy = read_json(args.class_mapping)
    # Inspect labels without constructing either model or changing its checkpoint.
    cls_metadata = torch.load(args.cls_ckpt, map_location="cpu", weights_only=True)
    class_names = list(cls_metadata["classes"])
    del cls_metadata
    resolved = compile_label_policy(policy, class_names, categories)
    reverse_indices = {external: internal for internal, external in index_map.items()}
    internal_policy = {name: None if cid is None else reverse_indices[cid] for name, cid in resolved.items()}
    model = WasteCascade(args.det_ckpt, args.cls_ckpt,
                         {i: categories[cid] for i, cid in index_map.items()}, args.device,
                         classifier_to_detector=internal_policy)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = metadata
        coco_gt.createIndex()
    settings = {key: getattr(args, key) for key in ("nms_iou", "no_nms", "proposal_threshold", "baseline_pr_threshold",
                "cascade_pr_threshold", "max_proposals", "cls_batch_size", "warmup_images")}
    records, summaries = run_experiments(model, coco_gt, args.test_images, index_map, **settings)
    report = {
        "test_images": len(metadata["images"]), "test_gt_boxes": len(metadata.get("annotations", [])),
        "categories": categories, "detector_index_to_coco_id": index_map,
        "provenance": provenance, "settings": settings,
        "label_policy": {"config": policy, "resolved_classifier_to_coco_id": resolved,
                         "sha256": sha256_file(args.class_mapping)},
        "inputs": {name: {"path": str(Path(getattr(args, name)).resolve()), "sha256": sha256_file(getattr(args, name))}
                   for name in ("test_coco", "det_ckpt", "cls_ckpt")},
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "torchvision": version("torchvision"), "pycocotools": version("pycocotools"), "device": str(model.device)},
        "protocol": {
            "metrics": "COCOeval bbox; AP IoU .50:.95; AP50; AR100 averaged over IoU .50:.95. "
                       "AP averages only categories with valid GT; absent categories yield undefined values (JSON null).",
            "precision_recall": "Micro P/R at IoU .50, fixed per-family score threshold, area all, maxDets=100 per image/category; "
                                "COCOeval matches and crowd ignores. Zero precision denominator -> 0; zero GT -> recall null.",
            "label_scope": "All five-class GT remains in evaluation. Unmapped/null classifier predictions are dropped and counted; "
                           "associated missed GT remains FN. No cardboard -> paper rule is inferred.",
            "nms": "Class-agnostic, detector score, float32 pixel xyxy, before proposal cap and classifier crops. "
                   "NMS changes the proposal set; not applied after classification.",
            "scores": "Detector sigmoid max per query; cascade det_score * classifier softmax max (uncalibrated heuristic). "
                      "PR thresholds never prefilter AP/AR exports. Proposal threshold/cap still bound attainable recall.",
            "latency": "Separate inference for all four variants, rotated order per image; CUDA synchronized. "
                       "End-to-end includes transforms, box processing/NMS, crop, forward and remapping inside predict; "
                       "excludes image I/O/decode, model construction, metric computation and configured warmup. "
                       "Detector-only runs do not crop or execute classifier. Timings are milliseconds per image.",
        },
        "experiments": summaries,
    }
    write_results(args.out_dir, report, records)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()