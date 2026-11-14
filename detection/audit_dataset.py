"""Read-only detection split audit. Writes a NEW JSON report, never edits inputs.

SHA256 proves byte identity. 64-bit DCT pHash is a candidate screen, not proof
of duplication or independence. No model construction/inference is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
import PIL
from PIL import Image, ImageOps
import scipy
from scipy.fftpack import dct

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"}
SPLITS = ("train", "valid", "test")
CLASSIFIER_NAMES = {"battery", "biological", "cardboard", "clothes", "glass", "metal", "paper", "plastic", "shoes", "trash"}
SKIP_DIRECTORIES = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "node_modules", "evaluation_runs"}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def phash(image):
    """ImageHash-style pHash: EXIF-normalized L, LANCZOS 32x32, DCT-II 8x8.

    Uses the median of all 64 coefficients including DC; > median yields a bit.
    DCT is unnormalized, first axis 0 then axis 1. The returned integer is 64-bit.
    """
    pixels = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    coefficients = dct(dct(pixels, axis=0), axis=1)[:8, :8]
    bits = (coefficients > np.median(coefficients)).ravel()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


class HammingTree:
    """Exact BK-tree radius queries. Insert UNIQUE hashes; no sampled comparisons."""
    def __init__(self):
        self.root = None

    def add(self, value):
        if self.root is None:
            self.root = (value, {})
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                return
            if distance not in node[1]:
                node[1][distance] = (value, {})
                return
            node = node[1][distance]

    def query(self, value, radius):
        stack = [self.root] if self.root is not None else []
        while stack:
            node = stack.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= radius:
                yield node[0], distance
            low, high = distance - radius, distance + radius
            stack.extend(child for edge, child in node[1].items() if low <= edge <= high)


def pair_counts(left, right=None):
    """Counts unordered image pairs by dataset group without materializing N^2 pairs."""
    counts = Counter()
    a = Counter(row["group"] for row in left)
    if right is None:
        for group, number in a.items():
            if number > 1:
                counts[f"{group}__{group}"] += number * (number - 1) // 2
        for x, y in combinations(sorted(a), 2):
            counts[f"{x}__{y}"] += a[x] * a[y]
    else:
        b = Counter(row["group"] for row in right)
        for x, nx in a.items():
            for y, ny in b.items():
                counts["__".join(sorted((x, y)))] += nx * ny
    return counts


def overlap_analysis(records, radius=6):
    exact_index = defaultdict(list)
    hash_index = defaultdict(list)
    for row in records:
        if row.get("sha256"):
            exact_index[row["sha256"]].append(row)
        if row.get("phash") is not None:
            hash_index[int(row["phash"], 16)].append(row)
    exact_groups, exact_totals = [], Counter()
    for digest, members in sorted(exact_index.items()):
        if len(members) < 2:
            continue
        counts = pair_counts(members)
        exact_totals.update(counts)
        exact_groups.append({"sha256": digest, "image_indices": [r["index"] for r in members],
                             "pair_counts": dict(counts)})
    tree = HammingTree()
    relations, totals, histogram = [], Counter(), Counter()
    participating = defaultdict(set)

    def add_relation(left_hash, right_hash, distance):
        left, right = hash_index[left_hash], hash_index[right_hash]
        counts = pair_counts(left, None if left_hash == right_hash else right)
        exact = Counter()
        # Same file content necessarily has the same pHash within this run.
        if left_hash == right_hash:
            by_sha = defaultdict(list)
            for row in left:
                by_sha[row["sha256"]].append(row)
            for group in by_sha.values():
                exact.update(pair_counts(group))
        nonexact = counts - exact
        if not nonexact:
            return
        totals.update(nonexact)
        histogram[distance] += sum(nonexact.values())
        for row in left + ([] if left_hash == right_hash else right):
            participating[row["group"]].add(row["index"])
        relations.append({"left_phash": f"{left_hash:016x}", "right_phash": f"{right_hash:016x}",
                          "hamming_distance": distance, "candidate_pairs_excluding_identical_sha256": sum(nonexact.values()),
                          "pair_counts_excluding_identical_sha256": dict(nonexact),
                          "review_status": "unreviewed_candidate"})

    for value in sorted(hash_index):
        if len(hash_index[value]) > 1:
            add_relation(value, value, 0)
        for previous, distance in tree.query(value, radius):
            add_relation(value, previous, distance)
        tree.add(value)
    return {
        "exact_sha256": {"duplicate_groups": exact_groups, "pair_counts": dict(exact_totals),
                         "duplicate_group_count": len(exact_groups), "image_pair_count": sum(exact_totals.values())},
        "perceptual_hash": {
            "hash_groups": {f"{h:016x}": [r["index"] for r in group] for h, group in sorted(hash_index.items())},
            "candidate_relations": relations, "candidate_relation_count": len(relations),
            "candidate_pair_counts_excluding_identical_sha256": dict(totals),
            "candidate_image_pair_count_excluding_identical_sha256": sum(totals.values()),
            "candidate_pair_distance_histogram": dict(sorted(histogram.items())),
            "participating_images_by_group": {k: len(v) for k, v in participating.items()},
        },
    }


def box_issues(annotation, image, category_ids):
    reasons = []
    if annotation.get("category_id") not in category_ids:
        reasons.append("unknown_category_id")
    bbox = annotation.get("bbox")
    if (not isinstance(bbox, list) or len(bbox) != 4
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in bbox)):
        return reasons + ["invalid_bbox_format_or_nonfinite"]
    x, y, width, height = bbox
    if width <= 0 or height <= 0:
        reasons.append("nonpositive_bbox_size")
    if image is not None:
        iw, ih = image.get("width"), image.get("height")
        if type(iw) in (int, float) and type(ih) in (int, float):
            if x < -1e-6 or y < -1e-6 or x+width > iw+1e-6 or y+height > ih+1e-6:
                reasons.append("bbox_outside_declared_image")
    if "area" in annotation:
        area = annotation["area"]
        if type(area) not in (int, float) or not math.isfinite(area) or area <= 0:
            reasons.append("invalid_area")
    if annotation.get("iscrowd", 0) not in (0, 1):
        reasons.append("invalid_iscrowd")
    return reasons


def check_coco(split_root, split, project):
    path = split_root / "annotations_coco.json"
    data = read_json(path)
    categories = data.get("categories", [])
    images, annotations = data.get("images", []), data.get("annotations", [])
    errors, warnings = [], []
    ids = [row.get("id") for row in categories]
    names = [str(row.get("name", "")).strip().lower() for row in categories]
    if len(categories) != 5 or len(set(ids)) != len(ids) or len(set(names)) != len(names):
        errors.append({"reason": "invalid_or_duplicate_five_class_categories", "categories": categories})
    if any(type(cid) is not int for cid in ids) or any(not name for name in names):
        errors.append({"reason": "invalid_category_id_or_name"})
    if set(ids) != set(range(5)):
        warnings.append({"reason": "category_ids_not_current_checkpoint_0_to_4", "ids": ids})
    image_map, references = {}, defaultdict(list)
    referenced_paths, missing, escaping = set(), [], []
    root = (split_root / "images").resolve()
    for row in images:
        image_id = row.get("id")
        if type(image_id) is not int or image_id in image_map:
            errors.append({"reason": "invalid_or_duplicate_image_id", "image_id": image_id})
        image_map[image_id] = row
        if any(type(row.get(k)) is not int or row[k] <= 0 for k in ("width", "height")):
            errors.append({"reason": "invalid_image_dimensions", "image_id": image_id})
        filename = row.get("file_name")
        if not isinstance(filename, str) or not filename:
            errors.append({"reason": "invalid_image_filename", "image_id": image_id})
            continue
        file = (root / filename.replace("\\", "/")).resolve()
        if not file.is_relative_to(root):
            escaping.append({"image_id": image_id, "file_name": filename})
            continue
        references[str(file)].append({"image_id": image_id, "width": row.get("width"), "height": row.get("height")})
        referenced_paths.add(file)
        if not file.is_file():
            missing.append({"image_id": image_id, "file_name": filename})
    if escaping:
        errors.append({"reason": "image_reference_escapes_root", "images": escaping})
    if missing:
        errors.append({"reason": "missing_referenced_images", "images": missing})
    duplicate_references = [{"path": file, "image_ids": [r["image_id"] for r in rows]}
                            for file, rows in references.items() if len(rows) > 1]
    if duplicate_references:
        errors.append({"reason": "same_file_referenced_by_multiple_image_records", "groups": duplicate_references})
    ann_ids, duplicate_boxes = set(), defaultdict(list)
    bbox_errors, box_reason_counts, per_category, boxes_per_image = [], Counter(), Counter(), Counter()
    missing_area, missing_crowd = 0, 0
    for ann in annotations:
        aid, image_id = ann.get("id"), ann.get("image_id")
        if type(aid) is not int or aid <= 0 or aid in ann_ids:
            errors.append({"reason": "invalid_or_duplicate_annotation_id", "annotation_id": aid})
        ann_ids.add(aid)
        reasons = box_issues(ann, image_map.get(image_id), set(ids))
        if image_id not in image_map:
            reasons.append("unknown_image_id")
        if reasons:
            bbox_errors.append({"annotation_id": aid, "image_id": image_id, "category_id": ann.get("category_id"),
                                "bbox": repr(ann.get("bbox")), "reasons": reasons})
            box_reason_counts.update(reasons)
        per_category[str(ann.get("category_id"))] += 1
        boxes_per_image[image_id] += 1
        missing_area += "area" not in ann
        missing_crowd += "iscrowd" not in ann
        if not reasons:
            duplicate_boxes[(image_id, ann.get("category_id"), tuple(ann["bbox"]))].append(aid)
    repeated_boxes = [{"image_id": key[0], "category_id": key[1], "bbox": list(key[2]), "annotation_ids": aids}
                      for key, aids in duplicate_boxes.items() if len(aids) > 1]
    if bbox_errors:
        errors.append({"reason": "invalid_annotations", "count": len(bbox_errors)})
    if repeated_boxes:
        warnings.append({"reason": "identical_annotation_geometry", "group_count": len(repeated_boxes)})
    files = sorted(p.resolve() for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    # Referenced image files are read even if an unusual extension was used.
    physical = set(files) | {p for p in referenced_paths if p.is_file()}
    unsupported = Counter(p.suffix.lower() or "<no_extension>" for p in root.rglob("*")
                          if p.is_file() and p.suffix.lower() not in IMAGE_EXTENSIONS and p.resolve() not in referenced_paths)
    orphan = sorted(str(p.relative_to(project)) for p in physical - referenced_paths)
    if orphan:
        warnings.append({"reason": "unreferenced_image_files", "count": len(orphan)})
    labels = split_root / "labels"
    label_files = sorted(labels.rglob("*.txt")) if labels.exists() else []
    summary = {
        "annotations_path": str(path.relative_to(project)), "annotations_sha256": sha256_file(path),
        "images_directory": str(root), "coco_image_records": len(images), "image_files_on_disk": len(physical),
        "bbox_count": len(annotations), "categories": categories, "boxes_by_category_id": dict(per_category),
        "images_without_boxes": sum(boxes_per_image[row.get("id")] == 0 for row in images),
        "invalid_bbox_annotation_count": len(bbox_errors), "bbox_issue_counts": dict(box_reason_counts),
        "bbox_issues": bbox_errors, "duplicate_annotation_groups": repeated_boxes,
        "missing_area_count": missing_area, "missing_iscrowd_count": missing_crowd,
        "missing_image_reference_count": len(missing), "missing_image_references": missing,
        "unreferenced_images": orphan, "non_image_files_in_images_directory": dict(unsupported),
        "labels_directory_exists": labels.is_dir(), "label_txt_file_count": len(label_files),
        "annotation_errors": errors, "annotation_warnings": warnings,
        "metadata": {k: v for k, v in data.items() if k not in ("images", "annotations", "categories")},
    }
    specs = [{"path": str(file), "relative_path": str(file.relative_to(project)), "group": split,
              "role": "detection", "split": split, "coco_references": references.get(str(file), [])}
             for file in sorted(physical)]
    return summary, specs


def discover_classification(project, detection_root):
    candidates, specs, unclassified = [], [], []
    for base, directories, filenames in os.walk(project, followlinks=False):
        base = Path(base)
        directories[:] = sorted(d for d in directories if d not in SKIP_DIRECTORIES
                                 and (base / d).resolve() != detection_root.resolve()
                                 and not (base / d).is_symlink())
        class_dirs = [d for d in directories if d.lower() in CLASSIFIER_NAMES]
        if class_dirs:
            parent = str(base.relative_to(project))
            candidates.append({"root": parent, "class_directories": class_dirs,
                               "training_membership": "unknown_without_training_manifest"})
        relative = base.relative_to(project)
        matched_class = next((p for p in relative.parts if p.lower() in CLASSIFIER_NAMES), None)
        for filename in sorted(filenames):
            file = base / filename
            if file.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if matched_class:
                index = next(i for i, p in enumerate(relative.parts) if p.lower() in CLASSIFIER_NAMES)
                parent = str(Path(*relative.parts[:index]))
                specs.append({"path": str(file.resolve()), "relative_path": str(file.relative_to(project)),
                              "group": f"classification/{parent}", "role": "classification_candidate",
                              "class_name": matched_class, "training_membership": "unverified", "coco_references": []})
            else:
                unclassified.append(str(file.relative_to(project)))
    return {"candidate_imagefolder_roots": candidates, "candidate_image_count": len(specs),
            "status": "candidates_found_training_membership_unverified" if specs else "no_classification_dataset_found_in_project",
            "unclassified_project_image_count": len(unclassified), "unclassified_project_images": unclassified,
            "scope": "Project only, excluding Detection dataset, .venv/.git/caches/evaluation_runs; "
                     "class-name ImageFolder discovery cannot identify arbitrary external/renamed archives."}, specs


def fingerprint(spec):
    row = dict(spec)
    path = Path(row["path"])
    try:
        before = path.stat()
        row["size_bytes"], row["mtime_ns"] = before.st_size, before.st_mtime_ns
        row["sha256"] = sha256_file(path)
        with Image.open(path) as image:
            image.load()  # Detect incomplete/unreadable pixels, not only header errors.
            row["width"], row["height"] = image.size
            row["format"] = image.format
            normalized = ImageOps.exif_transpose(image)
            row["phash"] = f"{phash(normalized):016x}"
            rgb = normalized.convert("RGB")
            row["decoded_rgb_sha256"] = hashlib.sha256(
                f"{rgb.width}x{rgb.height}:RGB:".encode("ascii") + rgb.tobytes()).hexdigest()
            row["grayscale_stddev"] = float(np.asarray(normalized.convert("L")).std())
        after = path.stat()
        row["changed_while_reading"] = (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
        row["decode_error"] = None
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
        row["decode_error"] = f"{type(error).__name__}: {error}"
    return row


def project_provenance(project, detection_root):
    evidence = []
    for folder in [detection_root, *(detection_root / s for s in SPLITS)]:
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in {".md", ".yaml", ".yml", ".txt"}:
                evidence.append({"path": str(path.relative_to(project)), "sha256": sha256_file(path),
                                 "text": path.read_text(encoding="utf-8-sig", errors="replace")[:12000]})
    checkpoint_evidence = []
    import torch  # Only metadata read; never instantiate a model or execute training.
    for relative in ["detection/checkpoints_wastedetect/best_model.pt", "checkpoints/dinov2_garbage_final.pth"]:
        path = project / relative
        if not path.is_file():
            continue
        record = {"path": relative, "sha256_before": sha256_file(path)}
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            record["top_level_keys"] = list(checkpoint)
            record["args"] = checkpoint.get("args")
            record["classes"] = checkpoint.get("classes")
            record["epoch"] = checkpoint.get("epoch")
            del checkpoint
        except Exception as error:
            record["metadata_read_error"] = f"{type(error).__name__}: {error}"
        checkpoint_evidence.append(record)
    legacy = project / "detection/annotations_coco.json"
    legacy_match = None
    if legacy.is_file():
        digest = sha256_file(legacy)
        legacy_match = {"path": "detection/annotations_coco.json", "sha256": digest,
                        "identical_to_split_annotations": [s for s in SPLITS
                            if sha256_file(detection_root / s / "annotations_coco.json") == digest]}
    return {"dataset_provenance_files": evidence, "checkpoint_metadata": checkpoint_evidence,
            "legacy_annotations": legacy_match,
            "split_history_verified": False,
            "limitation": "Directory names, disjoint files and checkpoint path strings do not prove which image contents "
                          "were used for training/model selection or absence of subject/session leakage. No verified split manifest supplied."}


def audit(project, detection_root, radius=6, workers=4):
    started = time.perf_counter()
    project, detection_root = Path(project).resolve(), Path(detection_root).resolve()
    summaries, specs = {}, []
    for split in SPLITS:
        summary, split_specs = check_coco(detection_root / split, split, project)
        summaries[split] = summary
        specs.extend(split_specs)
    classification, cls_specs = discover_classification(project, detection_root)
    specs.extend(cls_specs)
    for index, spec in enumerate(specs):
        spec["index"] = index
    provenance = project_provenance(project, detection_root)
    records = []
    print(f"Fingerprinting {len(specs)} image files with {workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for row in executor.map(fingerprint, specs):
            records.append(row)
            if len(records) % 1000 == 0 or len(records) == len(specs):
                print(f"Hashed/decoded {len(records)}/{len(specs)} images", flush=True)
    for split, summary in summaries.items():
        relevant = [r for r in records if r["group"] == split]
        unreadable = [{"path": r["relative_path"], "error": r["decode_error"]} for r in relevant if r.get("decode_error")]
        dimensions = []
        for row in relevant:
            if row.get("decode_error"):
                continue
            for ref in row["coco_references"]:
                if (row["width"], row["height"]) != (ref["width"], ref["height"]):
                    dimensions.append({"path": row["relative_path"], "image_id": ref["image_id"],
                                       "actual": [row["width"], row["height"]], "declared": [ref["width"], ref["height"]]})
        summary["unreadable_images"] = unreadable
        summary["dimension_mismatches"] = dimensions
        summary["decoded_image_count"] = len(relevant) - len(unreadable)
        summary["coco_valid_for_evaluation"] = not (summary["annotation_errors"] or unreadable or dimensions)
    categories_equal = all(summaries[s]["categories"] == summaries["train"]["categories"] for s in SPLITS)
    print("Searching all unique pHashes for Hamming distance <= " + str(radius), flush=True)
    overlap = overlap_analysis(records, radius)
    # A pHash hash collision is only a candidate. Pixel identity is stronger evidence.
    pixel_groups = defaultdict(list)
    for row in records:
        if row.get("decoded_rgb_sha256"):
            pixel_groups[row["decoded_rgb_sha256"]].append(row)
    overlap["identical_decoded_rgb"] = [
        {"decoded_rgb_sha256": digest, "image_indices": [r["index"] for r in members],
         "pair_counts": dict(pair_counts(members))}
        for digest, members in sorted(pixel_groups.items()) if len(members) > 1]
    changed = []
    for row in records:
        file = Path(row["path"])
        stat = file.stat() if file.exists() else None
        if (stat is None or (stat.st_size, stat.st_mtime_ns) != (row.get("size_bytes"), row.get("mtime_ns"))
                or row.get("changed_while_reading")):
            changed.append(row["relative_path"])
    annotations_stable = all(sha256_file(project / summary["annotations_path"]) == summary["annotations_sha256"]
                             for summary in summaries.values())
    for checkpoint in provenance["checkpoint_metadata"]:
        checkpoint["sha256_after"] = sha256_file(project / checkpoint["path"])
        checkpoint["unchanged"] = checkpoint["sha256_after"] == checkpoint["sha256_before"]
    exact = overlap["exact_sha256"]["pair_counts"]
    near = overlap["perceptual_hash"]["candidate_pair_counts_excluding_identical_sha256"]
    test_exact = sum(n for key, n in exact.items() if key in {"test__train", "test__valid"})
    test_near = sum(n for key, n in near.items() if key in {"test__train", "test__valid"})
    reasons = ["No verified split-generation/source/group manifest establishes the training history of BOTH models."]
    if test_exact:
        reasons.append(f"Confirmed exact test-vs-train/valid overlap: {test_exact} image pairs.")
    if test_near:
        reasons.append(f"Unreviewed non-byte-identical test-vs-train/valid pHash candidates: {test_near} image pairs.")
    if classification["candidate_image_count"] == 0:
        reasons.append("Classification train/validation images were not found in this project; Detection-test overlap with them is unassessed.")
    else:
        reasons.append("Classification candidates were compared, but actual checkpoint training membership is unverified.")
    if not categories_equal or not all(s["coco_valid_for_evaluation"] for s in summaries.values()):
        reasons.append("Category/annotation/image integrity problems must be resolved before evaluation.")
    if changed or not annotations_stable:
        reasons.append("Input files changed during the audit; results require a stable snapshot rerun.")
    return {
        "schema_version": 1, "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project), "detection_root": str(detection_root),
        "execution": {"mode": "read_only_inputs", "model_inference_run": False, "training_run": False,
                      "elapsed_seconds": time.perf_counter()-started,
                      "python": platform.python_version(), "Pillow": PIL.__version__, "numpy": np.__version__, "scipy": scipy.__version__},
        "methods": {"sha256": "SHA-256 of original file bytes; exhaustive all existing image files, including unreferenced images.",
                    "phash": "64-bit pHash: EXIF transpose, grayscale, LANCZOS 32x32, unnormalized DCT-II axes 0 then 1, "
                             "top-left 8x8 including DC, bits greater than median of all 64 coefficients.",
                    "phash_max_hamming_distance": radius, "search": "Exact BK-tree radius search over all unique pHashes; no sampling.",
                    "candidate_semantics": "pHash candidates exclude identical SHA256 pairs. They need visual/source review; "
                                           "simple/low-texture images may collide. Crop/rotation/session leakage can be missed.",
                    "pair_count_semantics": "Unordered image file pairs, not independent scenes. All members are listed by image inventory index; "
                                            "hash group relations compactly represent full Cartesian products, excluding exact-SHA pairs.",
                    "decoded_rgb": "SHA256 of EXIF-normalized RGB dimensions and decoded pixel bytes, without resizing; not used as a replacement for file SHA256.",
                    "bbox_validation": "COCO xywh finite, positive size, bounds within declared image (1e-6 tolerance), "
                                       "known IDs, positive area if provided; actual image dimensions also verified. No boxes clipped or modified.",
                    "labels_scope": "labels/*.txt are inventoried; COCO JSON is the evaluation ground truth. YOLO-vs-COCO semantic equivalence is not asserted."},
        "splits": summaries, "categories_consistent_across_splits": categories_equal,
        "classification_audit": classification, "provenance": provenance,
        "input_stability": {"changed_image_files": changed, "annotation_files_unchanged": annotations_stable},
        "overlap": overlap, "image_inventory": records,
        "independence_assessment": {
            "independent": False, "recommend_independent_true": False,
            "status": "exact_overlap_detected" if test_exact else "independence_not_established",
            "test_vs_detection_train_valid_exact_pairs": test_exact,
            "test_vs_detection_train_valid_near_candidate_pairs": test_near,
            "reasons": reasons,
            "next_evidence_needed": ["Review test-vs-train/valid near candidates and retain source-based decisions.",
                                     "Provide actual Classification train/validation data or a verifiable image-hash manifest tied to its checkpoint.",
                                     "Provide original dataset provenance and split method, including source/session/group independence and checkpoint training membership.",
                                     "Resolve any annotation errors without silently clipping/changing the audited inputs."],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--dataset", default="WasteDetect_Dataset")
    parser.add_argument("--out", default="evaluation_runs/dataset_overlap_report.json")
    parser.add_argument("--phash_distance", type=int, default=6)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 0 <= args.phash_distance <= 64 or args.workers < 1:
        parser.error("phash_distance must be 0..64 and workers >= 1")
    project = Path(args.project).resolve()
    dataset, output = (project / args.dataset).resolve(), (project / args.out).resolve()
    if output.exists():
        parser.error("Report already exists; select a new output path to preserve prior audits")
    report = audit(project, dataset, args.phash_distance, args.workers)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create protects existing reports too. Input images/annotations are never written.
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False, default=str)
    print(json.dumps({"report": str(output), "counts": {s: {
        "images": report["splits"][s]["coco_image_records"], "boxes": report["splits"][s]["bbox_count"],
        "invalid_boxes": report["splits"][s]["invalid_bbox_annotation_count"],
        "missing_images": report["splits"][s]["missing_image_reference_count"]} for s in SPLITS},
        "exact_pair_counts": report["overlap"]["exact_sha256"]["pair_counts"],
        "near_candidate_pair_counts": report["overlap"]["perceptual_hash"]["candidate_pair_counts_excluding_identical_sha256"],
        "assessment": report["independence_assessment"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
