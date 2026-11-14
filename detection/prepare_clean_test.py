"""Copy a SHA256-clean test candidate from a validated dataset audit.

Only exact-byte overlaps determine selection. No inference, COCOeval, label
merging, pHash-based exclusion, source mutation, or independence approval.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import copy
import csv
from datetime import datetime, timezone
from itertools import combinations, product
import json
from pathlib import Path
import re
import shutil

from PIL import Image

from audit_dataset import IMAGE_EXTENSIONS, SPLITS, check_coco, overlap_analysis, read_json, sha256_file


def require(condition, message):
    if not condition:
        raise ValueError(message)


def contained(root, relative):
    require(isinstance(relative, str) and bool(relative), "Missing relative filename")
    relative = Path(relative.replace("\\", "/"))
    require(not relative.is_absolute(), f"Absolute filename is not allowed: {relative}")
    result = (root / relative).resolve()
    require(result.is_relative_to(root.resolve()), f"Path escapes root: {relative}")
    return result


def validate_audit(report, project, dataset):
    """Validate schema AND recompute overlap indices from the cached inventory."""
    require(isinstance(report, dict) and report.get("schema_version") == 1, "Unsupported audit schema")
    for key, expected in [("splits", dict), ("image_inventory", list), ("overlap", dict), ("methods", dict)]:
        require(isinstance(report.get(key), expected), f"Audit field {key} has wrong/missing type")
    require(Path(report["project_root"]).resolve() == project, "Audit project_root differs from current project")
    require(Path(report["detection_root"]).resolve() == dataset, "Audit detection_root differs from current dataset")
    require(set(SPLITS).issubset(report["splits"]), "Audit must contain train/valid/test")
    stability = report.get("input_stability", {})
    require(stability.get("annotation_files_unchanged") is True and stability.get("changed_image_files") == [],
            "Audit did not establish stable input images/annotations")
    changes = stability.get("post_audit_directory_inventory_changes", {})
    require(not changes.get("added_image_paths") and not changes.get("removed_image_paths"), "Audit inventory was changing")
    inventory = report["image_inventory"]
    require(bool(inventory), "Empty audit image inventory")
    indices, paths = set(), set()
    for row in inventory:
        require(isinstance(row, dict), "Inventory entry must be an object")
        index, path = row.get("index"), row.get("path")
        require(type(index) is int and index >= 0 and index not in indices, "Duplicate/invalid inventory index")
        require(isinstance(path, str) and Path(path).resolve().is_relative_to(project), "Inventory path outside project")
        require(str(Path(path).resolve()) not in paths, "Duplicate inventory file path")
        require(re.fullmatch(r"[0-9a-f]{64}", row.get("sha256", "")) is not None, "Invalid inventory SHA256")
        require(re.fullmatch(r"[0-9a-f]{16}", row.get("phash", "")) is not None, "Missing/invalid cached pHash")
        require(row.get("decode_error") is None and row.get("changed_while_reading") is False,
                "Audit contains unreadable/changing images")
        require(isinstance(row.get("group"), str), "Missing inventory group")
        if row["group"] in SPLITS:
            require(row.get("role") == "detection" and row.get("split") == row["group"], "Inventory split/role mismatch")
            require(Path(path).resolve().is_relative_to(dataset / row["group"] / "images"), "Wrong split image path")
        indices.add(index)
        paths.add(str(Path(path).resolve()))
    radius = report["methods"].get("phash_max_hamming_distance")
    require(type(radius) is int and 0 <= radius <= 64, "Invalid audit pHash radius")
    # JSON object keys (notably Hamming histogram distances) become strings.
    expected = json.loads(json.dumps(overlap_analysis(inventory, radius)))
    actual = json.loads(json.dumps(report["overlap"]))
    require(actual.get("exact_sha256") == expected["exact_sha256"], "SHA256 duplicate-group/count structure is inconsistent")
    phash = actual.get("perceptual_hash", {})
    require(isinstance(phash, dict), "Invalid perceptual_hash structure")
    # Manual review notes/status may have been added; matching evidence must be unchanged.
    for key, value in expected["perceptual_hash"].items():
        if key == "candidate_relations":
            supplied = phash.get(key)
            require(isinstance(supplied, list) and len(supplied) == len(value), "pHash relation count mismatch")
            for original, computed in zip(supplied, value):
                require(isinstance(original, dict) and all(original.get(k) == v for k, v in computed.items() if k != "review_status"),
                        "pHash relation references/distance/counts are inconsistent")
        else:
            require(phash.get(key) == value, f"pHash structure/count mismatch: {key}")
    return inventory


def validate_source_metadata(report, project, dataset):
    metadata, rows_by_split = {}, {}
    for split in SPLITS:
        entry = report["splits"][split]
        path = dataset / split / "annotations_coco.json"
        require(sha256_file(path) == entry["annotations_sha256"], f"Stale audit: {split} COCO changed")
        require((project / entry["annotations_path"]).resolve() == path, "Audit annotation path mismatch")
        summary, specs = check_coco(dataset / split, split, project)
        require(not summary["annotation_errors"], f"Invalid source {split} COCO: {summary['annotation_errors']}")
        require(summary["coco_image_records"] == entry["coco_image_records"] and summary["bbox_count"] == entry["bbox_count"],
                "Audit image/annotation counts mismatch")
        data = read_json(path)
        require(summary["categories"] == entry["categories"], "Audit categories mismatch")
        rows = {str(Path(r["path"]).resolve()): r for r in report["image_inventory"] if r["group"] == split}
        require(set(rows) == {s["path"] for s in specs}, f"Stale audit: {split} directory file inventory differs")
        require(len(rows) == len(data["images"]), "Ambiguous/unreferenced source images; fresh audit/curation needed")
        for spec in specs:
            require(spec["coco_references"] == rows[spec["path"]]["coco_references"], "Audit COCO image references differ")
        metadata[split], rows_by_split[split] = data, rows
    mappings = [{row["id"]: row["name"] for row in metadata[s]["categories"]} for s in SPLITS]
    require(all(mapping == mappings[0] for mapping in mappings), "Category IDs/names differ across splits")
    return metadata, rows_by_split


def snapshot(paths, workers):
    paths = sorted(set(Path(p).resolve() for p in paths))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        hashes = list(executor.map(sha256_file, paths))
    return dict(zip(map(str, paths), hashes))


def test_rows_by_id(test_rows):
    result = {}
    for row in test_rows.values():
        require(len(row["coco_references"]) == 1, "One test file must correspond to exactly one COCO image")
        result[row["coco_references"][0]["image_id"]] = row
    return result


def annotation_signature(annotations):
    return sorted(json.dumps({k: v for k, v in ann.items() if k not in {"id", "image_id"}}, sort_keys=True)
                  for ann in annotations)


def plan_selection(data, inventory, rows_by_id):
    forbidden = defaultdict(list)
    for row in inventory:
        if row["group"] in {"train", "valid"}:
            forbidden[row["sha256"]].append(row)
    groups, annotations = defaultdict(list), defaultdict(list)
    for image in data["images"]:
        groups[rows_by_id[image["id"]]["sha256"]].append(image)
    for ann in data["annotations"]:
        annotations[ann["image_id"]].append(ann)
    kept, removed, duplicate_groups = set(), [], []
    for digest, images in sorted(groups.items()):
        ordered = sorted(images, key=lambda im: (im["id"], im["file_name"]))
        peers = forbidden.get(digest, [])
        representative = None if peers else ordered[0]["id"]
        if representative is not None:
            kept.add(representative)
        if len(ordered) > 1:
            signatures = [annotation_signature(annotations[im["id"]]) for im in ordered]
            duplicate_groups.append({"sha256": digest, "original_test_image_ids": [im["id"] for im in ordered],
                                     "file_names": [im["file_name"] for im in ordered],
                                     "kept_representative_image_id": representative,
                                     "all_excluded_by_train_valid_overlap": bool(peers),
                                     "annotations_agree_ignoring_ids": all(s == signatures[0] for s in signatures),
                                     "annotation_ids_by_image": {str(im["id"]): [a["id"] for a in annotations[im["id"]]] for im in ordered}})
        for image in ordered:
            if image["id"] == representative:
                continue
            peer_groups = {r["group"] for r in peers}
            reasons = [f"sha256_matches_{split}" for split in ("train", "valid") if split in peer_groups]
            if not peers:
                reasons = ["duplicate_sha256_within_test_candidate"]
            removed.append({"image_id": image["id"], "file_name": image["file_name"], "sha256": digest,
                            "primary_reason": "sha256_overlap_train_valid" if peers else "duplicate_sha256_within_test_candidate",
                            "reason_codes": reasons, "removed_bbox_count": len(annotations[image["id"]]),
                            "removed_annotation_ids": [a["id"] for a in annotations[image["id"]]],
                            "representative_image_id": representative,
                            "matching_reference_images": [{"group": r["group"], "path": r["path"],
                                                           "audit_index": r["index"]} for r in peers]})
    candidate = copy.deepcopy(data)
    candidate["images"] = [im for im in data["images"] if im["id"] in kept]
    candidate["annotations"] = [ann for ann in data["annotations"] if ann["image_id"] in kept]
    return candidate, sorted(removed, key=lambda r: r["image_id"]), duplicate_groups


def statistics(data):
    counts = Counter(ann["category_id"] for ann in data["annotations"])
    images = defaultdict(set)
    for ann in data["annotations"]:
        images[ann["category_id"]].add(ann["image_id"])
    return {"images": len(data["images"]), "bounding_boxes": len(data["annotations"]),
            "classes": [{"category_id": cat["id"], "name": cat["name"], "bounding_boxes": counts[cat["id"]],
                         "images_containing_class": len(images[cat["id"]])} for cat in data["categories"]]}


def annotation_review(data, duplicate_groups):
    """Provide original labels of retained representatives and their excluded copies."""
    images = {im["id"]: im for im in data["images"]}
    annotations = defaultdict(list)
    for ann in data["annotations"]:
        annotations[ann["image_id"]].append(ann)
    return [{**group, "review_status": "pending_manual_review", "source_records": [
        {"image": images[image_id], "annotations": annotations[image_id],
         "kept_representative": image_id == group["kept_representative_image_id"]}
        for image_id in group["original_test_image_ids"]]}
        for group in duplicate_groups
        if not group["annotations_agree_ignoring_ids"] and group["kept_representative_image_id"] is not None]


def phash_review_pairs(report, kept_ids, output):
    inventory = {row["index"]: row for row in report["image_inventory"]}
    groups = report["overlap"]["perceptual_hash"]["hash_groups"]
    result, seen = [], set()
    for relation in report["overlap"]["perceptual_hash"]["candidate_relations"]:
        a, b = relation["left_phash"], relation["right_phash"]
        pairs = combinations(groups[a], 2) if a == b else product(groups[a], groups[b])
        for ai, bi in pairs:
            left, right = inventory[ai], inventory[bi]
            if left["sha256"] == right["sha256"] or "test" not in (left["group"], right["group"]):
                continue
            key = tuple(sorted((ai, bi)))
            if key in seen:
                continue
            seen.add(key)
            if left["group"] != "test":
                left, right = right, left
            def side(row):
                image_id = row["coco_references"][0]["image_id"] if row["group"] == "test" else None
                retained = image_id in kept_ids if row["group"] == "test" else None
                path = row["path"]
                if retained:
                    path = str(output / "images" / Path(row["path"]).relative_to(Path(report["detection_root"]) / "test" / "images"))
                return {"group": row["group"], "original_path": row["path"], "review_path": path,
                        "image_id": image_id, "retained_in_candidate": retained,
                        "sha256": row["sha256"], "audit_index": row["index"]}
            left, right = side(left), side(right)
            actionable = left["retained_in_candidate"] and (right["group"] != "test" or right["retained_in_candidate"])
            result.append({"pair_id": len(result)+1, "left": left, "right": right,
                           "hamming_distance": relation["hamming_distance"],
                           "actionable_after_sha_clean": bool(actionable), "review_status": "pending_manual_review",
                           "manual_decision": "", "notes": ""})
    return result


def save_json(path, value):
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def save_review_csv(path, pairs):
    columns = ["pair_id", "hamming_distance", "actionable_after_sha_clean", "left_group", "left_image_id",
               "left_retained", "left_path", "left_original_path", "right_group", "right_image_id",
               "right_retained", "right_path", "right_original_path", "review_status", "manual_decision", "notes"]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for pair in pairs:
            row = {k: pair[k] for k in ["pair_id", "hamming_distance", "actionable_after_sha_clean", "review_status", "manual_decision", "notes"]}
            for label in ("left", "right"):
                side = pair[label]
                row.update({f"{label}_group": side["group"], f"{label}_image_id": side["image_id"],
                            f"{label}_retained": side["retained_in_candidate"], f"{label}_path": side["review_path"],
                            f"{label}_original_path": side["original_path"]})
            writer.writerow(row)


def build_candidate(project, dataset, audit_path, output, workers=4):
    project, dataset, audit_path, output = [Path(p).resolve() for p in (project, dataset, audit_path, output)]
    require(not output.exists(), "Candidate output already exists; nothing was overwritten")
    require(output != dataset and output.is_relative_to(dataset), "Candidate must be a new directory under dataset")
    require(all(not output.is_relative_to(dataset/s) for s in SPLITS), "Output cannot be inside an original split")
    audit_digest = sha256_file(audit_path)
    report = read_json(audit_path)
    print("Validating audit schema and recomputing overlap indices", flush=True)
    inventory = validate_audit(report, project, dataset)
    metadata, rows_by_split = validate_source_metadata(report, project, dataset)
    test_data = metadata["test"]
    by_id = test_rows_by_id(rows_by_split["test"])
    require(set(by_id) == {im["id"] for im in test_data["images"]}, "Test COCO/audit ID mismatch")
    print("Rechecking current SHA256 of every audited image (no model inference)", flush=True)
    guarded_paths = [row["path"] for row in inventory] + [dataset/s/"annotations_coco.json" for s in SPLITS]
    # Label files are copied unchanged as sidecars, not regenerated from COCO.
    test_label_files = list((dataset/"test"/"labels").rglob("*.txt"))
    guarded_paths += test_label_files
    guarded_paths += [project/row["path"] for row in report.get("provenance", {}).get("checkpoint_metadata", [])]
    before = snapshot(guarded_paths, workers)
    for row in inventory:
        require(before[row["path"]] == row["sha256"], f"Stale audit image hash: {row['path']}")
    for split in SPLITS:
        require(before[str(dataset/split/"annotations_coco.json")] == report["splits"][split]["annotations_sha256"],
                "Annotation changed while planning")
    candidate, removed, duplicate_groups = plan_selection(test_data, inventory, by_id)
    annotation_review_groups = annotation_review(test_data, duplicate_groups)
    kept_ids = {im["id"] for im in candidate["images"]}
    require(bool(kept_ids), "No images survive SHA256 cleaning; candidate was not created")
    pairs = phash_review_pairs(report, kept_ids, output)
    active_pairs = [pair for pair in pairs if pair["actionable_after_sha_clean"]]
    output.mkdir(parents=True, exist_ok=False)
    (output/"images").mkdir()
    (output/"labels").mkdir()
    manifest = {"split": "test", "independent": False, "build_status": "in_progress",
                "description": "SHA256-clean candidate only; pHash manual review, Classification overlap and split provenance remain unresolved.",
                "independent_of": [], "test_coco": "annotations_coco.json", "test_images": "images", "references": [],
                "source_audit": str(audit_path), "source_audit_sha256": audit_digest}
    save_json(output/"test_manifest.json", manifest)
    candidate_inventory, missing_labels = [], []
    print(f"Copying {len(kept_ids)} selected images into {output}", flush=True)
    for image in candidate["images"]:
        image_id, filename = image["id"], image["file_name"]
        source = contained(dataset/"test"/"images", filename)
        destination = contained(output/"images", filename)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        require(sha256_file(destination) == by_id[image_id]["sha256"], "Copied image hash mismatch")
        with Image.open(destination) as pil:
            pil.load()
            require(pil.size == (image["width"], image["height"]), "Copied image/COCO dimensions mismatch")
        label_relative = Path(filename.replace("\\", "/")).with_suffix(".txt").as_posix()
        label_source = contained(dataset/"test"/"labels", label_relative)
        label_destination = contained(output/"labels", label_relative)
        label_digest = None
        if label_source.is_file():
            label_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(label_source, label_destination)
            label_digest = sha256_file(label_destination)
            require(label_digest == before[str(label_source)], "Copied label hash mismatch")
        else:
            missing_labels.append({"image_id": image_id, "file_name": filename})
        candidate_inventory.append({"image_id": image_id, "file_name": filename,
                                    "source_path": str(source), "sha256": by_id[image_id]["sha256"],
                                    "phash": by_id[image_id]["phash"], "label_sha256": label_digest})
    save_json(output/"annotations_coco.json", candidate)
    validated, _ = check_coco(output, "test_clean_candidate", project)
    require(not validated["annotation_errors"] and not validated["unreferenced_images"], "New COCO validation failed")
    require(validated["coco_image_records"] == len(kept_ids), "Candidate image count mismatch")
    saved = read_json(output/"annotations_coco.json")
    require(saved == candidate, "Saved COCO differs from selected original records")
    require(candidate["categories"] == test_data["categories"], "Categories were changed")
    candidate_hashes = {row["sha256"] for row in candidate_inventory}
    reference_hashes = {row["sha256"] for row in inventory if row["group"] in {"train", "valid"}}
    require(len(candidate_hashes) == len(kept_ids) and not candidate_hashes.intersection(reference_hashes), "SHA256 cleaning failed")
    print("Verifying original image/annotation/label and checkpoint hashes are unchanged", flush=True)
    after = snapshot(guarded_paths, workers)
    require(before == after, "Source/checkpoint changed during build; candidate is not approved")
    require(sha256_file(audit_path) == audit_digest, "Source audit changed during build")
    # Reject additions/removals during the run, including images not in the audit.
    validate_source_metadata(report, project, dataset)
    reason_counts = Counter(row["primary_reason"] for row in removed)
    reason_boxes = Counter()
    for row in removed:
        reason_boxes[row["primary_reason"]] += row["removed_bbox_count"]
    cleaning_report = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "clean_candidate_pending_review", "independent": False,
        "source_audit": {"path": str(audit_path), "sha256": audit_digest, "schema_and_overlap_indices_validated": True},
        "source_test_coco": {"path": str(dataset/"test"/"annotations_coco.json"), "sha256": before[str(dataset/"test"/"annotations_coco.json")]},
        "candidate_coco_sha256": sha256_file(output/"annotations_coco.json"),
        "selection_policy": {"cross_split": "Exclude ALL original test images whose file SHA256 matches any audited train/valid image.",
                             "within_test": "Among remaining identical SHA256 images, keep the smallest original COCO image ID; filename is the tie-breaker.",
                             "annotations": "Keep all unchanged annotations of the selected representative. Never merge labels from excluded copies.",
                             "phash": "Export only; no candidate image removed because of pHash.",
                             "model_predictions_or_metrics_used": False},
        "statistics": {"original": statistics(test_data), "retained": statistics(candidate),
                       "removed_images_by_primary_reason": dict(reason_counts), "removed_boxes_by_primary_reason": dict(reason_boxes),
                       "original_within_test_duplicate_groups": len(duplicate_groups),
                       "original_images_in_within_test_duplicate_groups": sum(len(g["original_test_image_ids"]) for g in duplicate_groups),
                       "original_within_test_exact_pairs": report["overlap"]["exact_sha256"]["pair_counts"].get("test__test", 0),
                       "duplicate_groups_with_differing_annotations": sum(not g["annotations_agree_ignoring_ids"] for g in duplicate_groups)},
        "removed_images": removed, "within_test_sha256_groups": duplicate_groups,
        "annotation_review": {"retained_groups_with_differing_annotations": len(annotation_review_groups),
                              "review_file": "annotation_duplicate_review.json", "manual_review_completed": False},
        "phash_review": {"max_hamming_distance": report["methods"]["phash_max_hamming_distance"],
                         "original_test_related_pairs": len(pairs), "actionable_after_sha_clean_pairs": len(active_pairs),
                         "actionable_pairs_by_groups": dict(Counter("__".join(sorted((p["left"]["group"], p["right"]["group"]))) for p in active_pairs)),
                         "manual_review_completed": False, "pairs_removed_automatically": 0},
        "validation": {"image_paths_and_dimensions_verified": True, "copied_image_sha256_verified": True,
                       "coco_records_unchanged_except_image_annotation_filtering": True,
                       "remaining_within_candidate_exact_sha256_pairs": 0, "remaining_train_valid_exact_sha256_pairs": 0,
                       "original_guarded_files": len(before), "original_guarded_files_unchanged": before == after,
                       "source_audit_unchanged": True, "coco_annotation_errors": validated["annotation_errors"],
                       "coco_annotation_warnings": validated["annotation_warnings"], "missing_label_sidecars": missing_labels,
                       "label_sidecars_copied": len(candidate_inventory)-len(missing_labels),
                       "label_semantics": "Sidecar bytes preserved; COCO is evaluation ground truth. No YOLO-vs-COCO semantic equivalence inferred."},
        "unresolved": ["pHash candidates need manual visual/source review; pHash matches are not confirmed duplicates.",
                       "Classification training/validation images or a verifiable checkpoint-bound hash manifest have not been checked.",
                       "Original source/split-generation/checkpoint training history and subject/session/crop leakage remain unverified.",
                       "Any annotation differences among byte-identical copies require curator review; representative choice is ID-based only."],
        "model_inference_run": False, "cocoeval_run": False, "training_run": False,
    }
    save_json(output/"cleaning_report.json", cleaning_report)
    save_json(output/"image_manifest.json", candidate_inventory)
    save_json(output/"annotation_duplicate_review.json", annotation_review_groups)
    save_json(output/"phash_review_pairs.json", pairs)
    save_review_csv(output/"phash_review_all_test_pairs.csv", pairs)
    save_review_csv(output/"phash_review_candidate_pairs.csv", active_pairs)
    manifest["build_status"] = "complete_candidate_pending_review"
    save_json(output/"test_manifest.json", manifest)
    return cleaning_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--dataset", default="WasteDetect_Dataset")
    parser.add_argument("--audit_report", default="evaluation_runs/dataset_overlap_report.json")
    parser.add_argument("--out_dir", default="WasteDetect_Dataset/test_clean_candidate")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    project = Path(args.project).resolve()
    result = build_candidate(project, project/args.dataset, project/args.audit_report, project/args.out_dir, args.workers)
    print(json.dumps({"output": str(project/args.out_dir), "statistics": result["statistics"],
                      "phash_review": result["phash_review"], "validation": result["validation"],
                      "independent": result["independent"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
