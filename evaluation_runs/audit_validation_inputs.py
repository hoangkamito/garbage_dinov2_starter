"""Read-only validation/classifier overlap screening; never reads Test images."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "detection"))
from audit_dataset import phash, sha256_file, HammingTree
from PIL import Image, ImageOps
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder


def letterbox(image, size=256):
    ratio = size / max(image.size)
    width, height = [max(1, round(n * ratio)) for n in image.size]
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(image.resize((width, height), Image.Resampling.BILINEAR), ((size-width)//2, (size-height)//2))
    return canvas


def fingerprint(spec):
    row = dict(spec)
    for prefix in ["original", "standardized"]:
        path = Path(row[f"{prefix}_path"])
        before = path.stat()
        row[f"{prefix}_sha256"] = sha256_file(path)
        with Image.open(path) as image:
            image.load()
            row[f"{prefix}_phash"] = f"{phash(ImageOps.exif_transpose(image)):016x}"
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Image changed while reading: {path}")
        row[f"{prefix}_stat"] = [after.st_size, after.st_mtime_ns]
    return row


def run(output, classification_root, workers=4):
    output.mkdir(parents=True, exist_ok=True)
    valid_path = ROOT/"WasteDetect_Dataset/valid_clean_candidate/annotations_coco.json"
    valid = json.loads(valid_path.read_text(encoding="utf-8"))
    original_valid = json.loads((ROOT/"WasteDetect_Dataset/valid/annotations_coco.json").read_text(encoding="utf-8"))
    sha_valid = json.loads((ROOT/"WasteDetect_Dataset/valid_sha_clean_candidate/annotations_coco.json").read_text(encoding="utf-8"))
    counts = {"valid_original": [len(original_valid["images"]), len(original_valid["annotations"])],
              "valid_sha_clean_candidate": [len(sha_valid["images"]), len(sha_valid["annotations"])],
              "valid_clean_candidate": [len(valid["images"]), len(valid["annotations"])]}
    geometries = defaultdict(list)
    for ann in valid["annotations"]:
        geometries[(ann["image_id"], ann["category_id"], tuple(ann["bbox"]))].append(ann["id"])
    duplicate_annotations = [{"image_id": k[0], "category_id": k[1], "bbox": k[2], "annotation_ids": v}
                             for k, v in geometries.items() if len(v) > 1]
    audit = json.loads((ROOT/"evaluation_runs/dataset_overlap_report.json").read_text(encoding="utf-8"))
    old_rows = {Path(r["path"]).name.casefold(): r for r in audit["image_inventory"] if r["group"] == "valid"}
    valid_rows = []
    for image in valid["images"]:
        path = ROOT/"WasteDetect_Dataset/valid_clean_candidate/images"/image["file_name"]
        digest = sha256_file(path)
        if digest != old_rows[image["file_name"].casefold()]["sha256"]:
            raise ValueError("Validation content differs from audited source")
        with Image.open(path) as pil:
            pil.load()
            if pil.size != (image["width"], image["height"]):
                raise ValueError("Validation image/COCO dimensions mismatch")
            box_hash = f"{phash(letterbox(pil.convert('RGB'))):016x}"
        valid_rows.append({"image_id": image["id"], "file_name": image["file_name"], "path": str(path),
                           "sha256": digest, "native_phash": old_rows[image["file_name"].casefold()]["phash"], "letterbox_phash": box_hash})
    source = ImageFolder(str(classification_root/"standardized_256"))
    indices = np.arange(len(source))
    targets = np.asarray(source.targets)
    train, temp = train_test_split(indices, test_size=.20, random_state=42, stratify=targets)
    val, unused = train_test_split(temp, test_size=.50, random_state=42, stratify=targets[temp])
    subset = {int(i): name for name, values in [("cls_train", train), ("cls_val", val), ("cls_test", unused)] for i in values}
    specs = []
    for index, (path, category) in enumerate(source.samples):
        relative = Path(path).relative_to(classification_root/"standardized_256")
        specs.append({"index": index, "subset": subset[index], "class_name": source.classes[category],
                      "original_path": str(classification_root/"original"/relative), "standardized_path": path})
    cache_path = output/"classifier_fingerprints.jsonl"
    cached = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                cached[row["standardized_path"]] = row
            except json.JSONDecodeError:
                pass  # An interrupted last line will be recomputed.
    rows, pending = [], []
    for spec in specs:
        row = cached.get(spec["standardized_path"])
        fresh = row is not None and row["subset"] == spec["subset"]
        if fresh:
            for prefix in ["original", "standardized"]:
                stat = Path(spec[f"{prefix}_path"]).stat()
                fresh = fresh and row[f"{prefix}_stat"] == [stat.st_size, stat.st_mtime_ns]
        if fresh:
            rows.append(row)
        else:
            pending.append(spec)
    print(f"Classifier fingerprint cache: {len(rows)} reused, {len(pending)} pending", flush=True)
    with cache_path.open("a", encoding="utf-8") as stream, ThreadPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(fingerprint, pending):
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False)+"\n")
            stream.flush()
            if len(rows) % 1000 == 0:
                print(f"Classification images checked: {len(rows)}/{len(specs)}", flush=True)
    rows.sort(key=lambda r: r["index"])
    hashes = defaultdict(list)
    exact = defaultdict(list)
    for row in rows:
        for prefix in ["original", "standardized"]:
            hashes[int(row[f"{prefix}_phash"], 16)].append((row, prefix))
            exact[row[f"{prefix}_sha256"]].append((row, prefix))
    tree = HammingTree()
    for h in sorted(hashes):
        tree.add(h)
    pairs, exact_pairs = [], []
    for image in valid_rows:
        seen = {}
        for query_name in ["native_phash", "letterbox_phash"]:
            query = int(image[query_name], 16)
            for match, distance in tree.query(query, 6):
                for row, variant in hashes[match]:
                    key = row["index"]
                    if key not in seen or distance < seen[key]["hamming_distance"]:
                        seen[key] = {"valid_image_id": image["image_id"], "valid_file_name": image["file_name"],
                                     "valid_path": image["path"], "cls_index": key, "cls_split": row["subset"],
                                     "cls_class_name": row["class_name"], "classification_path": row[f"{variant}_path"],
                                     "classification_sha256": row[f"{variant}_sha256"], "valid_sha256": image["sha256"],
                                     "hamming_distance": distance, "matched_via": query_name+"_vs_"+variant,
                                     "review_status": "pending_manual_review", "manual_decision": ""}
        pairs.extend(seen.values())
        for row, variant in exact.get(image["sha256"], []):
            exact_pairs.append({"valid_image_id": image["image_id"], "cls_index": row["index"], "cls_split": row["subset"], "variant": variant})
    report = {"counts_from_current_coco": counts, "reported_4012_vs_actual": {
        "actual_sha_clean_boxes": len(sha_valid["annotations"]), "stale_summary_value": 4012,
        "authoritative_source": "valid_sha_clean_candidate/annotations_coco.json", "current_tuning_boxes": len(valid["annotations"])},
        "validation_coco_sha256": sha256_file(valid_path), "validation_image_manifest": valid_rows,
        "duplicate_annotation_geometry": duplicate_annotations,
        "classification_source": str(classification_root), "notebook_split_reconstruction": dict(Counter(subset.values())),
        "notebook_sha256": sha256_file(Path('C:/Users/hoang/Downloads/Copy_of_Trainning_DINOv2.ipynb')),
        "exact_pairs": exact_pairs, "phash_pairs": pairs,
        "phash_train_val_pending": sum(p["cls_split"] in {"cls_train", "cls_val"} for p in pairs),
        "phash_all_subset_pending": len(pairs), "independent": False,
        "changes_to_data": False, "classification_split_membership_basis": "ImageFolder ordering plus the notebook's two stratified seed-42 splits; checkpoint lacks a bound training hash manifest.",
        "limitations": "pHash is screening only; no candidate was auto-classified or removed. Native and approximate letterbox hashes checked; transformed crop/source leakage can remain."}
    (output/"validation_data_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if pairs:
        with (output/"validation_classification_review.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
            writer.writeheader(); writer.writerows(pairs)
    print(json.dumps({k: v for k, v in report.items() if k not in {"validation_image_manifest", "phash_pairs"}}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT/"evaluation_runs/val_grid_search_01/data_audit"))
    parser.add_argument("--classification_root", default="D:/Download Acer/Garbage Dataset")
    args = parser.parse_args()
    run(Path(args.out), Path(args.classification_root))
