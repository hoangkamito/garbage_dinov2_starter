"""Dataset Manifest Builder, COCO-to-YOLO Verifier, and Separated ZIP Packager for DG38 Colab Runs.

Enforces strict train/validation/test separation and cross-platform path portability:
- Classification Train/Val (9,807 + 1,226 = 11,033 images) is strictly separated from Test (1,226 images).
- Detection Train/Clean-Val (8,289 + 1,743 = 10,032 images) is strictly separated from Audited Test (871 images).
- Never bundles original Detection Validation (2,316 images, of which 506 exact SHA-256 and 28 pHash images
  overlap with Train and 37 SHA-256 + 2 pHash images are internal duplicates) into training packages.
- Converts COCO annotations to YOLO txt format from authoritative `annotations_coco.json` and verifies
  every bounding box (class IDs 0..4, finite coordinates, positive width/height, exact box count).
- Uses portable POSIX relative paths in all saved manifests so Google Colab Linux runtimes never encounter
  Windows `D:\\...` paths.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DG38_DIR = Path(__file__).resolve().parent
MANIFEST_DIR = DG38_DIR / "manifests"

DEFAULT_CLS_ROOT = Path(os.environ.get("DG38_LOCAL_CLS_ROOT", r"D:\Download Acer\Garbage Dataset\standardized_256"))
DEFAULT_DET_ROOT = PROJECT_ROOT / "WasteDetect_Dataset"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

EXPECTED_CLS_COUNTS = {"cls_train": 9807, "cls_val": 1226, "cls_test": 1226, "total": 12259}
EXPECTED_CLS_SPLIT_SIGNATURE = "1da7e63c9ee27d1831b3595350460321897565b0375ef33da0aea1ec0e5660c7"
EXPECTED_DET_COUNTS = {
    "train": {
        "images": 8289,
        "boxes": 18345,
        "coco_sha256": "a19eec883ccd8bfd76248577275ae15b01aa1c14878507fe2cda59fc4226c2eb",
    },
    "valid_clean_candidate": {
        "images": 1743,
        "boxes": 3932,
        "coco_sha256": "9faa0cc8aade9802f63905c7efe9a8c10f7118d866b186d46f2fd808c3464de8",
    },
    "test_final_candidate": {
        "images": 871,
        "boxes": 1884,
        "coco_sha256": "e26ff743f2a94ef9a6069a95cf3c3c69581e43295ff48bb5298b301bfde3eec1",
    },
}
EXPECTED_CHECKPOINTS = {
    "checkpoints/dinov2_garbage_final.pth": "bee29bc7364fa18d374e68bee7f435282917fd3d98b2a83dbca7dffe53e3ec75",
    "detection/checkpoints_wastedetect/best_model.pt": "218aa21101d27ffffe43b827300c8769e168a9d8998b792f1dacbb6548a49556",
}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json_obj(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def reconstruct_classification_manifest(
    cls_root: str | Path = DEFAULT_CLS_ROOT,
    compute_file_hashes: bool = True,
    existing_full_manifest_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Reconstruct exact 9,807 / 1,226 / 1,226 split from `Copy_of_Trainning_DINOv2.ipynb` (Cell 14-15)."""
    # Reuse cached per-file SHA-256 hashes if an already-verified full manifest exists on disk
    cached_hashes: Dict[str, str] = {}
    if existing_full_manifest_path and existing_full_manifest_path.exists():
        try:
            old_data = json.loads(existing_full_manifest_path.read_text(encoding="utf-8"))
            if old_data.get("split_signature_sha256") == EXPECTED_CLS_SPLIT_SIGNATURE:
                for sp_rows in old_data.get("splits", {}).values():
                    for r in sp_rows:
                        if "rel_path" in r and "sha256" in r:
                            cached_hashes[r["rel_path"]] = r["sha256"]
        except Exception:
            cached_hashes = {}

    root = Path(cls_root)
    if not root.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục Classification dataset: {root}")

    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    if len(classes) != 10:
        raise ValueError(f"Classification dataset phải có đúng 10 lớp, tìm thấy {len(classes)}: {classes}")
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    samples: List[Tuple[Path, int, str, str]] = []
    for cls_name in classes:
        cls_idx = class_to_idx[cls_name]
        cls_dir = root / cls_name
        for fname in sorted(os.listdir(cls_dir)):
            fpath = cls_dir / fname
            if fpath.is_file() and fpath.suffix.lower() in IMAGE_EXTS:
                rel_path = f"{cls_name}/{fname}"
                samples.append((fpath, cls_idx, cls_name, rel_path))

    if len(samples) != EXPECTED_CLS_COUNTS["total"]:
        raise ValueError(
            f"Tổng số ảnh trong {root} là {len(samples)}, khác với {EXPECTED_CLS_COUNTS['total']} của notebook gốc."
        )

    targets = np.array([s[1] for s in samples])
    indices = np.arange(len(samples))

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=42,
        stratify=targets,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=42,
        stratify=targets[temp_idx],
    )

    if (
        len(train_idx) != EXPECTED_CLS_COUNTS["cls_train"]
        or len(val_idx) != EXPECTED_CLS_COUNTS["cls_val"]
        or len(test_idx) != EXPECTED_CLS_COUNTS["cls_test"]
    ):
        raise RuntimeError(
            f"Split counts không khớp: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
        )

    def build_entries(idx_array: np.ndarray, split_name: str) -> List[Dict[str, Any]]:
        rows = []
        for global_i in idx_array.tolist():
            fpath, cls_idx, cls_name, rel_path = samples[int(global_i)]
            item: Dict[str, Any] = {
                "global_index": int(global_i),
                "split": split_name,
                "class_name": cls_name,
                "class_index": int(cls_idx),
                "rel_path": rel_path,
            }
            if compute_file_hashes:
                item["sha256"] = cached_hashes.get(rel_path) or sha256_file(fpath)
            rows.append(item)
        return rows

    train_entries = build_entries(train_idx, "cls_train")
    val_entries = build_entries(val_idx, "cls_val")
    test_entries = build_entries(test_idx, "cls_test")

    per_class_counts: Dict[str, Dict[str, int]] = {
        c: {"cls_train": 0, "cls_val": 0, "cls_test": 0, "total": 0} for c in classes
    }
    for e in train_entries:
        per_class_counts[e["class_name"]]["cls_train"] += 1
        per_class_counts[e["class_name"]]["total"] += 1
    for e in val_entries:
        per_class_counts[e["class_name"]]["cls_val"] += 1
        per_class_counts[e["class_name"]]["total"] += 1
    for e in test_entries:
        per_class_counts[e["class_name"]]["cls_test"] += 1
        per_class_counts[e["class_name"]]["total"] += 1

    split_signature = sha256_json_obj(
        {
            "classes": classes,
            "train_rel_paths": [e["rel_path"] for e in train_entries],
            "val_rel_paths": [e["rel_path"] for e in val_entries],
            "test_rel_paths": [e["rel_path"] for e in test_entries],
        }
    )

    return {
        "schema_version": 1,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset_name": "sumn2u/garbage-classification-v2 (standardized_256)",
        "source_root": "standardized_256",
        "split_protocol": "ImageFolder + train_test_split(test_size=0.20, random_state=42, stratify=targets) + train_test_split(test_size=0.50, random_state=42, stratify=targets[temp_idx])",
        "classes": classes,
        "class_to_idx": class_to_idx,
        "counts": {
            "cls_train": len(train_entries),
            "cls_val": len(val_entries),
            "cls_test": len(test_entries),
            "total": len(samples),
        },
        "per_class_counts": per_class_counts,
        "split_signature_sha256": split_signature,
        "splits": {
            "cls_train": train_entries,
            "cls_val": val_entries,
            "cls_test": test_entries,
        },
    }


def convert_coco_to_yolo_verified(
    coco_json_path: str | Path,
    images_dir: str | Path,
    out_labels_dir: Optional[str | Path] = None,
    write_files: bool = True,
    portable_coco_relpath: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert COCO annotations (`0..4`) to YOLO txt labels and strictly verify every bounding box."""
    coco_path = Path(coco_json_path)
    img_root = Path(images_dir)
    coco = json.loads(coco_path.read_text(encoding="utf-8-sig"))

    categories = {int(c["id"]): c["name"] for c in coco.get("categories", [])}
    if sorted(categories.keys()) != [0, 1, 2, 3, 4]:
        raise ValueError(f"COCO categories phải là 0..4, nhận được: {categories}")

    images_by_id = {int(im["id"]): im for im in coco.get("images", [])}
    anns_by_img: Dict[int, List[Dict[str, Any]]] = {img_id: [] for img_id in images_by_id}
    for ann in coco.get("annotations", []):
        img_id = int(ann["image_id"])
        if img_id not in anns_by_img:
            raise ValueError(f"Annotation {ann.get('id')} tham chiếu image_id không tồn tại: {img_id}")
        anns_by_img[img_id].append(ann)

    if write_files and out_labels_dir is not None:
        out_dir = Path(out_labels_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = None

    converted_box_count = 0
    clamped_boundary_count = 0
    per_class_boxes = {cid: 0 for cid in range(5)}
    files_written = 0

    for img_id, im in images_by_id.items():
        fname = im["file_name"]
        img_file = img_root / fname
        if not img_file.is_file():
            raise FileNotFoundError(f"Thiếu file ảnh Detection: {img_file}")
        W = float(im["width"])
        H = float(im["height"])
        if W <= 0 or H <= 0:
            raise ValueError(f"Kích thước ảnh không hợp lệ trong COCO: {fname} ({W}x{H})")

        lines: List[str] = []
        for ann in anns_by_img[img_id]:
            cid = int(ann["category_id"])
            if cid not in per_class_boxes:
                raise ValueError(f"category_id không hợp lệ: {cid}")
            x, y, w, h = [float(v) for v in ann["bbox"]]
            if not all(np.isfinite([x, y, w, h])) or w <= 0 or h <= 0:
                raise ValueError(f"Bbox không hợp lệ tại ann_id={ann.get('id')}: {ann['bbox']}")

            x1 = max(0.0, min(W, x))
            y1 = max(0.0, min(H, y))
            x2 = max(0.0, min(W, x + w))
            y2 = max(0.0, min(H, y + h))
            if abs(x1 - x) > 1e-4 or abs(y1 - y) > 1e-4 or abs(x2 - (x + w)) > 1e-4 or abs(y2 - (y + h)) > 1e-4:
                clamped_boundary_count += 1
            bw = x2 - x1
            bh = y2 - y1
            if bw <= 0 or bh <= 0:
                raise ValueError(f"Bbox suy biến sau kiểm tra biên tại ann_id={ann.get('id')}")

            cx = (x1 + bw / 2.0) / W
            cy = (y1 + bh / 2.0) / H
            nw = bw / W
            nh = bh / H
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < nw <= 1.0 and 0.0 < nh <= 1.0):
                raise ValueError(f"Tọa độ YOLO ngoài miền [0,1]: {cx}, {cy}, {nw}, {nh}")

            lines.append(f"{cid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            converted_box_count += 1
            per_class_boxes[cid] += 1

        if out_dir is not None:
            txt_name = Path(fname).with_suffix(".txt").name
            (out_dir / txt_name).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            files_written += 1

    return {
        "coco_json": portable_coco_relpath or f"WasteDetect_Dataset/{coco_path.parent.name}/annotations_coco.json",
        "coco_sha256": sha256_file(coco_path),
        "images_count": len(images_by_id),
        "converted_box_count": converted_box_count,
        "clamped_boundary_count": clamped_boundary_count,
        "per_class_boxes": {categories[k]: v for k, v in per_class_boxes.items()},
        "categories": categories,
        "labels_dir": str(out_dir) if out_dir else None,
        "label_files_written": files_written,
    }


def verify_detection_splits(det_root: str | Path = DEFAULT_DET_ROOT) -> Dict[str, Any]:
    """Verify Train, Clean Validation (`valid_clean_candidate`), and Final Test (`test_final_candidate`)."""
    root = Path(det_root)
    report: Dict[str, Any] = {
        "schema_version": 1,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "detection_root": "WasteDetect_Dataset",
        "policy_note": (
            "Strictly uses 'train' (8,289 images / 18,345 boxes) and 'valid_clean_candidate' (1,743 images / 3,932 boxes) "
            "for all training/validation. Original 'valid' (2,316 total images / 5,110 boxes) is excluded because audit "
            "confirmed 573 images to remove (506 exact SHA-256 matches with Train, 37 internal SHA-256 duplicates within valid, "
            "28 pHash near-duplicates with Train, and 2 internal pHash duplicates within valid, plus 1 duplicate box annotation_1847)."
        ),
        "original_valid_audit_breakdown": {
            "original_valid_total_images": 2316,
            "original_valid_total_boxes": 5110,
            "stage1_exact_sha256_with_train_removed": 506,
            "stage1_exact_sha256_within_valid_removed": 37,
            "stage1_sha_clean_retained_images": 1773,
            "stage2_phash_confirmed_with_train_removed": 28,
            "stage2_phash_confirmed_within_valid_removed": 2,
            "total_excluded_images": 573,
            "valid_clean_candidate_retained_images": 1743,
            "duplicate_geometry_annotation_1847_removed": 1,
            "valid_clean_candidate_retained_boxes": 3932,
        },
        "splits": {},
    }

    for split_name, expected in EXPECTED_DET_COUNTS.items():
        split_dir = root / split_name
        coco_path = split_dir / "annotations_coco.json"
        images_dir = split_dir / "images"
        if not coco_path.exists():
            raise FileNotFoundError(f"Không tìm thấy {coco_path}")
        actual_sha = sha256_file(coco_path)
        if actual_sha != expected["coco_sha256"]:
            raise ValueError(
                f"SHA-256 của {coco_path} ({actual_sha}) khác với bản kiểm toán ({expected['coco_sha256']})!"
            )

        conv = convert_coco_to_yolo_verified(
            coco_path,
            images_dir,
            out_labels_dir=None,
            write_files=False,
            portable_coco_relpath=f"WasteDetect_Dataset/{split_name}/annotations_coco.json",
        )
        if conv["images_count"] != expected["images"] or conv["converted_box_count"] != expected["boxes"]:
            raise ValueError(
                f"Số lượng ảnh/box của {split_name} không khớp: {conv['images_count']} ảnh / {conv['converted_box_count']} boxes"
            )
        report["splits"][split_name] = conv

    return report


def save_all_manifests(
    cls_root: str | Path = DEFAULT_CLS_ROOT,
    det_root: str | Path = DEFAULT_DET_ROOT,
    manifest_dir: str | Path = MANIFEST_DIR,
    compute_cls_file_hashes: bool = True,
) -> Dict[str, Path]:
    """Generate and save separated Train/Val vs Test manifests with portable relative paths."""
    out_dir = Path(manifest_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p_cls_tv = out_dir / "classification_train_val_manifest.json"
    p_cls_test = out_dir / "classification_test_manifest.json"
    p_cls_full = out_dir / "classification_full_manifest.json"
    p_det = out_dir / "detection_splits_manifest.json"

    cls_full = reconstruct_classification_manifest(
        cls_root,
        compute_file_hashes=compute_cls_file_hashes,
        existing_full_manifest_path=p_cls_full,
    )
    det_summary = verify_detection_splits(det_root)

    cls_train_val = {k: v for k, v in cls_full.items() if k != "splits"}
    cls_train_val["purpose"] = "TRAINING_AND_VALIDATION_ONLY (cls_test excluded by design)"
    cls_train_val["splits"] = {
        "cls_train": cls_full["splits"]["cls_train"],
        "cls_val": cls_full["splits"]["cls_val"],
    }

    cls_test_only = {k: v for k, v in cls_full.items() if k != "splits"}
    cls_test_only["purpose"] = "LOCKED_TEST_ONLY (Never use for model or hyperparameter selection)"
    cls_test_only["splits"] = {
        "cls_test": cls_full["splits"]["cls_test"],
    }

    p_cls_tv.write_text(json.dumps(cls_train_val, indent=2, ensure_ascii=False), encoding="utf-8")
    p_cls_test.write_text(json.dumps(cls_test_only, indent=2, ensure_ascii=False), encoding="utf-8")
    p_cls_full.write_text(json.dumps(cls_full, indent=2, ensure_ascii=False), encoding="utf-8")
    p_det.write_text(json.dumps(det_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    master_index = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "classification_split_signature_sha256": cls_full["split_signature_sha256"],
        "classification_train_val_manifest_sha256": sha256_file(p_cls_tv),
        "classification_test_manifest_sha256": sha256_file(p_cls_test),
        "detection_train_coco_sha256": det_summary["splits"]["train"]["coco_sha256"],
        "detection_valid_clean_coco_sha256": det_summary["splits"]["valid_clean_candidate"]["coco_sha256"],
        "detection_test_final_coco_sha256": det_summary["splits"]["test_final_candidate"]["coco_sha256"],
        "reference_checkpoints_sha256": EXPECTED_CHECKPOINTS,
        "files": {
            "classification_train_val_manifest": "evaluation_runs/dg38_additional_experiments/manifests/classification_train_val_manifest.json",
            "classification_test_manifest": "evaluation_runs/dg38_additional_experiments/manifests/classification_test_manifest.json",
            "classification_full_manifest": "evaluation_runs/dg38_additional_experiments/manifests/classification_full_manifest.json",
            "detection_splits_manifest": "evaluation_runs/dg38_additional_experiments/manifests/detection_splits_manifest.json",
        },
    }
    p_master = out_dir / "dataset_manifests_index.json"
    p_master.write_text(json.dumps(master_index, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "master_index": p_master,
        "cls_train_val": p_cls_tv,
        "cls_test": p_cls_test,
        "cls_full": p_cls_full,
        "det_manifest": p_det,
    }


def _record_zip_info(
    zip_path: Path,
    drive_rel_dest: str,
    colab_unpack_dest: str,
    contains_test: bool,
    description_vi: str,
) -> Dict[str, Any]:
    size_bytes = zip_path.stat().st_size
    return {
        "file_name": zip_path.name,
        "local_path": str(zip_path),
        "sha256": sha256_file(zip_path),
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 ** 2), 2),
        "drive_upload_path": drive_rel_dest,
        "colab_unpack_target": colab_unpack_dest,
        "contains_test": contains_test,
        "description_vi": description_vi,
    }


def build_separated_zip_packages(
    out_zip_dir: str | Path,
    cls_root: str | Path = DEFAULT_CLS_ROOT,
    det_root: str | Path = DEFAULT_DET_ROOT,
    include_source_and_checkpoints: bool = True,
    include_cls_zips: bool = True,
    include_det_zips: bool = True,
    include_crop_ablation_bundle: bool = True,
) -> Dict[str, Any]:
    """Create separated, verified ZIP archives ready for Google Drive upload."""
    zip_dir = Path(out_zip_dir)
    zip_dir.mkdir(parents=True, exist_ok=True)
    manifests = save_all_manifests(cls_root=cls_root, det_root=det_root)
    cls_full = json.loads(manifests["cls_full"].read_text(encoding="utf-8"))

    created_zips: Dict[str, Dict[str, Any]] = {}

    if include_source_and_checkpoints:
        # 1. Source Code, Manifests, Notebooks, and Validation SQLite Cache Bundle
        code_zip = zip_dir / "dg38_source_code_and_manifests.zip"
        code_files = [
            "model.py",
            "train.py",
            "evaluate.py",
            "prepare_split.py",
            "cu_budget_tracker.py",
            "src/model.py",
            "detection/Train.py",
            "detection/dinov2_deformable_detr.py",
            "detection/waste_cascade.py",
            "detection/evaluate_cascade.py",
            "detection/class_mapping.eval.json",
            "evaluation_runs/threshold_cache.py",
            "evaluation_runs/grid_metrics.py",
            "evaluation_runs/tune_validation_thresholds.py",
            "evaluation_runs/val_grid_search_01/locked_parameters.json",
            "evaluation_runs/val_grid_search_01/summary.csv",
            "evaluation_runs/val_grid_search_01/cascade_error_analysis.json",
            "evaluation_runs/val_grid_search_01/cascade_error_analysis.md",
            "evaluation_runs/val_grid_search_01/cache/fingerprint.json",
            "evaluation_runs/val_grid_search_01/cache/predictions.sqlite3",
            "evaluation_runs/val_grid_search_01/data_audit/validation_data_audit.json",
            "WasteDetect_Dataset/valid_clean_candidate/cleaning_report.json",
            "WasteDetect_Dataset/test_final_candidate/cleaning_report.json",
        ]
        with zipfile.ZipFile(code_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for rel_f in code_files:
                src_p = PROJECT_ROOT / rel_f
                if src_p.is_file():
                    zf.write(src_p, arcname=rel_f)
            # Add all files in evaluation_runs/dg38_additional_experiments (except colab_packages and runs)
            for p in sorted(DG38_DIR.rglob("*")):
                if not p.is_file():
                    continue
                rel_to_dg38 = p.relative_to(DG38_DIR).as_posix()
                if (
                    rel_to_dg38.startswith("colab_packages/")
                    or rel_to_dg38.startswith("runs/")
                    or "__pycache__" in rel_to_dg38
                ):
                    continue
                arc = f"evaluation_runs/dg38_additional_experiments/{rel_to_dg38}"
                zf.write(p, arcname=arc)
            # Also include tests/test_dg38_experiments.py
            test_f = PROJECT_ROOT / "tests/test_dg38_experiments.py"
            if test_f.is_file():
                zf.write(test_f, arcname="tests/test_dg38_experiments.py")

        created_zips[code_zip.name] = _record_zip_info(
            code_zip,
            drive_rel_dest="/content/drive/MyDrive/DG38_Colab_Package/zips/dg38_source_code_and_manifests.zip",
            colab_unpack_dest="/content/drive/MyDrive/DG38_Colab_Package/code",
            contains_test=False,
            description_vi="Gói mã nguồn Python, 6 Notebook Colab, các manifest SHA-256 và SQLite cache predictions.sqlite3.",
        )

        # 2. Reference Checkpoints ZIP (preserving exact paths: checkpoints/dinov2_garbage_final.pth & detection/checkpoints_wastedetect/best_model.pt)
        ckpt_zip = zip_dir / "dg38_reference_checkpoints.zip"
        with zipfile.ZipFile(ckpt_zip, "w", compression=zipfile.ZIP_STORED) as zf:
            for rel_ckpt, exp_sha in EXPECTED_CHECKPOINTS.items():
                ckpt_p = PROJECT_ROOT / rel_ckpt
                if not ckpt_p.is_file():
                    raise FileNotFoundError(f"Không tìm thấy checkpoint gốc tại đường dẫn chuẩn: {ckpt_p}")
                actual_sha = sha256_file(ckpt_p)
                if actual_sha != exp_sha:
                    raise ValueError(f"SHA-256 của {rel_ckpt} không khớp ({actual_sha} != {exp_sha})")
                zf.write(ckpt_p, arcname=rel_ckpt)

        created_zips[ckpt_zip.name] = _record_zip_info(
            ckpt_zip,
            drive_rel_dest="/content/drive/MyDrive/DG38_Colab_Package/zips/dg38_reference_checkpoints.zip",
            colab_unpack_dest="/content/drive/MyDrive/DG38_Colab_Package/code",
            contains_test=False,
            description_vi="Hai checkpoint gốc để chạy Crop Ablation & đối chiếu (`checkpoints/dinov2_garbage_final.pth` và `detection/checkpoints_wastedetect/best_model.pt`).",
        )

    if include_cls_zips:
        # 3. Classification Train + Val ZIP (excludes Test!)
        cls_tv_zip = zip_dir / "dg38_cls_train_val.zip"
        train_rel_set = {e["rel_path"] for e in cls_full["splits"]["cls_train"]}
        val_rel_set = {e["rel_path"] for e in cls_full["splits"]["cls_val"]}
        test_rel_set = {e["rel_path"] for e in cls_full["splits"]["cls_test"]}
        assert train_rel_set.isdisjoint(val_rel_set), "Train and Val must be disjoint!"
        assert train_rel_set.isdisjoint(test_rel_set), "Train and Test must be disjoint!"
        assert val_rel_set.isdisjoint(test_rel_set), "Val and Test must be disjoint!"

        with zipfile.ZipFile(cls_tv_zip, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.write(manifests["cls_train_val"], arcname="classification_train_val_manifest.json")
            for split_key in ("cls_train", "cls_val"):
                for item in cls_full["splits"][split_key]:
                    src = Path(cls_root) / item["rel_path"]
                    zf.write(src, arcname=f"standardized_256/{item['rel_path']}")

        created_zips[cls_tv_zip.name] = _record_zip_info(
            cls_tv_zip,
            drive_rel_dest="/content/drive/MyDrive/DG38_Colab_Package/zips/dg38_cls_train_val.zip",
            colab_unpack_dest="/content/dg38_data/classification",
            contains_test=False,
            description_vi="Tập Classification Train (9.807 ảnh) + Validation (1.226 ảnh) = 11.033 ảnh + manifest. Tuyệt đối không chứa ảnh Test.",
        )

        # 4. Classification Locked Test ZIP
        cls_test_zip = zip_dir / "dg38_cls_test_locked.zip"
        with zipfile.ZipFile(cls_test_zip, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.write(manifests["cls_test"], arcname="classification_test_manifest.json")
            for item in cls_full["splits"]["cls_test"]:
                src = Path(cls_root) / item["rel_path"]
                zf.write(src, arcname=f"standardized_256/{item['rel_path']}")

        created_zips[cls_test_zip.name] = _record_zip_info(
            cls_test_zip,
            drive_rel_dest="/content/drive/MyDrive/DG38_Colab_Package/zips/dg38_cls_test_locked.zip",
            colab_unpack_dest="/content/dg38_data/classification_test_locked",
            contains_test=True,
            description_vi="Tập Classification Test đã khóa (1.226 ảnh). Chỉ giải nén khi chạy đánh giá hậu nghiệm cuối cùng.",
        )

    if include_det_zips:
        # 5. Detection Train + Clean Validation ZIP (excludes dirty valid and excludes Test!)
        det_tv_zip = zip_dir / "dg38_det_train_val_clean.zip"
        with zipfile.ZipFile(det_tv_zip, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.write(manifests["det_manifest"], arcname="detection_splits_manifest.json")
            for split_name in ("train", "valid_clean_candidate"):
                s_dir = Path(det_root) / split_name
                zf.write(
                    s_dir / "annotations_coco.json",
                    arcname=f"WasteDetect_Dataset/{split_name}/annotations_coco.json",
                )
                img_dir = s_dir / "images"
                for img_p in sorted(img_dir.iterdir()):
                    if img_p.is_file() and img_p.suffix.lower() in IMAGE_EXTS:
                        zf.write(img_p, arcname=f"WasteDetect_Dataset/{split_name}/images/{img_p.name}")

        created_zips[det_tv_zip.name] = _record_zip_info(
            det_tv_zip,
            drive_rel_dest="/content/drive/MyDrive/DG38_Colab_Package/zips/dg38_det_train_val_clean.zip",
            colab_unpack_dest="/content/dg38_data/detection",
            contains_test=False,
            description_vi="Tập Detection Train (8.289 ảnh / 18.345 boxes) + Validation sạch valid_clean_candidate (1.743 ảnh / 3.932 boxes). Không chứa valid cũ và không chứa Test.",
        )

        # 6. Detection Locked Final Test ZIP (871 audited images / 1,884 boxes)
        det_test_zip = zip_dir / "dg38_det_test_locked.zip"
        test_s_dir = Path(det_root) / "test_final_candidate"
        with zipfile.ZipFile(det_test_zip, "w", compression=zipfile.ZIP_STORED) as zf:
            for meta_name in (
                "annotations_coco.json",
                "test_manifest.json",
                "image_manifest.json",
                "cleaning_report.json",
            ):
                mp = test_s_dir / meta_name
                if mp.is_file():
                    zf.write(mp, arcname=f"WasteDetect_Dataset/test_final_candidate/{meta_name}")
            img_dir = test_s_dir / "images"
            for img_p in sorted(img_dir.iterdir()):
                if img_p.is_file() and img_p.suffix.lower() in IMAGE_EXTS:
                    zf.write(img_p, arcname=f"WasteDetect_Dataset/test_final_candidate/images/{img_p.name}")

        created_zips[det_test_zip.name] = _record_zip_info(
            det_test_zip,
            drive_rel_dest="/content/drive/MyDrive/DG38_Colab_Package/zips/dg38_det_test_locked.zip",
            colab_unpack_dest="/content/dg38_data/detection_test_locked",
            contains_test=True,
            description_vi="Tập Detection Test đã kiểm toán test_final_candidate (871 ảnh / 1.884 boxes). Tách biệt hoàn toàn khỏi gói Train/Val.",
        )

    if include_crop_ablation_bundle:
        # 7. Lightweight bundle for Stage 7 Crop Preprocessing Ablation (Clean Val + SQLite Detector Cache)
        crop_zip = zip_dir / "dg38_crop_ablation_val_bundle.zip"
        val_cache_sqlite = PROJECT_ROOT / "evaluation_runs/val_grid_search_01/cache/predictions.sqlite3"
        val_cache_fp = PROJECT_ROOT / "evaluation_runs/val_grid_search_01/cache/fingerprint.json"
        locked_params = PROJECT_ROOT / "evaluation_runs/val_grid_search_01/locked_parameters.json"
        class_map = PROJECT_ROOT / "detection/class_mapping.eval.json"
        val_audit = PROJECT_ROOT / "evaluation_runs/val_grid_search_01/data_audit/validation_data_audit.json"

        with zipfile.ZipFile(crop_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p, arc in [
                (val_cache_sqlite, "val_cache/predictions.sqlite3"),
                (val_cache_fp, "val_cache/fingerprint.json"),
                (locked_params, "val_cache/locked_parameters.json"),
                (class_map, "detection/class_mapping.eval.json"),
                (val_audit, "val_cache/validation_data_audit.json"),
                (
                    Path(det_root) / "valid_clean_candidate/annotations_coco.json",
                    "WasteDetect_Dataset/valid_clean_candidate/annotations_coco.json",
                ),
            ]:
                if p.exists():
                    zf.write(p, arcname=arc)
            img_dir = Path(det_root) / "valid_clean_candidate/images"
            for img_p in sorted(img_dir.iterdir()):
                if img_p.is_file() and img_p.suffix.lower() in IMAGE_EXTS:
                    zf.write(img_p, arcname=f"WasteDetect_Dataset/valid_clean_candidate/images/{img_p.name}")

        created_zips[crop_zip.name] = _record_zip_info(
            crop_zip,
            drive_rel_dest="/content/drive/MyDrive/DG38_Colab_Package/zips/dg38_crop_ablation_val_bundle.zip",
            colab_unpack_dest="/content/dg38_data/crop_ablation",
            contains_test=False,
            description_vi="Gói độc lập siêu nhẹ cho Giai đoạn 7 Crop Ablation (Clean Val 1.743 ảnh + SQLite cache predictions.sqlite3).",
        )

    manifest_out = zip_dir / "zip_packages_sha256.json"
    manifest_out.write_text(json.dumps(created_zips, indent=2, ensure_ascii=False), encoding="utf-8")
    return created_zips


def verify_zip_archive_structure(zip_dir: str | Path) -> Dict[str, Any]:
    """Verify every created ZIP archive can be opened, passes CRC testzip(), and matches expected structure."""
    zdir = Path(zip_dir)
    meta_path = zdir / "zip_packages_sha256.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {meta_path}")
    packages = json.loads(meta_path.read_text(encoding="utf-8"))
    verification: Dict[str, Any] = {}

    for zname, info in packages.items():
        zpath = zdir / zname
        if not zpath.is_file():
            raise FileNotFoundError(f"Thiếu file ZIP: {zpath}")
        with zipfile.ZipFile(zpath, "r") as zf:
            bad_member = zf.testzip()
            if bad_member is not None:
                raise RuntimeError(f"File ZIP {zname} bị hỏng CRC tại thành phần: {bad_member}")
            names = zf.namelist()

        if zname == "dg38_cls_train_val.zip":
            img_files = [n for n in names if n.startswith("standardized_256/") and not n.endswith("/")]
            assert "classification_train_val_manifest.json" in names
            assert len(img_files) == 11033, f"Expected 11,033 train+val images, got {len(img_files)}"
        elif zname == "dg38_cls_test_locked.zip":
            img_files = [n for n in names if n.startswith("standardized_256/") and not n.endswith("/")]
            assert "classification_test_manifest.json" in names
            assert len(img_files) == 1226, f"Expected 1,226 test images, got {len(img_files)}"
        elif zname == "dg38_det_train_val_clean.zip":
            train_imgs = [n for n in names if n.startswith("WasteDetect_Dataset/train/images/") and not n.endswith("/")]
            val_imgs = [
                n for n in names if n.startswith("WasteDetect_Dataset/valid_clean_candidate/images/") and not n.endswith("/")
            ]
            dirty_val = [n for n in names if n.startswith("WasteDetect_Dataset/valid/")]
            test_leak = [n for n in names if "test" in n.lower()]
            assert len(train_imgs) == 8289, f"Expected 8,289 det train images, got {len(train_imgs)}"
            assert len(val_imgs) == 1743, f"Expected 1,743 clean val images, got {len(val_imgs)}"
            assert len(dirty_val) == 0, "Dirty valid must not be included!"
            assert len(test_leak) == 0, f"Test files leaked into train/val zip: {test_leak}"
        elif zname == "dg38_det_test_locked.zip":
            test_imgs = [
                n for n in names if n.startswith("WasteDetect_Dataset/test_final_candidate/images/") and not n.endswith("/")
            ]
            assert len(test_imgs) == 871, f"Expected 871 audited test images, got {len(test_imgs)}"
        elif zname == "dg38_reference_checkpoints.zip":
            assert "checkpoints/dinov2_garbage_final.pth" in names
            assert "detection/checkpoints_wastedetect/best_model.pt" in names
        elif zname == "dg38_source_code_and_manifests.zip":
            assert "evaluation_runs/val_grid_search_01/cache/predictions.sqlite3" in names
            assert "evaluation_runs/dg38_additional_experiments/00_DG38_Preflight_Colab.ipynb" in names

        verification[zname] = {
            "crc_ok": True,
            "entry_count": len(names),
            "size_mb": info["size_mb"],
            "sha256": info["sha256"],
        }

    return verification


def main() -> None:
    parser = argparse.ArgumentParser(description="Tạo SHA-256 manifests và gói dữ liệu ZIP tách biệt cho Colab.")
    parser.add_argument("--mode", choices=["manifest-only", "build-zips", "verify-zips"], default="manifest-only")
    parser.add_argument("--cls-root", default=str(DEFAULT_CLS_ROOT))
    parser.add_argument("--det-root", default=str(DEFAULT_DET_ROOT))
    parser.add_argument(
        "--out-zip-dir",
        "--output-dir",
        dest="out_zip_dir",
        default=str(DG38_DIR / "colab_packages"),
    )
    parser.add_argument(
        "--skip-cls-file-sha",
        action="store_true",
        help="Bỏ qua băm từng file ảnh Classification nếu chỉ cần tạo nhanh index",
    )
    args = parser.parse_args()

    if args.mode == "manifest-only":
        paths = save_all_manifests(
            cls_root=args.cls_root,
            det_root=args.det_root,
            compute_cls_file_hashes=not args.skip_cls_file_sha,
        )
        index_data = json.loads(paths["master_index"].read_text(encoding="utf-8"))
        print(json.dumps(index_data, indent=2, ensure_ascii=False))
    elif args.mode == "build-zips":
        res = build_separated_zip_packages(
            out_zip_dir=args.out_zip_dir,
            cls_root=args.cls_root,
            det_root=args.det_root,
        )
        ver = verify_zip_archive_structure(args.out_zip_dir)
        print(json.dumps({"packages": res, "verification": ver}, indent=2, ensure_ascii=False))
    elif args.mode == "verify-zips":
        ver = verify_zip_archive_structure(args.out_zip_dir)
        print(json.dumps(ver, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
