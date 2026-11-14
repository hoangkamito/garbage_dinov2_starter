#!/usr/bin/env python3
"""Build DG38 locked-Test ZIPs from original local datasets and existing locked manifests.

READ-ONLY for all original images/manifests. Does NOT run inference, train models,
resplit data, or open a validation-based model selection loop.

Usage examples:
  python dg38_recover_locked_test.py --only classification \
    --cls-root "D:/Download Acer/Garbage Dataset/standardized_256" \
    --cls-manifest "D:/.../classification_test_manifest.json" \
    --out-dir "D:/DG38_test_export"

  python dg38_recover_locked_test.py --only detection \
    --det-test-root "D:/.../WasteDetect_Dataset/test_final_candidate" \
    --out-dir "D:/DG38_test_export"
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import tempfile
import zipfile
from datetime import datetime, timezone

EXPECTED_CLASS_SIGNATURE = "1da7e63c9ee27d1831b3595350460321897565b0375ef33da0aea1ec0e5660c7"
EXPECTED_DET_COCO_SHA = "e26ff743f2a94ef9a6069a95cf3c3c69581e43295ff48bb5298b301bfde3eec1"
EXPECTED_CLS_TEST = 1226
EXPECTED_DET_IMAGES = 871
EXPECTED_DET_BOXES = 1884


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_child(root: Path, rel: str) -> Path:
    # Reject absolute paths, parent traversal, Windows separators in portable entries.
    if not isinstance(rel, str) or not rel or "\\" in rel:
        raise ValueError(f"Unsafe relative path: {rel!r}")
    p = PurePosixPath(rel)
    if p.is_absolute() or any(part in (".", "..", "") for part in p.parts):
        raise ValueError(f"Unsafe relative path: {rel!r}")
    full = root.joinpath(*p.parts)
    if not full.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes original dataset root: {rel!r}")
    return full


def atomic_zip(out: Path, content: list[tuple[Path, str]]) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(f"Archive already exists, not overwriting: {out}")
    tmp = out.with_name(out.name + ".partial")
    if tmp.exists():
        raise FileExistsError(f"Stale partial archive: {tmp}. Inspect/remove it before retrying.")
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
            for src, arcname in content:
                if not src.is_file():
                    raise FileNotFoundError(src)
                z.write(src, arcname)
        with zipfile.ZipFile(tmp, "r") as z:
            bad = z.testzip()
            if bad:
                raise ValueError(f"CRC failure in ZIP member: {bad}")
            expected_names = [arcname for _, arcname in content]
            if z.namelist() != expected_names:
                raise ValueError("ZIP members differ from manifest")
        tmp.rename(out)
        return {"filename": out.name, "path": str(out), "members": len(content),
                "size_bytes": out.stat().st_size, "sha256": sha256_file(out), "zip_crc_ok": True}
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise


def pack_cls(cls_root: Path, manifest_file: Path, out_dir: Path) -> dict:
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Missing locked Classification manifest: {manifest_file}")
    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    if data.get("split_signature_sha256") != EXPECTED_CLASS_SIGNATURE:
        raise ValueError("Classification split signature changed. STOP. Do not resplit.")
    entries = data.get("splits", {}).get("cls_test")
    if not isinstance(entries, list) or len(entries) != EXPECTED_CLS_TEST:
        raise ValueError(f"Expected {EXPECTED_CLS_TEST} cls_test entries, got {len(entries) if isinstance(entries, list) else None}")
    if data.get("counts", {}).get("cls_test") != EXPECTED_CLS_TEST:
        raise ValueError("Classification test manifest counts mismatch")
    seen = set()
    files = [(manifest_file, "classification_test_manifest.json")]
    for i, entry in enumerate(entries, 1):
        if entry.get("split") != "cls_test":
            raise ValueError(f"Entry {i} not cls_test")
        rel = entry["rel_path"]
        if rel in seen:
            raise ValueError(f"Duplicate classification Test path: {rel}")
        seen.add(rel)
        path = safe_child(cls_root, rel)
        if not path.is_file():
            raise FileNotFoundError(f"Classification Test image missing: {path}")
        expected_sha = entry.get("sha256")
        if not expected_sha or len(expected_sha) != 64:
            raise ValueError(f"Missing per-file SHA-256 for {rel}. Do not package unverified Test.")
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise ValueError(f"Classification Test image hash mismatch: {rel} (expected {expected_sha}, got {actual_sha})")
        files.append((path, f"standardized_256/{rel}"))
        if i % 200 == 0:
            print(f"[CLASSIFICATION] SHA-256 verified {i}/{len(entries)} images", flush=True)
    # If Train/Val manifest present locally, verify disjointness as an additional guard.
    tv = manifest_file.with_name("classification_train_val_manifest.json")
    if tv.is_file():
        tv_data = json.loads(tv.read_text(encoding="utf-8"))
        if tv_data.get("split_signature_sha256") != EXPECTED_CLASS_SIGNATURE:
            raise ValueError("Train/Val manifest split signature mismatch")
        for split in ("cls_train", "cls_val"):
            overlap = seen.intersection(e["rel_path"] for e in tv_data["splits"][split])
            if overlap:
                raise ValueError(f"Test overlaps {split}: {list(overlap)[:3]}")
        print("[CLASSIFICATION] Test disjoint from Train/Val: VERIFIED")
    else:
        print("[NOTE] Train/Val manifest not found locally; Test membership verified against locked Test manifest.")
    result = atomic_zip(out_dir / "dg38_cls_test_locked.zip", files)
    result.update({"split": "cls_test", "test_images": EXPECTED_CLS_TEST,
                   "split_signature_sha256": EXPECTED_CLASS_SIGNATURE,
                   "manifest_sha256": sha256_file(manifest_file), "per_image_sha256_verified": True})
    return result


def pack_det(det_test_root: Path, out_dir: Path) -> dict:
    coco = det_test_root / "annotations_coco.json"
    images_dir = det_test_root / "images"
    if not coco.is_file():
        raise FileNotFoundError(f"Missing audited Detection Test COCO: {coco}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing audited Detection Test images: {images_dir}")
    actual_sha = sha256_file(coco)
    if actual_sha != EXPECTED_DET_COCO_SHA:
        raise ValueError(f"Detection Test COCO SHA-256 mismatch: {actual_sha}. STOP. Original 'test' is not the audited 'test_final_candidate'.")
    data = json.loads(coco.read_text(encoding="utf-8"))
    if len(data.get("images", [])) != EXPECTED_DET_IMAGES or len(data.get("annotations", [])) != EXPECTED_DET_BOXES:
        raise ValueError("Detection Test image or GT box count mismatch")
    image_ids = [int(x["id"]) for x in data["images"]]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("Detection Test image IDs not unique")
    if not all(int(a["image_id"]) in set(image_ids) for a in data["annotations"]):
        raise ValueError("Some Test annotations point to unknown images")
    base_arc = "WasteDetect_Dataset/test_final_candidate"
    files = [(coco, f"{base_arc}/annotations_coco.json")]
    for optional in ("test_manifest.json", "image_manifest.json", "cleaning_report.json"):
        src = det_test_root / optional
        if src.is_file():
            files.append((src, f"{base_arc}/{optional}"))
    seen = set()
    for im in data["images"]:
        rel = im["file_name"]
        if rel in seen:
            raise ValueError(f"Duplicate Detection Test filename: {rel}")
        seen.add(rel)
        p = safe_child(images_dir, rel)
        if not p.is_file():
            raise FileNotFoundError(f"Detection Test image missing: {p}")
        files.append((p, f"{base_arc}/images/{rel}"))
    result = atomic_zip(out_dir / "dg38_det_test_locked.zip", files)
    result.update({"split": "test_final_candidate", "test_images": EXPECTED_DET_IMAGES,
                   "test_gt_boxes": EXPECTED_DET_BOXES, "coco_sha256": EXPECTED_DET_COCO_SHA,
                   "membership_verified_against_locked_coco": True})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Recover DG38 locked Test packages from original datasets (NO inference)")
    parser.add_argument("--only", choices=("classification", "detection", "both"), default="both")
    parser.add_argument("--cls-root", type=Path)
    parser.add_argument("--cls-manifest", type=Path)
    parser.add_argument("--det-test-root", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "no_training": True,
              "no_test_inference": True, "no_resplitting": True, "packages": {}}
    if args.only in ("classification", "both"):
        if args.cls_root is None or args.cls_manifest is None:
            parser.error("Classification requires --cls-root and --cls-manifest")
        result["packages"]["classification"] = pack_cls(args.cls_root, args.cls_manifest, args.out_dir)
        print("[OK] Classification Test package verified and created", flush=True)
    if args.only in ("detection", "both"):
        if args.det_test_root is None:
            parser.error("Detection requires --det-test-root")
        result["packages"]["detection"] = pack_det(args.det_test_root, args.out_dir)
        print("[OK] Detection Test package verified and created", flush=True)
    report = args.out_dir / f"locked_test_packaging_report_{args.only}.json"
    if report.exists():
        raise FileExistsError(f"Not overwriting existing packaging report: {report}")
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n======== DG38 LOCKED TEST PACKAGING ========")
    for group, package in result["packages"].items():
        print(f"{group}: {package['filename']} | images={package['test_images']} | crc={package['zip_crc_ok']}")
        print(f"SHA-256 ZIP: {package['sha256']}")
    print("Packaging report:", report)
    print("NO MODEL TRAINING / NO TEST INFERENCE / ORIGINAL INPUTS UNCHANGED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
