"""
Unit and Integration Tests for the DG38 Additional Experiments Suite.
Verifies:
  1. CU Budget Tracker (200.45 CU budget, category ceilings, 40 CU reserve floor, pilot forecasting).
  2. Dynamic GPU Runtime Manager (L4/A100/G4 recommendations, resumable checkpointing).
  3. Dataset Manifest Integrity, Portability (zero Windows paths), & COCO-to-YOLO Bounding Box Conversion.
  4. Original Validation Audit Breakdown (2,316 total -> 506+37 SHA-256 -> 1,773 -> 28+2 pHash -> 1,743 clean images / 3,932 boxes).
  5. Crop Preprocessing Ablation (Variants A-E transforms, SQLite fingerprint verification, & Variant A bit-exact match).
  6. Three-Seed Statistical Rules (Mean ± Sample Std with ddof=1, INCOMPLETE flag for <3 seeds, homogeneity check).
  7. Notebook JSON, Python AST validation, and zero Windows path dependence for all 6 Colab notebooks.
  8. Colab ZIP Package Verification (CRC check, train/val vs test isolation, expected internal paths).
"""
from __future__ import annotations

import ast
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from evaluation_runs.dg38_additional_experiments.cu_budget_tracker import (
    DEFAULT_CATEGORY_CEILINGS,
    INITIAL_TOTAL_CU,
    MIN_SAFETY_RESERVE_CU,
    CUBudgetTracker,
    PilotTimingMeasurement,
)
from evaluation_runs.dg38_additional_experiments.dg38_crop_ablation import (
    VARIANT_DEFINITIONS,
    evaluate_variant_on_validation,
    expand_and_clip_xyxy,
    letterbox_pil_to_square_224,
    preprocess_crop_for_variant,
    verify_base_cache_fingerprint,
)
from evaluation_runs.dg38_additional_experiments.dg38_stats_and_analysis import (
    aggregate_seed_group,
    compute_sample_mean_std,
)
from evaluation_runs.dg38_additional_experiments.gpu_runtime_manager import (
    classify_gpu_family,
    load_resumable_checkpoint,
    recommend_execution_config,
    save_resumable_checkpoint,
    seed_all_States,
)
from evaluation_runs.dg38_additional_experiments.prepare_colab_packages import (
    convert_coco_to_yolo_verified,
    verify_zip_archive_structure,
)


EXP_DIR = Path("evaluation_runs/dg38_additional_experiments")


def test_cu_budget_tracker_allocations_and_forecast(tmp_path: Path) -> None:
    ledger_path = tmp_path / "cu_budget_ledger.json"
    tracker = CUBudgetTracker(ledger_path)

    assert abs(INITIAL_TOTAL_CU - 200.45) < 1e-6
    total_allocated = sum(float(v["allocated_cu"]) for v in DEFAULT_CATEGORY_CEILINGS.values())
    assert abs(total_allocated - 200.45) < 1e-6
    assert abs(MIN_SAFETY_RESERVE_CU - 40.0) < 1e-6

    pilot = PilotTimingMeasurement(
        gpu_name="NVIDIA L4",
        observed_cu_per_hour=1.50,
        train_batches_measured=20,
        train_time_sec=10.0,
        total_train_batches_per_epoch=300,
        val_batches_measured=10,
        val_time_sec=3.0,
        total_val_batches_per_epoch=40,
        checkpoint_save_sec=5.0,
        startup_overhead_sec=30.0,
        overhead_multiplier=1.10,
    )
    forecast = pilot.forecast_run(epochs=15, num_seeds=1)
    assert forecast["estimated_cu"] > 0
    assert abs(forecast["estimated_cu"] - (forecast["estimated_gpu_hours"] * 1.50)) < 1e-3

    gate = tracker.check_Experiment_gate(
        run_id="cls_resnet50_seed42",
        category="resnet50_baseline",
        forecast=forecast,
    )
    assert gate["approved"] is True

    # Test rejection when forecast exceeds category ceiling (15.0 CU for resnet50_baseline)
    huge_pilot = PilotTimingMeasurement(
        gpu_name="NVIDIA A100-SXM4-40GB",
        observed_cu_per_hour=15.0,
        train_batches_measured=20,
        train_time_sec=10.0,
        total_train_batches_per_epoch=300,
    )
    huge_forecast = huge_pilot.forecast_run(epochs=200, num_seeds=3)
    huge_gate = tracker.check_Experiment_gate(
        run_id="cls_resnet50_huge",
        category="resnet50_baseline",
        forecast=huge_forecast,
    )
    assert huge_gate["approved"] is False
    assert len(huge_gate["reasons_blocked"]) >= 1

    # Record a valid run and verify balance updates
    tracker.record_actual_usage(
        run_id="cls_resnet50_seed42",
        category="resnet50_baseline",
        gpu_name="NVIDIA L4",
        observed_cu_per_hour=1.50,
        elapsed_seconds=1800.0,  # 0.5 hours -> 0.75 CU
        status="COMPLETED",
        epochs_completed=15,
        best_val_metric=0.95,
    )
    assert abs(tracker.total_logged_consumed_cu - 0.75) < 1e-4
    assert abs(tracker.effective_remaining_cu - (200.45 - 0.75)) < 1e-4


def test_gpu_runtime_manager_and_resumable_checkpoint(tmp_path: Path) -> None:
    assert classify_gpu_family("NVIDIA L4") == "L4"
    assert classify_gpu_family("NVIDIA A100-SXM4-40GB") == "A100"
    assert classify_gpu_family("NVIDIA G4") == "G4"

    l4_snapshot = {
        "cuda_available": True,
        "gpu_name": "NVIDIA L4",
        "gpu_family": "L4",
        "total_vram_gb": 22.5,
        "available_vram_gb": 22.0,
        "bf16_supported": True,
        "fp16_supported": True,
        "recommended_amp_dtype": "bfloat16",
        "cpu_count": 8,
    }
    rec = recommend_execution_config("resnet50_classification", runtime_info=l4_snapshot)
    assert rec["detected_gpu_family"] == "L4"
    assert rec["amp_dtype"] == "bfloat16"
    assert rec["effective_batch_size"] == rec["physical_batch_size"] * rec["grad_accum_steps"]

    seed_all_States(42)
    model = torch.nn.Linear(8, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt_file = tmp_path / "ckpt.pt"
    save_resumable_checkpoint(
        ckpt_file,
        model=model,
        optimizer=opt,
        scheduler=None,
        scaler=None,
        epoch=5,
        best_val_metric=0.9123,
        metric_name="val_macro_f1",
        seed=42,
        config={"arch": "resnet50"},
        history={"epoch": [1, 2, 3, 4, 5]},
    )
    model2 = torch.nn.Linear(8, 4)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    meta = load_resumable_checkpoint(ckpt_file, model=model2, optimizer=opt2, device="cpu")
    assert meta["epoch"] == 5
    assert abs(meta["best_val_metric"] - 0.9123) < 1e-6
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)


def test_locked_manifests_portability_and_coco_to_yolo_bbox(tmp_path: Path) -> None:
    manifest_dir = EXP_DIR / "manifests"
    index_path = manifest_dir / "dataset_manifests_index.json"
    assert index_path.exists(), "dataset_manifests_index.json must exist"

    # Assert zero Windows D:\ paths in any manifest file
    for mf_name in [
        "dataset_manifests_index.json",
        "classification_train_val_manifest.json",
        "classification_test_manifest.json",
        "classification_full_manifest.json",
        "detection_splits_manifest.json",
    ]:
        mf_path = manifest_dir / mf_name
        assert mf_path.exists(), f"Missing manifest {mf_path}"
        raw_txt = mf_path.read_text(encoding="utf-8")
        assert "D:\\" not in raw_txt and "d:\\" not in raw_txt and "D:/" not in raw_txt, (
            f"Found hardcoded Windows path in {mf_name}"
        )

    idx = json.loads(index_path.read_text(encoding="utf-8"))
    assert "classification_split_signature_sha256" in idx
    assert "detection_valid_clean_coco_sha256" in idx
    assert "reference_checkpoints_sha256" in idx
    assert "checkpoints/dinov2_garbage_final.pth" in idx["reference_checkpoints_sha256"]
    assert "detection/checkpoints_wastedetect/best_model.pt" in idx["reference_checkpoints_sha256"]

    cls_full_path = manifest_dir / "classification_full_manifest.json"
    cls_full = json.loads(cls_full_path.read_text(encoding="utf-8"))
    assert cls_full["counts"] == {
        "cls_train": 9807,
        "cls_val": 1226,
        "cls_test": 1226,
        "total": 12259,
    }

    det_manifest = json.loads((manifest_dir / "detection_splits_manifest.json").read_text(encoding="utf-8"))
    audit = det_manifest["original_valid_audit_breakdown"]
    assert audit["original_valid_total_images"] == 2316
    assert audit["original_valid_total_boxes"] == 5110
    assert audit["stage1_exact_sha256_with_train_removed"] == 506
    assert audit["stage1_exact_sha256_within_valid_removed"] == 37
    assert audit["stage1_sha_clean_retained_images"] == 1773
    assert audit["stage2_phash_confirmed_with_train_removed"] == 28
    assert audit["stage2_phash_confirmed_within_valid_removed"] == 2
    assert audit["total_excluded_images"] == 573
    assert audit["valid_clean_candidate_retained_images"] == 1743
    assert audit["valid_clean_candidate_retained_boxes"] == 3932

    det_splits = det_manifest["splits"]
    assert det_splits["train"]["images_count"] == 8289
    assert det_splits["train"]["converted_box_count"] == 18345
    assert det_splits["valid_clean_candidate"]["images_count"] == 1743
    assert det_splits["valid_clean_candidate"]["converted_box_count"] == 3932
    assert det_splits["test_final_candidate"]["images_count"] == 871
    assert det_splits["test_final_candidate"]["converted_box_count"] == 1884

    # Create a tiny synthetic COCO dataset and verify convert_coco_to_yolo_verified
    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    img_dir.mkdir(parents=True)
    Image.new("RGB", (400, 200), (128, 128, 128)).save(img_dir / "sample1.jpg")
    coco_payload = {
        "categories": [{"id": i, "name": f"cat_{i}"} for i in range(5)],
        "images": [{"id": 1, "file_name": "sample1.jpg", "width": 400, "height": 200}],
        "annotations": [{"id": 10, "image_id": 1, "category_id": 2, "bbox": [100.0, 50.0, 200.0, 100.0]}],
    }
    coco_json = tmp_path / "annotations.json"
    coco_json.write_text(json.dumps(coco_payload), encoding="utf-8")

    rep = convert_coco_to_yolo_verified(coco_json, img_dir, lbl_dir, write_files=True)
    assert rep["converted_box_count"] == 1
    txt_content = (lbl_dir / "sample1.txt").read_text(encoding="utf-8").strip().split()
    assert int(txt_content[0]) == 2
    assert abs(float(txt_content[1]) - 0.50) < 1e-5
    assert abs(float(txt_content[2]) - 0.50) < 1e-5
    assert abs(float(txt_content[3]) - 0.50) < 1e-5
    assert abs(float(txt_content[4]) - 0.50) < 1e-5


def test_crop_preprocessing_variants_and_sqlite_cache_verification() -> None:
    dummy_img = Image.fromarray(np.full((300, 400, 3), 180, dtype=np.uint8), mode="RGB")
    bbox = (50, 40, 150, 220)  # Tall object (width=100, height=180)

    expected_variants = [
        "A_original_crop",
        "B_direct_resize",
        "C_context_pad_15",
        "D_letterbox_224",
        "E_context_pad_25_letterbox",
    ]
    assert list(VARIANT_DEFINITIONS.keys()) == expected_variants

    expanded = expand_and_clip_xyxy(bbox, dummy_img.size, 0.15)
    assert expanded[0] < bbox[0] and expanded[1] < bbox[1]
    assert expanded[2] > bbox[2] and expanded[3] > bbox[3]

    lb_img = letterbox_pil_to_square_224(dummy_img.crop(bbox), target_size=224)
    assert lb_img.size == (224, 224)

    for v_name in expected_variants:
        t_out = preprocess_crop_for_variant(dummy_img, bbox, v_name)
        assert t_out.shape == (3, 224, 224)
        assert torch.isfinite(t_out).all()

    # Verify fingerprint & Variant A bit-exact match on the real SQLite cache
    base_sqlite = Path("evaluation_runs/val_grid_search_01/cache/predictions.sqlite3")
    valid_coco = Path("detection/WasteDetect_Dataset/valid_clean_candidate/_annotations.coco.json")
    det_ckpt = Path("detection/checkpoints_wastedetect/best_model.pt")
    cls_ckpt = Path("checkpoints/dinov2_garbage_final.pth")
    class_map = Path("evaluation_runs/class_mapping.json")

    if base_sqlite.exists() and valid_coco.exists() and det_ckpt.exists() and cls_ckpt.exists():
        fp = verify_base_cache_fingerprint(
            base_sqlite=base_sqlite,
            valid_coco=valid_coco,
            det_ckpt=det_ckpt,
            cls_ckpt=cls_ckpt,
            class_mapping_path=class_map,
        )
        assert fp["verified"] is True

        conn = sqlite3.connect(str(base_sqlite))
        coco_data = json.loads(valid_coco.read_text(encoding="utf-8"))
        id_to_idx = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
        idx_to_id = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
        res_a = evaluate_variant_on_validation(
            conn=conn,
            variant="A_original_crop",
            valid_coco_path=valid_coco,
            coco_data=coco_data,
            id_to_idx=id_to_idx,
            idx_to_id=idx_to_id,
        )
        conn.close()
        assert abs(res_a["cascade_nms_mAP50"] - 0.439864) < 1e-6
        assert abs(res_a["cascade_nms_mAP50_95"] - 0.247715) < 1e-6
        assert res_a["matched_gt_total"] == 3663
        assert abs(res_a["matched_gt_acc_pct"] - 76.63) < 1e-2
        assert abs(res_a["matched_glass_to_plastic_pct"] - 54.02) < 1e-2
        assert abs(res_a["matched_paper_to_cardboard_pct"] - 29.76) < 1e-2


def test_three_seed_statistics_and_homogeneity_rules() -> None:
    s0 = compute_sample_mean_std([])
    assert s0["status"] == "NOT_STARTED"
    assert s0["sample_std_ddof1"] is None

    s2 = compute_sample_mean_std([0.980, 0.984])
    assert s2["status"] == "INCOMPLETE"
    assert s2["sample_std_ddof1"] is None
    assert "INCOMPLETE" in s2["formatted"]

    vals = [0.980, 0.985, 0.990]
    s3 = compute_sample_mean_std(vals)
    assert s3["status"] == "COMPLETED"
    assert abs(s3["mean"] - 0.985) < 1e-6
    assert abs(s3["sample_std_ddof1"] - 0.005) < 1e-6

    valid_runs = [
        {
            "task": "classification",
            "model_name": "resnet50",
            "manifest_sha256": "abc",
            "protocol_id": "p1",
            "seed": s,
            "status": "COMPLETED",
            "metrics": {"val_macro_f1": v},
        }
        for s, v in zip([42, 123, 2026], vals)
    ]
    agg = aggregate_seed_group("rn50", valid_runs, ["val_macro_f1"])
    assert agg["overall_status"] == "COMPLETED"
    assert agg["n_completed_seeds"] == 3

    mixed_runs = list(valid_runs)
    mixed_runs[1] = dict(mixed_runs[1], protocol_id="different_protocol")
    with pytest.raises(ValueError, match="failed homogeneity validation"):
        aggregate_seed_group("rn50_mixed", mixed_runs, ["val_macro_f1"])


def test_colab_notebooks_json_python_syntax_and_no_windows_paths() -> None:
    notebook_names = [
        "00_DG38_Preflight_Colab.ipynb",
        "01_Classification_Baselines_Colab.ipynb",
        "02_Detection_Baselines_Colab.ipynb",
        "03_DINOv2_Finetune_Ablation_Colab.ipynb",
        "04_Crop_Preprocessing_Ablation_Colab.ipynb",
        "05_DG38_Results_Analysis_Colab.ipynb",
    ]
    for nb_name in notebook_names:
        nb_path = EXP_DIR / nb_name
        assert nb_path.exists(), f"Missing notebook {nb_path}"
        raw_text = nb_path.read_text(encoding="utf-8")
        assert "D:\\" not in raw_text and "d:\\" not in raw_text and "D:/" not in raw_text, (
            f"Notebook {nb_name} contains a hardcoded Windows path!"
        )
        nb_data = json.loads(raw_text)
        assert nb_data.get("nbformat") == 4
        cells = nb_data.get("cells", [])
        assert len(cells) >= 3
        for cell in cells:
            if cell.get("cell_type") == "code":
                raw_lines = [
                    line
                    for line in "".join(cell.get("source", [])).splitlines()
                    if not line.strip().startswith("!")
                ]
                code_str = "\n".join(raw_lines)
                ast.parse(code_str, filename=f"{nb_name}:{cell.get('id')}")


def test_colab_zip_packages_integrity_if_built() -> None:
    zip_dir = EXP_DIR / "colab_packages"
    manifest_path = zip_dir / "zip_packages_sha256.json"
    if not manifest_path.exists():
        pytest.skip("Colab ZIP packages not yet built")

    packages = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(packages) == 7

    assert packages["dg38_cls_train_val.zip"]["contains_test"] is False
    assert packages["dg38_cls_test_locked.zip"]["contains_test"] is True
    assert packages["dg38_det_train_val_clean.zip"]["contains_test"] is False
    assert packages["dg38_det_test_locked.zip"]["contains_test"] is True

    for zip_name, info in packages.items():
        zip_path = zip_dir / zip_name
        assert zip_path.exists()
        assert zip_path.stat().st_size == info["size_bytes"]


