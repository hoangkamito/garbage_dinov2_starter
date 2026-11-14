"""Unified Detection Baseline Trainer, COCOeval Evaluator, and Pilot Forecaster for DG38 (Stages 4, 5, 8).

Supports:
- `yolo11s` (Priority #2 Detection Baseline via Ultralytics)
- `yolo11n` (Explicit lightweight baseline if needed; never silently replaces `yolo11s`)
- `rtdetr-l` (Stage 8 optional expansion baseline via Ultralytics)
- `deformable_detr` (Stage 5/8 three-seed training for DINOv2 ViT-S/14 + Deformable DETR from `detection/Train.py`)

Mandatory Scientific & Engineering Rules (Stage 4 & 5):
1. Converts COCO annotations (`0..4`) to YOLO format from `train/annotations_coco.json` (8,289 images / 18,345 boxes)
   and `valid_clean_candidate/annotations_coco.json` (1,743 images / 3,932 boxes) and verifies every box.
2. NEVER uses the original dirty `valid` split (2,316 images with Train overlap) for checkpoint selection.
3. Evaluates all models with `pycocotools.cocoeval.COCOeval` (`maxDets=[1, 10, 100]`) for `mAP@50`, `mAP@50:95`,
   `AR@100`, `AP_S`, `AP_M`, `AP_L`, and per-class `AP@50` / `AP@50:95`.
4. Selects each model's operating score threshold independently on Clean Validation (`valid_clean_candidate`)
   by maximizing micro-F1@IoU=0.50 (never forces Deformable DETR's 0.30 threshold onto YOLO).
5. Measures inference latency (`ms/image` and `FPS`) with CUDA synchronization and tags the exact GPU name
   so L4/A100 FPS is never conflated with historical RTX 3050 Laptop GPU numbers.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as dt
import io
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "detection") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "detection"))
if str(PROJECT_ROOT / "evaluation_runs") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "evaluation_runs"))

from cu_budget_tracker import CUBudgetTracker, PilotTimingMeasurement
from grid_metrics import best_pr, metric_sweep
from gpu_runtime_manager import (
    inspect_runtime,
    load_resumable_checkpoint,
    recommend_execution_config,
    save_resumable_checkpoint,
    save_run_hardware_log,
    seed_all_States,
)
from prepare_colab_packages import (
    EXPECTED_DET_COUNTS,
    convert_coco_to_yolo_verified,
    sha256_file,
)


PR_SWEEP_THRESHOLDS = [round(x, 2) for x in np.arange(0.05, 0.91, 0.05).tolist()]


def prepare_yolo_clean_workspace(
    det_root: str | Path,
    yolo_work_dir: str | Path,
) -> Dict[str, Any]:
    """Create a verified YOLO dataset directory linking/copying `train` (8,289) and `valid_clean_candidate` (1,743)."""
    det_root = Path(det_root).resolve()
    work_dir = Path(yolo_work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    splits_map = {
        "train": det_root / "train",
        "val": det_root / "valid_clean_candidate",
    }

    verification: Dict[str, Any] = {}
    for yolo_split, src_split_dir in splits_map.items():
        coco_json = src_split_dir / "annotations_coco.json"
        src_images = src_split_dir / "images"
        expected_key = "train" if yolo_split == "train" else "valid_clean_candidate"
        expected = EXPECTED_DET_COUNTS[expected_key]

        actual_sha = sha256_file(coco_json)
        if actual_sha != expected["coco_sha256"]:
            raise ValueError(
                f"SHA-256 của {coco_json} ({actual_sha}) không khớp chuẩn kiểm toán ({expected['coco_sha256']})!"
            )

        dst_split_dir = work_dir / yolo_split
        dst_images = dst_split_dir / "images"
        dst_labels = dst_split_dir / "labels"
        dst_Split_ready = dst_split_dir / ".verified_complete.json"

        if not dst_Split_ready.exists():
            dst_split_dir.mkdir(parents=True, exist_ok=True)
            if not dst_images.exists():
                try:
                    # Prefer symlink/junction for zero disk & zero I/O overhead on Colab /content/
                    os.symlink(src_images, dst_images, target_is_directory=True)
                except OSError:
                    shutil.copytree(src_images, dst_images, dirs_exist_ok=True)

            conv = convert_coco_to_yolo_verified(
                coco_json_path=coco_json,
                images_dir=src_images,
                out_labels_dir=dst_labels,
                write_files=True,
            )
            if conv["images_count"] != expected["images"] or conv["converted_box_count"] != expected["boxes"]:
                raise RuntimeError(f"Lỗi kiểm tra số lượng box sau khi chuyển đổi sang YOLO cho {yolo_split}: {conv}")
            dst_Split_ready.write_text(json.dumps(conv, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            conv = json.loads(dst_Split_ready.read_text(encoding="utf-8"))

        verification[yolo_split] = conv

    yaml_path = work_dir / "wastedetect_clean.yaml"
    yaml_content = (
        f"path: {work_dir.as_posix()}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"nc: 5\n"
        f"names:\n"
        f"  0: glass\n"
        f"  1: metal\n"
        f"  2: organic\n"
        f"  3: paper\n"
        f"  4: plastic\n"
    )
    yaml_path.write_text(yaml_content, encoding="utf-8")

    return {
        "yaml_path": str(yaml_path),
        "work_dir": str(work_dir),
        "verification": verification,
    }


def evaluate_coco_predictions_detailed(
    coco_json_path: str | Path,
    predictions: List[Dict[str, Any]],
    pr_thresholds: Optional[List[float]] = None,
    locked_pr_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Run standard COCOeval (`iouType='bbox'`, `maxDets=[1, 10, 100]`) including per-class AP and area AP_S/M/L."""
    thresholds = pr_thresholds or PR_SWEEP_THRESHOLDS
    metadata = json.loads(Path(coco_json_path).read_text(encoding="utf-8-sig"))
    image_ids = [int(im["id"]) for im in metadata["images"]]
    categories = {int(c["id"]): c["name"] for c in metadata["categories"]}

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = copy.deepcopy(metadata)
        coco_gt.createIndex()

        if predictions:
            coco_dt = coco_gt.loadRes(copy.deepcopy(predictions))
        else:
            coco_dt = COCO()
            coco_dt.dataset = {
                "images": copy.deepcopy(metadata["images"]),
                "categories": copy.deepcopy(metadata["categories"]),
                "annotations": [],
            }
            coco_dt.createIndex()

        evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
        evaluator.params.imgIds = image_ids
        evaluator.params.catIds = sorted(categories.keys())
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    def stat_val(idx: int) -> Optional[float]:
        v = float(evaluator.stats[idx])
        return round(v, 6) if v >= 0 else None

    # Per-class AP50 and AP50:95 from evaluator.eval['precision']
    # shape: [T, R, K, A, M] where T=10 (0.50:0.95), A=0 (all), M=2 (maxDets=100)
    precisions = evaluator.eval["precision"]
    per_class_ap: Dict[str, Dict[str, Optional[float]]] = {}
    for k_idx, cid in enumerate(sorted(categories.keys())):
        cname = categories[cid]
        p_50 = precisions[0, :, k_idx, 0, 2]
        p_all = precisions[:, :, k_idx, 0, 2]
        ap50 = float(np.mean(p_50[p_50 > -1])) if np.any(p_50 > -1) else None
        ap50_95 = float(np.mean(p_all[p_all > -1])) if np.any(p_all > -1) else None
        per_class_ap[cname] = {
            "category_id": cid,
            "AP50": round(ap50, 6) if ap50 is not None else None,
            "AP50_95": round(ap50_95, 6) if ap50_95 is not None else None,
        }

    _, sweep = metric_sweep(coco_gt, predictions, image_ids, thresholds)
    selected_op = best_pr(sweep)
    if locked_pr_threshold is not None:
        _, locked_sweep = metric_sweep(coco_gt, predictions, image_ids, [locked_pr_threshold])
        operating_point = locked_sweep[0]
        operating_point["selection_mode"] = "locked_from_validation"
    else:
        operating_point = dict(selected_op)
        operating_point["selection_mode"] = "selected_on_validation_max_f1_iou50"

    return {
        "images_evaluated": len(image_ids),
        "gt_boxes_count": len(metadata.get("annotations", [])),
        "exported_predictions_count": len(predictions),
        "mAP50_95": stat_val(0),
        "mAP50": stat_val(1),
        "mAP75": stat_val(2),
        "AP_small": stat_val(3),
        "AP_medium": stat_val(4),
        "AP_large": stat_val(5),
        "AR1": stat_val(6),
        "AR10": stat_val(7),
        "AR100": stat_val(8),
        "AR_small": stat_val(9),
        "AR_medium": stat_val(10),
        "AR_large": stat_val(11),
        "per_class_ap": per_class_ap,
        "operating_point": operating_point,
        "validation_best_pr_point": selected_op,
        "pr_threshold_sweep": sweep,
    }


@torch.inference_mode()
def export_ultralytics_coco_predictions(
    weights_path: str | Path,
    coco_json_path: str | Path,
    images_dir: str | Path,
    imgsz: int = 640,
    conf_for_ap: float = 0.001,
    iou_nms: float = 0.70,
    max_det: int = 100,
    device: str = "cuda:0",
    warmup_images: int = 3,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run Ultralytics YOLO11/RT-DETR inference image-by-image with CUDA-synchronized latency measurement."""
    from ultralytics import RTDETR, YOLO

    w_str = str(weights_path)
    model = RTDETR(w_str) if "rtdetr" in w_str.lower() else YOLO(w_str)

    metadata = json.loads(Path(coco_json_path).read_text(encoding="utf-8-sig"))
    images = metadata["images"]
    img_root = Path(images_dir)

    use_cuda = device.startswith("cuda") and torch.cuda.is_available()

    # Warmup
    for im in images[:warmup_images]:
        pil = Image.open(img_root / im["file_name"]).convert("RGB")
        model.predict(pil, imgsz=imgsz, conf=conf_for_ap, iou=iou_nms, max_det=max_det, device=device, verbose=False)

    predictions: List[Dict[str, Any]] = []
    latencies_ms: List[float] = []

    for im in images:
        with Image.open(img_root / im["file_name"]) as p:
            pil = p.convert("RGB")
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        res_list = model.predict(
            pil, imgsz=imgsz, conf=conf_for_ap, iou=iou_nms, max_det=max_det, device=device, verbose=False
        )
        if use_cuda:
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        res = res_list[0]
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.detach().cpu().numpy()
            confs = res.boxes.conf.detach().cpu().numpy()
            clss = res.boxes.cls.detach().cpu().numpy()
            for box, score, cid in zip(xyxy, confs, clss):
                x1, y1, x2, y2 = [float(v) for v in box]
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                if w > 0 and h > 0:
                    predictions.append({
                        "image_id": int(im["id"]),
                        "category_id": int(cid),
                        "bbox": [round(x1, 3), round(y1, 3), round(w, 3), round(h, 3)],
                        "score": float(score),
                    })

    rt = inspect_runtime(device)
    latency_stats = {
        "gpu_name": rt["gpu_name"],
        "imgsz": imgsz,
        "images_measured": len(latencies_ms),
        "mean_ms_per_image": round(float(np.mean(latencies_ms)), 3) if latencies_ms else None,
        "p50_ms_per_image": round(float(np.percentile(latencies_ms, 50)), 3) if latencies_ms else None,
        "p95_ms_per_image": round(float(np.percentile(latencies_ms, 95)), 3) if latencies_ms else None,
        "approx_fps": round(1000.0 / float(np.mean(latencies_ms)), 2) if latencies_ms else None,
        "hardware_comparison_warning_vi": (
            f"Được đo trực tiếp trên GPU '{rt['gpu_name']}' với CUDA synchronization. "
            "Tuyệt đối không so sánh trực tiếp FPS này với số đo lịch sử trên RTX 3050 Laptop GPU như thể cùng phần cứng."
        ),
    }
    return predictions, latency_stats


def run_ultralytics_pipeline(args: argparse.Namespace, out_dir: Path, run_id: str) -> None:
    from ultralytics import RTDETR, YOLO

    yolo_ws = prepare_yolo_clean_workspace(
        det_root=args.det_root,
        yolo_work_dir=Path(args.yolo_work_dir or (out_dir.parent / "_yolo_clean_dataset")),
    )
    yaml_path = yolo_ws["yaml_path"]
    val_coco = Path(args.det_root) / "valid_clean_candidate/annotations_coco.json"
    val_images = Path(args.det_root) / "valid_clean_candidate/images"

    exec_cfg = recommend_execution_config("yolo11s_detection")
    batch_size = args.batch_size or int(exec_cfg["physical_batch_size"])
    device_str = "0" if torch.cuda.is_available() else "cpu"
    torch_dev = "cuda:0" if torch.cuda.is_available() else "cpu"

    weights_name = f"{args.arch}.pt"
    is_rtdetr = "rtdetr" in args.arch.lower()

    if args.mode == "smoke":
        model = RTDETR(weights_name) if is_rtdetr else YOLO(weights_name)
        t0 = time.perf_counter()
        model.train(
            data=yaml_path, epochs=1, imgsz=args.imgsz, batch=min(batch_size, 8),
            device=device_str, workers=exec_cfg["num_workers"], project=str(out_dir),
            name="smoke_run", exist_ok=True, fraction=0.02, seed=args.seed, verbose=False,
        )
        smoke_sec = time.perf_counter() - t0
        rt = inspect_runtime(torch_dev)
        smoke_res = {
            "status": "SMOKE_OK",
            "run_id": run_id,
            "arch": args.arch,
            "yolo_workspace": yolo_ws,
            "smoke_elapsed_sec": round(smoke_sec, 2),
            "gpu_name": rt["gpu_name"],
        }
        (out_dir / "smoke_result.json").write_text(json.dumps(smoke_res, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(smoke_res, indent=2, ensure_ascii=False))
        return

    if args.mode == "pilot":
        if not args.cu_per_hour or args.cu_per_hour <= 0:
            raise ValueError("Chế độ pilot yêu cầu --cu-per-hour <Usage Rate từ Colab UI>.")
        model = RTDETR(weights_name) if is_rtdetr else YOLO(weights_name)
        t0 = time.perf_counter()
        # Run 10% fraction of 1 epoch to measure realistic train+val speed
        model.train(
            data=yaml_path, epochs=1, imgsz=args.imgsz, batch=batch_size,
            device=device_str, workers=exec_cfg["num_workers"], project=str(out_dir),
            name="pilot_probe", exist_ok=True, fraction=0.10, seed=args.seed, verbose=False,
        )
        probe_sec = time.perf_counter() - t0
        total_train_batches = max(1, int(np.ceil(8289 / batch_size)))
        measured_batches = max(1, int(np.ceil(total_train_batches * 0.10)))

        rt = inspect_runtime(torch_dev)
        pilot = PilotTimingMeasurement(
            gpu_name=rt["gpu_name"],
            observed_cu_per_hour=args.cu_per_hour,
            train_batches_measured=measured_batches,
            train_time_sec=probe_sec,
            total_train_batches_per_epoch=total_train_batches,
            val_batches_measured=0,
            val_time_sec=0.0,
            total_val_batches_per_epoch=0,
            checkpoint_save_sec=4.0,
        )
        fc_1seed = pilot.forecast_run(epochs=args.epochs, num_seeds=1, extra_eval_sec_per_seed=90.0)
        fc_3seeds = pilot.forecast_run(epochs=args.epochs, num_seeds=3, extra_eval_sec_per_seed=90.0)

        tracker = CUBudgetTracker(Path(__file__).resolve().parent / "cu_budget_ledger.json")
        gate = tracker.check_Experiment_gate(
            run_id=f"{run_id}_1seed", category=args.budget_category,
            forecast=fc_1seed, allow_reserve_draw=args.allow_reserve,
        )
        report = {
            "run_id": run_id,
            "arch": args.arch,
            "hardware": rt,
            "forecast_1seed": fc_1seed,
            "forecast_3seeds": fc_3seeds,
            "budget_gate_1seed": gate,
            "lightweight_fallback_note_vi": (
                "Nếu dự báo CU của YOLO11s vượt hạn mức, có thể chạy bổ sung --arch yolo11n làm baseline nhẹ, "
                "nhưng không thay thế âm thầm YOLO11s."
            ),
        }
        (out_dir / "pilot_forecast.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(tracker.format_budget_table_markdown())
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    # Train or Eval mode
    best_pt = out_dir / "ultralytics_run/weights/best.pt"
    last_pt = out_dir / "ultralytics_run/weights/last.pt"

    if args.mode == "train":
        if not args.cu_per_hour or args.cu_per_hour <= 0:
            raise ValueError("Chế độ train yêu cầu --cu-per-hour <Usage Rate từ Colab UI>.")

        tracker = CUBudgetTracker(Path(__file__).resolve().parent / "cu_budget_ledger.json")
        t_start = time.perf_counter()
        if args.resume and last_pt.exists():
            model = RTDETR(str(last_pt)) if is_rtdetr else YOLO(str(last_pt))
            model.train(resume=True)
        else:
            model = RTDETR(weights_name) if is_rtdetr else YOLO(weights_name)
            model.train(
                data=yaml_path,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=batch_size,
                device=device_str,
                workers=exec_cfg["num_workers"],
                project=str(out_dir),
                name="ultralytics_run",
                exist_ok=True,
                seed=args.seed,
                deterministic=True,
                save=True,
                save_period=5,
            )
        train_elapsed = time.perf_counter() - t_start
    else:
        train_elapsed = 0.0
        tracker = None

    if not best_pt.exists():
        raise FileNotFoundError(f"Không tìm thấy best.pt sau khi train: {best_pt}")

    # Official COCOeval on Clean Validation (1,743 images)
    val_preds, val_latency = export_ultralytics_coco_predictions(
        weights_path=best_pt,
        coco_json_path=val_coco,
        images_dir=val_images,
        imgsz=args.imgsz,
        conf_for_ap=0.001,
        iou_nms=args.nms_iou,
        max_det=100,
        device=torch_dev,
    )
    val_coco_metrics = evaluate_coco_predictions_detailed(
        coco_json_path=val_coco,
        predictions=val_preds,
    )

    locked_val_op = {
        "model": args.arch,
        "seed": args.seed,
        "selected_pr_score_threshold": val_coco_metrics["operating_point"]["pr_score_threshold"],
        "nms_iou": args.nms_iou,
        "imgsz": args.imgsz,
        "selection_basis": "Clean Validation (1,743 images / 3,932 boxes) maximizing micro-F1@IoU=0.50 via COCOeval",
    }
    (out_dir / "locked_validation_threshold.json").write_text(
        json.dumps(locked_val_op, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "val_coco_predictions.json").write_text(
        json.dumps(val_preds, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "val_coco_metrics.json").write_text(
        json.dumps(val_coco_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    rt_final = inspect_runtime(torch_dev)
    save_run_hardware_log(out_dir, exec_cfg)

    usage_entry = None
    if tracker is not None and train_elapsed > 0:
        usage_entry = tracker.record_actual_usage(
            run_id=run_id,
            category=args.budget_category,
            gpu_name=rt_final["gpu_name"],
            observed_cu_per_hour=args.cu_per_hour,
            elapsed_seconds=train_elapsed,
            status="COMPLETED",
            epochs_completed=args.epochs,
            best_val_metric=val_coco_metrics["mAP50_95"],
            checkpoint_path=str(best_pt),
        )

    protocol_differences = {
        "architecture": args.arch,
        "imgsz": args.imgsz,
        "postprocessing": f"Built-in Ultralytics NMS (iou={args.nms_iou}, max_det=100, conf_export=0.001 for COCOeval AP/AR)",
        "operating_threshold_selected_on_val": locked_val_op["selected_pr_score_threshold"],
        "differences_vs_deformable_detr_vi": (
            f"{args.arch} huấn luyện ở độ phân giải {args.imgsz}x{args.imgsz} với data augmentation chuẩn của Ultralytics "
            "(Mosaic, HSV, HorizontalFlip) và hậu xử lý NMS tích hợp; trong khi Deformable DETR + frozen DINOv2 ViT-S/14 "
            "huấn luyện ở 518x518, dùng Hungarian matching 100 queries và được đánh giá cả khi không có NMS lẫn khi thêm NMS (IoU=0.70). "
            "Cả hai đều được đánh giá bằng cùng một hàm pycocotools.cocoeval.COCOeval trên tập Validation sạch (1.743 ảnh)."
        ),
    }

    summary = {
        "status": "COMPLETED",
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_id": run_id,
        "arch": args.arch,
        "seed": args.seed,
        "epochs": args.epochs,
        "training_time_seconds": round(train_elapsed, 2),
        "cu_usage": usage_entry,
        "best_weights_path": str(best_pt),
        "best_weights_sha256": sha256_file(best_pt),
        "clean_validation_coco_metrics": {
            k: v for k, v in val_coco_metrics.items() if k != "pr_threshold_sweep"
        },
        "latency_on_active_gpu": val_latency,
        "protocol_and_differences": protocol_differences,
    }
    (out_dir / "completed_run.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["clean_validation_coco_metrics"], indent=2, ensure_ascii=False))


def run_deformable_detr_seed_pipeline(args: argparse.Namespace, out_dir: Path, run_id: str) -> None:
    """Three-seed training & COCOeval runner for DINOv2 ViT-S/14 + Deformable DETR (`detection/Train.py`)."""
    from Train import (
        CocoWasteDataset,
        DINOv2DeformableDETR,
        SetCriterion,
        collate_fn,
        compute_class_weights,
        get_trainable_state_dict,
    )
    from waste_cascade import select_proposals

    train_coco = Path(args.det_root) / "train/annotations_coco.json"
    train_images = Path(args.det_root) / "train/images"
    val_coco = Path(args.det_root) / "valid_clean_candidate/annotations_coco.json"
    val_images = Path(args.det_root) / "valid_clean_candidate/images"

    if sha256_file(val_coco) != EXPECTED_DET_COUNTS["valid_clean_candidate"]["coco_sha256"]:
        raise ValueError("Deformable DETR phải dùng Validation sạch (valid_clean_candidate, 1.743 ảnh) để chọn checkpoint!")

    exec_cfg = recommend_execution_config("deformable_detr_training")
    phys_bs = args.batch_size or 1  # Keep batch_size=1 as in original Train.py unless overridden
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_ds = CocoWasteDataset(str(train_coco), str(train_images), img_size=518)
    val_ds = CocoWasteDataset(str(val_coco), str(val_images), img_size=518)

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=phys_bs, shuffle=True, collate_fn=collate_fn,
        num_workers=exec_cfg["num_workers"], pin_memory=(device.type == "cuda"), generator=g,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=phys_bs, shuffle=False, collate_fn=collate_fn,
        num_workers=exec_cfg["num_workers"], pin_memory=(device.type == "cuda"),
    )

    class_weights = compute_class_weights(str(train_coco), 5).to(device)
    model = DINOv2DeformableDETR(num_classes=5, num_queries=100, freeze_backbone=True).to(device)
    criterion = SetCriterion(5, class_weights=class_weights).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    if args.mode in ("smoke", "pilot"):
        model.train()
        n_batches = 3 if args.mode == "smoke" else min(args.pilot_batches, len(train_loader))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for idx, (imgs, targets) in enumerate(train_loader):
            if idx >= n_batches:
                break
            imgs = imgs.to(device)
            targets = [{"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)} for t in targets]
            outputs = model(imgs)
            loss, _ = criterion(outputs, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.1)
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0
        rt = inspect_runtime(device)
        if args.mode == "smoke":
            res = {
                "status": "SMOKE_OK",
                "run_id": run_id,
                "cuda_extension_note": "Pure PyTorch F.grid_sample MSDeformAttn verified (no custom C++/CUDA compilation needed).",
                "batches": n_batches,
                "elapsed_sec": round(elapsed, 3),
                "gpu_name": rt["gpu_name"],
            }
            (out_dir / "smoke_result.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
            print(json.dumps(res, indent=2, ensure_ascii=False))
            return

        if not args.cu_per_hour or args.cu_per_hour <= 0:
            raise ValueError("Chế độ pilot yêu cầu --cu-per-hour.")
        pilot = PilotTimingMeasurement(
            gpu_name=rt["gpu_name"],
            observed_cu_per_hour=args.cu_per_hour,
            train_batches_measured=n_batches,
            train_time_sec=elapsed,
            total_train_batches_per_epoch=len(train_loader),
            val_batches_measured=0,
            val_time_sec=0.0,
            total_val_batches_per_epoch=len(val_loader),
        )
        fc_1seed = pilot.forecast_run(epochs=args.epochs, num_seeds=1)
        fc_3seeds = pilot.forecast_run(epochs=args.epochs, num_seeds=3)
        tracker = CUBudgetTracker(Path(__file__).resolve().parent / "cu_budget_ledger.json")
        gate = tracker.check_Experiment_gate(
            run_id=f"{run_id}_1seed", category=args.budget_category, forecast=fc_1seed, allow_reserve_draw=args.allow_reserve
        )
        report = {"run_id": run_id, "hardware": rt, "forecast_1seed": fc_1seed, "forecast_3seeds": fc_3seeds, "budget_gate_1seed": gate}
        (out_dir / "pilot_forecast.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(tracker.format_budget_table_markdown())
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    # Full training with Clean Validation COCOeval checkpoint selection
    if not args.cu_per_hour or args.cu_per_hour <= 0:
        raise ValueError("Chế độ train yêu cầu --cu-per-hour.")

    start_epoch = 1
    best_map50_95 = -1.0
    history: Dict[str, List[Any]] = {"epoch": [], "train_loss": [], "val_mAP50": [], "val_mAP50_95": [], "epoch_sec": []}
    last_ckpt_path = out_dir / "last_checkpoint.pt"
    best_ckpt_path = out_dir / "best_model.pt"

    if args.resume and last_ckpt_path.exists():
        ckpt = torch.load(last_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_map50_95 = float(ckpt.get("best_mAP50_95", -1.0))
        history = ckpt.get("history", history)

    val_meta = json.loads(val_coco.read_text(encoding="utf-8-sig"))
    val_img_sizes = {int(im["id"]): (int(im["width"]), int(im["height"])) for im in val_meta["images"]}

    t_train0 = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        ep_t0 = time.perf_counter()
        model.train()
        total_loss = 0.0
        for imgs, targets in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = [{"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)} for t in targets]
            outputs = model(imgs)
            loss, _ = criterion(outputs, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.1)
            optimizer.step()
            total_loss += float(loss.item())
        scheduler.step()
        avg_loss = total_loss / max(len(train_loader), 1)

        # Evaluate on Clean Validation via COCOeval
        model.eval()
        val_preds: List[Dict[str, Any]] = []
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                outputs = model(imgs)
                probs = outputs["pred_logits"].sigmoid()
                boxes = outputs["pred_boxes"]
                for b in range(imgs.size(0)):
                    img_id = int(targets[b]["image_id"])
                    W, H = val_img_sizes[img_id]
                    scores, det_idx = probs[b].max(dim=-1)
                    q_idx, xyxy_boxes = select_proposals(
                        boxes[b], scores, (W, H), proposal_threshold=0.0, max_proposals=100, nms_iou=0.70
                    )
                    for q, box in zip(q_idx.tolist(), xyxy_boxes.tolist()):
                        coords = [round(v) for v in box]
                        if coords[2] > coords[0] and coords[3] > coords[1]:
                            val_preds.append({
                                "image_id": img_id,
                                "category_id": int(det_idx[q]),
                                "bbox": [coords[0], coords[1], coords[2] - coords[0], coords[3] - coords[1]],
                                "score": float(scores[q]),
                            })

        coco_res = evaluate_coco_predictions_detailed(val_coco, val_preds)
        cur_map = float(coco_res["mAP50_95"] or 0.0)
        cur_map50 = float(coco_res["mAP50"] or 0.0)
        ep_sec = time.perf_counter() - ep_t0

        history["epoch"].append(epoch)
        history["train_loss"].append(round(avg_loss, 5))
        history["val_mAP50"].append(cur_map50)
        history["val_mAP50_95"].append(cur_map)
        history["epoch_sec"].append(round(ep_sec, 2))

        payload = {
            "epoch": epoch,
            "seed": args.seed,
            "model_state_dict": get_trainable_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_mAP50_95": max(best_map50_95, cur_map),
            "args": vars(args),
            "history": history,
        }
        if cur_map > best_map50_95:
            best_map50_95 = cur_map
            torch.save(payload, best_ckpt_path)
            (out_dir / "val_coco_metrics.json").write_text(
                json.dumps(coco_res, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        torch.save(payload, last_ckpt_path)
        print(
            f"[{run_id}] Epoch {epoch:02d}/{args.epochs} ({ep_sec:.1f}s) | "
            f"loss={avg_loss:.4f} | CleanVal mAP50={cur_map50:.4f} mAP50:95={cur_map:.4f}",
            flush=True,
        )

    total_sec = time.perf_counter() - t_train0
    rt_final = inspect_runtime(device)
    tracker = CUBudgetTracker(Path(__file__).resolve().parent / "cu_budget_ledger.json")
    usage_entry = tracker.record_actual_usage(
        run_id=run_id, category=args.budget_category, gpu_name=rt_final["gpu_name"],
        observed_cu_per_hour=args.cu_per_hour, elapsed_seconds=total_sec,
        status="COMPLETED", epochs_completed=args.epochs, best_val_metric=best_map50_95,
        checkpoint_path=str(best_ckpt_path),
    )
    summary = {
        "status": "COMPLETED",
        "run_id": run_id,
        "arch": "deformable_detr",
        "seed": args.seed,
        "epochs": args.epochs,
        "best_val_mAP50_95": best_map50_95,
        "training_time_seconds": round(total_sec, 2),
        "cu_usage": usage_entry,
        "best_checkpoint_path": str(best_ckpt_path),
    }
    (out_dir / "completed_run.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Huấn luyện và đánh giá COCOeval cho YOLO11s, RT-DETR và Deformable DETR.")
    parser.add_argument("--mode", choices=["prepare-yolo", "smoke", "pilot", "train", "eval_val"], required=True)
    parser.add_argument("--arch", choices=["yolo11s", "yolo11n", "rtdetr-l", "deformable_detr"], default="yolo11s")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (42, 123, 2026)")
    parser.add_argument("--det-root", default=str(PROJECT_ROOT / "WasteDetect_Dataset"))
    parser.add_argument("--yolo-work-dir", "--yolo-dir", dest="yolo_work_dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--budget-category", default="yolo11s_baseline")
    parser.add_argument("--cu-per-hour", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--pilot-batches", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--allow-reserve", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or f"{args.arch}_seed{args.seed}"

    if args.mode == "prepare-yolo":
        yolo_ws = prepare_yolo_clean_workspace(
            det_root=args.det_root,
            yolo_work_dir=Path(args.yolo_work_dir or (out_dir.parent / "_yolo_clean_dataset")),
        )
        print(json.dumps(yolo_ws, indent=2, ensure_ascii=False))
        return

    completed_marker = out_dir / "completed_run.json"
    if args.mode == "train" and completed_marker.exists() and not args.force_retrain:
        print(f"[SKIP] Run '{run_id}' đã hoàn thành hợp lệ tại {completed_marker}. Không tốn thêm GPU CU!")
        print(completed_marker.read_text(encoding="utf-8"))
        return

    seed_all_States(args.seed)
    if args.arch == "deformable_detr":
        run_deformable_detr_seed_pipeline(args, out_dir, run_id)
    else:
        run_ultralytics_pipeline(args, out_dir, run_id)


if __name__ == "__main__":
    main()
