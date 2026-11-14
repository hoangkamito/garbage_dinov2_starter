"""
Generator for the 6 DG38 Google Colab Pro Notebooks (nbformat v4).
Ensures valid JSON structure, parameterized cells, Vietnamese instructions,
dynamic GPU detection (L4, A100, G4), CU budget gates, zero auto-run waste,
and 100% alignment with `prepare_colab_packages.py` ZIP layouts and CLI flags.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def _md_cell(cell_id: str, source: str) -> Dict[str, Any]:
    lines = [line + "\n" for line in source.strip().splitlines()]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": lines,
    }


def _code_cell(cell_id: str, source: str) -> Dict[str, Any]:
    lines = [line + "\n" for line in source.strip().splitlines()]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "id": cell_id,
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": lines,
    }


def _build_notebook(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {
                "provenance": [],
                "gpuType": "L4",
                "machine_shape": "hm",
            },
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3",
            },
            "language_info": {
                "name": "python",
            },
            "accelerator": "GPU",
        },
        "cells": cells,
    }


def create_00_preflight() -> Dict[str, Any]:
    cells = [
        _md_cell(
            "pf_title",
            """# 00 — DG38 Preflight, Kiểm Tra GPU & Dự Báo Ngân Sách Colab Pro (200.45 CU)

**Mục đích của Notebook này:**
1. Mount Google Drive, tự động giải nén gói mã nguồn (`dg38_source_code_and_manifests.zip`), gói checkpoints tham chiếu (`dg38_reference_checkpoints.zip`), gói SQLite cache (`dg38_crop_ablation_val_bundle.zip`) vào `DRIVE_ROOT/code` (nếu chưa giải nén) và giải nén nhanh dữ liệu Train/Val sang ổ SSD cục bộ của Colab (`/content/dg38_data`) để tránh nghẽn I/O.
2. Tự động nhận diện GPU hiện tại (**NVIDIA L4, A100, G4, T4...**), kiểm tra CUDA, PyTorch, Total/Available VRAM, BF16/FP16 và đề xuất cấu hình tối ưu.
3. Xác minh toàn vẹn SHA-256 của các manifest và checkpoints gốc:
   - **Classification**: `9,807` Train / `1,226` Val (`dg38_cls_train_val.zip`) và `1,226` Test tách riêng (`dg38_cls_test_locked.zip`).
   - **Detection**: `8,289` Train (`18,345` boxes) / `1,743` Clean Validation (`3,932` boxes) (`dg38_det_train_val_clean.zip`). *(Gói Test `871` ảnh / `1,884` boxes `dg38_det_test_locked.zip` được tách riêng, không nạp trong bước huấn luyện/chọn mô hình)*.
   - **Checkpoints gốc**: `checkpoints/dinov2_garbage_final.pth` và `detection/checkpoints_wastedetect/best_model.pt`.
4. Nhập **Usage Rate (CU/giờ)** thực tế hiển thị trên giao diện Google Colab (`Runtime -> View resources`) và ghi nhận số dư CU hiện tại vào `cu_budget_ledger.json`.
5. Chạy **Smoke Test** cho Classification, Detection (YOLO + Deformable DETR `MSDeformAttn`) trước khi chạy bất kỳ thí nghiệm nào.""",
        ),
        _code_cell(
            "pf_env_setup",
            """# [CELL 1] Mount Google Drive & Giải nén gói mã nguồn / checkpoints trên Drive (nếu cần)
import os
import sys
import shutil
import zipfile
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    if not Path("/content/drive/MyDrive").exists():
        drive.mount("/content/drive")

DRIVE_ROOT = Path(os.environ.get("DG38_DRIVE_ROOT", "/content/drive/MyDrive/DG38_Colab_Package"))
ZIPS_DIR = DRIVE_ROOT / "zips"
CODE_DIR = DRIVE_ROOT / "code"
ARTIFACTS_DIR = DRIVE_ROOT / "artifacts"
LOCAL_WORK_DIR = Path("/content/dg38_workspace")
LOCAL_DATA_DIR = Path("/content/dg38_data")

LOCAL_WORK_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
if DRIVE_ROOT.exists():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    CODE_DIR.mkdir(parents=True, exist_ok=True)

    # Tự động giải nén mã nguồn + manifests vào DRIVE_ROOT/code nếu chưa có model.py
    src_zip = ZIPS_DIR / "dg38_source_code_and_manifests.zip"
    if src_zip.exists() and not (CODE_DIR / "model.py").exists():
        print(f"Đang giải nén mã nguồn {src_zip} -> {CODE_DIR} ...")
        with zipfile.ZipFile(src_zip, "r") as zf:
            zf.extractall(CODE_DIR)

    # Tự động giải nén checkpoints gốc vào DRIVE_ROOT/code nếu chưa có
    ckpt_zip = ZIPS_DIR / "dg38_reference_checkpoints.zip"
    if ckpt_zip.exists() and not (CODE_DIR / "checkpoints" / "dinov2_garbage_final.pth").exists():
        print(f"Đang giải nén checkpoints tham chiếu {ckpt_zip} -> {CODE_DIR} ...")
        with zipfile.ZipFile(ckpt_zip, "r") as zf:
            zf.extractall(CODE_DIR)

    # Tự động giải nén gói SQLite cache cho Crop Ablation vào DRIVE_ROOT/code nếu chưa có
    crop_zip = ZIPS_DIR / "dg38_crop_ablation_val_bundle.zip"
    if crop_zip.exists() and not (CODE_DIR / "evaluation_runs" / "val_grid_search_01" / "cache" / "predictions.sqlite3").exists():
        print(f"Đang giải nén gói Crop Ablation cache {crop_zip} -> {CODE_DIR} ...")
        with zipfile.ZipFile(crop_zip, "r") as zf:
            zf.extractall(CODE_DIR)

REPO_ROOT = CODE_DIR if (CODE_DIR / "model.py").exists() else Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

print(f"IN_COLAB       : {IN_COLAB}")
print(f"DRIVE_ROOT     : {DRIVE_ROOT} (exists={DRIVE_ROOT.exists()})")
print(f"REPO_ROOT      : {REPO_ROOT} (model.py exists={(REPO_ROOT / 'model.py').exists()})")
print(f"LOCAL_DATA_DIR : {LOCAL_DATA_DIR}")""",
        ),
        _code_cell(
            "pf_install_deps",
            """# [CELL 2] Cài đặt các thư viện cần thiết trên Colab
!pip install -q timm transformers accelerate ultralytics pycocotools scikit-learn pandas pillow""",
        ),
        _code_cell(
            "pf_gpu_inspect",
            """# [CELL 3] Kiểm tra phần cứng GPU động (L4 / A100 / G4) & đề xuất cấu hình
import json
from evaluation_runs.dg38_additional_experiments.gpu_runtime_manager import (
    inspect_runtime,
    recommend_execution_config,
)

rt_info = inspect_runtime()
print("=== THÔNG TIN PHẦN CỨNG GPU HIỆN TẠI ===")
print(json.dumps(rt_info, indent=2, ensure_ascii=False))

print("\\n=== ĐỀ XUẤT CẤU HÌNH THEO NHIỆM VỤ TRÊN GPU HIỆN TẠI ===")
for task in [
    "smoke_test",
    "resnet50_classification",
    "yolo11s_detection",
    "crop_preprocessing_ablation",
    "dinov2_vitb14_finetune",
    "deformable_detr_training",
]:
    rec = recommend_execution_config(task, rt_info)
    print(
        f"[{task:27s}] GPU={rec['detected_gpu_family']:6s} | AMP={rec['amp_dtype']:8s} | "
        f"Batch={rec['physical_batch_size']:2d} x Accum={rec['grad_accum_steps']} (Eff={rec['effective_batch_size']:2d}) | "
        f"Khuyến nghị: {rec['recommended_primary_gpu']}"
    )""",
        ),
        _code_cell(
            "pf_unpack_verify",
            """# [CELL 4] Giải nén dữ liệu Train/Val sang ổ SSD `/content/dg38_data` & Kiểm tra SHA-256 Manifest + Checkpoints
from evaluation_runs.dg38_additional_experiments.prepare_colab_packages import sha256_file

EXP_DIR = REPO_ROOT / "evaluation_runs" / "dg38_additional_experiments"
MANIFEST_DIR = EXP_DIR / "manifests"

for zip_name, target_sub, sentinel_rel in [
    (
        "dg38_cls_train_val.zip",
        LOCAL_DATA_DIR / "classification",
        "standardized_256",
    ),
    (
        "dg38_det_train_val_clean.zip",
        LOCAL_DATA_DIR / "detection",
        "WasteDetect_Dataset/valid_clean_candidate/_annotations.coco.json",
    ),
]:
    zip_path = ZIPS_DIR / zip_name
    sentinel_path = target_sub / sentinel_rel
    if zip_path.exists() and not sentinel_path.exists():
        print(f"Đang giải nén {zip_path} -> {target_sub} ...")
        target_sub.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_sub)
        print(f"[OK] Hoàn tất giải nén {zip_name}!")
    else:
        print(f"[INFO] {zip_name}: zip_exists={zip_path.exists()}, unpacked={sentinel_path.exists()}")

# Kiểm tra các đường dẫn quan trọng
cls_ckpt = REPO_ROOT / "checkpoints" / "dinov2_garbage_final.pth"
det_ckpt = REPO_ROOT / "detection" / "checkpoints_wastedetect" / "best_model.pt"
sqlite_cache = REPO_ROOT / "evaluation_runs" / "val_grid_search_01" / "cache" / "predictions.sqlite3"

for label, p in [
    ("Classifier Checkpoint (checkpoints/dinov2_garbage_final.pth)", cls_ckpt),
    ("Detector Checkpoint (detection/checkpoints_wastedetect/best_model.pt)", det_ckpt),
    ("Validation SQLite Cache (predictions.sqlite3)", sqlite_cache),
]:
    if p.exists():
        print(f"[OK] {label}: size={p.stat().st_size / (1024**2):.2f} MB | sha256={sha256_file(p)[:16]}...")
    else:
        print(f"[CẢNH BÁO] Chưa tìm thấy {label} tại {p}")

index_path = MANIFEST_DIR / "dataset_manifests_index.json"
if index_path.exists():
    idx_data = json.loads(index_path.read_text(encoding="utf-8"))
    print("\\n=== THÔNG TIN MANIFEST ĐÃ KHOÁ (SHA-256) ===")
    print(json.dumps(idx_data, indent=2, ensure_ascii=False))
else:
    print(f"[CẢNH BÁO] Không tìm thấy {index_path}.")""",
        ),
        _code_cell(
            "pf_budget_init",
            """# [CELL 5] Nhập Usage Rate (CU/giờ) thực tế từ giao diện Colab & Đồng bộ Ngân sách
# HƯỚNG DẪN BẮT BUỘC: Mở menu Colab "Runtime -> View resources" để xem tốc độ tiêu thụ thực tế
# (Ví dụ tham khảo: L4 ~ 0.8 - 1.7 CU/h, A100 ~ 5.0 - 6.5 CU/h; vui lòng nhập đúng con số hiển thị trên Colab UI của bạn)
from evaluation_runs.dg38_additional_experiments.cu_budget_tracker import CUBudgetTracker

OBSERVED_CU_PER_HOUR = None     # <-- BẮT BUỘC: Thay None bằng số thực từ Colab UI (ví dụ: 1.48)
OBSERVED_REMAINING_CU = 200.45  # <-- Cập nhật số dư CU hiện tại trên giao diện Colab

if OBSERVED_CU_PER_HOUR is None or float(OBSERVED_CU_PER_HOUR) <= 0:
    raise ValueError(
        "Vui lòng mở 'Runtime -> View resources' trên Google Colab và nhập Usage Rate thực tế vào biến OBSERVED_CU_PER_HOUR (ví dụ: 1.48)!"
    )

LEDGER_PATH = EXP_DIR / "cu_budget_ledger.json"
tracker = CUBudgetTracker(LEDGER_PATH)
tracker.sync_user_reported_balance(
    remaining_cu_from_colab_ui=float(OBSERVED_REMAINING_CU),
    note=f"Đồng bộ từ giao diện Colab (GPU={rt_info['gpu_name']}, rate={OBSERVED_CU_PER_HOUR} CU/h)",
)
print(tracker.format_budget_table_markdown())""",
        ),
        _code_cell(
            "pf_smoke_test",
            """# [CELL 6] Smoke Test: Kiểm tra Forward/Backward ResNet50 & Deformable DETR MSDeformAttn trên GPU hiện tại
import time
import torch
import torchvision.models as tv_models
from detection.dinov2_deformable_detr import MSDeformAttn

t0 = time.perf_counter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Smoke test Torchvision ResNet50
m_rn50 = tv_models.resnet50(weights=None, num_classes=10).to(device)
x_dummy = torch.randn(4, 3, 224, 224, device=device)
out_rn50 = m_rn50(x_dummy)
loss_rn50 = out_rn50.sum()
loss_rn50.backward()
assert out_rn50.shape == (4, 10), f"Sai shape ResNet50: {out_rn50.shape}"
del m_rn50, out_rn50, loss_rn50

# 2. Smoke test Deformable DETR Pure-PyTorch MSDeformAttn
ms_attn = MSDeformAttn(d_model=256, n_levels=3, n_heads=8, n_points=4).to(device)
q = torch.randn(2, 100, 256, device=device)
ref_pts = torch.rand(2, 100, 3, 2, device=device)
spatial_shapes = torch.tensor([[36, 36], [18, 18], [9, 9]], dtype=torch.long, device=device)
total_tokens = int((spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum().item())
src_feat = torch.randn(2, total_tokens, 256, device=device)
level_start = torch.tensor([0, 1296, 1296 + 324], dtype=torch.long, device=device)
out_attn = ms_attn(q, ref_pts, src_feat, spatial_shapes, level_start)
assert out_attn.shape == (2, 100, 256), f"Sai shape MSDeformAttn: {out_attn.shape}"
del ms_attn, q, ref_pts, src_feat, out_attn

if torch.cuda.is_available():
    torch.cuda.synchronize()
elapsed_s = time.perf_counter() - t0
print(f"[OK] Smoke test thành công trên {rt_info['gpu_name']} trong {elapsed_s:.2f}s!")""",
        ),
    ]
    return _build_notebook(cells)


def create_01_classification() -> Dict[str, Any]:
    cells = [
        _md_cell(
            "cls_title",
            """# 01 — DG38 Classification Baselines & 3-Seed Experiments (ResNet50 / DINOv2 ViT-B/14 / ViT-B/16)

**Thứ tự ưu tiên theo kế hoạch DG38:**
1. **Ưu tiên số 1 — ResNet50 Baseline (`resnet50`)**:
   - Huấn luyện trên đúng tập Classification cố định: **9,807 Train / 1,226 Val** (10 lớp, trong `/content/dg38_data/classification/standardized_256`), chọn checkpoint theo **Validation Macro F1**.
   - Chạy Pilot 20 batches ở Cell 2 để đo tốc độ thực tế trên GPU hiện tại và dự báo CU trước khi chạy đầy đủ 15 epochs ở Cell 3.
   - Sau khi hoàn thành Seed `42`, tiếp tục chạy Seed `123` và `2026` để có **Mean ± Sample Std (`ddof=1`)**.
2. **Mở rộng 3 Seeds cho DINOv2 ViT-B/14 (`dinov2_vitb14`, fine-tune 4 block cuối)**:
   - Chạy `SEED = 123` và `SEED = 2026` khi ngân sách trong nhóm `three_seed_experiments` còn đủ.
3. **Đánh giá trên Tập Test Đã Khoá (`1,226` ảnh, Cell 4)**:
   - Tập Test (`dg38_cls_test_locked.zip`) chỉ được giải nén và đánh giá sau khi quá trình huấn luyện + chọn best checkpoint trên Validation đã hoàn tất (`RUN_LOCKED_TEST_EVAL = True`).

> **An toàn CU:** Biến `RUN_FULL_TRAINING` mặc định là `False` và `OBSERVED_CU_PER_HOUR` bắt buộc do người dùng nhập từ giao diện Colab (`Runtime -> View resources`).""",
        ),
        _code_cell(
            "cls_setup",
            """# [CELL 1] Thiết lập môi trường & tham số thí nghiệm Classification
import os
import sys
import zipfile
import subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    if not Path("/content/drive/MyDrive").exists():
        drive.mount("/content/drive")

DRIVE_ROOT = Path(os.environ.get("DG38_DRIVE_ROOT", "/content/drive/MyDrive/DG38_Colab_Package"))
REPO_ROOT = DRIVE_ROOT / "code" if (DRIVE_ROOT / "code" / "model.py").exists() else Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXP_DIR = REPO_ROOT / "evaluation_runs" / "dg38_additional_experiments"
MANIFEST_PATH = EXP_DIR / "manifests" / "classification_train_val_manifest.json"
TEST_MANIFEST_PATH = EXP_DIR / "manifests" / "classification_test_manifest.json"
DATA_ROOT = Path(os.environ.get("DG38_CLS_DATA_ROOT", "/content/dg38_data/classification/standardized_256"))
ARTIFACTS_DIR = (DRIVE_ROOT / "artifacts") if DRIVE_ROOT.exists() else EXP_DIR

# ================= THAM SỐ THÍ NGHIỆM =================
ARCH = "resnet50"                # Lựa chọn: "resnet50" | "dinov2_vitb14" | "vit_b_16"
SEED = 42                        # Lựa chọn: 42 | 123 | 2026
UNFREEZE_BLOCKS = 4              # Chỉ áp dụng cho dinov2_vitb14 (mặc định 4 block cuối)
EPOCHS = 15                      # 15 epochs chuẩn
LR = 3e-4 if ARCH == "resnet50" else 1e-4

# ================= CỔNG KIỂM SOÁT NGÂN SÁCH CU =================
OBSERVED_CU_PER_HOUR = None      # <-- BẮT BUỘC: Nhập Usage Rate thực tế từ Colab UI (ví dụ: 1.48)
BUDGET_CATEGORY = "resnet50_baseline" if (ARCH == "resnet50" and SEED == 42) else "three_seed_experiments"
RUN_FULL_TRAINING = False        # ĐỔI THÀNH True SAU KHI XEM BẢNG DỰ BÁO CU Ở CELL 2 ĐẠT APPROVED
RUN_LOCKED_TEST_EVAL = False     # ĐỔI THÀNH True Ở CELL 4 SAU KHI ĐÃ TRAIN XONG VÀ CHỌN BEST CHECKPOINT TRÊN VAL

if OBSERVED_CU_PER_HOUR is None or float(OBSERVED_CU_PER_HOUR) <= 0:
    raise ValueError("Vui lòng nhập OBSERVED_CU_PER_HOUR (Usage Rate từ 'Runtime -> View resources') trước khi chạy!")

print(f"DATA_ROOT     : {DATA_ROOT} (exists={DATA_ROOT.exists()})")
print(f"MANIFEST_PATH : {MANIFEST_PATH} (exists={MANIFEST_PATH.exists()})")""",
        ),
        _code_cell(
            "cls_pilot_gate",
            """# [CELL 2] Chạy Pilot 20 Batches để đo tốc độ thực tế & Kiểm tra Cổng Ngân Sách (CU Gate)
trainer_script = EXP_DIR / "dg38_cls_trainer.py"
out_dir = ARTIFACTS_DIR / "runs" / f"cls_{ARCH}_blocks{UNFREEZE_BLOCKS if ARCH=='dinov2_vitb14' else 'all'}_seed{SEED}"

cmd_pilot = [
    sys.executable,
    str(trainer_script),
    "--mode", "pilot",
    "--arch", ARCH,
    "--unfreeze-blocks", str(UNFREEZE_BLOCKS),
    "--seed", str(SEED),
    "--epochs", str(EPOCHS),
    "--lr", str(LR),
    "--data-root", str(DATA_ROOT),
    "--manifest", str(MANIFEST_PATH),
    "--output-dir", str(out_dir),
    "--cu-per-hour", str(OBSERVED_CU_PER_HOUR),
    "--budget-category", BUDGET_CATEGORY,
]
print("Đang chạy Pilot Benchmark:", " ".join(cmd_pilot))
if DATA_ROOT.exists():
    res = subprocess.run(cmd_pilot, text=True)
    if res.returncode != 0:
        raise RuntimeError("Pilot Benchmark gặp lỗi.")
else:
    print(f"[INFO] Không tìm thấy {DATA_ROOT}. Hãy chạy Cell 4 của Notebook 00 để giải nén dg38_cls_train_val.zip.")""",
        ),
        _code_cell(
            "cls_train_exec",
            """# [CELL 3] Huấn luyện đầy đủ trên Train/Val (Hỗ trợ Resume tự động nếu phiên Colab bị ngắt) & Ghi nhận CU thực tế
if not RUN_FULL_TRAINING:
    print("[DỪNG AN TOÀN] RUN_FULL_TRAINING đang là False. Sau khi kiểm tra dự báo CU ở Cell 2 đạt approved=True, hãy đặt RUN_FULL_TRAINING = True rồi chạy lại Cell này.")
else:
    cmd_train = [
        sys.executable,
        str(trainer_script),
        "--mode", "train",
        "--arch", ARCH,
        "--unfreeze-blocks", str(UNFREEZE_BLOCKS),
        "--seed", str(SEED),
        "--epochs", str(EPOCHS),
        "--lr", str(LR),
        "--data-root", str(DATA_ROOT),
        "--manifest", str(MANIFEST_PATH),
        "--output-dir", str(out_dir),
        "--cu-per-hour", str(OBSERVED_CU_PER_HOUR),
        "--budget-category", BUDGET_CATEGORY,
        "--resume",
    ]
    res = subprocess.run(cmd_train, text=True)
    if res.returncode != 0:
        raise RuntimeError("Quá trình huấn luyện gặp lỗi hoặc bị chặn bởi cổng ngân sách CU.")""",
        ),
        _code_cell(
            "cls_eval_test_locked",
            """# [CELL 4] Đánh giá Checkpoint tốt nhất (đã chọn theo Validation Macro F1) trên Tập Test Đã Khoá (1,226 ảnh)
if not RUN_LOCKED_TEST_EVAL:
    print("[DỪNG AN TOÀN] RUN_LOCKED_TEST_EVAL đang là False. Chỉ bật True sau khi Cell 3 đã huấn luyện xong và khoá best_model.pt.")
else:
    best_ckpt = out_dir / "best_model.pt"
    if not best_ckpt.exists():
        raise FileNotFoundError(f"Chưa có best_model.pt tại {best_ckpt}. Hãy hoàn thành huấn luyện ở Cell 3 trước.")

    # Giải nén gói Test đã khoá nếu chưa giải nén
    test_zip = DRIVE_ROOT / "zips" / "dg38_cls_test_locked.zip"
    if test_zip.exists():
        print(f"Đang giải nén gói Test {test_zip} -> {DATA_ROOT.parent} ...")
        with zipfile.ZipFile(test_zip, "r") as zf:
            zf.extractall(DATA_ROOT.parent)

    cmd_eval_test = [
        sys.executable,
        str(trainer_script),
        "--mode", "eval-test",
        "--arch", ARCH,
        "--unfreeze-blocks", str(UNFREEZE_BLOCKS),
        "--seed", str(SEED),
        "--data-root", str(DATA_ROOT),
        "--manifest", str(MANIFEST_PATH),
        "--test-manifest", str(TEST_MANIFEST_PATH),
        "--checkpoint", str(best_ckpt),
        "--output-dir", str(out_dir),
    ]
    res = subprocess.run(cmd_eval_test, text=True)
    if res.returncode != 0:
        raise RuntimeError("Đánh giá trên tập Test gặp lỗi.")""",
        ),
    ]
    return _build_notebook(cells)


def create_02_detection() -> Dict[str, Any]:
    cells = [
        _md_cell(
            "det_title",
            """# 02 — DG38 Detection Baselines & 3-Seed Experiments (YOLO11s / YOLO11n / RT-DETR / Deformable DETR)

**Quy tắc khoa học bắt buộc (Giai đoạn 4 & 5):**
1. **Dữ liệu cố định:**
   - Train: `WasteDetect_Dataset/train` (**8,289 ảnh / 18,345 boxes**, 5 lớp ID 0–4).
   - Validation sạch: `WasteDetect_Dataset/valid_clean_candidate` (**1,743 ảnh / 3,932 boxes**). **Tuyệt đối không dùng tập `valid` gốc (`2,316` ảnh, có `506` ảnh trùng SHA-256 và `28` ảnh trùng pHash với `train`).**
2. **Đánh giá công bằng bằng `COCOeval`:**
   - Mọi detector (`yolo11s`, `yolo11n`, `rtdetr-l`, `deformable_detr`) đều được xuất dự đoán chuẩn COCO JSON và chấm bằng `pycocotools.cocoeval.COCOeval`.
3. **Ngưỡng vận hành độc lập (Model-specific Operating Threshold):**
   - Tìm ngưỡng confidence tối ưu F1 trực tiếp trên **Clean Validation (`1,743` ảnh)** cho từng mô hình, lưu vào `locked_validation_threshold.json`.
4. **Kiểm soát ngân sách CU:**
   - Luôn chạy Pilot đo thời gian & kiểm tra cổng ngân sách ở Cell 3 trước khi bật `RUN_FULL_TRAINING = True` ở Cell 4.""",
        ),
        _code_cell(
            "det_setup",
            """# [CELL 1] Thiết lập môi trường & tham số thí nghiệm Detection
import os
import sys
import zipfile
import subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    if not Path("/content/drive/MyDrive").exists():
        drive.mount("/content/drive")

DRIVE_ROOT = Path(os.environ.get("DG38_DRIVE_ROOT", "/content/drive/MyDrive/DG38_Colab_Package"))
REPO_ROOT = DRIVE_ROOT / "code" if (DRIVE_ROOT / "code" / "model.py").exists() else Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXP_DIR = REPO_ROOT / "evaluation_runs" / "dg38_additional_experiments"
DET_ROOT = Path(os.environ.get("DG38_DET_ROOT", "/content/dg38_data/detection/WasteDetect_Dataset"))
YOLO_DATA_DIR = Path("/content/dg38_data/yolo_clean_val")
ARTIFACTS_DIR = (DRIVE_ROOT / "artifacts") if DRIVE_ROOT.exists() else EXP_DIR

# ================= THAM SỐ THÍ NGHIỆM DETECTION =================
ARCH = "yolo11s"                 # Lựa chọn: "yolo11s" | "yolo11n" | "rtdetr-l" | "deformable_detr"
SEED = 42                        # Lựa chọn: 42 | 123 | 2026
EPOCHS = 30                      # Số epochs huấn luyện
IMGSZ = 640 if ARCH != "deformable_detr" else 518

# ================= CỔNG KIỂM SOÁT NGÂN SÁCH CU =================
OBSERVED_CU_PER_HOUR = None      # <-- BẮT BUỘC: Nhập Usage Rate thực tế từ giao diện Colab (ví dụ: 1.48)
BUDGET_CATEGORY = "yolo11s_baseline" if (ARCH.startswith("yolo") and SEED == 42) else "three_seed_experiments"
RUN_FULL_TRAINING = False        # ĐỔI THÀNH True SAU KHI XÁC NHẬN DỰ BÁO CU Ở CELL 3 ĐẠT APPROVED
RUN_LOCKED_TEST_EVAL = False     # ĐỔI THÀNH True Ở CELL 5 SAU KHI ĐÃ TRAIN VÀ KHOÁ THRESHOLD TRÊN CLEAN VAL

if OBSERVED_CU_PER_HOUR is None or float(OBSERVED_CU_PER_HOUR) <= 0:
    raise ValueError("Vui lòng nhập OBSERVED_CU_PER_HOUR (Usage Rate từ 'Runtime -> View resources') trước khi chạy!")

print(f"DET_ROOT      : {DET_ROOT} (exists={DET_ROOT.exists()})")
print(f"YOLO_DATA_DIR : {YOLO_DATA_DIR}")""",
        ),
        _code_cell(
            "det_prepare_yolo",
            """# [CELL 2] Chuẩn bị dữ liệu YOLO từ COCO Annotations & Kiểm tra tính hợp lệ của Bounding Boxes (8,289 Train + 1,743 Clean Val)
det_script = EXP_DIR / "dg38_det_trainer.py"
out_dir = ARTIFACTS_DIR / "runs" / f"det_{ARCH}_seed{SEED}"

if DET_ROOT.exists() and ARCH != "deformable_detr":
    cmd_prep = [
        sys.executable,
        str(det_script),
        "--mode", "prepare-yolo",
        "--arch", ARCH,
        "--det-root", str(DET_ROOT),
        "--yolo-work-dir", str(YOLO_DATA_DIR),
        "--output-dir", str(out_dir),
    ]
    res = subprocess.run(cmd_prep, text=True)
    if res.returncode != 0:
        raise RuntimeError("Chuẩn bị dữ liệu YOLO gặp lỗi.")
else:
    print(f"[INFO] DET_ROOT={DET_ROOT} (exists={DET_ROOT.exists()}), ARCH={ARCH}")""",
        ),
        _code_cell(
            "det_pilot_and_gate",
            """# [CELL 3] Chạy Pilot Đo Tốc Độ Thực Tế & Dự Báo Chi Phí CU
if DET_ROOT.exists():
    cmd_pilot = [
        sys.executable,
        str(det_script),
        "--mode", "pilot",
        "--arch", ARCH,
        "--seed", str(SEED),
        "--epochs", str(EPOCHS),
        "--imgsz", str(IMGSZ),
        "--det-root", str(DET_ROOT),
        "--yolo-work-dir", str(YOLO_DATA_DIR),
        "--output-dir", str(out_dir),
        "--cu-per-hour", str(OBSERVED_CU_PER_HOUR),
        "--budget-category", BUDGET_CATEGORY,
    ]
    res = subprocess.run(cmd_pilot, text=True)
    if res.returncode != 0:
        raise RuntimeError("Pilot Detection gặp lỗi.")""",
        ),
        _code_cell(
            "det_train_and_cocoeval",
            """# [CELL 4] Huấn luyện đầy đủ, Đánh giá bằng COCOeval trên Clean Validation (1,743 ảnh) & Khoá Threshold riêng
if not RUN_FULL_TRAINING:
    print("[DỪNG AN TOÀN] RUN_FULL_TRAINING đang là False. Hãy kiểm tra bảng dự báo CU ở Cell 3 trước khi bật True.")
else:
    cmd_train = [
        sys.executable,
        str(det_script),
        "--mode", "train",
        "--arch", ARCH,
        "--seed", str(SEED),
        "--epochs", str(EPOCHS),
        "--imgsz", str(IMGSZ),
        "--det-root", str(DET_ROOT),
        "--yolo-work-dir", str(YOLO_DATA_DIR),
        "--output-dir", str(out_dir),
        "--cu-per-hour", str(OBSERVED_CU_PER_HOUR),
        "--budget-category", BUDGET_CATEGORY,
        "--resume",
    ]
    res = subprocess.run(cmd_train, text=True)
    if res.returncode != 0:
        raise RuntimeError("Huấn luyện Detection gặp lỗi hoặc bị chặn bởi cổng ngân sách CU.")""",
        ),
        _code_cell(
            "det_eval_test_locked",
            """# [CELL 5] Đánh giá Mô hình & Ngưỡng Đã Khoá trên Tập Test Đã Kiểm Toán (871 ảnh / 1,884 boxes)
if not RUN_LOCKED_TEST_EVAL:
    print("[DỪNG AN TOÀN] RUN_LOCKED_TEST_EVAL đang là False. Chỉ bật True sau khi Cell 4 đã hoàn tất và tạo locked_validation_threshold.json.")
else:
    test_zip = DRIVE_ROOT / "zips" / "dg38_det_test_locked.zip"
    test_coco = DET_ROOT / "test_final_candidate" / "_annotations.coco.json"
    if test_zip.exists() and not test_coco.exists():
        print(f"Đang giải nén gói Test Detection {test_zip} -> {DET_ROOT.parent} ...")
        with zipfile.ZipFile(test_zip, "r") as zf:
            zf.extractall(DET_ROOT.parent)

    cmd_eval_test = [
        sys.executable,
        str(det_script),
        "--mode", "eval-test",
        "--arch", ARCH,
        "--seed", str(SEED),
        "--imgsz", str(IMGSZ),
        "--det-root", str(DET_ROOT),
        "--yolo-work-dir", str(YOLO_DATA_DIR),
        "--output-dir", str(out_dir),
    ]
    res = subprocess.run(cmd_eval_test, text=True)
    if res.returncode != 0:
        raise RuntimeError("Đánh giá trên tập Test Detection gặp lỗi.")""",
        ),
    ]
    return _build_notebook(cells)


def create_03_finetune_ablation() -> Dict[str, Any]:
    cells = [
        _md_cell(
            "ft_title",
            """# 03 — DG38 DINOv2 ViT-B/14 Fine-Tune Blocks Ablation (`0`, `2`, `4`, `12` Blocks)

**Mục tiêu (Giai đoạn 6 — Hạn mức đề xuất: 15 CU):**
So sánh định lượng ảnh hưởng của số block Transformer cuối được mở khoá (unfreeze) trên cùng mô hình **DINOv2 ViT-B/14** (`facebook/dinov2-base`), cùng head `LayerNorm(768) -> Dropout(0.2) -> Linear(768, 10)`, cùng manifest cố định (`9,807` Train / `1,226` Val / `1,226` Test):
- **0 blocks (Linear Probe)**: Đóng băng toàn bộ backbone, chỉ train classification head + layernorm (`9,226` tham số, `0.01%`).
- **2 blocks cuối**: Mở khoá `encoder.layer.10..11` + `layernorm` (`14,184,970` tham số, `16.38%`).
- **4 blocks cuối (Cấu hình chính của đồ án)**: Mở khoá `encoder.layer.8..11` + `layernorm` (`28,360,714` tham số, `32.75%`).
- **12 blocks (Full Fine-tune)**: Mở khoá toàn bộ 12 block Transformer (`86,589,706` tham số, `100.00%`) *(chỉ chạy nếu dự báo CU nằm trong hạn mức)*.

Ghi nhận đầy đủ: **Trainable params, % params, thời gian/epoch, Peak VRAM, Validation Macro F1, Test Accuracy & Test Macro F1**.""",
        ),
        _code_cell(
            "ft_setup",
            """# [CELL 1] Thiết lập tham số Fine-Tune Blocks Ablation
import os
import sys
import subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    if not Path("/content/drive/MyDrive").exists():
        drive.mount("/content/drive")

DRIVE_ROOT = Path(os.environ.get("DG38_DRIVE_ROOT", "/content/drive/MyDrive/DG38_Colab_Package"))
REPO_ROOT = DRIVE_ROOT / "code" if (DRIVE_ROOT / "code" / "model.py").exists() else Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXP_DIR = REPO_ROOT / "evaluation_runs" / "dg38_additional_experiments"
MANIFEST_PATH = EXP_DIR / "manifests" / "classification_train_val_manifest.json"
DATA_ROOT = Path(os.environ.get("DG38_CLS_DATA_ROOT", "/content/dg38_data/classification/standardized_256"))
ARTIFACTS_DIR = (DRIVE_ROOT / "artifacts") if DRIVE_ROOT.exists() else EXP_DIR

BLOCKS_TO_RUN = [0, 2, 4]        # Có thể thêm 12: [0, 2, 4, 12] nếu dự báo CU cho phép
SEED = 42
EPOCHS = 15
OBSERVED_CU_PER_HOUR = None      # <-- BẮT BUỘC: Nhập Usage Rate thực tế từ giao diện Colab (ví dụ: 1.48)
RUN_FULL_ABLATION = False        # Đổi thành True sau khi kiểm tra bảng dự báo CU ở Cell 2

if OBSERVED_CU_PER_HOUR is None or float(OBSERVED_CU_PER_HOUR) <= 0:
    raise ValueError("Vui lòng nhập OBSERVED_CU_PER_HOUR (Usage Rate từ 'Runtime -> View resources') trước khi chạy!")

print(f"DATA_ROOT     : {DATA_ROOT} (exists={DATA_ROOT.exists()})")
print(f"BLOCKS_TO_RUN : {BLOCKS_TO_RUN}")""",
        ),
        _code_cell(
            "ft_pilot_forecast",
            """# [CELL 2] Chạy Pilot cho cấu hình nặng nhất trong BLOCKS_TO_RUN để dự báo CU
trainer_script = EXP_DIR / "dg38_cls_trainer.py"
max_b = max(BLOCKS_TO_RUN)
pilot_out_dir = ARTIFACTS_DIR / "runs" / f"ablation_dinov2_vitb14_blocks{max_b}_seed{SEED}"

if DATA_ROOT.exists():
    cmd_pilot = [
        sys.executable,
        str(trainer_script),
        "--mode", "pilot",
        "--arch", "dinov2_vitb14",
        "--unfreeze-blocks", str(max_b),
        "--seed", str(SEED),
        "--epochs", str(EPOCHS),
        "--data-root", str(DATA_ROOT),
        "--manifest", str(MANIFEST_PATH),
        "--output-dir", str(pilot_out_dir),
        "--cu-per-hour", str(OBSERVED_CU_PER_HOUR),
        "--budget-category", "finetune_blocks_ablation",
    ]
    res = subprocess.run(cmd_pilot, text=True)
    if res.returncode != 0:
        raise RuntimeError("Pilot Fine-tune Ablation gặp lỗi.")
else:
    print(f"[INFO] Không tìm thấy {DATA_ROOT}. Hãy giải nén dg38_cls_train_val.zip ở Notebook 00 trước.")""",
        ),
        _code_cell(
            "ft_run_ablation",
            """# [CELL 3] Thực thi huấn luyện lần lượt các cấu hình Fine-tune Blocks & Lưu bảng tổng hợp
if not RUN_FULL_ABLATION:
    print("[DỪNG AN TOÀN] RUN_FULL_ABLATION đang là False. Hãy kiểm tra dự báo CU ở Cell 2 trước.")
else:
    for b in BLOCKS_TO_RUN:
        out_dir = ARTIFACTS_DIR / "runs" / f"ablation_dinov2_vitb14_blocks{b}_seed{SEED}"
        cmd_train = [
            sys.executable,
            str(trainer_script),
            "--mode", "train",
            "--arch", "dinov2_vitb14",
            "--unfreeze-blocks", str(b),
            "--seed", str(SEED),
            "--epochs", str(EPOCHS),
            "--data-root", str(DATA_ROOT),
            "--manifest", str(MANIFEST_PATH),
            "--output-dir", str(out_dir),
            "--cu-per-hour", str(OBSERVED_CU_PER_HOUR),
            "--budget-category", "finetune_blocks_ablation",
            "--resume",
        ]
        print(f"=== ĐANG HUẤN LUYỆN DINOv2 ViT-B/14 (UNFREEZE_BLOCKS={b}) ===")
        res = subprocess.run(cmd_train, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Lỗi khi chạy cấu hình blocks={b}")""",
        ),
    ]
    return _build_notebook(cells)


def create_04_crop_ablation() -> Dict[str, Any]:
    cells = [
        _md_cell(
            "crop_title",
            """# 04 — DG38 Crop Preprocessing Ablation cho Cascade Detection -> Classification (Variants A–E)

**Mục tiêu (Giai đoạn 7 — Hạn mức đề xuất: 15 CU):**
- Giữ cố định **Detector (`detection/checkpoints_wastedetect/best_model.pt`)**, **Classifier (`checkpoints/dinov2_garbage_final.pth`)** và toàn bộ **`64,819` NMS proposals (`nms_iou=0.70`)** đã cache trong `evaluation_runs/val_grid_search_01/cache/predictions.sqlite3` trên **Clean Validation (`1,743` ảnh / `3,932` GT boxes)**.
- Tự động xác minh SHA-256 fingerprint của bộ cache gốc (`fingerprint.json`) trước khi chạy.
- So sánh 5 phương án tiền xử lý crop trước khi đưa vào DINOv2:
  - `A_original_crop`: Tight crop -> `Resize(256, 256)` -> `CenterCrop(224)` *(khớp bit-exact 100% với Validation gốc: `mAP@50 = 0.4399`, `76.63%` acc trên `3,663` matched GT boxes, `54.02%` `glass -> plastic`, `29.76%` `paper -> cardboard`)*.
  - `B_direct_resize`: Tight crop -> `Resize(224, 224)` (không cắt mất 23.4% diện tích viền).
  - `C_context_pad_15`: Mở rộng bbox thêm +15% mỗi cạnh -> `Resize(256, 256)` -> `CenterCrop(224)`.
  - `D_letterbox_224`: Tight crop -> Đệm vuông bảo toàn tỷ lệ khung hình `RGB(114, 114, 114)` -> `224x224`.
  - `E_context_pad_25_letterbox`: Mở rộng bbox thêm +25% mỗi cạnh -> Đệm vuông bảo toàn tỷ lệ khung hình -> `224x224`.
- Mỗi biến thể `B..E` được lưu trong bảng `variant_crops` của bản sao SQLite (`WORK_CACHE_DB`), **tuyệt đối không sửa đổi file `predictions.sqlite3` gốc hay kết quả Final Test cũ**.""",
        ),
        _code_cell(
            "crop_setup",
            """# [CELL 1] Thiết lập đường dẫn & kiểm tra SQLite Cache + Checkpoints của Clean Validation (1,743 ảnh)
import os
import sys
import subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    if not Path("/content/drive/MyDrive").exists():
        drive.mount("/content/drive")

DRIVE_ROOT = Path(os.environ.get("DG38_DRIVE_ROOT", "/content/drive/MyDrive/DG38_Colab_Package"))
REPO_ROOT = DRIVE_ROOT / "code" if (DRIVE_ROOT / "code" / "model.py").exists() else Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXP_DIR = REPO_ROOT / "evaluation_runs" / "dg38_additional_experiments"
ARTIFACTS_DIR = (DRIVE_ROOT / "artifacts") if DRIVE_ROOT.exists() else EXP_DIR

SOURCE_CACHE_DB = REPO_ROOT / "evaluation_runs" / "val_grid_search_01" / "cache" / "predictions.sqlite3"
WORK_CACHE_DB = Path("/content/dg38_data/crop_ablation_predictions.sqlite3")
VALID_CLEAN_DIR = Path(os.environ.get("DG38_VALID_CLEAN_DIR", "/content/dg38_data/detection/WasteDetect_Dataset/valid_clean_candidate"))
VALID_COCO = VALID_CLEAN_DIR / "_annotations.coco.json"
DET_CKPT = REPO_ROOT / "detection" / "checkpoints_wastedetect" / "best_model.pt"
CLS_CKPT = REPO_ROOT / "checkpoints" / "dinov2_garbage_final.pth"

OBSERVED_CU_PER_HOUR = None      # <-- BẮT BUỘC: Nhập Usage Rate thực tế từ giao diện Colab (ví dụ: 1.48)
RUN_FULL_CROP_ABLATION = False   # Đổi thành True sau khi kiểm tra Pilot ở Cell 2 đạt APPROVED

if OBSERVED_CU_PER_HOUR is None or float(OBSERVED_CU_PER_HOUR) <= 0:
    raise ValueError("Vui lòng nhập OBSERVED_CU_PER_HOUR (Usage Rate từ 'Runtime -> View resources') trước khi chạy!")

print(f"SOURCE_CACHE_DB : {SOURCE_CACHE_DB} (exists={SOURCE_CACHE_DB.exists()})")
print(f"VALID_COCO      : {VALID_COCO} (exists={VALID_COCO.exists()})")
print(f"DET_CKPT        : {DET_CKPT} (exists={DET_CKPT.exists()})")
print(f"CLS_CKPT        : {CLS_CKPT} (exists={CLS_CKPT.exists()})")""",
        ),
        _code_cell(
            "crop_run_ablation",
            """# [CELL 2] Chạy Pilot (đo tốc độ trên 25 ảnh) hoặc Toàn Bộ 5 Biến Thể Crop Preprocessing (A, B, C, D, E) trên Clean Validation
crop_script = EXP_DIR / "dg38_crop_ablation.py"
out_dir = ARTIFACTS_DIR / "runs" / "crop_ablation_val_clean"
mode = "run" if RUN_FULL_CROP_ABLATION else "pilot"

cmd = [
    sys.executable,
    str(crop_script),
    "--mode", mode,
    "--valid-coco", str(VALID_COCO),
    "--valid-images", str(VALID_CLEAN_DIR),
    "--det-ckpt", str(DET_CKPT),
    "--cls-ckpt", str(CLS_CKPT),
    "--base-sqlite", str(SOURCE_CACHE_DB),
    "--work-sqlite", str(WORK_CACHE_DB),
    "--output-dir", str(out_dir),
    "--cu-per-hour", str(OBSERVED_CU_PER_HOUR),
]
print(f"Đang chạy Crop Ablation ở chế độ --mode {mode} ...")
if VALID_COCO.exists() and SOURCE_CACHE_DB.exists() and CLS_CKPT.exists():
    res = subprocess.run(cmd, text=True)
    if res.returncode != 0:
        raise RuntimeError("Crop Ablation gặp lỗi.")
else:
    print("[INFO] Hãy đảm bảo VALID_COCO, SOURCE_CACHE_DB và CLS_CKPT đều tồn tại (chạy Notebook 00 trước).")""",
        ),
    ]
    return _build_notebook(cells)


def create_05_results_analysis() -> Dict[str, Any]:
    cells = [
        _md_cell(
            "an_title",
            """# 05 — DG38 Results Aggregation, Three-Seed Statistics (`ddof=1`) & Completion Report

**Chức năng của Notebook này:**
1. Quét toàn bộ thư mục `runs/` để cập nhật `experiment_registry.json`.
2. Tính toán thống kê **Mean ± Sample Std (`ddof=1`)** cho từng nhóm thí nghiệm 3 seeds (`42`, `123`, `2026`).
   - Tự động kiểm tra tính đồng nhất (cùng mô hình, cùng manifest SHA-256, cùng giao thức huấn luyện).
   - Nếu nhóm nào mới có 1 hoặc 2 seeds thì tự động gắn nhãn `INCOMPLETE` (không giả mạo độ lệch chuẩn của 3 seeds).
3. Xuất bảng đối chiếu kết quả mới với kết quả gốc trong báo cáo và cập nhật trạng thái hoàn thành DG38 (`dg38_completion_status.md`, `cu_budget_report.md`).""",
        ),
        _code_cell(
            "an_aggregate",
            """# [CELL 1] Quét toàn bộ kết quả thực nghiệm & hiển thị bảng trạng thái ngân sách CU
import os
import sys
import json
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    if not Path("/content/drive/MyDrive").exists():
        drive.mount("/content/drive")

DRIVE_ROOT = Path(os.environ.get("DG38_DRIVE_ROOT", "/content/drive/MyDrive/DG38_Colab_Package"))
REPO_ROOT = DRIVE_ROOT / "code" if (DRIVE_ROOT / "code" / "model.py").exists() else Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXP_DIR = REPO_ROOT / "evaluation_runs" / "dg38_additional_experiments"
ARTIFACTS_DIR = (DRIVE_ROOT / "artifacts") if DRIVE_ROOT.exists() else EXP_DIR

from evaluation_runs.dg38_additional_experiments.dg38_stats_and_analysis import (
    scan_and_update_registry,
    aggregate_seed_group,
)
from evaluation_runs.dg38_additional_experiments.cu_budget_tracker import CUBudgetTracker

registry = scan_and_update_registry(ARTIFACTS_DIR)
print(f"Tổng số thực nghiệm đã ghi nhận trong registry: {len(registry.get('experiments', {}))}")

tracker = CUBudgetTracker(EXP_DIR / "cu_budget_ledger.json")
print(tracker.format_budget_table_markdown())""",
        ),
        _code_cell(
            "an_group_summary",
            """# [CELL 2] Phân nhóm theo kiến trúc và kiểm tra trạng thái 3-Seed (42, 123, 2026)
experiments = list(registry.get("experiments", {}).values())

groups = {}
for exp in experiments:
    task = exp.get("task", "unknown")
    model = exp.get("model_name", "unknown")
    proto = exp.get("protocol_id", "default")
    gid = f"{task}::{model}::{proto}"
    groups.setdefault(gid, []).append(exp)

for gid, runs in sorted(groups.items()):
    task = runs[0].get("task")
    metric_keys = (
        ["val_macro_f1", "test_accuracy", "test_macro_f1"]
        if task == "classification"
        else ["val_clean_mAP_50", "val_clean_mAP_50_95", "val_clean_f1"]
    )
    try:
        agg = aggregate_seed_group(gid, runs, metric_keys=metric_keys)
        print(f"\\n=== NHÓM: {gid} | Trạng thái: {agg['overall_status']} ({agg['n_completed_seeds']}/3 seeds) ===")
        for mk, mstat in agg["metrics_summary"].items():
            print(f"  - {mk:22s}: {mstat['formatted']}")
    except ValueError as err:
        print(f"\\n[LỖI ĐỒNG NHẤT NHÓM {gid}]: {err}")""",
        ),
    ]
    return _build_notebook(cells)


def main() -> None:
    out_dir = Path("evaluation_runs/dg38_additional_experiments")
    out_dir.mkdir(parents=True, exist_ok=True)

    notebooks = {
        "00_DG38_Preflight_Colab.ipynb": create_00_preflight(),
        "01_Classification_Baselines_Colab.ipynb": create_01_classification(),
        "02_Detection_Baselines_Colab.ipynb": create_02_detection(),
        "03_DINOv2_Finetune_Ablation_Colab.ipynb": create_03_finetune_ablation(),
        "04_Crop_Preprocessing_Ablation_Colab.ipynb": create_04_crop_ablation(),
        "05_DG38_Results_Analysis_Colab.ipynb": create_05_results_analysis(),
    }

    for filename, nb_dict in notebooks.items():
        path = out_dir / filename
        path.write_text(json.dumps(nb_dict, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[OK] Generated {path} ({len(nb_dict['cells'])} cells)")


if __name__ == "__main__":
    main()
