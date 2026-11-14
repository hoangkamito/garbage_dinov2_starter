# Thực Nghiệm Ablation Tiền Xử Lý Crop Cho Cascade Detection -> Classification (Variants A–E) — DG38 Stage 7

## 1. Thiết Kế Thực Nghiệm & Tối Ưu Hoá Bằng SQLite Cache

- **Tập dữ liệu đánh giá:** `WasteDetect_Dataset/valid_clean_candidate` (**`1,743` ảnh / `3,932` GT boxes**, SHA-256: `9faa0cc8aade9802f63905c7efe9a8c10f7118d866b186d46f2fd808c3464de8`).
- **Tái sử dụng Proposals & Xác minh Fingerprint (`predictions.sqlite3` + `fingerprint.json`):**
  - Toàn bộ **`64,819` NMS proposals (`nms_iou=0.70`, trung bình `37.19` proposals/ảnh)** của **Deformable DETR (`detection/checkpoints_wastedetect/best_model.pt`)** và kết quả phân loại của **Variant A (`A_original_crop`)** trên `1,743` ảnh Clean Validation đã được lưu sẵn trong `evaluation_runs/val_grid_search_01/cache/predictions.sqlite3`.
  - Trước khi chạy, `verify_base_cache_fingerprint()` tự động kiểm tra khớp tuyệt đối 4 mã băm SHA-256 trong `fingerprint.json`:
    1. `validation_coco_sha256`: `9faa0cc8aade9802f63905c7efe9a8c10f7118d866b186d46f2fd808c3464de8`
    2. `det_ckpt_sha256` (`detection/checkpoints_wastedetect/best_model.pt`): `218aa21101d27ffffe43b827300c8769e168a9d8998b792f1dacbb6548a49556`
    3. `cls_ckpt_sha256` (`checkpoints/dinov2_garbage_final.pth`): `bee29bc7364fa18d374e68bee7f435282917fd3d98b2a83dbca7dffe53e3ec75`
    4. `class_mapping_sha256` (`evaluation_runs/class_mapping.json`): `640f73a90c42efe67ce27bbffbe5ef18e3b9cb3cd72a8ca6ebbc532c2fc67c08`
  - **Không sửa đổi DB gốc:** `dg38_crop_ablation.py` sao chép `predictions.sqlite3` sang file làm việc riêng (`WORK_CACHE_DB`) và tạo bảng `variant_crops` lưu kết quả riêng theo khoá `(variant, image_id, x1, y1, x2, y2)` cho từng biến thể `B..E`.

---

## 2. Định Nghĩa 5 Phương Án Tiền Xử Lý Crop (Variants A–E)

| Mã Phương Án | Tên Biến Thể (`VARIANT_DEFINITIONS`) | Mô Tả Toán Học & Biến Đổi Ảnh | Giả Thuyết Khoa Học |
|:---:|---|---|---|
| **A** | `A_original_crop` | Cắt sát bounding box $[x_1, y_1, x_2, y_2]$ -> `Resize((256, 256))` -> `CenterCrop(224)` | Baseline hiện tại của đồ án; dễ làm méo tỷ lệ vật thể thon dài (chai nhựa, ống hút) và mất ~23.4% diện tích viền do `CenterCrop(224)` từ `256`. |
| **B** | `B_direct_resize` | Cắt sát bounding box $[x_1, y_1, x_2, y_2]$ -> `Resize((224, 224))` (bỏ `CenterCrop`) | Giữ lại 100% diện tích bên trong bounding box để kiểm tra xem việc cắt mất 23.4% rìa ảnh ở Variant A có làm mất viền cổ chai/nắp chai hay không. |
| **C** | `C_context_pad_15` | Mở rộng mỗi cạnh bounding box thêm **+15% context** ($0.15 \times w, 0.15 \times h$, kẹp trong biên ảnh) -> `Resize((256, 256))` -> `CenterCrop(224)` | Bù lại phần rìa vật thể bị cắt bởi `CenterCrop(224)` và cung cấp thêm ngữ cảnh bề mặt xung quanh cho vật thể trong suốt (`glass` vs `plastic`). |
| **D** | `D_letterbox_224` | Cắt sát bounding box -> Đệm vuông bảo toàn tỷ lệ khung hình `RGB(114, 114, 114)` -> `Resize((224, 224))` | **Giữ nguyên 100% tỷ lệ khung hình (aspect ratio)** và không cắt mất rìa vật thể. |
| **E** | `E_context_pad_25_letterbox` | Mở rộng **+25% context** mỗi cạnh -> Đệm vuông bảo toàn tỷ lệ khung hình `RGB(114, 114, 114)` -> `Resize((224, 224))` | Kết hợp cả 2 ưu điểm: có ngữ cảnh nền +25% xung quanh vật thể và bảo toàn tuyệt đối tỷ lệ hình học. |

---

## 3. Bảng Kết Quả Trên Clean Validation (`1,743` Ảnh / `3,932` GT Boxes)

> **Xác minh thực nghiệm:** Kết quả của **Variant A (`A_original_crop`)** bên dưới đã được chạy kiểm chứng trực tiếp bằng lệnh `dg38_crop_ablation.py --mode run --variants A_original_crop` trên `predictions.sqlite3`, tái sử dụng `64,819 / 64,819` NMS proposals (`detector_runs = 0`, `new_crop_evals = 0`) và khớp tuyệt đối (`diff = 0.0`) với `evaluation_runs/val_grid_search_01/summary.csv` & `cascade_error_analysis.md`.

| Variant | Cascade+NMS mAP@50 (`det=0.05, nms=0.70, cls=0.75`) | Cascade+NMS mAP@50:95 | Cascade+NMS AR@100 | Cascade+NMS F1 (`@thr=0.25`) | Matched GT Boxes (`IoU>=0.50`) | Matched GT Acc (%) | Wrong In-Scope (%) | Out-of-Scope Reject (%) | Lỗi `glass -> plastic` (%) | Lỗi `paper -> cardboard` (%) | Trạng thái |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **A (`A_original_crop` — Baseline)** | **0.4399** (`0.439864`) | **0.2477** (`0.247715`) | **0.4169** (`0.416937`) | **0.5530** (`0.552997`) | **3,663** | **76.63%** (`2,807/3,663`) | **11.47%** (`420/3,663`) | **11.90%** (`436/3,663`) | **54.02%** (`168/311`) | **29.76%** (`136/457`) | **VERIFIED (Bit-Exact Match)** |
| **B (`B_direct_resize`)** | *Chờ chạy Colab* | *Chờ chạy Colab* | *Chờ chạy Colab* | *Chờ chạy Colab* | `3,663` | *Chờ chạy Colab* | *Chờ chạy Colab* | *Chờ chạy Colab* | *Chờ chạy Colab* | *Chờ chạy Colab* | **NOT_STARTED** |
| **C (`C_context_pad_15`)** | *Chờ chạy Colab* | *Chờ chạy Colab* | *Chờ chạy Colab* | *Chờ chạy Colab* | `3,663` | *Chờ chạy Colab* | *Chờ chạy Colab* | *Chờ chạy Colab* | *Chờ chạy Colab* | *Chờ chạy Colab* | **NOT_STARTED** |
| **D (`D_letterbox_224`)** | *Chờ chạy Colab* | *Chờ chạy Colab* | *Chờ chạy Colab* | *Chờ chạy Colab* | `3,663` | *Chờ chạy Colab* | *Chờ chạy Colab* | *Chờ chạy Colab* | *Chờ chạy Colab* | *Chờ chạy Colab* | **NOT_STARTED** |
| **E (`E_context_pad_25_letterbox`)** | *Chờ chạy Colab* | *Chờ chạy Colab* | *Chờ chạy Colab* | *Chờ chạy Colab* | `3,663` | *Chờ chạy Colab* | *Chờ chạy Colab* | *Chờ chạy Colab* | *Chờ chạy Colab* | *Chờ chạy Colab* | **NOT_STARTED** |

---

## 4. Quy Tắc Bảo Vệ Tập Test (`test_final_candidate` — `871` Ảnh)

1. Tuyệt đối **không chạy cả 5 biến thể A–E trên tập Test** để chọn biến thể có điểm Test cao nhất.
2. Không làm thay đổi các kết quả Final Test đã khoá trong `evaluation_runs/final_test_four_way_01/final_test_report.md` (`Detector Original mAP@50 = 0.7332`, `Detector + NMS mAP@50 = 0.7801`, `Cascade Original mAP@50 = 0.4134`, `Cascade + NMS mAP@50 = 0.4486`).
