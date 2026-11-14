# Hướng Dẫn Thực Thi Bộ Thực Nghiệm Bổ Sung DG38 Trên Google Colab Pro (Ngân Sách 200.45 CU)

Tài liệu này hướng dẫn từng bước tải gói ZIP đã đóng gói sẵn lên Google Drive và vận hành 6 Notebook Colab Pro sao cho **hoàn thành tối đa các yêu cầu DG38 trong hạn mức 200.45 Compute Units (CU)**, giữ vững vùng đệm an toàn **>= 40.00 CU**.

---

## 1. Kiến Trúc Thư Mục Trên Google Drive

Hãy tạo một thư mục gốc trên Google Drive có tên `DG38_Colab_Package` và thư mục con `zips` bên trong (`/content/drive/MyDrive/DG38_Colab_Package/zips`), sau đó tải các gói ZIP từ `evaluation_runs/dg38_additional_experiments/colab_packages/` lên:

```text
/content/drive/MyDrive/DG38_Colab_Package/
├── zips/                                           # Tải các file ZIP lên thư mục này
│   ├── dg38_source_code_and_manifests.zip          # Mã nguồn + 6 Notebooks + Manifests SHA-256
│   ├── dg38_reference_checkpoints.zip              # checkpoints/dinov2_garbage_final.pth + detection/checkpoints_wastedetect/best_model.pt
│   ├── dg38_crop_ablation_val_bundle.zip           # predictions.sqlite3 + fingerprint.json + locked_parameters.json
│   ├── dg38_cls_train_val.zip                      # 9,807 Train + 1,226 Val (standardized_256/...)
│   ├── dg38_cls_test_locked.zip                    # 1,226 Test (tách riêng, chỉ mở khi đánh giá cuối)
│   ├── dg38_det_train_val_clean.zip                # 8,289 Train + 1,743 Clean Val (WasteDetect_Dataset/...)
│   └── dg38_det_test_locked.zip                    # 871 Test đã kiểm toán (tách riêng, chỉ mở khi đánh giá cuối)
├── code/                                           # Notebook 00 tự động giải nén code + checkpoints + cache vào đây
│   ├── model.py
│   ├── checkpoints/dinov2_garbage_final.pth
│   ├── detection/checkpoints_wastedetect/best_model.pt
│   └── evaluation_runs/
│       ├── val_grid_search_01/cache/predictions.sqlite3
│       └── dg38_additional_experiments/
│           ├── 00_DG38_Preflight_Colab.ipynb
│           ├── 01_Classification_Baselines_Colab.ipynb
│           ├── 02_Detection_Baselines_Colab.ipynb
│           ├── 03_DINOv2_Finetune_Ablation_Colab.ipynb
│           ├── 04_Crop_Preprocessing_Ablation_Colab.ipynb
│           ├── 05_DG38_Results_Analysis_Colab.ipynb
│           └── manifests/
└── artifacts/                                      # Tự động lưu kết quả runs/ trên Google Drive
```

> **Lưu ý tiết kiệm thời gian I/O trên Colab:** Không đọc trực tiếp 10,000+ file ảnh rời từ Google Drive FUSE trong lúc train. Notebook `00_DG38_Preflight_Colab.ipynb` giải nén `dg38_cls_train_val.zip` và `dg38_det_train_val_clean.zip` từ Drive sang ổ SSD cục bộ `/content/dg38_data` chỉ mất ~15–25 giây.

---

## 2. Cách Chọn GPU Trên Giao Diện Colab Pro (L4 vs A100 vs G4)

Bạn chọn GPU trực tiếp trên menu Colab: **`Runtime -> Change runtime type -> Hardware accelerator`**.

| Loại GPU | Khi nào nên chọn | Nhiệm vụ phù hợp nhất |
|---|---|---|
| **NVIDIA L4 (24 GB)** | **Mặc định khuyến nghị cho ~80% tác vụ** | Preflight, Smoke test, **ResNet50 (3 seeds)**, **YOLO11s (3 seeds)**, **Crop Preprocessing Ablation (A–E)**, **DINOv2 Fine-tune Ablation (0, 2, 4 blocks)** |
| **NVIDIA A100 (40/80 GB)** | Khi cần throughput cao hoặc VRAM lớn | **DINOv2 Fine-tune 12 blocks (full)**, **3 seeds DINOv2 ViT-B/14**, hoặc **Deformable DETR multi-seed** |
| **NVIDIA G4** | Chỉ khi L4/A100 thiếu bộ nhớ hoặc Pilot chứng minh hiệu quả CU/epoch tốt hơn | Kiểm tra bằng Pilot trước khi chạy full |

### Cách lấy Usage Rate (CU/giờ) thực tế từ Colab
1. Sau khi kết nối Runtime trên Colab, bấm vào góc phải trên cùng: **`Runtime -> View resources`**.
2. Đọc dòng chữ dạng: *"Available: 200.45 compute units ... Usage rate: approximately X.XX per hour"*.
3. Điền con số `X.XX` vào biến `OBSERVED_CU_PER_HOUR` và số dư hiện tại vào `OBSERVED_REMAINING_CU` trong Notebook. **Tuyệt đối không dùng ước lượng cũ thay cho Usage Rate thực tế + Pilot.**

---

## 3. Trình Tự Chạy Từng Bước Để Không Lãng Phí CU

### Bước 1: Chạy `00_DG38_Preflight_Colab.ipynb` (Trên GPU L4)
- Mount Drive, tự động giải nén `dg38_source_code_and_manifests.zip`, `dg38_reference_checkpoints.zip`, `dg38_crop_ablation_val_bundle.zip` vào `DRIVE_ROOT/code` và giải nén dữ liệu Train/Val sang `/content/dg38_data`.
- Kiểm tra SHA-256 của manifest Classification (`9,807` / `1,226` / `1,226`), Detection (`8,289` / `1,743` Clean Val) và 2 checkpoints gốc (`checkpoints/dinov2_garbage_final.pth`, `detection/checkpoints_wastedetect/best_model.pt`).
- Nhập `OBSERVED_CU_PER_HOUR` thực tế từ Colab UI và chạy Smoke Test (ResNet50 + Deformable DETR `MSDeformAttn`).

### Bước 2: Chạy Ưu Tiên Số 1 — ResNet50 Baseline (`01_Classification_Baselines_Colab.ipynb`, GPU L4)
- Giữ `ARCH = "resnet50"`, `SEED = 42`, `RUN_FULL_TRAINING = False`.
- Nhập `OBSERVED_CU_PER_HOUR` từ Colab UI vào Cell 1.
- Chạy Cell 1 & Cell 2 để đo tốc độ thực tế 20 batches và xuất bảng **Kiểm Tra Cổng Ngân Sách (`pilot_forecast.json`)**.
- Khi thấy trạng thái **`approved: true`**, đổi `RUN_FULL_TRAINING = True` ở Cell 1 và chạy Cell 3 để hoàn tất 15 epochs trên Train/Val. Sau khi xong, bật `RUN_LOCKED_TEST_EVAL = True` và chạy Cell 4 để chấm trên tập Test đã khoá (`1,226` ảnh).

### Bước 3: Chạy Ưu Tiên Số 2 — YOLO11s Baseline (`02_Detection_Baselines_Colab.ipynb`, GPU L4)
- Giữ `ARCH = "yolo11s"`, `SEED = 42`, `RUN_FULL_TRAINING = False`.
- Chạy Cell 2 để chuyển đổi COCO -> YOLO trên `train` (`8,289` ảnh / `18,345` boxes) và `valid_clean_candidate` (`1,743` ảnh / `3,932` boxes sạch).
- Chạy Cell 3 (Pilot 10% của 1 epoch) để kiểm tra dự báo CU. Khi đạt **`approved: true`**, bật `RUN_FULL_TRAINING = True` và chạy Cell 4.

### Bước 4: Chạy Crop Preprocessing Ablation (`04_Crop_Preprocessing_Ablation_Colab.ipynb`, GPU L4)
- Tận dụng SQLite cache `predictions.sqlite3` có sẵn của `1,743` ảnh Clean Validation (`64,819` NMS proposals). Variant A tái sử dụng 100% cache gốc; các Variants B, C, D, E được tính và lưu riêng trong bảng `variant_crops` của `WORK_CACHE_DB`.

### Bước 5: Hoàn Thành 3 Seeds (`42`, `123`, `2026`) & Fine-Tune Blocks Ablation (`03`, `05`)
- Chạy thêm `SEED = 123` và `SEED = 2026` cho `resnet50` (Notebook `01`) và `yolo11s` (Notebook `02`).
- Chạy `03_DINOv2_Finetune_Ablation_Colab.ipynb` cho `BLOCKS_TO_RUN = [0, 2, 4]` (và `12` nếu dự báo CU cho phép).
- Mở `05_DG38_Results_Analysis_Colab.ipynb` để tự động tính `Mean ± Sample Std (ddof=1)` và xuất toàn bộ bảng báo cáo Markdown.
