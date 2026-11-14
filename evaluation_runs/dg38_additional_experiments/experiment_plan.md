# Kế Hoạch Thực Nghiệm Tổng Thể DG38 — Quản Trị Ngân Sách 200.45 CU

## 1. Bối Cảnh & Kiểm Toán Hiện Trạng (Giai Đoạn 0)

Qua kiểm tra toàn bộ mã nguồn, báo cáo kiểm toán dữ liệu và các checkpoint hiện có trong workspace `D:\Download Acer\garbage_dinov2_starter`:

### 1.1. Classification (10 lớp)
- **Nguồn dữ liệu gốc:** `standardized_256` (`12,259` ảnh RGB, 10 lớp: `battery`, `biological`, `cardboard`, `clothes`, `glass`, `metal`, `paper`, `plastic`, `shoes`, `trash`).
- **Chia tập cố định (tái dựng chính xác 100% từ `Copy_of_Trainning_DINOv2.ipynb`):**
  - `train_test_split(test_size=0.20, random_state=42, stratify=targets)` -> `train_test_split(test_size=0.50, random_state=42, stratify=targets[temp_idx])`.
  - **Train:** `9,807` ảnh | **Validation:** `1,226` ảnh | **Test:** `1,226` ảnh.
  - **SHA-256 Split Signature:** `1da7e63c9ee27d1831b3595350460321897565b0375ef33da0aea1ec0e5660c7`.
  - **SHA-256 Manifest Train+Val (portable):** `cd27f0e69e70691c0ae614e5d0b86dd4b84ec4636c823ae7e1539b6e446995b5`.
  - **SHA-256 Manifest Test (portable):** `7615b4fa1b67f0dd5d67b2e9ab7fa59f1602ea09a8352a3e5b07ef9c506fe86e`.
- **Giao thức huấn luyện DINOv2 ViT-B/14 gốc (`checkpoints/dinov2_garbage_final.pth`, `330.4 MB`, SHA-256 `bee29bc7364fa18d374e68bee7f435282917fd3d98b2a83dbca7dffe53e3ec75`):**
  - Kiến trúc: `facebook/dinov2-base` (embedding 768) + `LayerNorm(768) -> Dropout(0.2) -> Linear(768, 10)`.
  - Giai đoạn 1 (Linear Probe): 10 epochs, đóng băng toàn bộ backbone, `AdamW(lr=1e-3, weight_decay=1e-4)`, `CosineAnnealingLR`.
  - Giai đoạn 2 (Fine-tuning): 5 epochs, mở khoá **4 block Transformer cuối (`encoder.layer.8..11`) + `layernorm`**, `AdamW(backbone_lr=1e-5, head_lr=1e-4, weight_decay=1e-4)`.
  - Kết quả Test gốc (1 seed = 42): **Accuracy = 98.69%**, **Macro F1 = 98.43%**.

### 1.2. Detection & Cascade (5 lớp COCO ID 0–4)
- **Nguồn dữ liệu (`detection/WasteDetect_Dataset`):**
  - **Train gốc:** `8,289` ảnh / `18,345` boxes (COCO JSON SHA-256: `a19eec883ccd8bfd76248577275ae15b01aa1c14878507fe2cda59fc4226c2eb`).
  - **Validation gốc (`valid`):** Tổng cộng **`2,316` ảnh (`5,110` boxes)**. Theo hồ sơ kiểm toán `validation_data_audit.json` và `valid_clean_candidate/cleaning_report.json`, tập `valid` gốc đã được lọc qua 2 giai đoạn để loại bỏ tổng cộng **`573` ảnh** bị trùng lặp/rò rỉ:
    - **Giai đoạn 1 (Trùng khớp chính xác SHA-256):** Loại bỏ **`543` ảnh** (gồm **`506` ảnh trùng SHA-256 với tập `train`** và **`37` ảnh trùng SHA-256 nội bộ trong `valid`**), còn lại `1,773` ảnh (`valid_sha_clean_candidate`).
    - **Giai đoạn 2 (Trùng lặp thị giác Perceptual Hash `pHash <= 6`, rà soát thủ công `240` cặp nghi ngờ, xác nhận `36` cặp trùng):** Loại bỏ thêm **`30` ảnh** (gồm **`28` ảnh trùng pHash với tập `train`** và **`2` ảnh trùng pHash nội bộ trong `valid`**), còn lại **`1,743` ảnh sạch** (`valid_clean_candidate`).
    - Đồng thời loại bỏ `1` box trùng tọa độ hình học trên ảnh hợp lệ (`annotation_1847`), thu được tập Validation sạch cuối cùng gồm **`1,743` ảnh / `3,932` boxes**.
  - **Validation sạch (`valid_clean_candidate`):** `1,743` ảnh / `3,932` boxes (COCO JSON SHA-256: `9faa0cc8aade9802f63905c7efe9a8c10f7118d866b186d46f2fd808c3464de8`). Dùng cho toàn bộ việc chọn checkpoint, chọn threshold và chạy Crop Preprocessing Ablation.
  - **Test đã kiểm toán (`test_final_candidate`):** `871` ảnh / `1,884` boxes (COCO JSON SHA-256: `e26ff743f2a94ef9a6069a95cf3c3c69581e43295ff48bb5298b301bfde3eec1`, được làm sạch từ `1,158` ảnh `test` gốc bằng cách loại `259` ảnh trùng SHA-256 và `28` ảnh trùng pHash). Chỉ dùng để đánh giá cuối cùng sau khi mọi tham số đã khoá trên Validation sạch.
- **Checkpoint Detector gốc (`detection/checkpoints_wastedetect/best_model.pt`, `76.32 MB`, SHA-256 `218aa21101d27ffffe43b827300c8769e168a9d8998b792f1dacbb6548a49556`):**
  - Kiến trúc: `DINOv2DeformableDETR` (Frozen DINOv2 ViT-S/14 + Deformable DETR, `img_size=518`, `num_queries=100`, lưu ở epoch `29/30`).
  - Module `MSDeformAttn` trong `detection/dinov2_deformable_detr.py` được cài đặt bằng **pure PyTorch (`F.grid_sample`)**, đảm bảo tương thích trên L4, A100 và G4 mà không cần biên dịch CUDA extension ngoại lai.

---

## 2. Phân Bổ Ngân Sách 200.45 CU & Thứ Tự Ưu Tiên Thực Thi

| Thứ tự ưu tiên | Mã nhóm (`category`) | Nhiệm vụ cụ thể | Hạn mức trần (CU) | GPU khuyến nghị | Trạng thái đo chi phí CU |
|:---:|---|---|---:|---|---|
| **0** | `smoke_and_preflight` | Kiểm tra GPU, giải nén dữ liệu, xác minh SHA-256 manifest, smoke test forward/backward | **10.00 CU** | NVIDIA L4 | **UNVERIFIED_ESTIMATE** *(Chờ nhập Usage Rate Colab + chạy Pilot)* |
| **1** | `resnet50_baseline` | **Ưu tiên #1:** Huấn luyện ResNet50 pretrained ImageNet (Seed 42, 15 epochs) trên tập Classification cố định | **15.00 CU** | NVIDIA L4 | **UNVERIFIED_ESTIMATE** *(Bắt buộc đo Pilot 20 batches trước khi train)* |
| **2** | `yolo11s_baseline` | **Ưu tiên #2:** Huấn luyện YOLO11s (Seed 42, 30 epochs) trên Detection Train (`8,289`) + Clean Val (`1,743`), đánh giá bằng `COCOeval` | **35.00 CU** | NVIDIA L4 | **UNVERIFIED_ESTIMATE** *(Bắt buộc đo Pilot 10% epoch trước khi train)* |
| **3** | `crop_preprocessing_ablation` | Ablation 5 phương án Crop (A, B, C, D, E) trên `1,743` ảnh Clean Val bằng cách tái sử dụng SQLite cache `predictions.sqlite3` | **15.00 CU** | NVIDIA L4 | **UNVERIFIED_ESTIMATE** *(Bắt buộc đo Pilot 25 ảnh trước khi chạy full)* |
| **4** | `three_seed_experiments` | Hoàn thành 3 seeds (`42`, `123`, `2026`) theo thứ tự: (1) ResNet50 -> (2) YOLO11s -> (3) DINOv2 ViT-B/14 (4 blocks) -> (4) Deformable DETR (nếu còn ngân sách) | **60.00 CU** | NVIDIA L4 / A100 | **UNVERIFIED_ESTIMATE** *(Đo qua Pilot từng kiến trúc)* |
| **5** | `finetune_blocks_ablation` | Ablation mở khoá `0`, `2`, `4`, `12` blocks trên DINOv2 ViT-B/14 (cùng head, cùng manifest) | **15.00 CU** | NVIDIA L4 / A100 | **UNVERIFIED_ESTIMATE** *(Bắt buộc đo Pilot trước khi train)* |
| **6** | `reserve_and_expansion` | Quỹ dự phòng an toàn (giữ cứng **>= 40.00 CU** không tự động chạm vào; phần dư 10.45 CU chỉ dùng khi được người dùng phê duyệt) | **50.45 CU** | - | **LOCKED_RESERVE (Giữ >= 40.00 CU)** |
| **TỔNG** | | **Ngân sách tổng cộng** | **200.45 CU** | | |

---

## 3. Cơ Chế Bảo Vệ Ngân Sách & Tính Khoa Học

1. **Công thức dự báo bắt buộc trước mỗi lần chạy:**
   $$\text{estimated\_CU} = \text{estimated\_GPU\_hours} \times \text{observed\_CU\_per\_hour}$$
   Trong đó `observed_CU_per_hour` bắt buộc do người dùng đọc từ giao diện Google Colab (`Runtime -> View resources`) và $\text{estimated\_GPU\_hours}$ được đo thực nghiệm từ chế độ `--mode pilot` (bao gồm forward/backward, validation, lưu checkpoint và hệ số overhead $1.10\times$). Mọi con số ước tính sơ bộ trước khi chạy Pilot (ví dụ `0.6–1.2 CU` cho ResNet50) chỉ mang tính tham khảo lý thuyết (`UNVERIFIED_ESTIMATE`), tuyệt đối không được coi là số đo thực tế.
2. **Cổng phê duyệt tự động (`check_Experiment_gate`):**
   - Tự động chặn (`approved = False`) nếu chi phí dự báo vượt số dư của nhóm (`category_remaining_cu`) hoặc làm tổng ngân sách còn lại tụt xuống dưới mức sàn an toàn **40.00 CU**.
3. **Chống rò rỉ tập Test (Zero Test Leakage):**
   - Toàn bộ việc chọn epoch tốt nhất, chọn ngưỡng confidence cho YOLO11s, chọn số block fine-tune của DINOv2 và chọn phương án Crop (A–E) đều thực hiện **100% trên Validation (`1,226` ảnh Classification hoặc `1,743` ảnh Clean Validation Detection)**.
   - Tập Test (`1,226` ảnh Classification và `871` ảnh Detection) được đóng gói vào file ZIP riêng biệt (`dg38_cls_test_locked.zip`, `dg38_det_test_locked.zip`) và chỉ mở khoá ở bước đánh giá cuối cùng.
