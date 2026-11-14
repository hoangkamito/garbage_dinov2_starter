# Thực Nghiệm Ablation Số Block Fine-Tune Cho DINOv2 ViT-B/14 (`0`, `2`, `4`, `12` Blocks) — DG38 Stage 6

## 1. Thiết Kế Thực Nghiệm Chuẩn Hoá

- **Mô hình cơ sở:** `facebook/dinov2-base` (ViT-B/14, 12 Transformer Encoder layers `encoder.layer.0` đến `encoder.layer.11`, kích thước embedding `768`).
- **Classification Head cố định:** `LayerNorm(768) -> Dropout(0.2) -> Linear(768, 10)` (tương thích 100% với `model.py` và `Copy_of_Trainning_DINOv2.ipynb`).
- **Tập dữ liệu cố định (Manifest SHA-256 `d447f35ca8a4f189...`):**
  - Train: `9,807` ảnh | Validation: `1,226` ảnh | Test: `1,226` ảnh.
- **Tiêu chí chọn checkpoint:** Validation Macro F1 cao nhất trong 15 epochs.
- **Hạn mức ngân sách của nhóm `finetune_blocks_ablation`:** **15.00 CU**.

---

## 2. Bảng Phân Tích Cấu Hình & Kết Quả Ablation

| Cấu hình Ablation | Các tầng được mở khoá (Unfrozen Layers) | Tổng tham số | Tham số huấn luyện (Trainable Params) | Tỷ lệ tham số huấn luyện (%) | Thời gian / Epoch (s) | Peak VRAM (GB) | Best Val Macro F1 | Test Accuracy (%) | Test Macro F1 (%) | Trạng thái |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **0 Blocks (Linear Probe)** | Chỉ `layernorm` + `classifier` head | 86,589,706 | 9,226 | **0.01%** | *Chờ chạy Colab* | *Chờ chạy Colab* | *Chờ chạy Colab* | *Chờ chạy Colab* | *Chờ chạy Colab* | **NOT_STARTED** |
| **2 Blocks cuối** | `encoder.layer.10..11` + `layernorm` + `classifier` | 86,589,706 | 14,184,970 | **16.38%** | *Chờ chạy Colab* | *Chờ chạy Colab* | *Chờ chạy Colab* | *Chờ chạy Colab* | *Chờ chạy Colab* | **NOT_STARTED** |
| **4 Blocks cuối (Cấu hình chính)** | `encoder.layer.8..11` + `layernorm` + `classifier` | 86,589,706 | 28,360,714 | **32.75%** | *Đã có Ref / Chờ đo đồng nhất trên L4* | *Chờ đo trên L4* | *Đã khoá Ref* | **98.69%** *(Ref)* | **98.43%** *(Ref)* | **COMPLETED (1-Seed Ref) / Chờ đo đồng nhất tốc độ** |
| **12 Blocks (Full Fine-tune)** | Toàn bộ `embeddings` + `encoder.layer.0..11` + `layernorm` + `classifier` | 86,589,706 | 86,589,706 | **100.00%** | *Chờ chạy Colab (nếu đủ CU)* | *Chờ chạy Colab* | *Chờ chạy Colab* | *Chờ chạy Colab* | *Chờ chạy Colab* | **NOT_STARTED** |

---

## 3. Hướng Dẫn Thực Thi Trên Colab Pro

1. Mở notebook `03_DINOv2_Finetune_Ablation_Colab.ipynb` trên GPU **NVIDIA L4** (hoặc **A100** nếu muốn chạy cả cấu hình `12 blocks`).
2. Ở Cell 1, danh sách mặc định là `BLOCKS_TO_RUN = [0, 2, 4]`.
3. Chạy Cell 2 để đo tốc độ thực tế và kiểm tra bảng **Pre-Experiment Budget Gate**. Nếu tổng CU dự báo cho cả `[0, 2, 4, 12]` nằm gọn trong hạn mức `15.00 CU` của nhóm `finetune_blocks_ablation`, bạn có thể thêm `12` vào `BLOCKS_TO_RUN`.
4. Đổi `RUN_FULL_ABLATION = True` và chạy Cell 3. Kết quả sẽ tự động lưu vào `finetune_ablation_summary.json` và cập nhật sổ cái `cu_budget_ledger.json`.
