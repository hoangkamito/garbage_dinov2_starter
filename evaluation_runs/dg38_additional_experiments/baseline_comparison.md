# Báo Cáo Đối Chiếu Baselines (Classification & Detection) — DG38

> **Trạng thái hiện tại:** Đã hoàn tất kiểm toán kết quả gốc và chuẩn bị đầy đủ pipeline huấn luyện/đánh giá chuẩn hoá trên Google Colab Pro. Các ô `NOT_STARTED` sẽ được tự động điền thông qua `05_DG38_Results_Analysis_Colab.ipynb` ngay sau khi từng run trên Colab hoàn tất.

---

## 1. Bảng So Sánh Classification Baselines (10 Lớp)

**Tập dữ liệu cố định:** Train = `9,807` ảnh | Validation = `1,226` ảnh | Test = `1,226` ảnh  
**Split Signature SHA-256:** `1da7e63c9ee27d1831b3595350460321897565b0375ef33da0aea1ec0e5660c7`  
**Manifest SHA-256 (Train+Val, portable):** `cd27f0e69e70691c0ae614e5d0b86dd4b84ec4636c823ae7e1539b6e446995b5`  
**Tiêu chí chọn checkpoint:** Validation Macro F1 cao nhất.

| Mô hình | Đường dẫn Checkpoint / Pretraining | Tổng tham số (M) | Tham số huấn luyện (M / %) | Giao thức huấn luyện | Val Macro F1 | Test Accuracy (%) | Test Macro F1 (%) | Test Weighted F1 (%) | Phần cứng & Thời gian | Trạng thái |
|---|---|---:|---:|---|---:|---:|---:|---:|---|---|
| **DINOv2 ViT-B/14 (Gốc — 1 Seed=42)** | `checkpoints/dinov2_garbage_final.pth` (SSL LVD-142M) | 86.59 M | 28.36 M (32.75%) | 10 ep Linear Probe (`1e-3`) + 5 ep Fine-tune 4 blocks cuối (`1e-5`/`1e-4`) | *Đã khoá* | **98.69%** | **98.43%** | **98.69%** | Tesla T4 (Colab gốc) | **COMPLETED (1-Seed Reference)** |
| **ResNet50 (Ưu tiên #1 — Seed 42)** | Supervised ImageNet-1K (`IMAGENET1K_V2`) | 23.53 M | 23.53 M (100.0%) | 15 ep Full Fine-tune, `AdamW(lr=3e-4, wd=1e-4)`, CosineAnnealingLR | *Chờ chạy Colab* | *Chờ chạy Colab* | *Chờ chạy Colab* | *Chờ chạy Colab* | NVIDIA L4 (Colab Pro) | **NOT_STARTED** |
| **ViT-B/16 (Baseline phụ — Tuỳ chọn)** | Supervised ImageNet-21K/1K | 85.81 M | 85.81 M (100.0%) | 15 ep Fine-tune, `AdamW(lr=1e-4, wd=1e-4)`, CosineAnnealingLR | *Tuỳ chọn ngân sách* | *Tuỳ chọn ngân sách* | *Tuỳ chọn ngân sách* | *Tuỳ chọn ngân sách* | NVIDIA L4 / A100 | **NOT_STARTED** |

---

## 2. Bảng So Sánh Detection Baselines (5 Lớp COCO ID 0–4)

**Tập dữ liệu cố định:**
- **Train gốc:** `8,289` ảnh / `18,345` boxes (COCO JSON SHA-256: `a19eec883ccd8bfd76248577275ae15b01aa1c14878507fe2cda59fc4226c2eb`).
- **Validation sạch (`valid_clean_candidate`):** `1,743` ảnh / `3,932` boxes (COCO JSON SHA-256: `9faa0cc8aade9802f63905c7efe9a8c10f7118d866b186d46f2fd808c3464de8`). Được làm sạch từ `2,316` ảnh `valid` gốc sau khi loại `543` ảnh trùng SHA-256 (`506` trùng với `train` + `37` trùng nội bộ `valid`) và `30` ảnh trùng pHash (`28` trùng với `train` + `2` trùng nội bộ `valid`).
- **Test đã kiểm toán (`test_final_candidate`):** `871` ảnh / `1,884` boxes (COCO JSON SHA-256: `e26ff743f2a94ef9a6069a95cf3c3c69581e43295ff48bb5298b301bfde3eec1`).
- **Công cụ chấm điểm:** Tất cả mô hình đều xuất dự đoán định dạng COCO và chấm bằng cùng `pycocotools.cocoeval.COCOeval`.

| Mô hình / Chế độ | Checkpoint / Kiến trúc | Cơ chế Hậu xử lý (Post-processing) | Ngưỡng Conf vận hành (chọn trên Clean Val) | Clean Val mAP@50 | Clean Val mAP@50:95 | Test mAP@50 | Test mAP@50:95 | Test AR@100 | Test F1 | Phần cứng đo Latency | Trạng thái |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| **Deformable DETR + Frozen DINOv2 ViT-S/14 (Original)** | `detection/checkpoints_wastedetect/best_model.pt` (`518x518`, `100` queries, ep `29/30`) | Top-K Query Ranking (End-to-End, không NMS) | `0.30` | **0.7145** | **0.4335** | **0.7332** | **0.4408** | **0.6531** | **0.6317** | RTX 3050 4GB (Local: 147.87 ms/img) | **COMPLETED (1-Seed Reference)** |
| **Deformable DETR + Frozen DINOv2 ViT-S/14 (+ Class-aware NMS)** | `detection/checkpoints_wastedetect/best_model.pt` (`518x518`, `100` queries, ep `29/30`) | Class-aware NMS (`IoU=0.70`) | `0.30` | **0.7617** | **0.4445** | **0.7801** | **0.4517** | **0.5934** | **0.6889** | RTX 3050 4GB (Local: 148.32 ms/img) | **COMPLETED (1-Seed Reference)** |
| **Cascade Original (Detector + DINOv2 ViT-B/14)** | `best_model.pt` + `checkpoints/dinov2_garbage_final.pth` | Không NMS, `cls_thr=0.75` | `0.30` | **0.4083** | **0.2409** | **0.4134** | **0.2407** | **0.4912** | **0.5023** | RTX 3050 4GB (Local: 162.87 ms/img) | **COMPLETED (1-Seed Reference)** |
| **Cascade + NMS (Detector + NMS + DINOv2 ViT-B/14)** | `best_model.pt` + `checkpoints/dinov2_garbage_final.pth` | Class-aware NMS (`IoU=0.70`), `cls_thr=0.75` | `0.25` | **0.4399** | **0.2477** | **0.4486** | **0.2481** | **0.4227** | **0.5629** | RTX 3050 4GB (Local: 161.34 ms/img) | **COMPLETED (1-Seed Reference)** |
| **YOLO11s (Ưu tiên #2 — Seed 42)** | Ultralytics YOLO11s (CNN CSP/C3k2 + PANet) | Built-in Class-aware NMS (`iou=0.70` cho COCOeval, quét `conf_thr` tối ưu F1 trên Clean Val) | *Chọn riêng trên Clean Val* | *Chờ chạy Colab* | *Chờ chạy Colab* | *Chờ chạy Colab* | *Chờ chạy Colab* | *Chờ chạy Colab* | *Chờ chạy Colab* | NVIDIA L4 (Đo cùng GPU) | **NOT_STARTED** |
| **YOLO11n (Baseline nhẹ bổ sung nếu cần)** | Ultralytics YOLO11n | Built-in Class-aware NMS | *Chọn riêng trên Clean Val* | *Tuỳ chọn* | *Tuỳ chọn* | *Tuỳ chọn* | *Tuỳ chọn* | *Tuỳ chọn* | *Tuỳ chọn* | NVIDIA L4 | **NOT_STARTED** |
| **RT-DETR-L (Tuỳ chọn mở rộng)** | Hybrid CNN-Transformer End-to-End Detector | Top-K Decoder Queries | *Chọn riêng trên Clean Val* | *Tuỳ chọn* | *Tuỳ chọn* | *Tuỳ chọn* | *Tuỳ chọn* | *Tuỳ chọn* | *Tuỳ chọn* | NVIDIA L4 / A100 | **NOT_STARTED** |

> **Quy tắc so sánh tốc độ (Latency / FPS):** Không bao giờ so sánh trực tiếp FPS đo trên **NVIDIA L4 / A100 (Colab Pro)** với con số `147.87 ms/img` (`6.76 FPS`) của Deformable DETR đo trên **Laptop GPU RTX 3050 4GB**. Khi chạy Notebook `02_Detection_Baselines_Colab.ipynb`, tốc độ suy luận (ms/image) sẽ được ghi nhận kèm tên GPU cụ thể của phiên chạy đó.
