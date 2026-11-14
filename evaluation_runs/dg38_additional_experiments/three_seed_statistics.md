# Thống Kê Ba Random Seeds Độc Lập (`42`, `123`, `2026`) — DG38 Stage 5

## 1. Quy Định Thống Kê Bắt Buộc

1. **Ba Seeds bắt buộc:** `42`, `123`, `2026`.
2. **Định nghĩa một Seed hợp lệ:** Mỗi seed phải là **một lần khởi tạo trọng số ngẫu nhiên (cho phần head/detector) và huấn luyện độc lập từ đầu** trên cùng một manifest dữ liệu cố định. **Tuyệt đối không tính 3 lần chạy inference trên cùng 1 checkpoint là 3 seeds.**
3. **Công thức thống kê:**
   - Khi hoàn thành đủ $n = 3$ seeds độc lập: Báo cáo **$\text{Mean} \pm \text{Sample Std}$** với bậc tự do hiệu chỉnh mẫu **`ddof=1`**:
     $$s = \sqrt{\frac{1}{n - 1} \sum_{i=1}^{n} (x_i - \bar{x})^2}$$
   - Khi mới hoàn thành $1 \le n < 3$ seeds: Bắt buộc gắn nhãn **`INCOMPLETE (n/3 seeds)`**, không được báo cáo độ lệch chuẩn giả hoặc tuyên bố đã hoàn tất 3 seeds.
4. **Tính đồng nhất trong từng nhóm Seed (`validate_seed_group_homogeneity`):**
   - Không gộp các lần chạy khác kiến trúc, khác số block fine-tune, hoặc khác tập Validation vào chung một nhóm thống kê.

---

## 2. Thứ Tự Ưu Tiên Thực Hiện Theo Ngân Sách (Hạn Mức Nhóm: 60.00 CU)

1. **Ưu tiên 1:** Hoàn thành 3 seeds (`42`, `123`, `2026`) cho **ResNet50 Classification**.
2. **Ưu tiên 2:** Hoàn thành 3 seeds (`42`, `123`, `2026`) cho **YOLO11s Detection** (trên Clean Validation `1,743` ảnh).
3. **Ưu tiên 3:** Mở rộng 3 seeds (`42`, `123`, `2026`) cho **DINOv2 ViT-B/14 (fine-tune 4 block cuối)** khi ngân sách còn đủ.
4. **Ưu tiên 4:** Mở rộng 3 seeds cho **Deformable DETR + Frozen DINOv2 ViT-S/14** khi ngân sách và thời gian còn cho phép.

---

## 3. Bảng Theo Dõi Trạng Thái Từng Seed & Thống Kê `Mean ± Sample Std (ddof=1)`

### 3.1. Nhóm Classification (Split Signature SHA-256: `1da7e63c9ee27d18...`, Manifest SHA-256: `cd27f0e69e70691c...`)

| Nhóm mô hình (`group_id`) | Seed 42 (Val / Test Macro F1) | Seed 123 (Val / Test Macro F1) | Seed 2026 (Val / Test Macro F1) | Số Seeds hoàn tất | Test Accuracy (`Mean ± Std, ddof=1`) | Test Macro F1 (`Mean ± Std, ddof=1`) | Trạng thái nhóm |
|---|---|---|---|---:|---|---|---|
| `classification::resnet50::resnet50_15ep_adamw` | `NOT_STARTED` | `NOT_STARTED` | `NOT_STARTED` | `0 / 3` | `NOT_STARTED (0/3 seeds)` | `NOT_STARTED (0/3 seeds)` | **NOT_STARTED** |
| `classification::dinov2_vitb14::dinov2_finetune_blocks_4` | `Ref (checkpoints/dinov2_garbage_final.pth): Test Acc 98.69% / F1 98.43%` | `NOT_STARTED` | `NOT_STARTED` | `1 / 3` *(Ref)* | `0.9869 ± N/A` | `0.9843 ± N/A` | **INCOMPLETE (1/3 seeds)** |

### 3.2. Nhóm Detection (Clean Validation SHA-256: `9faa0cc8aade9802...`)

| Nhóm mô hình (`group_id`) | Seed 42 (Clean Val / Test mAP@50) | Seed 123 (Clean Val / Test mAP@50) | Seed 2026 (Clean Val / Test mAP@50) | Số Seeds hoàn tất | Clean Val mAP@50 (`Mean ± Std, ddof=1`) | Test mAP@50 (`Mean ± Std, ddof=1`) | Trạng thái nhóm |
|---|---|---|---|---:|---|---|---|
| `detection::yolo11s::yolo11s_30ep_cleanval` | `NOT_STARTED` | `NOT_STARTED` | `NOT_STARTED` | `0 / 3` | `NOT_STARTED (0/3 seeds)` | `NOT_STARTED (0/3 seeds)` | **NOT_STARTED** |
| `detection::deformable_detr::dinov2_vits14_frozen` | `Ref (detection/checkpoints_wastedetect/best_model.pt): Clean Val 0.7145 (NMS: 0.7617) / Test 0.7332 (NMS: 0.7801)` | `NOT_STARTED` | `NOT_STARTED` | `1 / 3` *(Ref)* | `0.7145 ± N/A (NMS: 0.7617 ± N/A)` | `0.7332 ± N/A (NMS: 0.7801 ± N/A)` | **INCOMPLETE (1/3 seeds)** |

> **Cập nhật tự động:** Ngay sau khi bạn chạy các seeds trên Colab bằng `01_Classification_Baselines_Colab.ipynb` và `02_Detection_Baselines_Colab.ipynb`, hãy mở `05_DG38_Results_Analysis_Colab.ipynb` để hàm `aggregate_seed_group()` tự động kiểm tra tính đồng nhất và điền đầy đủ giá trị `Mean ± Sample Std (ddof=1)` vào bảng này.
