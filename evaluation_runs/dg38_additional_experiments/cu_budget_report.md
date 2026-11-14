# Báo cáo Quản lý & Dự báo Ngân sách Google Colab Pro (200.45 CU)

- **Cập nhật lần cuối (UTC):** `2026-11-01T15:53:31.076279+00:00`
- **Tổng ngân sách khởi tạo:** **`200.45 CU`**
- **Tổng CU đã ghi nhận qua các lượt chạy:** **`0.00 CU`**
- **Số dư CU do người dùng đồng bộ từ Colab UI gần nhất:** **`200.45 CU`** *(lúc `2026-11-01T15:53:31.076265+00:00`)*
- **Số dư CU khả dụng hiệu dụng (`effective_remaining_cu`):** **`200.45 CU`**
- **Ngưỡng dự phòng an toàn tối thiểu bắt buộc:** **`40.00 CU`**

> [!IMPORTANT]
> Google Colab không cung cấp API Python chính thức để đọc trực tiếp số Compute Units (CU) còn lại. Mọi phép dự báo và ghi nhận chi phí đều dựa trên công thức:
> `estimated_CU = estimated_GPU_hours × observed_CU_per_hour`
> trong đó `observed_CU_per_hour` (Usage Rate) được người dùng nhập trực tiếp từ giao diện **Google Colab → View resources**.

---

## 1. Bảng Phân bổ và Trạng thái Ngân sách Hiện tại

| Mã nhóm (`category`) | Nhiệm vụ | Hạn mức cấp (CU) | Đã dùng (CU) | Còn lại trong nhóm (CU) | Tỷ lệ dùng (%) | Trạng thái |
|---|---|---:|---:|---:|---:|---|
| `smoke_and_preflight` | Smoke test, kiểm tra môi trường | 10.00 | 0.00 | 10.00 | 0.0% | CHƯA DÙNG |
| `resnet50_baseline` | ResNet50 Classification baseline | 15.00 | 0.00 | 15.00 | 0.0% | CHƯA DÙNG |
| `yolo11s_baseline` | YOLO11s Detection baseline | 35.00 | 0.00 | 35.00 | 0.0% | CHƯA DÙNG |
| `three_seed_experiments` | Three-seed experiments (Seeds 42, 123, 2026) | 60.00 | 0.00 | 60.00 | 0.0% | CHƯA DÙNG |
| `crop_preprocessing_ablation` | Crop preprocessing ablation (Validation cached) | 15.00 | 0.00 | 15.00 | 0.0% | CHƯA DÙNG |
| `finetune_blocks_ablation` | Fine-tune blocks ablation (0, 2, 4, 12 blocks) | 15.00 | 0.00 | 15.00 | 0.0% | CHƯA DÙNG |
| `reserve_and_expansion` | Dự phòng và mở rộng (cần phê duyệt, giữ tối thiểu 40 CU an toàn) | 50.45 | 0.00 | 50.45 | 0.0% | LOCKED_RESERVE (Giữ >= 40 CU an toàn) |
| **TỔNG CỘNG** | **Ngân sách Colab Pro hiện có** | **200.45** | **0.00** | **200.45** | **0.0%** | **Dự phòng tối thiểu: 40.00 CU** |

---

## 2. Lịch sử Dự báo Chi phí Trước Thí nghiệm (`pre_experiment_gate`)

*Chưa có lượt chạy pilot/forecast nào được ghi nhận trên Colab.*

---

## 3. Nhật ký Tiêu thụ CU Thực tế (`usage_logs`)

*Chưa có lượt chạy GPU thực tế nào được ghi nhận.*
