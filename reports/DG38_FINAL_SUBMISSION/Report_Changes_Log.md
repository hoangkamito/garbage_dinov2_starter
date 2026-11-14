# NHẬT KÝ CHỈNH SỬA VÀ HOÀN THIỆN BÁO CÁO (REPORT CHANGES LOG — DG38)

**Tệp báo cáo gốc được bảo toàn nguyên vẹn (không chỉnh sửa trực tiếp):**
- `evaluation_runs/report_final_review/Group 1 - 22ES - Backup Original.docx`
- `evaluation_runs/report_final_review/Group 1 - 22ES - Bao Cao - Final Submission.docx`
- `evaluation_runs/report_final_review/_submission_qa/submission_input_v6.docx`

**Tệp báo cáo hoàn chỉnh mới được xuất bản:**
- `reports/DG38_FINAL_SUBMISSION/DG38_Final_Complete_Report.docx`
- `reports/DG38_FINAL_SUBMISSION/DG38_Final_Complete_Report.pdf`

---

## 1. Tổng quan phương pháp tích hợp

Thay vì chỉ ghép nối thô văn bản từ tệp tổng kết thực nghiệm (`DG38_Bao_Cao_Tong_Ket_Thuc_Nghiem_FINAL.docx`) vào cuối báo cáo cũ, toàn bộ kết quả thực nghiệm bổ sung trên Google Colab Pro (**Notebooks 00–06**) đã được **tích hợp hữu cơ vào từng chương mục tương ứng** của báo cáo học thuật, đồng thời loại bỏ hoàn toàn các mâu thuẫn nội bộ giữa bản báo cáo trước đợt chạy Colab và kết quả khóa cuối cùng.

---

## 2. Chi tiết các nội dung giữ nguyên, chỉnh sửa và bổ sung mới theo từng phần

### A. Trang bìa và Phần đầu (Front Matter)
1. **Trang bìa (Giữ nguyên cấu trúc nhà trường):** Giữ nguyên thông tin Đại học Đà Nẵng – Trường Đại học Bách khoa, logo trường, tên môn học (Thị giác máy tính), tên đề tài, giảng viên hướng dẫn (TS. Huỳnh Hữu Hưng) và nhóm sinh viên thực hiện (Nhóm 1: Lê Minh Cường, Nguyễn Nhật Hoàng); cập nhật thời điểm hoàn thiện sang *tháng 11 năm 2026*.
2. **Mục lục tự động (Cập nhật mới qua LibreOffice UNO):** Tự động cập nhật số trang thực tế cho toàn bộ 7 chương, các mục cấp 2 và 3 phụ lục (Phụ lục A, B, C).
3. **Danh mục từ viết tắt, Danh mục bảng (26 bảng) và Danh mục hình (20 hình) (Bổ sung mới):** Đặt ở phần đánh số La Mã (`i, ii, iii...`) ngay sau Mục lục để thuận tiện tra cứu.
4. **Tóm tắt (Tiếng Việt) & Abstract (Tiếng Anh) (Viết mới & Chuẩn hóa):** Tổng hợp đầy đủ cả hai bài toán Classification (10 lớp) và Detection (5 lớp), giao thức 3 random seeds (`42, 123, 2026`), kết quả đối chứng ResNet50 vs. DINOv2 ViT-B/14, YOLO11s vs. RT-DETR-L, kết quả 2 nghiên cứu Ablation trên Validation (`0, 2, 4` blocks và `Crop A–E`), và nêu rõ nhánh Deformable DETR / Cascade (Crop C) mới đánh giá trên Validation sạch, chưa có Final Test mới trong gói Notebook 06.

### B. Chương 1 – Giới thiệu
- **Giữ nguyên:** Bối cảnh bài toán phân loại và phát hiện rác thải đô thị, mục tiêu ứng dụng và câu hỏi nghiên cứu nền tảng.
- **Chỉnh sửa & Bổ sung:** Cập nhật Mục 1.2 và 1.3 để phản ánh đầy đủ 4 mục tiêu thực nghiệm sau khi bổ sung theo góp ý của giảng viên (Baselines độc lập ResNet50, YOLO11s, RT-DETR-L; Giao thức 3 random seeds `42, 123, 2026`; Hai nghiên cứu Ablation trên Validation; Phân biệt rạch ròi Train / Validation / Final Test) và cấu trúc 7 chương + 3 phụ lục.

### C. Chương 2 – Cơ sở lý thuyết và nghiên cứu liên quan
- **Giữ nguyên:** Lý thuyết về Vision Transformer (ViT), DINOv2, DETR, Deformable DETR (bao gồm công thức **Office Math** chuẩn cho mô-đun `MSDeformAttn`) và hệ thống chỉ số đánh giá.
- **Bổ sung mới:**
  - Cơ sở lý thuyết về mạng tích chập sâu có kết nối tắt **ResNet50** (He và cs., 2016 [6]) làm baseline phân loại.
  - Cơ sở lý thuyết về kiến trúc phát hiện một giai đoạn **YOLO11s** (Ultralytics, 2024 [8]) và kiến trúc phát hiện Transformer thời gian thực **RT-DETR-L** (Zhao và cs., 2024 [7]).
  - Ghi chú kỹ thuật trung thực tại Mục 2.3 rằng mô-đun `MSDeformAttn` trong mã nguồn dự án (`detection/Train.py`) sử dụng phép lấy mẫu song tuyến **pure PyTorch (`F.grid_sample`)**, không phải bản cài đặt C++/CUDA extension gốc.
  - Lý thuyết hóa **Giả thuyết lệch miền vùng cắt (Crop Domain Shift Hypothesis)** tại Mục 2.4 và quy tắc thống kê **Mean ± Sample Standard Deviation (`ddof = 1`)** tại Mục 2.5.

### D. Chương 3 – Dữ liệu, Kiểm toán Chống Rò rỉ và Giao thức Thực nghiệm
- **Giữ nguyên:** Bảng phân bố 10 lớp của Garbage Classification v2 (`12.259` ảnh: `9.807` Train / `1.226` Validation / `1.226` Test) kèm quy tắc ánh xạ 10 lớp sang 5 lớp (**Bảng 1**), quy trình kiểm toán khử trùng lặp đa tầng SHA-256 + pHash ≤ 6 + xóa box trùng ID 1847 trên WasteDetect (`8.289` Train / `1.743` Validation sạch / `871` Final Test) (**Bảng 2**, **Bảng 3**).
- **Bổ sung mới (Mục 3.6 & Bảng 4):** Thiết kế giao thức thực nghiệm bổ sung trên Google Colab Pro (NVIDIA L4, quản trị ngân sách 100 CU), quy trình khóa 2 gói ZIP Final Test bằng mã băm SHA-256 (`dg38_cls_test_locked.zip` và `dg38_det_test_locked.zip`) và ma trận tổng hợp 7 Notebooks (`00` đến `06`) với 14/14 runs hoàn thành (**Bảng 4**).

### E. Chương 4 – Phương pháp Đề xuất và Hệ thống Đối chứng
- **Giữ nguyên:** Kiến trúc chi tiết DINOv2 ViT-B/14 (**Hình 2**), kiến trúc DINOv2 ViT-S/14 + Deformable DETR (**Hình 3**), bảng đối chiếu cấu hình với bài báo tham chiếu Islam và cs. (2025) (**Bảng 7**).
- **Bổ sung mới:**
  - **Hình 1 mới (`fig_full_pipeline_architecture.png`):** Sơ đồ tổng thể kiến trúc Pipeline Cascade hai giai đoạn (`Deformable DETR → NMS IoU=0,70 → Crop C → DINOv2 ViT-B/14 → Class Mapping`) đặt cạnh hai nhánh đối chứng độc lập (Classification 10 lớp & Detection 5 lớp).
  - **Bảng 5 mới:** Cấu hình đối chứng song song giữa DINOv2 ViT-B/14 và baseline ResNet50 qua 3 seeds (`42, 123, 2026`), kèm thiết kế 3 cấu hình Ablation mở khóa `0, 2, 4` khối Transformer.
  - **Bảng 6 mới:** Cấu hình đối chứng giữa nhánh Deformable DETR và hai baseline phát hiện độc lập YOLO11s (3 seeds) và RT-DETR-L (1 seed).
  - **Bảng 8 & Bảng 9 mới:** Thông số khóa từ Validation của nhánh Deformable DETR / Cascade (`Proposal thr = 0,00`, `NMS IoU = 0,70`, `Detector PR thr = 0,30`, `Cascade PR thr = 0,25`, `Max proposals = 100`, `Context padding = 15%`) và đặc tả kỹ thuật chi tiết của 5 biến thể tiền xử lý vùng cắt (**Crop Variants A–E**).

### F. Chương 5 – Kết quả Thực nghiệm và Đánh giá
1. **Mục 5.1 — Kết quả Classification qua 3 Seeds (Bổ sung mới — Bảng 10 & Hình 4):**
   - Bổ sung kết quả Validation và Final Test (`Mean ± Sample Std`, `n = 3`) của **ResNet50** và **DINOv2 ViT-B/14**, kèm phân tích sâu 4 khía cạnh: mức cải thiện (`+2,39` điểm % Accuracy và `+2,91` điểm % Macro F1 trên Test), độ ổn định qua 3 seeds, chênh lệch Validation–Test, và nguyên nhân biểu diễn DINOv2 hiệu quả (kèm lưu ý không khẳng định ý nghĩa thống kê khi chưa có kiểm định phù hợp).
2. **Mục 5.2 & 5.3 — Phân tích theo lớp (10 lớp), Confusion Matrix và Đối chiếu bài báo tham chiếu (Giữ nguyên & Chuẩn hóa — Hình 5–9, Bảng 11–13):**
   - Giữ nguyên bảng phân tích 10 lớp và Confusion Matrix `1.226` ảnh Test của checkpoint tham chiếu DINOv2 ViT-B/14 (seed 42), làm rõ sự nhất quán giữa kết quả 1 seed tham chiếu (`98,69%` Acc / `98,43%` Macro F1) và trung bình 3 seeds (`98,61% ± 0,29` / `98,40% ± 0,30`).
3. **Mục 5.4 — Kết quả Detection Baselines trên Validation và Final Test (Bổ sung mới — Bảng 14, Bảng 15, Bảng 16 & Hình 10):**
   - Bổ sung bảng kết quả Validation sạch (`1.743` ảnh) của YOLO11s (3 seeds), RT-DETR-L (1 seed) và Deformable DETR / Cascade (**Bảng 14**).
   - Bổ sung bảng kết quả Final Test đã khóa (`871` ảnh, `1.884` GT boxes) liệt kê từng seed (`42, 123, 2026`) và dòng tổng hợp `Mean ± Sample Std` cho **YOLO11s** (`mAP@50 = 82,78% ± 0,40`, `mAP@50:95 = 60,37% ± 0,42`, `AR@100 = 69,38% ± 0,21`) và **RT-DETR-L** 1 seed (`mAP@50 = 70,13%`, `mAP@50:95 = 48,57%`, `AR@100 = 69,55%`) (**Bảng 15**), cùng bảng đối chiếu chênh lệch Validation – Test (**Bảng 16**) và biểu đồ trực quan có thanh sai số chuẩn xác (**Hình 10**).
4. **Mục 5.5 — Nghiên cứu Ablation 1: Số khối Transformer Fine-tune (Bổ sung mới — Bảng 17 & Hình 11):**
   - Trình bày kết quả Validation Macro F1 của `0 blocks` (`96,46%`), `2 blocks` (`96,99%`) và `4 blocks` (`97,33%`) kèm tỷ lệ tham số huấn luyện (`0,01%`, `16,38%`, `32,75%`) và phân tích đánh đổi lý thuyết.
5. **Mục 5.6 — Nghiên cứu Ablation 2: Tiền xử lý vùng cắt Crop A–E (Bổ sung mới — Bảng 18 & Hình 12):**
   - Trình bày kết quả COCO `mAP@50`, `mAP@50:95`, `AR@100` của 5 biến thể `A–E` trên `1.743` ảnh Validation sạch, mô tả chính xác 5 bước của **Crop C** (`46,69%` mAP@50; `26,55%` mAP@50:95; `42,52%` AR@100) và giải thích cơ chế giảm lệch miền vùng cắt.
6. **Mục 5.7 & 5.8 — Phân tích nhánh Deformable DETR & Cascade trên Validation sạch và Trạng thái đánh giá cuối (Chỉnh sửa chuẩn hóa & Bổ sung — Hình 13–19, Bảng 19–23):**
   - **Thay đổi quan trọng để đảm bảo tính nhất quán tuyệt đối:** Thay thế các bảng/biểu đồ chạy thử Test cũ (vốn chỉ chạy với Crop A trước khi có quy trình Colab Notebooks 00–06) bằng bộ bảng và biểu đồ định lượng chuẩn xác trên tập **Validation sạch (`1.743` ảnh, `3.932` GT boxes)** lấy từ `evaluation_runs/val_grid_search_01/cascade_error_analysis.json` và `Notebook 04` (**Bảng 19, Bảng 20, Hình 16, Hình 17**), giữ nguyên bảng Factorial Ablation 6 cấu hình (**Bảng 21**) và ma trận nhầm lẫn chẩn đoán 5×10 trên Validation (**Bảng 22, Hình 18, Hình 19**), đồng thời bổ sung **Bảng 23** tổng hợp rõ trạng thái đánh giá cuối (nêu rõ Deformable DETR và Cascade Crop C chưa chạy Final Test mới).

### G. Chương 6, Chương 7, Tài liệu tham khảo và Phụ lục A, B, C
- **Chương 6 (Mở rộng toàn diện):** Thảo luận đầy đủ 8 luận điểm khoa học theo yêu cầu của giảng viên.
- **Chương 7 (Cập nhật):** Tổng kết chính xác các kết quả đã kiểm chứng và đề xuất 4 hướng phát triển tiếp theo (đưa việc chạy Final Test cho Deformable DETR / Cascade Crop C và mở rộng 3 seeds cho RT-DETR-L vào hướng phát triển).
- **Tài liệu tham khảo (Mở rộng lên 14 nguồn):** Bổ sung trích dẫn gốc cho ResNet50 (He và cs., 2016 [6]), RT-DETR (Zhao và cs., 2024 [7]) và Ultralytics YOLO11 (Jocher và cs., 2024 [8]).
- **Phụ lục A, B, C (Cập nhật & Bổ sung mới):** Bổ sung **Bảng A1** (danh sách Notebooks 00–06 và các module `.py`), **Bảng B1** (mã băm SHA-256 và trạng thái đồng bộ trung thực giữa Google Drive và Windows) và **Bảng C1** (ma trận đối chiếu hoàn thành góp ý của giảng viên).
