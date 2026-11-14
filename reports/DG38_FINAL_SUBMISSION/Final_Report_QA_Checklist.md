# BẢNG KIỂM ĐỊNH CHẤT LƯỢNG BÁO CÁO CUỐI CÙNG (FINAL REPORT QA CHECKLIST)

**Dự án:** Hệ thống Phân loại và Phát hiện Rác thải bằng Deep Learning (Mã đề tài: **DG38**)  
**Thư mục bàn giao:** `D:\Download Acer\garbage_dinov2_starter\reports\DG38_FINAL_SUBMISSION\`  
**Trạng thái kiểm định tổng thể:** **PASS (100% ĐẠT)**  

---

## 1. Kiểm tra Danh mục 6 Sản phẩm Bàn giao (Deliverables Verification)

| STT | Tên file bàn giao | Kích thước | Mã băm SHA-256 | Trạng thái |
|---:|---|---:|---|---|
| 1 | `DG38_Final_Complete_Report.docx` | 9.631.021 bytes (~9,63 MB) | `1155d5a93d8cfa04958aee605302d1c95612fe16cd28813840846b5d46eb2fe6` | **ĐẠT** |
| 2 | `DG38_Final_Complete_Report.pdf` | 4.676.547 bytes (~4,68 MB) | `076c7cabba2e2751cb3cbca62a23b6c339b17d86d44f5c7ee8c0de95a2191b82` | **ĐẠT (62 trang)** |
| 3 | `Teacher_Feedback_Completion_Matrix.md` | Đầy đủ 6 hạng mục góp ý | Đã kiểm chứng đối chiếu với Phụ lục C | **ĐẠT** |
| 4 | `Report_Changes_Log.md` | Nhật ký chi tiết theo chương/mục | Đã kiểm chứng toàn bộ nội dung giữ nguyên/chỉnh sửa/bổ sung | **ĐẠT** |
| 5 | `Evidence_Traceability.csv` | 47 dòng (Header + 26 Bảng + 20 Hình) | Truy xuất 100% nguồn dữ liệu cho Bảng 1–23, A1, B1, C1 và Hình 1–19, A1 | **ĐẠT** |
| 6 | `Final_Report_QA_Checklist.md` | Tài liệu hiện tại | Tổng hợp toàn bộ kết quả kiểm định nội dung, định dạng, tính khoa học | **ĐẠT** |

---

## 2. Kiểm tra Tính Nguyên vẹn của các File Báo cáo Gốc (Original Files Preservation)

Theo Quy tắc III.12 (*"Không chỉnh sửa trực tiếp báo cáo gốc; phải tạo file mới"*), mã băm SHA-256 của toàn bộ các bản báo cáo gốc trước khi chỉnh sửa được kiểm tra và đối chiếu với `evaluation_runs/report_final_review/_submission_qa/preservation_baseline.json`:

- `evaluation_runs/report_final_review/Group 1 - 22ES - Backup Original.docx`  
  - SHA-256: `5d05ec81dd9d2b08ab652769531a2bd0559b9d6e870f7b2ff7ff7b26bae41c2c` (**Nguyên vẹn 100%**)
- `evaluation_runs/report_final_review/Group 1 - 22ES - Bao Cao - Final Submission.docx`  
  - SHA-256: `cdffa9779f5847eecbb046616513102682945637308260028314e94707ec13a9` (**Nguyên vẹn 100%**)
- `evaluation_runs/report_final_review/_submission_qa/submission_input_v6.docx`  
  - SHA-256: `48a5b3c75350816f2ee4f2d85745769b2d77a58858596ef5e308983635148fc6` (**Nguyên vẹn 100%**)

---

## 3. Kiểm tra Cấu trúc Hình thức, Định dạng Học thuật và Duyệt Trực quan 62 Trang PDF

Bản báo cáo `DG38_Final_Complete_Report.docx` được biên dịch và cập nhật mục lục/số trang tự động qua **LibreOffice 26.8.2.2 UNO Server**, sau đó xuất trực tiếp sang `DG38_Final_Complete_Report.pdf` (62 trang) và đã được duyệt trực quan từng trang từ Trang 1 đến Trang 62:

| Hạng mục kiểm tra | Kết quả kiểm định thực tế | Trạng thái |
|---|---|---|
| **Trang bìa (Trang 1 PDF)** | Giữ nguyên mẫu trường ĐH Bách khoa – ĐH Đà Nẵng, Khoa Điện tử - Viễn thông, GVHD TS. Huỳnh Hữu Hưng, Nhóm 1 – Lớp 22ES (Lê Minh Cường `106220252`, Nguyễn Nhật Hoàng `106220259`), khung viền đôi trang bìa chuẩn mực, không đánh số trang trên bìa. | **ĐẠT** |
| **Khối đầu báo cáo (Trang i – ix, Trang 2–10 PDF)** | Đánh số trang La Mã (`i` đến `ix`) ở chân trang giữa: Mục lục tự động (`44` mục chương/mục có số trang chính xác), Danh mục từ viết tắt (18 thuật ngữ), Danh mục hình vẽ (20 hình: `Hình 1–19, A1`), Danh mục bảng biểu (26 bảng: `Bảng 1–23, A1, B1, C1`), Tóm tắt song ngữ Việt – Anh và Bảng phân công nhiệm vụ. | **ĐẠT** |
| **Khối nội dung chính (Trang 1 – 52, Trang 11–62 PDF)** | Đánh số trang Ả Rập (`1` đến `52`) bắt đầu từ Chương 1 đến hết Phụ lục C: Chương 1 (tr. 1–2), Chương 2 (tr. 3–5), Chương 3 (tr. 6–10), Chương 4 (tr. 11–18), Chương 5 (tr. 19–40), Chương 6 (tr. 41–43), Chương 7 (tr. 44–45), Tài liệu tham khảo (tr. 46–47), Phụ lục A–C (tr. 48–52). | **ĐẠT** |
| **Kiểu chữ và Bảng biểu (29 bảng Word / 26 bảng đánh số)** | Font `Times New Roman`, bảng biểu có header lặp lại khi qua trang (`tblHeader`), không cắt dòng giữa chừng (`cantSplit`), căn lề phải cho cột số liệu, căn lề trái cho cột văn bản, sử dụng dấu phẩy thập phân chuẩn tiếng Việt. | **ĐẠT** |
| **Hình vẽ và Biểu đồ (21 ảnh nhúng / 20 hình đánh số)** | Toàn bộ 20 hình (`Hình 1–19, A1`) cùng logo trang bìa hiển thị sắc nét (300 DPI đối với 7 biểu đồ mới), không bị chồng chữ, không bị tràn lề hoặc cắt chú thích, giữ đoạn với caption (`keep_with_next`). | **ĐẠT** |
| **Công thức Toán học** | Công thức `MSDeformAttn` tại Mục 4.3 được dựng bằng cấu trúc toán học gốc Office Math (`<m:oMath>`) hiển thị rõ ràng, chính xác. | **ĐẠT** |

---

## 4. Kiểm tra 12 Quy tắc Bắt buộc về Tính Khoa học và Nhất quán Nội bộ

| Quy tắc | Nội dung yêu cầu | Kết quả kiểm chứng trong báo cáo | Trạng thái |
|---:|---|---|---|
| **1** | Tuyệt đối không tự tạo số liệu, hình ảnh kết quả, checkpoint, kết quả Test hoặc tài liệu tham khảo. | Mọi con số, biểu đồ, mã băm SHA-256 và 14 tài liệu tham khảo đều có nguồn truy xuất rõ ràng trong `Evidence_Traceability.csv`. | **ĐẠT** |
| **2** | Phân biệt rõ Train, Validation và Final Test. | Mọi bảng biểu và nhận xét đều ghi rõ tập dữ liệu và số mẫu (`Train: 9.807 / 8.289`, `Val: 1.226 / 1.743`, `Final Test: 1.226 / 871`). | **ĐẠT** |
| **3** | Không biến số liệu Validation thành Test. | Toàn bộ kết quả của Deformable DETR và Pipeline Cascade (Crop A–E) được đóng khung nhất quán là kết quả trên **Validation sạch (`1.743` ảnh, `3.932` GT boxes)**. | **ĐẠT** |
| **4** | Không dùng kết quả Test để chọn lại mô hình, crop, NMS, learning rate hoặc confidence threshold. | Mọi siêu tham số (`4 blocks`, `Crop C: Context Padding 15%`, `NMS IoU = 0,70`, `conf = 0,25`) đều được khóa trên Validation trước khi mở khóa Final Test 1 lần tại Notebook 06. | **ĐẠT** |
| **5** | Phân biệt kết quả 3 seeds với kết quả 1 seed; không tự tính độ lệch chuẩn cho mô hình chỉ có 1 seed. | `ResNet50`, `DINOv2 ViT-B/14`, `YOLO11s` báo cáo `Mean ± Sample Std (n = 3)`. `RT-DETR-L` (`n = 1`, seed `42`), Fine-tuning Ablation (`0, 2, 4 blocks`) và Crop Ablation (`A–E`) báo cáo điểm đơn, cột SD ghi `— (1 seed)` và biểu đồ không vẽ thanh sai số giả tạo. | **ĐẠT** |
| **6** | Classification gồm 10 lớp, Detection gồm 5 lớp. Hai bài toán có dữ liệu và metrics khác nhau. | Phân tách rõ trong Chương 3, Chương 4, Chương 5 và Bảng 1, Bảng 2, Bảng 5. | **ĐẠT** |
| **7** | Không so sánh trực tiếp Classification Accuracy/Macro F1 với Detection mAP như thể đó là cùng một metric. | Hai nhánh được phân tích độc lập theo đúng metric chuẩn của từng bài toán (`Accuracy / Macro F1` vs. `COCO mAP@50 / mAP@50:95 / AR@100`). | **ĐẠT** |
| **8** | Không khẳng định Cascade đã vượt YOLO11s khi chưa có phép so sánh hợp lệ chứng minh điều đó. | Báo cáo nêu rõ trên Validation sạch (`1.743` ảnh), `YOLO11s` đạt `mAP@50 = 84,94% ± 0,09`, cao hơn đáng kể so với `Cascade Crop C` (`46,69%`), và phân tích trung thực 3 nguyên nhân kỹ thuật. | **ĐẠT** |
| **9** | Không gọi phương pháp Deformable DETR của project là bản triển khai chính thức nếu mã nguồn sử dụng phiên bản pure PyTorch với `grid_sample`. | Nêu rõ tại Tóm tắt, Chương 4 (Mục 4.3), Chương 6 (Mục 6.2, 6.3) và Phụ lục A rằng đồ án sử dụng bản xấp xỉ Pure PyTorch (`F.grid_sample`, single-scale feature map `16 × 16`) trong `models/detector.py`. | **ĐẠT** |
| **10** | Chỉ sử dụng bảng, biểu đồ, code, tài liệu tham khảo và nhận xét có thể truy xuất nguồn. | Kiểm chứng 100% tại `Evidence_Traceability.csv` và Phụ lục A, B. | **ĐẠT** |
| **11** | Giữ nguyên thông tin trường, khoa, giảng viên, lớp, nhóm, thành viên, phần phân công và cấu trúc phù hợp của báo cáo gốc. | Giữ nguyên đầy đủ thông tin hành chính và cấu trúc 7 chương + 3 phụ lục. | **ĐẠT** |
| **12** | Không chỉnh sửa trực tiếp báo cáo gốc; phải tạo file mới. | Toàn bộ sản phẩm mới đặt tại `reports/DG38_FINAL_SUBMISSION/`; các file gốc giữ nguyên SHA-256. | **ĐẠT** |

---

## 5. Kiểm tra Đối chiếu Toàn bộ Số liệu Bắt buộc (Mandatory Numbers Verification)

Script kiểm định tự động `run_qa_verification.py` đã xác nhận sự hiện diện chính xác tuyệt đối (`missing_mandatory_numbers: []`) của toàn bộ các số liệu cốt lõi trong văn bản và bảng biểu:

1. **Classification Validation (3 seeds: `42, 123, 2026`):**
   - `ResNet50`: Accuracy = `96,25% ± 0,08`, Macro F1 = `95,51% ± 0,21` (`n = 3`).
   - `DINOv2 ViT-B/14`: Accuracy = `97,63% ± 0,22`, Macro F1 = `97,33% ± 0,20` (`n = 3`).
2. **Classification Final Test (3 seeds: `42, 123, 2026`):**
   - `ResNet50`: Accuracy = `96,22% ± 0,29` (`0,962208 ± 0,002865`), Macro F1 = `95,50% ± 0,27` (`0,954968 ± 0,002713`).
   - `DINOv2 ViT-B/14`: Accuracy = `98,61% ± 0,29` (`0,986134 ± 0,002941`), Macro F1 = `98,40% ± 0,30` (`0,984033 ± 0,003002`).
3. **Detection Validation (`1.743` ảnh sạch):**
   - `YOLO11s (3 seeds)`: `mAP@50 = 84,94% ± 0,09`, `mAP@50:95 = 61,45% ± 0,16`.
   - `RT-DETR-L (1 seed, seed 42)`: `mAP@50 = 71,20%`, `mAP@50:95 = 49,43%`.
4. **Detection Final Test (`871` ảnh, `1.884` GT boxes):**
   - `YOLO11s` từng seed:
     - Seed `42`: `mAP@50 = 0,832441` (`83,24%`), `mAP@50:95 = 0,608381` (`60,84%`), `AR@100 = 0,695373` (`69,54%`).
     - Seed `123`: `mAP@50 = 0,825127` (`82,51%`), `mAP@50:95 = 0,602519` (`60,25%`), `AR@100 = 0,694642` (`69,46%`).
     - Seed `2026`: `mAP@50 = 0,825936` (`82,59%`), `mAP@50:95 = 0,600146` (`60,01%`), `AR@100 = 0,691502` (`69,15%`).
   - `YOLO11s` tổng hợp (`Mean ± Sample Std`, `n = 3`):
     - `mAP@50 = 82,78% ± 0,40` (`0,827835 ± 0,004010`)
     - `mAP@50:95 = 60,37% ± 0,42` (`0,603682 ± 0,004239`)
     - `AR@100 = 69,38% ± 0,21` (`0,693839 ± 0,002057`)
   - `RT-DETR-L (1 seed, seed 42)`: `mAP@50 = 0,701251` (`70,13%`), `mAP@50:95 = 0,485653` (`48,57%`), `AR@100 = 0,695482` (`69,55%`).
5. **Fine-tuning Ablation (Validation Macro F1):**
   - `0 blocks`: `96,46%`; `2 blocks`: `96,99%`; `4 blocks`: `97,33%`.
6. **Crop Preprocessing Ablation (`1.743` ảnh Validation sạch):**
   - `A – Original Crop`: `mAP@50 = 43,99%`, `mAP@50:95 = 24,77%`, `AR@100 = 41,69%`.
   - `B – Direct Resize`: `mAP@50 = 46,19%`, `mAP@50:95 = 26,32%`, `AR@100 = 42,27%`.
   - `C – Context Padding 15%` (Tối ưu): `mAP@50 = 46,69%`, `mAP@50:95 = 26,55%`, `AR@100 = 42,52%`.
   - `D – Letterbox 224`: `mAP@50 = 45,47%`, `mAP@50:95 = 25,95%`, `AR@100 = 41,73%`.
   - `E – Context Padding 25% + Letterbox`: `mAP@50 = 45,75%`, `mAP@50:95 = 26,10%`, `AR@100 = 41,65%`.
7. **Trạng thái đánh giá Deformable DETR và Cascade:**
   - Báo cáo trung thực tại Tóm tắt, Mục 5.4 (`Bảng 14`), Mục 5.7 (`Bảng 18–21`), Mục 5.8 (`Bảng 22–23`), Mục 6.3 và Mục 7.1 rằng nhánh `Deformable DETR` và `Cascade (Crop C)` đã hoàn tất đánh giá trên tập **Validation sạch (`1.743` ảnh)** nhưng **chưa chạy lượt Final Test mới** trên tập Test đã khóa (`871` ảnh) tại Notebook 06. Tuyệt đối không chèn số liệu Test giả định.
