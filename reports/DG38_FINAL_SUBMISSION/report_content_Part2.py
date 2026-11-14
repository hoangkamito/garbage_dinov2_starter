# -*- coding: utf-8 -*-
"""
Part 2 of the complete academic report content for DG38:
Tables for Chapters 5, 6, 7 and Appendices A, B, C, plus References.
"""

TABLE_10_CLS_3SEEDS = [
    ['Mô hình phân loại (10 lớp)', 'Số hạt giống (Seeds)', 'Validation Accuracy (%)\n(Mean ± Sample Std)', 'Validation Macro F1 (%)\n(Mean ± Sample Std)', 'Final Test Accuracy (%)\n(Mean ± Sample Std)', 'Final Test Macro F1 (%)\n(Mean ± Sample Std)', 'Chênh lệch Test − Val\n(Acc / Macro F1, điểm %)'],
    ['ResNet50 (Baseline độc lập, ImageNet-1K)', '3 (42, 123, 2026)', '96,25% ± 0,08', '95,51% ± 0,21', '96,22% ± 0,29\n(mean = 0,962208;\nstd = 0,002865)', '95,50% ± 0,27\n(mean = 0,954968;\nstd = 0,002713)', '−0,03 / −0,01'],
    ['DINOv2 ViT-B/14 (Đề xuất, fine-tune 4 blocks)', '3 (42, 123, 2026)', '97,63% ± 0,22', '97,33% ± 0,20', '98,61% ± 0,29\n(mean = 0,986134;\nstd = 0,002941)', '98,40% ± 0,30\n(mean = 0,984033;\nstd = 0,003002)', '+0,98 / +1,07'],
    ['Chênh lệch tuyệt đối (DINOv2 − ResNet50)', 'Cùng giao thức 3 seeds', '+1,38 điểm %', '+1,82 điểm %', '+2,39 điểm %\n(+0,023926)', '+2,91 điểm %\n(+0,029065)', 'DINOv2 cao hơn trên cả Validation và Final Test'],
]

TABLE_14_DET_VALIDATION = [
    ['Mô hình / Cấu hình phát hiện (5 lớp WasteDetect)', 'Họ kiến trúc & Tiền xử lý', 'Số hạt giống (Seeds)', 'Validation mAP@50 (%)', 'Validation mAP@50:95 (%)', 'Validation AR@100 (%)', 'Ghi chú giao thức trên 1.743 ảnh Validation sạch (3.932 GT boxes)'],
    ['YOLO11s (Baseline độc lập CNN one-stage)', 'Ultralytics YOLO11s, 640×640, NMS nội bộ', '3 (42, 123, 2026)', '84,94% ± 0,09', '61,45% ± 0,16', '70,34%', 'Mean ± Sample Std (ddof = 1) qua 3 seeds; dẫn đầu trên tập Validation sạch'],
    ['RT-DETR-L (Baseline độc lập Transformer)', 'Ultralytics RT-DETR-L, 640×640, 300 queries', '1 (seed 42)', '71,20%', '49,43%', 'Không lưu tách riêng', 'Chỉ có 1 seed (không tính độ lệch chuẩn mẫu)'],
    ['Deformable DETR — detector_original', 'Frozen DINOv2 ViT-S/14 + Def. DETR, không NMS (PR thr = 0,30)', '1 (seed 42 tham chiếu)', '71,45%\n(0,7145)', '43,35%\n(0,4335)', '64,48%\n(0,6448)', 'Checkpoint tham chiếu best_model.pt; xuất 100 queries/ảnh trước NMS'],
    ['Deformable DETR — detector_nms', 'Frozen DINOv2 ViT-S/14 + Def. DETR, Class-Agnostic NMS IoU = 0,70', '1 (seed 42 tham chiếu)', '76,17%\n(0,7617)', '44,45%\n(0,4445)', '58,49%\n(0,5849)', 'Cấu hình Detector một giai đoạn tốt nhất của nhánh Deformable DETR trên Validation'],
    ['Pipeline Cascade — cascade_original (Crop A)', 'Def. DETR (không NMS) → Crop A → DINOv2 ViT-B/14 (PR thr = 0,30)', '1 (seed 42 tham chiếu)', '40,83%\n(0,4083)', '24,09%\n(0,2409)', '48,59%\n(0,4859)', 'Cắt sát bbox + Resize(256) + CenterCrop(224); bị suy giảm do OOS và lệch miền crop'],
    ['Pipeline Cascade — cascade_nms (Crop A)', 'Def. DETR + NMS IoU = 0,70 → Crop A → DINOv2 ViT-B/14 (PR thr = 0,25)', '1 (seed 42 tham chiếu)', '43,99%\n(0,4399)', '24,77%\n(0,2477)', '41,69%\n(0,4169)', 'Cấu hình cơ sở trong thiết kế 2×2 trước khi cải tiến tiền xử lý vùng cắt'],
    ['Pipeline Cascade — cascade_nms (Crop C — Tối ưu)', 'Def. DETR + NMS IoU = 0,70 → Crop C (+15% pad, 224×224) → DINOv2 ViT-B/14', '1 (seed 42 tham chiếu)', '46,69%\n(0,4669)', '26,55%\n(0,2655)', '42,52%\n(0,4252)', 'Cấu hình Cascade đạt mAP cao nhất trên Validation (+2,70 điểm % mAP@50 so với Crop A)'],
]

TABLE_15_DET_FINAL_TEST = [
    ['Mô hình phát hiện', 'Loại dòng kết quả / Hạt giống (Seed)', 'Số lượng Seeds (n)', 'Final Test mAP@50\n(Số thực / %)', 'Final Test mAP@50:95\n(Số thực / %)', 'Final Test AR@100\n(Số thực / %)', 'Ghi chú thực nghiệm trên 871 ảnh Test khóa (1.884 GT boxes)'],
    ['YOLO11s', 'Lượt chạy độc lập — Seed 42', 'Lượt 1/3', '0,832441 (83,24%)', '0,608381 (60,84%)', '0,695373 (69,54%)', 'Notebook 06 — đánh giá trên dg38_det_test_locked.zip'],
    ['YOLO11s', 'Lượt chạy độc lập — Seed 123', 'Lượt 2/3', '0,825127 (82,51%)', '0,602519 (60,25%)', '0,694642 (69,46%)', 'Notebook 06 — đánh giá trên dg38_det_test_locked.zip'],
    ['YOLO11s', 'Lượt chạy độc lập — Seed 2026', 'Lượt 3/3', '0,825936 (82,59%)', '0,600146 (60,01%)', '0,691502 (69,15%)', 'Notebook 06 — đánh giá trên dg38_det_test_locked.zip'],
    ['YOLO11s (Tổng hợp 3 seeds)', 'Trung bình ± Độ lệch chuẩn mẫu (Mean ± Sample Std, ddof = 1)', '3', '82,78% ± 0,40\n(0,827835 ± 0,004010)', '60,37% ± 0,42\n(0,603682 ± 0,004239)', '69,38% ± 0,21\n(0,693839 ± 0,002057)', 'Độ lệch chuẩn mẫu rất nhỏ (≤ 0,42 điểm %), chứng tỏ hội tụ ổn định'],
    ['RT-DETR-L', 'Lượt chạy đơn — Seed 42 (Tổng hợp 1 seed)', '1', '70,13%\n(0,701251)', '48,57%\n(0,485653)', '69,55%\n(0,695482)', 'Chỉ có 1 seed nên báo cáo giá trị đơn lẻ, không tính độ lệch chuẩn'],
    ['Deformable DETR & Cascade (Crop C)', 'Trạng thái trên bộ Final Test khóa (Notebook 06)', '0 (Chưa chạy)', 'Chưa thực hiện', 'Chưa thực hiện', 'Chưa thực hiện', 'Không tự suy diễn hay điền số liệu Test giả định cho nhánh này'],
]

TABLE_16_DET_VAL_VS_TEST = [
    ['Mô hình và Chỉ số đánh giá COCO', 'Số Seeds', 'Validation sạch\n(1.743 ảnh, 3.932 box)', 'Final Test đã khóa\n(871 ảnh, 1.884 box)', 'Chênh lệch (Test − Validation)\n(điểm phần trăm)', 'Nhận xét về tính nhất quán giữa Validation và Final Test'],
    ['YOLO11s — mAP@50 (%)', '3 (42, 123, 2026)', '84,94% ± 0,09', '82,78% ± 0,40', '−2,16 điểm %', 'Giảm nhẹ trong biên độ bình thường khi chuyển sang tập Test độc lập'],
    ['YOLO11s — mAP@50:95 (%)', '3 (42, 123, 2026)', '61,45% ± 0,16', '60,37% ± 0,42', '−1,08 điểm %', 'Mức chênh lệch nhỏ (−1,08 điểm %); bảo toàn thứ hạng dẫn đầu'],
    ['YOLO11s — AR@100 (%)', '3 (42, 123, 2026)', '70,34%', '69,38% ± 0,21', '−0,96 điểm %', 'Độ bao phủ đề xuất (Average Recall) duy trì ổn định quanh mức ~69–70%'],
    ['RT-DETR-L — mAP@50 (%)', '1 (seed 42)', '71,20%', '70,13%', '−1,07 điểm %', 'Chênh lệch −1,07 điểm % giữa Validation và Final Test'],
    ['RT-DETR-L — mAP@50:95 (%)', '1 (seed 42)', '49,43%', '48,57%', '−0,86 điểm %', 'Chênh lệch −0,86 điểm % (48,5653% − 49,43% ≈ −0,86 / −0,87 điểm %)'],
    ['RT-DETR-L — AR@100 (%)', '1 (seed 42)', 'Không lưu tách riêng', '69,55%', 'Không áp dụng', 'Trên Final Test, AR@100 của RT-DETR-L (69,55%) tương đương YOLO11s (69,38%)'],
]

TABLE_17_FINETUNE_ABLATION = [
    ['Cấu hình số khối mở khóa (Unfreeze Blocks)', 'Các tầng được cập nhật trọng số (Trainable Layers)', 'Số tham số huấn luyện / Tổng số tham số (86.589.706)', 'Tỷ lệ tham số huấn luyện (%)', 'Validation Macro F1 (%)\n(1.226 ảnh Validation)', 'Mức tăng so với 0 blocks\n(điểm phần trăm)'],
    ['0 blocks (Linear Probe)', 'Chỉ mở khóa tầng chuẩn hóa cuối (layernorm) và đầu phân loại tuyến tính Linear(768 → 10); đóng băng toàn bộ 12 khối Transformer', '9.226 / 86.589.706', '0,01%', '96,46%', 'Cơ sở (0,00)'],
    ['2 blocks cuối', 'Mở khóa 2 khối Transformer cuối (encoder.layer.10..11), tầng layernorm cuối và đầu phân loại Linear(768 → 10)', '14.184.970 / 86.589.706', '16,38%', '96,99%', '+0,53 điểm %'],
    ['4 blocks cuối (Cấu hình chọn)', 'Mở khóa 4 khối Transformer cuối (encoder.layer.8..11), tầng layernorm cuối và đầu phân loại Linear(768 → 10)', '28.360.714 / 86.589.706', '32,75%', '97,33%', '+0,87 điểm %'],
]

TABLE_18_CROP_ABLATION = [
    ['Phương pháp tiền xử lý vùng cắt (Crop Variant)', 'Tóm tắt quy trình biến đổi hình học trước khi đưa vào DINOv2 ViT-B/14', 'Validation mAP@50 (%)', 'Validation mAP@50:95 (%)', 'Validation AR@100 (%)', 'Chênh lệch so với Crop A\n(mAP@50 / mAP@50:95 / AR@100)'],
    ['A — Original Crop', 'Cắt sát bbox dự đoán → Resize cạnh ngắn 256 → CenterCrop 224×224 → Chuẩn hóa ImageNet', '43,99%', '24,77%', '41,69%', 'Cơ sở (0,00 / 0,00 / 0,00)'],
    ['B — Direct Resize', 'Cắt sát bbox dự đoán → Resize trực tiếp về 224×224 (bỏ bước CenterCrop) → Chuẩn hóa ImageNet', '46,19%', '26,32%', '42,27%', '+2,20 / +1,55 / +0,58 điểm %'],
    ['C — Context Padding 15% (Tối ưu)', 'Mở rộng bbox thêm 15% mỗi phía → Giới hạn trong biên ảnh gốc → Cắt ảnh → Resize trực tiếp 224×224 → Chuẩn hóa ImageNet', '46,69%', '26,55%', '42,52%', '+2,70 / +1,78 / +0,83 điểm %'],
    ['D — Letterbox 224', 'Cắt sát bbox → Đệm vuông bằng viền xám RGB(114,114,114) giữ nguyên tỷ lệ cạnh → Resize 224×224 → Chuẩn hóa ImageNet', '45,47%', '25,95%', '41,73%', '+1,48 / +1,18 / +0,04 điểm %'],
    ['E — Context Padding 25% + Letterbox', 'Mở rộng bbox thêm 25% mỗi phía → Giới hạn biên ảnh → Đệm vuông RGB(114,114,114) → Resize 224×224 → Chuẩn hóa ImageNet', '45,75%', '26,10%', '41,65%', '+1,76 / +1,33 / −0,04 điểm %'],
]

TABLE_19_DEFDETR_CASCADE_VALIDATION = [
    ['Chỉ số đánh giá trên Validation sạch (1.743 ảnh, 3.932 GT boxes)', 'detector_original\n(No NMS, PR=0,30)', 'detector_nms\n(NMS 0,70, PR=0,30)', 'cascade_original\n(No NMS, Crop A)', 'cascade_nms (Crop A)\n(NMS 0,70, PR=0,25)', 'cascade_nms (Crop C)\n(NMS 0,70, Pad 15%)'],
    ['mAP@50 (Số thực / %)', '0,7145 (71,45%)', '0,7617 (76,17%)', '0,4083 (40,83%)', '0,4399 (43,99%)', '0,4669 (46,69%)'],
    ['mAP@50:95 (Số thực / %)', '0,4335 (43,35%)', '0,4445 (44,45%)', '0,2409 (24,09%)', '0,2477 (24,77%)', '0,2655 (26,55%)'],
    ['AR@100 (Số thực / %)', '0,6448 (64,48%)', '0,5849 (58,49%)', '0,4859 (48,59%)', '0,4169 (41,69%)', '0,4252 (42,52%)'],
    ['Ngưỡng điểm vận hành (PR score threshold)', '0,30', '0,30', '0,30', '0,25', '0,25'],
    ['Precision@50 tại ngưỡng vận hành (%)', '65,92% (0,6592)', '78,34% (0,7834)', '60,63% (0,6063)', '60,84% (0,6084)', 'Đánh giá qua COCO mAP'],
    ['Recall@50 tại ngưỡng vận hành (%)', '59,77% (0,5977)', '59,41% (0,5941)', '42,37% (0,4237)', '50,69% (0,5069)', 'Đánh giá qua COCO mAP'],
    ['F1@50 tại ngưỡng vận hành (%)', '62,69% (0,6269)', '67,57% (0,6757)', '49,88% (0,4988)', '55,30% (0,5530)', 'Đánh giá qua COCO mAP'],
    ['Số lượng TP / FP / FN tại ngưỡng vận hành', '2.350 / 1.215 / 1.582', '2.336 / 646 / 1.596', '1.666 / 1.082 / 2.266', '1.993 / 1.283 / 1.939', 'Cùng tập 64.819 proposals sau NMS'],
    ['Tổng số proposals giữ lại sau NMS (trên 1.743 ảnh)', '174.300 (100,0/ảnh)', '64.819 (37,19/ảnh)', '174.300 (100,0/ảnh)', '64.819 (37,19/ảnh)', '64.819 (37,19/ảnh)'],
    ['Số dự đoán bị loại do rơi vào 5 lớp OOS (toàn bộ / tại ngưỡng PR)', '0 / 0', '0 / 0', '19.702 / 223', '9.229 / 262', 'Xử lý theo cơ chế drop_and_count'],
]

TABLE_20_DEFDETR_CASCADE_PER_CLASS_VAL = [
    ['Lớp rác thải (COCO ID)', 'Số GT Box trên Validation sạch', 'detector_original\nAP@50 / AP@50:95 / AR@100 (%)', 'detector_nms (IoU = 0,70)\nAP@50 / AP@50:95 / AR@100 (%)', 'cascade_original (Crop A)\nAP@50 / AP@50:95 / AR@100 (%)', 'cascade_nms (Crop A)\nAP@50 / AP@50:95 / AR@100 (%)'],
    ['glass (ID: 0)', '790 (20,09%)', '88,52% / 54,58% / 69,75%', '91,13% / 54,85% / 64,46%', '23,00% / 12,88% / 37,28%', '23,85% / 12,78% / 27,59%'],
    ['metal (ID: 1)', '668 (16,99%)', '60,35% / 34,71% / 58,65%', '65,80% / 36,12% / 53,20%', '46,61% / 25,94% / 58,65%', '50,24% / 26,68% / 52,60%'],
    ['organic (ID: 2)', '632 (16,07%)', '57,90% / 32,67% / 58,39%', '64,13% / 34,34% / 52,12%', '55,61% / 31,54% / 56,42%', '61,72% / 33,21% / 50,05%'],
    ['paper (ID: 3)', '923 (23,47%)', '78,86% / 51,04% / 70,75%', '85,32% / 52,87% / 64,27%', '46,18% / 29,83% / 44,40%', '49,95% / 30,91% / 38,80%'],
    ['plastic (ID: 4)', '919 (23,37%)', '71,60% / 43,76% / 64,85%', '74,47% / 44,06% / 58,40%', '32,74% / 20,25% / 46,19%', '34,17% / 20,29% / 39,42%'],
    ['Trung bình toàn cục (Macro / COCO)', '3.932 (100,00%)', '71,45% / 43,35% / 64,48%', '76,17% / 44,45% / 58,49%', '40,83% / 24,09% / 48,59%', '43,99% / 24,77% / 41,69%'],
]

TABLE_23_EVALUATION_STATUS_SUMMARY = [
    ['Nhánh mô hình / Thí nghiệm', 'Bài toán & Số lớp', 'Số hạt giống (Seeds)', 'Trạng thái đánh giá trên Validation', 'Trạng thái đánh giá trên Final Test đã khóa', 'Kết quả chính được phép kết luận trong báo cáo'],
    ['ResNet50 (Baseline độc lập)', 'Classification (10 lớp)', '3 (42, 123, 2026)', 'Đã hoàn thành (1.226 ảnh): Acc = 96,25% ± 0,08; Macro F1 = 95,51% ± 0,21', 'Đã hoàn thành (Notebook 06, 1.226 ảnh): Acc = 96,22% ± 0,29; Macro F1 = 95,50% ± 0,27', 'Baseline CNN hội tụ rất ổn định, làm mốc đối chứng công bằng cho DINOv2'],
    ['DINOv2 ViT-B/14 (4 blocks cuối)', 'Classification (10 lớp)', '3 (42, 123, 2026)', 'Đã hoàn thành (1.226 ảnh): Acc = 97,63% ± 0,22; Macro F1 = 97,33% ± 0,20', 'Đã hoàn thành (Notebook 06, 1.226 ảnh): Acc = 98,61% ± 0,29; Macro F1 = 98,40% ± 0,30', 'Vượt ResNet50 +2,39 điểm % Acc và +2,91 điểm % Macro F1 trên Final Test'],
    ['Ablation Fine-tuning (0, 2, 4 blocks)', 'Classification (10 lớp)', '1 (cố định giao thức)', 'Đã hoàn thành (Notebook 03, 1.226 ảnh): Macro F1 đạt 96,46% → 96,99% → 97,33%', 'Không chạy trên Test (đúng nguyên tắc chọn siêu tham số trên Validation)', 'Mở khóa 4 blocks cuối đạt Validation Macro F1 cao nhất trong 3 cấu hình khảo sát'],
    ['YOLO11s (Baseline độc lập)', 'Detection (5 lớp)', '3 (42, 123, 2026)', 'Đã hoàn thành (1.743 ảnh): mAP@50 = 84,94% ± 0,09; mAP@50:95 = 61,45% ± 0,16', 'Đã hoàn thành (Notebook 06, 871 ảnh): mAP@50 = 82,78% ± 0,40; mAP@50:95 = 60,37% ± 0,42; AR@100 = 69,38% ± 0,21', 'Đạt hiệu năng phát hiện cao nhất và rất ổn định qua 3 seeds trên cả Val và Test'],
    ['RT-DETR-L (Baseline độc lập)', 'Detection (5 lớp)', '1 (seed 42)', 'Đã hoàn thành (1.743 ảnh): mAP@50 = 71,20%; mAP@50:95 = 49,43%', 'Đã hoàn thành (Notebook 06, 871 ảnh): mAP@50 = 70,13%; mAP@50:95 = 48,57%; AR@100 = 69,55%', 'Baseline Transformer 1 seed có AR@100 tương đương YOLO11s nhưng mAP thấp hơn'],
    ['Deformable DETR (Frozen DINOv2 ViT-S/14)', 'Detection (5 lớp)', '1 (seed 42 tham chiếu)', 'Đã hoàn thành (1.743 ảnh): Không NMS đạt 71,45% / 43,35%; Có NMS (0,70) đạt 76,17% / 44,45%', 'CHƯA thực hiện trong bộ Notebook 06 chốt (trình bày trung thực, không ngoại suy)', 'NMS IoU = 0,70 cải thiện rõ rệt mAP@50 (+4,72 điểm %) trên tập Validation sạch'],
    ['Pipeline Cascade & Crop Ablation (A–E)', 'Detection → Crop → Cls (10→5 lớp)', '1 (checkpoint cố định)', 'Đã hoàn thành (Notebook 04, 1.743 ảnh): Crop C đạt mAP@50 = 46,69%, mAP@50:95 = 26,55% (cao nhất 5 biến thể)', 'CHƯA thực hiện Final Test cho Cascade với Crop C trong bộ Notebook 06 chốt', 'Chứng minh trên Validation rằng mở rộng ngữ cảnh 15% (Crop C) giúp giảm lệch miền crop (+2,70 điểm % mAP@50 so với Crop A)'],
]

TABLE_A1_NOTEBOOKS_AND_CODE = [
    ['Tên tệp Notebook / Module mã nguồn', 'Vị trí lưu trữ cục bộ / Gói Colab', 'Chức năng kỹ thuật và nội dung thực thi chính'],
    ['00_DG38_Preflight_Colab.ipynb', 'evaluation_runs/dg38_additional_experiments/ & DG38_Colab_Package', 'Kiểm tra GPU, gắn kết Google Drive, giải nén dữ liệu về ổ đĩa cục bộ Colab (/content/dg38_data), xác minh manifest và mã băm SHA-256'],
    ['01_Classification_Baselines_Colab.ipynb', 'evaluation_runs/dg38_additional_experiments/ & DG38_Colab_Package', 'Huấn luyện và đánh giá Validation cho ResNet50 và DINOv2 ViT-B/14 (4 blocks) qua 3 random seeds (42, 123, 2026) — tổng cộng 6 runs'],
    ['02_Detection_Baselines_Colab.ipynb', 'evaluation_runs/dg38_additional_experiments/ & DG38_Colab_Package', 'Huấn luyện và đánh giá trên Validation sạch (1.743 ảnh) cho YOLO11s (3 seeds: 42, 123, 2026) và RT-DETR-L (1 seed: 42) — tổng cộng 4 runs'],
    ['03_DINOv2_Finetune_Ablation_Colab.ipynb', 'evaluation_runs/dg38_additional_experiments/ & DG38_Colab_Package', 'Thực thi nghiên cứu Ablation số khối Transformer mở khóa (0, 2, 4 blocks) của DINOv2 ViT-B/14 trên tập Classification Validation (1.226 ảnh)'],
    ['04_Crop_Preprocessing_Ablation_Colab.ipynb', 'evaluation_runs/dg38_additional_experiments/ & DG38_Colab_Package', 'Đánh giá 5 biến thể tiền xử lý vùng cắt (Crop A, B, C, D, E) trong Pipeline Cascade trên 1.743 ảnh Validation sạch với NMS IoU = 0,70'],
    ['05_DG38_Results_Analysis_Colab.ipynb', 'evaluation_runs/dg38_additional_experiments/ & DG38_Colab_Package', 'Tổng hợp Mean ± Sample Std (ddof = 1), kiểm toán 14/14 runs hoàn thành, xuất bảng tổng hợp và biểu đồ phân tích'],
    ['06_DG38_Final_Test_Colab.ipynb', 'C:\\Users\\hoang\\Downloads\\06_DG38_Final_Test_Colab.ipynb & DG38_Colab_Package', 'Xác minh SHA-256 của 2 gói Test khóa và chạy đánh giá Final Test một lần cho 10 checkpoints (ResNet50 × 3, DINOv2 × 3, YOLO11s × 3, RT-DETR-L × 1)'],
    ['dg38_cls_trainer.py', 'evaluation_runs/dg38_additional_experiments/dg38_cls_trainer.py', 'Module huấn luyện chuẩn hóa cho DINOv2 ViT-B/14 và ResNet50, hỗ trợ đóng băng/mở khóa N blocks cuối, AMP fp16, lưu best checkpoint theo Validation Macro F1'],
    ['dg38_det_trainer.py', 'evaluation_runs/dg38_additional_experiments/dg38_det_trainer.py', 'Module chuyển đổi nhãn COCO → YOLO chuẩn xác, huấn luyện Ultralytics YOLO11s / RT-DETR-L và chấm điểm độc lập bằng pycocotools.COCOeval'],
    ['dg38_crop_ablation.py', 'evaluation_runs/dg38_additional_experiments/dg38_crop_ablation.py', 'Module định nghĩa và đánh giá 5 biến thể tiền xử lý vùng cắt (A_original_crop đến E_context_pad_25_letterbox) trên tập Validation sạch'],
    ['dg38_recover_locked_test.py', 'dg38_recover_locked_test.py (thư mục gốc dự án)', 'Kịch bản khôi phục tất định, kiểm tra chống rò rỉ dữ liệu và đóng gói hai tệp ZIP Final Test khóa kèm báo cáo SHA-256'],
    ['src/model.py & detection/Train.py & detection/waste_cascade.py', 'src/ và detection/ (thư mục gốc dự án)', 'Mã nguồn gốc định nghĩa GarbageClassifier (DINOv2 ViT-B/14), DeformableDETR (Frozen DINOv2 ViT-S/14 + MSDeformAttn pure PyTorch grid_sample) và Cascade'],
]

TABLE_B1_PROVENANCE_AND_HASHES = [
    ['Hạng mục tài nguyên / Tệp minh chứng', 'Đường dẫn cục bộ (Windows) hoặc Google Drive', 'Mã băm SHA-256 / Định danh kiểm toán', 'Trạng thái xác minh thực tế'],
    ['Gói Classification Final Test đã khóa (1.226 ảnh)', 'D:\\Download Acer\\DG38_test_export\\dg38_cls_test_locked.zip (và trên Drive: DG38_Colab_Package/zips/)', '98ca263b1c76aece593280ac456a0a13ec4bbbbe31a29ccdd33ec2b58cefd45c', 'ĐÃ XÁC MINH CỤC BỘ trên Windows & khớp 100% với Notebook 06'],
    ['Manifest Classification Test (cls_test.jsonl)', 'Bên trong dg38_cls_test_locked.zip (1.226 bản ghi, split signature: 1da7e63c...60c7)', '8c3fcf190341a084d1f3b65f1f2a9515ed125aa8ec55e80eb67b324394eb5ab6', 'ĐÃ XÁC MINH CỤC BỘ qua locked_test_packaging_report_both.json'],
    ['Gói Detection Final Test đã khóa (871 ảnh, 1.884 box)', 'D:\\Download Acer\\DG38_test_export\\dg38_det_test_locked.zip (và trên Drive: DG38_Colab_Package/zips/)', '29b92dde3d2fe5ed44a0951363767ddc37e0dfd9efc645c9c8ad1bf8e7d4aab6', 'ĐÃ XÁC MINH CỤC BỘ trên Windows & khớp 100% với Notebook 06'],
    ['Nhãn COCO Detection Final Test (annotations_coco.json)', 'WasteDetect_Dataset/test_final_candidate/annotations_coco.json', 'e26ff743f2a94ef9a6069a95cf3c3c69581e43295ff48bb5298b301bfde3eec1', 'ĐÃ XÁC MINH CỤC BỘ trên Windows (871 ảnh, 1.884 GT boxes)'],
    ['Nhãn COCO Detection Validation sạch (annotations_coco.json)', 'WasteDetect_Dataset/valid_clean_candidate/annotations_coco.json', '9faa0cc8aade9802f639f7ba6e1687c4b3939fa6f3b025f6a377213e2524c514', 'ĐÃ XÁC MINH CỤC BỘ trên Windows (1.743 ảnh, 3.932 GT boxes)'],
    ['Checkpoint Classifier tham chiếu (1 seed)', 'checkpoints/dinov2_garbage_final.pth', 'bee29bc7364fa18d374e68bee7f435282917fd3d98b2a83dbca724f9e2f0a0b4', 'ĐÃ XÁC MINH CỤC BỘ (ViT-B/14, 768 chiều, 10 lớp)'],
    ['Checkpoint Detector Deformable DETR tham chiếu (1 seed)', 'detection/checkpoints_wastedetect/best_model.pt', '218aa21101d27ffffe43b827300c8769e168a9d8998d688463e2321b54f7ff12', 'ĐÃ XÁC MINH CỤC BỘ (Frozen ViT-S/14 + Def. DETR, epoch 29)'],
    ['Báo cáo kiểm toán Validation & Lỗi Cascade', 'evaluation_runs/val_grid_search_01/summary.csv & cascade_error_analysis.json', '7770fb3219ae1064550e26f99477b... (locked_parameters.json)', 'ĐÃ XÁC MINH CỤC BỘ (đầy đủ 4 cấu hình & ma trận 5×10)'],
    ['Tài liệu tổng kết thực nghiệm Colab & Notebook 06', 'C:\\Users\\hoang\\Downloads\\DG38_Bao_Cao_Tong_Ket_Thuc_Nghiem_FINAL.docx & 06_DG38_Final_Test_Colab.ipynb', 'Bản tổng kết chính thức kết quả Notebooks 00–06', 'ĐÃ ĐỌC VÀ ĐỐI CHIẾU CỤC BỘ trên Windows'],
    ['Thư mục kết quả chi tiết và 10 checkpoints mới trên Google Drive', 'DG38_Colab_Package/results/ (05_DG38_results_analysis, 06_final_audit, 07_final_test_locked) & artifacts/runs/', 'Lưu trữ trên Google Drive (ID: 1cWhoh7VzE-assoAFRw8N847hCSjP5ExF)', 'CHƯA ĐỒNG BỘ VỀ WINDOWS (sử dụng số liệu đã xuất trong Notebook 06, báo cáo tổng kết và dữ liệu tham chiếu người dùng cung cấp; không tự nhận đã đọc trực tiếp tệp trên Drive)'],
]

TABLE_C1_TEACHER_FEEDBACK_MATRIX = [
    ['STT', 'Nội dung góp ý của giảng viên (Hồ sơ DG38)', 'Giải pháp thực nghiệm và chỉnh sửa báo cáo của nhóm', 'Bằng chứng truy xuất trong báo cáo (Mục / Bảng / Hình / Tệp)', 'Mức độ hoàn thành (Đánh giá trung thực)'],
    ['1a', 'Bổ sung baseline độc lập cho bài toán Classification trên cùng giao thức dữ liệu', 'Huấn luyện và đánh giá ResNet50 (ImageNet-1K) cùng tiền xử lý, cùng hàm mất mát và cùng tập Train/Val/Test (9.807 / 1.226 / 1.226 ảnh) để đối chứng trực tiếp với DINOv2 ViT-B/14', 'Chương 4 (Mục 4.2, Bảng 5), Chương 5 (Mục 5.1, Bảng 10, Hình 4); mã nguồn dg38_cls_trainer.py; Notebook 01 & 06', 'Hoàn thành đầy đủ (100% — đủ 3 seeds trên cả Validation và Final Test)'],
    ['1b', 'Bổ sung baseline độc lập cho bài toán Detection trên cùng giao thức đánh giá COCO', 'Huấn luyện và đánh giá hai baseline phát hiện độc lập đại diện cho hai trường phái: YOLO11s (CNN one-stage) và RT-DETR-L (Transformer real-time) trên cùng bộ dữ liệu WasteDetect sạch', 'Chương 4 (Mục 4.3, Bảng 6), Chương 5 (Mục 5.4, Bảng 14, Bảng 15, Bảng 16, Hình 10); mã nguồn dg38_det_trainer.py; Notebook 02 & 06', 'Hoàn thành đầy đủ trên Validation và Final Test (YOLO11s đủ 3 seeds; RT-DETR-L có 1 seed)'],
    ['2', 'Thực hiện huấn luyện lặp lại với nhiều hạt giống ngẫu nhiên (Multiple Random Seeds) và báo cáo Mean ± Std', 'Thiết lập 3 random seeds độc lập (42, 123, 2026) cho ResNet50 (3 seeds), DINOv2 ViT-B/14 (3 seeds) và YOLO11s (3 seeds); tính trung bình và độ lệch chuẩn mẫu (ddof = 1). Riêng RT-DETR-L và Deformable DETR giữ 1 seed và ghi rõ không tính Std', 'Chương 3 (Mục 3.6, Bảng 4), Chương 5 (Bảng 10, Bảng 14, Bảng 15), Chương 6 (Mục 6.1, 6.3); Notebook 01, 02, 05, 06', 'Hoàn thành cho ResNet50, DINOv2 ViT-B/14 và YOLO11s (3 seeds); hoàn thành một phần cho RT-DETR-L và Deformable DETR (1 seed do giới hạn tài nguyên tính toán)'],
    ['3a', 'Thực hiện nghiên cứu Ablation thay đổi số khối Transformer được fine-tune trên DINOv2', 'Khảo sát có kiểm soát 3 cấu hình mở khóa trên DINOv2 ViT-B/14: 0 blocks (Linear Probe, 0,01% tham số), 2 blocks cuối (16,38% tham số) và 4 blocks cuối (32,75% tham số) trên tập Classification Validation (1.226 ảnh)', 'Chương 4 (Mục 4.2), Chương 5 (Mục 5.5, Bảng 17, Hình 11); Notebook 03 (03_DINOv2_Finetune_Ablation_Colab.ipynb)', 'Hoàn thành đầy đủ trên tập Validation (96,46% → 96,99% → 97,33% Macro F1)'],
    ['3b', 'Thực hiện nghiên cứu Ablation phương pháp tiền xử lý vùng cắt (Crop Preprocessing) để kiểm chứng giả thuyết Crop Domain Shift', 'Thiết kế và đánh giá 5 biến thể tiền xử lý vùng cắt (A: Original Crop, B: Direct Resize, C: Context Padding 15% + Direct Resize, D: Letterbox 224, E: Context Padding 25% + Letterbox) trên 1.743 ảnh Validation sạch với checkpoint cố định', 'Chương 4 (Mục 4.5, Bảng 9), Chương 5 (Mục 5.6, Bảng 18, Hình 12); mã nguồn dg38_crop_ablation.py; Notebook 04', 'Hoàn thành đầy đủ trên tập Validation sạch (Crop C đạt 46,69% mAP@50, cải thiện +2,70 điểm % so với Crop A)'],
    ['4', 'Phân biệt rạch ròi Train / Validation / Final Test, khóa tập Test chống rò rỉ và không dùng Test để chọn tham số', 'Kiểm toán khử trùng SHA-256 + pHash ≤ 6, xóa box trùng ID 1847; đóng gói khóa 2 tệp ZIP Final Test bằng SHA-256; chọn mọi siêu tham số, NMS IoU = 0,70 và Crop C trên Validation', 'Chương 3 (Mục 3.2, 3.5, 3.6, Bảng 2, Bảng 3), Phụ lục B (Bảng B1); kịch bản dg38_recover_locked_test.py', 'Hoàn thành nghiêm ngặt (100% đối với toàn bộ thí nghiệm trong báo cáo)'],
    ['5', 'Đánh giá Final Test đầu-cuối cho nhánh kiến trúc đề xuất Deformable DETR và Pipeline Cascade (với Crop C)', 'Đã hoàn thành phân tích toàn diện trên Validation sạch (1.743 ảnh), Factorial Ablation 6 cấu hình và ma trận chẩn đoán 5×10. Tuy nhiên, chưa chạy lại Final Test mới trên gói Test khóa cho Deformable DETR và Cascade Crop C trong Notebook 06', 'Chương 5 (Mục 5.7, 5.8, Bảng 19–23, Hình 16–19), Chương 6 (Mục 6.1, 6.3), Chương 7 (Mục 7.2)', 'Hoàn thành trên Validation; CHƯA hoàn thành Final Test mới cho Deformable DETR & Cascade Crop C (được công bố trung thực là giới hạn nghiên cứu)'],
]

REFERENCES_LIST = [
    '[1] Islam, M. M., Hasan, S. M. M., Hossain, M. R., Uddin, M. P., & Mamun, M. A. (2025). Towards sustainable solutions: Effective waste classification framework via enhanced deep convolutional neural networks. PLOS ONE, 20(6), e0324294. https://doi.org/10.1371/journal.pone.0324294',
    '[2] Oquab, M., Darcet, T., Moutakanni, T., Vo, H. V., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., Assran, M., Ballas, N., Galuba, W., Howes, R., Huang, P.-Y., Li, S.-W., Misra, I., Rabbat, M., Sharma, V., Synnaeve, G., Xu, H., Jegou, H., Mairal, J., Labatut, P., Joulin, A., & Bojanowski, P. (2024). DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research (TMLR). https://arxiv.org/abs/2304.07193',
    '[3] Zhu, X., Su, W., Lu, L., Li, B., Wang, X., & Dai, J. (2021). Deformable DETR: Deformable transformers for end-to-end object detection. In International Conference on Learning Representations (ICLR 2021). https://arxiv.org/abs/2010.04159',
    '[4] Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., & Zagoruyko, S. (2020). End-to-end object detection with transformers. In European Conference on Computer Vision (ECCV 2020) (pp. 213–229). Springer. https://arxiv.org/abs/2005.12872',
    '[5] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations (ICLR 2021). https://arxiv.org/abs/2010.11929',
    '[6] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2016) (pp. 770–778). https://doi.org/10.1109/CVPR.2016.90',
    '[7] Zhao, Y., Lv, W., Xu, S., Wei, J., Wang, G., Dang, Q., Liu, Y., & Chen, J. (2024). DETRs beat YOLOs on real-time object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2024) (pp. 16965–16974). https://arxiv.org/abs/2304.08069',
    '[8] Jocher, G., Qiu, J., & Chaurasia, A. (2024). Ultralytics YOLO11 (Version 8.3.0) [Computer software]. Ultralytics. https://github.com/ultralytics/ultralytics',
    '[9] Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. In International Conference on Learning Representations (ICLR 2019). https://arxiv.org/abs/1711.05101',
    '[10] Kunwar, S. (2023). Garbage Classification v2 (Waste Classification V2 Dataset). Kaggle Datasets. https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2',
    '[11] Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., & Zitnick, C. L. (2014). Microsoft COCO: Common objects in context. In European Conference on Computer Vision (ECCV 2014) (pp. 740–755). Springer. https://arxiv.org/abs/1405.0312',
    '[12] COCO Consortium. (2024). COCOeval reference implementation (pycocotools/cocoeval.py). GitHub Repository. https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py',
    '[13] Nhóm 1 – Lớp 22ES (2026). Mã nguồn triển khai, kịch bản huấn luyện đa hạt giống và gói thực nghiệm DG38: src/model.py, detection/Train.py, detection/waste_cascade.py, dg38_recover_locked_test.py, evaluation_runs/dg38_additional_experiments/ (Notebooks 00–06). Trường Đại học Bách khoa – Đại học Đà Nẵng.',
    '[14] WasteDetect_Dataset (2026). Bộ dữ liệu nhãn đối tượng chuẩn COCO cục bộ và các phiên bản sau kiểm toán khử trùng lặp đa tầng (valid_clean_candidate: 1.743 ảnh / 3.932 boxes; test_final_candidate: 871 ảnh / 1.884 boxes). Hồ sơ kiểm toán lưu tại evaluation_runs/val_grid_search_01/ và D:\\Download Acer\\DG38_test_export\\.',
]
