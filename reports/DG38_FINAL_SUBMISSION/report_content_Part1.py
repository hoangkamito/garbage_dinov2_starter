# -*- coding: utf-8 -*-
"""
Part 1 of the complete academic report content for DG38:
Front Matter (List of Figures/Tables, Abbreviations, Vietnamese Abstract, English Abstract)
and Chapters 1, 2, 3, 4.
"""

ABBREVIATIONS_ROWS = [
    ['Ký hiệu / Từ viết tắt', 'Thuật ngữ tiếng Anh đầy đủ', 'Ý nghĩa tiếng Việt và phạm vi áp dụng'],
    ['AP / mAP', 'Average Precision / mean Average Precision', 'Độ chính xác trung bình theo diện tích dưới đường cong P–R; chỉ số chuẩn của bài toán Detection (5 lớp)'],
    ['mAP@50', 'mean Average Precision at IoU = 0.50', 'mAP tính tại ngưỡng giao trên hợp IoU = 0,50 theo chuẩn PASCAL VOC / COCO'],
    ['mAP@50:95', 'mean Average Precision across IoU 0.50:0.05:0.95', 'Chỉ số chính của chuẩn COCO, lấy trung bình mAP qua 10 ngưỡng IoU từ 0,50 đến 0,95 (bước nhảy 0,05)'],
    ['AR@100', 'Average Recall at maxDets = 100', 'Độ nhạy trung bình qua các ngưỡng IoU 0,50:0,95 với tối đa 100 dự đoán trên mỗi ảnh'],
    ['CLS', 'Classification Token ([CLS])', 'Token đại diện toàn cục ở đầu ra của kiến trúc Vision Transformer (768 chiều với ViT-B/14; 384 chiều với ViT-S/14)'],
    ['CNN', 'Convolutional Neural Network', 'Mạng nơ-ron tích chập; đại diện trong báo cáo bởi baseline ResNet50 và nhánh đặc trưng trong YOLO11s'],
    ['CU', 'Compute Units (Google Colab Pro)', 'Đơn vị tính toán trên nền tảng đám mây Google Colab Pro dùng để quản trị ngân sách huấn luyện GPU L4'],
    ['DETR', 'DEtection TRansformer', 'Họ mô hình phát hiện đối tượng đầu-cuối dựa trên Transformer và ghép cặp tối ưu một-đối-một Hungarian'],
    ['DINOv2', 'Self-Distillation with No Labels v2', 'Mô hình nền tảng thị giác tự giám sát của Meta AI (Oquab và cs., 2024)'],
    ['FP / FN / TP', 'False Positive / False Negative / True Positive', 'Dự đoán dương tính giả, âm tính giả và dương tính thật tại một điểm vận hành xác định'],
    ['IoU', 'Intersection over Union', 'Tỷ số diện tích phần giao trên phần hợp giữa bounding box dự đoán và nhãn thực (Ground Truth)'],
    ['Macro F1', 'Macro-averaged F1-score', 'Trung bình cộng không trọng số của F1-score qua toàn bộ các lớp; chỉ số đánh giá chính cho Classification 10 lớp'],
    ['MSDeformAttn', 'Multi-Scale Deformable Attention', 'Mô-đun chú ý biến dạng đa tỷ lệ trong Deformable DETR (triển khai bằng pure PyTorch F.grid_sample trong dự án)'],
    ['NMS', 'Non-Maximum Suppression', 'Thuật toán lọc triệt tiêu phi cực đại nhằm loại bỏ các bounding box dự đoán chồng lấp dư thừa'],
    ['OOS', 'Out-of-Scope Classes', 'Nhóm 5 lớp phân loại (battery, cardboard, clothes, shoes, trash) nằm ngoài không gian 5 lớp của bộ dữ liệu Detection'],
    ['pHash', 'Perceptual Hash', 'Mã băm cảm nhận thị giác 64-bit dùng để phát hiện các cặp ảnh gần trùng nhau (ngưỡng khoảng cách Hamming ≤ 6)'],
    ['RT-DETR', 'Real-Time DEtection TRansformer', 'Kiến trúc phát hiện đối tượng Transformer thời gian thực sử dụng bộ mã hóa lai (Zhao và cs., 2024)'],
    ['SHA-256', 'Secure Hash Algorithm 256-bit', 'Thuật toán băm mật mã 256-bit dùng để kiểm toán trùng lặp tệp tuyệt đối và khóa gói dữ liệu Final Test'],
    ['Std (Sample Std)', 'Sample Standard Deviation (ddof = 1)', 'Độ lệch chuẩn mẫu tính trên n = 3 hạt giống ngẫu nhiên (seeds 42, 123, 2026); không áp dụng cho cấu hình 1 seed'],
    ['ViT', 'Vision Transformer', 'Kiến trúc biến đổi thị giác chia ảnh thành chuỗi các mảng (patches) phẳng (Dosovitskiy và cs., 2021)'],
    ['YOLO11s', 'You Only Look Once version 11 (small)', 'Mô hình phát hiện đối tượng một giai đoạn (one-stage detector) của Ultralytics dùng làm baseline độc lập'],
]

LIST_OF_TABLES_ROWS = [
    ['Số hiệu', 'Tên bảng số liệu', 'Phạm vi tập dữ liệu & Ghi chú'],
    ['Bảng 1', 'Phân bố số ảnh theo 10 lớp trong Garbage Classification v2 và quy tắc ánh xạ sang hệ 5 lớp phát hiện', 'Train / Val / Test (12.259 ảnh)'],
    ['Bảng 2', 'Quy mô và vai trò các tập dữ liệu phân tầng đã khóa của Classification (10 lớp) và Detection (5 lớp)', 'Manifest đã khóa SHA-256'],
    ['Bảng 3', 'Thống kê chi tiết bộ dữ liệu phát hiện WasteDetect qua các giai đoạn kiểm toán khử trùng lặp đa tầng', 'Train / Valid sạch / Final Test'],
    ['Bảng 4', 'Tổng hợp ma trận thiết kế thực nghiệm bổ sung theo góp ý giảng viên (Notebooks 00–06)', 'Toàn bộ gói thực nghiệm DG38'],
    ['Bảng 5', 'Cấu hình kiến trúc và siêu tham số huấn luyện nhánh phân loại: DINOv2 ViT-B/14 và Baseline ResNet50', 'Classification (3 seeds: 42, 123, 2026)'],
    ['Bảng 6', 'Cấu hình kiến trúc và siêu tham số huấn luyện nhánh phát hiện: Deformable DETR, YOLO11s và RT-DETR-L', 'Detection (5 lớp WasteDetect)'],
    ['Bảng 7', 'So sánh cấu hình thực nghiệm giữa bài báo tham chiếu Islam và cs. (2025) và mô hình DINOv2 của nhóm', 'Đối chiếu tham khảo học thuật'],
    ['Bảng 8', 'Cấu hình các phương pháp trong nhánh Deformable DETR / Cascade và bộ thông số khóa từ Validation', 'Khóa trên Validation sạch (1.743 ảnh)'],
    ['Bảng 9', 'Đặc tả kỹ thuật 5 biến thể tiền xử lý vùng cắt (Crop Variants A–E) khảo sát trong Pipeline Cascade', 'Validation sạch (1.743 ảnh)'],
    ['Bảng 10', 'Kết quả Classification (10 lớp) của ResNet50 và DINOv2 ViT-B/14 qua 3 seeds trên Validation và Final Test', 'Validation & Final Test (Mean ± Std, n=3)'],
    ['Bảng 11', 'Kết quả phân loại chi tiết theo từng lớp (10 lớp) của checkpoint tham chiếu DINOv2 ViT-B/14 (seed 42)', 'Classification Test (1.226 ảnh, 1 seed)'],
    ['Bảng 12', 'Đối chiếu tham khảo các chỉ số tổng thể giữa bài báo Islam và cs. (2025) và mô hình DINOv2 của nhóm', 'Đối chiếu tham khảo (khác split)'],
    ['Bảng 13', 'Đối chiếu tham khảo theo từng lớp trên 5 lớp rác chung giữa bài báo Islam và cs. (2025) và DINOv2', 'Đối chiếu tham khảo (5 lớp chung)'],
    ['Bảng 14', 'Kết quả đánh giá bài toán Detection (5 lớp) trên tập Validation sạch của YOLO11s, RT-DETR-L và Def. DETR', 'Validation sạch (1.743 ảnh, 3.932 box)'],
    ['Bảng 15', 'Kết quả Detection Final Test đã khóa theo từng seed và tổng hợp Mean ± Sample Std (YOLO11s & RT-DETR-L)', 'Final Test đã khóa (871 ảnh, 1.884 box)'],
    ['Bảng 16', 'Đối chiếu tính nhất quán giữa tập Validation sạch và Final Test đã khóa của YOLO11s và RT-DETR-L', 'Validation vs. Final Test'],
    ['Bảng 17', 'Kết quả Ablation 1: Ảnh hưởng của số Transformer blocks được fine-tune (0, 2, 4 blocks) trên DINOv2 ViT-B/14', 'Classification Validation (1.226 ảnh)'],
    ['Bảng 18', 'Kết quả Ablation 2: Ảnh hưởng của 5 phương pháp tiền xử lý vùng cắt (Crop A–E) trong Pipeline Cascade', 'Detection Validation sạch (1.743 ảnh)'],
    ['Bảng 19', 'Kết quả định lượng chi tiết 4 cấu hình Deformable DETR và Cascade trên Validation sạch kèm đối chiếu Crop C', 'Detection Validation sạch (1.743 ảnh)'],
    ['Bảng 20', 'Hiệu năng phát hiện chi tiết theo từng lớp rác (AP@50, AP@50:95, AR@100) trên tập Validation sạch', 'Detection Validation sạch (1.743 ảnh)'],
    ['Bảng 21', 'Kết quả phân tích tách biệt nhân tố (Factorial Ablation — 6 cấu hình) của nhánh Cascade trên Validation', 'Detection Validation sạch (1.743 ảnh)'],
    ['Bảng 22', 'Ma trận nhầm lẫn chẩn đoán giữa 5 lớp GT và 10 lớp dự đoán DINOv2 trên 3.663 GT boxes ở Validation', 'Detection Validation sạch (3.663 box)'],
    ['Bảng 23', 'Tổng hợp trạng thái đánh giá cuối cùng (Validation vs. Final Test) của toàn bộ các mô hình trong đồ án', 'Tổng hợp minh bạch phạm vi kết luận'],
    ['Bảng A1', 'Danh mục các Google Colab Notebooks (00–06) và các mô-đun mã nguồn thực nghiệm của dự án DG38', 'Phụ lục A — Minh chứng mã nguồn'],
    ['Bảng B1', 'Hồ sơ truy xuất nguồn gốc tệp dữ liệu, mã băm SHA-256 và trạng thái đồng bộ Google Drive – Windows', 'Phụ lục B — Hồ sơ tái lập'],
    ['Bảng C1', 'Ma trận đối chiếu chi tiết việc hoàn thành các yêu cầu chỉnh sửa và bổ sung của giảng viên', 'Phụ lục C — Đối chiếu góp ý GV'],
]

LIST_OF_FIGURES_ROWS = [
    ['Số hiệu', 'Tên hình vẽ / Biểu đồ trực quan', 'Nguồn dữ liệu / Tệp minh chứng'],
    ['Hình 1', 'Kiến trúc tổng thể hệ thống Pipeline Cascade hai giai đoạn và hai nhánh đối chứng độc lập DG38', 'Sơ đồ kiến trúc tổng hợp toàn dự án'],
    ['Hình 2', 'Pipeline phân loại ảnh đơn đối tượng DINOv2 ViT-B/14 (embedding 768 chiều, tinh chỉnh 4 block cuối)', 'Dựng từ src/model.py & notebook'],
    ['Hình 3', 'Kiến trúc bộ phát hiện đa tỷ lệ DINOv2 ViT-S/14 (frozen backbone) kết hợp Deformable DETR', 'Dựng từ detection/Train.py'],
    ['Hình 4', 'So sánh Classification (10 lớp): ResNet50 vs. DINOv2 ViT-B/14 trên Validation và Final Test (3 seeds)', 'Notebook 01, 05 & 06 (Mean ± Std)'],
    ['Hình 5', 'Classification Report chi tiết của checkpoint tham chiếu DINOv2 ViT-B/14 (seed 42) trên 1.226 ảnh Test', 'scientific_verification_report.json'],
    ['Hình 6', 'Confusion Matrix của DINOv2 ViT-B/14 trên 1.226 ảnh Classification Test (1.210 ảnh đúng, 16 ảnh sai)', 'scientific_verification_report.json'],
    ['Hình 7', 'Biểu đồ chỉ số tổng thể của mô hình DenseNet201 + Parallel SE-CNN trong bài báo Islam và cs. (2025)', 'PLOS ONE 2025, Fig 6 [1]'],
    ['Hình 8', 'Confusion Matrix trên tập Waste Classification V2 của bài báo tham chiếu Islam và cs. (2025)', 'PLOS ONE 2025, Fig 5 [1]'],
    ['Hình 9', 'So sánh tham khảo Recall và F1-score trên 5 lớp chung giữa bài báo [1] và DINOv2 ViT-B/14 của nhóm', 'Bảng 13 (khác tập kiểm thử)'],
    ['Hình 10', 'So sánh Detection Baselines (5 lớp): YOLO11s (3 seeds) vs. RT-DETR-L (1 seed) trên Validation và Test', 'Notebook 02, 05 & 06'],
    ['Hình 11', 'Kết quả Ablation 1: Ảnh hưởng của số Transformer blocks fine-tune (0, 2, 4 blocks) trên DINOv2 ViT-B/14', 'Notebook 03 (Validation 1.226 ảnh)'],
    ['Hình 12', 'Kết quả Ablation 2: Ảnh hưởng của 5 biến thể tiền xử lý vùng cắt (Crop A–E) trong Pipeline Cascade', 'Notebook 04 (Validation sạch 1.743 ảnh)'],
    ['Hình 13', 'Diễn biến chỉ số mAP@50 và mAP@50:95 qua 29 epoch được lưu trên tập Validation gốc (2.316 ảnh)', 'training_history.json (epoch 1–29)'],
    ['Hình 14', 'Diễn biến Precision@50, Recall@50 và F1@50 tại score = 0,05 qua 29 epoch được lưu trên Validation gốc', 'training_history.json (epoch 1–29)'],
    ['Hình 15', 'Diễn biến Train Loss và Validation Loss qua 29 epoch được lưu trong training_history.json', 'training_history.json (epoch 1–29)'],
    ['Hình 16', 'So sánh hiệu năng nhánh Deformable DETR và Pipeline Cascade (Crop A vs. Crop C) trên Validation sạch', 'val_grid_search_01 & Notebook 04'],
    ['Hình 17', 'Phân rã AP@50 theo từng lớp rác trên tập Validation sạch (1.743 ảnh) giữa Deformable DETR và Cascade', 'cascade_error_analysis.json'],
    ['Hình 18', 'Phân rã cơ chế suy giảm mAP của kiến trúc Cascade trên Validation sạch: ma trận 5×10 (a) và nhân tố (b)', 'cascade_error_analysis.json'],
    ['Hình 19', 'Trực quan hóa định tính 4 trường hợp điển hình (Case A, B, C, D) của hệ thống phát hiện và phân loại', 'Trích xuất trực quan 4 trường hợp'],
    ['Hình A1', 'Mã nguồn ViT-B/14 thực tế, cấu hình fine-tune 4 block trong notebook và đối chiếu kích thước checkpoint', 'src/model.py & notebook cells 50–53'],
]

TABLE_2_SPLIT_SUMMARY = [
    ['Tác vụ & Bộ dữ liệu', 'Tập phân chia (Split)', 'Số lượng ảnh', 'Số lượng Bounding Box', 'Tỷ lệ / Trạng thái kiểm toán & Khóa Manifest'],
    ['Classification — Garbage Classification v2 (10 lớp đơn đối tượng)', 'Train (cls_train)', '9.807', 'Không áp dụng (nhãn mức ảnh)', '80,00% — Phân tầng (stratified), kiểm tra SHA-256 tách biệt'],
    ['Classification — Garbage Classification v2 (10 lớp đơn đối tượng)', 'Validation (cls_val)', '1.226', 'Không áp dụng (nhãn mức ảnh)', '10,00% — Dùng để chọn checkpoint và thực hiện Ablation 0/2/4 blocks'],
    ['Classification — Garbage Classification v2 (10 lớp đơn đối tượng)', 'Final Test (cls_test)', '1.226', 'Không áp dụng (nhãn mức ảnh)', '10,00% — Đã khóa manifest & gói dg38_cls_test_locked.zip (SHA-256 xác minh)'],
    ['Classification — Garbage Classification v2 (10 lớp đơn đối tượng)', 'Tổng cộng Classification', '12.259', 'Không áp dụng', '100,00% — Đủ 10 lớp: battery, biological, cardboard, clothes, glass, metal, paper, plastic, shoes, trash'],
    ['Detection — WasteDetect_Dataset (5 lớp đa đối tượng)', 'Train (det_train)', '8.289', '18.345', 'Tập huấn luyện chuẩn COCO JSON (5 lớp: glass, metal, organic, paper, plastic)'],
    ['Detection — WasteDetect_Dataset (5 lớp đa đối tượng)', 'Validation gốc (trước làm sạch)', '2.316', '5.221', 'Chứa 573 ảnh trùng/gần trùng với Train hoặc nội bộ và 1 box trùng ID 1847'],
    ['Detection — WasteDetect_Dataset (5 lớp đa đối tượng)', 'Validation sạch (valid_clean_candidate)', '1.743', '3.932', 'Đã loại 573 ảnh trùng (506 SHA-256 + 37 nội bộ + 28 pHash + 2 nội bộ) & xóa box ID 1847'],
    ['Detection — WasteDetect_Dataset (5 lớp đa đối tượng)', 'Test gốc (trước làm sạch)', '1.195', '2.645', 'Chứa 324 ảnh trùng/gần trùng xuyên tập với Train, Validation và Classification'],
    ['Detection — WasteDetect_Dataset (5 lớp đa đối tượng)', 'Final Test đã khóa (test_final_candidate)', '871', '1.884', 'Đã loại 324 ảnh trùng (286 SHA-256 + 5 nội bộ + 31 pHash + 2 xuyên tập); khóa gói dg38_det_test_locked.zip'],
]

TABLE_4_EXPERIMENT_MATRIX = [
    ['Notebook / Giai đoạn', 'Mục tiêu thực nghiệm (Đáp ứng góp ý GV)', 'Mô hình / Cấu hình khảo sát', 'Hạt giống ngẫu nhiên (Seeds)', 'Tập dữ liệu đánh giá & Trạng thái'],
    ['00_DG38_Preflight_Colab.ipynb', 'Kiểm tra môi trường, giải nén dữ liệu, xác minh manifest và mã băm SHA-256 trước khi huấn luyện', 'Kiểm toán tự động toàn bộ gói dữ liệu và cây thư mục', 'Cố định manifest (seed gốc 42)', 'Hoàn thành 100% — 0 cảnh báo rò rỉ dữ liệu'],
    ['01_Classification_Baselines_Colab.ipynb', 'Yêu cầu 1 & 2: Bổ sung baseline phân loại độc lập và huấn luyện lặp lại với 3 random seeds', 'ResNet50 (ImageNet-1K) vs. DINOv2 ViT-B/14 (fine-tune 4 blocks cuối)', '3 seeds độc lập: 42, 123, 2026 (tổng 6 runs)', 'Hoàn thành 6/6 runs — Chọn best checkpoint trên Validation (1.226 ảnh)'],
    ['02_Detection_Baselines_Colab.ipynb', 'Yêu cầu 1 & 2: Bổ sung baseline phát hiện độc lập (CNN one-stage & Transformer) trên cùng dữ liệu', 'YOLO11s (3 seeds) và RT-DETR-L (1 seed)', 'YOLO11s: 42, 123, 2026 (3 runs); RT-DETR-L: 42 (1 run)', 'Hoàn thành 4/4 runs — Đánh giá trên Validation sạch (1.743 ảnh, 3.932 box)'],
    ['03_DINOv2_Finetune_Ablation_Colab.ipynb', 'Yêu cầu 3a: Nghiên cứu Ablation thay đổi số khối Transformer được mở khóa tinh chỉnh', 'DINOv2 ViT-B/14 với 0 blocks (Linear Probe), 2 blocks và 4 blocks', 'Cố định giao thức so sánh trên Validation (seed 42)', 'Hoàn thành — Đánh giá trên Classification Validation (1.226 ảnh)'],
    ['04_Crop_Preprocessing_Ablation_Colab.ipynb', 'Yêu cầu 3b: Nghiên cứu Ablation 5 biến thể tiền xử lý vùng cắt (Crop A–E) trong Cascade', 'Pipeline Deformable DETR → NMS (0,70) → Crop A/B/C/D/E → DINOv2 ViT-B/14', 'Checkpoint Detector & Classifier cố định', 'Hoàn thành — Chọn Crop C (+15% pad, 224×224) trên Validation sạch (1.743 ảnh)'],
    ['05_DG38_Results_Analysis_Colab.ipynb', 'Tổng hợp thống kê Mean ± Sample Std (ddof = 1), xuất bảng LaTeX/CSV và kiểm toán 14/14 runs', 'Toàn bộ 14 completed runs (6 Cls + 4 Det + 3 Finetune + 1 Crop Ablation)', 'Tổng hợp 3 seeds (n = 3) và tách riêng cấu hình 1 seed (n = 1)', 'Hoàn thành — Xác nhận 14/14 runs & checkpoints hợp lệ, 0 warnings'],
    ['06_DG38_Final_Test_Colab.ipynb', 'Đánh giá một lần trên gói Final Test đã khóa SHA-256 cho các mô hình baseline và đề xuất', 'ResNet50 (3 seeds), DINOv2 ViT-B/14 (3 seeds), YOLO11s (3 seeds), RT-DETR-L (1 seed)', '10 checkpoints đã khóa từ Validation', 'Hoàn thành cho 4 mô hình (10 checkpoints); Deformable DETR & Cascade chưa chạy Test mới'],
]

TABLE_5_CLS_CONFIG = [
    ['Thành phần / Siêu tham số', 'Mô hình đề xuất: DINOv2 ViT-B/14', 'Baseline độc lập: ResNet50'],
    ['Kiến trúc mạng xương sống (Backbone)', 'Vision Transformer ViT-B/14 (facebook/dinov2-base), 12 khối Transformer Encoder, kích thước mảng 14×14, chiều nhúng 768', 'Mạng tích chập sâu có kết nối tắt ResNet50 (torchvision ResNet50_Weights.IMAGENET1K_V2), 50 tầng tích chập, chiều đặc trưng 2.048'],
    ['Phương thức tiền huấn luyện', 'Tự giám sát không nhãn (Self-supervised learning DINOv2) trên tập LVD-142M (142 triệu ảnh chọn lọc)', 'Giám sát có nhãn (Supervised pre-training) trên tập ImageNet-1K (1,28 triệu ảnh, 1.000 lớp)'],
    ['Cấu trúc đầu phân loại (Classification Head)', 'LayerNorm(768) → Dropout(p = 0,2) → Linear(768 → 10 lớp)', 'Linear(2048 → 10 lớp) thay thế tầng fc cuối của ResNet50'],
    ['Chiến lược mở khóa tham số (Fine-tuning)', 'Chính thức (3 seeds): Mở khóa 4 khối Transformer cuối (blocks 8–11), tầng LayerNorm cuối và Head (28.360.714 / 86.589.706 tham số ≈ 32,75%). Khảo sát Ablation: 0, 2, 4 blocks', 'Fine-tune toàn bộ mạng (Full fine-tuning): 23.528.522 / 23.528.522 tham số (100%)'],
    ['Tiền xử lý và tăng cường dữ liệu (Transforms)', 'Train: RandomResizedCrop(224, scale=(0,8; 1,0)), RandomHorizontalFlip(0,5), RandomRotation(10°), ColorJitter(0,2). Val/Test: Resize(256) → CenterCrop(224) → ImageNet Normalize', 'Đồng nhất hoàn toàn với DINOv2: cùng chuỗi tăng cường dữ liệu khi Train và Resize(256) → CenterCrop(224) → ImageNet Normalize khi Val/Test'],
    ['Thuật toán tối ưu & Tốc độ học (Learning Rate)', 'AdamW (weight_decay = 1e-4); phân tầng tốc độ học: lr_backbone = 2e-5, lr_head = 2e-4 (trong dg38_cls_trainer.py); lịch giảm CosineAnnealingLR', 'AdamW (weight_decay = 1e-4); lr_backbone = 1e-4, lr_head = 5e-4; lịch giảm CosineAnnealingLR'],
    ['Hàm mất mát, Batch size & AMP', 'CrossEntropyLoss(label_smoothing = 0,05); batch_size = 64; tổng số epoch = 15; chế độ độ chính xác hỗn hợp tự động (AMP fp16)', 'CrossEntropyLoss(label_smoothing = 0,05); batch_size = 64; tổng số epoch = 15; chế độ AMP fp16'],
    ['Hạt giống ngẫu nhiên & Tiêu chí chọn mô hình', '3 hạt giống độc lập: 42, 123, 2026; lưu và khóa checkpoint đạt Validation Macro F1 cao nhất trên 1.226 ảnh Validation', '3 hạt giống độc lập: 42, 123, 2026; lưu và khóa checkpoint đạt Validation Macro F1 cao nhất trên 1.226 ảnh Validation'],
]

TABLE_6_DET_CONFIG = [
    ['Thành phần / Siêu tham số', 'Nhánh đề xuất: DINOv2 ViT-S/14 + Deformable DETR', 'Baseline độc lập 1: YOLO11s (CNN One-stage)', 'Baseline độc lập 2: RT-DETR-L (Transformer Real-time)'],
    ['Họ kiến trúc & Cơ chế phát hiện', 'Two-part Hybrid: Backbone ViT-S/14 đóng băng trích xuất 4 mức đặc trưng (blocks 3, 6, 9, 12) + 3 tầng Encoder / 3 tầng Decoder Deformable DETR (100 object queries)', 'One-stage Anchor-free CNN Detector (C3k2 + SPPF + C2PSA neck + Decoupled detection head đa tỷ lệ P3/P4/P5)', 'End-to-end Real-Time Detection Transformer: CNN backbone + Hybrid Encoder (AIFI + CCFM) + IoU-aware Query Selection (300 queries)'],
    ['Đặc điểm triển khai kỹ thuật cần lưu ý', 'Mô-đun MSDeformAttn trong detection/Train.py được triển khai bằng pure PyTorch với F.grid_sample (không phải bản CUDA C++ extension gốc của Zhu và cs.)', 'Triển khai chính thức qua thư viện Ultralytics (yolo11s.pt tiền huấn luyện trên COCO 2017)', 'Triển khai chính thức qua thư viện Ultralytics (rtdetr-l.pt tiền huấn luyện trên COCO 2017)'],
    ['Kích thước ảnh đầu vào', 'Resize nội suy song tuyến về 800 × 800 (hoặc 518 × 518 tùy chế độ cấu hình); chuẩn hóa ImageNet', 'imgsz = 640 × 640 pixel (chuẩn hóa và letterbox nội bộ theo giao thức Ultralytics)', 'imgsz = 640 × 640 pixel (chuẩn hóa nội bộ theo giao thức Ultralytics)'],
    ['Cấu hình huấn luyện & Tối ưu hóa', 'AdamW (lr = 1e-4, weight_decay = 1e-4), CosineAnnealingLR; cấu hình 30 epochs (lịch sử lưu 29 epochs trên Validation gốc)', 'Epochs = 30, batch_size = 16, workers = 4, AMP = True, patience = 10, cache = ram (trên GPU NVIDIA L4)', 'Epochs = 30, batch_size = 8, workers = 4, AMP = True, patience = 10, cache = ram (trên GPU NVIDIA L4)'],
    ['Hạt giống ngẫu nhiên (Random Seeds)', '1 seed tham chiếu (seed = 42, checkpoint best_model.pt)', '3 seeds độc lập: 42, 123, 2026 (báo cáo Mean ± Sample Std)', '1 seed độc lập: 42 (do chi phí tính toán cao của kiến trúc Transformer lớn; không tính độ lệch chuẩn)'],
    ['Cơ chế hậu xử lý & Đánh giá COCO', 'Khảo sát đối chứng không NMS và có Class-Agnostic NMS (IoU = 0,70); chấm điểm bằng pycocotools.COCOeval (maxDets = 100)', 'NMS chuẩn nội bộ của YOLO11; trích xuất dự đoán và chấm điểm độc lập bằng pycocotools.COCOeval trên cùng tệp COCO JSON', 'Dự đoán tập hợp trực tiếp (NMS-free decoding); chấm điểm độc lập bằng pycocotools.COCOeval trên cùng tệp COCO JSON'],
]

TABLE_8_CASCADE_LOCKED_CONFIG = [
    ['Phương pháp / Cấu hình', 'Nhánh mô hình phát hiện & phân loại', 'Cơ chế lọc NMS', 'Ngưỡng NMS IoU', 'Ngưỡng điểm vận hành (PR Threshold)', 'Tiền xử lý vùng cắt (Crop Variant)', 'Trạng thái đánh giá trong báo cáo'],
    ['detector_original', 'Chỉ dùng bộ phát hiện DINOv2 ViT-S/14 + Deformable DETR (5 lớp)', 'Không áp dụng (No NMS)', 'Không áp dụng', '0,30 (chọn từ Validation)', 'Không cắt ảnh (dự đoán trực tiếp từ 100 queries)', 'Đã đánh giá trên Validation sạch (1.743 ảnh); chưa chạy Final Test mới trong Notebook 06'],
    ['detector_nms', 'Chỉ dùng bộ phát hiện DINOv2 ViT-S/14 + Deformable DETR (5 lớp)', 'Class-Agnostic NMS', '0,70 (khóa từ Validation)', '0,30 (khóa từ Validation)', 'Không cắt ảnh (giữ nguyên nhãn Detector sau NMS)', 'Đã đánh giá trên Validation sạch (1.743 ảnh); đạt mAP@50 = 76,17%, mAP@50:95 = 44,45%'],
    ['cascade_original', 'Ghép nối tuần tự: Deformable DETR → Crop → DINOv2 ViT-B/14 (10→5 lớp)', 'Không áp dụng (No NMS)', 'Không áp dụng', '0,30 (khóa từ Validation)', 'Crop A — Cắt sát bbox + Resize(256) + CenterCrop(224)', 'Đã đánh giá trên Validation sạch (1.743 ảnh); đạt mAP@50 = 40,83%, mAP@50:95 = 24,09%'],
    ['cascade_nms (Crop A)', 'Ghép nối tuần tự: Deformable DETR → NMS → Crop A → DINOv2 ViT-B/14', 'Class-Agnostic NMS', '0,70 (khóa từ Validation)', '0,25 (khóa từ Validation)', 'Crop A — Cắt sát bbox + Resize(256) + CenterCrop(224)', 'Đã đánh giá trên Validation sạch (1.743 ảnh); đạt mAP@50 = 43,99%, mAP@50:95 = 24,77%'],
    ['cascade_nms (Crop C — Tối ưu)', 'Ghép nối tuần tự: Deformable DETR → NMS → Crop C → DINOv2 ViT-B/14', 'Class-Agnostic NMS', '0,70 (khóa từ Validation)', '0,25 (khóa từ Validation)', 'Crop C — Mở rộng bbox 15% mỗi phía + Clip + Resize trực tiếp 224×224', 'Cấu hình Cascade tốt nhất trên Validation sạch (mAP@50 = 46,69%, mAP@50:95 = 26,55%); chưa có Final Test mới'],
]

TABLE_9_CROP_VARIANTS_SPEC = [
    ['Mã biến thể', 'Tên phương pháp tiền xử lý vùng cắt', 'Hệ số mở rộng ngữ cảnh (Context Padding)', 'Quy trình biến đổi hình học chi tiết (Geometric Transform)', 'Đặc tính bảo toàn thông tin thị giác'],
    ['Biến thể A', 'A_original_crop (Cấu hình cơ sở ban đầu)', '0% (cắt sát tọa độ bounding box [x1, y1, x2, y2])', 'Cắt sát bbox → Resize cạnh ngắn về 256 pixel (giữ tỷ lệ khung hình) → CenterCrop(224 × 224) → Chuẩn hóa ImageNet', 'Mất ngữ cảnh nền xung quanh và bị cắt cụt thêm ~12,5% vùng rìa vật thể do bước CenterCrop(224) từ 256'],
    ['Biến thể B', 'B_direct_resize (Thay đổi kích thước trực tiếp)', '0% (cắt sát tọa độ bounding box [x1, y1, x2, y2])', 'Cắt sát bbox → Resize trực tiếp về kích thước cố định 224 × 224 (không qua CenterCrop) → Chuẩn hóa ImageNet', 'Giữ trọn vẹn 100% phần diện tích bên trong bounding box nhưng vẫn thiếu ngữ cảnh viền và có thể biến dạng tỷ lệ cạnh'],
    ['Biến thể C (Tối ưu)', 'C_context_pad_15 (Mở rộng ngữ cảnh 15% + Resize trực tiếp)', '+15% chiều rộng (0,15·w) về hai phía trái/phải và +15% chiều cao (0,15·h) về hai phía trên/dưới; giới hạn trong biên ảnh gốc [0, W] × [0, H]', 'Cắt vùng bbox đã mở rộng 15% → Resize trực tiếp về 224 × 224 (T.Resize((224, 224))) → Chuẩn hóa theo mean/std ImageNet', 'Bổ sung đường viền vật thể và ngữ cảnh nền lân cận (tương tự phân bố ảnh huấn luyện DINOv2), không cắt mất rìa vật thể'],
    ['Biến thể D', 'D_letterbox_224 (Đệm vuông bảo toàn tỷ lệ khung hình)', '0% (cắt sát tọa độ bounding box [x1, y1, x2, y2])', 'Cắt sát bbox → Đệm viền xám trung tính RGB(114, 114, 114) để tạo ảnh vuông cạnh max(w, h) → Resize(224 × 224) → Chuẩn hóa ImageNet', 'Bảo toàn tuyệt đối tỷ lệ dài/rộng của vật thể nhưng tạo ra các dải màu xám nhân tạo vốn không xuất hiện trong tập Train phân loại'],
    ['Biến thể E', 'E_context_pad_25_letterbox (Mở rộng 25% kết hợp đệm vuông)', '+25% chiều rộng (0,25·w) mỗi phía và +25% chiều cao (0,25·h) mỗi phía; giới hạn trong biên ảnh gốc', 'Cắt vùng bbox mở rộng 25% → Đệm vuông bằng màu xám RGB(114, 114, 114) → Resize(224 × 224) → Chuẩn hóa ImageNet', 'Cung cấp nhiều ngữ cảnh hơn nhưng diện tích vật thể chính bị thu nhỏ và chịu ảnh hưởng của dải đệm xám nhân tạo'],
]
