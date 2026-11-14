from pathlib import Path
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

out_dir = Path(r'D:\Download Acer\garbage_dinov2_starter\reports\DG38_FINAL_SUBMISSION')
fig_dir = out_dir / 'figures'
fig_dir.mkdir(parents=True, exist_ok=True)

summary_docx = Path(r'C:\Users\hoang\.gemini\antigravity\brain\0c8461c5-bef2-4b19-a785-60e56be2c4ab\.user_uploaded\media_1794370285830_56b91366.docx')
if summary_docx.exists():
    ref_media_dir = fig_dir / 'colab_summary_extracted'
    ref_media_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(summary_docx, 'r') as zf:
        for name in zf.namelist():
            if name.startswith('word/media/'):
                target = ref_media_dir / Path(name).name
                target.write_bytes(zf.read(name))

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10.2,
    'axes.titlesize': 11.2,
    'axes.titleweight': 'bold',
    'axes.labelsize': 10.2,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'legend.fontsize': 9.2,
    'savefig.dpi': 260,
})

# ==============================================================================
# FIGURE 1: Classification Baselines (ResNet50 vs DINOv2 ViT-B/14, 3 Seeds)
# ==============================================================================
fig, axs = plt.subplots(1, 2, figsize=(11.8, 5.1), sharey=True)

splits = ['Validation\n(n = 1.226 ảnh)', 'Final Test đã khóa\n(n = 1.226 ảnh)']
x = np.arange(len(splits))
width = 0.34

resnet_acc = [96.25, 96.22]
resnet_acc_std = [0.08, 0.29]
dinov2_acc = [97.63, 98.61]
dinov2_acc_std = [0.22, 0.29]

resnet_f1 = [95.51, 95.50]
resnet_f1_std = [0.21, 0.27]
dinov2_f1 = [97.33, 98.40]
dinov2_f1_std = [0.20, 0.30]

panels = [
    (axs[0], 'Accuracy (%) — 3 Random Seeds (42, 123, 2026)', resnet_acc, resnet_acc_std, dinov2_acc, dinov2_acc_std),
    (axs[1], 'Macro F1 (%) — 3 Random Seeds (42, 123, 2026)', resnet_f1, resnet_f1_std, dinov2_f1, dinov2_f1_std),
]

for ax, title, r_mean, r_std, d_mean, d_std in panels:
    b1 = ax.bar(x - width/2, r_mean, width, yerr=r_std, capsize=6,
                label='ResNet50 (ImageNet-1K, n = 3 seeds)',
                color='#4C72B0', edgecolor='#1B365D', linewidth=1.1,
                error_kw={'elinewidth': 1.6, 'ecolor': '#1B365D'})
    b2 = ax.bar(x + width/2, d_mean, width, yerr=d_std, capsize=6,
                label='DINOv2 ViT-B/14 (4 blocks, n = 3 seeds)',
                color='#2E8B57', edgecolor='#145A32', linewidth=1.1,
                error_kw={'elinewidth': 1.6, 'ecolor': '#145A32'})
    ax.set_title(title, pad=10)
    ax.set_xticks(x, splits)
    ax.set_ylim(93.5, 100.4)
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    for bar, m, s in zip(b1, r_mean, r_std):
        ax.text(bar.get_x() + bar.get_width()/2, m + s + 0.16,
                f'{m:.2f}%\n(±{s:.2f})', ha='center', va='bottom', fontsize=8.8, fontweight='bold', color='#1B365D')
    for bar, m, s in zip(b2, d_mean, d_std):
        ax.text(bar.get_x() + bar.get_width()/2, m + s + 0.16,
                f'{m:.2f}%\n(±{s:.2f})', ha='center', va='bottom', fontsize=8.8, fontweight='bold', color='#145A32')

axs[0].set_ylabel('Chỉ số đánh giá (%)')
axs[0].legend(loc='lower right', framealpha=0.94)
fig.suptitle('So sánh Classification (10 lớp): ResNet50 vs. DINOv2 ViT-B/14\ntrên tập Validation và Final Test (Mean ± Sample Std, n = 3 seeds)',
             fontsize=11.8, fontweight='bold', y=0.98)
fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.90])
fig.savefig(fig_dir / 'fig_cls_3seeds_val_vs_test.png')
plt.close(fig)

# ==============================================================================
# FIGURE 2: Detection Baselines (YOLO11s 3 Seeds vs RT-DETR-L 1 Seed)
# ==============================================================================
fig, axs = plt.subplots(1, 2, figsize=(12.0, 5.1), sharey=True)

metrics_val = ['mAP@50', 'mAP@50:95', 'AR@100']
x_val = np.arange(len(metrics_val))

yolo_val_mean = [84.94, 61.45, 70.34]
yolo_val_std = [0.09, 0.16, None]

ax = axs[0]
b_yv = ax.bar(x_val - width/2, yolo_val_mean, width,
              yerr=[0.09, 0.16, 0.0], capsize=5,
              label='YOLO11s (n = 3 seeds: 42, 123, 2026)',
              color='#2B5C8F', edgecolor='#16324F', linewidth=1.1,
              error_kw={'elinewidth': 1.5, 'ecolor': '#16324F'})
b_rv = ax.bar(np.array([0, 1]) + width/2, [71.20, 49.43], width,
              label='RT-DETR-L (n = 1 seed: 42, không tính Std)',
              color='#D97724', edgecolor='#7A3E0A', linewidth=1.1)

ax.set_title('Detection Validation sạch\n(1.743 ảnh, 3.932 GT boxes)', pad=10)
ax.set_xticks(x_val, metrics_val)
ax.set_ylabel('Chỉ số COCO (%)')
ax.set_ylim(0, 96)
ax.grid(axis='y', linestyle='--', alpha=0.35)

for bar, m, s in zip(b_yv, yolo_val_mean, yolo_val_std):
    lbl = f'{m:.2f}%\n(±{s:.2f})' if s is not None else f'{m:.2f}%'
    offset = (s if s is not None else 0.0) + 1.2
    ax.text(bar.get_x() + bar.get_width()/2, m + offset, lbl,
            ha='center', va='bottom', fontsize=8.6, fontweight='bold', color='#16324F')

for bar, m in zip(b_rv, [71.20, 49.43]):
    ax.text(bar.get_x() + bar.get_width()/2, m + 1.2, f'{m:.2f}%\n(1 seed)',
            ha='center', va='bottom', fontsize=8.6, fontweight='bold', color='#7A3E0A')
ax.text(2 + width/2, 4.0, 'N/A\n(không lưu)', ha='center', va='bottom', fontsize=8.0, color='#666666', style='italic')

ax = axs[1]
yolo_test_mean = [82.78, 60.37, 69.38]
yolo_test_std = [0.40, 0.42, 0.21]
rtdetr_test = [70.13, 48.57, 69.55]

b_yt = ax.bar(x_val - width/2, yolo_test_mean, width,
              yerr=yolo_test_std, capsize=5,
              label='YOLO11s (n = 3 seeds, Mean ± Sample Std)',
              color='#2B5C8F', edgecolor='#16324F', linewidth=1.1,
              error_kw={'elinewidth': 1.5, 'ecolor': '#16324F'})
b_rt = ax.bar(x_val + width/2, rtdetr_test, width,
              label='RT-DETR-L (n = 1 seed, không tính Std)',
              color='#D97724', edgecolor='#7A3E0A', linewidth=1.1)

ax.set_title('Detection Final Test đã khóa\n(871 ảnh, 1.884 GT boxes)', pad=10)
ax.set_xticks(x_val, metrics_val)
ax.grid(axis='y', linestyle='--', alpha=0.35)

for bar, m, s in zip(b_yt, yolo_test_mean, yolo_test_std):
    ax.text(bar.get_x() + bar.get_width()/2, m + s + 1.2, f'{m:.2f}%\n(±{s:.2f})',
            ha='center', va='bottom', fontsize=8.6, fontweight='bold', color='#16324F')

for bar, m in zip(b_rt, rtdetr_test):
    ax.text(bar.get_x() + bar.get_width()/2, m + 1.2, f'{m:.2f}%\n(1 seed)',
            ha='center', va='bottom', fontsize=8.6, fontweight='bold', color='#7A3E0A')

axs[1].legend(loc='upper right', framealpha=0.94)
fig.suptitle('So sánh Detection Baselines (5 lớp WasteDetect): YOLO11s (3 seeds) vs. RT-DETR-L (1 seed)',
             fontsize=11.8, fontweight='bold', y=0.98)
fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.92])
fig.savefig(fig_dir / 'fig_det_baselines_val_vs_test.png')
plt.close(fig)

# ==============================================================================
# FIGURE 3: Fine-tuning Ablation (DINOv2 ViT-B/14: 0, 2, 4 Transformer Blocks)
# ==============================================================================
fig, ax1 = plt.subplots(figsize=(9.2, 4.8))

blocks_lbl = ['0 blocks\n(Linear Probe:\nLayerNorm + Head)', '2 blocks cuối\n(Blocks 10–11 +\nLayerNorm + Head)', '4 blocks cuối\n(Blocks 8–11 +\nLayerNorm + Head)']
val_macro_f1 = [96.46, 96.99, 97.33]
trainable_pct = [0.01, 16.38, 32.75]
trainable_params = ['9.226', '14.184.970', '28.360.714']

x_b = np.arange(len(blocks_lbl))
bars = ax1.bar(x_b - 0.16, val_macro_f1, width=0.33, color=['#7F9EB2', '#4B86B4', '#1F4E78'],
               edgecolor='#102A43', linewidth=1.2, label='Validation Macro F1 (%)')
ax1.set_ylabel('Validation Macro F1 (%)', color='#1F4E78', fontweight='bold')
ax1.set_ylim(95.5, 98.05)
ax1.set_xticks(x_b, blocks_lbl)
ax1.grid(axis='y', linestyle='--', alpha=0.35)

for bar, f1 in zip(bars, val_macro_f1):
    ax1.text(bar.get_x() + bar.get_width()/2, f1 + 0.06, f'{f1:.2f}%',
             ha='center', va='bottom', fontsize=9.8, fontweight='bold', color='#102A43')

ax2 = ax1.twinx()
bars2 = ax2.bar(x_b + 0.18, trainable_pct, width=0.30, color='#E08244', alpha=0.85,
                edgecolor='#8C3F00', linewidth=1.1, label='Tỷ lệ tham số mở khóa (%)')
ax2.set_ylabel('Tỷ lệ tham số huấn luyện / 86,59M (%)', color='#8C3F00', fontweight='bold')
ax2.set_ylim(0, 43)

for bar, pct, p_str in zip(bars2, trainable_pct, trainable_params):
    ax2.text(bar.get_x() + bar.get_width()/2, pct + 0.8, f'{pct:.2f}%\n({p_str})',
             ha='center', va='bottom', fontsize=8.4, fontweight='bold', color='#8C3F00')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', framealpha=0.94)
ax1.set_title('Ablation 1: Ảnh hưởng của số Transformer blocks fine-tune trên DINOv2 ViT-B/14\n(Đánh giá trên Classification Validation: 1.226 ảnh, 10 lớp)',
              fontsize=11.0, pad=10)
fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.98])
fig.savefig(fig_dir / 'fig_finetune_blocks_ablation.png')
plt.close(fig)

# ==============================================================================
# FIGURE 4: Crop Preprocessing Ablation (5 Variants A-E on Clean Validation)
# ==============================================================================
fig, ax = plt.subplots(figsize=(11.0, 5.1))

variants = [
    'A — Original Crop\n(Crop sát + Resize 256\n+ CenterCrop 224)',
    'B — Direct Resize\n(Crop sát +\nResize 224×224)',
    'C — Context Pad 15%\n(Mở rộng 15% + Clip\n+ Resize 224×224) ★',
    'D — Letterbox 224\n(Crop sát + Pad vuông\nRGB 114 + Resize 224)',
    'E — Pad 25% + Letterbox\n(Mở rộng 25% + Pad vuông\nRGB 114 + Resize 224)',
]
map50 = [43.99, 46.19, 46.69, 45.47, 45.75]
map50_95 = [24.77, 26.32, 26.55, 25.95, 26.10]
ar100 = [41.69, 42.27, 42.52, 41.73, 41.65]

x_c = np.arange(len(variants))
w_c = 0.25

b_m50 = ax.bar(x_c - w_c, map50, w_c, label='Validation mAP@50 (%)', color='#2E6B9E', edgecolor='#163854', linewidth=1.0)
b_m95 = ax.bar(x_c, map50_95, w_c, label='Validation mAP@50:95 (%)', color='#2E8B57', edgecolor='#145A32', linewidth=1.0)
b_ar = ax.bar(x_c + w_c, ar100, w_c, label='Validation AR@100 (%)', color='#D9822B', edgecolor='#7A430D', linewidth=1.0)

b_m50[2].set_facecolor('#1B4F72')
b_m95[2].set_facecolor('#196F3D')
b_ar[2].set_facecolor('#B95804')

for bars_grp, vals, col in [(b_m50, map50, '#163854'), (b_m95, map50_95, '#145A32'), (b_ar, ar100, '#7A430D')]:
    for idx, (bar, v) in enumerate(zip(bars_grp, vals)):
        fw = 'bold' if idx == 2 else 'normal'
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.55, f'{v:.2f}%',
                ha='center', va='bottom', fontsize=8.4, fontweight=fw, color=col)

ax.set_xticks(x_c, variants)
ax.set_ylabel('Chỉ số COCO trên Validation sạch (%)')
ax.set_ylim(0, 56.5)
ax.grid(axis='y', linestyle='--', alpha=0.35)
ax.legend(loc='upper right', ncol=3, framealpha=0.95)
ax.set_title('Ablation 2: Ảnh hưởng của 5 biến thể tiền xử lý vùng cắt (Crop A–E) trong Cascade\n(Đánh giá trên 1.743 ảnh Validation sạch, 3.932 GT boxes; NMS IoU = 0,70; PR thr = 0,25)',
             fontsize=11.0, pad=10)
fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.98])
fig.savefig(fig_dir / 'fig_crop_preprocessing_ablation.png')
plt.close(fig)

# ==============================================================================
# FIGURE 5: Full System Architecture & Experimental Design Diagram
# ==============================================================================
im = Image.new('RGB', (1740, 1030), 'white')
dr = ImageDraw.Draw(im)

f_title = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', 29)
f_sub = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', 21)
f_box_t = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', 19)
f_box_b = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 16)
f_note = ImageFont.truetype('C:/Windows/Fonts/ariali.ttf', 17)

dr.text((45, 22), 'KIẾN TRÚC PIPELINE HAI GIAI ĐOẠN (CASCADE) VÀ HỆ THỐNG ĐỐI CHỨNG ĐỘC LẬP DG38', font=f_title, fill='#102A43')
dr.text((45, 60), 'Phân biệt rõ nhánh kiến trúc đề xuất (Deformable DETR → Crop C → DINOv2) và các baseline độc lập trên giao thức chuẩn hóa', font=f_note, fill='#334E68')

# Top Lane: Proposed Two-Stage Cascade Pipeline
dr.rounded_rectangle((35, 100, 1705, 540), radius=14, fill='#F4F8FB', outline='#2B5C8F', width=3)
dr.text((58, 114), 'NHÁNH 1: KIẾN TRÚC HAI GIAI ĐOẠN ĐỀ XUẤT (TWO-STAGE CASCADE PIPELINE — ĐÁNH GIÁ TRÊN VALIDATION SẠCH)', font=f_sub, fill='#1B365D')

cascade_boxes = [
    (55, 160, 290, 360, '#E3EEF8', '#2B5C8F', '1. Ảnh đầu vào\n(WasteDetect)',
     '• Ảnh chứa đa vật thể\n• Resize 800×800\n• Chuẩn hóa ImageNet\n• Val sạch: 1.743 ảnh\n  (3.932 GT boxes)'),
    (320, 160, 635, 360, '#E3EEF8', '#2B5C8F', '2. Bộ phát hiện (Detector)\nDINOv2 ViT-S/14 + Def. DETR',
     '• Backbone ViT-S/14 đóng băng\n• 4 mức đặc trưng (blocks 3,6,9,12)\n• MSDeformAttn (PyTorch grid_sample)\n• 100 object queries / ảnh\n• Proposal thr = 0,00'),
    (665, 160, 950, 360, '#E8F6EF', '#1E8449', '3. Lọc NMS & Tiền xử lý\nCrop C (Tối ưu từ Val)',
     '• Class-Agnostic NMS IoU = 0,70\n• Mở rộng bbox 15% mỗi phía\n• Giới hạn biên ảnh gốc (clip)\n• Resize trực tiếp 224×224\n• Chuẩn hóa ImageNet'),
    (980, 160, 1295, 360, '#E8F6EF', '#1E8449', '4. Bộ phân loại (Classifier)\nDINOv2 ViT-B/14 (10 lớp)',
     '• Backbone ViT-B/14 (768 chiều)\n• Fine-tune 4 blocks cuối (8–11)\n• Head: LayerNorm → Dropout(0,2)\n  → Linear(768, 10)\n• Dự đoán xác suất cls_score'),
    (1325, 160, 1685, 360, '#FEF5E7', '#B9770E', '5. Ánh xạ nhãn 10 → 5 lớp\n& Hậu xử lý Score Fusion',
     '• Giữ 4 lớp: glass, metal, paper,\n  plastic; biological → organic\n• 5 lớp OOS (battery, cardboard,\n  clothes, shoes, trash): loại bỏ\n• final_score = det_score × cls_score'),
]

for x1, y1, x2, y2, fill_c, out_c, b_title, b_body in cascade_boxes:
    dr.rounded_rectangle((x1, y1, x2, y2), radius=10, fill=fill_c, outline=out_c, width=2)
    dr.text((x1 + 12, y1 + 12), b_title, font=f_box_t, fill=out_c)
    dr.multiline_text((x1 + 12, y1 + 64), b_body, font=f_box_b, fill='#1C2833', spacing=6)

for x_s, x_e in [(290, 320), (635, 665), (950, 980), (1295, 1325)]:
    ym = 260
    dr.line((x_s, ym, x_e, ym), fill='#1B365D', width=4)
    dr.polygon([(x_e, ym), (x_e - 11, ym - 6), (x_e - 11, ym + 6)], fill='#1B365D')

dr.rounded_rectangle((55, 382, 1685, 522), radius=10, fill='#FFFFFF', outline='#5D6D7E', width=2)
dr.text((75, 394), 'Cấu hình khóa từ Validation & Trạng thái đánh giá nhánh Deformable DETR / Cascade:', font=f_box_t, fill='#1B365D')
dr.multiline_text((75, 424),
    '• Thông số khóa: Proposal threshold = 0,00 | NMS IoU = 0,70 | Detector PR thr = 0,30 | Cascade PR thr = 0,25 | Max proposals = 100 | Context padding = 15% (Crop C).\n'
    '• Kết quả Validation sạch (1.743 ảnh): Detector NMS đạt mAP@50 = 76,17%, mAP@50:95 = 44,45%; Cascade NMS (Crop A) = 43,99% / 24,77% → nâng lên 46,69% / 26,55% nhờ Crop C.\n'
    '• Trạng thái Final Test: Nhánh Deformable DETR và Cascade Crop C CHƯA chạy Final Test mới trong gói Notebook 06 (trình bày trung thực, không ngoại suy số liệu Test).',
    font=f_box_b, fill='#1C2833', spacing=7)

# Bottom Left: Independent Classification Track
dr.rounded_rectangle((35, 565, 855, 995), radius=14, fill='#F4FBF7', outline='#1E8449', width=3)
dr.text((58, 580), 'NHÁNH 2: BÀI TOÁN CLASSIFICATION ĐỘC LẬP (10 LỚP)', font=f_sub, fill='#145A32')
dr.multiline_text((58, 616),
    '• Bộ dữ liệu: Garbage Classification v2 (12.259 ảnh đơn đối tượng, 10 lớp)\n'
    '  Train: 9.807 ảnh | Validation: 1.226 ảnh | Locked Final Test: 1.226 ảnh\n'
    '• Giao thức đa hạt giống: 3 Random Seeds độc lập (42, 123, 2026)\n'
    '• Đối chứng độc lập trên Final Test (Notebook 01 & 06, Mean ± Sample Std):\n'
    '  1) Baseline ResNet50: Acc = 96,22% ± 0,29 | Macro F1 = 95,50% ± 0,27\n'
    '  2) Đề xuất DINOv2 ViT-B/14: Acc = 98,61% ± 0,29 | Macro F1 = 98,40% ± 0,30\n'
    '• Nghiên cứu Ablation Fine-tuning (Notebook 03, trên Validation):\n'
    '  0 blocks (96,46%) → 2 blocks (96,99%) → 4 blocks (97,33% Macro F1)\n'
    '• Chỉ số đánh giá: Accuracy, Precision, Recall, Macro F1 (không so trực tiếp với mAP)',
    font=f_box_b, fill='#1C2833', spacing=8)

# Bottom Right: Independent Detection Baselines Track
dr.rounded_rectangle((885, 565, 1705, 995), radius=14, fill='#FDF6EE', outline='#B9770E', width=3)
dr.text((908, 580), 'NHÁNH 3: BÀI TOÁN DETECTION BASELINES ĐỘC LẬP (5 LỚP)', font=f_sub, fill='#7E5109')
dr.multiline_text((908, 616),
    '• Bộ dữ liệu: WasteDetect (5 lớp: glass, metal, organic, paper, plastic)\n'
    '  Train: 8.289 ảnh (18.345 box) | Val sạch: 1.743 ảnh (3.932 box) | Test: 871 ảnh (1.884 box)\n'
    '• Đối chứng độc lập (Notebook 02 & 06):\n'
    '  1) Baseline YOLO11s (3 seeds: 42, 123, 2026 — Mean ± Sample Std):\n'
    '     Val: mAP@50 = 84,94% ± 0,09 | mAP@50:95 = 61,45% ± 0,16 | AR@100 = 70,34%\n'
    '     Test: mAP@50 = 82,78% ± 0,40 | mAP@50:95 = 60,37% ± 0,42 | AR@100 = 69,38% ± 0,21\n'
    '  2) Baseline RT-DETR-L (1 seed: 42 — không tính độ lệch chuẩn):\n'
    '     Val: mAP@50 = 71,20% | mAP@50:95 = 49,43%\n'
    '     Test: mAP@50 = 70,13% | mAP@50:95 = 48,57% | AR@100 = 69,55%\n'
    '• Chỉ số đánh giá: COCO mAP@50, mAP@50:95, AR@100 (pycocotools chuẩn)',
    font=f_box_b, fill='#1C2833', spacing=8)

im.save(fig_dir / 'fig_full_pipeline_architecture.png')
print('Regenerated all 5 figures cleanly')

# ==============================================================================
# FIGURE 6: Deformable DETR & Cascade Overall Comparison on Clean Validation
# ==============================================================================
fig, ax = plt.subplots(figsize=(10.8, 4.9))

cfg_labels = [
    'detector_original\n(No NMS, PR=0,30)',
    'detector_nms\n(NMS 0,70, PR=0,30)',
    'cascade_original\n(No NMS, Crop A)',
    'cascade_nms (Crop A)\n(NMS 0,70, Crop sát)',
    'cascade_nms (Crop C)\n(NMS 0,70, Pad 15%) ★',
]
v_map50 = [71.45, 76.17, 40.83, 43.99, 46.69]
v_map95 = [43.35, 44.45, 24.09, 24.77, 26.55]
v_ar100 = [64.48, 58.49, 48.59, 41.69, 42.52]

x_v = np.arange(len(cfg_labels))
w_v = 0.25

b1 = ax.bar(x_v - w_v, v_map50, w_v, label='Validation mAP@50 (%)', color='#2B5C8F', edgecolor='#16324F', linewidth=1.0)
b2 = ax.bar(x_v, v_map95, w_v, label='Validation mAP@50:95 (%)', color='#2E8B57', edgecolor='#145A32', linewidth=1.0)
b3 = ax.bar(x_v + w_v, v_ar100, w_v, label='Validation AR@100 (%)', color='#D97724', edgecolor='#7A3E0A', linewidth=1.0)

for bars_grp, vals, col in [(b1, v_map50, '#16324F'), (b2, v_map95, '#145A32'), (b3, v_ar100, '#7A3E0A')]:
    for idx, (bar, v) in enumerate(zip(bars_grp, vals)):
        fw = 'bold' if idx in (1, 4) else 'normal'
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.8, f'{v:.2f}%',
                ha='center', va='bottom', fontsize=8.3, fontweight=fw, color=col)

ax.set_xticks(x_v, cfg_labels)
ax.set_ylabel('Chỉ số COCO trên Validation sạch (%)')
ax.set_ylim(0, 86.0)
ax.grid(axis='y', linestyle='--', alpha=0.35)
ax.legend(loc='upper right', ncol=3, framealpha=0.94)
ax.set_title('Hiệu năng nhánh Deformable DETR (Frozen DINOv2 ViT-S/14) và Pipeline Cascade\ntrên tập Validation sạch (1.743 ảnh, 3.932 GT boxes; 1 seed tham chiếu)',
             fontsize=11.0, pad=10)
fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.98])
fig.savefig(fig_dir / 'fig_defdetr_cascade_val_comparison.png')
plt.close(fig)

# ==============================================================================
# FIGURE 7: Per-Class AP@50 on Clean Validation (Deformable DETR vs Cascade)
# ==============================================================================
fig, ax = plt.subplots(figsize=(10.6, 4.8))

cls_names = ['glass\n(790 box)', 'metal\n(668 box)', 'organic\n(632 box)', 'paper\n(923 box)', 'plastic\n(919 box)']
det_orig_ap50 = [88.52, 60.35, 57.90, 78.86, 71.60]
det_nms_ap50 = [91.13, 65.80, 64.13, 85.32, 74.47]
cas_orig_ap50 = [23.00, 46.61, 55.61, 46.18, 32.74]
cas_nms_ap50 = [23.85, 50.24, 61.72, 49.95, 34.17]

x_cl = np.arange(len(cls_names))
w_cl = 0.20

b_do = ax.bar(x_cl - 1.5*w_cl, det_orig_ap50, w_cl, label='detector_original (No NMS)', color='#5B84B1', edgecolor='#1B365D', linewidth=0.9)
b_dn = ax.bar(x_cl - 0.5*w_cl, det_nms_ap50, w_cl, label='detector_nms (NMS IoU=0,70)', color='#1F4E78', edgecolor='#102A43', linewidth=0.9)
b_co = ax.bar(x_cl + 0.5*w_cl, cas_orig_ap50, w_cl, label='cascade_original (No NMS, Crop A)', color='#E69F00', edgecolor='#7A430D', linewidth=0.9)
b_cn = ax.bar(x_cl + 1.5*w_cl, cas_nms_ap50, w_cl, label='cascade_nms (NMS IoU=0,70, Crop A)', color='#C05621', edgecolor='#652B0E', linewidth=0.9)

for bars_grp, vals, col in [
    (b_do, det_orig_ap50, '#1B365D'),
    (b_dn, det_nms_ap50, '#102A43'),
    (b_co, cas_orig_ap50, '#7A430D'),
    (b_cn, cas_nms_ap50, '#652B0E'),
]:
    for bar, v in zip(bars_grp, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.8, f'{v:.1f}',
                ha='center', va='bottom', fontsize=7.8, fontweight='bold', color=col)

ax.set_xticks(x_cl, cls_names)
ax.set_ylabel('Validation AP@50 (%)')
ax.set_ylim(0, 114.0)
ax.grid(axis='y', linestyle='--', alpha=0.35)
ax.legend(loc='upper right', ncol=2, framealpha=0.94)
ax.set_title('Phân rã AP@50 theo từng lớp rác trên tập Validation sạch (1.743 ảnh, 3.932 GT boxes)\ngiữa Deformable DETR và Pipeline Cascade (trước khi áp dụng Crop C)',
             fontsize=11.0, pad=10)
fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.98])
fig.savefig(fig_dir / 'fig_defdetr_cascade_val_per_class.png')
plt.close(fig)
print('Generated figures 6 and 7 cleanly')
