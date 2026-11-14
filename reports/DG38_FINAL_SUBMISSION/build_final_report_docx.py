# -*- coding: utf-8 -*-
"""
Builds the complete academic project report:
D:\\Download Acer\\garbage_dinov2_starter\\reports\\DG38_FINAL_SUBMISSION\\DG38_Final_Complete_Report.docx
from the base template without modifying any original report files.
"""
import sys
import json
import re
from copy import deepcopy
from pathlib import Path
from PIL import Image
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(r'D:\Download Acer\garbage_dinov2_starter')
OUT_DIR = ROOT / 'reports' / 'DG38_FINAL_SUBMISSION'
FIG_DIR = OUT_DIR / 'figures'
QA_ASSETS = ROOT / 'evaluation_runs' / 'report_final_review' / '_qa' / 'assets'
DOC0_MEDIA = ROOT / 'evaluation_runs' / 'report_final_review' / '_qa' / 'doc0_media'
SOURCE_TEMPLATE = ROOT / 'evaluation_runs' / 'report_final_review' / '_submission_qa' / 'submission_input_v6.docx'
BUILD_INPUT_DOCX = OUT_DIR / '_build_input.docx'

sys.path.insert(0, str(OUT_DIR))
from report_content_Part1 import (
    ABBREVIATIONS_ROWS,
    LIST_OF_TABLES_ROWS,
    LIST_OF_FIGURES_ROWS,
    TABLE_2_SPLIT_SUMMARY,
    TABLE_4_EXPERIMENT_MATRIX,
    TABLE_5_CLS_CONFIG,
    TABLE_6_DET_CONFIG,
    TABLE_8_CASCADE_LOCKED_CONFIG,
    TABLE_9_CROP_VARIANTS_SPEC,
)
from report_content_Part2 import (
    TABLE_10_CLS_3SEEDS,
    TABLE_14_DET_VALIDATION,
    TABLE_15_DET_FINAL_TEST,
    TABLE_16_DET_VAL_VS_TEST,
    TABLE_17_FINETUNE_ABLATION,
    TABLE_18_CROP_ABLATION,
    TABLE_19_DEFDETR_CASCADE_VALIDATION,
    TABLE_20_DEFDETR_CASCADE_PER_CLASS_VAL,
    TABLE_23_EVALUATION_STATUS_SUMMARY,
    TABLE_A1_NOTEBOOKS_AND_CODE,
    TABLE_B1_PROVENANCE_AND_HASHES,
    TABLE_C1_TEACHER_FEEDBACK_MATRIX,
    REFERENCES_LIST,
)

base_tables = json.loads((OUT_DIR / 'base_tables_cache.json').read_text(encoding='utf-8'))
TABLE_1_GARBAGE_CLS = base_tables['T00']
TABLE_3_WASTEDETECT_AUDIT = base_tables['T02']
TABLE_7_REF_PAPER_CONFIG = base_tables['T05']
TABLE_11_CLS_10CLASSES_SEED42 = base_tables['T07']
TABLE_12_REF_OVERALL = base_tables['T08']
TABLE_13_REF_5COMMON = base_tables['T09']
TABLE_21_FACTORIAL_ABLATION = base_tables['T14']
TABLE_22_CONFUSION_5X10 = base_tables['T15']

doc = Document(str(SOURCE_TEMPLATE))
body = doc._element.body

# Extract the structured Office Math element from paragraph 43 before clearing body after cover page
math_omath_el = None
for p in doc.paragraphs:
    omaths = p._p.xpath('.//m:oMath')
    if omaths:
        math_omath_el = deepcopy(omaths[0])
        break
assert math_omath_el is not None, "Could not locate native Office Math element in template"

# Preserve Cover Page (paragraphs 0..11) and body.sectPr; remove everything else from body
children = list(body)
cover_end_p = doc.paragraphs[11]._p
cover_end_idx = children.index(cover_end_p)
for el in children[cover_end_idx + 1:]:
    if el is not body.sectPr:
        body.remove(el)

# Update cover page date to reflect final submission
doc.paragraphs[11].text = 'Đà Nẵng, tháng 11 năm 2026'
for r in doc.paragraphs[11].runs:
    r.font.name = 'Times New Roman'
    r.font.size = Pt(13)
    r.italic = True

# Remove any existing sectPr inside cover paragraphs so we can set clean section breaks
base_sectPr = deepcopy(doc.sections[0]._sectPr)
for node in list(body.iter(qn('w:sectPr'))):
    if node is not body.sectPr:
        node.getparent().remove(node)

def make_sectPr(fmt=None, start=None):
    s = deepcopy(base_sectPr)
    for n in list(s):
        if n.tag in [qn('w:footerReference'), qn('w:headerReference'), qn('w:pgNumType'), qn('w:titlePg'), qn('w:type')]:
            s.remove(n)
    typ = OxmlElement('w:type')
    typ.set(qn('w:val'), 'nextPage')
    s.insert(0, typ)
    if fmt:
        pn = OxmlElement('w:pgNumType')
        pn.set(qn('w:fmt'), fmt)
        pn.set(qn('w:start'), str(start))
        s.append(pn)
    return s

# End cover page section (unnumbered)
cover_end_p.get_or_add_pPr().append(make_sectPr())

# Helper functions to append well-styled content
def set_run_font(r, size=12, bold=False, italic=False, color=None, font_name='Times New Roman'):
    r.font.name = font_name
    r._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'), font_name)
    r.font.size = Pt(size)
    r.bold = bold
    r.italic = italic
    if color is not None:
        r.font.color.rgb = RGBColor(*color)

def add_h1(text, page_break=False):
    p = doc.add_paragraph(style='Heading 1')
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.page_break_before = page_break
    r = p.add_run(text)
    set_run_font(r, size=14, bold=True, color=(0, 0, 0))
    # Explicit outlineLvl = 0 for TOC
    pPr = p._p.get_or_add_pPr()
    for node in list(pPr.findall(qn('w:outlineLvl'))):
        pPr.remove(node)
    ol = OxmlElement('w:outlineLvl')
    ol.set(qn('w:val'), '0')
    pPr.append(ol)
    return p

def add_h2(text):
    p = doc.add_paragraph(style='Heading 2')
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text)
    set_run_font(r, size=13, bold=True, color=(0, 0, 0))
    pPr = p._p.get_or_add_pPr()
    for node in list(pPr.findall(qn('w:outlineLvl'))):
        pPr.remove(node)
    ol = OxmlElement('w:outlineLvl')
    ol.set(qn('w:val'), '1')
    pPr.append(ol)
    return p

def add_h3(text):
    p = doc.add_paragraph(style='Normal')
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(text)
    set_run_font(r, size=12, bold=True, italic=True, color=(0, 0, 0))
    return p

def add_p(text, bold_prefix=None, italic=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_after=5, line_spacing=1.15, size=12):
    p = doc.add_paragraph(style='Normal')
    p.alignment = align
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = line_spacing
    if bold_prefix:
        rb = p.add_run(bold_prefix)
        set_run_font(rb, size=size, bold=True, italic=italic)
    r = p.add_run(text)
    set_run_font(r, size=size, bold=False, italic=italic)
    return p

def add_table_with_caption(caption_text, rows_data, col_weights=None, font_size=9.5):
    cap = doc.add_paragraph(style='Normal')
    cap.alignment = WD_ALIGN_PARAGRAPH.LEFT
    cap.paragraph_format.keep_with_next = True
    cap.paragraph_format.keep_together = True
    cap.paragraph_format.space_before = Pt(6)
    cap.paragraph_format.space_after = Pt(3)
    rc = cap.add_run(caption_text)
    set_run_font(rc, size=11, bold=True, color=(0, 0, 0))

    nrows = len(rows_data)
    ncols = len(rows_data[0])
    t = doc.add_table(rows=nrows, cols=ncols)
    t.autofit = False

    section = doc.sections[0]
    available = section.page_width - section.left_margin - section.right_margin
    if col_weights is None:
        if ncols == 2:
            col_weights = [1.15, 2.55]
        elif ncols == 3:
            col_weights = [1.0, 1.35, 1.35]
        else:
            col_weights = [1.35] + [1.0] * (ncols - 1)
    widths = [int(available * w / sum(col_weights)) for w in col_weights]
    for col, w in zip(t.columns, widths):
        col.width = w

    pr = t._tbl.tblPr
    ww = pr.find(qn('w:tblW'))
    if ww is None:
        ww = OxmlElement('w:tblW')
        pr.append(ww)
    ww.set(qn('w:type'), 'dxa')
    ww.set(qn('w:w'), str(int(available / 635)))

    borders = OxmlElement('w:tblBorders')
    for name in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
        e = OxmlElement('w:' + name)
        e.set(qn('w:val'), 'single')
        e.set(qn('w:sz'), '4')
        e.set(qn('w:color'), 'BFBFBF')
        borders.append(e)
    pr.append(borders)

    margins = OxmlElement('w:tblCellMar')
    for name, val in [('top', 55), ('bottom', 55), ('left', 65), ('right', 65)]:
        e = OxmlElement('w:' + name)
        e.set(qn('w:w'), str(val))
        e.set(qn('w:type'), 'dxa')
        margins.append(e)
    pr.append(margins)

    for ri, row_vals in enumerate(rows_data):
        row = t.rows[ri]
        trpr = row._tr.get_or_add_trPr()
        if trpr.find(qn('w:cantSplit')) is None:
            trpr.append(OxmlElement('w:cantSplit'))
        if ri == 0 and trpr.find(qn('w:tblHeader')) is None:
            trpr.append(OxmlElement('w:tblHeader'))
        for ci, val in enumerate(row_vals):
            cell = row.cells[ci]
            cell.width = widths[ci]
            cell.vertical_alignment = 1
            if ri == 0:
                sh = OxmlElement('w:shd')
                sh.set(qn('w:fill'), 'E7ECF1')
                cell._tc.get_or_add_tcPr().append(sh)
            cell.text = str(val).replace('_', '_\u200b')
            for p in cell.paragraphs:
                p.paragraph_format.space_before = Pt(1.5)
                p.paragraph_format.space_after = Pt(1.5)
                p.paragraph_format.line_spacing = 1.05
                p.paragraph_format.keep_together = True
                p.paragraph_format.keep_with_next = (ri == 0)
                if ri == 0:
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                else:
                    p.alignment = WD_ALIGN_PARAGRAPH.LEFT if (ci == 0 or ncols <= 3) else WD_ALIGN_PARAGRAPH.CENTER
                for r in p.runs:
                    set_run_font(r, size=font_size, bold=(ri == 0 or (ci == 0 and ncols > 4)))

    sp = doc.add_paragraph(style='Normal')
    sp.paragraph_format.space_before = Pt(0)
    sp.paragraph_format.space_after = Pt(3)
    return t

def add_figure_with_caption(img_path, caption_text, max_width_in=6.15, max_height_in=6.8):
    img_path = Path(img_path)
    assert img_path.exists(), f"Missing image: {img_path}"
    img = Image.open(img_path)
    w = max_width_in
    h = w * img.height / img.width
    if h > max_height_in:
        w *= max_height_in / h

    p_img = doc.add_paragraph(style='Normal')
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.keep_with_next = True
    p_img.paragraph_format.keep_together = True
    p_img.paragraph_format.space_before = Pt(6)
    p_img.paragraph_format.space_after = Pt(3)
    p_img.add_run().add_picture(str(img_path), width=Inches(w))

    for prop in p_img._p.xpath('.//wp:docPr'):
        prop.set('descr', caption_text[:500])

    p_cap = doc.add_paragraph(style='Normal')
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.keep_together = True
    p_cap.paragraph_format.keep_with_next = False
    p_cap.paragraph_format.space_before = Pt(2)
    p_cap.paragraph_format.space_after = Pt(8)
    rc = p_cap.add_run(caption_text)
    set_run_font(rc, size=11, italic=True, color=(0, 0, 0))
    return p_img, p_cap

# ==============================================================================
# FRONT MATTER (ROMAN NUMERALS i, ii, iii...)
# ==============================================================================
p_toc_title = doc.add_paragraph(style='Normal')
p_toc_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_toc_title.paragraph_format.keep_with_next = True
p_toc_title.paragraph_format.space_before = Pt(6)
p_toc_title.paragraph_format.space_after = Pt(8)
rt = p_toc_title.add_run('MỤC LỤC')
set_run_font(rt, size=14, bold=True)

p_toc_field = doc.add_paragraph(style='Normal')
p_toc_field.paragraph_format.space_before = Pt(0)
p_toc_field.paragraph_format.space_after = Pt(6)
rf = p_toc_field.add_run()
set_run_font(rf, size=11)
for kind in ['begin', 'instruction', 'separate', 'end']:
    if kind == 'instruction':
        n = OxmlElement('w:instrText')
        n.set(qn('xml:space'), 'preserve')
        n.text = ' TOC \\o "1-2" \\h \\z \\u '
    else:
        n = OxmlElement('w:fldChar')
        n.set(qn('w:fldCharType'), kind)
    rf._r.append(n)

# Danh mục chữ viết tắt, danh mục bảng, danh mục hình
p_abbr = doc.add_paragraph(style='Normal')
p_abbr.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_abbr.paragraph_format.page_break_before = True
p_abbr.paragraph_format.keep_with_next = True
p_abbr.paragraph_format.space_after = Pt(6)
set_run_font(p_abbr.add_run('DANH MỤC TỪ VIẾT TẮT VÀ KÝ HIỆU'), size=13, bold=True)
add_table_with_caption('Bảng danh mục các từ viết tắt và ký hiệu toán học sử dụng trong báo cáo', ABBREVIATIONS_ROWS, col_weights=[0.85, 1.35, 1.80], font_size=9.5)

p_lot = doc.add_paragraph(style='Normal')
p_lot.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_lot.paragraph_format.page_break_before = True
p_lot.paragraph_format.keep_with_next = True
p_lot.paragraph_format.space_after = Pt(6)
set_run_font(p_lot.add_run('DANH MỤC BẢNG SỐ LIỆU'), size=13, bold=True)
add_table_with_caption('Danh mục 26 bảng số liệu trong báo cáo (Bảng 1–23 và Bảng A1, B1, C1)', LIST_OF_TABLES_ROWS, col_weights=[0.55, 2.25, 1.20], font_size=9.0)

p_lof = doc.add_paragraph(style='Normal')
p_lof.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_lof.paragraph_format.page_break_before = True
p_lof.paragraph_format.keep_with_next = True
p_lof.paragraph_format.space_after = Pt(6)
set_run_font(p_lof.add_run('DANH MỤC HÌNH VẼ VÀ BIỂU ĐỒ'), size=13, bold=True)
add_table_with_caption('Danh mục 20 hình vẽ và biểu đồ thực nghiệm trong báo cáo (Hình 1–19 và Hình A1)', LIST_OF_FIGURES_ROWS, col_weights=[0.55, 2.25, 1.20], font_size=9.0)

# Tóm tắt (Tiếng Việt) & Abstract (Tiếng Anh)
add_h1('Tóm tắt (Tiếng Việt và Tiếng Anh)', page_break=True)
add_p(
    'Báo cáo trình bày nghiên cứu thực nghiệm có kiểm soát về phân loại và phát hiện rác thải bằng học sâu (Deep Learning), kết hợp giữa mô hình nền tảng thị giác tự giám sát DINOv2, họ kiến trúc phát hiện đối tượng Transformer (Deformable DETR, RT-DETR-L), mạng tích chập một giai đoạn (YOLO11s) và mạng tích chập sâu (ResNet50). Nghiên cứu giải quyết đồng thời hai bài toán có cấu trúc nhãn khác biệt: (i) Phân loại ảnh đơn đối tượng trên bộ dữ liệu Garbage Classification v2 gồm 12.259 ảnh thuộc 10 lớp (9.807 Train / 1.226 Validation / 1.226 Final Test đã khóa SHA-256); và (ii) Phát hiện đa đối tượng bằng bounding box trên bộ dữ liệu WasteDetect gồm 5 lớp (glass, metal, organic, paper, plastic), trong đó tập Validation sạch (1.743 ảnh, 3.932 GT boxes) và tập Final Test đã khóa (871 ảnh, 1.884 GT boxes) đã trải qua quy trình kiểm toán khử trùng lặp đa tầng bằng mã băm SHA-256 kết hợp mã băm cảm nhận thị giác pHash (ngưỡng khoảng cách Hamming ≤ 6) và loại bỏ nhãn trùng ID 1847.',
    bold_prefix='Tóm tắt Tiếng Việt: ', size=11.5, line_spacing=1.12
)
add_p(
    'Thực hiện đầy đủ góp ý chuyên môn của giảng viên hướng dẫn (Hồ sơ DG38), nhóm đã bổ sung các baseline độc lập, thiết lập giao thức lặp lại với ba hạt giống ngẫu nhiên (seeds 42, 123, 2026 — báo cáo Mean ± Sample Standard Deviation với ddof = 1) và thực hiện hai nghiên cứu cắt bỏ/thay đổi thành phần (Ablation Study) trên tập Validation trước khi mở gói Final Test đã khóa. Trên bài toán Classification (10 lớp), DINOv2 ViT-B/14 (fine-tune 4 khối Transformer cuối) đạt Validation Accuracy 97,63% ± 0,22 và Macro F1 97,33% ± 0,20; trên Final Test đã khóa, DINOv2 ViT-B/14 đạt Accuracy 98,61% ± 0,29 và Macro F1 98,40% ± 0,30, vượt qua baseline ResNet50 (Test Accuracy 96,22% ± 0,29; Macro F1 95,50% ± 0,27) lần lượt +2,39 điểm phần trăm về Accuracy và +2,91 điểm phần trăm về Macro F1. Trên bài toán Detection (5 lớp), baseline YOLO11s (3 seeds) đạt Validation mAP@50 = 84,94% ± 0,09, mAP@50:95 = 61,45% ± 0,16 và đạt Final Test mAP@50 = 82,78% ± 0,40, mAP@50:95 = 60,37% ± 0,42, AR@100 = 69,38% ± 0,21; trong khi baseline RT-DETR-L (1 seed = 42) đạt Validation mAP@50 = 71,20%, mAP@50:95 = 49,43% và Final Test mAP@50 = 70,13%, mAP@50:95 = 48,57%, AR@100 = 69,55%.',
    size=11.5, line_spacing=1.12
)
add_p(
    'Hai nghiên cứu Ablation trên tập Validation làm sáng tỏ cơ chế thích nghi miền và ghép nối hai giai đoạn: (1) Khi thay đổi số khối Transformer được mở khóa tinh chỉnh trên DINOv2 ViT-B/14 từ 0 blocks (Linear Probe, 0,01% tham số) lên 2 blocks (16,38% tham số) và 4 blocks (32,75% tham số), Validation Macro F1 tăng tương ứng từ 96,46% lên 96,99% và 97,33%; (2) Trong kiến trúc Pipeline Cascade hai giai đoạn (Frozen DINOv2 ViT-S/14 + Deformable DETR → Class-Agnostic NMS IoU = 0,70 → Crop → DINOv2 ViT-B/14 Classifier → Ánh xạ 10 sang 5 lớp), khảo sát 5 biến thể tiền xử lý vùng cắt (Crop A–E) trên 1.743 ảnh Validation sạch chứng minh rằng phương pháp Crop C (mở rộng bounding box thêm 15% mỗi phía, giới hạn trong biên ảnh gốc, cắt ảnh và resize trực tiếp về 224 × 224) nâng Validation mAP@50 từ 43,99% (Crop A) lên 46,69% (+2,70 điểm phần trăm) và mAP@50:95 từ 24,77% lên 26,55% (+1,78 điểm phần trăm). Do nhánh Deformable DETR và Pipeline Cascade (với cấu hình tối ưu Crop C) chưa thực hiện lượt chạy Final Test mới trong gói Notebook 06 đã khóa, báo cáo tuân thủ nghiêm ngặt tính trung thực khoa học: chỉ kết luận hiệu quả của Deformable DETR và Cascade trên tập Validation sạch, không tự suy diễn số liệu Test và không tuyên bố Cascade vượt qua YOLO11s.',
    size=11.5, line_spacing=1.12
)
add_p(
    'This report presents a controlled empirical investigation of deep learning-based waste image classification and multi-object detection, integrating the self-supervised vision foundation model DINOv2, Transformer-based object detectors (Deformable DETR and RT-DETR-L), a one-stage convolutional detector (YOLO11s), and a residual convolutional network baseline (ResNet50). Two distinct tasks are addressed under strict data-hygiene protocols: (i) single-object image classification on Garbage Classification v2 (12,259 images across 10 classes: 9,807 Train / 1,226 Validation / 1,226 SHA-256 locked Final Test); and (ii) multi-object bounding-box detection on WasteDetect across 5 classes (glass, metal, organic, paper, plastic), featuring 8,289 Train images (18,345 boxes), 1,743 deduplicated Clean Validation images (3,932 boxes), and 871 SHA-256 locked Final Test images (1,884 boxes) audited via SHA-256 and perceptual hashing (pHash Hamming distance ≤ 6). Across three independent random seeds (42, 123, 2026), DINOv2 ViT-B/14 achieves 98.61% ± 0.29% Accuracy and 98.40% ± 0.30% Macro F1 on the locked Classification Final Test, outperforming ResNet50 (96.22% ± 0.29% Accuracy; 95.50% ± 0.27% Macro F1) by +2.39 and +2.91 percentage points, respectively. On the locked Detection Final Test, YOLO11s (3 seeds) achieves 82.78% ± 0.40% mAP@50, 60.37% ± 0.42% mAP@50:95, and 69.38% ± 0.21% AR@100, while single-seed RT-DETR-L achieves 70.13% mAP@50, 48.57% mAP@50:95, and 69.55% AR@100. Validation ablations confirm that unfreezing 4 DINOv2 Transformer blocks yields the highest Validation Macro F1 (97.33% vs. 96.46% for 0 blocks and 96.99% for 2 blocks), and expanding proposal crops by 15% context padding with direct 224×224 resizing (Crop C) improves Cascade Validation mAP@50 from 43.99% to 46.69% (+2.70 percentage points). Because Deformable DETR and Cascade (Crop C) were not re-evaluated on the locked Final Test in Notebook 06, their conclusions are strictly bounded to the Clean Validation split.',
    bold_prefix='Abstract (English): ', italic=True, size=11.0, line_spacing=1.10
)
add_p(
    'phân loại rác thải, phát hiện đối tượng, DINOv2, ResNet50, Deformable DETR, YOLO11s, RT-DETR-L, Pipeline Cascade hai giai đoạn, tiền xử lý vùng cắt (Crop Preprocessing), lệch miền vùng cắt (Crop Domain Shift), đa hạt giống ngẫu nhiên (Multiple Random Seeds).',
    bold_prefix='Từ khóa: ', size=11.0, space_after=4
)

# Section break ending Roman front matter (lowerRoman starting at i) and starting Arabic main body (decimal starting at 1)
p_sect_end = doc.add_paragraph(style='Normal')
p_sect_end.paragraph_format.space_before = Pt(0)
p_sect_end.paragraph_format.space_after = Pt(0)
p_sect_end.paragraph_format.line_spacing = Pt(1)
p_sect_end.add_run().font.size = Pt(1)
p_sect_end._p.get_or_add_pPr().append(make_sectPr('lowerRoman', 1))

print('Front matter built successfully')

# ==============================================================================
# CHƯƠNG 1. GIỚI THIỆU (TỔNG QUAN ĐỀ TÀI)
# ==============================================================================
add_h1('1. Giới thiệu', page_break=False)
add_h2('1.1. Đặt vấn đề và bối cảnh ứng dụng')
add_p(
    'Sự gia tăng nhanh chóng của khối lượng chất thải rắn sinh hoạt tại các đô thị hiện đại đang tạo ra áp lực lớn lên hạ tầng thu gom, xử lý và tái chế. Việc phân loại rác tại nguồn đóng vai trò then chốt nhằm thu hồi vật liệu tái chế (thủy tinh, kim loại, giấy, nhựa, bìa cứng), tách riêng chất thải hữu cơ sinh học và cô lập rác thải nguy hại (pin, ắc quy). Tuy nhiên, quy trình phân loại thủ công không chỉ tốn kém nhân lực, hiệu suất thấp mà còn tiềm ẩn nguy cơ phơi nhiễm độc hại đối với người lao động.'
)
add_p(
    'Trong bối cảnh đó, việc ứng dụng Thị giác máy tính (Computer Vision) và Học sâu (Deep Learning) để tự động hóa nhận diện rác thải đã trở thành một hướng nghiên cứu trọng tâm. Tuy nhiên, bài toán nhận diện rác thải trong thực tế đặt ra hai thách thức kỹ thuật rất khác nhau: (1) Ở bài toán phân loại ảnh đơn đối tượng (Image Classification), mô hình phải phân biệt tinh vi nhiều nhóm vật liệu có bề mặt bị biến dạng, nhàu nát, bám bẩn hoặc phản quang (như chai nhựa trong suốt so với lọ thủy tinh, hoặc bìa carton so với giấy vụn); và (2) Ở bài toán phát hiện đa đối tượng (Multi-Object Detection), hệ thống phải đồng thời định vị tọa độ khung bao (bounding box) và gán nhãn cho nhiều mẫu rác nằm chồng lấp, kích thước đa tỷ lệ trên nền phức tạp.'
)

add_h2('1.2. Mục tiêu nghiên cứu, hướng tiếp cận và hoàn thiện theo góp ý giảng viên')
add_p(
    'Đề tài hướng tới việc nghiên cứu, thiết kế và đánh giá thực nghiệm nghiêm ngặt các kiến trúc học sâu hiện đại cho cả hai bài toán phân loại và phát hiện rác thải, đồng thời khảo sát khả năng ghép nối tuần tự hai giai đoạn (Detector → Crop → Classifier). Cụ thể, đề tài tập trung giải quyết bốn mục tiêu và câu hỏi nghiên cứu cốt lõi sau:'
)
add_p(
    'Nghiên cứu khai thác mô hình nền tảng thị giác tự giám sát DINOv2 (ViT-B/14, 768 chiều) cho bài toán phân loại 10 lớp rác thải trên bộ dữ liệu Garbage Classification v2 (12.259 ảnh), đồng thời đặt trong phép đối chứng công bằng trên cùng giao thức dữ liệu với mạng tích chập sâu kinh điển ResNet50 nhằm trả lời câu hỏi: liệu đặc trưng tự giám sát của Vision Transformer có mang lại ưu thế thực sự và ổn định hơn mạng CNN truyền thống hay không.',
    bold_prefix='• Mục tiêu 1 — Đánh giá và đối chứng độc lập bài toán Classification (10 lớp): '
)
add_p(
    'Triển khai bộ phát hiện đa tỷ lệ kết hợp backbone DINOv2 ViT-S/14 đóng băng với kiến trúc Deformable DETR trên bộ dữ liệu phát hiện 5 lớp WasteDetect, đồng thời bổ sung hai baseline phát hiện độc lập đại diện cho hai trường phái kiến trúc hiện đại là YOLO11s (mạng CNN một giai đoạn) và RT-DETR-L (kiến trúc Transformer thời gian thực) trên cùng tập dữ liệu và giao thức đánh giá chuẩn COCO (mAP@50, mAP@50:95, AR@100).',
    bold_prefix='• Mục tiêu 2 — Đánh giá và đối chứng độc lập bài toán Detection (5 lớp): '
)
add_p(
    'Thiết lập giao thức huấn luyện và đánh giá lặp lại với ba hạt giống ngẫu nhiên độc lập (random seeds 42, 123, 2026) đối với ResNet50, DINOv2 ViT-B/14 và YOLO11s để định lượng độ ổn định qua trung bình và độ lệch chuẩn mẫu (Mean ± Sample Standard Deviation, ddof = 1); phân biệt minh bạch với các cấu hình một hạt giống (RT-DETR-L, Deformable DETR) mà không tự suy diễn sai số.',
    bold_prefix='• Mục tiêu 3 — Đảm bảo độ ổn định thống kê và tính trung thực khoa học: '
)
add_p(
    'Thực hiện hai nghiên cứu Ablation có kiểm soát trên tập Validation gồm: (i) Thay đổi số khối Transformer được mở khóa tinh chỉnh trên DINOv2 ViT-B/14 (0, 2 và 4 blocks) để phân tích đánh đổi giữa dung lượng tham số thích nghi và hiệu năng phân loại; và (ii) Khảo sát 5 biến thể tiền xử lý vùng cắt (Crop Variants A–E) trong kiến trúc Pipeline Cascade hai giai đoạn để kiểm chứng thực nghiệm giả thuyết lệch miền vùng cắt (Crop Domain Shift Hypothesis) và vai trò của ngữ cảnh xung quanh vật thể.',
    bold_prefix='• Mục tiêu 4 — Thực hiện các nghiên cứu cắt bỏ và thay đổi thành phần (Ablation Studies): '
)

add_h2('1.3. Phạm vi nghiên cứu và bố cục báo cáo')
add_p(
    'Về phạm vi dữ liệu và quy ước thực nghiệm: Báo cáo phân biệt rạch ròi giữa hai bài toán có không gian nhãn và thước đo khác nhau — bài toán Classification đánh giá trên 10 lớp rác bằng Accuracy và Macro F1, trong khi bài toán Detection đánh giá trên 5 lớp rác bằng COCO mAP@50, mAP@50:95 và AR@100. Toàn bộ quá trình chọn mô hình, chọn số khối fine-tune, chọn ngưỡng NMS IoU (0,70), ngưỡng điểm vận hành (PR threshold) và chọn phương pháp tiền xử lý vùng cắt (Crop C) đều được thực hiện nghiêm ngặt trên tập Validation; tập Final Test đã được kiểm toán chống rò rỉ và đóng gói khóa bằng mã băm SHA-256 chỉ được dùng để đo hiệu năng cuối cùng của ResNet50, DINOv2 ViT-B/14, YOLO11s và RT-DETR-L. Riêng nhánh Deformable DETR và Pipeline Cascade (với cấu hình tối ưu Crop C) được báo cáo trung thực là đã hoàn thành đánh giá và chẩn đoán trên tập Validation sạch nhưng chưa thực hiện lượt chạy Final Test mới trên bộ Test đã khóa.'
)
add_p(
    'Bố cục báo cáo được tổ chức thành 7 chương và 3 phụ lục: Chương 1 giới thiệu tổng quan đề tài; Chương 2 trình bày cơ sở lý thuyết về CNN/ResNet50, ViT/DINOv2, DETR/Deformable DETR/RT-DETR/YOLO11s, kiến trúc Cascade và các chỉ số đánh giá; Chương 3 trình bày chi tiết các bộ dữ liệu, quy trình kiểm toán khử trùng lặp đa tầng (SHA-256 + pHash) và giao thức thực nghiệm đa hạt giống; Chương 4 mô tả kiến trúc phương pháp đề xuất, các baseline độc lập và thiết kế hai nghiên cứu Ablation; Chương 5 trình bày và phân tích toàn diện kết quả thực nghiệm; Chương 6 thảo luận chuyên sâu 8 luận điểm khoa học và các giới hạn nghiên cứu; Chương 7 tổng kết kết luận và đề xuất hướng phát triển; các Phụ lục A, B, C cung cấp minh chứng mã nguồn, danh mục Notebooks 00–06, mã băm SHA-256 truy xuất nguồn gốc và ma trận đối chiếu hoàn thành góp ý của giảng viên.'
)

# ==============================================================================
# CHƯƠNG 2. CƠ SỞ LÝ THUYẾT VÀ NGHIÊN CỨU LIÊN QUAN
# ==============================================================================
add_h1('2. Cơ sở lý thuyết và nghiên cứu liên quan', page_break=True)
add_h2('2.1. Phân loại ảnh rác thải, mạng tích chập CNN và kiến trúc ResNet50')
add_p(
    'Phân loại ảnh (Image Classification) là bài toán ánh xạ một bức ảnh đầu vào chứa đối tượng chính về một phân phối xác suất trên tập C lớp định trước. Trong suốt thập kỷ qua, Mạng nơ-ron tích chập (Convolutional Neural Networks — CNN) đóng vai trò nền tảng nhờ hai tính chất quy nạp (inductive biases) quan trọng: tính cục bộ không gian (spatial locality) thông qua các nhân lọc trượt và tính bất biến tịnh tiến (translation equivariance) thông qua cơ chế chia sẻ trọng số.'
)
add_p(
    'Tuy nhiên, khi tăng chiều sâu của mạng CNN thuần túy, hiện tượng suy thoái đạo hàm (degradation problem) khiến sai số huấn luyện lẫn kiểm thử đều bão hòa và tăng ngược trở lại. He và cs. (2016) [6] đã giải quyết vấn đề này bằng kiến trúc Mạng Học Phần dư (Residual Network — ResNet) thông qua các khối kết nối tắt (skip/shortcut connections) học hàm phần dư F(x) = H(x) − x, cho phép tín hiệu đạo hàm truyền trực tiếp qua nhiều tầng mà không bị triệt tiêu. Trong báo cáo này, kiến trúc ResNet50 (50 tầng trọng số với cấu trúc bottleneck 1×1 → 3×3 → 1×1, xuất đặc trưng toàn cục 2.048 chiều sau tầng Global Average Pooling, tiền huấn luyện giám sát trên ImageNet-1K) được lựa chọn làm baseline độc lập chuẩn mực cho bài toán phân loại 10 lớp rác thải, đồng thời kế thừa bối cảnh từ các nghiên cứu phân loại rác dựa trên CNN gần đây như công trình của Islam và cs. (2025) [1].'
)

add_h2('2.2. Vision Transformer (ViT), DINOv2 và học biểu diễn tự giám sát')
add_p(
    'Khác với CNN xây dựng trường tiếp nhận từ cục bộ đến toàn cục, kiến trúc Vision Transformer (ViT) của Dosovitskiy và cs. (2021) [5] chia bức ảnh kích thước H × W × C thành một chuỗi N = (H·W)/P² các mảng phẳng không chồng lấp (image patches) kích thước P × P. Mỗi mảng được chiếu tuyến tính thành một vector nhúng D chiều, cộng với mã hóa vị trí khả học (learnable positional embeddings) và ghép thêm một token phân loại đặc biệt [CLS] ở đầu chuỗi trước khi đi qua L khối Transformer Encoder gồm cơ chế Tự chú ý Đa đầu (Multi-Head Self-Attention — MSA) và mạng truyền thẳng MLP. Nhờ cơ chế MSA, ViT có khả năng mô hình hóa mối tương quan ngữ cảnh tầm xa (long-range dependencies) giữa mọi vùng trên bức ảnh ngay từ những tầng đầu tiên.'
)
add_p(
    'Mặc dù vậy, ViT thiếu các định kiến quy nạp không gian của CNN nên thường đòi hỏi tập dữ liệu huấn luyện cực lớn. Mô hình nền tảng thị giác DINOv2 (Self-Distillation with No Labels v2) của Oquab và cs. (2024) [2] khắc phục hạn chế này bằng phương pháp học tự giám sát chưng cất tri thức không nhãn (self-supervised teacher–student distillation) kết hợp mục tiêu mức ảnh (DINO loss trên token [CLS]) và mục tiêu mức mảng bị che (iBOT masked patch loss) trên tập dữ liệu LVD-142M gồm 142 triệu ảnh được tuyển chọn tự động. Các đặc trưng rút trích từ DINOv2 nắm bắt đồng thời cấu trúc hình học vật thể, ranh giới ngữ nghĩa và kết cấu chất liệu bề mặt mà không bị thiên lệch bởi bộ nhãn cố định của ImageNet-1K. Trong đồ án này, nhóm sử dụng hai biến thể: (1) DINOv2 ViT-B/14 (12 blocks, patch size 14×14, chiều nhúng D = 768, 86,59 triệu tham số) làm bộ phân loại chính; và (2) DINOv2 ViT-S/14 (12 blocks, patch size 14×14, chiều nhúng D = 384) đóng băng làm backbone trích xuất đặc trưng cho bộ phát hiện Deformable DETR.'
)

add_h2('2.3. Họ mô hình DETR, Deformable DETR, RT-DETR và YOLO11s trong phát hiện đối tượng')
add_p(
    'Họ mô hình DETR (DEtection TRansformer) của Carion và cs. (2020) [4] định hình lại bài toán phát hiện đối tượng dưới dạng dự đoán tập hợp trực tiếp (direct set prediction) thông qua cơ chế Transformer Encoder–Decoder và thuật toán ghép cặp lưỡng phân tối ưu một-đối-một Hungarian Matcher, loại bỏ nhu cầu thiết kế anchor thủ công. Tuy nhiên, DETR nguyên bản có chi phí tính toán bậc hai theo số lượng điểm ảnh trên bản đồ đặc trưng và hội tụ chậm.'
)
add_p(
    'Deformable DETR của Zhu và cs. (2021) [3] cải thiện tốc độ hội tụ và khả năng phát hiện vật thể đa tỷ lệ bằng mô-đun Chú ý Biến dạng Đa tỷ lệ (Multi-Scale Deformable Attention — MSDeformAttn). Thay vì tính trọng số chú ý trên toàn bộ không gian bản đồ đặc trưng, mỗi truy vấn chỉ lấy mẫu tại một số lượng nhỏ K điểm lân cận có độ lệch khả học xung quanh điểm tham chiếu trên L mức bản đồ đặc trưng theo công thức:'
)

# Insert the preserved native Office Math paragraph for MSDeformAttn
p_math = doc.add_paragraph(style='Normal')
p_math.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_math.paragraph_format.keep_together = True
p_math.paragraph_format.space_before = Pt(4)
p_math.paragraph_format.space_after = Pt(4)
p_math._p.append(deepcopy(math_omath_el))

add_p(
    'Trong đó M là số đầu chú ý (attention heads), L là số mức đặc trưng đa tỷ lệ, K là số điểm lấy mẫu trên mỗi mức cho mỗi đầu chú ý, p̂_q là tọa độ điểm tham chiếu chuẩn hóa của truy vấn thứ q, Δp_mqlk là độ lệch lấy mẫu khả học (learnable sampling offset) và A_mqlk là trọng số chú ý đã chuẩn hóa softmax. Cần lưu ý rõ về mặt kỹ thuật triển khai: trong mã nguồn của dự án này (detection/Train.py), mô-đun MSDeformAttn được hiện thực hóa bằng phép nội suy song tuyến pure PyTorch (F.grid_sample) thay vì nhân mở rộng C++/CUDA (MultiScaleDeformableAttention CUDA kernel) của nhóm tác giả gốc [3]; báo cáo ghi nhận trung thực đặc điểm này để đảm bảo tính chính xác khi phân tích tốc độ và hiệu năng.'
)
add_p(
    'Để đối chứng khách quan với nhánh Deformable DETR của dự án theo yêu cầu của giảng viên, nhóm bổ sung hai baseline phát hiện độc lập đại diện cho hai hướng tiếp cận hàng đầu hiện nay: (1) YOLO11s [8] — đại diện cho trường phái mạng CNN một giai đoạn (one-stage detector) thế hệ mới của Ultralytics, sử dụng khối C3k2, SPPF, mô-đun chú ý không gian C2PSA và đầu dự đoán tách biệt (decoupled head) tối ưu hóa bằng CIoU Loss và Distribution Focal Loss (DFL); và (2) RT-DETR-L (Real-Time DEtection TRansformer) của Zhao và cs. (2024) [7] — kiến trúc Transformer phát hiện thời gian thực sử dụng Bộ mã hóa Lai (Hybrid Encoder gồm AIFI tương tác nội tầng và CCFM hợp nhất đa tỷ lệ) kết hợp cơ chế lựa chọn truy vấn nhận biết IoU (IoU-aware Query Selection).'
)

add_h2('2.4. Kiến trúc ghép nối hai giai đoạn (Cascade) và giả thuyết lệch miền vùng cắt')
add_p(
    'Trong các hệ thống thị giác máy tính mô-đun hóa, kiến trúc ghép nối hai giai đoạn (Two-Stage Cascade: Detection → Crop → Classification) tách biệt nhiệm vụ định vị vật thể (localization) khỏi nhiệm vụ phân loại chi tiết (fine-grained classification). Bộ phát hiện ở giai đoạn 1 đề xuất các vùng chứa rác tiềm năng (object proposals); sau đó mỗi vùng được cắt khỏi ảnh gốc, biến đổi kích thước và đưa vào bộ phân loại chuyên biệt ở giai đoạn 2 (DINOv2 ViT-B/14).'
)
add_p(
    'Mặc dù bộ phân loại DINOv2 đạt độ chính xác rất cao trên tập ảnh đơn đối tượng, việc ghép nối trực tiếp vào sau bộ phát hiện phải đối mặt với hai rào cản lý thuyết quan trọng: (1) Bất tương thích hệ thống nhãn (Taxonomy Mismatch), khi bộ phân loại được huấn luyện trên 10 lớp nhưng bộ dữ liệu phát hiện chỉ định nghĩa 5 lớp đích; và (2) Giả thuyết lệch miền vùng cắt (Crop Domain Shift Hypothesis): ảnh trong tập huấn luyện phân loại đơn đối tượng thường có khoảng lề nền tự nhiên bao quanh vật thể, trong khi bounding box do bộ phát hiện xuất ra lại bám sát biên vật thể. Nếu cắt sát bounding box rồi tiếp tục áp dụng phép biến đổi chuẩn Resize(256) → CenterCrop(224), khoảng 12,5% diện tích rìa ở mỗi cạnh của vật thể bị cắt mất, làm biến mất thông tin hình dáng tổng thể (như cổ chai, đáy lon, mép hộp giấy) và khiến bộ phân loại chỉ nhìn thấy một mảng bề mặt phóng đại ở tâm vật thể. Giả thuyết này là động lực trực tiếp để nhóm thiết kế nghiên cứu Ablation 5 phương pháp tiền xử lý vùng cắt (Crop Variants A–E) tại Chương 4 và Chương 5.'
)

add_h2('2.5. Các chỉ số và giao thức đánh giá chuẩn mực')
add_p(
    'Để đảm bảo tính chuẩn mực học thuật và tuân thủ tuyệt đối các quy tắc khoa học, báo cáo phân biệt rõ thước đo của từng bài toán và tuyệt đối không so sánh trực tiếp chỉ số của bài toán Classification với chỉ số của bài toán Detection:'
)
add_p(
    'Sử dụng Độ chính xác tổng thể (Accuracy — tỷ lệ mẫu phân loại đúng trên toàn tập kiểm thử) và Macro F1-score (trung bình cộng không trọng số của F1-score qua cả 10 lớp: Macro F1 = (1/10) ∑ F1_c). Do bộ dữ liệu phân loại có sự chênh lệch số lượng mẫu giữa các lớp (ví dụ lớp clothes có 1.888 ảnh trong khi lớp trash chỉ có 482 ảnh), Macro F1 là thước đo công bằng nhất vì mọi lớp — kể cả lớp thiểu số — đều có đóng góp ngang nhau. Bên cạnh đó, nhóm vẫn báo cáo Weighted F1 khi đối chiếu tham khảo với bài báo [1].',
    bold_prefix='• Chỉ số đánh giá bài toán Classification (10 lớp): '
)
add_p(
    'Sử dụng thư viện chuẩn pycocotools.cocoeval.COCOeval [11, 12] với tối đa 100 dự đoán trên mỗi ảnh (maxDets = 100) để tính: (i) mAP@50 (mean Average Precision tại ngưỡng IoU = 0,50 qua 5 lớp); (ii) mAP@50:95 (chỉ số chính của chuẩn COCO, lấy trung bình mAP qua 10 ngưỡng IoU từ 0,50 đến 0,95 với bước nhảy 0,05, phản ánh đồng thời độ chính xác phân loại và độ khít của khung bao); và (iii) AR@100 (Average Recall với tối đa 100 dự đoán/ảnh qua các ngưỡng IoU 0,50:0,95). Ngoài ra, tại ngưỡng điểm vận hành khóa (PR score threshold) trên tập Validation, nhóm trích xuất thêm Precision@50, Recall@50 và Micro-F1@50 từ kết quả ghép cặp một-đối-một tại IoU = 0,50 để phục vụ chẩn đoán lỗi.',
    bold_prefix='• Chỉ số đánh giá bài toán Detection (5 lớp): '
)
add_p(
    'Đối với các mô hình được huấn luyện và đánh giá lặp lại trên n = 3 hạt giống ngẫu nhiên độc lập (seeds 42, 123, 2026), báo cáo trình bày Trung bình mẫu (Sample Mean) kèm Độ lệch chuẩn mẫu (Sample Standard Deviation với bậc tự do hiệu chỉnh Bessel ddof = 1: s = sqrt([1/(n−1)] ∑ (x_i − x̄)²)). Đối với các mô hình chỉ có n = 1 hạt giống (RT-DETR-L và nhánh Deformable DETR/Cascade tham chiếu), báo cáo chỉ trình bày giá trị đơn lẻ và không tự tính độ lệch chuẩn.',
    bold_prefix='• Quy tắc tổng hợp thống kê qua nhiều hạt giống ngẫu nhiên (Multiple Random Seeds): '
)

# ==============================================================================
# CHƯƠNG 3. DỮ LIỆU, KIỂM TOÁN CHỐNG RÒ RỈ VÀ GIAO THỨC THỰC NGHIỆM
# ==============================================================================
add_h1('3. Dữ liệu, Kiểm toán Chống Rò rỉ và Giao thức Thực nghiệm', page_break=True)
add_h2('3.1. Bộ dữ liệu phân loại Garbage Classification v2 (10 lớp) và quy tắc ánh xạ nhãn')
add_p(
    'Nhánh phân loại sử dụng bộ dữ liệu Garbage Classification v2 (Kaggle Garbage Dataset của Suman Kunwar [10]), bao gồm tổng cộng 12.259 ảnh RGB đơn đối tượng được phân bố vào 10 lớp rác thải sinh hoạt và tái chế: battery (pin/ắc quy), biological (rác hữu cơ sinh học), cardboard (bìa cứng/carton), clothes (quần áo/vải vóc), glass (thủy tinh), metal (kim loại), paper (giấy), plastic (nhựa), shoes (giày dép) và trash (rác hỗn hợp khác). Bảng 1 trình bày phân bố số lượng ảnh chính xác của từng lớp trên toàn bộ tập dữ liệu cũng như trên ba tập con Train (9.807 ảnh), Validation (1.226 ảnh) và Final Test (1.226 ảnh).'
)
add_table_with_caption(
    'Bảng 1. Phân bố số ảnh theo 10 lớp trong bộ dữ liệu Garbage Classification v2 (12.259 ảnh) và quy tắc ánh xạ sang hệ 5 lớp phát hiện (Class Mapping)',
    TABLE_1_GARBAGE_CLS,
    col_weights=[0.95, 0.90, 0.75, 0.75, 0.75, 1.15, 1.35],
    font_size=8.8
)
add_p(
    'Bảng 1 đồng thời quy định bảng đối chiếu ánh xạ nhãn (Class Mapping) tường minh giữa không gian 10 lớp của bộ phân loại và không gian 5 lớp của bộ dữ liệu phát hiện WasteDetect (quy ước COCO ID: glass = 0, metal = 1, organic = 2, paper = 3, plastic = 4). Cụ thể: 4 lớp giữ nguyên tên định danh (glass, metal, paper, plastic); lớp biological được ánh xạ tường minh sang lớp organic (ID: 2) theo cấu hình dự án (detection/class_mapping.eval.json). Năm lớp còn lại gồm battery, cardboard, clothes, shoes và trash không tồn tại trong hệ nhãn 5 lớp của WasteDetect nên được xếp vào nhóm Ngoài phạm vi (Out-of-Scope — OOS) và xử lý theo quy tắc drop_and_count (loại bỏ dự đoán khỏi đầu ra COCO 5 lớp và ghi nhận số lượng để thống kê minh bạch, tuyệt đối không tự ý gộp cardboard vào paper).'
)

add_h2('3.2. Phân chia tập dữ liệu phân loại và khóa Manifest chống rò rỉ')
add_p(
    'Toàn bộ 12.259 ảnh của bộ dữ liệu Garbage Classification v2 được phân chia phân tầng (stratified split với random_state = 42) theo tỷ lệ 80% Train (9.807 ảnh), 10% Validation (1.226 ảnh) và 10% Final Test (1.226 ảnh). Trong bản thảo sơ bộ trước đây của nhóm từng xuất hiện lỗi ghi chép số lượng (ghi nhầm 8.289 ảnh của tập Detection Train vào cột Classification Train dẫn đến tổng 13.169 > 12.259); báo cáo này đã đính chính triệt để theo đúng manifest phân tầng thực tế (Bảng 2).'
)
add_p(
    'Để đảm bảo tính bất biến và chống rò rỉ dữ liệu giữa các lượt chạy trên máy cục bộ Windows và trên Google Colab Pro, kịch bản kiểm toán dg38_recover_locked_test.py đã kiểm tra mã băm mật mã SHA-256 của toàn bộ 12.259 tệp ảnh, xác nhận sự tách biệt giữa ba tập Train / Validation / Test và đóng gói tập kiểm thử 1.226 ảnh thành gói khóa dg38_cls_test_locked.zip (SHA-256: 98ca263b1c76aece593280ac456a0a13ec4bbbbe31a29ccdd33ec2b58cefd45c; chữ ký phân chia split_signature: 1da7e63c9ee27d1831b3595350460321897565b0375ef33da0aea1ec0e5660c7).'
)
add_table_with_caption(
    'Bảng 2. Quy mô và vai trò các tập dữ liệu phân tầng đã khóa của bài toán Classification (10 lớp) và bài toán Detection (5 lớp WasteDetect)',
    TABLE_2_SPLIT_SUMMARY,
    col_weights=[1.35, 1.05, 0.65, 0.85, 1.70],
    font_size=9.0
)

add_h2('3.3. Tiền xử lý và tăng cường dữ liệu cho từng bài toán')
add_p(
    'Trên nhánh phân loại ảnh đơn đối tượng (10 lớp): Tập huấn luyện (9.807 ảnh) áp dụng chuỗi tăng cường dữ liệu chuẩn hóa gồm RandomResizedCrop(224, scale=(0,8; 1,0)), RandomHorizontalFlip(p = 0,5), RandomRotation(10°) và ColorJitter(brightness = 0,2, contrast = 0,2, saturation = 0,2), sau đó chuyển sang Tensor và chuẩn hóa theo thống kê ImageNet (mean = [0,485; 0,456; 0,406], std = [0,229; 0,224; 0,225]). Trên tập Validation (1.226 ảnh) và Final Test (1.226 ảnh), cả ResNet50 và DINOv2 ViT-B/14 đều áp dụng chung chuỗi biến đổi tất định: Resize(256) → CenterCrop(224) → ToTensor → ImageNet Normalize.'
)
add_p(
    'Trên nhánh phát hiện đa đối tượng (5 lớp WasteDetect): Dữ liệu gốc được chuẩn hóa về định dạng nhãn COCO JSON (annotations_coco.json) cho nhánh Deformable DETR và đồng thời được chuyển đổi tự động, bảo toàn chính xác tuyệt đối số lượng khung bao sang cấu trúc thư mục YOLO (hệ tọa độ chuẩn hóa [class_id, x_center, y_center, width, height]) thông qua hàm prepare_yolo_dataset_from_coco trong dg38_det_trainer.py để phục vụ huấn luyện YOLO11s và RT-DETR-L.'
)

add_h2('3.4. Dữ liệu của bài báo tham chiếu (Islam và cs., 2025)')
add_p(
    'Bài báo tham chiếu của Islam và cs. (PLOS ONE, 2025) [1] huấn luyện mô hình kết hợp DenseNet201 và hai nhánh CNN song song có gắn cơ chế chú ý kênh Squeeze-and-Excitation (Parallel SE-CNN) trên một phiên bản mở rộng của bộ dữ liệu Waste Classification V2 gồm 21.983 ảnh chia theo tỷ lệ 70:10:20. Do phiên bản dữ liệu cục bộ của nhóm gồm 12.259 ảnh với tỷ lệ chia 80:10:10 và quy trình tiền xử lý khác nhau, mọi phép so sánh giữa mô hình của nhóm với bài báo [1] trong Chương 5 được xác định rõ là đối chiếu tham khảo học thuật, không phải thực nghiệm chạy lại trên cùng tập kiểm thử.'
)

add_h2('3.5. Bộ dữ liệu phát hiện WasteDetect và Kiểm toán khử trùng lặp đa tầng')
add_p(
    'Bộ dữ liệu phát hiện đối tượng WasteDetect_Dataset bao gồm 5 lớp rác tái chế và hữu cơ: glass (ID: 0), metal (ID: 1), organic (ID: 2), paper (ID: 3) và plastic (ID: 4). Trước khi thực hiện đánh giá mô hình và các thực nghiệm bổ sung trên Google Colab, nhóm đã tiến hành quy trình kiểm toán chống rò rỉ dữ liệu đa tầng (Multi-stage Data Leakage Audit) để loại bỏ triệt để các ảnh trùng lặp và gần trùng lặp giữa các tập phân chia:'
)
add_p(
    'Kiểm tra mã băm mật mã SHA-256 trên toàn bộ các tệp ảnh. Trên tập Validation gốc (2.316 ảnh), phát hiện và loại bỏ 506 ảnh trùng tuyệt đối với tập Train và 37 ảnh trùng nội bộ, đưa tập Validation về 1.773 ảnh. Trên tập Test gốc (1.195 ảnh / 2.645 boxes), phát hiện và loại bỏ 286 ảnh trùng với tập Train hoặc Validation và 5 ảnh trùng nội bộ, đưa tập Test về 904 ảnh / 2.002 boxes.',
    bold_prefix='1. Giai đoạn 1 — Khử trùng lặp cấp độ tệp nhị phân bằng SHA-256: '
)
add_p(
    'Để phát hiện các ảnh bị nén lại, thay đổi nhẹ kích thước hoặc chỉnh sáng giữa các tập, nhóm tính mã băm cảm nhận thị giác pHash 64-bit và duyệt toàn bộ các cặp ứng viên có khoảng cách Hamming ≤ 6. Trên tập Validation, loại bỏ thêm 28 ảnh gần trùng với Train và 2 ảnh gần trùng nội bộ (tổng cộng loại 573 ảnh so với Validation gốc), giữ lại 1.743 ảnh sạch; đồng thời phát hiện và xóa 1 nhãn bounding box bị gán đúp (annotation ID 1847 trùng với ID 1846 trên ảnh 009134.jpg), chuẩn hóa tổng số nhãn của tập Validation sạch (valid_clean_candidate) ở mức chính xác 1.743 ảnh và 3.932 bounding boxes. Trên tập Test, loại bỏ thêm 31 ảnh xác nhận gần trùng với Train/Validation, đưa tập Test về 873 ảnh / 1.889 boxes.',
    bold_prefix='2. Giai đoạn 2 — Khử trùng lặp thị giác bằng mã băm cảm nhận (Perceptual Hash — pHash ≤ 6) và làm sạch annotation: '
)
add_p(
    'Đối chiếu 873 ảnh Detection Test còn lại với toàn bộ 12.259 ảnh của bộ dữ liệu Classification (Garbage Dataset) để ngăn ngừa rò rỉ xuyên tác vụ khi đánh giá Pipeline Cascade. Qua duyệt toàn bộ các cặp ứng viên pHash ≤ 6, nhóm phát hiện và loại bỏ thêm 2 ảnh trùng lặp xuyên tập (trong khi giữ lại 23 cặp khác biệt thực sự sau kiểm tra trực quan), chốt tập Detection Final Test đã khóa (test_final_candidate / dg38_det_test_locked.zip, SHA-256: 29b92dde3d2fe5ed44a0951363767ddc37e0dfd9efc645c9c8ad1bf8e7d4aab6) ở mức chính xác 871 ảnh và 1.884 bounding boxes (giảm tổng cộng 324 ảnh so với Test gốc). Bảng 3 thống kê chi tiết phân bố số lượng bounding box theo từng lớp qua các giai đoạn kiểm toán.',
    bold_prefix='3. Giai đoạn 3 — Kiểm toán xuyên tập dữ liệu (Cross-Dataset Audit) giữa Detection và Classification: '
)
add_table_with_caption(
    'Bảng 3. Thống kê chi tiết bộ dữ liệu phát hiện WasteDetect_Dataset qua các giai đoạn kiểm toán khử trùng lặp đa tầng',
    TABLE_3_WASTEDETECT_AUDIT,
    col_weights=[0.95, 0.90, 0.90, 0.90, 0.90, 0.90, 1.05],
    font_size=8.8
)
add_p(
    'Ghi chú Bảng 3: Tỷ lệ phần trăm của mỗi lớp được tính bằng số lượng bounding box của lớp đó chia cho tổng số bounding box của chính tập phân chia tương ứng (làm tròn hai chữ số thập phân). Do một bức ảnh trong bài toán phát hiện đa đối tượng có thể chứa nhiều loại rác khác nhau, cột số lượng ảnh chứa lớp ở Final Test có sự giao nhau giữa các dòng; tổng số ảnh độc nhất của tập Final Test là đúng 871 ảnh.',
    italic=True, size=10.5
)

add_h2('3.6. Giao thức thực nghiệm bổ sung theo góp ý giảng viên: Đa hạt giống ngẫu nhiên và môi trường thực thi')
add_p(
    'Nhằm hoàn thiện toàn diện báo cáo theo ba nhóm yêu cầu bổ sung của giảng viên hướng dẫn, nhóm đã thiết kế gói thực nghiệm chuẩn hóa DG38_Colab_Package gồm 7 Google Colab Notebooks (Notebooks 00–06, tổng hợp tại Bảng 4) thực thi trên hạ tầng đám mây Google Colab Pro (sử dụng GPU NVIDIA L4 24GB VRAM, kết hợp quản trị ngân sách tính toán 100 Compute Units bằng cách giải nén dữ liệu từ Google Drive sang ổ cứng cục bộ tốc độ cao /content/dg38_data trước khi huấn luyện và lưu đồng bộ ngay mọi checkpoint, history.json, metrics.json về Google Drive sau mỗi lượt chạy).'
)
add_p(
    'Đối với yêu cầu đa hạt giống ngẫu nhiên (Multiple Random Seeds), nhóm khóa cố định bộ ba hạt giống 42, 123 và 2026 (thiết lập đồng bộ qua hàm seed_everything cho Python random, NumPy, PyTorch CPU/CUDA và bộ tạo mẫu DataLoader). Toàn bộ 14 lượt huấn luyện và đánh giá trên Validation (gồm 6 runs Classification Baselines, 4 runs Detection Baselines, 3 runs Fine-tuning Ablation và 1 bộ đánh giá 5 biến thể Crop Preprocessing Ablation) đã hoàn thành 100% (14/14 completed runs) và vượt qua bước kiểm toán tự động trong Notebook 05 với 0 cảnh báo trước khi mở gói Final Test trong Notebook 06.'
)
add_table_with_caption(
    'Bảng 4. Tổng hợp ma trận thiết kế thực nghiệm bổ sung theo góp ý giảng viên (Notebooks 00–06, số hạt giống và trạng thái hoàn thành)',
    TABLE_4_EXPERIMENT_MATRIX,
    col_weights=[1.15, 1.35, 1.25, 0.95, 1.30],
    font_size=8.8
)

# ==============================================================================
# CHƯƠNG 4. PHƯƠNG PHÁP ĐỀ XUẤT VÀ HỆ THỐNG ĐỐI CHỨNG
# ==============================================================================
add_h1('4. Phương pháp Đề xuất và Hệ thống Đối chứng', page_break=True)
add_h2('4.1. Tổng quan kiến trúc hệ thống và các nhánh thực nghiệm')
add_p(
    'Hình 1 minh họa kiến trúc tổng thể của hệ thống nhận diện rác thải trong đồ án DG38. Để đảm bảo tính rõ ràng về mặt phương pháp luận, báo cáo phân tách tường minh giữa: (1) Nhánh kiến trúc ghép nối hai giai đoạn đề xuất của dự án (Proposed Two-Stage Cascade Pipeline: Ảnh đầu vào → Bộ phát hiện Deformable DETR sử dụng backbone DINOv2 ViT-S/14 đóng băng → Tập đề xuất vật thể Object Proposals → Lọc Class-Agnostic NMS IoU = 0,70 → Tiền xử lý vùng cắt tối ưu Crop C → Bộ phân loại DINOv2 ViT-B/14 → Ánh xạ nhãn 10 lớp sang 5 lớp và xử lý OOS → Dự đoán phát hiện cuối cùng); và (2) Hai nhánh thực nghiệm đối chứng độc lập trên từng bài toán chuẩn hóa theo yêu cầu của giảng viên, gồm nhánh Classification độc lập 10 lớp (đối chứng DINOv2 ViT-B/14 với baseline ResNet50 qua 3 seeds) và nhánh Detection độc lập 5 lớp (đối chứng YOLO11s 3 seeds và RT-DETR-L 1 seed).'
)
add_figure_with_caption(
    FIG_DIR / 'fig_full_pipeline_architecture.png',
    'Hình 1. Kiến trúc tổng thể hệ thống Pipeline Cascade hai giai đoạn đề xuất (Deformable DETR → NMS → Crop C → DINOv2 ViT-B/14 → Class Mapping) và hai nhánh đối chứng độc lập (Classification 10 lớp & Detection 5 lớp)',
    max_width_in=6.25
)

add_h2('4.2. Nhánh phân loại: Mô hình đề xuất DINOv2 ViT-B/14, Baseline ResNet50 và Ablation số khối Fine-tune')
add_p(
    'Trên nhánh phân loại ảnh đơn đối tượng 10 lớp (Hình 2), mô hình đề xuất sử dụng mạng xương sống DINOv2 ViT-B/14 (facebook/dinov2-base) gồm 12 khối Transformer Encoder với chiều nhúng ẩn D = 768. Từ bức ảnh đầu vào kích thước 224 × 224 × 3, tầng Patch Embedding sử dụng phép tích chập hạt nhân 14 × 14 với bước trượt 14 để chia ảnh thành 16 × 16 = 256 patch tokens, ghép thêm 1 token phân loại toàn cục [CLS] để tạo thành chuỗi 257 tokens đi qua 12 khối Transformer. Ở đầu ra của khối cuối cùng, vector đặc trưng của token [CLS] (768 chiều) được đưa qua tầng chuẩn hóa LayerNorm(768), tầng điều chuẩn Dropout(p = 0,2) và tầng chiếu tuyến tính Linear(768 → 10) để tính logits cho 10 lớp rác thải.'
)
add_figure_with_caption(
    QA_ASSETS / 'figure1.png',
    'Hình 2. Kiến trúc chi tiết nhánh phân loại DINOv2 ViT-B/14 (embedding 768 chiều, mở khóa tinh chỉnh các khối Transformer cuối và đầu phân loại tuyến tính)',
    max_width_in=5.85
)
add_p(
    'Để đối chứng công bằng với DINOv2 ViT-B/14 theo góp ý thứ nhất của giảng viên, nhóm triển khai baseline độc lập ResNet50 (khởi tạo từ trọng số chuẩn ResNet50_Weights.IMAGENET1K_V2 của torchvision, thay tầng fc cuối bằng Linear(2048 → 10)). Cả hai mô hình được huấn luyện và đánh giá trên cùng một tập dữ liệu phân tầng (9.807 Train / 1.226 Validation / 1.226 Final Test), cùng chuỗi biến đổi tăng cường dữ liệu, cùng hàm mất mát CrossEntropyLoss với label_smoothing = 0,05, cùng kích thước lô batch_size = 64 trong 15 epochs và cùng lặp lại trên 3 hạt giống ngẫu nhiên độc lập (42, 123, 2026). Bảng 5 tổng hợp toàn bộ cấu hình kiến trúc và siêu tham số của hai mô hình phân loại.'
)
add_table_with_caption(
    'Bảng 5. Cấu hình kiến trúc và siêu tham số huấn luyện nhánh phân loại: Mô hình đề xuất DINOv2 ViT-B/14 và Baseline độc lập ResNet50 (3 random seeds: 42, 123, 2026)',
    TABLE_5_CLS_CONFIG,
    col_weights=[1.15, 1.65, 1.50],
    font_size=8.8
)
add_p(
    'Bên cạnh thực nghiệm đối chứng chính thức 3 seeds (với cấu hình mở khóa 4 khối Transformer cuối trên DINOv2 ViT-B/14), nhóm thiết kế Nghiên cứu Ablation 1 (Notebook 03 — 03_DINOv2_Finetune_Ablation_Colab.ipynb) nhằm định lượng ảnh hưởng của độ sâu tinh chỉnh (fine-tuning depth) trên tập Classification Validation (1.226 ảnh). Ba cấu hình được khảo sát có kiểm soát gồm: (i) 0 blocks (Linear Probe): đóng băng toàn bộ 12 khối Transformer của DINOv2, chỉ cập nhật tầng chuẩn hóa cuối layernorm và đầu phân loại Linear(768 → 10) với vỏn vẹn 9.226 tham số huấn luyện (0,01% trên tổng số 86.589.706 tham số); (ii) 2 blocks cuối: mở khóa thêm khối Transformer thứ 10 và 11 (encoder.layer.10..11), nâng số tham số huấn luyện lên 14.184.970 (16,38%); và (iii) 4 blocks cuối: mở khóa các khối Transformer từ 8 đến 11 (encoder.layer.8..11) cùng layernorm và head, tương ứng 28.360.714 tham số huấn luyện (32,75%).'
)

add_h2('4.3. Nhánh phát hiện: DINOv2 ViT-S/14 + Deformable DETR và hai Baseline độc lập YOLO11s, RT-DETR-L')
add_p(
    'Trên nhánh phát hiện đối tượng của dự án (Hình 3), nhóm thiết kế kiến trúc lai ghép giữa mạng xương sống DINOv2 ViT-S/14 đóng băng (frozen backbone, chiều nhúng 384) và đầu phát hiện Deformable DETR đa tỷ lệ (detection/Train.py). Do Vision Transformer xuất ra các bản đồ đặc trưng có cùng độ phân giải không gian (H/14 × W/14) ở mọi khối, mô-đun DINOv2MultiScaleBackbone rút trích các trạng thái ẩn trung gian tại 4 khối Transformer (blocks 3, 6, 9, 12), chiếu qua các tầng tích chập 1×1 sang không gian 256 kênh và áp dụng các phép co giãn không gian (interpolations/strided convolutions) để kiến tạo tháp đặc trưng 4 mức tỷ lệ đưa vào 3 tầng Deformable Transformer Encoder và 3 tầng Deformable Transformer Decoder với 100 truy vấn vật thể khả học (object queries).'
)
add_figure_with_caption(
    DOC0_MEDIA / 'image4.png',
    'Hình 3. Kiến trúc bộ phát hiện đa tỷ lệ kết hợp mạng xương sống DINOv2 ViT-S/14 đóng băng và Deformable DETR',
    max_width_in=5.95
)
add_p(
    'Như đã nêu tại Mục 2.3, bản triển khai Deformable DETR của dự án sử dụng phép lấy mẫu song tuyến pure PyTorch (F.grid_sample) cho mô-đun chú ý biến dạng đa tỷ lệ thay vì nhân CUDA C++ tùy chỉnh. Để đánh giá khách quan vị trí của kiến trúc này và đáp ứng yêu cầu bổ sung baseline phát hiện độc lập của giảng viên, nhóm triển khai trong Notebook 02 (02_Detection_Baselines_Colab.ipynb) hai mô hình phát hiện hiện đại trên cùng bộ dữ liệu WasteDetect (8.289 ảnh Train / 1.743 ảnh Validation sạch / 871 ảnh Final Test): (1) YOLO11s với 3 hạt giống ngẫu nhiên độc lập (42, 123, 2026); và (2) RT-DETR-L với 1 hạt giống ngẫu nhiên (seed = 42, do chi phí tính toán và bộ nhớ GPU của kiến trúc Transformer quy mô lớn). Để đảm bảo tính thống nhất tuyệt đối của thước đo, toàn bộ dự đoán bounding box xuất ra từ YOLO11s và RT-DETR-L đều được chuyển về định dạng danh sách dự đoán COCO và chấm điểm bằng chính hàm evaluate_coco_predictions (sử dụng pycocotools.COCOeval với maxDets = 100) giống hệt nhánh Deformable DETR. Bảng 6 đối chiếu chi tiết cấu hình của ba kiến trúc phát hiện.'
)
add_table_with_caption(
    'Bảng 6. Cấu hình kiến trúc và siêu tham số huấn luyện nhánh phát hiện: DINOv2 ViT-S/14 + Deformable DETR và hai Baseline độc lập YOLO11s, RT-DETR-L',
    TABLE_6_DET_CONFIG,
    col_weights=[1.05, 1.35, 1.25, 1.25],
    font_size=8.6
)

add_h2('4.4. Giao thức đối chiếu tham khảo với bài báo Islam và cs. (2025)')
add_p(
    'Bảng 7 tóm tắt các điểm khác biệt cốt lõi về cấu hình thực nghiệm giữa mô hình phân loại DINOv2 ViT-B/14 của nhóm và mô hình DenseNet201 + Parallel SE-CNN trong bài báo tham chiếu của Islam và cs. (2025) [1]. Việc đặt bảng so sánh cấu hình này ngay trong chương phương pháp giúp làm rõ vì sao các số liệu đối chiếu ở Mục 5.3 chỉ mang tính chất tham khảo học thuật chứ không phải phép so sánh trực tiếp trên cùng một tập kiểm thử.'
)
add_table_with_caption(
    'Bảng 7. So sánh cấu hình thực nghiệm giữa bài báo tham chiếu Islam và cs. (2025) [1] và mô hình phân loại DINOv2 ViT-B/14 của nhóm',
    TABLE_7_REF_PAPER_CONFIG,
    col_weights=[1.10, 1.55, 1.65],
    font_size=9.0
)

add_h2('4.5. Hệ thống Pipeline Cascade hai giai đoạn và Thiết kế Ablation 5 biến thể tiền xử lý vùng cắt (Crop A–E)')
add_p(
    'Phương pháp trọng tâm của dự án là hệ thống Pipeline Cascade hai giai đoạn ghép nối tuần tự bộ phát hiện Deformable DETR với bộ phân loại DINOv2 ViT-B/14 (detection/waste_cascade.py và dg38_crop_ablation.py). Quy trình suy luận đầy đủ của một bức ảnh đầu vào diễn ra qua 5 bước chặt chẽ:'
)
add_p(
    'Ảnh đầu vào được đưa qua bộ phát hiện DINOv2 ViT-S/14 + Deformable DETR để sinh ra tối đa 100 đề xuất vật thể (object proposals), mỗi đề xuất thứ i gồm tọa độ khung bao b_i = [x1, y1, x2, y2], nhãn dự đoán sơ bộ của detector và điểm tự tin định vị det_score_i (với ngưỡng đề xuất khởi tạo proposal_threshold = 0,00).',
    bold_prefix='• Bước 1 — Sinh đề xuất vật thể (Object Proposal Generation): '
)
add_p(
    'Do cơ chế giải mã không dùng NMS của Deformable DETR trong giai đoạn huấn luyện ban đầu vẫn sinh ra nhiều truy vấn chồng lấp xung quanh cùng một vật thể, hệ thống áp dụng thuật toán Lọc triệt tiêu phi cực đại không phân biệt lớp (Class-Agnostic NMS) với ngưỡng giao trên hợp khóa từ Validation là NMS IoU = 0,70 (giữ lại tối đa max_proposals = 100 đề xuất sau lọc).',
    bold_prefix='• Bước 2 — Lọc đề xuất trùng lặp bằng Class-Agnostic NMS: '
)
add_p(
    'Mỗi khung bao đề xuất hợp lệ sau NMS được cắt khỏi bức ảnh gốc và biến đổi về kích thước chuẩn 224 × 224 trước khi đưa vào bộ phân loại DINOv2 ViT-B/14 để thu được nhãn phân loại dự đoán c_cls thuộc tập 10 lớp và xác suất tự tin tương ứng cls_score_i.',
    bold_prefix='• Bước 3 — Tiền xử lý vùng cắt (RoI Cropping) và Phân loại bằng DINOv2 ViT-B/14: '
)
add_p(
    'Nhãn phân loại c_cls (10 lớp) được ánh xạ sang không gian 5 lớp của bài toán Detection theo quy tắc tại Bảng 1 (glass → 0, metal → 1, biological → organic (2), paper → 3, plastic → 4). Nếu c_cls rơi vào một trong 5 lớp ngoài phạm vi (OOS: battery, cardboard, clothes, shoes, trash), dự đoán đó bị loại bỏ khỏi tập kết quả phát hiện 5 lớp và được ghi nhận vào bộ đếm kiểm toán (drop_and_count).',
    bold_prefix='• Bước 4 — Ánh xạ nhãn 10 lớp sang 5 lớp và Xử lý dự đoán ngoài phạm vi (OOS Handling): '
)
add_p(
    'Đối với các đề xuất thuộc 5 lớp hợp lệ, điểm tự tin cuối cùng của dự đoán Cascade được tính theo công thức kết hợp nhân tính: final_score_i = det_score_i × cls_score_i. Khi đánh giá chỉ số chuẩn COCO (mAP@50, mAP@50:95, AR@100), toàn bộ danh sách dự đoán hợp lệ được đưa vào COCOeval; khi tính Precision@50, Recall@50 và F1@50 tại điểm vận hành thực tế, hệ thống sử dụng ngưỡng khóa từ Validation là Detector PR threshold = 0,30 và Cascade PR threshold = 0,25 (Bảng 8).',
    bold_prefix='• Bước 5 — Kết hợp điểm tự tin (Score Fusion) và Xuất dự đoán cuối: '
)
add_table_with_caption(
    'Bảng 8. Cấu hình các phương pháp trong nhánh Deformable DETR / Cascade và bộ thông số khóa từ tập Validation sạch (1.743 ảnh)',
    TABLE_8_CASCADE_LOCKED_CONFIG,
    col_weights=[0.95, 1.25, 0.80, 0.70, 0.75, 1.15, 1.30],
    font_size=8.5
)
add_p(
    'Để giải quyết triệt để câu hỏi liệu phương pháp cắt ảnh sát bounding box (Crop A) có phải là nguyên nhân gây suy giảm hiệu năng của bộ phân loại DINOv2 trong Cascade hay không (Yêu cầu bổ sung thứ 3b của giảng viên), nhóm đã thiết kế Nghiên cứu Ablation 2 (Notebook 04 — 04_Crop_Preprocessing_Ablation_Colab.ipynb, mã nguồn dg38_crop_ablation.py) khảo sát 5 biến thể tiền xử lý vùng cắt (Crop Variants A, B, C, D, E) trên toàn bộ 1.743 ảnh Validation sạch (3.932 GT boxes) với cùng checkpoint Detector (best_model.pt), cùng checkpoint Classifier (dinov2_garbage_final.pth) và cùng ngưỡng khóa NMS IoU = 0,70, PR threshold = 0,25. Đặc tả kỹ thuật của cả 5 biến thể được trình bày chi tiết tại Bảng 9.'
)
add_table_with_caption(
    'Bảng 9. Đặc tả kỹ thuật 5 biến thể tiền xử lý vùng cắt (Crop Preprocessing Variants A–E) được khảo sát trong Pipeline Cascade trên tập Validation sạch',
    TABLE_9_CROP_VARIANTS_SPEC,
    col_weights=[0.75, 1.15, 1.15, 1.55, 1.40],
    font_size=8.6
)
print('Chapters 1-4 appended successfully')

# ==============================================================================
# CHƯƠNG 5. KẾT QUẢ THỰC NGHIỆM VÀ ĐÁNH GIÁ
# ==============================================================================
add_h1('5. Kết quả Thực nghiệm và Đánh giá', page_break=True)

add_h2('5.1. Kết quả Classification (10 lớp) trên Validation và Final Test qua 3 Random Seeds')
add_p(
    'Bảng 10 và Hình 4 trình bày kết quả đối chứng độc lập giữa baseline ResNet50 và mô hình đề xuất DINOv2 ViT-B/14 (cấu hình mở khóa 4 khối Transformer cuối) trên tập Validation (1.226 ảnh) và tập Final Test đã khóa (1.226 ảnh) qua ba hạt giống ngẫu nhiên độc lập (seeds 42, 123, 2026). Toàn bộ số liệu được tổng hợp theo Trung bình mẫu ± Độ lệch chuẩn mẫu (Mean ± Sample Standard Deviation với ddof = 1, đơn vị tính bằng phần trăm %; kèm theo số thực gốc 6 chữ số thập phân từ kết quả kiểm toán tại Notebook 05 và Notebook 06).'
)
add_table_with_caption(
    'Bảng 10. Kết quả thực nghiệm Classification (10 lớp) của ResNet50 và DINOv2 ViT-B/14 trên tập Validation (1.226 ảnh) và Final Test đã khóa (1.226 ảnh) qua 3 random seeds (42, 123, 2026 — Mean ± Sample Std)',
    TABLE_10_CLS_3SEEDS,
    col_weights=[1.25, 0.85, 0.95, 0.95, 1.05, 1.05, 0.95],
    font_size=8.6
)
add_figure_with_caption(
    FIG_DIR / 'fig_cls_3seeds_val_vs_test.png',
    'Hình 4. So sánh hiệu năng Classification (10 lớp) giữa ResNet50 và DINOv2 ViT-B/14 trên tập Validation và Final Test đã khóa qua 3 random seeds (Mean ± Sample Std, n = 3)',
    max_width_in=6.15
)
add_p(
    'Từ số liệu thực nghiệm tại Bảng 10 và Hình 4, nhóm rút ra bốn phân tích định lượng trọng tâm sau:'
)
add_p(
    'Trên tập Validation (1.226 ảnh), DINOv2 ViT-B/14 đạt Accuracy 97,63% ± 0,22 và Macro F1 97,33% ± 0,20, cao hơn ResNet50 (Accuracy 96,25% ± 0,08; Macro F1 95,51% ± 0,21) lần lượt +1,38 điểm phần trăm về Accuracy và +1,82 điểm phần trăm về Macro F1. Khi đánh giá một lần trên tập Final Test đã khóa (1.226 ảnh), DINOv2 ViT-B/14 đạt Accuracy trung bình 98,61% ± 0,29 (mean = 0,986134; sample std = 0,002941) và Macro F1 trung bình 98,40% ± 0,30 (mean = 0,984033; sample std = 0,003002), vượt qua baseline ResNet50 (Accuracy mean = 0,962208 ± 0,002865 tức 96,22% ± 0,29; Macro F1 mean = 0,954968 ± 0,002713 tức 95,50% ± 0,27) tới +2,39 điểm phần trăm về Accuracy (+0,023926) và +2,91 điểm phần trăm về Macro F1 (+0,029065). Xét theo tỷ lệ giảm lỗi tương đối trên tập Test, DINOv2 giúp giảm 63,3% tổng số mẫu phân loại sai so với ResNet50 (sai số trung bình giảm từ 3,78% xuống 1,39%).',
    bold_prefix='1. Mức cải thiện của DINOv2 ViT-B/14 so với ResNet50: '
)
add_p(
    'Cả hai kiến trúc đều thể hiện độ ổn định rất cao khi thay đổi hạt giống ngẫu nhiên khởi tạo đầu phân loại và thứ tự xáo trộn lô dữ liệu (seeds 42, 123, 2026). Độ lệch chuẩn mẫu (sample std) trên tập Validation chỉ nằm trong khoảng 0,08 đến 0,22 điểm phần trăm, và trên tập Final Test chỉ dao động từ 0,27 đến 0,30 điểm phần trăm. Khoảng biến thiên hẹp này cho thấy quy trình huấn luyện với tối ưu hóa AdamW, lịch giảm tốc độ học CosineAnnealingLR và label smoothing = 0,05 hội tụ nhất quán, không bị phụ thuộc ngẫu nhiên vào một hạt giống đơn lẻ.',
    bold_prefix='2. Độ ổn định qua ba hạt giống ngẫu nhiên (3 Random Seeds): '
)
add_p(
    'Đối với baseline ResNet50, hiệu năng gần như giữ nguyên tuyệt đối khi chuyển từ Validation sang Final Test (Accuracy từ 96,25% sang 96,22%, chênh lệch −0,03 điểm phần trăm; Macro F1 từ 95,51% sang 95,50%, chênh lệch −0,01 điểm phần trăm). Đối với DINOv2 ViT-B/14, kết quả trên Final Test (Accuracy 98,61%; Macro F1 98,40%) cao hơn trên Validation (Accuracy 97,63%; Macro F1 97,33%) lần lượt +0,98 điểm phần trăm và +1,07 điểm phần trăm. Hiện tượng điểm Test nhỉnh hơn nhẹ so với điểm Validation (~1 điểm phần trăm trên tập mẫu kích thước 1.226 ảnh, tương ứng chênh lệch khoảng 12 ảnh phân loại đúng) xuất phát từ sự biến thiên tự nhiên về độ khó mẫu giữa hai tập con 10% được chia phân tầng ngẫu nhiên. Như đã xác minh bằng mã băm SHA-256 tại Chương 3, tập Test hoàn toàn tách biệt với tập Train/Validation ở cấp độ tệp ảnh; tuy nhiên, sự chênh lệch dương này cũng không tự nó chứng minh hay phủ nhận hoàn toàn khả năng có các ảnh cùng bối cảnh chụp trong bộ dữ liệu gốc của Kaggle.',
    bold_prefix='3. Phân tích chênh lệch giữa Validation và Final Test: '
)
add_p(
    'Ưu thế của DINOv2 ViT-B/14 trước ResNet50 trong bài toán phân loại rác thải đơn đối tượng có thể giải thích bởi ba yếu tố kỹ thuật: (i) Đặc trưng tiền huấn luyện tự giám sát trên tập LVD-142M (142 triệu ảnh) bằng mục tiêu chưng cất mức ảnh và mức mảng bị che (iBOT) giúp mô hình học được đặc trưng hình học vật thể và kết cấu bề mặt chất liệu (độ bóng của kim loại, độ trong suốt/khúc xạ của thủy tinh và nhựa, thớ sợi của giấy và bìa carton) giàu thông tin hơn so với đặc trưng tiền huấn luyện có giám sát theo nhãn lớp của ImageNet-1K; (ii) Cơ chế Tự chú ý Đa đầu (Multi-Head Self-Attention) trên lưới mảng 14 × 14 cho phép token [CLS] tổng hợp ngữ cảnh toàn cục của toàn bộ hình dáng vật thể ngay cả khi rác bị bóp méo, thay vì bị giới hạn bởi trường tiếp nhận tích chập cục bộ ở các tầng đầu của ResNet50; và (iii) Chiến lược tinh chỉnh phân tầng (chỉ mở khóa 4 khối Transformer cuối với tốc độ học nhỏ cho backbone và tốc độ học lớn hơn cho head) giúp thích nghi hiệu quả với miền dữ liệu rác thải mà không làm phá vỡ tri thức tiền huấn luyện. Lưu ý chuẩn mực khoa học: do cỡ mẫu hạt giống là n = 3 seeds và nhóm không thực hiện kiểm định giả thuyết thống kê tham số/phi tham số trên phân phối hạt giống, báo cáo ghi nhận mức vượt trội +2,39 điểm % Accuracy và +2,91 điểm % Macro F1 như một kết quả quan sát thực nghiệm nhất quán trên bộ dữ liệu hiện tại, không tuyên bố kết luận có ý nghĩa thống kê (p-value) tổng quát cho mọi tập dữ liệu.',
    bold_prefix='4. Các yếu tố giúp biểu diễn DINOv2 hiệu quả và lưu ý về mặt thống kê: '
)

add_h2('5.2. Phân tích chi tiết theo từng lớp (10 lớp) và Confusion Matrix của DINOv2 ViT-B/14')
add_p(
    'Để đi sâu vào hành vi phân loại ở cấp độ từng danh mục vật liệu rác thải, Hình 5, Bảng 11 và Hình 6 trình bày báo cáo phân loại chi tiết (Precision, Recall, F1-score, Support) và ma trận nhầm lẫn (Confusion Matrix) trên đầy đủ 1.226 ảnh của tập Classification Test đối với checkpoint tham chiếu DINOv2 ViT-B/14 (seed = 42, tệp checkpoints/dinov2_garbage_final.pth, đạt Accuracy đơn lẻ 98,69% và Macro F1 98,43% — hoàn toàn nằm trong dải phân bố 3 seeds 98,61% ± 0,29 và 98,40% ± 0,30 ở Bảng 10).'
)
add_figure_with_caption(
    QA_ASSETS / 'figure3.png',
    'Hình 5. Báo cáo phân loại chi tiết (Classification Report) của checkpoint tham chiếu DINOv2 ViT-B/14 (seed 42) trên 1.226 ảnh Test, dựng lại từ per_class_metrics đã kiểm toán',
    max_width_in=5.65
)
add_table_with_caption(
    'Bảng 11. Kết quả phân loại chi tiết theo từng lớp (10 lớp) của checkpoint tham chiếu DINOv2 ViT-B/14 (seed 42) trên tập Classification Test (1.226 ảnh)',
    TABLE_11_CLS_10CLASSES_SEED42,
    col_weights=[1.35, 0.95, 0.95, 0.95, 1.15],
    font_size=9.2
)
add_figure_with_caption(
    QA_ASSETS / 'figure4.png',
    'Hình 6. Ma trận nhầm lẫn (Confusion Matrix) của bộ phân loại DINOv2 ViT-B/14 trên 1.226 ảnh Classification Test (1.210 ảnh dự đoán đúng, 16 ảnh nhầm lẫn)',
    max_width_in=5.35
)
add_p(
    'Phân tích Bảng 11 và Hình 6 cho thấy: trên tổng số 1.226 ảnh kiểm thử, mô hình phân loại đúng 1.210 ảnh và chỉ nhầm lẫn 16 ảnh. Các lớp có đặc trưng hình thái riêng biệt như clothes (189 ảnh, F1 = 99,74%), shoes (147 ảnh, Recall = 100,00%, F1 = 99,32%), biological (70 ảnh, Precision = 100,00%, F1 = 99,28%) và glass (174 ảnh, F1 = 99,14%) đạt độ chính xác gần như tuyệt đối. Lớp có F1-score thấp nhất là trash (48 ảnh kiểm thử, Precision = 97,83%, Recall = 93,75%, F1 = 95,74%) do đây là lớp thiểu số (chỉ chiếm 3,93% bộ dữ liệu) và mang tính chất danh mục tổng hợp chứa nhiều mẫu rác hỗn hợp khó định hình. Các cặp nhầm lẫn cụ thể được ghi nhận trên ma trận gồm: cardboard dự đoán nhầm thành paper (3 ảnh — do bề mặt bìa cứng phẳng tương đồng với giấy dày), trash nhầm thành metal (2 ảnh), cùng 11 trường hợp nhầm lẫn đơn lẻ (1 ảnh) ở các cặp lớp khác (chẳng hạn paper → cardboard 1 ảnh, plastic → glass 1 ảnh).'
)

add_h2('5.3. Đối chiếu tham khảo với bài báo Islam và cs. (2025)')
add_p(
    'Hình 7, Bảng 12, Bảng 13, Hình 8 và Hình 9 trình bày kết quả đối chiếu tham khảo giữa mô hình DINOv2 ViT-B/14 của nhóm và mô hình DenseNet201 + Parallel SE-CNN trong bài báo công bố của Islam và cs. (PLOS ONE, 2025) [1]. Tuân thủ chặt chẽ nguyên tắc trung thực học thuật, do bài báo gốc không nêu rõ công thức lấy trung bình tổng thể (Macro hay Weighted), Bảng 12 đặt song song cả hai cột Macro Average và Weighted Average của nhóm bên cạnh số liệu công bố của bài báo.'
)
add_figure_with_caption(
    DOC0_MEDIA / 'image2.png',
    'Hình 7. Biểu đồ các chỉ số tổng thể của mô hình DenseNet201 + Parallel SE-CNN trong bài báo tham chiếu Islam và cs. (2025) [1], Fig 6',
    max_width_in=5.40
)
add_table_with_caption(
    'Bảng 12. Đối chiếu tham khảo các chỉ số tổng thể giữa bài báo Islam và cs. (2025) [1] và mô hình DINOv2 ViT-B/14 của nhóm',
    TABLE_12_REF_OVERALL,
    col_weights=[1.15, 1.15, 1.05, 1.05, 1.20],
    font_size=9.0
)
add_p(
    'Ghi chú Bảng 12: Cột chênh lệch lấy giá trị ở cột Weighted Avg của nhóm trừ đi số công bố trong bài báo tham chiếu; chỉ số Accuracy là độ chính xác toàn cục. Do hai nghiên cứu khác phiên bản quy mô dữ liệu (21.983 ảnh so với 12.259 ảnh) và khác tỷ lệ chia tập kiểm thử (70:10:20 so với 80:10:10), phép đối chiếu này chỉ mang ý nghĩa tham khảo học thuật, không dùng để kết luận ưu thế tuyệt đối.',
    italic=True, size=10.5
)
add_table_with_caption(
    'Bảng 13. Đối chiếu tham khảo theo từng lớp trên 5 lớp rác chung giữa bài báo Islam và cs. (2025) [1] và mô hình DINOv2 ViT-B/14 của nhóm',
    TABLE_13_REF_5COMMON,
    col_weights=[1.05, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85],
    font_size=9.0
)
add_figure_with_caption(
    DOC0_MEDIA / 'image9.png',
    'Hình 8. Confusion Matrix trên tập Waste Classification V2 của bài báo tham chiếu Islam và cs. (2025) [1], Fig 5',
    max_width_in=5.35
)
add_figure_with_caption(
    QA_ASSETS / 'figure7.png',
    'Hình 9. Biểu đồ đối chiếu tham khảo Recall và F1-score trên 5 lớp chung theo Bảng 13 (hai mô hình khác tập dữ liệu và phân chia kiểm thử)',
    max_width_in=5.85
)

add_h2('5.4. Kết quả bài toán Detection (5 lớp): YOLO11s (3 Seeds) và RT-DETR-L (1 Seed) trên Validation và Final Test')
add_p(
    'Đáp ứng yêu cầu bổ sung baseline độc lập cho bài toán phát hiện đối tượng 5 lớp trên bộ dữ liệu WasteDetect, nhóm đã huấn luyện và đánh giá YOLO11s (với 3 hạt giống ngẫu nhiên: 42, 123, 2026) và RT-DETR-L (với 1 hạt giống ngẫu nhiên: 42) trong Notebook 02 (Validation) và Notebook 06 (Final Test). Bảng 14 tổng hợp toàn bộ kết quả đánh giá trên tập Validation sạch (1.743 ảnh, 3.932 GT boxes), đặt cạnh kết quả Validation đã xác minh của nhánh Deformable DETR và Pipeline Cascade của dự án trên đúng cùng tập 1.743 ảnh Validation sạch này.'
)
add_table_with_caption(
    'Bảng 14. Kết quả đánh giá bài toán Detection (5 lớp WasteDetect) trên tập Validation sạch (1.743 ảnh, 3.932 GT boxes) của YOLO11s (3 seeds), RT-DETR-L (1 seed) và các cấu hình Deformable DETR / Cascade (1 seed)',
    TABLE_14_DET_VALIDATION,
    col_weights=[1.25, 1.25, 0.75, 0.80, 0.80, 0.75, 1.40],
    font_size=8.4
)
add_p(
    'Tiếp theo, Bảng 15 trình bày đầy đủ kết quả đánh giá trên tập Detection Final Test đã khóa (871 ảnh, 1.884 GT boxes — gói dg38_det_test_locked.zip trong Notebook 06) cho từng lượt chạy theo hạt giống (seed 42, seed 123, seed 2026) cũng như dòng tổng hợp Trung bình ± Độ lệch chuẩn mẫu (Mean ± Sample Std, ddof = 1) của YOLO11s (3 seeds) và kết quả đơn lẻ của RT-DETR-L (1 seed). Bảng 16 và Hình 10 đối chiếu trực tiếp tính nhất quán giữa tập Validation sạch và tập Final Test đã khóa của hai baseline phát hiện độc lập.'
)
add_table_with_caption(
    'Bảng 15. Kết quả đánh giá Detection Final Test đã khóa (871 ảnh, 1.884 GT boxes) theo từng hạt giống (42, 123, 2026) và tổng hợp Mean ± Sample Std cho YOLO11s (3 seeds) và RT-DETR-L (1 seed)',
    TABLE_15_DET_FINAL_TEST,
    col_weights=[1.05, 1.25, 0.65, 0.95, 0.95, 0.90, 1.35],
    font_size=8.5
)
add_table_with_caption(
    'Bảng 16. Đối chiếu tính nhất quán giữa tập Validation sạch (1.743 ảnh) và tập Final Test đã khóa (871 ảnh) của hai baseline phát hiện YOLO11s và RT-DETR-L',
    TABLE_16_DET_VAL_VS_TEST,
    col_weights=[1.20, 0.75, 0.95, 0.95, 0.90, 1.45],
    font_size=8.6
)
add_figure_with_caption(
    FIG_DIR / 'fig_det_baselines_val_vs_test.png',
    'Hình 10. So sánh hiệu năng Detection Baselines (5 lớp WasteDetect) giữa YOLO11s (3 seeds, Mean ± Sample Std) và RT-DETR-L (1 seed, không vẽ thanh sai số) trên tập Validation sạch và Final Test đã khóa',
    max_width_in=6.15
)
add_p(
    'Từ kết quả định lượng tại Bảng 14, Bảng 15, Bảng 16 và Hình 10, nhóm rút ra các nhận định khoa học sau:'
)
add_p(
    'Từ ba lượt chạy độc lập trên tập Final Test đã khóa (seed 42: mAP@50 = 0,832441, mAP@50:95 = 0,608381, AR@100 = 0,695373; seed 123: mAP@50 = 0,825127, mAP@50:95 = 0,602519, AR@100 = 0,694642; seed 2026: mAP@50 = 0,825936, mAP@50:95 = 0,600146, AR@100 = 0,691502), mô hình YOLO11s đạt trung bình và độ lệch chuẩn mẫu (ddof = 1) là: mAP@50 = 82,78% ± 0,40 (0,827835 ± 0,004010), mAP@50:95 = 60,37% ± 0,42 (0,603682 ± 0,004239) và AR@100 = 69,38% ± 0,21 (0,693839 ± 0,002057). Độ lệch chuẩn mẫu dưới 0,43 điểm phần trăm trên cả ba chỉ số chứng tỏ quá trình huấn luyện YOLO11s trên bộ dữ liệu WasteDetect cực kỳ ổn định.',
    bold_prefix='1. Hiệu năng và độ ổn định của YOLO11s qua 3 hạt giống ngẫu nhiên: '
)
add_p(
    'Trên tập Validation sạch, YOLO11s (84,94% mAP@50; 61,45% mAP@50:95) vượt RT-DETR-L một hạt giống (71,20% mAP@50; 49,43% mAP@50:95) lần lượt +13,74 điểm phần trăm về mAP@50 và +12,02 điểm phần trăm về mAP@50:95. Trên tập Final Test đã khóa, thứ tự hiệu năng hoàn toàn giữ nguyên: YOLO11s (82,78% mAP@50; 60,37% mAP@50:95) cao hơn RT-DETR-L ở seed 42 (mAP@50 = 0,701251 tức 70,13%; mAP@50:95 = 0,485653 tức 48,57%) khoảng +12,66 điểm phần trăm về mAP@50 và +11,80 điểm phần trăm về mAP@50:95. Đáng chú ý, chỉ số độ nhạy trung bình AR@100 trên Final Test của RT-DETR-L (69,55%) và YOLO11s (69,38% ± 0,21) gần như ngang bằng nhau; điều này cho thấy bộ giải mã 300 truy vấn của RT-DETR-L có khả năng bao phủ vật thể thật (recall) tốt, nhưng bị suy giảm mAP do sinh nhiều dự đoán dương tính giả có điểm tự tin cao hoặc định vị khung bao kém khít hơn so với đầu dự đoán CNN đa tỷ lệ của YOLO11s trên tập dữ liệu quy mô vừa (8.289 ảnh Train). Do RT-DETR-L chỉ có 1 seed, nhóm không thực hiện kiểm định thống kê giữa hai mô hình này.',
    bold_prefix='2. Đối chiếu giữa YOLO11s và RT-DETR-L cũng như tính nhất quán Validation – Test: '
)
add_p(
    'Trên tập Validation sạch (Bảng 14), cấu hình tốt nhất của nhánh Deformable DETR một giai đoạn (detector_nms: mAP@50 = 76,17%, mAP@50:95 = 44,45%) đạt mAP@50 cao hơn RT-DETR-L (+4,97 điểm %), nhưng thấp hơn YOLO11s (−8,77 điểm % mAP@50 và −17,00 điểm % mAP@50:95). Trong khi đó, cấu hình Cascade tốt nhất trên Validation (cascade_nms với Crop C: mAP@50 = 46,69%, mAP@50:95 = 26,55%) vẫn thấp hơn đáng kể so với cả Detector độc lập lẫn YOLO11s. Do đó, báo cáo khẳng định rõ ràng: kiến trúc Cascade chưa vượt qua YOLO11s, và nhánh Deformable DETR / Cascade hiện chỉ dừng ở kết quả đánh giá trên Validation sạch, chưa có kết quả Final Test mới trong gói Notebook 06.',
    bold_prefix='3. Đặt nhánh Deformable DETR và Cascade trong bức tranh tổng thể trên Validation: '
)

add_h2('5.5. Nghiên cứu Ablation 1: Ảnh hưởng của số Transformer blocks được fine-tune trên DINOv2 ViT-B/14')
add_p(
    'Bảng 17 và Hình 11 trình bày kết quả của Nghiên cứu Ablation 1 (Notebook 03 — 03_DINOv2_Finetune_Ablation_Colab.ipynb) khảo sát ảnh hưởng của việc thay đổi số khối Transformer Encoder cuối cùng được mở khóa tinh chỉnh (0, 2 và 4 blocks) trên mô hình DINOv2 ViT-B/14. Cần nhấn mạnh rõ về mặt giao thức: đây là kết quả nghiên cứu Ablation thực hiện trên tập Classification Validation (1.226 ảnh, 10 lớp) với một hạt giống cố định (seed = 42) để phục vụ lựa chọn cấu hình mô hình, tuyệt đối không phải là kết quả trên tập Test hay trung bình Mean ± Std của 3 seeds.'
)
add_table_with_caption(
    'Bảng 17. Kết quả nghiên cứu Ablation 1 trên tập Classification Validation (1.226 ảnh): Ảnh hưởng của số khối Transformer được fine-tune (0, 2, 4 blocks) trên DINOv2 ViT-B/14',
    TABLE_17_FINETUNE_ABLATION,
    col_weights=[1.05, 1.55, 1.00, 0.80, 0.90, 0.85],
    font_size=8.8
)
add_figure_with_caption(
    FIG_DIR / 'fig_finetune_blocks_ablation.png',
    'Hình 11. Kết quả Ablation 1 trên tập Classification Validation (1.226 ảnh): Mối quan hệ giữa số khối Transformer được mở khóa tinh chỉnh (0, 2, 4 blocks), tỷ lệ tham số huấn luyện và Validation Macro F1 (%)',
    max_width_in=5.85
)
add_p(
    'Phân tích sự đánh đổi giữa số khối mở khóa, hiệu năng và chi phí tính toán từ Bảng 17 và Hình 11:'
)
add_p(
    'Ngay cả khi đóng băng hoàn toàn 12 khối Transformer của DINOv2 ViT-B/14 và chỉ huấn luyện tầng chuẩn hóa cuối cùng với đầu phân loại tuyến tính (0 blocks — Linear Probe, chỉ cập nhật 9.226 tham số, tương đương 0,01% tổng dung lượng mạng), mô hình đã đạt Validation Macro F1 = 96,46% (cao hơn cả mạng ResNet50 fine-tune 100% tham số vốn đạt 95,51% ± 0,21). Khi mở khóa 2 khối Transformer cuối (blocks 10–11, tương ứng 14.184.970 tham số hay 16,38%), Validation Macro F1 tăng lên 96,99% (+0,53 điểm phần trăm). Khi mở khóa 4 khối Transformer cuối (blocks 8–11, tương ứng 28.360.714 tham số hay 32,75%), Validation Macro F1 tiếp tục tăng lên 97,33% (+0,87 điểm phần trăm so với 0 blocks và +0,34 điểm phần trăm so với 2 blocks). Kết quả này chứng tỏ việc cho phép các tầng tự chú ý cấp cao ở cuối backbone điều chỉnh trọng số giúp không gian nhúng [CLS] phân tách tốt hơn các lớp rác có bề mặt tương đồng trong miền dữ liệu mục tiêu.',
    bold_prefix='• Về hiệu năng thích nghi miền (Domain Adaptation Performance): '
)
add_p(
    'Về mặt nguyên lý tính toán (nhóm phân tích định tính trên cơ sở cấu trúc giải tích của mạng, không tự tạo số đo thời gian hay bộ nhớ GPU khi không lưu benchmark riêng cho từng cấu hình trong Notebook 03): Ở cấu hình 0 blocks, quá trình lan truyền ngược (backpropagation) chỉ diễn ra tại tầng Linear cuối nên không cần lưu đồ thị tính toán đạo hàm trung gian (activation tensors) của 12 khối Transformer, giúp tiết kiệm tối đa bộ nhớ VRAM và thời gian cập nhật trọng số. Khi tăng số khối mở khóa lên 2 blocks và 4 blocks, mặc dù chi phí lan truyền tiến (forward pass) qua 12 khối là như nhau, bộ tối ưu hóa AdamW phải lưu trữ thêm trạng thái moment bậc nhất và bậc hai cho 14,18 triệu (ở 2 blocks) hoặc 28,36 triệu tham số (ở 4 blocks), đồng thời phải lưu trữ các tensor kích hoạt trung gian và tính đạo hàm ngược xuyên qua 2 hoặc 4 khối cuối cùng. Cấu hình 4 blocks đạt điểm cân bằng hợp lý nhất trong phạm vi khảo sát: mở khóa khoảng 1/3 dung lượng mạng ở các tầng ngữ nghĩa cao để đạt Macro F1 đỉnh (97,33%), trong khi vẫn giữ đóng băng 8 khối đầu (67,25% tham số) để bảo toàn các bộ lọc đặc trưng thị giác cơ sở và ngăn ngừa hiện tượng quá khớp (overfitting). Do phạm vi thí nghiệm chỉ khảo sát 0, 2 và 4 blocks, báo cáo không suy diễn rằng mở khóa nhiều hơn 4 blocks sẽ tiếp tục tăng hiệu năng.',
    bold_prefix='• Về đánh đổi chi phí tính toán và bộ nhớ (Computational & Memory Trade-off): '
)

add_h2('5.6. Nghiên cứu Ablation 2: Ảnh hưởng của 5 phương pháp tiền xử lý vùng cắt (Crop A–E) trong Cascade')
add_p(
    'Bảng 18 và Hình 12 trình bày kết quả của Nghiên cứu Ablation 2 (Notebook 04 — 04_Crop_Preprocessing_Ablation_Colab.ipynb, thực thi qua module dg38_crop_ablation.py) trên toàn bộ 1.743 ảnh Validation sạch (3.932 GT boxes). Trong thí nghiệm này, cả 5 biến thể tiền xử lý vùng cắt (Crop A, B, C, D, E) đều sử dụng cùng một bộ 64.819 đề xuất vật thể đã lọc qua Class-Agnostic NMS (IoU = 0,70) từ checkpoint Detector cố định (best_model.pt) và cùng đưa vào một checkpoint Classifier cố định (dinov2_garbage_final.pth) với ngưỡng vận hành khóa PR threshold = 0,25. Lưu ý quan trọng: toàn bộ các chỉ số trong Bảng 18 là chỉ số phát hiện đối tượng chuẩn COCO (mAP@50, mAP@50:95, AR@100) của toàn bộ hệ thống Pipeline Cascade trên tập Validation sạch, tuyệt đối không phải là Classification Accuracy trên ảnh đơn lẻ.'
)
add_table_with_caption(
    'Bảng 18. Kết quả nghiên cứu Ablation 2 trên tập Detection Validation sạch (1.743 ảnh, 3.932 GT boxes): Ảnh hưởng của 5 phương pháp tiền xử lý vùng cắt (Crop Variants A–E) trong Pipeline Cascade',
    TABLE_18_CROP_ABLATION,
    col_weights=[1.10, 1.75, 0.80, 0.85, 0.75, 1.15],
    font_size=8.6
)
add_figure_with_caption(
    FIG_DIR / 'fig_crop_preprocessing_ablation.png',
    'Hình 12. Kết quả Ablation 2 trên 1.743 ảnh Validation sạch: So sánh mAP@50, mAP@50:95 và AR@100 giữa 5 phương pháp tiền xử lý vùng cắt (Crop A–E) trong Pipeline Cascade',
    max_width_in=6.10
)
add_p(
    'Kết quả thực nghiệm tại Bảng 18 và Hình 12 cung cấp bằng chứng định lượng trực tiếp kiểm chứng giả thuyết lệch miền vùng cắt (Crop Domain Shift Hypothesis):'
)
add_p(
    'Trong cả 5 biến thể được khảo sát, Phương pháp C (C_context_pad_15 — Context Padding 15% + Direct Resize) đạt hiệu năng cao nhất đồng loạt trên cả ba thước đo chuẩn COCO: Validation mAP@50 đạt 46,69%, Validation mAP@50:95 đạt 26,55% và Validation AR@100 đạt 42,52%. Quy trình chính xác 5 bước của Phương pháp C trong mã nguồn thực thi (dg38_crop_ablation.py) gồm: (1) Từ tọa độ bounding box dự đoán [x1, y1, x2, y2] có chiều rộng w = x2 − x1 và chiều cao h = y2 − y1, mở rộng khung bao thêm 15% về mỗi phía: x1_new = x1 − 0,15·w, y1_new = y1 − 0,15·h, x2_new = x2 + 0,15·w, y2_new = y2 + 0,15·h; (2) Giới hạn (clip) các tọa độ mở rộng nằm gọn trong phạm vi kích thước ảnh gốc [0, W] × [0, H]; (3) Cắt (crop) vùng ảnh theo tọa độ đã mở rộng; (4) Thay đổi kích thước trực tiếp (direct resize bằng nội suy song tuyến T.Resize((224, 224))) về đúng kích thước đầu vào 224 × 224 của DINOv2 ViT-B/14; và (5) Chuẩn hóa giá trị pixel theo mean và std của ImageNet.',
    bold_prefix='1. Mô tả chính xác quy trình của cấu hình tối ưu Crop C (Context Padding 15%): '
)
add_p(
    'So sánh bước nhảy từ Biến thể A (mAP@50 = 43,99%; mAP@50:95 = 24,77%) sang Biến thể B (mAP@50 = 46,19%; mAP@50:95 = 26,32%) và sang Biến thể C (mAP@50 = 46,69%; mAP@50:95 = 26,55%) cho phép bóc tách rõ hai cơ chế gây suy giảm ở Crop A ban đầu: Thứ nhất, việc bỏ phép cắt tâm CenterCrop(224) từ 256 (chuyển từ A sang B) ngay lập tức giúp mAP@50 tăng +2,20 điểm phần trăm và mAP@50:95 tăng +1,55 điểm phần trăm. Điều này chứng minh rằng khi cắt sát bounding box của Detector, phép CenterCrop(224) đã gọt mất phần viền hình dáng quan trọng của vật thể (như miệng chai, vành lon, đường viền mảnh giấy). Thứ hai, việc mở rộng thêm 15% ngữ cảnh xung quanh mỗi cạnh của bounding box trước khi resize trực tiếp (chuyển từ B sang C) giúp tăng thêm +0,50 điểm phần trăm mAP@50 và +0,23 điểm phần trăm mAP@50:95, nâng tổng mức cải thiện của Crop C so với Crop A lên +2,70 điểm phần trăm mAP@50 (+6,14% tương đối) và +1,78 điểm phần trăm mAP@50:95 (+7,19% tương đối). Vùng đệm ngữ cảnh 15% giúp khôi phục tương phản giữa biên vật thể và nền xung quanh, làm cho phân phối hình thái của vùng cắt gần lại với phân phối của các ảnh đơn đối tượng trong tập huấn luyện Garbage Classification v2.',
    bold_prefix='2. Giải thích cơ chế giảm lệch miền vùng cắt (Crop Domain Shift) giữa Crop A, B và C: '
)
add_p(
    'Hai biến thể sử dụng kỹ thuật đệm vuông giữ nguyên tỷ lệ cạnh bằng màu xám trung tính RGB(114, 114, 114) là Biến thể D (Letterbox 224: mAP@50 = 45,47%, mAP@50:95 = 25,95%) và Biến thể E (Context Padding 25% + Letterbox: mAP@50 = 45,75%, mAP@50:95 = 26,10%) đều vượt qua Crop A (+1,48 và +1,76 điểm % mAP@50) nhờ không cắt mất rìa vật thể, nhưng lại thấp hơn Biến thể B (46,19%) và Biến thể C (46,69%). Nguyên nhân là do bộ phân loại DINOv2 ViT-B/14 được huấn luyện trên các ảnh phủ kín khung hình 224 × 224 mà không có dải viền xám nhân tạo; khi đưa các vùng cắt có dải đệm xám lớn (ở các vật thể thon dài như chai nhựa, thanh kim loại) hoặc mở rộng ngữ cảnh quá rộng tới 25% (làm thu nhỏ diện tích vật thể chính và lọt thêm các vật thể rác lân cận vào cùng một vùng cắt), các patch tokens ở vùng biên bị nhiễu, làm giảm độ tự tin phân loại so với phương án mở rộng vừa phải 15% và resize trực tiếp (Crop C).',
    bold_prefix='3. Vì sao các biến thể đệm vuông Letterbox (D và E) kém hiệu quả hơn Resize trực tiếp có ngữ cảnh 15% (C): '
)

add_h2('5.7. Phân tích chuyên sâu nhánh Deformable DETR và Pipeline Cascade trên tập Validation')
add_p(
    'Để cung cấp bức tranh toàn diện và minh bạch về nhánh phương pháp chính của dự án (DINOv2 ViT-S/14 đóng băng + Deformable DETR và Pipeline Cascade hai giai đoạn), mục này trình bày lịch sử huấn luyện của bộ phát hiện cùng toàn bộ kết quả định lượng và phân rã theo lớp trên tập Validation sạch (1.743 ảnh, 3.932 GT boxes).'
)
add_p(
    'Hình 13, Hình 14 và Hình 15 minh họa diễn biến quá trình huấn luyện ban đầu của bộ phát hiện Deformable DETR qua 29 điểm lưu (epochs 1–29 trong tệp detection/checkpoints_wastedetect/training_history.json) trên tập Validation gốc (2.316 ảnh, trước khi kiểm toán khử trùng). Tại epoch 29, checkpoint best_model.pt đạt mAP@50 = 0,7562 và mAP@50:95 = 0,4846 trên tập Validation gốc. Lưu ý tại Hình 14: ở điểm vận hành mặc định trong lịch sử huấn luyện (ngưỡng điểm rất thấp score_threshold = 0,05 và chưa áp dụng NMS), Recall@50 đạt tới 92,37% nhưng Precision@50 chỉ đạt 7,45% do bộ giải mã xuất toàn bộ các truy vấn điểm thấp; chính quan sát lịch sử này đã thúc đẩy nhóm thực hiện bước tìm kiếm lưới trên tập Validation sạch (1.743 ảnh) để khóa ngưỡng Class-Agnostic NMS IoU = 0,70 và ngưỡng điểm vận hành PR threshold = 0,30 (cho Detector) / 0,25 (cho Cascade).'
)
add_figure_with_caption(
    QA_ASSETS / 'figure8.png',
    'Hình 13. Diễn biến chỉ số mAP@50 và mAP@50:95 qua 29 epoch được lưu của bộ phát hiện Deformable DETR trên tập Validation gốc (2.316 ảnh)',
    max_width_in=5.55
)
add_figure_with_caption(
    QA_ASSETS / 'figure9.png',
    'Hình 14. Diễn biến Precision@50, Recall@50 và F1@50 tại ngưỡng điểm lịch sử score = 0,05 (chưa NMS) qua 29 epoch được lưu trên tập Validation gốc',
    max_width_in=5.55
)
add_figure_with_caption(
    QA_ASSETS / 'figure10.png',
    'Hình 15. Diễn biến hàm mất mát huấn luyện (Train Loss) và kiểm định (Validation Loss) qua 29 epoch được lưu trong training_history.json',
    max_width_in=5.55
)
add_p(
    'Trên tập Validation sạch sau kiểm toán khử trùng (1.743 ảnh, 3.932 GT boxes), Bảng 19 và Hình 16 đối chiếu toàn diện 4 cấu hình trong thiết kế 2 × 2 (detector_original, detector_nms, cascade_original, cascade_nms với Crop A) đặt cạnh cấu hình nâng cấp cascade_nms với Crop C (Context Padding 15%). Tiếp đó, Bảng 20 và Hình 17 phân rã chi tiết hiệu năng phát hiện theo từng lớp trong 5 lớp rác (glass, metal, organic, paper, plastic).'
)
add_table_with_caption(
    'Bảng 19. Kết quả định lượng chi tiết của các cấu hình Deformable DETR và Pipeline Cascade trên tập Validation sạch (1.743 ảnh, 3.932 GT boxes, 1 seed tham chiếu)',
    TABLE_19_DEFDETR_CASCADE_VALIDATION,
    col_weights=[1.55, 0.90, 0.90, 0.90, 0.95, 0.95],
    font_size=8.4
)
add_figure_with_caption(
    FIG_DIR / 'fig_defdetr_cascade_val_comparison.png',
    'Hình 16. So sánh tổng thể mAP@50, mAP@50:95 và AR@100 của nhánh Deformable DETR và Pipeline Cascade (Crop A vs. Crop C) trên tập Validation sạch (1.743 ảnh, 3.932 GT boxes)',
    max_width_in=6.05
)
add_table_with_caption(
    'Bảng 20. Hiệu năng phát hiện chi tiết theo từng lớp rác (AP@50 / AP@50:95 / AR@100) trên tập Validation sạch (1.743 ảnh, 3.932 GT boxes) giữa 4 cấu hình Deformable DETR và Cascade',
    TABLE_20_DEFDETR_CASCADE_PER_CLASS_VAL,
    col_weights=[0.95, 0.85, 1.05, 1.05, 1.05, 1.05],
    font_size=8.4
)
add_figure_with_caption(
    FIG_DIR / 'fig_defdetr_cascade_val_per_class.png',
    'Hình 17. Phân rã AP@50 theo từng lớp rác trên tập Validation sạch (1.743 ảnh, 3.932 GT boxes) giữa bộ phát hiện Deformable DETR và Pipeline Cascade',
    max_width_in=6.00
)
add_p(
    'Phân tích Bảng 19 và Bảng 20 làm nổi bật hai kết quả quan trọng trên tập Validation sạch: (1) Việc áp dụng Class-Agnostic NMS (IoU = 0,70) giúp loại bỏ 109.481 / 174.300 truy vấn chồng lấp dư thừa (giảm 62,81% số lượng proposals, từ 100,0 xuống trung bình 37,19 proposals/ảnh), qua đó giảm số dự đoán dương tính giả (FP) của Detector tại ngưỡng PR = 0,30 từ 1.215 xuống 646 (giảm 46,83%), nâng Precision@50 từ 65,92% lên 78,34% (+12,42 điểm %) và nâng Validation mAP@50 của Detector từ 71,45% lên 76,17% (+4,72 điểm %), đồng thời nâng Validation mAP@50 của Cascade (Crop A) từ 40,83% lên 43,99% (+3,16 điểm %); (2) Khi kết hợp thêm phương pháp tiền xử lý vùng cắt tối ưu Crop C (+15% context padding), hiệu năng của Cascade NMS tiếp tục tăng lên mAP@50 = 46,69% và mAP@50:95 = 26,55%.'
)

add_h2('5.8. Phân tích tách biệt nhân tố (Factorial Ablation), Chẩn đoán lỗi 5×10 và Trạng thái đánh giá cuối')
add_p(
    'Vì sao ngay cả khi bộ phân loại DINOv2 ViT-B/14 đạt Macro F1 tới 98,40% trên tập ảnh đơn đối tượng, việc ghép nối vào sau Deformable DETR trong kiến trúc Cascade vẫn có mAP trên Validation thấp hơn việc giữ nguyên nhãn của Detector (46,69% với Crop C hoặc 43,99% với Crop A so với 76,17% của detector_nms)? Để giải đáp thấu đáo hiện tượng này, nhóm trình bày phép phân tích tách biệt nhân tố gồm 6 cấu hình chẩn đoán trên tập Validation sạch tại Bảng 21 và ma trận nhầm lẫn chẩn đoán 5 lớp Ground Truth × 10 lớp DINOv2 trên 3.663 GT boxes định vị thành công (IoU ≥ 0,50) tại Bảng 22, Hình 18 và Hình 19.'
)
add_table_with_caption(
    'Bảng 21. Kết quả phân tích tách biệt nhân tố (Factorial Ablation — 6 cấu hình) của nhánh Cascade trên tập Validation sạch (1.743 ảnh, 3.932 GT boxes, NMS IoU = 0,70)',
    TABLE_21_FACTORIAL_ABLATION,
    col_weights=[1.55, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80],
    font_size=8.4
)
add_p(
    'Ghi chú Bảng 21: Best F1 trong bảng chẩn đoán sử dụng ngưỡng điểm tối ưu riêng của từng cấu hình trên tập Validation (lần lượt là 0,30; 0,25; 0,25; 0,25; 0,25; 0,25); các tỷ lệ phần trăm đóng góp (71,22% và 28,78%) phụ thuộc vào thứ tự phân rã tuần tự từ Cấu hình 1 → Cấu hình 3 → Cấu hình 5.',
    italic=True, size=10.2
)
add_table_with_caption(
    'Bảng 22. Ma trận nhầm lẫn chẩn đoán giữa 5 lớp Ground Truth (cột) và 10 lớp dự đoán DINOv2 (hàng) trên 3.663 GT boxes định vị được (IoU ≥ 0,50) ở tập Validation sạch',
    TABLE_22_CONFUSION_5X10,
    col_weights=[1.35, 0.80, 0.80, 0.80, 0.80, 0.80, 0.90],
    font_size=8.4
)
add_figure_with_caption(
    FIG_DIR / 'image19.png',
    'Hình 18. Phân rã cơ chế suy giảm mAP của kiến trúc Cascade trên tập Validation sạch (1.743 ảnh): ma trận nhầm lẫn chuẩn hóa 5×10 (a) và tác động từng bước của các nhân tố (b)',
    max_width_in=6.10
)
add_figure_with_caption(
    FIG_DIR / 'image20.png',
    'Hình 19. Trực quan hóa định tính 4 trường hợp điển hình (Case A: Đồng thuận đúng; Case B: Lọc OOS đúng; Case C: Nhầm lẫn nội bộ glass → plastic; Case D: Tổn thất do gán nhãn OOS paper → cardboard/trash)',
    max_width_in=6.10
)
add_p(
    'Từ Bảng 21, Bảng 22 và Hình 18–19, cơ chế suy giảm mAP của nhánh Cascade được giải mã định lượng qua ba nguyên nhân trực tiếp: (1) Trong 3.663 GT boxes được Deformable DETR định vị đúng vị trí (IoU ≥ 0,50), bản thân đầu phân loại 5 lớp của Detector (vốn được huấn luyện trực tiếp cùng ngữ cảnh toàn ảnh của WasteDetect) đã gán đúng nhãn cho 3.618/3.663 vật thể (98,77%); trong khi bộ phân loại DINOv2 ViT-B/14 (huấn luyện trên miền ảnh đơn đối tượng Kaggle) khi nhận vùng cắt Crop A chỉ khớp nhãn 5 lớp trên 2.536/3.663 vật thể (69,23%); (2) Ở nhóm nhầm lẫn nội bộ giữa 5 lớp (chiếm 71,22% độ sụt giảm mAP trước bước nhân điểm theo thứ tự phân rã 1 → 3 → 5), hiện tượng nghiêm trọng nhất xảy ra tại lớp glass: trong 311 GT boxes lớp glass định vị được, DINOv2 chỉ giữ đúng 121 box (38,91%) nhưng gán nhầm tới 168 box (54,02%) sang lớp plastic do đặc tính trong suốt và phản quang của mảnh thủy tinh trong vùng cắt hẹp rất giống chai nhựa; và (3) Ở nhóm loại bỏ ngoài phạm vi OOS (chiếm 28,78% độ sụt giảm còn lại), có tổng cộng 436/3.663 GT boxes (11,90%) bị DINOv2 gán vào 5 lớp ngoài phạm vi nên bị loại bỏ theo cơ chế drop_and_count, tập trung nặng nhất ở lớp paper khi 136/457 box (29,76%) bị gán thành cardboard và 48/457 box (10,50%) bị gán thành trash (tổng cộng 188/457 box paper = 41,14% bị mất do rơi vào nhóm OOS).'
)
add_p(
    'Để khép lại Chương 5 một cách hoàn toàn minh bạch, Bảng 23 tổng hợp trạng thái đánh giá cuối cùng (Validation vs. Final Test) của toàn bộ các mô hình và nhánh thực nghiệm trong đồ án. Bảng 23 khẳng định rõ: trong khi ResNet50 (3 seeds), DINOv2 ViT-B/14 (3 seeds), YOLO11s (3 seeds) và RT-DETR-L (1 seed) đã hoàn tất đánh giá trên gói Final Test khóa tại Notebook 06, nhánh Deformable DETR và Pipeline Cascade (đặc biệt là cấu hình tối ưu Crop C được chọn từ Notebook 04) chưa thực hiện lượt chạy Final Test mới trong gói Notebook chốt. Báo cáo tuyệt đối không sử dụng lại các bảng chạy thử nghiệm cục bộ trước đây (vốn chỉ dùng Crop A trên 1 seed trước khi khóa quy trình Colab) để gán làm kết quả Final Test của cấu hình hoàn thiện.'
)
add_table_with_caption(
    'Bảng 23. Tổng hợp trạng thái đánh giá cuối cùng (Validation vs. Final Test) của toàn bộ các mô hình và nhánh thực nghiệm trong đồ án DG38',
    TABLE_23_EVALUATION_STATUS_SUMMARY,
    col_weights=[1.15, 0.85, 0.75, 1.25, 1.25, 1.35],
    font_size=8.4
)

# ==============================================================================
# CHƯƠNG 6. THẢO LUẬN VÀ GIỚI HẠN NGHIÊN CỨU
# ==============================================================================
add_h1('6. Thảo luận và Giới hạn Nghiên cứu', page_break=True)
add_h2('6.1. Thảo luận tổng hợp 8 luận điểm khoa học từ kết quả thực nghiệm')
add_p(
    'Dựa trên toàn bộ bằng chứng định lượng đã trình bày tại Chương 5, nhóm đi sâu thảo luận 8 vấn đề cốt lõi về mặt học thuật và kỹ thuật hệ thống:'
)
add_p(
    'Trên cả tập Validation và Final Test (qua 3 hạt giống 42, 123, 2026), DINOv2 ViT-B/14 vượt qua ResNet50 một cách nhất quán (+2,39 điểm % Accuracy và +2,91 điểm % Macro F1 trên Final Test). Sự vượt trội này xuất phát từ bản chất biểu diễn của mô hình nền tảng tự giám sát (Self-supervised Vision Foundation Model): khác với ResNet50 học các bộ lọc tích chập bị ràng buộc bởi nhiệm vụ phân biệt 1.000 lớp cụ thể của ImageNet-1K, DINOv2 được huấn luyện bằng cơ chế tự chưng cất không nhãn trên 142 triệu ảnh (LVD-142M) với hàm mục tiêu kết hợp mức ảnh (DINO) và mức mảng bị che (iBOT). Nhờ đó, các vector nhúng 768 chiều của DINOv2 mã hóa đồng thời cấu trúc hình học toàn cục (nhờ trường chú ý toàn ảnh của ViT) và đặc tính vật lý tinh vi của bề mặt vật liệu (nhờ mục tiêu dự đoán patch ẩn). Khi kết hợp với chiến lược mở khóa 4 khối Transformer cuối cùng (Mục 5.5), mô hình thích nghi nhanh chóng với 10 lớp rác thải mà không bị suy thoái đặc trưng.',
    bold_prefix='1. Vì sao DINOv2 ViT-B/14 vượt trội hơn ResNet50 trên bài toán Classification: '
)
add_p(
    'Trên tập dữ liệu phát hiện WasteDetect (8.289 ảnh Train), YOLO11s đạt mAP@50:95 trên Final Test là 60,37% ± 0,42 (mAP@50 = 82,78% ± 0,40), vượt khá xa RT-DETR-L một hạt giống (mAP@50:95 = 48,57%; mAP@50 = 70,13%). Nguyên nhân cốt lõi nằm ở định kiến quy nạp (inductive bias) và đặc tính hội tụ trên tập dữ liệu quy mô trung bình: kiến trúc CNN một giai đoạn của YOLO11s với tháp đặc trưng đa tỷ lệ P3/P4/P5, cơ chế gán nhãn động Task-Aligned Assigner và hàm mất mát DFL + CIoU có khả năng định vị biên vật thể cực kỳ sắc nét với số lượng epoch huấn luyện vừa phải (30 epochs). Ngược lại, mặc dù RT-DETR-L đạt độ nhạy AR@100 ngang ngửa YOLO11s (69,55% so với 69,38%), bộ giải mã truy vấn Transformer và ghép cặp Hungarian một-đối-một đòi hỏi lịch huấn luyện dài hơn hoặc tinh chỉnh sâu tốc độ học để triệt tiêu các khung bao dự đoán dư thừa xung quanh các mẫu rác có hình dạng bất quy tắc.',
    bold_prefix='2. Vì sao YOLO11s vượt trội hơn RT-DETR-L trên bài toán Detection hiện tại: '
)
add_p(
    'Trong báo cáo này, ResNet50, DINOv2 ViT-B/14 và YOLO11s đã được huấn luyện và đánh giá đủ 3 hạt giống ngẫu nhiên (42, 123, 2026), cho phép định lượng độ biến thiên qua Sample Standard Deviation. Tuy nhiên, do giới hạn về ngân sách tính toán (100 Compute Units trên Google Colab Pro) và thời gian huấn luyện mỗi chu kỳ của các mô hình Transformer phát hiện đối tượng quy mô lớn, RT-DETR-L và nhánh Deformable DETR chỉ mới được thực hiện với 1 hạt giống ngẫu nhiên (seed = 42). Việc chỉ có 1 seed đồng nghĩa với việc chúng ta chưa thể ước lượng phương sai do khởi tạo ngẫu nhiên và thứ tự nạp dữ liệu của hai mô hình này; do đó, mọi phép so sánh liên quan đến RT-DETR-L và Deformable DETR đều được diễn giải thận trọng như kết quả của một lượt chạy đại diện, tuyệt đối không gán độ lệch chuẩn giả định.',
    bold_prefix='3. Hạn chế của RT-DETR-L và Deformable DETR khi chỉ có 1 hạt giống ngẫu nhiên (1 Seed): '
)
add_p(
    'Một nguyên tắc phương pháp luận bắt buộc trong báo cáo là tuyệt đối không đặt chỉ số Accuracy (98,61%) hay Macro F1 (98,40%) của bài toán Classification cạnh chỉ số mAP@50 (82,78%) hay mAP@50:95 (60,37%) của bài toán Detection để kết luận rằng "mô hình phân loại tốt hơn mô hình phát hiện". Hai bài toán này giải quyết hai cấp độ nhận thức thị giác hoàn toàn khác nhau trên hai bộ dữ liệu khác nhau: Classification chỉ yêu cầu dự đoán 1 nhãn duy nhất cho toàn bộ bức ảnh đã chứa sẵn một vật thể ở trung tâm (10 lớp), trong khi Detection đòi hỏi mô hình phải tự tìm kiếm vị trí không gian của nhiều vật thể trên nền phức tạp, hồi quy chính xác 4 tọa độ khung bao vượt qua chuỗi ngưỡng giao trên hợp khắt khe từ IoU = 0,50 đến IoU = 0,95 và đồng thời phân loại đúng nhãn (5 lớp), lại bị phạt nặng bởi mọi dự đoán dương tính giả (FP) trên nền ảnh.',
    bold_prefix='4. Vì sao tuyệt đối không so sánh trực tiếp Classification Accuracy/Macro F1 với Detection mAP: '
)
add_p(
    'Nghiên cứu Ablation 2 (Mục 5.6) trên 1.743 ảnh Validation sạch đã chứng minh thực nghiệm rằng giả thuyết lệch miền vùng cắt (Crop Domain Shift) là hoàn toàn có thật: chỉ bằng cách thay đổi tiền xử lý vùng cắt từ Crop A (cắt sát bbox + Resize 256 + CenterCrop 224) sang Crop C (mở rộng bbox 15% mỗi phía + giới hạn biên ảnh + Resize trực tiếp 224 × 224) mà không cần huấn luyện lại bất kỳ trọng số nào, Validation mAP@50 của Cascade tăng thêm +2,70 điểm phần trăm (43,99% → 46,69%) và mAP@50:95 tăng +1,78 điểm phần trăm (24,77% → 26,55%). Tuy nhiên, nhóm tuân thủ nghiêm ngặt ranh giới kết luận: cải thiện này hiện mới được chứng minh trên tập Validation sạch (nơi dùng để chọn ra cấu hình Crop C); do chưa thực thi lượt chạy Final Test độc lập cho Cascade với Crop C, báo cáo chỉ kết luận về hiệu quả của Crop C trên phạm vi Validation.',
    bold_prefix='5. Ý nghĩa của kết quả Crop C đối với kiến trúc Cascade và phạm vi hiệu lực trên Validation: '
)
add_p(
    'Bên cạnh hiện tượng lệch miền vùng cắt hình học, phân tích ma trận chẩn đoán 5 × 10 tại Mục 5.8 chỉ ra nút thắt cấu trúc thứ hai của hệ thống Cascade nằm ở sự bất tương thích hệ thống nhãn (Taxonomy Mismatch) giữa bộ phân loại 10 lớp và bộ phát hiện 5 lớp. Trong bộ dữ liệu Garbage Classification v2, giấy báo/giấy viết (paper) và thùng bìa cứng (cardboard) được tách thành hai lớp riêng biệt; trong khi ở bộ dữ liệu WasteDetect, nhiều mẫu bìa giấy hoặc hộp giấy lại được gộp chung vào nhãn paper (ID: 3) và không có lớp cardboard. Hệ quả là khi DINOv2 phân loại hoàn toàn hợp lý về mặt thị giác một mảnh hộp giấy thành cardboard (136/457 GT boxes của lớp paper trên Validation), cơ chế lọc ngoài phạm vi (OOS drop_and_count) lại loại bỏ dự đoán đó, gây sụt giảm trực tiếp Recall và mAP của lớp paper.',
    bold_prefix='6. Ảnh hưởng của sự bất tương thích nhãn (Taxonomy Mismatch) giữa Classification 10 lớp và Detection 5 lớp: '
)
add_p(
    'Mặc dù nhóm đã thực hiện quy trình kiểm toán khử trùng lặp đa tầng rất nghiêm ngặt ở cấp độ ảnh (loại bỏ 573 ảnh trùng trên Validation và 324 ảnh trùng trên Test bằng SHA-256 và pHash ≤ 6), các bộ dữ liệu công khai thu thập từ Kaggle và Roboflow vẫn thiếu siêu dữ liệu về phiên chụp, thiết bị camera và địa điểm thu thập. Do đó, kiểm toán mức ảnh không thể loại trừ hoàn toàn trường hợp hai bức ảnh khác nhau được chụp từ cùng một vật thể ở các góc độ rất khác nhau. Ngoài ra, hiệu năng cao trên tập dữ liệu hiện tại chưa mặc nhiên đảm bảo khả năng tổng quát hóa khi triển khai ngoài thực địa dưới điều kiện ánh sáng yếu, bóng đổ gắt, rác bị bùn đất che phủ nặng hoặc chuyển động nhanh trên băng chuyền.',
    bold_prefix='7. Hạn chế về dữ liệu, tính tổng quát hóa (Generalization) và hạ tầng phần cứng: '
)
add_p(
    'Như đã tổng hợp tại Bảng 23, trong đợt thực nghiệm bổ sung trên Google Colab (Notebooks 00–06), nhóm đã ưu tiên hoàn thành trọn vẹn toàn bộ các baseline độc lập đa hạt giống (ResNet50, DINOv2 ViT-B/14, YOLO11s, RT-DETR-L) và hai nghiên cứu Ablation theo đúng yêu cầu của giảng viên, nhưng chưa kịp thực thi kịch bản đánh giá Final Test mới trên gói Test khóa cho riêng nhánh Deformable DETR và Pipeline Cascade (với cấu hình tối ưu Crop C). Việc công bố minh bạch trạng thái chưa hoàn thành Final Test của nhánh này — thay vì lấy số liệu Validation hay số liệu chạy thử Crop A cũ để gán ghép — là cam kết cao nhất của nhóm về đạo đức và tính liêm chính trong nghiên cứu khoa học.',
    bold_prefix='8. Công bố trung thực hạn chế về việc chưa hoàn thành Final Test mới cho Deformable DETR và Cascade: '
)

add_h2('6.2. Giá trị và phạm vi của việc đối chiếu với bài báo tham chiếu (Islam và cs., 2025)')
add_p(
    'Việc đối chiếu mô hình phân loại DINOv2 ViT-B/14 của nhóm với công trình của Islam và cs. (PLOS ONE, 2025) [1] mang lại giá trị định hướng quan trọng: nó giúp đặt kết quả của đồ án vào mạch nghiên cứu quốc tế đương đại trên cùng họ dữ liệu Waste Classification V2. Tuy nhiên, báo cáo luôn giữ thái độ khách quan khi chỉ ra rằng sự khác biệt về tổng số ảnh (12.259 ảnh so với 21.983 ảnh), tỷ lệ chia tập kiểm thử (10% so với 20%) và phương thức tiền xử lý khiến mọi độ chênh lệch số học chỉ mang tính chất tham khảo định tính và định lượng tương đối, trong khi phép so sánh nội bộ giữa DINOv2 ViT-B/14 và ResNet50 trên cùng manifest khóa qua 3 seeds (Mục 5.1) mới là bằng chứng đối chứng trực tiếp và chặt chẽ nhất.'
)

add_h2('6.3. Tổng hợp các giới hạn khoa học và kỹ thuật cần lưu ý khi sử dụng kết quả')
add_p(
    'Tóm lược lại, người đọc và hội đồng đánh giá cần lưu ý năm giới hạn cụ thể khi tham chiếu các kết quả của báo cáo này: (i) Chỉ có ResNet50, DINOv2 ViT-B/14 và YOLO11s có đủ 3 random seeds (42, 123, 2026); RT-DETR-L và nhánh Deformable DETR/Cascade chỉ có 1 seed; (ii) Mô-đun MSDeformAttn của nhánh Deformable DETR trong dự án là bản triển khai pure PyTorch sử dụng F.grid_sample, không phải nhân CUDA C++ chính thức; (iii) Kết quả Ablation số khối fine-tune (0, 2, 4 blocks) và Ablation tiền xử lý vùng cắt (Crop A–E) là kết quả trên tập Validation, không phải kết quả Final Test; (iv) Nhánh Deformable DETR và Cascade (với cấu hình tối ưu Crop C) chưa có kết quả Final Test mới trong gói Notebook 06; và (v) Các phép đo độ trễ suy luận trước đây được thực hiện trên GPU Laptop (RTX 3050) và GPU đám mây (NVIDIA L4), chưa bao gồm thử nghiệm thực tế với luồng camera thời gian thực trên thiết bị nhúng biên (Edge AI).'
)

# ==============================================================================
# CHƯƠNG 7. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN
# ==============================================================================
add_h1('7. Kết luận và Hướng phát triển', page_break=True)
add_h2('7.1. Kết luận')
add_p(
    'Đồ án môn học Thị giác máy tính (Mã đề tài DG38) đã hoàn thành một quy trình nghiên cứu thực nghiệm có kiểm soát, minh bạch và khả tái lập cho cả hai bài toán phân loại ảnh rác thải đơn đối tượng (10 lớp) và phát hiện rác thải đa đối tượng (5 lớp), đồng thời giải quyết thỏa đáng các yêu cầu bổ sung của giảng viên hướng dẫn. Những kết quả chính đã được kiểm chứng thực nghiệm bao gồm:'
)
add_p(
    'Đã hoàn tất kiểm toán khử trùng lặp đa tầng bằng mã băm mật mã SHA-256 và mã băm cảm nhận thị giác pHash (khoảng cách Hamming ≤ 6), loại bỏ 573 ảnh trùng trên tập Detection Validation (còn 1.743 ảnh sạch / 3.932 GT boxes) và 324 ảnh trùng trên tập Detection Final Test (còn 871 ảnh khóa / 1.884 GT boxes), đồng thời khóa manifest phân tầng của tập Classification Final Test (1.226 ảnh / 10 lớp). Mọi quyết định chọn mô hình, chọn số khối fine-tune, chọn ngưỡng NMS IoU (0,70) và chọn phương pháp tiền xử lý vùng cắt (Crop C) đều được thực hiện nghiêm ngặt trên tập Validation trước khi mở gói Final Test.',
    bold_prefix='1. Về chuẩn hóa dữ liệu và giao thức chống rò rỉ (Data Hygiene & Protocol Integrity): '
)
add_p(
    'Trên tập Classification Validation (1.226 ảnh), DINOv2 ViT-B/14 (fine-tune 4 blocks cuối) đạt Accuracy 97,63% ± 0,22 và Macro F1 97,33% ± 0,20, vượt qua baseline ResNet50 (Accuracy 96,25% ± 0,08; Macro F1 95,51% ± 0,21). Trên tập Classification Final Test đã khóa (1.226 ảnh), DINOv2 ViT-B/14 đạt Accuracy 98,61% ± 0,29 và Macro F1 98,40% ± 0,30, vượt qua ResNet50 (Accuracy 96,22% ± 0,29; Macro F1 95,50% ± 0,27) lần lượt +2,39 điểm phần trăm về Accuracy và +2,91 điểm phần trăm về Macro F1 với độ lệch chuẩn mẫu rất nhỏ (≤ 0,30 điểm %).',
    bold_prefix='2. Về đối chứng độc lập bài toán Classification (10 lớp, 3 random seeds 42, 123, 2026): '
)
add_p(
    'Trên tập Detection Validation sạch (1.743 ảnh, 3.932 GT boxes), YOLO11s (3 seeds) đạt mAP@50 = 84,94% ± 0,09 và mAP@50:95 = 61,45% ± 0,16; RT-DETR-L (1 seed) đạt mAP@50 = 71,20% và mAP@50:95 = 49,43%; trong khi cấu hình Detector tốt nhất của nhánh Deformable DETR (detector_nms, 1 seed) đạt mAP@50 = 76,17% và mAP@50:95 = 44,45%. Trên tập Detection Final Test đã khóa (871 ảnh, 1.884 GT boxes), YOLO11s (3 seeds) đạt mAP@50 = 82,78% ± 0,40, mAP@50:95 = 60,37% ± 0,42 và AR@100 = 69,38% ± 0,21, khẳng định vị trí dẫn đầu trong các mô hình phát hiện được khảo sát; RT-DETR-L (1 seed) đạt mAP@50 = 70,13%, mAP@50:95 = 48,57% và AR@100 = 69,55%.',
    bold_prefix='3. Về đối chứng độc lập bài toán Detection (5 lớp WasteDetect): '
)
add_p(
    'Nghiên cứu Ablation 1 trên tập Classification Validation (1.226 ảnh) chỉ ra rằng mở khóa 4 khối Transformer cuối của DINOv2 ViT-B/14 (32,75% tham số) mang lại Validation Macro F1 cao nhất (97,33%) so với 2 khối (96,99%) và 0 khối — Linear Probe (96,46%). Nghiên cứu Ablation 2 trên tập Detection Validation sạch (1.743 ảnh) chứng minh rằng phương pháp tiền xử lý vùng cắt Crop C (mở rộng bounding box 15% mỗi phía, giới hạn trong biên ảnh gốc, cắt ảnh và resize trực tiếp về 224 × 224) giúp nâng Validation mAP@50 của Pipeline Cascade từ 43,99% (Crop A) lên 46,69% (+2,70 điểm %) và mAP@50:95 từ 24,77% lên 26,55% (+1,78 điểm %). Tuy nhiên, báo cáo cũng ghi nhận trung thực rằng nhánh Deformable DETR và Pipeline Cascade (với Crop C) chưa thực hiện lượt chạy Final Test mới trên bộ Test đã khóa.',
    bold_prefix='4. Về hai nghiên cứu Ablation và chẩn đoán hệ thống hai giai đoạn trên Validation: '
)

add_h2('7.2. Hướng phát triển tiếp theo')
add_p(
    'Từ những kết quả và giới hạn đã được phân tích minh bạch, nhóm đề xuất bốn hướng phát triển cụ thể và khả thi cho giai đoạn nghiên cứu tiếp theo:'
)
add_p(
    'Thực thi kịch bản đánh giá một lần trên gói Detection Final Test đã khóa (dg38_det_test_locked.zip, 871 ảnh, 1.884 GT boxes) đối với bộ phát hiện Deformable DETR (detector_original, detector_nms) và Pipeline Cascade sử dụng cấu hình tiền xử lý vùng cắt tối ưu Crop C (đã khóa từ Validation tại Notebook 04) nhằm hoàn tất bảng đối chiếu đầu-cuối trên tập Test.',
    bold_prefix='1. Hoàn tất đánh giá Final Test đã khóa cho nhánh Deformable DETR và Pipeline Cascade (Crop C): '
)
add_p(
    'Huấn luyện bổ sung thêm 2 hạt giống ngẫu nhiên (seeds 123 và 2026) cho RT-DETR-L và nhánh Deformable DETR để toàn bộ các mô hình phát hiện đều có đủ bộ ba hạt giống (42, 123, 2026), đồng thời biên dịch nhân mở rộng C++/CUDA chính thức cho mô-đun MultiScaleDeformableAttention để tăng tốc độ huấn luyện và suy luận.',
    bold_prefix='2. Mở rộng đủ 3 hạt giống ngẫu nhiên cho các mô hình Transformer Detection và tối ưu hóa nhân CUDA: '
)
add_p(
    'Thay vì cho phép bộ phân loại DINOv2 ghi đè vô điều kiện lên mọi đề xuất của Detector và loại bỏ cứng toàn bộ 5 lớp OOS, thiết kế cơ chế kết hợp có cổng kiểm soát (Gated Score Fusion) dựa trên độ tin cậy tương đối giữa Detector và Classifier, đồng thời chuẩn hóa lại quy ước gộp nhãn giấy/bìa cứng (paper/cardboard) ngay từ khâu xây dựng tập dữ liệu để triệt tiêu tổn thất 11,90% do lệch hệ thống nhãn.',
    bold_prefix='3. Thiết kế cơ chế Gating chọn lọc có điều kiện và đồng nhất hóa hệ thống nhãn (Taxonomy Alignment): '
)
add_p(
    'Khai thác ưu thế độ chính xác và tốc độ của baseline dẫn đầu YOLO11s (mAP@50 = 82,78%) hoặc ghép nối YOLO11s làm bộ sinh đề xuất tốc độ cao phía trước DINOv2 (với Crop C), sau đó xuất mô hình sang định dạng TensorRT / ONNX FP16 để kiểm thử thực nghiệm với camera thời gian thực trên thiết bị nhúng biên (NVIDIA Jetson Orin Nano).',
    bold_prefix='4. Tối ưu hóa mô hình cho thiết bị nhúng biên (Edge AI Deployment): '
)

# ==============================================================================
# TÀI LIỆU THAM KHẢO
# ==============================================================================
add_h1('Tài liệu tham khảo', page_break=True)
for ref_str in REFERENCES_LIST:
    add_p(ref_str, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_after=5, line_spacing=1.12, size=11.0)

# ==============================================================================
# PHỤ LỤC A. MINH CHỨNG MÃ NGUỒN, KIẾN TRÚC VÀ DANH SÁCH NOTEBOOKS 00–06
# ==============================================================================
add_h1('Phụ lục A. Minh chứng mã nguồn, kiến trúc mô hình và danh sách Notebooks 00–06', page_break=True)
add_p(
    'Phụ lục A cung cấp minh chứng mã nguồn thực tế của bộ phân loại DINOv2 ViT-B/14 (tệp src/model.py, cấu hình mở khóa 4 khối Transformer cuối trong notebook huấn luyện và kích thước tensor xác minh trực tiếp từ tệp checkpoint checkpoints/dinov2_garbage_final.pth tại Hình A1), cùng danh mục đầy đủ 7 Google Colab Notebooks (00–06) và các module mã nguồn Python phục vụ gói thực nghiệm bổ sung DG38 tại Bảng A1.'
)
add_figure_with_caption(
    QA_ASSETS / 'figureA1.png',
    'Hình A1. Mã nguồn bộ phân loại DINOv2 ViT-B/14 hiện tại (src/model.py), cấu hình fine-tune 4 block cuối trong notebook và đối chiếu kích thước tensor trong checkpoint cuối',
    max_width_in=6.10
)
add_table_with_caption(
    'Bảng A1. Danh mục các Google Colab Notebooks (00–06) và các mô-đun mã nguồn thực nghiệm của dự án DG38',
    TABLE_A1_NOTEBOOKS_AND_CODE,
    col_weights=[1.25, 1.35, 2.00],
    font_size=8.6
)

# ==============================================================================
# PHỤ LỤC B. HỒ SƠ TÁI LẬP, MÃ BĂM SHA-256 VÀ TRẠNG THÁI ĐỒNG BỘ DỮ LIỆU
# ==============================================================================
add_h1('Phụ lục B. Hồ sơ tái lập, mã băm SHA-256 và trạng thái đồng bộ dữ liệu Google Drive – Windows', page_break=True)
add_p(
    'Tuân thủ nghiêm ngặt yêu cầu về tính minh bạch truy xuất nguồn gốc dữ liệu, Phụ lục B liệt kê chi tiết đường dẫn, mã băm mật mã SHA-256 và trạng thái xác minh thực tế của từng gói dữ liệu, manifest, checkpoint và thư mục kết quả (Bảng B1). Đặc biệt, báo cáo phân biệt rõ ràng giữa: (1) Những tệp đã có mặt và được xác minh mã băm trực tiếp trên máy cục bộ Windows (bao gồm toàn bộ mã nguồn dự án, hai gói ZIP Final Test đã khóa tại D:\\Download Acer\\DG38_test_export\\, các tệp nhãn COCO JSON sau làm sạch, các báo cáo kiểm toán Validation tại evaluation_runs/val_grid_search_01/, Notebook 06 và tài liệu tổng kết thực nghiệm DG38_Bao_Cao_Tong_Ket_Thuc_Nghiem_FINAL.docx); và (2) Những thư mục kết quả chi tiết và 10 tệp trọng số checkpoint mới sinh ra trên Google Drive (DG38_Colab_Package/results/... và artifacts/runs/...) vốn chưa được tải đồng bộ nguyên thư mục về ổ cứng cục bộ Windows — đối với nhóm tệp này, báo cáo sử dụng số liệu đã xuất trong Notebook 06, báo cáo tổng kết thực nghiệm và dữ liệu tham chiếu chính thức của nhóm, tuyệt đối không tự nhận đã mở đọc trực tiếp các tệp từ xa chưa đồng bộ.'
)
add_table_with_caption(
    'Bảng B1. Hồ sơ truy xuất nguồn gốc tệp dữ liệu, mã băm SHA-256 đã xác minh và trạng thái đồng bộ giữa máy cục bộ Windows và Google Drive (DG38_Colab_Package)',
    TABLE_B1_PROVENANCE_AND_HASHES,
    col_weights=[1.15, 1.35, 1.30, 1.30],
    font_size=8.4
)

# ==============================================================================
# PHỤ LỤC C. MA TRẬN ĐỐI CHIẾU HOÀN THÀNH GÓP Ý CỦA GIẢNG VIÊN
# ==============================================================================
add_h1('Phụ lục C. Bảng đối chiếu chi tiết việc hoàn thành góp ý của giảng viên (Teacher Feedback Completion Matrix)', page_break=True)
add_p(
    'Bảng C1 tổng hợp đối chiếu trực tiếp từng yêu cầu chỉnh sửa và bổ sung của giảng viên hướng dẫn (Hồ sơ DG38) với giải pháp kỹ thuật đã thực hiện, vị trí bằng chứng cụ thể trong báo cáo (chương, mục, bảng, hình, notebook) và đánh giá trung thực mức độ hoàn thành của từng hạng mục.'
)
add_table_with_caption(
    'Bảng C1. Ma trận đối chiếu chi tiết việc hoàn thành các yêu cầu chỉnh sửa và bổ sung của giảng viên (Teacher Feedback Completion Matrix)',
    TABLE_C1_TEACHER_FEEDBACK_MATRIX,
    col_weights=[0.40, 1.25, 1.55, 1.15, 1.05],
    font_size=8.4
)

# ==============================================================================
# FINAL SECTION & FOOTER PAGINATION SETUP (Cover empty, Front Roman, Body Arabic)
# ==============================================================================
main_sectPr = make_sectPr('decimal', 1)
body.replace(body.sectPr, main_sectPr)

for i, s in enumerate(doc.sections):
    s.footer.is_linked_to_previous = False
    s.different_first_page_header_footer = False
    fp = s.footer.paragraphs[0]
    fp.clear()
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fp.paragraph_format.space_before = Pt(0)
    fp.paragraph_format.space_after = Pt(0)
    if i > 0:
        rr = fp.add_run()
        set_run_font(rr, size=11)
        a = OxmlElement('w:fldChar')
        a.set(qn('w:fldCharType'), 'begin')
        rr._r.append(a)
        a = OxmlElement('w:instrText')
        a.set(qn('xml:space'), 'preserve')
        a.text = ' PAGE '
        rr._r.append(a)
        a = OxmlElement('w:fldChar')
        a.set(qn('w:fldCharType'), 'separate')
        rr._r.append(a)
        a = OxmlElement('w:t')
        a.text = 'i' if i == 1 else '1'
        rr._r.append(a)
        a = OxmlElement('w:fldChar')
        a.set(qn('w:fldCharType'), 'end')
        rr._r.append(a)

for name in ['TOC 1', 'TOC 2']:
    if name in doc.styles:
        st = doc.styles[name]
        st.font.name = 'Times New Roman'
        st.font.size = Pt(11)
        st.paragraph_format.space_after = Pt(3)
        st.paragraph_format.line_spacing = 1.05

# Clean up any unused image relationships from the base template
used_rids = {x.get(qn('r:embed')) for x in doc._element.xpath('.//a:blip')}
for rid, rel in list(doc.part.rels.items()):
    if rel.reltype.endswith('/image') and rid not in used_rids:
        del doc.part.rels[rid]

# Ensure no unrequested field refresh prompt pops up after LibreOffice materializes TOC
for up in list(doc.settings.element.findall(qn('w:updateFields'))):
    doc.settings.element.remove(up)

doc.save(str(BUILD_INPUT_DOCX))
print(f'Successfully built {BUILD_INPUT_DOCX}')
print(f'Sections: {len(doc.sections)}, Paragraphs: {len(doc.paragraphs)}, Tables: {len(doc.tables)}, InlineShapes: {len(doc.inline_shapes)}')
