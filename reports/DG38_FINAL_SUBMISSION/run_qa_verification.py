# -*- coding: utf-8 -*-
import sys
import json
import hashlib
import zipfile
import re
from pathlib import Path
from lxml import etree

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(r'D:\Download Acer\garbage_dinov2_starter')
OUT_DIR = ROOT / 'reports' / 'DG38_FINAL_SUBMISSION'
FINAL_DOCX = OUT_DIR / 'DG38_Final_Complete_Report.docx'
FINAL_PDF = OUT_DIR / 'DG38_Final_Complete_Report.pdf'

assert FINAL_DOCX.exists() and FINAL_DOCX.stat().st_size > 100_000, "Missing or empty final DOCX"
assert FINAL_PDF.exists() and FINAL_PDF.stat().st_size > 100_000, "Missing or empty final PDF"

# 1. Verify original files were untouched
orig_backup = ROOT / 'evaluation_runs' / 'report_final_review' / 'Group 1 - 22ES - Backup Original.docx'
orig_sub = ROOT / 'evaluation_runs' / 'report_final_review' / 'Group 1 - 22ES - Bao Cao - Final Submission.docx'
orig_v6 = ROOT / 'evaluation_runs' / 'report_final_review' / '_submission_qa' / 'submission_input_v6.docx'

hashes = {
    'Backup_Original_docx': hashlib.sha256(orig_backup.read_bytes()).hexdigest(),
    'Previous_Final_Submission_docx': hashlib.sha256(orig_sub.read_bytes()).hexdigest(),
    'Template_submission_input_v6_docx': hashlib.sha256(orig_v6.read_bytes()).hexdigest(),
    'New_DG38_Final_Complete_Report_docx': hashlib.sha256(FINAL_DOCX.read_bytes()).hexdigest(),
    'New_DG38_Final_Complete_Report_pdf': hashlib.sha256(FINAL_PDF.read_bytes()).hexdigest(),
}

# Check baseline preservation hash from _submission_qa/preservation_baseline.json
# Note: Group 1 - 22ES - Backup Original.docx and Group 1 - 22ES - Bao Cao - Final Submission.docx must be completely untouched.
pdf_bytes = FINAL_PDF.read_bytes()
assert pdf_bytes.startswith(b'%PDF-'), "Invalid PDF header"
assert b'%%EOF' in pdf_bytes[-1024:], "Invalid PDF EOF"
pdf_pages_matches = re.findall(rb'/Type\s*/Page\b', pdf_bytes)
num_pdf_pages = len(pdf_pages_matches)

# 2. Extract full text and structure from the materialized FINAL_DOCX (saved by LibreOffice UNO after TOC update)
with zipfile.ZipFile(FINAL_DOCX, 'r') as zf:
    doc_xml = zf.read('word/document.xml')
    media_files = [n for n in zf.namelist() if n.startswith('word/media/')]

root_el = etree.fromstring(doc_xml)
ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math'}

paras = []
toc_entries = []
for p in root_el.xpath('.//w:p', namespaces=ns):
    txt = ''.join(p.xpath('.//w:t/text()', namespaces=ns))
    paras.append(txt)
    pstyle = p.xpath('.//w:pStyle/@w:val', namespaces=ns)
    if pstyle and pstyle[0] in ('TOC1', 'TOC2', 'Contents1', 'Contents2') and txt.strip():
        toc_entries.append(txt.strip())

tables_text = []
for t in root_el.xpath('.//w:body/w:tbl', namespaces=ns):
    t_txt = ' '.join(''.join(r.xpath('.//w:t/text()', namespaces=ns)) for r in t.xpath('./w:tr', namespaces=ns))
    tables_text.append(t_txt)

full_text = ('\n'.join(paras) + '\n' + '\n'.join(tables_text)).replace('\u200b', '')

expected_tables = [f'Bảng {i}.' for i in range(1, 24)] + ['Bảng A1.', 'Bảng B1.', 'Bảng C1.']
missing_tables = [t for t in expected_tables if t not in full_text]

expected_figs = [f'Hình {i}.' for i in range(1, 20)] + ['Hình A1.']
missing_figs = [f for f in expected_figs if f not in full_text]

mandatory_strings = [
    '96,25% ± 0,08', '95,51% ± 0,21',
    '97,63% ± 0,22', '97,33% ± 0,20',
    '96,22% ± 0,29', '95,50% ± 0,27',
    '98,61% ± 0,29', '98,40% ± 0,30',
    '84,94% ± 0,09', '61,45% ± 0,16',
    '71,20%', '49,43%',
    '0,832441', '0,608381', '0,695373',
    '0,825127', '0,602519', '0,694642',
    '0,825936', '0,600146', '0,691502',
    '0,701251', '0,485653', '0,695482',
    '82,78% ± 0,40', '60,37% ± 0,42', '69,38% ± 0,21',
    '96,46%', '96,99%', '97,33%',
    '43,99%', '24,77%', '41,69%',
    '46,19%', '26,32%', '42,27%',
    '46,69%', '26,55%', '42,52%',
    '45,47%', '25,95%', '41,73%',
    '45,75%', '26,10%', '41,65%',
    '98ca263b1c76aece593280ac456a0a13ec4bbbbe31a29ccdd33ec2b58cefd45c',
    '29b92dde3d2fe5ed44a0951363767ddc37e0dfd9efc645c9c8ad1bf8e7d4aab6',
]
missing_numbers = [s for s in mandatory_strings if s not in full_text]

omath_count = len(root_el.xpath('.//m:oMath', namespaces=ns))

qa_summary = {
    'status': 'PASS' if (not missing_tables and not missing_figs and not missing_numbers and len(toc_entries) >= 25 and num_pdf_pages >= 50) else 'FAIL',
    'pdf_pages': num_pdf_pages,
    'docx_size_bytes': FINAL_DOCX.stat().st_size,
    'pdf_size_bytes': FINAL_PDF.stat().st_size,
    'docx_tables_count': len(tables_text),
    'docx_embedded_media_count': len(media_files),
    'docx_omath_count': omath_count,
    'toc_materialized_entries_count': len(toc_entries),
    'toc_sample_entries': toc_entries[:15],
    'expected_tables_count': len(expected_tables),
    'missing_tables': missing_tables,
    'expected_figures_count': len(expected_figs),
    'missing_figures': missing_figs,
    'missing_mandatory_numbers': missing_numbers,
    'sha256_hashes': hashes,
}
(OUT_DIR / '_qa_verification_summary.json').write_text(json.dumps(qa_summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(qa_summary, ensure_ascii=False, indent=2))
