# -*- coding: utf-8 -*-
import os
import sys
import json
import time
import subprocess
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(r'D:\Download Acer\garbage_dinov2_starter')
OUT_DIR = ROOT / 'reports' / 'DG38_FINAL_SUBMISSION'
RENDERER_PROG = ROOT / 'evaluation_runs' / 'report_final_review' / '_submission_qa' / 'renderer' / 'program'
SOFFICE = RENDERER_PROG / 'soffice.exe'

os.environ['SAL_NO_ACCESSIBILITY'] = '1'
os.environ['SAL_DISABLE_SYNCHRONOUS_PRINTER_DETECTION'] = '1'

import uno
from com.sun.star.beans import PropertyValue

def prop(k, v):
    p = PropertyValue()
    p.Name = k
    p.Value = v
    return p

profile_dir = OUT_DIR / '_lo_profile'
profile_dir.mkdir(parents=True, exist_ok=True)
profile_url = uno.systemPathToFileUrl(str(profile_dir.resolve()))

cmd = [
    str(SOFFICE),
    '--headless',
    '--invisible',
    '--nocrashreport',
    '--nodefault',
    '--nofirststartwizard',
    '--nologo',
    '--norestore',
    f'-env:UserInstallation={profile_url}',
    '--accept=socket,host=127.0.0.1,port=2018;urp;StarOffice.ServiceManager',
]

print('Starting LibreOffice UNO server...', flush=True)
proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

try:
    local = uno.getComponentContext()
    resolver = local.ServiceManager.createInstanceWithContext('com.sun.star.bridge.UnoUrlResolver', local)
    ctx = None
    for attempt in range(120):
        try:
            ctx = resolver.resolve('uno:socket,host=127.0.0.1,port=2018;urp;StarOffice.ComponentContext')
            break
        except Exception:
            time.sleep(0.5)
    if ctx is None:
        raise RuntimeError('Failed to connect to LibreOffice UNO server on port 2018')

    print('UNO connected!', flush=True)
    sm = ctx.ServiceManager
    desktop = sm.createInstanceWithContext('com.sun.star.frame.Desktop', ctx)
    url = lambda p: uno.systemPathToFileUrl(str(Path(p).resolve()))

    loadprops = (
        prop('Hidden', True),
        prop('ReadOnly', False),
        prop('MacroExecutionMode', uno.getConstantByName('com.sun.star.document.MacroExecMode.NEVER_EXECUTE')),
        prop('UpdateDocMode', uno.getConstantByName('com.sun.star.document.UpdateDocMode.FULL_UPDATE')),
    )

    input_docx = OUT_DIR / '_build_input.docx'
    final_docx = OUT_DIR / 'DG38_Final_Complete_Report.docx'
    final_pdf = OUT_DIR / 'DG38_Final_Complete_Report.pdf'

    doc = desktop.loadComponentFromURL(url(input_docx), '_blank', 0, loadprops)
    assert doc is not None, 'Failed to load _build_input.docx'
    print('Loaded _build_input.docx', flush=True)

    indexes = doc.getDocumentIndexes()
    print('Initial INDEX_COUNT:', indexes.getCount(), flush=True)
    if indexes.getCount() == 0:
        search = doc.createSearchDescriptor()
        search.SearchString = 'MỤC LỤC'
        found = doc.findFirst(search)
        assert found is not None, 'Could not find MỤC LỤC title'
        cursor = doc.Text.createTextCursorByRange(found.End)
        cursor.gotoNextParagraph(False)
        idx = doc.createInstance('com.sun.star.text.ContentIndex')
        idx.CreateFromOutline = True
        idx.Level = 2
        idx.Title = ''
        doc.Text.insertTextContent(cursor, idx, False)
    else:
        idx = indexes.getByIndex(0)
        idx.Title = ''
        idx.CreateFromOutline = True
        idx.Level = 2

    styles = doc.StyleFamilies.getByName('ParagraphStyles')
    for name in ['Contents 1', 'Contents 2', 'Contents1', 'Contents2', 'TOC 1', 'TOC 2']:
        if styles.hasByName(name):
            st = styles.getByName(name)
            st.CharFontName = 'Times New Roman'
            st.CharHeight = 11
            st.ParaTopMargin = 0
            st.ParaBottomMargin = 70

    for rep in range(3):
        doc.refresh()
        doc.TextFields.refresh()
        idx.update()
        vc = doc.CurrentController.ViewCursor
        vc.jumpToLastPage()
        print(f'Layout pass {rep}: {vc.Page} pages', flush=True)

    doc.storeAsURL(url(final_docx), (prop('FilterName', 'Office Open XML Text'), prop('Overwrite', True)))
    toc_text = idx.Anchor.String
    print('Saved DG38_Final_Complete_Report.docx; TOC length:', len(toc_text), flush=True)
    doc.close(True)

    # Reopen exported DOCX, refresh fields/index once more, save and export PDF
    doc = desktop.loadComponentFromURL(url(final_docx), '_blank', 0, loadprops)
    indexes = doc.getDocumentIndexes()
    assert indexes.getCount() >= 1
    for rep in range(2):
        doc.refresh()
        doc.TextFields.refresh()
        indexes.getByIndex(0).update()
        vc = doc.CurrentController.ViewCursor
        vc.jumpToLastPage()
        print(f'Reopen layout pass {rep}: {vc.Page} pages', flush=True)
    doc.store()
    toc_final = indexes.getByIndex(0).Anchor.String
    toc_count = indexes.getCount()
    if final_pdf.exists():
        try:
            final_pdf.unlink()
        except Exception:
            pass
    doc.storeToURL(url(final_pdf), (prop('FilterName', 'writer_pdf_Export'), prop('Overwrite', True)))
    page_count = vc.Page
    doc.close(True)

    meta = {
        'renderer': 'LibreOffice 26.8.2.2 task-local UNO',
        'toc_count': toc_count,
        'toc_text': toc_final,
        'page_count': page_count,
        'docx': str(final_docx),
        'pdf': str(final_pdf),
    }
    (OUT_DIR / '_uno_export_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Successfully exported PDF ({page_count} pages): {final_pdf}', flush=True)
finally:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
