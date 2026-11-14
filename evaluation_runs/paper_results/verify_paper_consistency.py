"""Automated consistency verification between source JSON/CSV files and paper_results."""
from __future__ import annotations

import csv
import json
import pathlib

ROOT = pathlib.Path(r"D:\Download Acer\garbage_dinov2_starter")
VAL_DIR = ROOT / "evaluation_runs" / "val_grid_search_01"
TEST_DIR = ROOT / "evaluation_runs" / "final_test_four_way_01"
PAPER_DIR = ROOT / "evaluation_runs" / "paper_results"
TAB_DIR = PAPER_DIR / "tables"
FIG_DIR = PAPER_DIR / "figures"
QUAL_DIR = PAPER_DIR / "qualitative"


def main():
    val_err = json.loads((VAL_DIR / "cascade_error_analysis.json").read_text(encoding="utf-8"))
    val_rep = json.loads((VAL_DIR / "validation_grid_search_report.json").read_text(encoding="utf-8"))
    test_met = json.loads((TEST_DIR / "metrics.json").read_text(encoding="utf-8"))
    manuscript = (PAPER_DIR / "experimental_results_vi.md").read_text(encoding="utf-8")

    checks = []

    def record(name: str, passed: bool, detail: str):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})
        if not passed:
            raise AssertionError(f"Check failed: {name} -> {detail}")

    # 1. Verify Validation localized GT confusion statistics from cascade_error_analysis.json
    loc = val_err["matching_and_confusion_nms070"]["one_to_one_localized_gt_all_proposals"]
    cm_cls = val_err["matching_and_confusion_nms070"]["confusion_matrices_1to1_all_proposals"]["gt_5_vs_classifier_10"]
    cm_det = val_err["matching_and_confusion_nms070"]["confusion_matrices_1to1_all_proposals"]["gt_5_vs_detector_5"]

    record(
        "val_localized_gt_counts",
        loc["total_gt"] == 3932 and loc["localized_gt"] == 3663,
        f"total_gt={loc['total_gt']}, localized_gt={loc['localized_gt']}",
    )
    record(
        "val_detector_correct_rate",
        loc["detector_correct_count"] == 3618 and abs(loc["detector_correct_rate_of_localized"] - 3618 / 3663) < 1e-6,
        f"3618/3663 = {3618/3663:.6f} (98.77%)",
    )
    record(
        "val_classifier_correct_rate",
        loc["classifier_correct_count"] == 2807 and abs(loc["classifier_correct_rate_of_localized"] - 2807 / 3663) < 1e-6,
        f"2807/3663 = {2807/3663:.6f} (76.63%)",
    )
    record(
        "val_classifier_oos_rate",
        loc["classifier_oos_lost_count"] == 436 and abs(loc["classifier_oos_lost_rate_of_localized"] - 436 / 3663) < 1e-6,
        f"436/3663 = {436/3663:.6f} (11.90%)",
    )
    record(
        "val_classifier_inscope_wrong_rate",
        loc["classifier_wrong_inscope_count"] == 420 and abs(loc["classifier_wrong_inscope_rate_of_localized"] - 420 / 3663) < 1e-6,
        f"420/3663 = {420/3663:.6f} (11.47%)",
    )
    glass_tot = sum(cm_cls["glass"].values())
    glass_to_plastic = cm_cls["glass"]["plastic"]
    record(
        "val_glass_to_plastic_rate",
        glass_tot == 311 and glass_to_plastic == 168 and abs(glass_to_plastic / glass_tot - 168 / 311) < 1e-6,
        f"168/311 = {168/311:.6f} (54.02%)",
    )

    # 2. Verify Test proposal counts and NMS behavior
    exps = test_met["experiments"]
    record(
        "test_nms_protocol_is_class_agnostic",
        "Class-agnostic" in test_met["protocol"]["nms"],
        test_met["protocol"]["nms"],
    )
    det_nms_ret = exps["detector_nms"]["proposal_counts"]["retained"]
    casc_nms_ret = exps["cascade_nms"]["proposal_counts"]["retained"]
    casc_nms_exp = exps["cascade_nms"]["exported_prediction_count"]
    casc_nms_oos = exps["cascade_nms"]["outside_scope_dropped"]
    record(
        "test_proposals_per_image",
        det_nms_ret == 32993
        and casc_nms_ret == 32993
        and casc_nms_exp == 28334
        and casc_nms_oos == 4659
        and abs(det_nms_ret / 871 - 37.8794489) < 1e-5,
        f"retained=32993 (37.88/img), cascade_nms exported=28334 (32.53/img), oos=4659 (5.35/img)",
    )

    # 3. Verify Table 1 CSV matches final_test_four_way_01/metrics.json
    with (TAB_DIR / "table1_main_test_comparison.csv").open("r", encoding="utf-8-sig") as f:
        t1_rows = list(csv.DictReader(f))
    for r in t1_rows:
        m = r["Method"]
        em = exps[m]["metrics"]
        record(
            f"table1_{m}_metrics",
            abs(float(r["mAP@50"]) - em["mAP50"]) < 1e-4
            and abs(float(r["mAP@50:95"]) - em["mAP50_95"]) < 1e-4
            and abs(float(r["AR@100"]) - em["AR100"]) < 1e-4
            and abs(float(r["Precision@50"]) - em["precision_iou50"]) < 1e-4
            and abs(float(r["Recall@50"]) - em["recall_iou50"]) < 1e-4
            and abs(float(r["F1@50"]) - em["f1_iou50"]) < 1e-4
            and int(r["TP"]) == em["TP"]
            and int(r["FP"]) == em["FP"]
            and int(r["FN"]) == em["FN"]
            and int(r["Proposals Retained"]) == exps[m]["proposal_counts"]["retained"]
            and int(r["Exported Predictions"]) == exps[m]["exported_prediction_count"]
            and int(r["OOS Dropped"]) == exps[m]["outside_scope_dropped"],
            f"{m} verified against metrics.json",
        )

    # 4. Verify Table 5 CSV matches cascade_error_analysis.json
    with (TAB_DIR / "table5_validation_confusion_matrix.csv").open("r", encoding="utf-8-sig") as f:
        t5_rows = list(csv.DictReader(f))
    record("table5_row_count", len(t5_rows) == 6, f"rows={len(t5_rows)} (5 classes + Total)")
    tot_row = t5_rows[-1]
    record(
        "table5_totals",
        int(tot_row["Total Localized GT"]) == 3663
        and "76.63% (2807/3663)" in tot_row["DINOv2 Correct (%)"]
        and "11.47% (420/3663)" in tot_row["DINOv2 In-Scope Wrong (%)"]
        and "11.90% (436/3663)" in tot_row["DINOv2 OOS Lost (%)"]
        and "98.77% (3618/3663)" in tot_row["Detector Correct (%)"],
        str(tot_row),
    )

    # 5. Verify Manuscript contains exact required strings and zero erroneous strings
    required_manuscript_tokens = [
        "3.618 / 3.663",
        "98,77%",
        "2.807 / 3.663",
        "76,63%",
        "436 / 3.663",
        "11,90%",
        "420 / 3.663",
        "11,47%",
        "54,02%",
        "29,76%",
        "32.993",
        "37.88",
        "28.334",
        "32.53",
        "Class-Agnostic NMS",
        "0.7332",
        "0.7801",
        "0.4134",
        "0.4486",
    ]
    for tok in required_manuscript_tokens:
        record(f"manuscript_has_{tok}", tok in manuscript, f"Token '{tok}' present in experimental_results_vi.md")

    forbidden_manuscript_tokens = [
        "91.29", "91,29",
        "64.07", "64,07",
        "641/3663", "641/3,663", "641 / 3,663",
        "17.50%", "17,50%",
        "22.39%", "22,39%",
        "38.28%", "38,28%",
        "44.71 boxes", "44,71 boxes",
        "Class-wise NMS", "class-wise NMS",
    ]
    for ftok in forbidden_manuscript_tokens:
        record(f"manuscript_free_of_{ftok}", ftok not in manuscript, f"Forbidden token '{ftok}' absent")

    # 5b. Erratum 2026-10-30: '641' has no source; organic->biological is 1618/1719 in the source matrix
    organic_row = cm_cls["organic"]
    record(
        "val_organic_to_biological_is_1618_of_1719",
        organic_row.get("biological") == 1618 and sum(organic_row.values()) == 1719,
        f"organic->biological={organic_row.get('biological')}, organic_total={sum(organic_row.values())}",
    )
    record(
        "val_no_cell_equals_641",
        all(v != 641 for row in cm_cls.values() for v in row.values()),
        "No confusion-matrix cell equals 641",
    )
    audit_text = (PAPER_DIR / "consistency_audit.md").read_text(encoding="utf-8")
    record(
        "audit_free_of_retracted_641_explanation",
        "`641` là số box GT lớp `organic`" not in audit_text,
        "Retracted explanation of 641 no longer asserted in consistency_audit.md",
    )

    # 6. Verify all 8 figures (.png and .pdf) and 5 tables (.csv, .md, .tex) exist and are non-empty
    for i in range(1, 9):
        png_files = list(FIG_DIR.glob(f"fig{i}_*.png"))
        pdf_files = list(FIG_DIR.glob(f"fig{i}_*.pdf"))
        record(f"fig{i}_exists", len(png_files) == 1 and len(pdf_files) == 1 and png_files[0].stat().st_size > 10000, f"fig{i} verified")

    for i in range(1, 6):
        csv_files = list(TAB_DIR.glob(f"table{i}_*.csv"))
        md_files = list(TAB_DIR.glob(f"table{i}_*.md"))
        tex_files = list(TAB_DIR.glob(f"table{i}_*.tex"))
        record(f"table{i}_exists", len(csv_files) == 1 and len(md_files) == 1 and len(tex_files) == 1, f"table{i} verified")

    out_summary = {
        "status": "ALL_CHECKS_PASSED",
        "total_checks": len(checks),
        "passed_checks": sum(1 for c in checks if c["passed"]),
        "failed_checks": sum(1 for c in checks if not c["passed"]),
    }
    (PAPER_DIR / "consistency_verification_results.json").write_text(
        json.dumps({"summary": out_summary, "checks": checks}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(out_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
