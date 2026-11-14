import csv
import json
import pathlib
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

ROOT = pathlib.Path(r"D:\Download Acer\garbage_dinov2_starter")
VAL_DIR = ROOT / "evaluation_runs" / "val_grid_search_01"
TEST_DIR = ROOT / "evaluation_runs" / "final_test_four_way_01"
TEST_DATA_DIR = ROOT / "WasteDetect_Dataset" / "test_final_candidate"
OUT_DIR = ROOT / "evaluation_runs" / "paper_results"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"
QUAL_DIR = OUT_DIR / "qualitative"

for d in (OUT_DIR, FIG_DIR, TAB_DIR, QUAL_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Configure publication-grade matplotlib styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Segoe UI"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.labelweight": "bold",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linestyle": "--",
})

METHOD_ORDER = ["detector_original", "detector_nms", "cascade_original", "cascade_nms"]
METHOD_LABELS = {
    "detector_original": "Detector\n(Original, No NMS)",
    "detector_nms": "Detector + NMS\n(IoU = 0.70)",
    "cascade_original": "Cascade\n(Original, No NMS)",
    "cascade_nms": "Cascade + NMS\n(IoU = 0.70)",
}
COLORS = {
    "detector_original": "#4C78A8",
    "detector_nms": "#1F4E78",
    "cascade_original": "#F58518",
    "cascade_nms": "#D62728",
}
CLASS_ORDER = ["glass", "metal", "organic", "paper", "plastic"]
CLASS_LABELS = {
    "glass": "Glass",
    "metal": "Metal",
    "organic": "Organic",
    "paper": "Paper",
    "plastic": "Plastic",
}


def save_fig(fig, stem):
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300)
    fig.savefig(FIG_DIR / f"{stem}.pdf")
    plt.close(fig)


def write_csv_md_tex(stem, headers, rows, caption=""):
    # CSV
    csv_path = TAB_DIR / f"{stem}.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)

    # Markdown
    md_lines = [f"| {' | '.join(headers)} |", f"| {' | '.join(['---'] * len(headers))} |"]
    for r in rows:
        md_lines.append(f"| {' | '.join(str(x) for x in r)} |")
    (TAB_DIR / f"{stem}.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # LaTeX
    col_spec = "l" + "c" * (len(headers) - 1)
    tex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}" if caption else "",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\hline",
        " & ".join(f"\\textbf{{{h.replace('%', '\\%').replace('_', '\\_')}}}" for h in headers) + " \\\\",
        "\\hline",
    ]
    for r in rows:
        escaped = [str(x).replace("%", "\\%").replace("_", "\\_") for x in r]
        tex_lines.append(" & ".join(escaped) + " \\\\")
    tex_lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])
    (TAB_DIR / f"{stem}.tex").write_text("\n".join([l for l in tex_lines if l]) + "\n", encoding="utf-8")


def get_val_cascade_original_at_thr(val_report, target_thr=0.25):
    sweep = val_report["no_nms"]["cascade_original"]["pr_threshold_sweep"]
    for row in sweep:
        if abs(float(row["pr_score_threshold"]) - target_thr) < 1e-9:
            return row
    raise KeyError(f"Threshold {target_thr} not found in cascade_original sweep")


def generate_tables_and_charts(test_metrics, val_report, val_err):
    exps = test_metrics["experiments"]

    # 1. Table 1: Main Test 4-Method Comparison
    t1_headers = [
        "Method", "NMS IoU", "PR Thr", "mAP@50", "mAP@50:95", "AR@100",
        "Precision@50", "Recall@50", "F1@50", "TP", "FP", "FN",
        "Proposals Retained", "Exported Predictions", "OOS Dropped",
        "Latency Mean (ms)", "Latency P50 (ms)", "Latency P95 (ms)", "FPS"
    ]
    t1_rows = []
    for m in METHOD_ORDER:
        e = exps[m]
        met = e["metrics"]
        lat = e["latency_ms"]
        fps = 1000.0 / lat["end_to_end_mean"]
        t1_rows.append([
            m,
            "None" if e["nms_iou"] is None else f"{e['nms_iou']:.2f}",
            f"{e['pr_score_threshold']:.2f}",
            f"{met['mAP50']:.4f}",
            f"{met['mAP50_95']:.4f}",
            f"{met['AR100']:.4f}",
            f"{met['precision_iou50']:.4f}",
            f"{met['recall_iou50']:.4f}",
            f"{met['f1_iou50']:.4f}",
            met["TP"],
            met["FP"],
            met["FN"],
            e["proposal_counts"]["retained"],
            e["exported_prediction_count"],
            e["outside_scope_dropped"],
            f"{lat['end_to_end_mean']:.2f}",
            f"{lat['end_to_end_p50']:.2f}",
            f"{lat['end_to_end_p95']:.2f}",
            f"{fps:.2f}",
        ])
    write_csv_md_tex(
        "table1_main_test_comparison",
        t1_headers,
        t1_rows,
        "Comparison of the four evaluated configurations on the final held-out Test set (871 images, 1,884 bounding boxes).",
    )

    # 2. Table 2: Per-Class AP50 and AP50:95 on Test
    t2_headers = [
        "Class", "GT Boxes",
        "detector_original AP50", "detector_nms AP50", "cascade_original AP50", "cascade_nms AP50",
        "detector_original AP50:95", "detector_nms AP50:95", "cascade_original AP50:95", "cascade_nms AP50:95",
        "Delta AP50 (NMS on Det)", "Delta AP50 (Cascade vs Det NMS)"
    ]
    t2_rows = []
    gt_counts = test_metrics["gt_boxes_by_class"]
    for c in CLASS_ORDER:
        d_orig_50 = exps["detector_original"]["per_class_ap"][c]["AP50"]
        d_nms_50 = exps["detector_nms"]["per_class_ap"][c]["AP50"]
        c_orig_50 = exps["cascade_original"]["per_class_ap"][c]["AP50"]
        c_nms_50 = exps["cascade_nms"]["per_class_ap"][c]["AP50"]

        d_orig_95 = exps["detector_original"]["per_class_ap"][c]["AP50_95"]
        d_nms_95 = exps["detector_nms"]["per_class_ap"][c]["AP50_95"]
        c_orig_95 = exps["cascade_original"]["per_class_ap"][c]["AP50_95"]
        c_nms_95 = exps["cascade_nms"]["per_class_ap"][c]["AP50_95"]

        t2_rows.append([
            c, gt_counts[c],
            f"{d_orig_50:.4f}", f"{d_nms_50:.4f}", f"{c_orig_50:.4f}", f"{c_nms_50:.4f}",
            f"{d_orig_95:.4f}", f"{d_nms_95:.4f}", f"{c_orig_95:.4f}", f"{c_nms_95:.4f}",
            f"{d_nms_50 - d_orig_50:+.4f}", f"{c_nms_50 - d_nms_50:+.4f}",
        ])
    t2_rows.append([
        "Mean (mAP)", sum(gt_counts.values()),
        f"{exps['detector_original']['metrics']['mAP50']:.4f}",
        f"{exps['detector_nms']['metrics']['mAP50']:.4f}",
        f"{exps['cascade_original']['metrics']['mAP50']:.4f}",
        f"{exps['cascade_nms']['metrics']['mAP50']:.4f}",
        f"{exps['detector_original']['metrics']['mAP50_95']:.4f}",
        f"{exps['detector_nms']['metrics']['mAP50_95']:.4f}",
        f"{exps['cascade_original']['metrics']['mAP50_95']:.4f}",
        f"{exps['cascade_nms']['metrics']['mAP50_95']:.4f}",
        f"{exps['detector_nms']['metrics']['mAP50'] - exps['detector_original']['metrics']['mAP50']:+.4f}",
        f"{exps['cascade_nms']['metrics']['mAP50'] - exps['detector_nms']['metrics']['mAP50']:+.4f}",
    ])
    write_csv_md_tex(
        "table2_per_class_ap_test",
        t2_headers,
        t2_rows,
        "Per-class AP@50 and AP@50:95 across the five waste categories on the final Test set.",
    )

    # 3. Table 3: Validation vs Test Consistency
    # Note: On Validation summary.csv, cascade_original's own best PR threshold was 0.30 (F1@50=0.4988),
    # while at the locked Cascade family threshold 0.25 (used on Test), Val F1@50 is 0.4911.
    val_map = {r["experiment"]: r for r in val_report["summary_at_shared_nms"]}
    val_casc_orig_025 = get_val_cascade_original_at_thr(val_report, 0.25)
    t3_headers = [
        "Method", "Locked PR Thr",
        "Val mAP@50", "Test mAP@50", "Diff mAP@50",
        "Val mAP@50:95", "Test mAP@50:95", "Diff mAP@50:95",
        "Val F1@50 (at Locked Thr)", "Test F1@50 (at Locked Thr)", "Diff F1@50 (Locked Thr)",
        "Val AR@100", "Test AR@100"
    ]
    t3_rows = []
    val_f1_locked_map = {}
    for m in METHOD_ORDER:
        vr = val_map[m]
        tm = exps[m]["metrics"]
        locked_thr = exps[m]["pr_score_threshold"]
        if m == "cascade_original":
            val_f1_locked = val_casc_orig_025["f1_iou50"]
            val_f1_str = f"{val_f1_locked:.4f} (0.4988 at thr=0.30)"
        else:
            val_f1_locked = vr["f1_iou50"]
            val_f1_str = f"{val_f1_locked:.4f}"
        val_f1_locked_map[m] = val_f1_locked
        t3_rows.append([
            m,
            f"{locked_thr:.2f}",
            f"{vr['mAP50']:.4f}", f"{tm['mAP50']:.4f}", f"{tm['mAP50'] - vr['mAP50']:+.4f}",
            f"{vr['mAP50_95']:.4f}", f"{tm['mAP50_95']:.4f}", f"{tm['mAP50_95'] - vr['mAP50_95']:+.4f}",
            val_f1_str, f"{tm['f1_iou50']:.4f}", f"{tm['f1_iou50'] - val_f1_locked:+.4f}",
            f"{vr['AR100']:.4f}", f"{tm['AR100']:.4f}",
        ])
    write_csv_md_tex(
        "table3_validation_vs_test_consistency",
        t3_headers,
        t3_rows,
        "Consistency of evaluation metrics between Validation (1,743 images, 3,932 boxes) and Test (871 images, 1,884 boxes) at locked parameters.",
    )

    # 4. Table 4: Validation Factorial Ablation
    t4_headers = [
        "Ablation ID", "Configuration Formula", "Exported Preds", "OOS Dropped",
        "mAP@50", "mAP@50:95", "AR@100", "Best F1@50",
        "glass AP50", "metal AP50", "organic AP50", "paper AP50", "plastic AP50"
    ]
    t4_rows = []
    for k, v in val_err["diagnostic_factorial_ablation_nms070"].items():
        pc = v["per_class"]
        t4_rows.append([
            k, v["description"], v["exported_predictions"], v["outside_scope_dropped"],
            f"{v['mAP50']:.4f}", f"{v['mAP50_95']:.4f}", f"{v['AR100']:.4f}", f"{v['best_pr']['f1_iou50']:.4f}",
            f"{pc['glass']['AP50']:.4f}", f"{pc['metal']['AP50']:.4f}", f"{pc['organic']['AP50']:.4f}",
            f"{pc['paper']['AP50']:.4f}", f"{pc['plastic']['AP50']:.4f}",
        ])
    write_csv_md_tex(
        "table4_validation_factorial_ablation",
        t4_headers,
        t4_rows,
        "Diagnostic factorial ablation on Validation (NMS IoU = 0.70) isolating class replacement, out-of-scope dropping, and score fusion.",
    )

    # 5. Table 5: Validation Confusion Matrix (5 GT x 10 Classifier at IoU >= 0.5)
    cls_cols = ["glass", "metal", "biological", "paper", "plastic", "cardboard", "trash", "battery", "clothes", "shoes"]
    inscope_cols = {"glass", "metal", "biological", "paper", "plastic"}
    oos_cols = {"cardboard", "trash", "battery", "clothes", "shoes"}
    cm_data = val_err["matching_and_confusion_nms070"]["confusion_matrices_1to1_all_proposals"]["gt_5_vs_classifier_10"]
    cm_det = val_err["matching_and_confusion_nms070"]["confusion_matrices_1to1_all_proposals"]["gt_5_vs_detector_5"]
    t5_headers = (
        ["GT Class"]
        + cls_cols
        + ["Total Localized GT", "DINOv2 Correct (%)", "DINOv2 In-Scope Wrong (%)", "DINOv2 OOS Lost (%)", "Detector Correct (%)"]
    )
    t5_rows = []
    map_correct_col = {"glass": "glass", "metal": "metal", "organic": "biological", "paper": "paper", "plastic": "plastic"}
    col_totals = [0] * len(cls_cols)
    grand_tot = 0
    grand_cls_ok = 0
    grand_cls_in_wrong = 0
    grand_cls_oos = 0
    grand_det_ok = 0
    for g in CLASS_ORDER:
        row_counts = [cm_data[g][c] for c in cls_cols]
        for idx_c, val_c in enumerate(row_counts):
            col_totals[idx_c] += val_c
        tot = sum(row_counts)
        cls_ok = cm_data[g][map_correct_col[g]]
        cls_oos = sum(cm_data[g][c] for c in oos_cols)
        cls_in_wrong = sum(cm_data[g][c] for c in inscope_cols) - cls_ok
        det_ok = cm_det[g][g]
        grand_tot += tot
        grand_cls_ok += cls_ok
        grand_cls_in_wrong += cls_in_wrong
        grand_cls_oos += cls_oos
        grand_det_ok += det_ok
        t5_rows.append(
            [g]
            + row_counts
            + [
                tot,
                f"{100.0 * cls_ok / tot:.2f}%",
                f"{100.0 * cls_in_wrong / tot:.2f}%",
                f"{100.0 * cls_oos / tot:.2f}%",
                f"{100.0 * det_ok / tot:.2f}%",
            ]
        )
    t5_rows.append(
        ["Total"]
        + col_totals
        + [
            grand_tot,
            f"{100.0 * grand_cls_ok / grand_tot:.2f}% ({grand_cls_ok}/{grand_tot})",
            f"{100.0 * grand_cls_in_wrong / grand_tot:.2f}% ({grand_cls_in_wrong}/{grand_tot})",
            f"{100.0 * grand_cls_oos / grand_tot:.2f}% ({grand_cls_oos}/{grand_tot})",
            f"{100.0 * grand_det_ok / grand_tot:.2f}% ({grand_det_ok}/{grand_tot})",
        ]
    )
    write_csv_md_tex(
        "table5_validation_confusion_matrix",
        t5_headers,
        t5_rows,
        "Confusion matrix between 5 Detection Ground-Truth classes and 10 DINOv2 Classifier labels on 3,663 localized GT boxes (1-to-1 greedy matching at IoU >= 0.50, Validation NMS IoU = 0.70).",
    )

    # --- FIGURE 1: mAP@50 Comparison on Test ---
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    x = np.arange(len(METHOD_ORDER))
    vals = [exps[m]["metrics"]["mAP50"] for m in METHOD_ORDER]
    bars = ax.bar(x, vals, width=0.56, color=[COLORS[m] for m in METHOD_ORDER], edgecolor="#222222", linewidth=0.8, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER])
    ax.set_ylabel("mAP@0.50 (COCOeval)")
    ax.set_ylim(0, 0.97)
    ax.set_title("Comparison of mAP@0.50 Across Four Methods on Final Test Set (n = 871 images)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.4f}\n({v*100:.2f}%)", ha="center", va="bottom", fontweight="bold", fontsize=9)
    # Annotate NMS improvement brackets using percentage points (pp) and relative %
    ax.annotate(
        "+0.0469 (+4.69 pp | +6.39% rel)",
        xy=(1, vals[1] + 0.075), xytext=(0.5, 0.895),
        ha="center", fontsize=8.5, fontweight="bold", color="#1F4E78",
        bbox=dict(boxstyle="round,pad=0.25", fc="#EAF2F8", ec="#1F4E78", lw=0.8)
    )
    ax.annotate(
        "+0.0352 (+3.52 pp | +8.51% rel)",
        xy=(3, vals[3] + 0.075), xytext=(2.5, 0.57),
        ha="center", fontsize=8.5, fontweight="bold", color="#B01E1E",
        bbox=dict(boxstyle="round,pad=0.25", fc="#FDEDEC", ec="#D62728", lw=0.8)
    )
    save_fig(fig, "fig1_map50_comparison")

    # --- FIGURE 2: mAP@50:95 and AR@100 Comparison on Test ---
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    w = 0.34
    map95_vals = [exps[m]["metrics"]["mAP50_95"] for m in METHOD_ORDER]
    ar100_vals = [exps[m]["metrics"]["AR100"] for m in METHOD_ORDER]
    b1 = ax.bar(x - w/2, map95_vals, width=w, label="mAP@0.50:0.95", color="#1F4E78", edgecolor="#222222", linewidth=0.8, zorder=3)
    b2 = ax.bar(x + w/2, ar100_vals, width=w, label="AR@100 (IoU 0.50:0.95)", color="#59A14F", edgecolor="#222222", linewidth=0.8, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER])
    ax.set_ylabel("Score (COCOeval)")
    ax.set_ylim(0, 0.76)
    ax.set_title("mAP@0.50:0.95 and AR@100 Across Four Methods on Final Test Set")
    ax.legend(loc="upper right", frameon=True)
    for b in list(b1) + list(b2):
        v = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.4f}", ha="center", va="bottom", fontsize=8.2, fontweight="bold")
    save_fig(fig, "fig2_map50_95_comparison")

    # --- FIGURE 3: Precision@50, Recall@50, F1@50 at Locked Operating Thresholds ---
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    w = 0.25
    p_vals = [exps[m]["metrics"]["precision_iou50"] for m in METHOD_ORDER]
    r_vals = [exps[m]["metrics"]["recall_iou50"] for m in METHOD_ORDER]
    f1_vals = [exps[m]["metrics"]["f1_iou50"] for m in METHOD_ORDER]
    bp = ax.bar(x - w, p_vals, width=w, label="Precision@0.50", color="#2B5C8F", edgecolor="#222222", linewidth=0.7, zorder=3)
    br = ax.bar(x, r_vals, width=w, label="Recall@0.50", color="#38A6A5", edgecolor="#222222", linewidth=0.7, zorder=3)
    bf = ax.bar(x + w, f1_vals, width=w, label="Micro-F1@0.50", color="#E17C05", edgecolor="#222222", linewidth=0.7, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([
        "Detector (Original)\n[thr = 0.30]",
        "Detector + NMS\n[IoU=0.70, thr=0.30]",
        "Cascade (Original)\n[thr = 0.25]",
        "Cascade + NMS\n[IoU=0.70, thr=0.25]",
    ])
    ax.set_ylabel("Operating Point Metric at IoU = 0.50")
    ax.set_ylim(0, 0.92)
    ax.set_title("Precision, Recall, and F1@0.50 at Validation-Locked Operating Thresholds (Test Set)")
    ax.legend(loc="upper right", frameon=True)
    for b in list(bp) + list(br) + list(bf):
        v = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=7.8, fontweight="bold")
    save_fig(fig, "fig3_precision_recall_f1")

    # --- FIGURE 4: Latency Comparison & Breakdown ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.8), gridspec_kw={"wspace": 0.25})
    det_ms = np.array([exps[m]["latency_ms"]["detector_model_mean"] for m in METHOD_ORDER])
    cls_ms = np.array([exps[m]["latency_ms"]["classifier_model_mean"] for m in METHOD_ORDER])
    e2e_ms = np.array([exps[m]["latency_ms"]["end_to_end_mean"] for m in METHOD_ORDER])
    p50_ms = np.array([exps[m]["latency_ms"]["end_to_end_p50"] for m in METHOD_ORDER])
    other_ms = np.maximum(0.0, e2e_ms - det_ms - cls_ms)

    ax1.bar(x, det_ms, width=0.52, label="Deformable DETR Forward", color="#1F4E78", edgecolor="#222222", linewidth=0.7, zorder=3)
    ax1.bar(x, cls_ms, width=0.52, bottom=det_ms, label="DINOv2 Classifier Forward", color="#D62728", edgecolor="#222222", linewidth=0.7, zorder=3)
    ax1.bar(x, other_ms, width=0.52, bottom=det_ms + cls_ms, label="Crop / Transforms / NMS Overhead", color="#F28E2B", edgecolor="#222222", linewidth=0.7, zorder=3)
    ax1.scatter(x, p50_ms, color="#111111", marker="D", s=36, zorder=5, label="Median (p50)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER])
    ax1.set_ylabel("Inference Latency per Image (ms)")
    ax1.set_ylim(0, 3150)
    ax1.set_title("(a) End-to-End Latency Breakdown (Mean & p50)")
    ax1.legend(loc="upper left", frameon=True, fontsize=8.2)
    for i, tot in enumerate(e2e_ms):
        ax1.text(i, tot + 50, f"{tot:.1f} ms\n(p50: {p50_ms[i]:.1f})", ha="center", va="bottom", fontsize=8.0, fontweight="bold")

    # Panel (b): Accuracy (mAP@50) vs Throughput (FPS) — non-overlapping callout boxes
    fps_vals = 1000.0 / e2e_ms
    callout_positions = {
        "detector_original": (5.2, 0.655),
        "detector_nms": (5.2, 0.825),
        "cascade_original": (3.1, 0.365),
        "cascade_nms": (3.4, 0.525),
    }
    for i, m in enumerate(METHOD_ORDER):
        ax2.scatter(fps_vals[i], vals[i], s=140, color=COLORS[m], edgecolors="#111111", linewidth=1.2, zorder=4)
        tx, ty = callout_positions[m]
        ax2.annotate(
            f"{m}\n({fps_vals[i]:.2f} FPS, mAP50={vals[i]:.4f})",
            xy=(fps_vals[i], vals[i]),
            xytext=(tx, ty),
            fontsize=8.2, fontweight="bold",
            ha="center", va="center",
            arrowprops=dict(arrowstyle="->", color=COLORS[m], lw=1.1, shrinkA=4, shrinkB=6),
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=COLORS[m], lw=1.0),
            zorder=5,
        )
    ax2.set_xlabel("Throughput (Frames Per Second, FPS)")
    ax2.set_ylabel("mAP@0.50 (COCOeval)")
    ax2.set_xlim(-0.4, 9.4)
    ax2.set_ylim(0.30, 0.90)
    ax2.set_title("(b) Accuracy (mAP@0.50) vs. Inference Speed Trade-off")
    save_fig(fig, "fig4_latency_comparison")

    # --- FIGURE 5: Per-Class AP@50 on Test ---
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    xc = np.arange(len(CLASS_ORDER))
    w = 0.19
    offsets = [-1.5 * w, -0.5 * w, 0.5 * w, 1.5 * w]
    for offset, m in zip(offsets, METHOD_ORDER):
        c_vals = [exps[m]["per_class_ap"][c]["AP50"] for c in CLASS_ORDER]
        rects = ax.bar(xc + offset, c_vals, width=w, label=m, color=COLORS[m], edgecolor="#222222", linewidth=0.65, zorder=3)
        for r in rects:
            v = r.get_height()
            ax.text(r.get_x() + r.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7.2, fontweight="bold", rotation=90)
    ax.set_xticks(xc)
    ax.set_xticklabels([f"{CLASS_LABELS[c]}\n(GT = {gt_counts[c]})" for c in CLASS_ORDER])
    ax.set_ylabel("Category AP@0.50 (COCOeval)")
    ax.set_ylim(0, 1.15)
    ax.set_title("Per-Class AP@0.50 Across Five Waste Categories on Final Test Set")
    ax.legend(loc="upper right", ncol=2, frameon=True)
    save_fig(fig, "fig5_per_class_ap50")

    # --- FIGURE 6: Validation vs. Test Consistency (at Matched Locked Parameters) ---
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.7), gridspec_kw={"wspace": 0.24})
    metric_specs = [
        ("mAP50", "mAP@0.50", (0, 0.90)),
        ("mAP50_95", "mAP@0.50:0.95", (0, 0.54)),
        ("f1_iou50", "Operating Point F1@0.50 (Locked Thr)", (0, 0.80)),
    ]
    handles_f6, labels_f6 = None, None
    for ax_i, (mkey, mtitle, ylim) in zip(axes, metric_specs):
        if mkey == "f1_iou50":
            v_list = [val_f1_locked_map[m] for m in METHOD_ORDER]
        else:
            v_list = [val_map[m][mkey] for m in METHOD_ORDER]
        t_list = [exps[m]["metrics"][mkey] for m in METHOD_ORDER]
        w = 0.35
        b_val = ax_i.bar(x - w/2, v_list, width=w, label="Validation (1,743 images, 3,932 boxes)", color="#76B7B2", edgecolor="#222222", linewidth=0.7, zorder=3)
        b_tst = ax_i.bar(x + w/2, t_list, width=w, label="Test (871 images, 1,884 boxes)", color="#1F4E78", edgecolor="#222222", linewidth=0.7, zorder=3)
        ax_i.set_xticks(x)
        ax_i.set_xticklabels(["Det\nOrig", "Det\nNMS", "Casc\nOrig", "Casc\nNMS"])
        ax_i.set_title(mtitle)
        ax_i.set_ylim(*ylim)
        if ax_i is axes[0]:
            ax_i.set_ylabel("Score")
            handles_f6, labels_f6 = ax_i.get_legend_handles_labels()
        for b in list(b_val) + list(b_tst):
            v = b.get_height()
            ax_i.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=7.2, fontweight="bold", rotation=90)
    fig.legend(handles_f6, labels_f6, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=True, fontsize=9.2)
    fig.suptitle("Cross-Split Consistency Between Validation Set and Final Held-Out Test Set", fontweight="bold", fontsize=12, y=1.08)
    save_fig(fig, "fig6_validation_vs_test")

    # --- FIGURE 7: Cascade Error Breakdown (Confusion Matrix + Factorial Ablation) ---
    fig, (ax_cm, ax_ab) = plt.subplots(1, 2, figsize=(14.4, 5.2), gridspec_kw={"width_ratios": [1.3, 1.0], "wspace": 0.48})
    mat_counts = np.array([[cm_data[g][c] for c in cls_cols] for g in CLASS_ORDER], dtype=float)
    row_sums = mat_counts.sum(axis=1, keepdims=True)
    mat_pct = np.divide(mat_counts * 100.0, row_sums, out=np.zeros_like(mat_counts), where=row_sums > 0)

    ax_cm.imshow(mat_pct, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax_cm.set_xticks(np.arange(len(cls_cols)))
    ax_cm.set_xticklabels([c + ("*" if c in oos_cols else "") for c in cls_cols], rotation=35, ha="right")
    ax_cm.set_yticks(np.arange(len(CLASS_ORDER)))
    ax_cm.set_yticklabels([f"{g} (n={int(row_sums[i,0])})" for i, g in enumerate(CLASS_ORDER)])
    ax_cm.set_xlabel("DINOv2 Predicted Class (* = Out-of-Scope Dropped)")
    ax_cm.set_ylabel("Detection Ground-Truth Class")
    ax_cm.set_title("(a) GT vs. DINOv2 Confusion Matrix at IoU >= 0.5 (Validation)")
    ax_cm.grid(False)
    for i in range(len(CLASS_ORDER)):
        for j in range(len(cls_cols)):
            cnt = int(mat_counts[i, j])
            pct = mat_pct[i, j]
            if cnt > 0:
                txt_color = "white" if pct > 45 else "#111111"
                ax_cm.text(j, i, f"{cnt}\n({pct:.1f}%)", ha="center", va="center", color=txt_color, fontsize=7.2, fontweight="bold" if pct >= 10 else "normal")

    # Panel (b): Factorial Ablation on Validation
    ab_keys = [
        "1_detector_baseline",
        "2_score_fusion_only",
        "3_class_replace_inscope_fallback_oos_det_score",
        "4_class_replace_inscope_fallback_oos_score_fusion",
        "5_class_replace_drop_oos_det_score",
        "6_full_cascade",
    ]
    ab_labels = [
        "1. Detector + NMS\n(det_class, det_score)",
        "2. Score Fusion Only\n(det_class, det*cls)",
        "3. In-Scope Cls Only\n(fallback OOS, det_score)",
        "4. In-Scope Cls + Fusion\n(fallback OOS, det*cls)",
        "5. Cls + Drop OOS\n(drop OOS, det_score)",
        "6. Full Cascade + NMS\n(drop OOS, det*cls)",
    ]
    ab_vals = [val_err["diagnostic_factorial_ablation_nms070"][k]["mAP50"] for k in ab_keys]
    ab_colors = ["#1F4E78", "#4C78A8", "#F28E2B", "#E17C05", "#E15759", "#D62728"]
    y_pos = np.arange(len(ab_keys))
    bars_ab = ax_ab.barh(y_pos, ab_vals, height=0.58, color=ab_colors, edgecolor="#222222", linewidth=0.75, zorder=3)
    ax_ab.set_yticks(y_pos)
    ax_ab.set_yticklabels(ab_labels, fontsize=8.5)
    ax_ab.invert_yaxis()
    ax_ab.set_xlabel("Validation mAP@0.50")
    ax_ab.set_xlim(0, 0.90)
    ax_ab.set_title("(b) Factorial Ablation of Cascade Components (NMS = 0.70)")
    for b, v in zip(bars_ab, ab_vals):
        diff = v - ab_vals[0]
        diff_str = " (baseline)" if abs(diff) < 1e-6 else f" ({diff:+.4f})"
        ax_ab.text(v + 0.012, b.get_y() + b.get_height() / 2, f"{v:.4f}{diff_str}", va="center", ha="left", fontsize=8.0, fontweight="bold")
    save_fig(fig, "fig7_cascade_error_breakdown")


def select_and_render_qualitative_examples():
    coco = json.loads((TEST_DATA_DIR / "annotations_coco.json").read_text(encoding="utf-8"))
    cat_names = {c["id"]: c["name"] for c in coco["categories"]}
    img_by_id = {im["id"]: im for im in coco["images"]}
    gt_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        x, y, w, h = ann["bbox"]
        gt_by_img[ann["image_id"]].append({
            "bbox_xyxy": [x, y, x + w, y + h],
            "class_name": cat_names[ann["category_id"]],
            "category_id": ann["category_id"],
        })

    # Load raw predictions of the 4 configurations from evaluation_runs/final_test_four_way_01/
    raw_by_method = {}
    for m in METHOD_ORDER:
        raw_list = json.loads((TEST_DIR / m / "raw_predictions.json").read_text(encoding="utf-8"))
        by_img = defaultdict(list)
        for r in raw_list:
            by_img[r["image_id"]].append(r)
        raw_by_method[m] = by_img

    case1_candidates = []
    case2_candidates = []
    case3_candidates = []
    case4_candidates = []

    for iid, im in img_by_id.items():
        gts = gt_by_img[iid]
        if not (1 <= len(gts) <= 4):
            continue
        d_orig = [p for p in raw_by_method["detector_original"][iid] if p["score"] >= 0.30]
        d_nms = [p for p in raw_by_method["detector_nms"][iid] if p["score"] >= 0.30]
        c_nms = [p for p in raw_by_method["cascade_nms"][iid] if p["score"] >= 0.25]
        c_nms_inscope = [p for p in c_nms if p["category_id"] is not None]
        c_nms_oos = [p for p in c_nms if p["category_id"] is None]

        gt_classes = [g["class_name"] for g in gts]
        d_nms_classes = [p["class_name"] for p in d_nms]

        # Case 1: Duplicate suppression by NMS
        if len(d_nms) == len(gts) and sorted(d_nms_classes) == sorted(gt_classes) and len(d_orig) >= len(d_nms) + 2:
            case1_candidates.append((len(d_orig) - len(d_nms), iid))

        # Case 2: Both Detector+NMS and Cascade+NMS match GT exactly on >= 2 objects
        c_nms_in_classes = [("organic" if p["class_name"] == "biological" else p["class_name"]) for p in c_nms_inscope]
        if len(gts) >= 2 and len(d_nms) == len(gts) and len(c_nms_inscope) == len(gts) and len(c_nms_oos) == 0:
            if sorted(d_nms_classes) == sorted(gt_classes) and sorted(c_nms_in_classes) == sorted(gt_classes):
                case2_candidates.append((len(gts), iid))

        # Case 3: Paper GT -> Detector+NMS predicts paper, Cascade+NMS predicts cardboard (OOS)
        if "paper" in gt_classes and len(d_nms) == len(gts) and sorted(d_nms_classes) == sorted(gt_classes):
            if any(p["class_name"] == "cardboard" for p in c_nms_oos):
                case3_candidates.append((len(c_nms_oos), iid))

        # Case 4: Glass GT -> Detector+NMS predicts glass, Cascade+NMS predicts plastic
        if "glass" in gt_classes and len(d_nms) == len(gts) and sorted(d_nms_classes) == sorted(gt_classes):
            if any(p["class_name"] == "plastic" for p in c_nms_inscope):
                case4_candidates.append((len(gts), iid))

    case1_candidates.sort(reverse=True)
    case2_candidates.sort(reverse=True)
    case3_candidates.sort(reverse=True)
    case4_candidates.sort(reverse=True)

    chosen = [
        ("case1_nms_duplicate_suppression", case1_candidates[0][1], "Case A: NMS Suppresses Redundant Boxes"),
        ("case2_cascade_agreement", case2_candidates[0][1], "Case B: Detector & Cascade Agreement"),
        ("case3_paper_to_cardboard_oos", case3_candidates[0][1], "Case C: Taxonomy Mismatch (paper -> cardboard OOS)"),
        ("case4_glass_to_plastic_shift", case4_candidates[0][1], "Case D: In-Scope Misclassification (glass -> plastic)"),
    ]

    CLASS_PALETTE = {
        "glass": "#00A8E8",
        "metal": "#FF9F1C",
        "organic": "#2EC4B6",
        "biological": "#2EC4B6",
        "paper": "#FFD166",
        "plastic": "#EF476F",
        "OOS": "#9B5DE5",
    }

    def render_panel(base_img, boxes_info):
        canvas = base_img.copy().convert("RGB")
        W, H = canvas.size
        target_h = 340
        target_w = int(round(W * (target_h / H)))
        scale_x = target_w / W
        scale_y = target_h / H
        canvas = canvas.resize((target_w, target_h), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(canvas)
        for b in boxes_info:
            x1, y1, x2, y2 = b["bbox_xyxy"]
            rx1, ry1, rx2, ry2 = x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y
            color = b["color"]
            draw.rectangle([rx1, ry1, rx2, ry2], outline=color, width=3)
            label = b["label"]
            tw = max(48, len(label) * 6 + 6)
            th = 15
            ty1 = max(0, ry1 - th)
            draw.rectangle([rx1, ty1, min(target_w, rx1 + tw), ty1 + th], fill=color)
            draw.text((rx1 + 3, ty1 + 1), label, fill="#000000" if color in {"#FFD166", "#2EC4B6", "#00A8E8"} else "#FFFFFF")
        return canvas

    summary_records = []
    combined_rows = []
    for slug, iid, case_title in chosen:
        im_meta = img_by_id[iid]
        img_path = TEST_DATA_DIR / "images" / im_meta["file_name"]
        pil_img = Image.open(img_path).convert("RGB")

        gts = gt_by_img[iid]
        d_orig = [p for p in raw_by_method["detector_original"][iid] if p["score"] >= 0.30]
        d_nms = [p for p in raw_by_method["detector_nms"][iid] if p["score"] >= 0.30]
        c_orig = [p for p in raw_by_method["cascade_original"][iid] if p["score"] >= 0.25]
        c_nms = [p for p in raw_by_method["cascade_nms"][iid] if p["score"] >= 0.25]

        panels = [
            ("Ground Truth", [
                {"bbox_xyxy": g["bbox_xyxy"], "label": g["class_name"], "color": CLASS_PALETTE.get(g["class_name"], "#2EC4B6")}
                for g in gts
            ]),
            (f"detector_original (n={len(d_orig)})", [
                {"bbox_xyxy": p["bbox_xyxy"], "label": f"{p['class_name']} {p['score']:.2f}", "color": CLASS_PALETTE.get(p["class_name"], "#00A8E8")}
                for p in d_orig
            ]),
            (f"detector_nms (n={len(d_nms)})", [
                {"bbox_xyxy": p["bbox_xyxy"], "label": f"{p['class_name']} {p['score']:.2f}", "color": CLASS_PALETTE.get(p["class_name"], "#00A8E8")}
                for p in d_nms
            ]),
            (f"cascade_original (n={len([p for p in c_orig if p['category_id'] is not None])}, OOS={len([p for p in c_orig if p['category_id'] is None])})", [
                {
                    "bbox_xyxy": p["bbox_xyxy"],
                    "label": f"{p['class_name']}{'[OOS]' if p['category_id'] is None else ''} {p['score']:.2f}",
                    "color": CLASS_PALETTE["OOS"] if p["category_id"] is None else CLASS_PALETTE.get(p["class_name"], "#EF476F"),
                }
                for p in c_orig
            ]),
            (f"cascade_nms (n={len([p for p in c_nms if p['category_id'] is not None])}, OOS={len([p for p in c_nms if p['category_id'] is None])})", [
                {
                    "bbox_xyxy": p["bbox_xyxy"],
                    "label": f"{p['class_name']}{'[OOS]' if p['category_id'] is None else ''} {p['score']:.2f}",
                    "color": CLASS_PALETTE["OOS"] if p["category_id"] is None else CLASS_PALETTE.get(p["class_name"], "#EF476F"),
                }
                for p in c_nms
            ]),
        ]

        fig, axes = plt.subplots(1, 5, figsize=(16.5, 3.8))
        for ax, (ptitle, binfo) in zip(axes, panels):
            rendered = render_panel(pil_img, binfo)
            ax.imshow(rendered)
            ax.set_title(ptitle, fontsize=9.2, fontweight="bold")
            ax.axis("off")
        fig.suptitle(f"{case_title} — Test Image {im_meta['file_name']} (image_id = {iid})", fontsize=11, fontweight="bold", y=1.02)
        out_case_path = QUAL_DIR / f"{slug}_{im_meta['file_name'].rsplit('.', 1)[0]}.png"
        fig.savefig(out_case_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        combined_rows.append((case_title, im_meta["file_name"], iid, pil_img, panels))
        summary_records.append({
            "slug": slug,
            "case_title": case_title,
            "image_id": iid,
            "file_name": im_meta["file_name"],
            "output_file": str(out_case_path),
            "gt_boxes": gts,
            "detector_original_ge_030": d_orig,
            "detector_nms_ge_030": d_nms,
            "cascade_original_ge_025": c_orig,
            "cascade_nms_ge_025": c_nms,
        })

    # Combined 4x5 publication grid figure (fig8_qualitative_examples.png)
    fig, axes = plt.subplots(4, 5, figsize=(16.5, 13.2))
    for row_idx, (case_title, fname, iid, pil_img, panels) in enumerate(combined_rows):
        for col_idx, (ptitle, binfo) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            rendered = render_panel(pil_img, binfo)
            ax.imshow(rendered)
            if row_idx == 0:
                col_header = ["Ground Truth", "detector_original\n(thr >= 0.30)", "detector_nms\n(IoU=0.70, thr >= 0.30)", "cascade_original\n(thr >= 0.25)", "cascade_nms\n(IoU=0.70, thr >= 0.25)"][col_idx]
                ax.set_title(col_header, fontsize=9.5, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                short_row_title = case_title.split(":")[0]
                ax.set_ylabel(f"{short_row_title}\n({fname}, id={iid})", fontsize=9.2, fontweight="bold")
            else:
                ax.set_xlabel(ptitle.split("(", 1)[-1].rstrip(")") if "(" in ptitle else "", fontsize=8.0)
    fig.suptitle("Qualitative Comparison of Detector and Cascade Configurations on Held-Out Test Images\n(Purple [OOS] = Out-of-Scope DINOv2 prediction dropped during 5-class evaluation)", fontsize=12, fontweight="bold", y=1.01)
    save_fig(fig, "fig8_qualitative_examples")

    (QUAL_DIR / "qualitative_examples_metadata.json").write_text(
        json.dumps(summary_records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary_records


def main():
    test_metrics = json.loads((TEST_DIR / "metrics.json").read_text(encoding="utf-8"))
    val_report = json.loads((VAL_DIR / "validation_grid_search_report.json").read_text(encoding="utf-8"))
    val_err = json.loads((VAL_DIR / "cascade_error_analysis.json").read_text(encoding="utf-8"))

    generate_tables_and_charts(test_metrics, val_report, val_err)
    qual_records = select_and_render_qualitative_examples()
    print(json.dumps({
        "status": "ok",
        "figures_generated": sorted(p.name for p in FIG_DIR.iterdir()),
        "tables_generated": sorted(p.name for p in TAB_DIR.iterdir()),
        "qualitative_cases": [
            {"slug": r["slug"], "image_id": r["image_id"], "file_name": r["file_name"], "output_file": r["output_file"]}
            for r in qual_records
        ],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
