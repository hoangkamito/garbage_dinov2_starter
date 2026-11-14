"""One COCOeval matching pass per prediction set, shared across PR thresholds."""
import contextlib
import copy
import io

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def metric_sweep(coco_gt, predictions, image_ids, thresholds):
    with contextlib.redirect_stdout(io.StringIO()):
        if predictions:
            dt = coco_gt.loadRes(copy.deepcopy(predictions))
        else:
            dt = COCO()
            dt.dataset = {"images": copy.deepcopy(coco_gt.dataset["images"]),
                          "categories": copy.deepcopy(coco_gt.dataset["categories"]), "annotations": []}
            dt.createIndex()
        evaluator = COCOeval(coco_gt, dt, "bbox")
        evaluator.params.imgIds = list(image_ids)
        evaluator.params.catIds = sorted(coco_gt.getCatIds())
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    stats = {key: float(evaluator.stats[i]) if evaluator.stats[i] >= 0 else None
             for key, i in [("mAP50_95", 0), ("mAP50", 1), ("AR100", 8)]}
    area = list(evaluator.params.areaRng[0])
    iou_index = int(np.flatnonzero(np.isclose(evaluator.params.iouThrs, .5))[0])
    scores, matched, ignored = [], [], []
    gt_count = 0
    for row in evaluator.evalImgs:
        if row is None or list(row["aRng"]) != area or row["maxDet"] != 100:
            continue
        scores.extend(row["dtScores"])
        matched.extend(row["dtMatches"][iou_index] > 0)
        ignored.extend(row["dtIgnore"][iou_index].astype(bool))
        gt_count += int(np.count_nonzero(row["gtIgnore"] == 0))
    scores, matched, ignored = np.asarray(scores), np.asarray(matched, bool), np.asarray(ignored, bool)
    sweep = []
    for threshold in thresholds:
        selected = scores >= threshold
        tp = int(np.count_nonzero(selected & matched & ~ignored))
        fp = int(np.count_nonzero(selected & ~matched & ~ignored))
        precision = tp/max(tp+fp, 1)
        recall = tp/gt_count if gt_count else None
        f1 = None if recall is None else (2*precision*recall/(precision+recall) if precision+recall else 0.)
        sweep.append({"pr_score_threshold": threshold, "precision_iou50": precision, "recall_iou50": recall,
                      "f1_iou50": f1, "TP": tp, "FP": fp, "FN": gt_count-tp,
                      "ignored_predictions": int(np.count_nonzero(selected & ignored))})
    return stats, sweep


def best_pr(sweep):
    valid = [row for row in sweep if row["f1_iou50"] is not None]
    if not valid:
        raise ValueError("No valid ground truth for threshold selection")
    return max(valid, key=lambda r: (r["f1_iou50"], r["precision_iou50"], -r["pr_score_threshold"]))
