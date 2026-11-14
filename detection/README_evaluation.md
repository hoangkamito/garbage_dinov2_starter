# Four-experiment evaluation

No independent test split is established by the current `annotations_coco.json`:
it contains 8,289 image records, 18,345 annotations and five categories, but no
split/provenance metadata. `test_images/carton1.jpg` is an inference example,
not a labelled independent detection test set. No real evaluation metrics have
been produced as part of this implementation.

## Controlled comparisons

The default run exports all four variants on exactly the same image IDs and GT:

| Directory | Classification | NMS before crop |
| --- | --- | --- |
| `detector_original` | off | off |
| `detector_nms` | off | on |
| `cascade_original` | on | off |
| `cascade_nms` | on | on |

NMS is class-agnostic and uses detector scores and original-image float32 xyxy
boxes, before the proposal cap and crop. The original path retains its original
top-k-before-invalid-filter behavior. `--no_nms` skips the two NMS variants;
their report rows say `skipped_nms_disabled`, not zero accuracy.

Each variant executes independently. Detector-only variants never transform
crops or run the classifier. All variants use the same checkpoint, score/cap
settings, and original classifier preprocessing (`Resize(256)`,
`CenterCrop(224)`, ImageNet normalization). No classifier interpolation changes
are introduced. The cascade score remains the uncalibrated product of detector
and classifier confidence.

## Labels

`categories` in the test COCO file supplies the five external IDs and names.
The current detector was trained directly on category IDs 0..4. A test file with
another ID scheme requires `--detector_class_map`: JSON with keys `"0"` through
`"4"`, values equal to COCO category names or integer IDs. Example for the current
class order: `{"0":"glass","1":"metal","2":"organic","3":"paper","4":"plastic"}`.
Do not guess this order for a different checkpoint.

`--class_mapping` is a **classifier -> detector** policy, separate from the old
inference `label_map.json` whose direction is detector -> classifier.
All checkpoint classifier labels must appear explicitly; use `null` for labels
outside the evaluation taxonomy. Non-identity mappings require a nonempty
`evidence` entry, whose scientific validity the dataset curator must establish.
Many-to-one mappings are supported when explicitly documented.

`class_mapping.eval.json` preserves exact-name matches and the existing configured
`biological -> organic` alias, recording its source and need for annotation-rule
review. It leaves `cardboard` outside scope; there is no inferred
`cardboard -> paper` rule. Out-of-scope predictions are exported and counted by
class and score threshold, rather than silently merged. All five-class GT stays
in evaluation: missed GT can therefore still be a false negative. No test images
or GT are removed because the classifier predicted an out-of-scope label.

## Independent test provenance

Copy `test_manifest.example.json` to a new manifest. It intentionally has
`independent: false`, so it cannot start a scientific run unchanged. Supply real
provenance for a held-out test independent of both models' train/validation data.
Manifest dataset paths and reference paths are relative to the manifest folder;
absolute paths are also accepted. Bind `test_coco` and `test_images` to the actual
evaluation paths. Only then set `independent` to true.

Optional `references` entries have `coco` and optionally `images`, for example:
`{"coco":"../train/annotations_coco.json","images":"../train/images"}`.
With image roots, SHA256 content overlap is rejected, even if files were renamed.
Without image roots, duplicate filenames are conservatively rejected. Equal
annotation files and train-labelled paths/metadata are rejected. COCO image IDs
alone are not used to infer overlap, since split exports often restart IDs.

The manifest is an explicit provenance declaration, **not proof** of independence.
References are audited when supplied; unprovided training/classifier source data
cannot be checked. Subject/session/source leakage must also be checked during
dataset curation. The report records this limitation and all performed audits.

## Metrics and latency

The official [COCOeval implementation](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)
computes AP50, AP averaged over IoU 0.50:0.05:0.95, and AR100 averaged over those
IoU thresholds. COCO uses a maximum of 100 detections per image/category.
Categories without non-ignored GT are excluded by COCO from the AP average;
undefined statistics are JSON null, never fabricated zero scores.

Precision/recall are **micro** values at IoU 0.50 and fixed score thresholds,
using COCOeval's raw matches (`area=all`, `maxDets=100`), including ignored crowd
matches. They are not means of interpolated AP entries. TP, FP, FN and ignored
detection counts are included. Zero precision denominator is reported as zero;
recall with no valid GT is null. Empty predictions against valid GT yield AP/AR
zero without crashing `COCO.loadRes([])`.

AP/AR use all exported scores; the PR threshold does not filter AP predictions.
The proposal threshold and cap still limit achievable recall. The default
proposal threshold is zero, retaining all finite queries before the cap/NMS.
Choose all thresholds and the label policy on validation, freeze them before
test, and compare original/NMS variants with the same values. Use separate fixed
PR thresholds for detector and cascade scores; those scores are not calibrated.

End-to-end timing is measured per variant with CUDA synchronization, includes
preprocessing, proposal selection/NMS, crop and classifier work inside `predict`,
and excludes file read/decode, model loading, metrics and warmup. The default
warms up on one image in all active variants; `--warmup_images 0` disables it.
Execution order rotates per test image. Reports include mean/p50/p95 end-to-end
latency and mean detector/classifier forward time. These timings are hardware
dependent and do not include model construction or energy/memory measurements.

## Outputs and tests

Use a fresh `--out_dir` for each run. Each of the four subdirectories contains
COCO `predictions.json`, `raw_predictions.json` (including out-of-scope labels),
`out_of_scope_predictions.json`, and `metrics.json`. Top-level `metrics.json`
records thresholds, policy, provenance, input/checkpoint hashes, software versions,
counts removed at each proposal stage, out-of-scope counts and latency.
`summary.csv` has four experiment rows. Skipped experiments have null metrics.
Below-PR-threshold counts are diagnostic: those predictions remain in AP exports.

Run the offline synthetic tests from the project root:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```

Synthetic test scores validate the evaluator; they are not model performance
results. No training, checkpoint changes or network downloads are required.

After an independent test and manifest exist:

```powershell
.\.venv\Scripts\python.exe -X utf8 detection\evaluate_cascade.py `
  --det_ckpt detection\checkpoints_wastedetect\best_model.pt `
  --cls_ckpt checkpoints\dinov2_garbage_final.pth `
  --test_coco "D:\WasteTestIndependent\annotations_coco.json" `
  --test_images "D:\WasteTestIndependent\images" `
  --test_manifest "D:\WasteTestIndependent\test_manifest.json" `
  --class_mapping detection\class_mapping.eval.json `
  --proposal_threshold 0.0 `
  --baseline_pr_threshold 0.05 `
  --cascade_pr_threshold 0.05 `
  --nms_iou 0.5 `
  --out_dir evaluation_runs\four_way_test_01
```

The shown thresholds are starting examples, not validation-optimized settings.
