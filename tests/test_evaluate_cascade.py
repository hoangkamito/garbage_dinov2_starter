"""Synthetic-only scientific-evaluation checks, using real COCOeval/NMS."""
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch
from PIL import Image
from pycocotools.coco import COCO

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "detection"))
import evaluate_cascade as evaluation
from waste_cascade import WasteCascade

CLASSES = ["battery", "biological", "cardboard", "clothes", "glass", "metal",
           "paper", "plastic", "shoes", "trash"]
CATEGORIES = {0: "glass", 1: "metal", 2: "organic", 3: "paper", 4: "plastic"}


def metadata(annotations=None):
    if annotations is None:
        annotations = [{"id": 1, "image_id": 1, "category_id": 3,
                        "bbox": [10, 10, 30, 30], "area": 900, "iscrowd": 0}]
    return {"images": [{"id": 1, "file_name": "sample.png", "width": 100, "height": 100}],
            "annotations": copy.deepcopy(annotations),
            "categories": [{"id": i, "name": name} for i, name in CATEGORIES.items()]}


def coco(data):
    with contextlib.redirect_stdout(io.StringIO()):
        result = COCO()
        result.dataset = copy.deepcopy(data)
        result.createIndex()
        return result


def prediction(score=.9, bbox=None, category_id=3):
    return {"image_id": 1, "category_id": category_id, "bbox": bbox or [10, 10, 30, 30], "score": score}


def label_policy():
    return evaluation.read_json(Path(__file__).resolve().parents[1] / "detection/class_mapping.eval.json")


class MappingTests(unittest.TestCase):
    def test_categories_preserve_external_ids(self):
        data = metadata()
        for row in data["categories"]:
            row["id"] += 10
        categories = evaluation.detector_categories(data)
        self.assertEqual(categories[13], "paper")
        with self.assertRaises(ValueError):
            evaluation.detector_index_map(categories)
        mapping = evaluation.detector_index_map(categories, {str(i): name for i, name in CATEGORIES.items()})
        self.assertEqual(mapping, {i: i+10 for i in range(5)})

    def test_duplicate_categories_and_invalid_index_mapping_rejected(self):
        data = metadata()
        data["categories"][1]["id"] = 0
        with self.assertRaises(ValueError):
            evaluation.detector_categories(data)
        with self.assertRaises(ValueError):
            evaluation.detector_index_map(CATEGORIES, {str(i): "paper" for i in range(5)})

    def test_cardboard_not_implicitly_paper(self):
        resolved = evaluation.compile_label_policy(label_policy(), CLASSES, CATEGORIES)
        self.assertIsNone(resolved["cardboard"])
        self.assertEqual(resolved["paper"], 3)
        self.assertEqual(resolved["biological"], 2)

    def test_many_to_one_requires_explicit_evidence(self):
        config = label_policy()
        config["mapping"]["cardboard"] = "paper"
        with self.assertRaises(ValueError):
            evaluation.compile_label_policy(config, CLASSES, CATEGORIES)
        config["evidence"]["cardboard"] = "Synthetic test fixture explicitly defines a merged category."
        resolved = evaluation.compile_label_policy(config, CLASSES, CATEGORIES)
        self.assertEqual(resolved["cardboard"], resolved["paper"])

    def test_missing_unknown_and_invalid_targets_rejected(self):
        for mode in ["missing", "extra", "target"]:
            config = label_policy()
            if mode == "missing":
                del config["mapping"]["battery"]
            elif mode == "extra":
                config["mapping"]["invented"] = None
            else:
                config["mapping"]["cardboard"] = "unknown_category"
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                evaluation.compile_label_policy(config, CLASSES, CATEGORIES)

    def test_constructor_accepts_forward_mapping_without_changing_transforms(self):
        resolved = evaluation.compile_label_policy(label_policy(), CLASSES, CATEGORIES)
        resolved["cardboard"] = 3  # Explicit many-to-one mapping already compiled by caller.
        with patch("waste_cascade.load_detector", return_value=(Mock(), 518, {})), \
             patch("waste_cascade.load_classifier", return_value=(Mock(), CLASSES)):
            original = WasteCascade("unused", "unused", CATEGORIES, "cpu", {"organic": "biological"})
            configured = WasteCascade("unused", "unused", CATEGORIES, "cpu", classifier_to_detector=resolved)
        self.assertEqual(configured.det_name_to_id["cardboard"], 3)
        self.assertEqual(configured.det_name_to_id["paper"], 3)
        self.assertEqual(repr(configured.cls_tf), repr(original.cls_tf))
        self.assertEqual(repr(configured.det_tf), repr(original.det_tf))


class MetricTests(unittest.TestCase):
    def test_perfect_prediction(self):
        result = evaluation.coco_metrics(coco(metadata()), [prediction()], [1])
        for key in ["mAP50", "mAP50_95", "AR100", "precision_iou50", "recall_iou50"]:
            self.assertAlmostEqual(result[key], 1., places=6)
        self.assertEqual((result["TP"], result["FP"], result["FN"]), (1, 0, 0))

    def test_duplicate_counts_as_false_positive(self):
        result = evaluation.coco_metrics(coco(metadata()), [prediction(.9), prediction(.8)], [1])
        self.assertEqual((result["TP"], result["FP"], result["FN"]), (1, 1, 0))
        self.assertEqual(result["precision_iou50"], .5)

    def test_pr_threshold_does_not_change_ap(self):
        predictions = [prediction(.9, [60, 60, 10, 10]), prediction(.8)]
        low = evaluation.coco_metrics(coco(metadata()), predictions, [1], .05)
        high = evaluation.coco_metrics(coco(metadata()), predictions, [1], .85)
        self.assertEqual(low["mAP50"], high["mAP50"])
        self.assertEqual(low["TP"], 1)
        self.assertEqual(high["TP"], 0)
        self.assertEqual(high["FN"], 1)

    def test_crowd_predictions_are_ignored(self):
        data = metadata()
        data["annotations"].append({"id": 2, "image_id": 1, "category_id": 3,
                                    "bbox": [60, 60, 30, 30], "area": 900, "iscrowd": 1})
        predictions = [prediction(), prediction(.8, [65, 65, 10, 10]), prediction(.7, [70, 70, 10, 10])]
        result = evaluation.coco_metrics(coco(data), predictions, [1])
        self.assertEqual((result["TP"], result["FP"], result["FN"]), (1, 0, 0))
        self.assertEqual(result["ignored_predictions"], 2)

    def test_empty_predictions_with_gt(self):
        result = evaluation.coco_metrics(coco(metadata()), [], [1])
        self.assertEqual(result["mAP50"], 0.)
        self.assertEqual(result["mAP50_95"], 0.)
        self.assertEqual(result["AR100"], 0.)
        self.assertEqual((result["TP"], result["FP"], result["FN"]), (0, 0, 1))

    def test_no_gt_is_undefined_instead_of_fake_accuracy(self):
        for predictions in [[], [prediction()]]:
            with self.subTest(predictions=predictions):
                result = evaluation.coco_metrics(coco(metadata([])), predictions, [1])
                self.assertIsNone(result["mAP50"])
                self.assertIsNone(result["recall_iou50"])
                self.assertEqual(result["status"], "no_valid_ground_truth")

    def test_no_images_is_explicit(self):
        result = evaluation.coco_metrics(coco(metadata([])), [], [])
        self.assertEqual(result["status"], "no_images")
        self.assertIsNone(result["mAP50"])

    def test_max_detections_100_applies_to_pr(self):
        predictions = [prediction(1. - i*.001) for i in range(101)]
        result = evaluation.coco_metrics(coco(metadata()), predictions, [1])
        self.assertEqual((result["TP"], result["FP"]), (1, 99))

    def test_wrong_category_is_fp_and_fn(self):
        result = evaluation.coco_metrics(coco(metadata()), [prediction(category_id=4)], [1])
        self.assertEqual((result["TP"], result["FP"], result["FN"]), (0, 1, 1))


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root / "heldout_images"
        self.images.mkdir()
        Image.new("RGB", (100, 100), "white").save(self.images / "sample.png")
        self.annotation_path = self.root / "test_coco.json"
        self.annotation_path.write_text(json.dumps(metadata()), encoding="utf-8")
        self.manifest_path = self.root / "manifest.json"
        self.manifest = {"split": "test", "independent": True, "description": "Synthetic test fixture only",
                         "independent_of": sorted(evaluation.INDEPENDENT_OF),
                         "test_coco": "test_coco.json", "test_images": "heldout_images", "references": []}

    def validate(self):
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        return evaluation.validate_test_split(self.manifest_path, self.annotation_path, self.images, metadata())

    def test_bound_manifest_records_declaration(self):
        result = self.validate()
        self.assertEqual(result["test_unique_image_sha256_count"], 1)
        self.assertIn("User-declared", result["independence_basis"])

    def test_missing_independence_and_wrong_binding_rejected(self):
        for field, value in [("independent", False), ("split", "train"),
                             ("independent_of", ["detector_train"]), ("test_coco", "another.json")]:
            original = copy.deepcopy(self.manifest)
            self.manifest[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate()
            self.manifest = original

    def test_renamed_training_image_content_overlap_is_rejected(self):
        ref_images = self.root / "reference_images"
        ref_images.mkdir()
        (ref_images / "renamed.png").write_bytes((self.images / "sample.png").read_bytes())
        ref = metadata()
        ref["images"][0]["file_name"] = "renamed.png"
        (self.root / "train_coco.json").write_text(json.dumps(ref), encoding="utf-8")
        self.manifest["references"] = [{"coco": "train_coco.json", "images": "reference_images"}]
        with self.assertRaisesRegex(ValueError, "content overlaps"):
            self.validate()

    def test_train_metadata_and_root_are_rejected(self):
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        data = metadata()
        data["info"] = {"split": "train"}
        with self.assertRaises(ValueError):
            evaluation.validate_test_split(self.manifest_path, self.annotation_path, self.images, data)
        train_root = self.root / "train"
        train_root.mkdir()
        self.manifest["test_images"] = "train"
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        with self.assertRaises(ValueError):
            evaluation.validate_test_split(self.manifest_path, self.annotation_path, train_root, metadata())

    def test_path_escape_and_empty_coco_rejected(self):
        with self.assertRaises(ValueError):
            evaluation.image_path(self.images, "../outside.png")
        data = metadata()
        data["images"] = []
        with self.assertRaises(ValueError):
            evaluation.validate_coco(data, CATEGORIES)

    def test_unverified_manifest_stops_before_checkpoint_loading(self):
        self.manifest["independent"] = False
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        argv = ["evaluate_cascade.py", "--det_ckpt", "unused.pt", "--cls_ckpt", "unused.pth",
                "--test_coco", str(self.annotation_path), "--test_images", str(self.images),
                "--test_manifest", str(self.manifest_path), "--class_mapping", "unused.json",
                "--out_dir", str(self.root / "results")]
        with patch.object(sys, "argv", argv), patch.object(evaluation.torch, "load") as load, \
             patch.object(evaluation, "WasteCascade") as factory, self.assertRaises(ValueError):
            evaluation.main()
        load.assert_not_called()
        factory.assert_not_called()
        self.assertFalse((self.root / "results").exists())


class FourExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        Image.new("RGB", (100, 100)).save(self.root / "sample.png")

    def pipeline(self, outside_scope=False):
        model = WasteCascade.__new__(WasteCascade)
        model.device = torch.device("cpu")
        model.det_categories = CATEGORIES
        model.cls_names = CLASSES
        model.det_name_to_id = evaluation.compile_label_policy(label_policy(), CLASSES, CATEGORIES)
        model.det_tf = Mock(return_value=torch.zeros(3, 14, 14))
        model.cls_tf = Mock(return_value=torch.zeros(3, 14, 14))
        probabilities = torch.full((1, 3, 5), .001)
        probabilities[0, :, 3] = torch.tensor([.9, .8, .7])
        model.detector = Mock(return_value={"pred_logits": torch.logit(probabilities),
                                           "pred_boxes": torch.tensor([[[.25, .25, .3, .3],
                                                                         [.25, .25, .3, .3],
                                                                         [.75, .75, .3, .3]]])})
        def classify(batch):
            logits = torch.full((len(batch), 10), -20.)
            logits[:, 2 if outside_scope else 6] = 20.
            return logits
        model.classifier = Mock(side_effect=classify)
        return model

    def run_four(self, model, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return evaluation.run_experiments(model, coco(metadata()), self.root,
                                               {i: i for i in range(5)}, warmup_images=0, **kwargs)

    def test_four_variants_nms_and_original_baselines(self):
        model = self.pipeline()
        records, summary = self.run_four(model)
        self.assertEqual([len(records[name]["predictions"]) for name in evaluation.EXPERIMENTS], [3, 2, 3, 2])
        self.assertEqual(summary["detector_original"]["proposal_counts"]["nms_removed"], 0)
        self.assertEqual(summary["detector_nms"]["proposal_counts"]["nms_removed"], 1)
        self.assertEqual(summary["cascade_original"]["metrics"]["FP"], 2)
        self.assertEqual(summary["cascade_nms"]["metrics"]["FP"], 1)
        # Only cascade variants crop/classify: original 3 + NMS 2.
        self.assertEqual(model.cls_tf.call_count, 5)
        self.assertEqual(model.classifier.call_count, 2)
        self.assertEqual(summary["detector_nms"]["latency_ms"]["classifier_model_mean"], 0.)

    def test_no_nms_skips_variants_without_faking_zero_metrics(self):
        records, summary = self.run_four(self.pipeline(), no_nms=True)
        self.assertEqual(len(records["detector_original"]["predictions"]), 3)
        self.assertEqual(len(records["cascade_original"]["predictions"]), 3)
        self.assertEqual(summary["cascade_nms"]["status"], "skipped_nms_disabled")
        self.assertIsNone(summary["cascade_nms"]["metrics"])

    def test_outside_scope_counted_without_removing_gt(self):
        records, summary = self.run_four(self.pipeline(outside_scope=True))
        self.assertEqual(summary["cascade_original"]["outside_scope_by_class"], {"cardboard": 3})
        self.assertEqual(summary["cascade_nms"]["outside_scope_dropped"], 2)
        self.assertEqual(summary["cascade_nms"]["metrics"]["FN"], 1)
        self.assertEqual(summary["cascade_nms"]["metrics"]["mAP50"], 0.)
        self.assertEqual(records["cascade_nms"]["predictions"], [])
        self.assertEqual(len(records["cascade_nms"]["out_of_scope"]), 2)

    def test_empty_predictions_in_all_four_variants(self):
        model = self.pipeline()
        _, summary = self.run_four(model, proposal_threshold=1.)
        for result in summary.values():
            self.assertEqual(result["exported_prediction_count"], 0)
            self.assertEqual(result["metrics"]["FN"], 1)
            self.assertEqual(result["metrics"]["mAP50"], 0.)
        model.classifier.assert_not_called()

    def test_output_four_sets_and_csv_is_serializable(self):
        records, summaries = self.run_four(self.pipeline(), no_nms=True)
        output = self.root / "results"
        report = {"experiments": summaries, "settings": {"proposal_threshold": 0.},
                  "label_policy": {"config": label_policy()}}
        evaluation.write_results(output, report, records)
        with (output / "summary.csv").open(encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[1]["mAP50"], "")
        for name in evaluation.EXPERIMENTS:
            for filename in ["predictions.json", "raw_predictions.json", "out_of_scope_predictions.json", "metrics.json"]:
                self.assertTrue((output / name / filename).is_file())
        with self.assertRaises(FileExistsError):
            evaluation.write_results(output, report, records)

    def test_external_ids_exported_without_renumbering(self):
        data = metadata()
        for row in data["categories"]:
            row["id"] += 10
        data["annotations"][0]["category_id"] += 10
        with contextlib.redirect_stdout(io.StringIO()):
            records, summaries = evaluation.run_experiments(
                self.pipeline(), coco(data), self.root, {i: i+10 for i in range(5)}, warmup_images=0)
        self.assertEqual(records["cascade_nms"]["predictions"][0]["category_id"], 13)
        self.assertEqual(summaries["cascade_nms"]["metrics"]["TP"], 1)

    def test_complete_cli_with_synthetic_models_and_data(self):
        annotation_path = self.root / "test_coco.json"
        annotation_path.write_text(json.dumps(metadata()), encoding="utf-8")
        manifest_path = self.root / "manifest.json"
        manifest_path.write_text(json.dumps({
            "split": "test", "independent": True, "description": "Synthetic fixture; not real model evaluation",
            "independent_of": sorted(evaluation.INDEPENDENT_OF),
            "test_coco": "test_coco.json", "test_images": ".", "references": [],
        }), encoding="utf-8")
        for filename in ["det.pt", "cls.pth"]:
            (self.root / filename).write_bytes(b"Synthetic checkpoint placeholder; no real weights")
        config_path = self.root / "policy.json"
        config_path.write_text(json.dumps(label_policy()), encoding="utf-8")
        output = self.root / "full_cli_results"
        argv = ["evaluate_cascade.py", "--det_ckpt", str(self.root / "det.pt"),
                "--cls_ckpt", str(self.root / "cls.pth"), "--test_coco", str(annotation_path),
                "--test_images", str(self.root), "--test_manifest", str(manifest_path),
                "--class_mapping", str(config_path), "--out_dir", str(output), "--warmup_images", "0"]
        with patch.object(sys, "argv", argv), \
             patch.object(evaluation.torch, "load", return_value={"classes": CLASSES}), \
             patch.object(evaluation, "WasteCascade", return_value=self.pipeline()) as factory, \
             contextlib.redirect_stdout(io.StringIO()):
            evaluation.main()
        report = evaluation.read_json(output / "metrics.json")
        self.assertEqual(len(report["experiments"]), 4)
        self.assertEqual(report["experiments"]["cascade_nms"]["outside_scope_dropped"], 0)
        self.assertIsNone(factory.call_args.kwargs["classifier_to_detector"]["cardboard"])
        self.assertEqual(report["settings"]["proposal_threshold"], 0.)
        self.assertEqual(len(report["inputs"]["det_ckpt"]["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
