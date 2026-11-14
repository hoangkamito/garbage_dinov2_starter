"""Offline NMS tests: no checkpoints, downloads, training or output images."""
import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "detection"))
import waste_cascade
from waste_cascade import WasteCascade, select_proposals


class ProposalTests(unittest.TestCase):
    def select(self, boxes, scores, **kwargs):
        return select_proposals(
            torch.tensor(boxes, dtype=torch.float64).reshape(-1, 4),
            torch.tensor(scores, dtype=torch.float64), (200, 100), **kwargs,
        )

    def test_duplicate_keeps_highest_score(self):
        indices, boxes = self.select([[.5, .5, .4, .4]] * 2, [.7, .9], nms_iou=.5)
        self.assertEqual(indices.tolist(), [1])
        self.assertEqual(boxes.dtype, torch.float32)
        torch.testing.assert_close(boxes, torch.tensor([[60., 30., 140., 70.]]))

    def test_partial_overlap_and_iou_threshold(self):
        boxes = [[.4, .5, .4, .4], [.5, .5, .4, .4]]  # IoU = 0.6
        self.assertEqual(self.select(boxes, [.9, .8], nms_iou=.5)[0].tolist(), [0])
        self.assertEqual(self.select(boxes, [.9, .8], nms_iou=.7)[0].tolist(), [0, 1])

    def test_disjoint_boxes_survive(self):
        indices, _ = self.select([[.2, .5, .2, .4], [.8, .5, .2, .4]],
                                 [.8, .9], nms_iou=.5)
        self.assertEqual(indices.tolist(), [1, 0])

    def test_nms_precedes_proposal_limit(self):
        indices, _ = self.select([[.2, .5, .2, .4]] * 2 + [[.8, .5, .2, .4]],
                                 [.9, .8, .7], max_proposals=2, nms_iou=.5)
        self.assertEqual(indices.tolist(), [0, 2])

    def test_disabled_nms_retains_duplicate_top_k(self):
        indices, _ = self.select([[.2, .5, .2, .4]] * 2 + [[.8, .5, .2, .4]],
                                 [.9, .8, .7], max_proposals=2, nms_iou=None)
        self.assertEqual(indices.tolist(), [0, 1])

    def test_score_filter_precedes_nms(self):
        with patch.object(waste_cascade, "nms", wraps=waste_cascade.nms) as op:
            indices, _ = self.select([[.5, .5, .4, .4]] * 2, [.9, .1],
                                     proposal_threshold=.5, nms_iou=.5)
        self.assertEqual(indices.tolist(), [0])
        self.assertEqual(op.call_args.args[0].shape, (1, 4))
        self.assertEqual(op.call_args.args[1].dtype, torch.float32)

    def test_clipping_before_nms_in_original_pixels(self):
        with patch.object(waste_cascade, "nms", wraps=waste_cascade.nms) as op:
            _, boxes = self.select([[0., .5, .4, 2.]], [.9], nms_iou=.5)
        expected = torch.tensor([[0., 0., 40., 100.]])
        torch.testing.assert_close(boxes, expected)
        torch.testing.assert_close(op.call_args.args[0], expected)

    def test_invalid_boxes_and_scores_are_removed(self):
        boxes = [[.5, .5, 0., .4], [.5, .5, -.1, .4], [2., .5, .1, .4],
                 [float("nan"), .5, .4, .4], [float("inf"), .5, .4, .4],
                 [.5, .5, .4, .4], [.5, .5, .4, .4], [.5, .5, .4, .4]]
        indices, _ = self.select(boxes, [.9] * 5 + [float("nan"), float("inf"), .8],
                                 nms_iou=.5)
        self.assertEqual(indices.tolist(), [7])

    def test_empty_candidates_do_not_call_nms(self):
        for boxes, scores, threshold in [([], [], .05), ([[.5, .5, .2, .2]], [.1], .5),
                                         ([[.5, .5, 0., .2]], [.9], .05)]:
            with self.subTest(boxes=boxes), patch.object(waste_cascade, "nms") as op:
                indices, xyxy = self.select(boxes, scores, proposal_threshold=threshold,
                                            nms_iou=.5)
                self.assertEqual(indices.numel(), 0)
                self.assertEqual(xyxy.shape, (0, 4))
                self.assertEqual(xyxy.dtype, torch.float32)
                op.assert_not_called()

    def test_zero_proposal_limit(self):
        indices, boxes = self.select([[.5, .5, .4, .4]], [.9],
                                     max_proposals=0, nms_iou=.5)
        self.assertEqual(indices.numel(), 0)
        self.assertEqual(boxes.shape, (0, 4))

    def test_invalid_iou_is_rejected(self):
        for value in [-.1, 1.1, float("nan")]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.select([], [], nms_iou=value)


class CascadeTests(unittest.TestCase):
    def make_pipeline(self, boxes=None):
        pipeline = WasteCascade.__new__(WasteCascade)
        pipeline.device = torch.device("cpu")
        pipeline.det_categories = {0: "paper", 1: "plastic"}
        pipeline.cls_names = ["paper", "plastic"] + [f"other{i}" for i in range(8)]
        pipeline.det_name_to_id = {"paper": 0, "plastic": 1}
        pipeline.det_tf = Mock(return_value=torch.zeros(3, 14, 14))
        pipeline.cls_tf = Mock(return_value=torch.zeros(3, 14, 14))
        # Duplicate boxes intentionally have DIFFERENT detector labels.
        probs = torch.tensor([[[.9, .1], [.1, .8], [.1, .7]]])
        if boxes is None:
            boxes = [[.2, .4, .2, .4], [.2, .4, .2, .4], [.8, .4, .2, .4]]
        pipeline.detector = Mock(return_value={"pred_logits": torch.logit(probs),
                                               "pred_boxes": torch.tensor([boxes])})
        pipeline.classifier = Mock(side_effect=lambda x: torch.zeros(len(x), 10))
        return pipeline

    def test_nms_before_crop_and_preserves_query_labels(self):
        pipeline = self.make_pipeline()
        result = pipeline.predict(Image.new("RGB", (100, 50)), max_proposals=2,
                                  cls_batch_size=1, nms_iou=.5)
        self.assertEqual(pipeline.cls_tf.call_count, 2)
        self.assertEqual(pipeline.classifier.call_count, 2)
        self.assertEqual([p["category_id"] for p in result["baseline"]], [0, 1])
        self.assertEqual([p["bbox_xyxy"] for p in result["baseline"]],
                         [[10, 10, 30, 30], [70, 10, 90, 30]])
        self.assertEqual(len(result["cascade"]), 2)
        self.assertAlmostEqual(result["cascade"][1]["det_score"], .7, places=5)

    def test_default_programmatic_api_preserves_no_nms_evaluation(self):
        pipeline = self.make_pipeline()
        result = pipeline.predict(Image.new("RGB", (100, 50)), max_proposals=2)
        self.assertEqual(len(result["cascade"]), 2)
        self.assertEqual(result["baseline"][0]["bbox_xyxy"],
                         result["baseline"][1]["bbox_xyxy"])

    def test_empty_pipeline_skips_crop_and_classifier(self):
        for mode in [None, .5]:
            for case in ["threshold", "invalid", "rounded"]:
                with self.subTest(nms_iou=mode, case=case):
                    boxes = None if case == "threshold" else (
                        [[.5, .5, 0., .4]] * 3 if case == "invalid" else
                        [[.5, .5, .00001, .00001]] * 3)
                    pipeline = self.make_pipeline(boxes)
                    result = pipeline.predict(Image.new("RGB", (100, 50)),
                                              proposal_threshold=1. if case == "threshold" else .05,
                                              nms_iou=mode)
                    self.assertEqual(result["baseline"], [])
                    self.assertEqual(result["cascade"], [])
                    pipeline.cls_tf.assert_not_called()
                    pipeline.classifier.assert_not_called()


class CliTests(unittest.TestCase):
    def run_cli(self, extra):
        argv = ["waste_cascade.py", "--det_ckpt", "unused.pt", "--cls_ckpt", "unused.pth",
                "--det_classes", "glass,metal,organic,paper,plastic", "--image", "unused.jpg"]
        with patch.object(sys, "argv", argv + extra), \
             patch.object(waste_cascade, "WasteCascade") as factory, \
             patch.object(waste_cascade.Image, "open", return_value=Image.new("RGB", (10, 10))), \
             patch.object(waste_cascade, "draw_predictions"), \
             contextlib.redirect_stdout(io.StringIO()):
            factory.return_value.predict.return_value = {
                "baseline": [], "cascade": [], "detector_ms": 0., "classifier_ms": 0.,
            }
            waste_cascade.main()
            return factory.return_value.predict.call_args

    def test_default_enables_nms(self):
        self.assertEqual(self.run_cli([]).kwargs["nms_iou"], .5)

    def test_custom_iou(self):
        self.assertEqual(self.run_cli(["--nms_iou", "0.7"]).kwargs["nms_iou"], .7)

    def test_disable_nms(self):
        self.assertIsNone(self.run_cli(["--no_nms"]).kwargs["nms_iou"])

    def test_invalid_iou_fails_before_model_loading(self):
        with patch.object(sys, "argv", ["waste_cascade.py", "--det_ckpt", "x", "--cls_ckpt", "x",
                                       "--image", "x", "--nms_iou", "1.5"]), \
             patch.object(waste_cascade, "WasteCascade") as factory, \
             contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            waste_cascade.main()
        self.assertEqual(error.exception.code, 2)
        factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
