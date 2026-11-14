"""Synthetic validation of audit matching and read-only image fingerprinting."""
from collections import Counter
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "detection"))
from audit_dataset import HammingTree, box_issues, check_coco, fingerprint, overlap_analysis, phash


class HashTests(unittest.TestCase):
    def test_exact_radius_search_matches_brute_force(self):
        generator = random.Random(2026)
        values = [generator.getrandbits(64) for _ in range(100)]
        values += [v ^ 0b111 for v in values[:20]]
        tree = HammingTree()
        for value in values:
            tree.add(value)
        for query in values:
            self.assertEqual(set(tree.query(query, 6)),
                             {(v, (v ^ query).bit_count()) for v in set(values) if (v ^ query).bit_count() <= 6})

    def test_tree_handles_empty_and_boundary_distance(self):
        tree = HammingTree()
        self.assertEqual(list(tree.query(0, 6)), [])
        tree.add(0)
        tree.add(0)
        self.assertEqual(list(tree.query(63, 6)), [(0, 6)])
        self.assertEqual(list(tree.query(127, 6)), [])

    def test_resize_and_jpeg_candidates_do_not_modify_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = Image.new("RGB", (200, 120), "white")
            draw = ImageDraw.Draw(image)
            draw.rectangle((10, 20, 90, 110), fill="red")
            draw.ellipse((100, 0, 190, 85), fill="blue")
            image.save(root / "original.png")
            image.resize((100, 60), Image.Resampling.LANCZOS).save(root / "recompressed.jpg", quality=80)
            before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.iterdir()}
            records = [fingerprint({"path": str(root / name), "index": i, "group": group})
                       for i, (name, group) in enumerate([("original.png", "train"), ("recompressed.jpg", "test")])]
            self.assertNotEqual(records[0]["sha256"], records[1]["sha256"])
            self.assertLessEqual((int(records[0]["phash"], 16) ^ int(records[1]["phash"], 16)).bit_count(), 6)
            self.assertTrue(all(not row["changed_while_reading"] for row in records))
            self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.iterdir()})

    def test_group_counts_do_not_double_count_exact_pairs_as_near(self):
        rows = [{"index": i, "group": group, "sha256": digest, "phash": "0000000000000000"}
                for i, (group, digest) in enumerate([("train", "A"), ("train", "A"), ("test", "B"), ("valid", "B")])]
        result = overlap_analysis(rows, 6)
        self.assertEqual(result["exact_sha256"]["pair_counts"], {"train__train": 1, "test__valid": 1})
        self.assertEqual(result["perceptual_hash"]["candidate_pair_counts_excluding_identical_sha256"],
                         {"test__train": 2, "train__valid": 2})
        self.assertEqual(result["perceptual_hash"]["candidate_image_pair_count_excluding_identical_sha256"], 4)

    def test_overlap_index_matches_brute_force_file_pairs(self):
        rows = [{"index": i, "group": group, "sha256": str(i), "phash": f"{value:016x}"}
                for i, (group, value) in enumerate([("train", 0), ("test", 63), ("valid", 127), ("test", 63)])]
        expected = Counter()
        for i, left in enumerate(rows):
            for right in rows[i+1:]:
                if (int(left["phash"], 16) ^ int(right["phash"], 16)).bit_count() <= 6:
                    expected["__".join(sorted((left["group"], right["group"])))] += 1
        result = overlap_analysis(rows, 6)
        self.assertEqual(result["perceptual_hash"]["candidate_pair_counts_excluding_identical_sha256"], dict(expected))

    def test_corrupt_image_records_failure_instead_of_skipping(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.jpg"
            path.write_bytes(b"not an image")
            result = fingerprint({"path": str(path)})
            self.assertTrue(result["sha256"])
            self.assertTrue(result["decode_error"])


class AnnotationTests(unittest.TestCase):
    def test_invalid_geometry_and_unknown_class(self):
        image = {"width": 100, "height": 100}
        self.assertEqual(box_issues({"bbox": [0, 0, 100, 100], "category_id": 0}, image, {0}), [])
        for box, expected in [([0, 0, 0, 10], "nonpositive_bbox_size"),
                              ([-1, 0, 10, 10], "bbox_outside_declared_image"),
                              ([90, 0, 20, 10], "bbox_outside_declared_image"),
                              ([0, 0, float("nan"), 10], "invalid_bbox_format_or_nonfinite")]:
            with self.subTest(box=box):
                self.assertIn(expected, box_issues({"bbox": box, "category_id": 0}, image, {0}))
        self.assertIn("unknown_category_id", box_issues({"bbox": [0, 0, 10, 10], "category_id": 9}, image, {0}))

    def test_missing_references_and_duplicate_annotation_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            split = root / "train"
            (split / "images").mkdir(parents=True)
            data = {"categories": [{"id": i, "name": f"class{i}"} for i in range(5)],
                    "images": [{"id": 1, "file_name": "missing.jpg", "width": 100, "height": 100}],
                    "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [0, 0, 20, 20]}] * 2}
            path = split / "annotations_coco.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            original = path.read_bytes()
            report, specs = check_coco(split, "train", root)
            self.assertEqual(report["missing_image_reference_count"], 1)
            self.assertIn("invalid_or_duplicate_annotation_id", {r["reason"] for r in report["annotation_errors"]})
            self.assertEqual(specs, [])
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
