"""Synthetic end-to-end tests: SHA-only selection, preservation and stale audit rejection."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "detection"))
from audit_dataset import check_coco, fingerprint, overlap_analysis, sha256_file
from prepare_clean_test import build_candidate, read_json, validate_audit


class CleanCandidateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name).resolve()
        self.dataset = self.project/"WasteDetect_Dataset"
        self.output = self.dataset/"test_clean_candidate"
        categories = [{"id": i, "name": name} for i, name in enumerate(["glass", "metal", "organic", "paper", "plastic"])]
        for split, colors in [("train", ["red"]), ("valid", ["blue"]),
                              ("test", ["red", "blue", "green", "green", "yellow"])]:
            root = self.dataset/split
            (root/"images").mkdir(parents=True)
            (root/"labels").mkdir()
            images, annotations = [], []
            for image_id, color in enumerate(colors, 1):
                filename = f"{image_id:02}.png"
                Image.new("RGB", (20, 20), color).save(root/"images"/filename)
                images.append({"id": image_id, "file_name": filename, "width": 20, "height": 20,
                               "license": 1, "custom_image_field": f"preserve-{image_id}"})
                annotations.append({"id": image_id+100, "image_id": image_id,
                                    "category_id": 1 if image_id == 4 else 0,
                                    "bbox": [1, 1, 18, 18], "area": 324, "iscrowd": 0,
                                    "segmentation": [], "custom_annotation_field": "preserve"})
                (root/"labels"/f"{image_id:02}.txt").write_text("0 0.5 0.5 0.9 0.9\n", encoding="utf-8")
            data = {"info": {"description": "synthetic"}, "licenses": [{"id": 1, "name": "fixture"}],
                    "categories": categories, "images": images, "annotations": annotations,
                    "custom_root_field": "must remain"}
            (root/"annotations_coco.json").write_text(json.dumps(data), encoding="utf-8")
        summaries, inventory = {}, []
        for split in ["train", "valid", "test"]:
            summaries[split], specs = check_coco(self.dataset/split, split, self.project)
            for spec in specs:
                spec["index"] = len(inventory)
                inventory.append(fingerprint(spec))
        self.report = {"schema_version": 1, "project_root": str(self.project), "detection_root": str(self.dataset),
                       "splits": summaries, "image_inventory": inventory,
                       "overlap": overlap_analysis(inventory, 6), "methods": {"phash_max_hamming_distance": 6},
                       "input_stability": {"changed_image_files": [], "annotation_files_unchanged": True},
                       "provenance": {"checkpoint_metadata": []}}
        self.audit_path = self.project/"audit.json"
        self.audit_path.write_text(json.dumps(self.report), encoding="utf-8")

    def run_build(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return build_candidate(self.project, self.dataset, self.audit_path, self.output, workers=2)

    def test_sha_only_selection_and_originals_unchanged(self):
        before = {str(p): sha256_file(p) for p in self.dataset.rglob("*") if p.is_file()}
        result = self.run_build()
        candidate = read_json(self.output/"annotations_coco.json")
        self.assertEqual([row["id"] for row in candidate["images"]], [3, 5])
        self.assertEqual([row["id"] for row in candidate["annotations"]], [103, 105])
        self.assertEqual(result["statistics"]["removed_images_by_primary_reason"],
                         {"sha256_overlap_train_valid": 2, "duplicate_sha256_within_test_candidate": 1})
        self.assertTrue(all(sha256_file(Path(p)) == digest for p, digest in before.items()))
        self.assertTrue(result["validation"]["original_guarded_files_unchanged"])
        self.assertEqual(result["validation"]["remaining_train_valid_exact_sha256_pairs"], 0)

    def test_preserves_categories_metadata_and_selected_annotations_exactly(self):
        source = read_json(self.dataset/"test"/"annotations_coco.json")
        result = self.run_build()
        saved = read_json(self.output/"annotations_coco.json")
        for key in ["categories", "info", "licenses", "custom_root_field"]:
            self.assertEqual(saved[key], source[key])
        for image in saved["images"]:
            self.assertIn(image, source["images"])
        for ann in saved["annotations"]:
            self.assertIn(ann, source["annotations"])
        self.assertEqual(result["statistics"]["duplicate_groups_with_differing_annotations"], 1)
        review = read_json(self.output/"annotation_duplicate_review.json")
        self.assertEqual(len(review), 1)
        self.assertEqual([record["image"]["id"] for record in review[0]["source_records"]], [3, 4])
        self.assertEqual(result["annotation_review"]["retained_groups_with_differing_annotations"], 1)
        self.assertEqual(result["validation"]["label_sidecars_copied"], 2)

    def test_phash_collisions_are_exported_and_never_auto_removed(self):
        result = self.run_build()
        self.assertEqual(result["statistics"]["retained"]["images"], 2)
        pairs = read_json(self.output/"phash_review_pairs.json")
        active = [p for p in pairs if p["actionable_after_sha_clean"]]
        self.assertTrue(active)
        self.assertTrue(any(p["left"]["group"] == p["right"]["group"] == "test" for p in active))
        self.assertTrue(all(p["manual_decision"] == "" for p in pairs))
        self.assertEqual(result["phash_review"]["pairs_removed_automatically"], 0)
        self.assertFalse(read_json(self.output/"test_manifest.json")["independent"])
        self.assertFalse(result["cocoeval_run"])

    def test_changed_source_image_rejected_before_output_creation(self):
        Image.new("RGB", (20, 20), "black").save(self.dataset/"train"/"images"/"01.png")
        with self.assertRaisesRegex(ValueError, "Stale audit image hash"):
            self.run_build()
        self.assertFalse(self.output.exists())

    def test_changed_annotation_rejected_before_output_creation(self):
        path = self.dataset/"test"/"annotations_coco.json"
        data = read_json(path)
        data["annotations"][0]["category_id"] = 2
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Stale audit"):
            self.run_build()
        self.assertFalse(self.output.exists())

    def test_malformed_or_inconsistent_audit_is_rejected(self):
        for mode in ["schema", "indices", "sha_groups", "phash_distance", "phash_missing"]:
            report = copy.deepcopy(self.report)
            if mode == "schema":
                report["schema_version"] = 2
            elif mode == "indices":
                report["image_inventory"][1]["index"] = report["image_inventory"][0]["index"]
            elif mode == "sha_groups":
                report["overlap"]["exact_sha256"]["duplicate_groups"][0]["sha256"] = "0"*64
            elif mode == "phash_distance":
                report["overlap"]["perceptual_hash"]["candidate_relations"][0]["hamming_distance"] = 65
            else:
                report["overlap"]["perceptual_hash"]["candidate_relations"] = []
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                validate_audit(report, self.project, self.dataset)

    def test_existing_destination_is_not_overwritten(self):
        self.output.mkdir()
        sentinel = self.output/"keep.txt"
        sentinel.write_text("unchanged", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.run_build()
        self.assertEqual(sentinel.read_text(), "unchanged")

    def test_new_unreported_reference_image_is_rejected(self):
        Image.new("RGB", (20, 20), "black").save(self.dataset/"valid"/"images"/"new.png")
        with self.assertRaisesRegex(ValueError, "file inventory differs"):
            self.run_build()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
