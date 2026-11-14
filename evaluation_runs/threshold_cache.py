"""CPU/disk prediction cache. Selections run on the real inference device."""
import json
import sqlite3
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from waste_cascade import canonical_name, select_proposals
from evaluate_cascade import as_coco_detection, sha256_file


def selection_key(proposal_threshold, max_proposals, nms_iou):
    return json.dumps([float(proposal_threshold), int(max_proposals), nms_iou], separators=(",", ":"))


class DiskCache:
    def __init__(self, path, fingerprint):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS images (id INTEGER PRIMARY KEY, sha TEXT NOT NULL, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS selections (image_id INTEGER, key TEXT, data TEXT NOT NULL, PRIMARY KEY(image_id,key));
            CREATE TABLE IF NOT EXISTS crops (image_id INTEGER, coords TEXT, probs TEXT NOT NULL, PRIMARY KEY(image_id,coords));
        """)
        serialized = json.dumps(fingerprint, sort_keys=True)
        row = self.db.execute("SELECT value FROM metadata WHERE key='fingerprint'").fetchone()
        if row is not None and row[0] != serialized:
            self.db.close()
            raise ValueError("Cache fingerprint differs (data/checkpoint/preprocessing/device/code). Use a new cache directory.")
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('fingerprint',?)", (serialized,))

    def close(self):
        self.db.close()

    def image(self, image_id):
        row = self.db.execute("SELECT sha,data FROM images WHERE id=?", (image_id,)).fetchone()
        return (row[0], json.loads(row[1])) if row else None

    def save_image(self, image_id, digest, raw):
        with self.db:
            self.db.execute("INSERT INTO images VALUES (?,?,?)", (image_id, digest, json.dumps(raw)))

    def selection(self, image_id, key):
        row = self.db.execute("SELECT data FROM selections WHERE image_id=? AND key=?", (image_id, key)).fetchone()
        return json.loads(row[0]) if row else None

    def save_selection(self, image_id, key, rows):
        with self.db:
            self.db.execute("INSERT INTO selections VALUES (?,?,?)", (image_id, key, json.dumps(rows)))

    def crops(self, image_id):
        return {tuple(json.loads(coords)): probs for coords, probs in
                ((c, json.loads(p)) for c, p in self.db.execute("SELECT coords,probs FROM crops WHERE image_id=?", (image_id,)))}

    def save_crops(self, image_id, predictions):
        with self.db:
            self.db.executemany("INSERT INTO crops VALUES (?,?,?)", [
                (image_id, json.dumps(list(coords)), json.dumps(probabilities)) for coords, probabilities in predictions.items()])


def select_raw(raw, device, proposal_threshold, max_proposals, nms_iou):
    scores = torch.tensor(raw["scores"], dtype=torch.float32, device=device)
    boxes = torch.tensor(raw["norm_boxes"], dtype=torch.float32, device=device).reshape(-1, 4)
    indices, pixel_boxes = select_proposals(boxes, scores, raw["size"], proposal_threshold, max_proposals, nms_iou)
    selected = []
    for query, box in zip(indices.tolist(), pixel_boxes.tolist()):
        coords = [round(v) for v in box]
        if coords[2] > coords[0] and coords[3] > coords[1]:
            selected.append({"query": query, "coords": coords})
    return selected


@torch.inference_mode()
def ensure_image(cache, model, image_metadata, path, nms_grid, *, proposal_threshold=0., max_proposals=100,
                 cls_batch_size=2, expected_sha=None):
    image_id = image_metadata["id"]
    digest = sha256_file(path)
    if expected_sha is not None and digest != expected_sha:
        raise ValueError(f"Validation image changed since data audit: {path}")
    stats = {"detector_inferences": 0, "classifier_batches": 0, "new_unique_crops": 0, "reused_unique_crops": 0, "oom_retries": 0}
    cached = cache.image(image_id)
    image = None
    if cached is not None:
        if cached[0] != digest:
            raise ValueError(f"Cached image content changed: {path}")
        raw = cached[1]
    else:
        with Image.open(path) as pil:
            image = pil.convert("RGB")
        if image.size != (image_metadata["width"], image_metadata["height"]):
            raise ValueError(f"Image/COCO dimensions mismatch: {path}")
        tensor = model.det_tf(image).unsqueeze(0).to(model.device)
        output = model.detector(tensor)
        scores, det_idx = output["pred_logits"].sigmoid()[0].max(-1)
        raw = {"scores": scores.cpu().tolist(), "det_idx": det_idx.cpu().tolist(),
               "norm_boxes": output["pred_boxes"][0].float().cpu().tolist(), "size": list(image.size)}
        del tensor, output, scores, det_idx
        cache.save_image(image_id, digest, raw)  # Detector survives an interrupted classifier pass.
        stats["detector_inferences"] += 1
    union = {}
    for iou in [None] + list(nms_grid):
        key = selection_key(proposal_threshold, max_proposals, iou)
        selected = cache.selection(image_id, key)
        if selected is None:
            selected = select_raw(raw, model.device, proposal_threshold, max_proposals, iou)
            cache.save_selection(image_id, key, selected)
        for item in selected:
            union.setdefault(tuple(item["coords"]), None)
    existing = cache.crops(image_id)
    missing = [coords for coords in union if coords not in existing]
    stats["reused_unique_crops"] = len(union)-len(missing)
    if missing and image is None:
        with Image.open(path) as pil:
            image = pil.convert("RGB")
    predictions = {}
    k, batch_size = 0, cls_batch_size
    while k < len(missing):
        coords_batch = missing[k:k+batch_size]
        x = None
        try:
            # CPU transforms only for the next small batch; never keep all crops on GPU.
            x = torch.stack([model.cls_tf(image.crop(coords)) for coords in coords_batch]).to(model.device)
            probabilities = model.classifier(x).softmax(-1).cpu().tolist()
        except torch.cuda.OutOfMemoryError:
            del x
            torch.cuda.empty_cache()
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size//2)
            stats["oom_retries"] += 1
            continue
        del x
        predictions.update(zip(coords_batch, probabilities))
        k += len(coords_batch)
        stats["classifier_batches"] += 1
    cache.save_crops(image_id, predictions)  # Atomic per image, no partial/corrupt vector cache.
    stats["new_unique_crops"] = len(predictions)
    return stats


def cascade_score(detector_score, classifier_confidence):
    # WasteCascade multiplies the Python det score by a CPU float32 tensor.
    return float(np.float32(detector_score) * np.float32(classifier_confidence))


def predictions_for_image(cache, image_id, cls_names, forward_map, index_map, *, classify,
                          nms_iou, proposal_threshold=0., max_proposals=100):
    raw = cache.image(image_id)[1]
    selected = cache.selection(image_id, selection_key(proposal_threshold, max_proposals, nms_iou))
    if selected is None:
        raise ValueError("Missing cached proposal selection")
    crop_probs = cache.crops(image_id) if classify else {}
    result = []
    for item in selected:
        query, coords = item["query"], item["coords"]
        detector_score = raw["scores"][query]
        if classify:
            probabilities = crop_probs[tuple(coords)]
            cls_index = int(np.argmax(probabilities))
            conf = probabilities[cls_index]
            name = cls_names[cls_index]
            internal = forward_map.get(canonical_name(name))
            row = {"bbox_xyxy": coords, "class_name": name, "category_id": None if internal is None else index_map[internal],
                   "score": cascade_score(detector_score, conf), "det_score": detector_score, "cls_score": conf}
        else:
            internal = raw["det_idx"][query]
            row = {"bbox_xyxy": coords, "category_id": index_map[internal], "score": detector_score}
        result.append(row)
    return result


def build_predictions(cache, image_ids, cls_names, forward_map, index_map, **settings):
    exported, outside = [], []
    for image_id in image_ids:
        for row in predictions_for_image(cache, image_id, cls_names, forward_map, index_map, **settings):
            coco = as_coco_detection(row, image_id)
            if coco is None:
                outside.append(dict(row, image_id=image_id))
            else:
                exported.append(coco)
    return exported, outside
