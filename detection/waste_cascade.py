"""Inference for frozen DINOv2 ViT-S/14 + Deformable DETR -> crop -> DINOv2 ViT-B/14 classifier.

This is designed for the files uploaded with this project:
  Train.py, best_model.pt, Copy_of_Trainning_DINOv2.ipynb's fine-tuned classifier checkpoint.

Run: python waste_cascade.py --det_ckpt best_model.pt \
    --cls_ckpt dinov2_base_finetuned_best.pth \
    --det_coco_json WasteDetect_Dataset/valid/annotations_coco.json \
    --image demo.jpg --out demo_cascade.jpg

Notes: Detection and classification scores are not calibrated. Product is an
experimental ranking heuristic. Use a held-out test COCO file for official mAP.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision import transforms as T
from torchvision.ops import nms

from Train import DINOv2DeformableDETR

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def canonical_name(name: str) -> str:
    return str(name).strip().lower().replace("_", " ").replace("-", " ")


def get_detector_categories(coco_path=None, class_names=None):
    if coco_path:
        with open(coco_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if not meta.get("categories"):
            raise ValueError("COCO JSON không có 'categories' với tên lớp.")
        cats = {int(c["id"]): c["name"] for c in meta["categories"]}
    elif class_names:
        cats = {i: name.strip() for i, name in enumerate(class_names.split(","))}
    else:
        raise ValueError("Truyền --det_coco_json hoặc --det_classes (theo đúng thứ tự index 0..4).")
    if sorted(cats) != list(range(len(cats))):
        raise ValueError(
            "Dataset code đang train trực tiếp trên COCO category_id 0..4; "
            f"nhận được ID={sorted(cats)}. Cần kiểm tra nhãn gốc, không tự đổi ID."
        )
    if len(cats) != 5:
        raise ValueError(f"Checkpoint Detection có 5 lớp, metadata cung cấp {len(cats)} lớp.")
    return cats


class DINOv2Classifier(torch.nn.Module):
    """Exact classifier network described in cell 23 of the supplied notebook."""

    def __init__(self, num_classes=10, model_name="facebook/dinov2-base", dropout=0.2):
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as e:
            raise ImportError("Cài transformers: pip install transformers") from e
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_size),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, num_classes),
        )
        self.backbone_frozen = False

    def forward(self, x):
        x = self.backbone(pixel_values=x).last_hidden_state[:, 0, :]
        return self.classifier(x)


def load_classifier(path: str, device: torch.device):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if "model_state_dict" not in ckpt or "classes" not in ckpt:
        raise ValueError("Checkpoint phân loại phải chứa model_state_dict và classes.")
    classes = list(ckpt["classes"])
    if len(classes) != 10:
        raise ValueError(f"Classifier notebook cần 10 lớp, nhận được {len(classes)} lớp.")
    model = DINOv2Classifier(num_classes=len(classes))
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, classes


def load_detector(path: str, device: torch.device):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    args = ckpt.get("args", {})
    state = ckpt["model_state_dict"]
    num_classes = int(args.get("num_classes", 5))
    num_queries = int(args.get("num_queries", 100))
    img_size = int(args.get("img_size", 518))
    if img_size % 14:
        raise ValueError("Kích thước detection phải chia hết cho 14.")
    model = DINOv2DeformableDETR(
        num_classes=num_classes,
        num_queries=num_queries,
        freeze_backbone=True,
    )
    status = model.load_state_dict(state, strict=False)
    # The uploaded checkpoint intentionally excludes the frozen Meta DINOv2 backbone.
    unexpected = status.unexpected_keys
    missing_critical = [k for k in status.missing_keys if not k.startswith("backbone.dinov2.")]
    if unexpected or missing_critical:
        raise RuntimeError(
            f"Checkpoint không khớp Train.py: unexpected={unexpected[:8]}, "
            f"missing_critical={missing_critical[:8]}"
        )
    model.to(device).eval()
    return model, img_size, ckpt


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def select_proposals(norm_boxes, scores, image_size, proposal_threshold=0.05,
                     max_proposals=100, nms_iou=None, return_stats=False):
    """Return query indices and float32 pixel xyxy boxes, before cropping.

    NMS is class-agnostic and ranks by detector score. None disables NMS;
    that path retains the original top-k-before-invalid-box-filter behavior.
    """
    if nms_iou is not None and not 0 <= nms_iou <= 1:
        raise ValueError("nms_iou phải nằm trong [0, 1] hoặc None để tắt NMS.")
    if max_proposals < 0:
        raise ValueError("max_proposals phải >= 0.")
    indices = torch.where(torch.isfinite(scores) & (scores >= proposal_threshold))[0]
    stats = {
        "queries_total": scores.numel(),
        "nonfinite_scores": int((~torch.isfinite(scores)).sum()),
        "below_threshold": int((torch.isfinite(scores) & (scores < proposal_threshold)).sum()),
        "threshold_passed": indices.numel(),
        "pre_limit_removed": 0, "invalid_boxes": 0,
        "nms_removed": 0, "post_limit_removed": 0,
    }
    indices = indices[torch.argsort(scores[indices], descending=True)]
    if nms_iou is None:
        stats["pre_limit_removed"] = max(0, indices.numel() - max_proposals)
        indices = indices[:max_proposals]

    boxes = norm_boxes[indices].to(dtype=torch.float32)
    centers, sizes = boxes[:, :2], boxes[:, 2:]
    boxes = torch.cat((centers - sizes / 2, centers + sizes / 2), dim=-1)
    width, height = image_size
    boxes = boxes * boxes.new_tensor([width, height, width, height])
    boxes[:, 0::2].clamp_(0, width)
    boxes[:, 1::2].clamp_(0, height)
    valid = (torch.isfinite(norm_boxes[indices]).all(dim=-1)
             & torch.isfinite(boxes).all(dim=-1)
             & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]))
    stats["invalid_boxes"] = int((~valid).sum())
    indices, boxes = indices[valid], boxes[valid]
    if nms_iou is not None and indices.numel():
        keep = nms(boxes, scores[indices].to(dtype=torch.float32), nms_iou)
        stats["nms_removed"] = indices.numel() - keep.numel()
        stats["post_limit_removed"] = max(0, keep.numel() - max_proposals)
        keep = keep[:max_proposals]
        indices, boxes = indices[keep], boxes[keep]
    if return_stats:
        return indices, boxes, stats
    return indices, boxes


class WasteCascade:
    def __init__(self, det_ckpt, cls_ckpt, det_categories, device=None, label_map=None,
                 classifier_to_detector=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.detector, self.det_img_size, self.det_ckpt = load_detector(det_ckpt, self.device)
        self.classifier, self.cls_names = load_classifier(cls_ckpt, self.device)
        self.det_categories = det_categories
        # Category name matching is required for fair COCO evaluation on the 5 classes.
        # Optional map: {"detection category name": "classification name"}.
        # Require ALL detection classes to be mapped; do not silently report incorrect mAP.
        if label_map is not None and classifier_to_detector is not None:
            raise ValueError("Không kết hợp label_map và classifier_to_detector.")
        label_map = {canonical_name(k): canonical_name(v) for k, v in (label_map or {}).items()}
        cls_names_set = {canonical_name(v) for v in self.cls_names}
        self.det_name_to_id = {}
        if classifier_to_detector is None:
            for cid, name in det_categories.items():
                det_name = canonical_name(name)
                cls_name = label_map.get(det_name, det_name)
                if cls_name not in cls_names_set:
                    raise ValueError(f"Lớp Detection '{name}' không tương ứng với lớp Classification nào. "
                                     "Hãy bổ sung --label_map JSON (tên_detection -> tên_classification).")
                if cls_name in self.det_name_to_id:
                    raise ValueError("Nhiều lớp Detection trùng một lớp Classification; mapping không một-một.")
                self.det_name_to_id[cls_name] = cid
        else:
            forward_map = {canonical_name(k): v for k, v in classifier_to_detector.items()}
            if set(forward_map) != cls_names_set:
                raise ValueError("classifier_to_detector phải cấu hình đủ các lớp classifier.")
            if any(v is not None and v not in det_categories for v in forward_map.values()):
                raise ValueError("classifier_to_detector chứa ID detector không hợp lệ.")
            self.det_name_to_id = forward_map
        self.det_tf = T.Compose([
            T.Resize((self.det_img_size, self.det_img_size)),
            T.ToTensor(), T.Normalize(NORM_MEAN, NORM_STD),
        ])
        self.cls_tf = T.Compose([
            T.Resize(256), T.CenterCrop(224),
            T.ToTensor(), T.Normalize(NORM_MEAN, NORM_STD),
        ])

    @torch.inference_mode()
    def predict(self, image: Image.Image, proposal_threshold=0.05, max_proposals=100,
                cls_batch_size=8, nms_iou=None, classify=True):
        # Keep programmatic callers (including evaluate_cascade.py) unchanged;
        # the image CLI explicitly enables NMS by default.
        image = image.convert("RGB")
        W, H = image.size
        image_tensor = self.det_tf(image).unsqueeze(0).to(self.device)

        synchronize(self.device)
        start = time.perf_counter()
        output = self.detector(image_tensor)
        synchronize(self.device)
        detector_ms = (time.perf_counter() - start) * 1000

        scores, det_idx = output["pred_logits"].sigmoid()[0].max(-1)
        norm_boxes = output["pred_boxes"][0]
        indices, boxes, proposal_stats = select_proposals(
            norm_boxes, scores, (W, H), proposal_threshold, max_proposals, nms_iou,
            return_stats=True,
        )
        proposals, crops = [], []
        proposal_stats["rounded_invalid"] = 0
        for q, box in zip(indices.tolist(), boxes.tolist()):
            coords = [round(v) for v in box]
            if coords[2] <= coords[0] or coords[3] <= coords[1]:
                proposal_stats["rounded_invalid"] += 1
                continue
            cid = int(det_idx[q])
            proposals.append({
                "bbox_xyxy": coords,
                "class_name": self.det_categories[cid],
                "category_id": cid,
                "score": float(scores[q]),
            })
            if classify:
                crops.append(self.cls_tf(image.crop(tuple(coords))))

        synchronize(self.device)
        start = time.perf_counter()
        if crops:
            batch_probs = []
            for k in range(0, len(crops), cls_batch_size):
                x = torch.stack(crops[k:k + cls_batch_size]).to(self.device)
                batch_probs.append(self.classifier(x).softmax(-1).cpu())
            class_probs = torch.cat(batch_probs, dim=0)
        else:
            class_probs = torch.empty(0, len(self.cls_names))
        synchronize(self.device)
        classifier_ms = (time.perf_counter() - start) * 1000

        cascade = []
        for i, det in enumerate(proposals if classify else []):
            conf, class_index = class_probs[i].max(-1)
            class_name = self.cls_names[int(class_index)]
            mapped_id = self.det_name_to_id.get(canonical_name(class_name))
            cascade.append({
                "bbox_xyxy": det["bbox_xyxy"],
                "class_name": class_name,
                "category_id": mapped_id,  # None if classifier predicts outside 5 detection classes
                "score": float(det["score"] * conf),  # heuristic, not calibrated probability
                "det_score": det["score"],
                "cls_score": float(conf),
            })
        proposal_stats["retained"] = len(proposals)
        return {"baseline": proposals, "cascade": cascade, "proposal_stats": proposal_stats,
                "detector_ms": detector_ms, "classifier_ms": classifier_ms if classify else 0.}


def draw_predictions(img: Image.Image, predictions, color="red"):
    image = img.copy().convert("RGB")
    draw = ImageDraw.Draw(image)
    for p in predictions:
        x1, y1, x2, y2 = p["bbox_xyxy"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1, max(0, y1 - 12)), f"{p['class_name']} {p['score']:.2f}", fill=color)
    return image


def main():
    parser = argparse.ArgumentParser(description="Cascade: Detection -> Crop -> Classification")
    parser.add_argument("--det_ckpt", required=True)
    parser.add_argument("--cls_ckpt", required=True)
    parser.add_argument("--det_coco_json", default=None, help="COCO annotations JSON chứa categories 0..4")
    parser.add_argument("--det_classes", default=None, help="Nếu không có COCO: tên 5 lớp theo ID, cách nhau bằng dấu phẩy")
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default="cascade_result.jpg")
    parser.add_argument("--score_threshold", type=float, default=0.05, help="Ngưỡng chọn proposal của detector")
    parser.add_argument("--nms_iou", type=float, default=0.5,
                        help="Ngưỡng IoU NMS không phân biệt lớp, trong [0, 1] (mặc định: 0.5)")
    parser.add_argument("--no_nms", action="store_true", help="Tắt NMS để so sánh với pipeline gốc")
    parser.add_argument("--max_proposals", type=int, default=100)
    parser.add_argument("--cls_batch_size", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--label_map", default=None, help="JSON map tên lớp Detection -> Classification nếu khác nhau")
    args = parser.parse_args()
    if not 0 <= args.nms_iou <= 1:
        parser.error("--nms_iou phải nằm trong [0, 1]")
    if args.max_proposals < 0:
        parser.error("--max_proposals phải >= 0")
    categories = get_detector_categories(args.det_coco_json, args.det_classes)
    aliases = json.loads(Path(args.label_map).read_text(encoding="utf-8")) if args.label_map else None
    predictor = WasteCascade(args.det_ckpt, args.cls_ckpt, categories, args.device, aliases)
    img = Image.open(args.image).convert("RGB")
    nms_iou = None if args.no_nms else args.nms_iou
    result = predictor.predict(img, args.score_threshold, args.max_proposals,
                               args.cls_batch_size, nms_iou=nms_iou)
    draw_predictions(img, result["cascade"]).save(args.out)
    print(json.dumps({"out": args.out, "n_baseline": len(result["baseline"]),
                      "nms_iou": nms_iou,
                      "n_cascade": len(result["cascade"]),
                      "detector_ms": result["detector_ms"],
                      "classifier_ms": result["classifier_ms"],
                      "predictions": result["cascade"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
