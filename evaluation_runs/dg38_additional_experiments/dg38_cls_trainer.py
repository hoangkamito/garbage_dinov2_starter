"""Unified Classification Trainer, Pilot Forecaster, and Evaluator for DG38 (Stages 3, 5, 6, 8).

Supported architectures:
- `resnet50`: ImageNet-pretrained ResNet-50 (`IMAGENET1K_V2`), Priority #1 Classification Baseline.
- `dinov2_vitb14`: `facebook/dinov2-base` (ViT-B/14, 768-dim) with exact notebook head
  `LayerNorm(768) -> Dropout(0.2) -> Linear(768, 10)`, supporting `unfreeze_blocks` in {0, 2, 4, 12}.
- `vit_b_16`: ImageNet-pretrained ViT-B/16 (`IMAGENET1K_V1`), Stage 8 optional expansion baseline.

Scientific Guarantees:
1. Uses the frozen `classification_train_val_manifest.json` (9,807 train / 1,226 val).
   Never re-splits the dataset across seeds (42, 123, 2026).
2. Selects the best model strictly by Validation Macro F1 (`val_macro_f1`).
3. Supports mixed precision (AMP BF16/FP16), gradient accumulation, and cross-GPU checkpoint resume.
4. Enforces the CU budget gate (`CUBudgetTracker.check_Experiment_gate`) before full training.
5. Never touches the Test split (`cls_test`, 1,226 images) during training or hyperparameter selection.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from cu_budget_tracker import CUBudgetTracker, PilotTimingMeasurement
from gpu_runtime_manager import (
    inspect_runtime,
    load_resumable_checkpoint,
    recommend_execution_config,
    save_resumable_checkpoint,
    save_run_hardware_log,
    seed_all_States,
)
from prepare_colab_packages import sha256_file


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_classification_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Exact transforms from `Copy_of_Trainning_DINOv2.ipynb` (Cells 12 & 13)."""
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_tf, val_tf


class ManifestClassificationDataset(Dataset):
    """Dataset bound to a fixed JSON split manifest (`cls_train`, `cls_val`, or `cls_test`)."""

    def __init__(
        self,
        data_root: str | Path,
        manifest_path: str | Path,
        split_name: str,
        transform: Optional[transforms.Compose] = None,
        verify_sha256_on_init: bool = False,
    ):
        self.data_root = Path(data_root)
        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if split_name not in self.manifest.get("splits", {}):
            raise ValueError(
                f"Split '{split_name}' không tồn tại trong manifest {self.manifest_path}. "
                f"Các split hiện có: {list(self.manifest.get('splits', {}))}"
            )
        self.split_name = split_name
        self.entries: List[Dict[str, Any]] = self.manifest["splits"][split_name]
        self.classes: List[str] = list(self.manifest["classes"])
        self.class_to_idx: Dict[str, int] = dict(self.manifest["class_to_idx"])
        self.split_signature_sha256: str = self.manifest["split_signature_sha256"]
        self.transform = transform

        # Verify all files exist (and optionally SHA-256)
        for item in self.entries:
            p = self.data_root / item["rel_path"]
            if not p.is_file():
                raise FileNotFoundError(f"Thiếu ảnh trong split {split_name}: {p}")
            if verify_sha256_on_init and "sha256" in item:
                actual = sha256_file(p)
                if actual != item["sha256"]:
                    raise ValueError(f"SHA-256 không khớp cho {p}: expected {item['sha256']}, got {actual}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        item = self.entries[index]
        img_path = self.data_root / item["rel_path"]
        with Image.open(img_path) as pil:
            img = pil.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(item["class_index"])


class DINOv2NotebookClassifier(nn.Module):
    """Exact DINOv2 ViT-B/14 architecture from `Copy_of_Trainning_DINOv2.ipynb` (Cell 23 & 45-53)."""

    def __init__(
        self,
        num_classes: int = 10,
        model_name: str = "facebook/dinov2-base",
        dropout: float = 0.2,
        unfreeze_blocks: int = 4,
    ):
        super().__init__()
        from transformers import AutoModel

        self.model_name = model_name
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = int(self.backbone.config.hidden_size)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )
        self.unfreeze_blocks = int(unfreeze_blocks)
        self.backbone_frozen = False
        self.configure_unfreeze_blocks(self.unfreeze_blocks)

    def configure_unfreeze_blocks(self, unfreeze_blocks: int) -> None:
        self.unfreeze_blocks = int(unfreeze_blocks)
        for param in self.backbone.parameters():
            param.requires_grad = False

        total_layers = len(self.backbone.encoder.layer)
        if self.unfreeze_blocks <= 0:
            self.backbone_frozen = True
        else:
            self.backbone_frozen = False
            n = min(self.unfreeze_blocks, total_layers)
            for block in self.backbone.encoder.layer[-n:]:
                for param in block.parameters():
                    param.requires_grad = True
            if hasattr(self.backbone, "layernorm"):
                for param in self.backbone.layernorm.parameters():
                    param.requires_grad = True
            if self.unfreeze_blocks >= total_layers and hasattr(self.backbone, "embeddings"):
                for param in self.backbone.embeddings.parameters():
                    param.requires_grad = True

        for param in self.classifier.parameters():
            param.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbone_frozen:
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone_frozen:
            self.backbone.eval()
            with torch.no_grad():
                outputs = self.backbone(pixel_values=x)
        else:
            outputs = self.backbone(pixel_values=x)
        features = outputs.last_hidden_state[:, 0, :]
        return self.classifier(features)


def build_classification_model(
    arch: str,
    num_classes: int = 10,
    dropout: float = 0.2,
    unfreeze_blocks: int = 4,
    pretrained: bool = True,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Construct ResNet50, DINOv2 ViT-B/14, or ViT-B/16 and return model + metadata."""
    arch_key = arch.lower().strip()
    if arch_key == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = models.resnet50(weights=weights)
        in_features = net.fc.in_features
        net.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        meta = {
            "arch": "resnet50",
            "pretrained_source": "ImageNet-1K (Torchvision ResNet50_Weights.IMAGENET1K_V2)" if pretrained else "scratch",
            "embedding_dim": in_features,
            "unfreeze_blocks": "all_cnn_layers",
            "protocol_comparison_note_vi": (
                "ResNet50 pretrained giám sát trên ImageNet-1K (25.56M tham số), "
                "khác với DINOv2 ViT-B/14 pretrained tự giám sát LVD-142M (86.59M tham số). "
                "Cả hai dùng đúng cùng manifest (9.807 train / 1.226 val), cùng phép biến đổi ảnh 224x224, "
                "cùng tổng ngân sách 15 epochs và quy tắc chọn checkpoint theo Validation Macro F1."
            ),
        }
    elif arch_key in ("dinov2_vitb14", "dinov2"):
        net = DINOv2NotebookClassifier(
            num_classes=num_classes,
            model_name="facebook/dinov2-base",
            dropout=dropout,
            unfreeze_blocks=unfreeze_blocks,
        )
        meta = {
            "arch": "dinov2_vitb14",
            "pretrained_source": "Meta DINOv2 LVD-142M (facebook/dinov2-base)",
            "embedding_dim": 768,
            "unfreeze_blocks": int(unfreeze_blocks),
            "protocol_comparison_note_vi": (
                f"DINOv2 ViT-B/14 với đầu phân loại LayerNorm(768)->Dropout({dropout})->Linear(768, {num_classes}), "
                f"mở khóa {unfreeze_blocks} block Transformer cuối."
            ),
        }
    elif arch_key == "vit_b_16":
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.vit_b_16(weights=weights)
        in_features = net.heads.head.in_features
        net.heads.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        meta = {
            "arch": "vit_b_16",
            "pretrained_source": "ImageNet-1K (Torchvision ViT_B_16_Weights.IMAGENET1K_V1)" if pretrained else "scratch",
            "embedding_dim": in_features,
            "unfreeze_blocks": 12,
            "protocol_comparison_note_vi": "Baseline mở rộng ViT-B/16 pretrained giám sát trên ImageNet-1K.",
        }
    else:
        raise ValueError(f"Kiến trúc không hỗ trợ: {arch}. Chọn trong ['resnet50', 'dinov2_vitb14', 'vit_b_16'].")

    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    meta["total_parameters"] = int(total_params)
    meta["trainable_parameters"] = int(trainable_params)
    meta["trainable_ratio_pct"] = round(100.0 * trainable_params / max(total_params, 1), 4)
    return net, meta


def build_optimizer_and_scheduler(
    model: nn.Module,
    arch: str,
    epochs: int,
    lr: float,
    backbone_lr: float,
    weight_decay: float,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    arch_key = arch.lower().strip()
    if arch_key in ("dinov2_vitb14", "dinov2") and isinstance(model, DINOv2NotebookClassifier):
        if model.unfreeze_blocks <= 0:
            optimizer = torch.optim.AdamW(
                model.classifier.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        else:
            backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                [
                    {"params": backbone_params, "lr": backbone_lr},
                    {"params": model.classifier.parameters(), "lr": lr},
                ],
                weight_decay=weight_decay,
            )
    elif arch_key == "resnet50":
        # Differential learning rate for pretrained CNN backbone vs fresh classification head
        head_params = list(model.fc.parameters())
        head_ids = {id(p) for p in head_params}
        backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": backbone_lr},
                {"params": head_params, "lr": lr},
            ],
            weight_decay=weight_decay,
        )
    else:
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=backbone_lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    return optimizer, scheduler


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    grad_accum_steps: int = 1,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.float16,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_samples = 0
    all_preds: List[int] = []
    all_targets: List[int] = []
    batches_run = 0

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    if is_train:
        optimizer.zero_grad(set_to_none=True)

    for step, (images, targets) in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        images = images.to(device, non_blocking=(device.type == "cuda"))
        targets = targets.to(device, non_blocking=(device.type == "cuda"))

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=(use_amp and device.type == "cuda"),
            ):
                logits = model(images)
                raw_loss = criterion(logits, targets)
                loss = raw_loss / max(grad_accum_steps, 1)

            if is_train:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
                    if scaler is not None and scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        bs = int(images.size(0))
        total_loss += float(raw_loss.item()) * bs
        total_samples += bs
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_targets.extend(targets.detach().cpu().tolist())
        batches_run += 1

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    avg_loss = total_loss / max(total_samples, 1)
    acc = float(accuracy_score(all_targets, all_preds)) if all_targets else 0.0
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0)) if all_targets else 0.0
    return {
        "loss": round(avg_loss, 6),
        "accuracy": round(acc, 6),
        "macro_f1": round(macro_f1, 6),
        "samples": total_samples,
        "batches": batches_run,
        "elapsed_sec": round(elapsed, 4),
        "preds": all_preds,
        "targets": all_targets,
    }


def compute_detailed_classification_metrics(
    targets: List[int],
    preds: List[int],
    class_names: List[str],
    out_dir: Optional[str | Path] = None,
    prefix: str = "val",
) -> Dict[str, Any]:
    """Compute Accuracy, Macro P/R/F1, Weighted F1, Per-class F1, and Confusion Matrix."""
    acc = float(accuracy_score(targets, preds))
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        targets, preds, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        targets, preds, average="weighted", zero_division=0
    )
    per_p, per_r, per_f1, support = precision_recall_fscore_support(
        targets, preds, labels=list(range(len(class_names))), average=None, zero_division=0
    )
    cm = confusion_matrix(targets, preds, labels=list(range(len(class_names))))

    per_class = {}
    for idx, cname in enumerate(class_names):
        per_class[cname] = {
            "precision": round(float(per_p[idx]), 6),
            "recall": round(float(per_r[idx]), 6),
            "f1": round(float(per_f1[idx]), 6),
            "support": int(support[idx]),
        }

    result = {
        "split": prefix,
        "total_samples": len(targets),
        "accuracy": round(acc, 6),
        "macro_precision": round(float(macro_p), 6),
        "macro_recall": round(float(macro_r), 6),
        "macro_f1": round(float(macro_f1), 6),
        "weighted_precision": round(float(weighted_p), 6),
        "weighted_recall": round(float(weighted_r), 6),
        "weighted_f1": round(float(weighted_f1), 6),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{prefix}_metrics.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        report_txt = classification_report(
            targets, preds, target_names=class_names, digits=4, zero_division=0
        )
        (out / f"{prefix}_classification_report.txt").write_text(
            f"Accuracy: {acc:.6f}\nMacro F1: {macro_f1:.6f}\nWeighted F1: {weighted_f1:.6f}\n\n{report_txt}",
            encoding="utf-8",
        )
        # Save Confusion Matrix CSV & PNG
        with (out / f"{prefix}_confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["true\\pred"] + class_names)
            for cname, row in zip(class_names, cm.tolist()):
                writer.writerow([cname] + row)

        fig, ax = plt.subplots(figsize=(9, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
        ax.set_title(f"Confusion Matrix ({prefix})")
        fig.tight_layout()
        fig.savefig(out / f"{prefix}_confusion_matrix.png", dpi=160)
        plt.close(fig)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Huấn luyện và đánh giá các mô hình Classification DG38.")
    parser.add_argument("--mode", choices=["smoke", "pilot", "train", "eval_val", "eval_test_posthoc"], required=True)
    parser.add_argument("--arch", choices=["resnet50", "dinov2_vitb14", "vit_b_16"], default="resnet50")
    parser.add_argument("--unfreeze-blocks", type=int, default=4, help="Số block mở khóa cho DINOv2 (0, 2, 4, 12)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (bắt buộc: 42, 123, 2026)")
    parser.add_argument("--data-root", required=True, help="Thư mục gốc chứa ảnh 10 lớp (standardized_256)")
    parser.add_argument(
        "--manifest",
        default=str(Path(__file__).resolve().parent / "manifests/classification_train_val_manifest.json"),
        help="Đường dẫn file manifest cố định",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--budget-category", default="resnet50_baseline")
    parser.add_argument("--cu-per-hour", type=float, default=None, help="Usage Rate (CU/giờ) thực tế trên Colab")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--effective-batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate cho classification head")
    parser.add_argument("--backbone-lr", type=float, default=1e-4, help="Learning rate cho backbone (ResNet50=1e-4, DINOv2=1e-5)")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pilot-batches", type=int, default=20, help="Số batch đo tốc độ trong chế độ pilot")
    parser.add_argument("--init-checkpoint", default=None, help="Khởi tạo từ checkpoint Stage 1 (ví dụ cho fine-tune ablation)")
    parser.add_argument("--resume", action="store_true", help="Tự động khôi phục từ last_checkpoint.pt nếu có")
    parser.add_argument("--force-retrain", action="store_true", help="Chạy lại kể cả khi đã có completed_run.json")
    parser.add_argument("--allow-reserve", action="store_true")
    parser.add_argument("--allow-posthoc-test", action="store_true", help="Bắt buộc khi chạy eval_test_posthoc")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or f"{args.arch}_blocks{args.unfreeze_blocks}_seed{args.seed}"

    # Guard against repeating an already completed run
    completed_marker = out_dir / "completed_run.json"
    if args.mode == "train" and completed_marker.exists() and not args.force_retrain:
        print(f"[SKIP] Run '{run_id}' đã hoàn thành hợp lệ tại {completed_marker}. Không tốn thêm GPU CU!")
        print(completed_marker.read_text(encoding="utf-8"))
        return

    seed_all_States(args.seed)
    task_key = "resnet50_classification" if args.arch == "resnet50" else "dinov2_vitb14_finetune"
    exec_cfg = recommend_execution_config(task_key, target_effective_batch_size=args.effective_batch_size)
    phys_bs = args.batch_size or int(exec_cfg["physical_batch_size"])
    eff_bs = args.effective_batch_size or int(exec_cfg["effective_batch_size"])
    grad_accum = max(1, eff_bs // phys_bs)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if exec_cfg["amp_dtype"] == "bfloat16" else torch.float16
    use_amp = bool(exec_cfg["use_amp"] and device.type == "cuda")

    train_tf, val_tf = get_classification_transforms()

    if args.mode == "eval_test_posthoc":
        if not args.allow_posthoc_test:
            raise PermissionError(
                "Đánh giá hậu nghiệm trên tập Test đòi hỏi cờ --allow-posthoc-test sau khi đã khóa cấu hình trên Validation."
            )
        test_ds = ManifestClassificationDataset(args.data_root, args.manifest, "cls_test", transform=val_tf)
        test_loader = DataLoader(test_ds, batch_size=phys_bs, shuffle=False, num_workers=exec_cfg["num_workers"])
        model, _ = build_classification_model(
            args.arch, num_classes=len(test_ds.classes), dropout=args.dropout, unfreeze_blocks=args.unfreeze_blocks, pretrained=False
        )
        best_ckpt = out_dir / "best_model.pt"
        load_resumable_checkpoint(best_ckpt, model=model, device=device)
        model.to(device).eval()
        epoch_res = run_one_epoch(model, test_loader, nn.CrossEntropyLoss(), device, use_amp=use_amp, amp_dtype=amp_dtype)
        metrics = compute_detailed_classification_metrics(
            epoch_res["targets"], epoch_res["preds"], test_ds.classes, out_dir=out_dir, prefix="posthoc_test"
        )
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        return

    train_ds = ManifestClassificationDataset(args.data_root, args.manifest, "cls_train", transform=train_tf)
    val_ds = ManifestClassificationDataset(args.data_root, args.manifest, "cls_val", transform=val_tf)

    # Seeded generator for reproducible DataLoader shuffling per seed
    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=phys_bs,
        shuffle=True,
        num_workers=exec_cfg["num_workers"],
        pin_memory=exec_cfg["pin_memory"],
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=phys_bs,
        shuffle=False,
        num_workers=exec_cfg["num_workers"],
        pin_memory=exec_cfg["pin_memory"],
    )

    model, model_meta = build_classification_model(
        args.arch,
        num_classes=len(train_ds.classes),
        dropout=args.dropout,
        unfreeze_blocks=args.unfreeze_blocks,
        pretrained=True,
    )
    if args.init_checkpoint:
        init_ckpt = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        state = init_ckpt.get("model_state_dict") or init_ckpt.get("model_state")
        model.load_state_dict(state, strict=True)
        model_meta["initialized_from_checkpoint"] = str(args.init_checkpoint)

    model.to(device)
    criterion = nn.CrossEntropyLoss()

    if args.arch in ("dinov2_vitb14", "dinov2") and args.backbone_lr == 1e-4:
        # Match original notebook's 1e-5 backbone LR for DINOv2 fine-tuning
        effective_backbone_lr = 1e-5
    else:
        effective_backbone_lr = args.backbone_lr

    optimizer, scheduler = build_optimizer_and_scheduler(
        model,
        arch=args.arch,
        epochs=args.epochs,
        lr=args.lr,
        backbone_lr=effective_backbone_lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    if args.mode == "smoke":
        tr_res = run_one_epoch(
            model, train_loader, criterion, device, optimizer=optimizer, scaler=scaler,
            grad_accum_steps=grad_accum, use_amp=use_amp, amp_dtype=amp_dtype, max_batches=2
        )
        va_res = run_one_epoch(
            model, val_loader, criterion, device, optimizer=None,
            use_amp=use_amp, amp_dtype=amp_dtype, max_batches=2
        )
        rt = inspect_runtime(device)
        smoke_out = {
            "status": "SMOKE_OK",
            "run_id": run_id,
            "model_meta": model_meta,
            "train_2batches_sec": tr_res["elapsed_sec"],
            "val_2batches_sec": va_res["elapsed_sec"],
            "peak_vram_mb": rt["peak_vram_allocated_mb"],
        }
        (out_dir / "smoke_result.json").write_text(json.dumps(smoke_out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(smoke_out, indent=2, ensure_ascii=False))
        return

    if args.mode == "pilot":
        if not args.cu_per_hour or args.cu_per_hour <= 0:
            raise ValueError("Chế độ pilot yêu cầu truyền --cu-per-hour <Usage Rate từ Colab UI>.")
        n_tr = min(args.pilot_batches, len(train_loader))
        n_va = min(max(5, args.pilot_batches // 2), len(val_loader))
        tr_res = run_one_epoch(
            model, train_loader, criterion, device, optimizer=optimizer, scaler=scaler,
            grad_accum_steps=grad_accum, use_amp=use_amp, amp_dtype=amp_dtype, max_batches=n_tr
        )
        va_res = run_one_epoch(
            model, val_loader, criterion, device, optimizer=None,
            use_amp=use_amp, amp_dtype=amp_dtype, max_batches=n_va
        )
        t_ckpt0 = time.perf_counter()
        save_resumable_checkpoint(
            out_dir / "pilot_ckpt_probe.pt",
            model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            epoch=0, best_val_metric=0.0, metric_name="val_macro_f1", seed=args.seed,
            config=vars(args), history={},
        )
        ckpt_sec = time.perf_counter() - t_ckpt0
        (out_dir / "pilot_ckpt_probe.pt").unlink(missing_ok=True)

        rt = inspect_runtime(device)
        pilot = PilotTimingMeasurement(
            gpu_name=rt["gpu_name"],
            observed_cu_per_hour=args.cu_per_hour,
            train_batches_measured=tr_res["batches"],
            train_time_sec=tr_res["elapsed_sec"],
            total_train_batches_per_epoch=len(train_loader),
            val_batches_measured=va_res["batches"],
            val_time_sec=va_res["elapsed_sec"],
            total_val_batches_per_epoch=len(val_loader),
            checkpoint_save_sec=max(ckpt_sec, 1.0),
        )
        forecast_1seed = pilot.forecast_run(epochs=args.epochs, num_seeds=1)
        forecast_3seeds = pilot.forecast_run(epochs=args.epochs, num_seeds=3)

        tracker = CUBudgetTracker(Path(__file__).resolve().parent / "cu_budget_ledger.json")
        gate_1seed = tracker.check_Experiment_gate(
            run_id=f"{run_id}_1seed",
            category=args.budget_category,
            forecast=forecast_1seed,
            allow_reserve_draw=args.allow_reserve,
        )
        pilot_report = {
            "run_id": run_id,
            "model_meta": model_meta,
            "hardware": rt,
            "forecast_1seed": forecast_1seed,
            "forecast_3seeds": forecast_3seeds,
            "budget_gate_1seed": gate_1seed,
        }
        (out_dir / "pilot_forecast.json").write_text(json.dumps(pilot_report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(tracker.format_budget_table_markdown())
        print(json.dumps(pilot_report, indent=2, ensure_ascii=False))
        return

    if args.mode == "eval_val":
        best_ckpt = out_dir / "best_model.pt"
        load_resumable_checkpoint(best_ckpt, model=model, device=device)
        model.to(device).eval()
        va_res = run_one_epoch(model, val_loader, criterion, device, use_amp=use_amp, amp_dtype=amp_dtype)
        metrics = compute_detailed_classification_metrics(
            va_res["targets"], va_res["preds"], val_ds.classes, out_dir=out_dir, prefix="val"
        )
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        return

    # Mode == "train"
    if not args.cu_per_hour or args.cu_per_hour <= 0:
        raise ValueError("Chế độ train yêu cầu --cu-per-hour <Usage Rate thực tế trên Colab UI> để kiểm soát ngân sách 200.45 CU.")

    start_epoch = 1
    best_val_f1 = -1.0
    history: Dict[str, List[Any]] = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "train_macro_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "epoch_sec": [],
        "lr": [],
    }

    last_ckpt_path = out_dir / "last_checkpoint.pt"
    best_ckpt_path = out_dir / "best_model.pt"
    if args.resume and last_ckpt_path.exists():
        ckpt = load_resumable_checkpoint(
            last_ckpt_path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, device=device
        )
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_f1 = float(ckpt.get("best_val_metric", -1.0))
        history = ckpt.get("history", history)
        print(f"[RESUME] Khôi phục từ epoch {ckpt['epoch']} (best_val_macro_f1={best_val_f1:.4f}) trên {device}")

    # If no pilot_forecast.json exists yet, run a quick 10-batch inline pilot gate before full epochs
    tracker = CUBudgetTracker(Path(__file__).resolve().parent / "cu_budget_ledger.json")
    remaining_epochs = max(0, args.epochs - start_epoch + 1)
    if remaining_epochs > 0 and not (out_dir / "pilot_forecast.json").exists():
        print("[BUDGET GATE] Đang đo 10 batches thử nghiệm để dự báo chi phí CU trước khi train đầy đủ...")
        probe_tr = run_one_epoch(
            model, train_loader, criterion, device, optimizer=None,
            use_amp=use_amp, amp_dtype=amp_dtype, max_batches=min(10, len(train_loader))
        )
        rt_probe = inspect_runtime(device)
        pilot = PilotTimingMeasurement(
            gpu_name=rt_probe["gpu_name"],
            observed_cu_per_hour=args.cu_per_hour,
            train_batches_measured=probe_tr["batches"],
            train_time_sec=probe_tr["elapsed_sec"] * 2.2,  # scale forward-only probe to full train+backward
            total_train_batches_per_epoch=len(train_loader),
            val_batches_measured=probe_tr["batches"],
            val_time_sec=probe_tr["elapsed_sec"],
            total_val_batches_per_epoch=len(val_loader),
        )
        fc = pilot.forecast_run(epochs=remaining_epochs, num_seeds=1)
        gate = tracker.check_Experiment_gate(
            run_id=run_id, category=args.budget_category, forecast=fc, allow_reserve_draw=args.allow_reserve
        )
        print(tracker.format_budget_table_markdown())
        if not gate["approved"]:
            raise RuntimeError(
                f"[BUDGET GATE BLOCKED] Dừng trước khi train đầy đủ! Lý do: {gate['reasons_blocked']} | "
                f"Phương án giảm chi phí: {gate['cost_reduction_recommendations']}"
            )

    run_config = {
        "run_id": run_id,
        "arch": args.arch,
        "unfreeze_blocks": args.unfreeze_blocks,
        "seed": args.seed,
        "epochs": args.epochs,
        "physical_batch_size": phys_bs,
        "grad_accum_steps": grad_accum,
        "effective_batch_size": phys_bs * grad_accum,
        "lr": args.lr,
        "backbone_lr": effective_backbone_lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "manifest_path": str(args.manifest),
        "manifest_split_signature_sha256": train_ds.split_signature_sha256,
        "model_meta": model_meta,
    }

    train_wall_t0 = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        ep_t0 = time.perf_counter()
        tr_res = run_one_epoch(
            model, train_loader, criterion, device,
            optimizer=optimizer, scaler=scaler, grad_accum_steps=grad_accum,
            use_amp=use_amp, amp_dtype=amp_dtype,
        )
        va_res = run_one_epoch(
            model, val_loader, criterion, device,
            optimizer=None, use_amp=use_amp, amp_dtype=amp_dtype,
        )
        current_lr = float(optimizer.param_groups[-1]["lr"])
        scheduler.step()
        ep_sec = time.perf_counter() - ep_t0

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_res["loss"])
        history["train_acc"].append(tr_res["accuracy"])
        history["train_macro_f1"].append(tr_res["macro_f1"])
        history["val_loss"].append(va_res["loss"])
        history["val_acc"].append(va_res["accuracy"])
        history["val_macro_f1"].append(va_res["macro_f1"])
        history["epoch_sec"].append(round(ep_sec, 2))
        history["lr"].append(current_lr)

        is_best = va_res["macro_f1"] > best_val_f1
        if is_best:
            best_val_f1 = va_res["macro_f1"]
            save_resumable_checkpoint(
                best_ckpt_path,
                model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                epoch=epoch, best_val_metric=best_val_f1, metric_name="val_macro_f1",
                seed=args.seed, config=run_config, history=history,
                extra_payload={
                    "classes": train_ds.classes,
                    "class_to_idx": train_ds.class_to_idx,
                    "val_acc": va_res["accuracy"],
                    "val_macro_f1": va_res["macro_f1"],
                },
            )

        save_resumable_checkpoint(
            last_ckpt_path,
            model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            epoch=epoch, best_val_metric=best_val_f1, metric_name="val_macro_f1",
            seed=args.seed, config=run_config, history=history,
            extra_payload={"classes": train_ds.classes, "class_to_idx": train_ds.class_to_idx},
        )

        print(
            f"[{run_id}] Epoch {epoch:02d}/{args.epochs} ({ep_sec:.1f}s) | "
            f"Train loss={tr_res['loss']:.4f} acc={tr_res['accuracy']:.4f} F1={tr_res['macro_f1']:.4f} | "
            f"Val loss={va_res['loss']:.4f} acc={va_res['accuracy']:.4f} F1={va_res['macro_f1']:.4f}"
            + (" ★ BEST" if is_best else ""),
            flush=True,
        )

    total_train_sec = time.perf_counter() - train_wall_t0

    # Evaluate best checkpoint on Validation for full report
    load_resumable_checkpoint(best_ckpt_path, model=model, device=device)
    model.to(device).eval()
    best_va_res = run_one_epoch(model, val_loader, criterion, device, use_amp=use_amp, amp_dtype=amp_dtype)
    val_metrics = compute_detailed_classification_metrics(
        best_va_res["targets"], best_va_res["preds"], val_ds.classes, out_dir=out_dir, prefix="val"
    )

    rt_final = inspect_runtime(device)
    save_run_hardware_log(out_dir, exec_cfg)
    usage_entry = tracker.record_actual_usage(
        run_id=run_id,
        category=args.budget_category,
        gpu_name=rt_final["gpu_name"],
        observed_cu_per_hour=args.cu_per_hour,
        elapsed_seconds=total_train_sec,
        status="COMPLETED",
        epochs_completed=args.epochs,
        best_val_metric=val_metrics["macro_f1"],
        checkpoint_path=str(best_ckpt_path),
    )

    summary_payload = {
        "status": "COMPLETED",
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_id": run_id,
        "config": run_config,
        "model_meta": model_meta,
        "training_time_seconds": round(total_train_sec, 2),
        "training_time_minutes": round(total_train_sec / 60.0, 2),
        "peak_vram_allocated_mb": rt_final["peak_vram_allocated_mb"],
        "peak_vram_reserved_mb": rt_final["peak_vram_reserved_mb"],
        "cu_usage": usage_entry,
        "best_validation_metrics": val_metrics,
        "history": history,
        "best_checkpoint_path": str(best_ckpt_path),
        "best_checkpoint_sha256": sha256_file(best_ckpt_path),
    }
    completed_marker.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary_payload["best_validation_metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
