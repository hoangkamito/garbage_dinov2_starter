import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from model import DINOv2Classifier


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loaders(data_dir, batch_size, num_workers):
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(
            224, scale=(0.75, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
        ),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = datasets.ImageFolder(Path(data_dir) / "train", transform=train_tfms)
    val_ds = datasets.ImageFolder(Path(data_dir) / "val", transform=eval_tfms)

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )

    return train_ds, val_ds, train_loader, val_loader


def run_epoch(model, loader, criterion, device, optimizer=None, linear_mode=False, use_amp=True):
    training = optimizer is not None

    if training:
        model.train()
        # Khi linear probing, backbone nên giữ ở eval mode.
        if linear_mode:
            model.backbone.eval()
    else:
        model.eval()

    total_loss = 0.0
    preds_all, targets_all = [], []

    scaler = torch.cuda.amp.GradScaler(enabled=(training and use_amp and device.type == "cuda"))

    context = torch.enable_grad if training else torch.no_grad
    with context():
        pbar = tqdm(loader, leave=False)

        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, targets)

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * images.size(0)

            preds = logits.argmax(dim=1)
            preds_all.extend(preds.detach().cpu().tolist())
            targets_all.extend(targets.detach().cpu().tolist())

            pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(targets_all, preds_all)
    macro_f1 = f1_score(targets_all, preds_all, average="macro")

    return avg_loss, acc, macro_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--model_name", type=str, default="dinov2_vits14")
    parser.add_argument("--mode", choices=["linear", "finetune"], default="linear")
    parser.add_argument("--unfreeze_blocks", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--head_lr", type=float, default=None)
    parser.add_argument("--backbone_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init_checkpoint", type=str, default=None,
                        help="Checkpoint dùng để khởi tạo, ví dụ best linear-probe checkpoint.")
    parser.add_argument("--no_amp", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, train_loader, val_loader = make_loaders(
        args.data_dir, args.batch_size, args.num_workers
    )

    print("Classes:", train_ds.classes)
    print("Train:", len(train_ds), "| Val:", len(val_ds))

    model = DINOv2Classifier(
        num_classes=len(train_ds.classes),
        model_name=args.model_name,
        mode=args.mode,
        unfreeze_blocks=args.unfreeze_blocks,
    )

    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model_state"], strict=True)
        print("Loaded init checkpoint:", args.init_checkpoint)

    model.to(device)

    if args.head_lr is None:
        head_lr = 1e-3 if args.mode == "linear" else 1e-4
    else:
        head_lr = args.head_lr

    if args.mode == "linear":
        optimizer = torch.optim.AdamW(
            model.head.parameters(),
            lr=head_lr,
            weight_decay=args.weight_decay,
        )
    else:
        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": args.backbone_lr},
                {"params": model.head.parameters(), "lr": head_lr},
            ],
            weight_decay=args.weight_decay,
        )

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1)
    )

    log_path = output_dir / "history.csv"
    best_f1 = -1.0
    bad_epochs = 0

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss", "train_acc", "train_macro_f1",
            "val_loss", "val_acc", "val_macro_f1",
        ])

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc, train_f1 = run_epoch(
                model, train_loader, criterion, device,
                optimizer=optimizer,
                linear_mode=(args.mode == "linear"),
                use_amp=not args.no_amp,
            )

            val_loss, val_acc, val_f1 = run_epoch(
                model, val_loader, criterion, device,
                optimizer=None,
                linear_mode=(args.mode == "linear"),
                use_amp=not args.no_amp,
            )

            scheduler.step()

            writer.writerow([
                epoch,
                train_loss, train_acc, train_f1,
                val_loss, val_acc, val_f1,
            ])
            f.flush()

            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"train loss={train_loss:.4f} acc={train_acc:.4f} f1={train_f1:.4f} | "
                f"val loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f}"
            )

            if val_f1 > best_f1:
                best_f1 = val_f1
                bad_epochs = 0

                checkpoint = {
                    "model_state": model.state_dict(),
                    "class_to_idx": train_ds.class_to_idx,
                    "model_name": args.model_name,
                    "mode": args.mode,
                    "unfreeze_blocks": args.unfreeze_blocks,
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "val_macro_f1": val_f1,
                }

                torch.save(checkpoint, output_dir / "best.pt")
                print("  -> saved best.pt")
            else:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    print("Early stopping.")
                    break

    print("Best validation Macro F1:", best_f1)
    print("Output:", output_dir.resolve())


if __name__ == "__main__":
    main()
