import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from model import DINOv2Classifier


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    class_to_idx = ckpt["class_to_idx"]
    classes = [None] * len(class_to_idx)
    for name, idx in class_to_idx.items():
        classes[idx] = name

    tfms = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    test_ds = datasets.ImageFolder(Path(args.data_dir) / "test", transform=tfms)

    if test_ds.class_to_idx != class_to_idx:
        raise ValueError(
            "Mapping class của test set không giống checkpoint.\n"
            f"test={test_ds.class_to_idx}\nckpt={class_to_idx}"
        )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = DINOv2Classifier(
        num_classes=len(classes),
        model_name=ckpt["model_name"],
        mode=ckpt.get("mode", "linear"),
        unfreeze_blocks=ckpt.get("unfreeze_blocks", 2),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    preds_all, targets_all = [], []

    with torch.no_grad():
        for images, targets in tqdm(test_loader):
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu()

            preds_all.extend(preds.tolist())
            targets_all.extend(targets.tolist())

    acc = accuracy_score(targets_all, preds_all)
    macro_f1 = f1_score(targets_all, preds_all, average="macro")
    report = classification_report(
        targets_all,
        preds_all,
        target_names=classes,
        digits=4,
        zero_division=0,
    )

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Macro F1: {macro_f1:.4f}")
    print(report)

    (output_dir / "classification_report.txt").write_text(
        f"Accuracy: {acc:.6f}\nMacro F1: {macro_f1:.6f}\n\n{report}",
        encoding="utf-8",
    )

    cm = confusion_matrix(targets_all, preds_all)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, xticks_rotation=45, cmap=None, colorbar=False)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    print("Saved evaluation results to:", output_dir.resolve())


if __name__ == "__main__":
    main()
