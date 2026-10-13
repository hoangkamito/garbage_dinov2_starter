import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.model import DINOv2Classifier


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def main():
    # =========================
    # ARGUMENTS
    # =========================
    parser = argparse.ArgumentParser(
        description="DINOv2 Garbage Classification"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to input image"
    )

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/dinov2_garbage_final.pth",
        help="Path to trained checkpoint"
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="Number of predictions to display"
    )

    args = parser.parse_args()

    # =========================
    # DEVICE
    # =========================
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # =========================
    # CHECK PATHS
    # =========================
    checkpoint_path = Path(args.checkpoint)
    image_path = Path(args.image)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # =========================
    # LOAD CHECKPOINT
    # =========================
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    class_names = checkpoint["classes"]

    print("Classes:", class_names)

    # =========================
    # CREATE MODEL
    # =========================
    model = DINOv2Classifier(
        num_classes=len(class_names),
        model_name="facebook/dinov2-base",
        dropout=0.2
    )

    # Load fine-tuned weights
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # Inference only
    model.freeze_backbone()

    model = model.to(device)
    model.eval()

    print("Model loaded successfully.")

    # =========================
    # TRANSFORM
    # =========================
    transform = transforms.Compose([
        transforms.Resize(
            256,
            interpolation=transforms.InterpolationMode.BICUBIC
        ),

        transforms.CenterCrop(224),

        transforms.ToTensor(),

        transforms.Normalize(
            IMAGENET_MEAN,
            IMAGENET_STD
        )
    ])

    # =========================
    # LOAD IMAGE
    # =========================
    image = Image.open(
        image_path
    ).convert("RGB")

    x = transform(image)
    x = x.unsqueeze(0).to(device)

    # =========================
    # PREDICTION
    # =========================
    with torch.no_grad():

        logits = model(x)

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

    # =========================
    # TOP K
    # =========================
    k = min(
        args.topk,
        len(class_names)
    )

    values, indices = torch.topk(
        probabilities,
        k
    )

    print("\n===== PREDICTION =====")

    for rank, (score, idx) in enumerate(
        zip(values.tolist(), indices.tolist()),
        start=1
    ):

        print(
            f"{rank}. "
            f"{class_names[idx]:15s} "
            f"{score * 100:.2f}%"
        )

    # =========================
    # FINAL RESULT
    # =========================
    prediction = class_names[
        indices[0].item()
    ]

    confidence = values[0].item()

    print("\n===== FINAL RESULT =====")

    print(
        f"Prediction : {prediction}"
    )

    print(
        f"Confidence : {confidence * 100:.2f}%"
    )


if __name__ == "__main__":
    main()