import argparse

import torch
from PIL import Image
from torchvision import transforms

from model import DINOv2Classifier


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    class_to_idx = ckpt["class_to_idx"]
    classes = [None] * len(class_to_idx)
    for name, idx in class_to_idx.items():
        classes[idx] = name

    model = DINOv2Classifier(
        num_classes=len(classes),
        model_name=ckpt["model_name"],
        mode=ckpt.get("mode", "linear"),
        unfreeze_blocks=ckpt.get("unfreeze_blocks", 2),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    tfms = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    image = Image.open(args.image).convert("RGB")
    x = tfms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]

    k = min(args.topk, len(classes))
    values, indices = torch.topk(probs, k)

    for score, idx in zip(values.tolist(), indices.tolist()):
        print(f"{classes[idx]:20s} {score*100:.2f}%")

if __name__ == "__main__":
    main()
