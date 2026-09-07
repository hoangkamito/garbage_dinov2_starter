import argparse
import random
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def collect_samples(raw_dir: Path):
    classes = sorted([p for p in raw_dir.iterdir() if p.is_dir()])
    if len(classes) < 2:
        raise ValueError(
            f"{raw_dir} không giống ImageFolder root. "
            "Thư mục này phải chứa trực tiếp các folder class."
        )

    samples, labels = [], []
    valid_classes = []

    for cls_dir in classes:
        imgs = [p for p in cls_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        if imgs:
            valid_classes.append(cls_dir.name)
            for p in imgs:
                samples.append(p)
                labels.append(cls_dir.name)

    if len(valid_classes) < 2:
        raise ValueError("Không tìm thấy đủ class có ảnh.")

    return samples, labels, valid_classes

def copy_split(paths, labels, out_dir: Path, split_name: str):
    for src, label in zip(paths, labels):
        dst_dir = out_dir / split_name / label
        dst_dir.mkdir(parents=True, exist_ok=True)

        dst = dst_dir / src.name
        if dst.exists():
            dst = dst_dir / f"{src.stem}_{abs(hash(str(src))) % 10_000_000}{src.suffix}"
        shutil.copy2(src, dst)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, required=True,
                        help="Folder chứa trực tiếp các thư mục class.")
    parser.add_argument("--out_dir", type=str, default="data_split")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio phải bằng 1.")

    random.seed(args.seed)

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)

    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"{out_dir} đã có dữ liệu. Dùng --overwrite nếu muốn tạo lại."
        )

    samples, labels, classes = collect_samples(raw_dir)

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        samples,
        labels,
        test_size=(1.0 - args.train_ratio),
        random_state=args.seed,
        stratify=labels,
    )

    val_fraction_of_temp = args.val_ratio / (args.val_ratio + args.test_ratio)

    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=(1.0 - val_fraction_of_temp),
        random_state=args.seed,
        stratify=temp_labels,
    )

    copy_split(train_paths, train_labels, out_dir, "train")
    copy_split(val_paths, val_labels, out_dir, "val")
    copy_split(test_paths, test_labels, out_dir, "test")

    print("Classes:", classes)
    print(f"Train: {len(train_paths)}")
    print(f"Val:   {len(val_paths)}")
    print(f"Test:  {len(test_paths)}")
    print(f"Saved to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
