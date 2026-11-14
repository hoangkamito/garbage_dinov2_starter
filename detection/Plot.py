"""
Vẽ biểu đồ training/validation metrics TRỰC TIẾP từ 1 file checkpoint .pt
(dùng khi chưa có file training_history.json riêng — history đã được lưu kèm
bên trong checkpoint mỗi lần torch.save() trong quá trình train).

Bấm Run trực tiếp (không cần đối số):
    - Tự động tìm checkpoint mới nhất trong thư mục checkpoints_wastedetect/
      (ưu tiên best_model.pt, nếu không có thì lấy model_epochX.pt có epoch lớn nhất)
    - Tự động lưu biểu đồ vào thư mục plots/
    - Tự động xuất kèm training_history.json để dùng lại về sau

Vẫn có thể chạy tuỳ chỉnh qua dòng lệnh nếu muốn:
    python PlotValidation.py --checkpoint checkpoints_wastedetect/model_epoch2.pt
"""

import json
import argparse
import os
import glob
import torch
import matplotlib.pyplot as plt


DEFAULT_CHECKPOINT_DIR = "checkpoints_wastedetect"
DEFAULT_EXPORT_JSON = os.path.join(DEFAULT_CHECKPOINT_DIR, "training_history.json")


def find_latest_checkpoint(checkpoint_dir: str) -> str:
    """
    Tự động tìm checkpoint mới nhất trong thư mục, theo thứ tự ưu tiên:
    1. best_model.pt (nếu có)
    2. model_epochX.pt có epoch lớn nhất
    Trả về None nếu không tìm thấy file nào.
    """
    best_path = os.path.join(checkpoint_dir, "best_model.pt")
    if os.path.exists(best_path):
        return best_path

    epoch_files = glob.glob(os.path.join(checkpoint_dir, "model_epoch*.pt"))
    if not epoch_files:
        return None

    def epoch_num(path):
        name = os.path.basename(path)
        digits = "".join(ch for ch in name if ch.isdigit())
        return int(digits) if digits else -1

    epoch_files.sort(key=epoch_num, reverse=True)
    return epoch_files[0]


def plot_history(history: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # --- 1. Train loss / Val loss theo epoch ---
    if history.get("train_loss"):
        plt.figure(figsize=(8, 5))
        epochs_train = range(1, len(history["train_loss"]) + 1)
        plt.plot(epochs_train, history["train_loss"], label="Train loss", marker="o")

        if history.get("val_loss"):
            val_every = max(1, len(history["train_loss"]) // max(len(history["val_loss"]), 1))
            epochs_val = [i * val_every for i in range(1, len(history["val_loss"]) + 1)]
            epochs_val = epochs_val[:len(history["val_loss"])]
            plt.plot(epochs_val, history["val_loss"], label="Val loss", marker="s")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Train / Validation Loss")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=150)
        plt.close()
    else:
        print("⚠ Chưa đủ dữ liệu train_loss để vẽ (checkpoint có thể mới lưu ở epoch quá sớm).")

    # --- 2. mAP@50 và mAP@50:95 theo lần validation ---
    if history.get("mAP50"):
        plt.figure(figsize=(8, 5))
        x = range(1, len(history["mAP50"]) + 1)
        plt.plot(x, history["mAP50"], label="mAP@50", marker="o")
        plt.plot(x, history["mAP50_95"], label="mAP@50:95", marker="s")
        plt.xlabel("Lần validation")
        plt.ylabel("mAP")
        plt.title("mAP theo quá trình training")
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "map_curve.png"), dpi=150)
        plt.close()
    else:
        print("⚠ Chưa có mAP nào được ghi nhận (có thể chưa chạy đủ 1 lần validation --val_every).")

    # --- 3. Precision / Recall / F1 theo lần validation ---
    if history.get("precision50"):
        precision = history["precision50"]
        recall = history["recall50"]
        f1 = [2 * p * r / max(p + r, 1e-8) for p, r in zip(precision, recall)]

        plt.figure(figsize=(8, 5))
        x = range(1, len(precision) + 1)
        plt.plot(x, precision, label="Precision@50", marker="o")
        plt.plot(x, recall, label="Recall@50", marker="s")
        plt.plot(x, f1, label="F1-score@50", marker="^")
        plt.xlabel("Lần validation")
        plt.ylabel("Giá trị")
        plt.title("Precision / Recall / F1-score")
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "precision_recall_f1.png"), dpi=150)
        plt.close()

    print(f"Đã lưu biểu đồ (nếu đủ dữ liệu) vào thư mục: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None,
                         help=f"Đường dẫn file .pt. Nếu bỏ trống, tự tìm checkpoint mới nhất trong '{DEFAULT_CHECKPOINT_DIR}/'")
    parser.add_argument("--out_dir", default="plots")
    parser.add_argument("--export_json", default=DEFAULT_EXPORT_JSON,
                         help="Path lưu history ra .json. Mặc định: checkpoints_wastedetect/training_history.json")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint(DEFAULT_CHECKPOINT_DIR)
        if checkpoint_path is None:
            raise SystemExit(
                f"❌ Không tìm thấy checkpoint nào trong '{DEFAULT_CHECKPOINT_DIR}/'. "
                f"Hãy train ít nhất 1 epoch trước, hoặc truyền --checkpoint thủ công."
            )
        print(f"→ Tự động dùng checkpoint mới nhất: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    history = ckpt.get("history", {})

    if not history:
        print("❌ Checkpoint này không chứa key 'history'. Kiểm tra lại đúng file checkpoint chưa.")
    else:
        plot_history(history, args.out_dir)

        if args.export_json:
            os.makedirs(os.path.dirname(args.export_json), exist_ok=True)
            with open(args.export_json, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
            print(f"Đã xuất thêm file JSON: {args.export_json}")