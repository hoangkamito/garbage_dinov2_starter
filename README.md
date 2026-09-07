# Garbage Classification with DINOv2

Starter project cho Garbage Classification V2.

## 1. Cài môi trường local

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

## 2. Chuẩn bị split

`raw_dir` phải chứa trực tiếp các folder class.

```bash
python prepare_split.py \
  --raw_dir /path/to/raw_dataset \
  --out_dir data_split \
  --overwrite
```

Output:

```text
data_split/
  train/<class>/*.jpg
  val/<class>/*.jpg
  test/<class>/*.jpg
```

## 3. Linear probing

```bash
python train.py \
  --data_dir data_split \
  --output_dir outputs/linear \
  --mode linear \
  --epochs 25 \
  --batch_size 32
```

## 4. Fine-tuning 2 block cuối

```bash
python train.py \
  --data_dir data_split \
  --output_dir outputs/finetune \
  --mode finetune \
  --unfreeze_blocks 2 \
  --epochs 20 \
  --batch_size 16 \
  --init_checkpoint outputs/linear/best.pt
```

## 5. Evaluate

```bash
python evaluate.py \
  --data_dir data_split \
  --checkpoint outputs/finetune/best.pt \
  --output_dir eval_results
```

## 6. Predict

```bash
python predict.py \
  --image sample.jpg \
  --checkpoint outputs/finetune/best.pt
```
