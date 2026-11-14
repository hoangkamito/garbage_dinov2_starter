"""Dynamic GPU Runtime Inspector, Config Recommender & Cross-GPU Checkpoint Manager (Stage 2).

Supports NVIDIA L4, NVIDIA A100, NVIDIA G4 (and fallback GPUs/CPU for smoke tests).
Never claims Python can switch Colab hardware directly — the user selects the runtime
in the Google Colab UI. When switching GPUs across Colab sessions, this module restores
model, optimizer, scheduler, GradScaler, and RNG states while adapting physical
`batch_size` and `grad_accum_steps` to keep `effective_batch_size` invariant.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import platform
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


TASK_GPU_POLICY: Dict[str, Dict[str, Any]] = {
    "smoke_test": {
        "preferred_gpus": ["L4", "T4", "A100", "G4"],
        "recommended_primary": "NVIDIA L4",
        "rationale_vi": "Smoke test và kiểm tra môi trường chỉ cần GPU mức CU/giờ thấp (ưu tiên L4).",
        "target_effective_batch_size": 32,
    },
    "resnet50_classification": {
        "preferred_gpus": ["L4", "A100", "G4"],
        "recommended_primary": "NVIDIA L4",
        "rationale_vi": "ResNet50 (25.6M params, ảnh 224x224) chạy rất nhanh với AMP trên L4 (~24GB VRAM), tối ưu tỷ lệ hiệu năng/CU.",
        "target_effective_batch_size": 64,
    },
    "yolo11s_detection": {
        "preferred_gpus": ["L4", "A100", "G4"],
        "recommended_primary": "NVIDIA L4",
        "rationale_vi": "YOLO11s tối ưu rất tốt cho Tensor Cores thế hệ Ada (L4), VRAM < 10GB ở batch 16-32.",
        "target_effective_batch_size": 32,
    },
    "crop_preprocessing_ablation": {
        "preferred_gpus": ["L4", "A100", "G4"],
        "recommended_primary": "NVIDIA L4",
        "rationale_vi": "Chỉ chạy inference trên tập Validation (1.743 ảnh) với detector proposals đã cache; L4 thừa VRAM và tiết kiệm CU.",
        "target_effective_batch_size": 64,
    },
    "dinov2_vitb14_finetune": {
        "preferred_gpus": ["A100", "L4", "G4"],
        "recommended_primary": "NVIDIA A100 (hoặc L4 cho 0/2/4 blocks)",
        "rationale_vi": "DINOv2 ViT-B/14 (86.6M params) hưởng lợi từ băng thông bộ nhớ và BF16 của A100 khi chạy nhiều seeds hoặc 12 blocks; 0/2/4 blocks cũng chạy tốt trên L4.",
        "target_effective_batch_size": 32,
    },
    "deformable_detr_training": {
        "preferred_gpus": ["A100", "L4", "G4"],
        "recommended_primary": "NVIDIA A100",
        "rationale_vi": "Deformable DETR + DINOv2 ViT-S/14 (518x518, multi-scale grid_sample) nặng về tính toán và bộ nhớ; ưu tiên A100 để giảm thời gian mỗi epoch.",
        "target_effective_batch_size": 4,
    },
}


def classify_gpu_family(gpu_name: str) -> str:
    upper = (gpu_name or "").upper()
    if "A100" in upper:
        return "A100"
    if "L4" in upper and "L40" not in upper:
        return "L4"
    if "G4" in upper or "RTX PRO" in upper or "L40" in upper or "A10G" in upper:
        return "G4"
    if "T4" in upper:
        return "T4"
    if "V100" in upper:
        return "V100"
    if "3050" in upper:
        return "RTX3050_LOCAL"
    return "OTHER_GPU" if gpu_name and gpu_name != "CPU" else "CPU"


def inspect_runtime(device: Optional[str | torch.device] = None) -> Dict[str, Any]:
    """Inspect current hardware, CUDA, PyTorch, VRAM, and AMP capabilities."""
    cuda_available = torch.cuda.is_available()
    if device is None:
        dev = torch.device("cuda:0" if cuda_available else "cpu")
    else:
        dev = torch.device(device)

    if dev.type == "cuda" and cuda_available:
        idx = dev.index if dev.index is not None else 0
        props = torch.cuda.get_device_properties(idx)
        gpu_name = torch.cuda.get_device_name(idx)
        free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
        bf16_supported = bool(torch.cuda.is_bf16_supported())
        fp16_supported = props.major >= 7
        capability = f"{props.major}.{props.minor}"
        peak_alloc_mb = round(torch.cuda.max_memory_allocated(idx) / (1024 ** 2), 2)
        peak_reserved_mb = round(torch.cuda.max_memory_reserved(idx) / (1024 ** 2), 2)
    else:
        gpu_name = "CPU"
        total_bytes = 0
        free_bytes = 0
        bf16_supported = False
        fp16_supported = False
        capability = None
        peak_alloc_mb = 0.0
        peak_reserved_mb = 0.0

    family = classify_gpu_family(gpu_name)
    return {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if cuda_available else None,
        "device": str(dev),
        "gpu_name": gpu_name,
        "gpu_family": family,
        "compute_capability": capability,
        "total_vram_gb": round(total_bytes / (1024 ** 3), 2),
        "total_vram_mb": round(total_bytes / (1024 ** 2), 1),
        "available_vram_gb": round(free_bytes / (1024 ** 3), 2),
        "available_vram_mb": round(free_bytes / (1024 ** 2), 1),
        "bf16_supported": bf16_supported,
        "fp16_supported": fp16_supported,
        "recommended_amp_dtype": "bfloat16" if bf16_supported else ("float16" if fp16_supported else "float32"),
        "peak_vram_allocated_mb": peak_alloc_mb,
        "peak_vram_reserved_mb": peak_reserved_mb,
        "cpu_count": os.cpu_count() or 2,
    }


def recommend_execution_config(
    task_type: str,
    runtime_info: Optional[Dict[str, Any]] = None,
    target_effective_batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Recommend batch size, gradient accumulation, precision, and worker count for current GPU."""
    info = runtime_info or inspect_runtime()
    policy = TASK_GPU_POLICY.get(task_type, TASK_GPU_POLICY["resnet50_classification"])
    eff_bs = target_effective_batch_size or int(policy["target_effective_batch_size"])
    vram_gb = float(info.get("available_vram_gb") or info.get("total_vram_gb") or 0.0)
    family = info.get("gpu_family", "CPU")

    if task_type == "deformable_detr_training":
        if vram_gb >= 32.0:
            phys_bs = min(4, eff_bs)
        elif vram_gb >= 18.0:
            phys_bs = min(2, eff_bs)
        else:
            phys_bs = 1
    elif task_type == "dinov2_vitb14_finetune":
        if vram_gb >= 32.0:
            phys_bs = min(32, eff_bs)
        elif vram_gb >= 18.0:
            phys_bs = min(32, eff_bs)
        elif vram_gb >= 10.0:
            phys_bs = min(16, eff_bs)
        else:
            phys_bs = min(8, eff_bs)
    elif task_type == "yolo11s_detection":
        if vram_gb >= 32.0:
            phys_bs = min(32, eff_bs)
        elif vram_gb >= 18.0:
            phys_bs = min(16, eff_bs)
        else:
            phys_bs = min(8, eff_bs)
    else:  # resnet50_classification, crop_preprocessing_ablation, smoke_test
        if vram_gb >= 32.0:
            phys_bs = min(64, eff_bs)
        elif vram_gb >= 18.0:
            phys_bs = min(64, eff_bs)
        elif vram_gb >= 10.0:
            phys_bs = min(32, eff_bs)
        else:
            phys_bs = min(16, eff_bs)

    phys_bs = max(1, phys_bs)
    grad_accum = max(1, eff_bs // phys_bs)
    actual_eff_bs = phys_bs * grad_accum
    num_workers = min(4, max(2, (info.get("cpu_count") or 2) - 1)) if info["cuda_available"] else 0

    g4_warning = None
    if family == "G4":
        g4_warning = (
            "Đang chạy trên GPU nhóm G4: Theo quy tắc Giai đoạn 2, chỉ tiếp tục dùng G4 nếu "
            "L4/A100 thiếu VRAM hoặc kết quả pilot chứng minh chi phí CU/epoch của G4 thấp hơn."
        )

    return {
        "task_type": task_type,
        "detected_gpu_name": info["gpu_name"],
        "detected_gpu_family": family,
        "recommended_primary_gpu": policy["recommended_primary"],
        "policy_rationale_vi": policy["rationale_vi"],
        "g4_advisory_vi": g4_warning,
        "physical_batch_size": phys_bs,
        "grad_accum_steps": grad_accum,
        "effective_batch_size": actual_eff_bs,
        "use_amp": bool(info["bf16_supported"] or info["fp16_supported"]),
        "amp_dtype": info["recommended_amp_dtype"],
        "num_workers": num_workers,
        "pin_memory": bool(info["cuda_available"]),
        "hardware_snapshot": info,
    }


def seed_all_States(seed: int) -> None:
    """Deterministically seed Python, NumPy, and PyTorch (CPU & CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def capture_rng_states() -> Dict[str, Any]:
    """Capture RNG states for exact mid-training resume across sessions."""
    state: Dict[str, Any] = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        try:
            state["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        except Exception:
            pass
    return state


def restore_rng_states(rng_dict: Optional[Dict[str, Any]]) -> bool:
    """Restore RNG states if compatible; safely handles cross-GPU device count changes."""
    if not rng_dict:
        return False
    try:
        if "python_random_state" in rng_dict:
            random.setstate(rng_dict["python_random_state"])
        if "numpy_random_state" in rng_dict:
            np.random.set_state(rng_dict["numpy_random_state"])
        if "torch_cpu_rng_state" in rng_dict:
            cpu_state = rng_dict["torch_cpu_rng_state"]
            if isinstance(cpu_state, torch.Tensor):
                torch.set_rng_state(cpu_state.cpu().to(dtype=torch.uint8))
        if torch.cuda.is_available() and "torch_cuda_rng_state_all" in rng_dict:
            cuda_states = rng_dict["torch_cuda_rng_state_all"]
            if isinstance(cuda_states, list) and len(cuda_states) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all([s.cpu().to(dtype=torch.uint8) for s in cuda_states])
        return True
    except Exception:
        return False


def save_resumable_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    scaler: Optional[Any],
    epoch: int,
    best_val_metric: float,
    metric_name: str,
    seed: int,
    config: Dict[str, Any],
    history: Dict[str, Any],
    extra_payload: Optional[Dict[str, Any]] = None,
) -> Path:
    """Atomically write a full resumable checkpoint supporting cross-GPU resume."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    payload: Dict[str, Any] = {
        "schema_version": 2,
        "saved_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "epoch": int(epoch),
        "best_val_metric": float(best_val_metric),
        "metric_name": metric_name,
        "seed": int(seed),
        "config": config,
        "history": history,
        "runtime_info": inspect_runtime(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "rng_states": capture_rng_states(),
    }
    if extra_payload:
        payload.update(extra_payload)

    torch.save(payload, tmp_path)
    tmp_path.replace(out_path)
    return out_path


def load_resumable_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """Load checkpoint and restore model, optimizer, scheduler, scaler, and RNG states across GPUs."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict") or ckpt.get("model_state")
    if state_dict is None:
        raise KeyError(f"Checkpoint {path} không có 'model_state_dict'.")
    model.load_state_dict(state_dict, strict=strict)

    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        try:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        except Exception:
            pass
    restore_rng_states(ckpt.get("rng_states"))
    return ckpt


def save_run_hardware_log(out_dir: str | Path, exec_config: Dict[str, Any], peak_stats: Optional[Dict[str, Any]] = None) -> Path:
    """Persist actual GPU configuration and peak VRAM statistics for a run."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    current = inspect_runtime()
    record = {
        "execution_config": exec_config,
        "final_runtime_snapshot": current,
        "peak_stats": peak_stats or {
            "peak_vram_allocated_mb": current["peak_vram_allocated_mb"],
            "peak_vram_reserved_mb": current["peak_vram_reserved_mb"],
        },
    }
    target = out / "gpu_execution_profile.json"
    target.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return target
