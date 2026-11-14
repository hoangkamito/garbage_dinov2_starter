"""Colab Pro Compute Units (CU) Budget Tracker & Cost Forecaster for DG38 Experiments.

Key Principles (Stage 1 Requirements):
1. Total initial budget: 200.45 CU.
2. Google Colab does NOT provide a verified public Python API to read live CU balance.
   Therefore, Usage Rate (`observed_cu_per_hour`) and any balance reconciliation MUST be
   provided by the user from the Colab UI ("View resources").
3. Forecast formula:
       estimated_CU = estimated_GPU_hours * observed_CU_per_hour
   where `estimated_GPU_hours` explicitly includes:
   - Training forward/backward time across all planned epochs & seeds
   - Validation evaluation time (including COCOeval / metric computation)
   - Checkpoint saving I/O to Google Drive
   - Startup / data staging / runtime overhead factor
4. Pre-experiment gate (`pre_experiment_gate`):
   - Prints a full cost forecast & budget status table before any full run.
   - Blocks full training if `estimated_CU` exceeds the category's remaining ceiling
     or would reduce the unapproved reserve below the 40.0 CU safety floor (or use
     the 50.45 CU reserve without explicit user authorization).
   - Provides actionable cost-reduction alternatives when a forecast exceeds budget.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


INITIAL_TOTAL_CU: float = 200.45
MIN_SAFETY_RESERVE_CU: float = 40.00

DEFAULT_CATEGORY_CEILINGS: Dict[str, Dict[str, Any]] = {
    "smoke_and_preflight": {
        "label_vi": "Smoke test, kiểm tra môi trường",
        "allocated_cu": 10.00,
        "is_reserve": False,
    },
    "resnet50_baseline": {
        "label_vi": "ResNet50 Classification baseline",
        "allocated_cu": 15.00,
        "is_reserve": False,
    },
    "yolo11s_baseline": {
        "label_vi": "YOLO11s Detection baseline",
        "allocated_cu": 35.00,
        "is_reserve": False,
    },
    "three_seed_experiments": {
        "label_vi": "Three-seed experiments (Seeds 42, 123, 2026)",
        "allocated_cu": 60.00,
        "is_reserve": False,
    },
    "crop_preprocessing_ablation": {
        "label_vi": "Crop preprocessing ablation (Validation cached)",
        "allocated_cu": 15.00,
        "is_reserve": False,
    },
    "finetune_blocks_ablation": {
        "label_vi": "Fine-tune blocks ablation (0, 2, 4, 12 blocks)",
        "allocated_cu": 15.00,
        "is_reserve": False,
    },
    "reserve_and_expansion": {
        "label_vi": "Dự phòng và mở rộng (cần phê duyệt, giữ tối thiểu 40 CU an toàn)",
        "allocated_cu": 50.45,
        "is_reserve": True,
    },
}


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


@dataclass
class PilotTimingMeasurement:
    """Empirical timing measured from a short pilot run on the active Colab GPU."""

    gpu_name: str
    observed_cu_per_hour: float
    train_batches_measured: int
    train_time_sec: float
    total_train_batches_per_epoch: int
    val_batches_measured: int = 0
    val_time_sec: float = 0.0
    total_val_batches_per_epoch: int = 0
    checkpoint_save_sec: float = 5.0
    startup_overhead_sec: float = 30.0
    overhead_multiplier: float = 1.10
    notes: str = ""

    def validate(self) -> None:
        if self.observed_cu_per_hour <= 0 or not math.isfinite(self.observed_cu_per_hour):
            raise ValueError(
                "observed_cu_per_hour phải > 0 và lấy từ giao diện Google Colab (View resources)."
            )
        if self.train_batches_measured <= 0 or self.train_time_sec <= 0:
            raise ValueError("Phải đo ít nhất 1 batch huấn luyện thực tế với thời gian > 0.")
        if self.total_train_batches_per_epoch <= 0:
            raise ValueError("total_train_batches_per_epoch phải > 0.")
        if self.overhead_multiplier < 1.0:
            raise ValueError("overhead_multiplier phải >= 1.0 để tính hao phí hợp lý.")

    @property
    def sec_per_train_batch(self) -> float:
        self.validate()
        return self.train_time_sec / self.train_batches_measured

    @property
    def sec_per_val_batch(self) -> float:
        if self.val_batches_measured > 0 and self.val_time_sec > 0:
            return self.val_time_sec / self.val_batches_measured
        # Conservative fallback: validation batch takes ~45% of train batch if not separately timed
        return 0.45 * self.sec_per_train_batch

    def estimate_epoch_seconds(self, val_every: int = 1, save_checkpoint_every_epoch: bool = True) -> Dict[str, float]:
        self.validate()
        train_epoch_sec = self.sec_per_train_batch * self.total_train_batches_per_epoch
        val_epoch_sec = (
            self.sec_per_val_batch * self.total_val_batches_per_epoch
            if self.total_val_batches_per_epoch > 0
            else 0.0
        )
        effective_val_sec = val_epoch_sec / max(val_every, 1)
        ckpt_sec = self.checkpoint_save_sec if save_checkpoint_every_epoch else (self.checkpoint_save_sec * 0.25)
        raw_epoch_sec = train_epoch_sec + effective_val_sec + ckpt_sec
        buffered_epoch_sec = raw_epoch_sec * self.overhead_multiplier
        return {
            "train_epoch_sec": round(train_epoch_sec, 3),
            "val_epoch_sec": round(effective_val_sec, 3),
            "checkpoint_io_sec": round(ckpt_sec, 3),
            "raw_epoch_sec": round(raw_epoch_sec, 3),
            "buffered_epoch_sec": round(buffered_epoch_sec, 3),
        }

    def forecast_run(
        self,
        epochs: int,
        num_seeds: int = 1,
        val_every: int = 1,
        extra_eval_sec_per_seed: float = 30.0,
    ) -> Dict[str, Any]:
        """Forecast total GPU hours and Compute Units (CU) for `epochs` x `num_seeds`."""
        self.validate()
        if epochs <= 0 or num_seeds <= 0:
            raise ValueError("epochs và num_seeds phải > 0.")
        epoch_breakdown = self.estimate_epoch_seconds(val_every=val_every)
        per_seed_sec = (
            self.startup_overhead_sec
            + epoch_breakdown["buffered_epoch_sec"] * epochs
            + extra_eval_sec_per_seed
        )
        total_sec = per_seed_sec * num_seeds
        estimated_gpu_hours = total_sec / 3600.0
        estimated_cu = estimated_gpu_hours * self.observed_cu_per_hour
        return {
            "gpu_name": self.gpu_name,
            "observed_cu_per_hour": round(self.observed_cu_per_hour, 4),
            "epochs_per_seed": epochs,
            "num_seeds": num_seeds,
            "val_every": val_every,
            "sec_per_train_batch": round(self.sec_per_train_batch, 4),
            "sec_per_val_batch": round(self.sec_per_val_batch, 4),
            "epoch_breakdown_sec": epoch_breakdown,
            "per_seed_estimated_sec": round(per_seed_sec, 2),
            "per_seed_estimated_minutes": round(per_seed_sec / 60.0, 2),
            "per_seed_estimated_cu": round((per_seed_sec / 3600.0) * self.observed_cu_per_hour, 4),
            "total_estimated_sec": round(total_sec, 2),
            "total_estimated_minutes": round(total_sec / 60.0, 2),
            "estimated_gpu_hours": round(estimated_gpu_hours, 4),
            "estimated_cu": round(estimated_cu, 4),
            "formula": "estimated_CU = estimated_GPU_hours * observed_CU_per_hour",
        }


class CUBudgetTracker:
    """Persistent manager for the 200.45 CU budget and experiment registry."""

    def __init__(
        self,
        ledger_path: str | Path,
        registry_path: Optional[str | Path] = None,
        report_md_path: Optional[str | Path] = None,
    ):
        self.ledger_path = Path(ledger_path)
        self.registry_path = (
            Path(registry_path)
            if registry_path is not None
            else self.ledger_path.parent / "experiment_registry.json"
        )
        self.report_md_path = (
            Path(report_md_path)
            if report_md_path is not None
            else self.ledger_path.parent / "cu_budget_report.md"
        )
        self.state = self._load_or_init()

    def _load_or_init(self) -> Dict[str, Any]:
        if self.ledger_path.exists():
            return json.loads(self.ledger_path.read_text(encoding="utf-8"))
        state = {
            "schema_version": 1,
            "created_utc": utc_now_iso(),
            "updated_utc": utc_now_iso(),
            "initial_total_cu": INITIAL_TOTAL_CU,
            "min_safety_reserve_cu": MIN_SAFETY_RESERVE_CU,
            "user_reported_remaining_cu": INITIAL_TOTAL_CU,
            "last_user_balance_sync_utc": utc_now_iso(),
            "categories": {
                k: {
                    "label_vi": v["label_vi"],
                    "allocated_cu": float(v["allocated_cu"]),
                    "is_reserve": bool(v["is_reserve"]),
                    "consumed_cu": 0.0,
                }
                for k, v in DEFAULT_CATEGORY_CEILINGS.items()
            },
            "forecasts": [],
            "usage_logs": [],
            "balance_sync_history": [
                {
                    "timestamp_utc": utc_now_iso(),
                    "reported_remaining_cu": INITIAL_TOTAL_CU,
                    "note": "Ngân sách khởi tạo ban đầu theo yêu cầu DG38 (200.45 CU).",
                }
            ],
        }
        self._save_state(state)
        return state

    def _save_state(self, state: Optional[Dict[str, Any]] = None) -> None:
        if state is not None:
            self.state = state
        self.state["updated_utc"] = utc_now_iso()
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.ledger_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.state, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.ledger_path)

    @property
    def total_logged_consumed_cu(self) -> float:
        return round(
            sum(float(c["consumed_cu"]) for c in self.state["categories"].values()),
            4,
        )

    @property
    def effective_remaining_cu(self) -> float:
        """Uses the more conservative of (initial - logged) vs user-reported remaining balance."""
        by_logs = round(float(self.state["initial_total_cu"]) - self.total_logged_consumed_cu, 4)
        user_rep = float(self.state.get("user_reported_remaining_cu", by_logs))
        return round(min(by_logs, user_rep), 4)

    def sync_user_reported_balance(self, remaining_cu_from_colab_ui: float, note: str = "") -> Dict[str, Any]:
        """Update remaining CU balance manually read by the user from Google Colab UI."""
        if remaining_cu_from_colab_ui < 0 or remaining_cu_from_colab_ui > INITIAL_TOTAL_CU:
            raise ValueError(f"Số dư CU phải nằm trong khoảng [0, {INITIAL_TOTAL_CU}].")
        self.state["user_reported_remaining_cu"] = round(float(remaining_cu_from_colab_ui), 4)
        self.state["last_user_balance_sync_utc"] = utc_now_iso()
        entry = {
            "timestamp_utc": utc_now_iso(),
            "reported_remaining_cu": round(float(remaining_cu_from_colab_ui), 4),
            "logged_consumed_cu_at_sync": self.total_logged_consumed_cu,
            "implied_untracked_drift_cu": round(
                (float(self.state["initial_total_cu"]) - self.total_logged_consumed_cu)
                - float(remaining_cu_from_colab_ui),
                4,
            ),
            "note": note or "Cập nhật thủ công từ giao diện Colab.",
        }
        self.state["balance_sync_history"].append(entry)
        self._save_state()
        self.write_markdown_report()
        return entry

    def check_Experiment_gate(
        self,
        run_id: str,
        category: str,
        forecast: Dict[str, Any],
        allow_reserve_draw: bool = False,
        warn_threshold_ratio: float = 0.80,
    ) -> Dict[str, Any]:
        """Evaluate whether a planned experiment fits within its category ceiling and safety floor."""
        if category not in self.state["categories"]:
            raise KeyError(
                f"Category không hợp lệ: {category}. Chọn trong {list(self.state['categories'])}"
            )
        cat_info = self.state["categories"][category]
        est_cu = float(forecast["estimated_cu"])
        cat_allocated = float(cat_info["allocated_cu"])
        cat_consumed = float(cat_info["consumed_cu"])
        cat_remaining = round(cat_allocated - cat_consumed, 4)

        total_remaining = self.effective_remaining_cu
        projected_total_remaining = round(total_remaining - est_cu, 4)
        min_floor = float(self.state["min_safety_reserve_cu"])

        reasons: List[str] = []
        warnings: List[str] = []
        recommendations: List[str] = []

        if cat_info.get("is_reserve") and not allow_reserve_draw:
            reasons.append(
                "Không tự động sử dụng ngân sách 'reserve_and_expansion' (50.45 CU) khi chưa được người dùng phê duyệt."
            )

        if est_cu > cat_remaining:
            if not allow_reserve_draw:
                reasons.append(
                    f"Chi phí dự báo ({est_cu:.2f} CU) vượt quá hạn mức còn lại của nhóm '{category}' ({cat_remaining:.2f} / {cat_allocated:.2f} CU)."
                )
            else:
                warnings.append(
                    f"Chi phí dự báo ({est_cu:.2f} CU) vượt hạn mức nhóm '{category}' ({cat_remaining:.2f} CU), đã được phép trích dự phòng."
                )

        if projected_total_remaining < min_floor:
            reasons.append(
                f"Sau khi chạy sẽ chỉ còn {projected_total_remaining:.2f} CU, thấp hơn mức dự phòng an toàn tối thiểu ({min_floor:.2f} CU)."
            )

        if cat_allocated > 0 and (cat_consumed + est_cu) / cat_allocated >= warn_threshold_ratio:
            warnings.append(
                f"Sau lượt chạy này, nhóm '{category}' sẽ tiêu thụ {((cat_consumed + est_cu) / cat_allocated) * 100:.1f}% hạn mức được giao."
            )

        if reasons:
            recommendations.extend(
                [
                    "1. Giảm số lượng seeds chạy một lượt (ví dụ: chạy hoàn chỉnh 1 seed = 42 trước, kiểm tra kết quả rồi mới chạy seed 123 và 2026).",
                    "2. Bật Mixed Precision (AMP FP16/BF16) và tăng batch_size + num_workers phù hợp VRAM để tăng throughput (giảm số giờ GPU).",
                    "3. Chuyển các tác vụ nhẹ (ResNet50, YOLO11s, Crop Ablation) sang GPU NVIDIA L4 thay vì A100/G4 để giảm Usage Rate (CU/giờ).",
                    "4. Kiểm tra và tái sử dụng cache (ví dụ: SQLite proposal cache cho Crop Ablation) hoặc resume từ checkpoint đã lưu trên Google Drive.",
                ]
            )

        approved = len(reasons) == 0
        gate_record = {
            "timestamp_utc": utc_now_iso(),
            "run_id": run_id,
            "category": category,
            "category_label_vi": cat_info["label_vi"],
            "approved": approved,
            "estimated_cu": round(est_cu, 4),
            "estimated_gpu_hours": forecast.get("estimated_gpu_hours"),
            "observed_cu_per_hour": forecast.get("observed_cu_per_hour"),
            "gpu_name": forecast.get("gpu_name"),
            "category_allocated_cu": cat_allocated,
            "category_consumed_cu_before": cat_consumed,
            "category_remaining_cu_before": cat_remaining,
            "total_remaining_cu_before": total_remaining,
            "projected_total_remaining_cu_after": projected_total_remaining,
            "min_safety_reserve_cu": min_floor,
            "reasons_blocked": reasons,
            "warnings": warnings,
            "cost_reduction_recommendations": recommendations,
            "forecast_detail": forecast,
        }
        self.state["forecasts"].append(gate_record)
        self._save_state()
        self.write_markdown_report()
        return gate_record

    def record_actual_usage(
        self,
        run_id: str,
        category: str,
        gpu_name: str,
        observed_cu_per_hour: float,
        elapsed_seconds: float,
        status: str = "COMPLETED",
        epochs_completed: int = 0,
        best_val_metric: Optional[float] = None,
        checkpoint_path: str = "",
        notes: str = "",
    ) -> Dict[str, Any]:
        """Record actual GPU time and compute consumed CU for a run."""
        if category not in self.state["categories"]:
            raise KeyError(f"Unknown category: {category}")
        if observed_cu_per_hour <= 0:
            raise ValueError("observed_cu_per_hour phải > 0.")
        if elapsed_seconds < 0:
            raise ValueError("elapsed_seconds không được âm.")
        gpu_hours = elapsed_seconds / 3600.0
        actual_cu = round(gpu_hours * observed_cu_per_hour, 4)

        self.state["categories"][category]["consumed_cu"] = round(
            float(self.state["categories"][category]["consumed_cu"]) + actual_cu,
            4,
        )
        log_entry = {
            "timestamp_utc": utc_now_iso(),
            "run_id": run_id,
            "category": category,
            "gpu_name": gpu_name,
            "observed_cu_per_hour": round(observed_cu_per_hour, 4),
            "elapsed_seconds": round(elapsed_seconds, 2),
            "elapsed_gpu_hours": round(gpu_hours, 4),
            "actual_consumed_cu": actual_cu,
            "status": status,
            "epochs_completed": epochs_completed,
            "best_val_metric": best_val_metric,
            "checkpoint_path": checkpoint_path,
            "notes": notes,
            "reminder_vi": "Vui lòng kiểm tra số dư CU thực tế trên giao diện Colab (View resources) và gọi sync_user_reported_balance() trước lượt chạy tiếp theo.",
        }
        self.state["usage_logs"].append(log_entry)
        self._save_state()
        self._update_registry_run(log_entry)
        self.write_markdown_report()
        return log_entry

    def _update_registry_run(self, log_entry: Dict[str, Any]) -> None:
        if not self.registry_path.exists():
            return
        try:
            reg = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except Exception:
            return
        runs = reg.get("experiments", [])
        for r in runs:
            if r.get("run_id") == log_entry["run_id"]:
                r["gpu_name"] = log_entry["gpu_name"]
                r["usage_rate_cu_per_hour"] = log_entry["observed_cu_per_hour"]
                r["epochs_completed"] = log_entry["epochs_completed"]
                if log_entry["best_val_metric"] is not None:
                    r["best_validation_metric"] = log_entry["best_val_metric"]
                r["training_time_seconds"] = log_entry["elapsed_seconds"]
                r["actual_cu"] = log_entry["actual_consumed_cu"]
                if log_entry["checkpoint_path"]:
                    r["checkpoint_path"] = log_entry["checkpoint_path"]
                r["status"] = log_entry["status"]
                r["updated_utc"] = utc_now_iso()
                break
        reg["updated_utc"] = utc_now_iso()
        self.registry_path.write_text(json.dumps(reg, indent=2, ensure_ascii=False), encoding="utf-8")

    def format_budget_table_markdown(self) -> str:
        lines = [
            "| Mã nhóm (`category`) | Nhiệm vụ | Hạn mức cấp (CU) | Đã dùng (CU) | Còn lại trong nhóm (CU) | Tỷ lệ dùng (%) | Trạng thái |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
        for key, info in self.state["categories"].items():
            alloc = float(info["allocated_cu"])
            used = float(info["consumed_cu"])
            rem = alloc - used
            pct = (used / alloc * 100.0) if alloc > 0 else 0.0
            if info.get("is_reserve"):
                status = "LOCKED_RESERVE (Giữ >= 40 CU an toàn)"
            elif used == 0:
                status = "CHƯA DÙNG"
            elif rem < 0:
                status = "VƯỢT HẠN MỨC NHÓM"
            elif pct >= 80.0:
                status = "SẮP ĐẦY (>=80%)"
            else:
                status = "ĐANG TRONG HẠN MỨC"
            lines.append(
                f"| `{key}` | {info['label_vi']} | {alloc:.2f} | {used:.2f} | {rem:.2f} | {pct:.1f}% | {status} |"
            )
        lines.append(
            f"| **TỔNG CỘNG** | **Ngân sách Colab Pro hiện có** | **{self.state['initial_total_cu']:.2f}** | **{self.total_logged_consumed_cu:.2f}** | **{self.effective_remaining_cu:.2f}** | **{(self.total_logged_consumed_cu / self.state['initial_total_cu']) * 100:.1f}%** | **Dự phòng tối thiểu: {self.state['min_safety_reserve_cu']:.2f} CU** |"
        )
        return "\n".join(lines)

    def write_markdown_report(self) -> None:
        self.report_md_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Báo cáo Quản lý & Dự báo Ngân sách Google Colab Pro (200.45 CU)",
            "",
            f"- **Cập nhật lần cuối (UTC):** `{self.state['updated_utc']}`",
            f"- **Tổng ngân sách khởi tạo:** **`{self.state['initial_total_cu']:.2f} CU`**",
            f"- **Tổng CU đã ghi nhận qua các lượt chạy:** **`{self.total_logged_consumed_cu:.2f} CU`**",
            f"- **Số dư CU do người dùng đồng bộ từ Colab UI gần nhất:** **`{self.state['user_reported_remaining_cu']:.2f} CU`** *(lúc `{self.state['last_user_balance_sync_utc']}`)*",
            f"- **Số dư CU khả dụng hiệu dụng (`effective_remaining_cu`):** **`{self.effective_remaining_cu:.2f} CU`**",
            f"- **Ngưỡng dự phòng an toàn tối thiểu bắt buộc:** **`{self.state['min_safety_reserve_cu']:.2f} CU`**",
            "",
            "> [!IMPORTANT]",
            "> Google Colab không cung cấp API Python chính thức để đọc trực tiếp số Compute Units (CU) còn lại. Mọi phép dự báo và ghi nhận chi phí đều dựa trên công thức:",
            "> `estimated_CU = estimated_GPU_hours × observed_CU_per_hour`",
            "> trong đó `observed_CU_per_hour` (Usage Rate) được người dùng nhập trực tiếp từ giao diện **Google Colab → View resources**.",
            "",
            "---",
            "",
            "## 1. Bảng Phân bổ và Trạng thái Ngân sách Hiện tại",
            "",
            self.format_budget_table_markdown(),
            "",
            "---",
            "",
            "## 2. Lịch sử Dự báo Chi phí Trước Thí nghiệm (`pre_experiment_gate`)",
            "",
        ]
        if not self.state["forecasts"]:
            lines.append("*Chưa có lượt chạy pilot/forecast nào được ghi nhận trên Colab.*")
        else:
            lines.extend(
                [
                    "| Thời gian (UTC) | `run_id` | Nhóm ngân sách | GPU | Usage Rate (CU/h) | Giờ GPU dự kiến | CU dự kiến | Phê duyệt | Ghi chú / Cảnh báo |",
                    "|---|---|---|---|---:|---:|---:|:---:|---|",
                ]
            )
            for f in self.state["forecasts"]:
                notes = "; ".join(f.get("reasons_blocked", []) + f.get("warnings", [])) or "Đạt điều kiện ngân sách"
                lines.append(
                    f"| `{f['timestamp_utc'][:19]}` | `{f['run_id']}` | `{f['category']}` | `{f.get('gpu_name', 'N/A')}` | "
                    f"{f.get('observed_cu_per_hour', 0.0):.2f} | {f.get('estimated_gpu_hours', 0.0):.3f} | "
                    f"**{f['estimated_cu']:.2f}** | {'ĐẠT' if f['approved'] else 'CHẶN'} | {notes} |"
                )

        lines.extend(
            [
                "",
                "---",
                "",
                "## 3. Nhật ký Tiêu thụ CU Thực tế (`usage_logs`)",
                "",
            ]
        )
        if not self.state["usage_logs"]:
            lines.append("*Chưa có lượt chạy GPU thực tế nào được ghi nhận.*")
        else:
            lines.extend(
                [
                    "| Thời gian (UTC) | `run_id` | Nhóm ngân sách | GPU | Usage Rate (CU/h) | Thời gian (phút) | CU thực tế | Epochs | Best Val Metric | Trạng thái |",
                    "|---|---|---|---|---:|---:|---:|---:|---:|---|",
                ]
            )
            for u in self.state["usage_logs"]:
                metric_str = f"{u['best_val_metric']:.4f}" if isinstance(u.get("best_val_metric"), (int, float)) else "N/A"
                lines.append(
                    f"| `{u['timestamp_utc'][:19]}` | `{u['run_id']}` | `{u['category']}` | `{u['gpu_name']}` | "
                    f"{u['observed_cu_per_hour']:.2f} | {u['elapsed_seconds'] / 60.0:.2f} | "
                    f"**{u['actual_consumed_cu']:.2f}** | {u['epochs_completed']} | {metric_str} | `{u['status']}` |"
                )

        self.report_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quản lý và dự báo ngân sách 200.45 CU cho thực nghiệm DG38.")
    parser.add_argument(
        "--ledger",
        default=str(
            Path(__file__).resolve().parent / "cu_budget_ledger.json"
        ),
        help="Đường dẫn file JSON lưu trạng thái ngân sách CU.",
    )
    sub = parser.add_subparsers(dest="cmd", required=False)

    sub.add_parser("status", help="In bảng trạng thái ngân sách hiện tại và cập nhật cu_budget_report.md")

    p_sync = sub.add_parser("sync-balance", help="Cập nhật số dư CU thực tế đọc từ giao diện Google Colab")
    p_sync.add_argument("--remaining-cu", type=float, required=True)
    p_sync.add_argument("--note", default="")

    p_fc = sub.add_parser("forecast", help="Chạy dự báo chi phí từ số đo pilot và kiểm tra cổng ngân sách")
    p_fc.add_argument("--run-id", required=True)
    p_fc.add_argument("--category", required=True, choices=list(DEFAULT_CATEGORY_CEILINGS))
    p_fc.add_argument("--gpu-name", required=True)
    p_fc.add_argument("--cu-per-hour", type=float, required=True)
    p_fc.add_argument("--train-batches-measured", type=int, required=True)
    p_fc.add_argument("--train-time-sec", type=float, required=True)
    p_fc.add_argument("--total-train-batches", type=int, required=True)
    p_fc.add_argument("--val-batches-measured", type=int, default=0)
    p_fc.add_argument("--val-time-sec", type=float, default=0.0)
    p_fc.add_argument("--total-val-batches", type=int, default=0)
    p_fc.add_argument("--epochs", type=int, required=True)
    p_fc.add_argument("--seeds", type=int, default=1)
    p_fc.add_argument("--val-every", type=int, default=1)
    p_fc.add_argument("--checkpoint-sec", type=float, default=5.0)
    p_fc.add_argument("--allow-reserve", action="store_true")

    p_log = sub.add_parser("log-usage", help="Ghi nhận thời gian và số CU tiêu thụ thực tế của một run")
    p_log.add_argument("--run-id", required=True)
    p_log.add_argument("--category", required=True, choices=list(DEFAULT_CATEGORY_CEILINGS))
    p_log.add_argument("--gpu-name", required=True)
    p_log.add_argument("--cu-per-hour", type=float, required=True)
    p_log.add_argument("--elapsed-sec", type=float, required=True)
    p_log.add_argument("--epochs-completed", type=int, default=0)
    p_log.add_argument("--best-val-metric", type=float, default=None)
    p_log.add_argument("--checkpoint-path", default="")
    p_log.add_argument("--status", default="COMPLETED")
    p_log.add_argument("--notes", default="")

    args = parser.parse_args()
    tracker = CUBudgetTracker(args.ledger)

    if args.cmd in (None, "status"):
        tracker.write_markdown_report()
        print(tracker.format_budget_table_markdown())
    elif args.cmd == "sync-balance":
        res = tracker.sync_user_reported_balance(args.remaining_cu, args.note)
        print(json.dumps(res, indent=2, ensure_ascii=False))
    elif args.cmd == "forecast":
        pilot = PilotTimingMeasurement(
            gpu_name=args.gpu_name,
            observed_cu_per_hour=args.cu_per_hour,
            train_batches_measured=args.train_batches_measured,
            train_time_sec=args.train_time_sec,
            total_train_batches_per_epoch=args.total_train_batches,
            val_batches_measured=args.val_batches_measured,
            val_time_sec=args.val_time_sec,
            total_val_batches_per_epoch=args.total_val_batches,
            checkpoint_save_sec=args.checkpoint_sec,
        )
        fc = pilot.forecast_run(epochs=args.epochs, num_seeds=args.seeds, val_every=args.val_every)
        gate = tracker.check_Experiment_gate(
            run_id=args.run_id,
            category=args.category,
            forecast=fc,
            allow_reserve_draw=args.allow_reserve,
        )
        print(tracker.format_budget_table_markdown())
        print("\n=== KẾT QUẢ DỰ BÁO & KIỂM TRA NGÂN SÁCH ===")
        print(json.dumps(gate, indent=2, ensure_ascii=False))
    elif args.cmd == "log-usage":
        res = tracker.record_actual_usage(
            run_id=args.run_id,
            category=args.category,
            gpu_name=args.gpu_name,
            observed_cu_per_hour=args.cu_per_hour,
            elapsed_seconds=args.elapsed_sec,
            status=args.status,
            epochs_completed=args.epochs_completed,
            best_val_metric=args.best_val_metric,
            checkpoint_path=args.checkpoint_path,
            notes=args.notes,
        )
        print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
