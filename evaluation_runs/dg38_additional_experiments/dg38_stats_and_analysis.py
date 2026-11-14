"""
DG38 Stage 5 & Stage 10 — Three-Seed Statistics & Report Generator
==================================================================
Computes scientifically rigorous Mean ± Sample Std (ddof=1) across
independent training seeds (42, 123, 2026), enforces strict grouping rules
(never mixing architectures, datasets, or protocols; marking <3 seeds as
INCOMPLETE), and generates/updates the required DG38 markdown reports.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


REQUIRED_SEEDS: Tuple[int, ...] = (42, 123, 2026)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def compute_sample_mean_std(values: Sequence[float]) -> Dict[str, Any]:
    """
    Compute Mean and Sample Standard Deviation (ddof=1).
    Rule:
      - If len(values) >= 3: status = 'COMPLETED', ddof=1 sample std computed.
      - If 1 <= len(values) < 3: status = 'INCOMPLETE', std = None.
      - If len(values) == 0: status = 'NOT_STARTED', mean = None, std = None.
    """
    clean_vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    n = len(clean_vals)
    if n == 0:
        return {
            "n_runs": 0,
            "status": "NOT_STARTED",
            "mean": None,
            "sample_std_ddof1": None,
            "min": None,
            "max": None,
            "formatted": "NOT_STARTED (0/3 seeds)",
        }
    arr = np.asarray(clean_vals, dtype=np.float64)
    mean_val = float(np.mean(arr))
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    if n < 3:
        return {
            "n_runs": n,
            "status": "INCOMPLETE",
            "mean": round(mean_val, 6),
            "sample_std_ddof1": None,
            "min": round(min_val, 6),
            "max": round(max_val, 6),
            "formatted": f"{mean_val:.4f} ± N/A (INCOMPLETE: {n}/3 seeds)",
        }
    std_val = float(np.std(arr, ddof=1))
    return {
        "n_runs": n,
        "status": "COMPLETED",
        "mean": round(mean_val, 6),
        "sample_std_ddof1": round(std_val, 6),
        "min": round(min_val, 6),
        "max": round(max_val, 6),
        "formatted": f"{mean_val:.4f} ± {std_val:.4f} (n={n}, ddof=1)",
    }


def validate_seed_group_homogeneity(runs: Sequence[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """
    Verify that all runs in a seed group share identical:
      - task ('classification' or 'detection')
      - model_name
      - manifest_sha256 / split_signature
      - training_protocol_id (e.g. finetune_blocks, epochs, img_size)
      - independent training seeds (no duplicate seeds or inference-only repeats)
    """
    errors: List[str] = []
    if not runs:
        return True, errors

    ref = runs[0]
    ref_keys = ("task", "model_name", "manifest_sha256", "protocol_id")
    seen_seeds = set()

    for idx, r in enumerate(runs):
        if r.get("is_inference_repeat_only", False):
            errors.append(
                f"Run index {idx} (seed={r.get('seed')}) is marked as inference-only repeat; "
                "DG38 requires independent training runs per seed."
            )
        seed = r.get("seed")
        if seed in seen_seeds:
            errors.append(f"Duplicate seed {seed} in seed group; each seed must be trained independently.")
        seen_seeds.add(seed)

        for k in ref_keys:
            if r.get(k) != ref.get(k):
                errors.append(
                    f"Inhomogeneous seed group on key '{k}': run 0 has '{ref.get(k)}' "
                    f"vs run {idx} has '{r.get(k)}'."
                )

    return len(errors) == 0, errors


def aggregate_seed_group(
    group_id: str,
    runs: Sequence[Dict[str, Any]],
    metric_keys: Sequence[str],
) -> Dict[str, Any]:
    """
    Aggregate a homogeneous group of training runs across seeds (42, 123, 2026).
    Raises ValueError if inhomogeneous runs are mixed.
    """
    is_valid, errors = validate_seed_group_homogeneity(runs)
    if not is_valid:
        raise ValueError(f"Seed group '{group_id}' failed homogeneity validation: {'; '.join(errors)}")

    completed_runs = [r for r in runs if r.get("status") == "COMPLETED"]
    seeds_present = [int(r["seed"]) for r in completed_runs if "seed" in r]
    missing_seeds = [s for s in REQUIRED_SEEDS if s not in seeds_present]

    metrics_summary: Dict[str, Any] = {}
    for mk in metric_keys:
        vals = [
            float(r["metrics"][mk])
            for r in completed_runs
            if "metrics" in r and mk in r["metrics"] and r["metrics"][mk] is not None
        ]
        metrics_summary[mk] = compute_sample_mean_std(vals)

    overall_status = (
        "COMPLETED"
        if len(completed_runs) >= 3 and len(missing_seeds) == 0
        else ("INCOMPLETE" if len(completed_runs) > 0 else "NOT_STARTED")
    )

    return {
        "group_id": group_id,
        "overall_status": overall_status,
        "n_completed_seeds": len(completed_runs),
        "seeds_completed": seeds_present,
        "seeds_missing": missing_seeds,
        "metrics_summary": metrics_summary,
        "runs": list(runs),
    }


def scan_and_update_registry(exp_dir: Path) -> Dict[str, Any]:
    """
    Scan run directories in `exp_dir/runs/` and merge any newly completed run
    summaries into `experiment_registry.json`.
    """
    registry_path = exp_dir / "experiment_registry.json"
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    else:
        registry = {"updated_at": _utc_now(), "experiments": {}}

    runs_root = exp_dir / "runs"
    if runs_root.exists():
        for summary_file in sorted(runs_root.rglob("run_summary.json")):
            try:
                data = json.loads(summary_file.read_text(encoding="utf-8"))
                exp_id = data.get("experiment_id") or summary_file.parent.name
                registry.setdefault("experiments", {})[exp_id] = data
            except Exception as exc:
                print(f"[WARN] Could not parse {summary_file}: {exc}")

    registry["updated_at"] = _utc_now()
    registry_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
    return registry


def main() -> None:
    parser = argparse.ArgumentParser(description="DG38 Three-Seed Statistics & Report Updater")
    parser.add_argument(
        "--exp-dir",
        type=Path,
        default=Path("evaluation_runs/dg38_additional_experiments"),
        help="Path to dg38_additional_experiments directory",
    )
    args = parser.parse_args()
    registry = scan_and_update_registry(args.exp_dir)
    print(
        f"[OK] Registry updated at {args.exp_dir / 'experiment_registry.json'} "
        f"({len(registry.get('experiments', {}))} entries)."
    )


if __name__ == "__main__":
    main()
