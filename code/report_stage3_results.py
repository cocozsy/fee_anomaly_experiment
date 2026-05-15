#!/usr/bin/env python3
"""Generate stage-3 comparison charts (env_full_v1 vs env_stage3_v1)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_metrics(path: Path) -> pd.DataFrame:
    rows: List[Dict] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    df["val_f1"] = df["val_metric"].apply(lambda x: float(x["f1"]))
    return df


def _plot_stage_compare(base_report: Dict, stage3_report: Dict, stage: str, out_png: Path) -> None:
    metrics = ["f1", "recall", "precision", "acc"]
    a = [float(base_report[stage]["test"][m]) for m in metrics]
    b = [float(stage3_report[stage]["test"][m]) for m in metrics]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(metrics))
    width = 0.35
    ax.bar([i - width / 2 for i in x], a, width=width, label="env_full_v1")
    ax.bar([i + width / 2 for i in x], b, width=width, label="env_stage3_v1")
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics)
    ax.set_ylim(0.75, 1.01)
    ax.set_title(f"{stage} test metrics: env_full_v1 vs env_stage3_v1")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _plot_val_f1_curves(base_metrics: pd.DataFrame, stage3_metrics: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for idx, stage in enumerate(["teacher", "student"]):
        ax = axes[idx]
        b = base_metrics[base_metrics["stage"] == stage].sort_values("epoch")
        s3 = stage3_metrics[stage3_metrics["stage"] == stage].sort_values("epoch")
        ax.plot(b["epoch"], b["val_f1"], marker="o", label="env_full_v1")
        ax.plot(s3["epoch"], s3["val_f1"], marker="o", label="env_stage3_v1")
        ax.set_title(f"{stage} val_f1 by epoch")
        ax.set_xlabel("epoch")
        ax.set_ylabel("val_f1")
        ax.set_ylim(0.82, 1.01)
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report stage-3 env_stage3_v1 comparison results.")
    parser.add_argument("--env_full_report", type=Path, default=Path("checkpoints/env_full_v1/train_report.json"))
    parser.add_argument("--env_full_metrics", type=Path, default=Path("checkpoints/env_full_v1/metrics.jsonl"))
    parser.add_argument("--env_stage3_report", type=Path, default=Path("checkpoints/env_stage3_v1/train_report.json"))
    parser.add_argument("--env_stage3_metrics", type=Path, default=Path("checkpoints/env_stage3_v1/metrics.jsonl"))
    parser.add_argument("--figures_dir", type=Path, default=Path("experiments/figures"))
    args = parser.parse_args()

    base_report = _load_json(args.env_full_report)
    stage3_report = _load_json(args.env_stage3_report)
    base_metrics = _load_metrics(args.env_full_metrics)
    stage3_metrics = _load_metrics(args.env_stage3_metrics)

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    _plot_stage_compare(base_report, stage3_report, "student", args.figures_dir / "stage3_student_metric_compare.png")
    _plot_stage_compare(base_report, stage3_report, "teacher", args.figures_dir / "stage3_teacher_metric_compare.png")
    _plot_val_f1_curves(base_metrics, stage3_metrics, args.figures_dir / "stage3_training_val_f1_curves.png")
    print(f"[OK] stage-3 charts written: {args.figures_dir}")


if __name__ == "__main__":
    main()

