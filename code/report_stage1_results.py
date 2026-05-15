#!/usr/bin/env python3
"""Generate stage-1 experiment charts and markdown report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _load_report(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_metrics(path: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"metrics file is empty: {path}")
    df = pd.DataFrame(rows)
    df["val_f1"] = df["val_metric"].apply(lambda x: float(x["f1"]))
    df["val_precision"] = df["val_metric"].apply(lambda x: float(x["precision"]))
    df["val_recall"] = df["val_metric"].apply(lambda x: float(x["recall"]))
    return df


def _plot_metric_compare(
    baseline_report: Dict,
    env_report: Dict,
    stage: str,
    output_png: Path,
) -> None:
    metrics = ["f1", "recall", "precision", "acc"]
    baseline_values = [float(baseline_report[stage]["test"][m]) for m in metrics]
    env_values = [float(env_report[stage]["test"][m]) for m in metrics]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = range(len(metrics))
    width = 0.35
    ax.bar([i - width / 2 for i in x], baseline_values, width=width, label="baseline_v0")
    ax.bar([i + width / 2 for i in x], env_values, width=width, label="env_self_v1")
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics)
    ax.set_ylim(0.75, 1.01)
    ax.set_ylabel("score")
    ax.set_title(f"{stage} test metrics: baseline_v0 vs env_self_v1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def _plot_f1_curves(
    baseline_metrics: pd.DataFrame,
    env_metrics: pd.DataFrame,
    output_png: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for idx, stage in enumerate(["teacher", "student"]):
        ax = axes[idx]
        b = baseline_metrics[baseline_metrics["stage"] == stage]
        e = env_metrics[env_metrics["stage"] == stage]
        ax.plot(b["epoch"], b["val_f1"], marker="o", label="baseline_v0")
        ax.plot(e["epoch"], e["val_f1"], marker="o", label="env_self_v1")
        ax.set_title(f"{stage} val f1 by epoch")
        ax.set_xlabel("epoch")
        ax.set_ylabel("val f1")
        ax.set_ylim(0.82, 0.91 if stage == "teacher" else 0.905)
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def _plot_intermediate_feature_snapshot(env_csv: Path, output_png: Path) -> None:
    use_cols = [
        "history_len",
        "env_self_fee_vs_roll3_z",
        "env_self_energy_vs_roll3_z",
        "env_self_unit_price_dev_vs_roll3_z",
    ]
    df = pd.read_csv(env_csv, usecols=use_cols).sample(n=20000, random_state=42)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, col in zip(axes.flatten(), use_cols):
        ax.hist(df[col].fillna(0.0), bins=40)
        ax.set_title(col)
        ax.grid(alpha=0.2)
    fig.suptitle("Stage-1 intermediate feature distributions (sample=20k)", y=1.02)
    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def _write_markdown_report(
    output_md: Path,
    env_meta: Dict,
    baseline_report: Dict,
    env_report: Dict,
) -> None:
    b_student = baseline_report["student"]["test"]
    e_student = env_report["student"]["test"]
    b_teacher = baseline_report["teacher"]["test"]
    e_teacher = env_report["teacher"]["test"]

    md = f"""# Stage 1 Results (env_self_v1)

## 1) Stage-1 feature build (A模块前半部分)

- Input: `data/aligned/aligned_customer_month.csv`
- Output: `data/aligned/aligned_customer_month_env.csv`
- Row count: `{env_meta["row_count"]}`
- Added feature count: `{env_meta["new_feature_count"]}`
- Added feature prefix: `env_self_*`
- Rolling anti-leakage rule: all stats use `groupby("客户ID").shift(1)` history only.

## 2) Metric comparison (test set)

- Student F1: `{b_student["f1"]:.6f}` -> `{e_student["f1"]:.6f}` (`{e_student["f1"] - b_student["f1"]:+.6f}`)
- Student Recall: `{b_student["recall"]:.6f}` -> `{e_student["recall"]:.6f}` (`{e_student["recall"] - b_student["recall"]:+.6f}`)
- Teacher F1: `{b_teacher["f1"]:.6f}` -> `{e_teacher["f1"]:.6f}` (`{e_teacher["f1"] - b_teacher["f1"]:+.6f}`)
- Teacher Recall: `{b_teacher["recall"]:.6f}` -> `{e_teacher["recall"]:.6f}` (`{e_teacher["recall"] - b_teacher["recall"]:+.6f}`)

## 3) Charts

![Student metric compare](figures/stage1_student_metric_compare.png)

![Teacher metric compare](figures/stage1_teacher_metric_compare.png)

![Training val-f1 curves](figures/stage1_training_val_f1_curves.png)

![Intermediate feature distributions](figures/stage1_intermediate_feature_distributions.png)
"""
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(md, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate stage-1 metric charts and markdown report.")
    parser.add_argument("--baseline_report", type=Path, default=Path("checkpoints/baseline_v0/train_report.json"))
    parser.add_argument("--baseline_metrics", type=Path, default=Path("checkpoints/baseline_v0/metrics.jsonl"))
    parser.add_argument("--env_report", type=Path, default=Path("checkpoints/env_self_v1/train_report.json"))
    parser.add_argument("--env_metrics", type=Path, default=Path("checkpoints/env_self_v1/metrics.jsonl"))
    parser.add_argument("--env_meta", type=Path, default=Path("data/aligned/aligned_customer_month_env_metadata.json"))
    parser.add_argument("--env_csv", type=Path, default=Path("data/aligned/aligned_customer_month_env.csv"))
    parser.add_argument("--output_dir", type=Path, default=Path("experiments"))
    args = parser.parse_args()

    baseline_report = _load_report(args.baseline_report)
    env_report = _load_report(args.env_report)
    baseline_metrics = _load_metrics(args.baseline_metrics)
    env_metrics = _load_metrics(args.env_metrics)
    env_meta = _load_report(args.env_meta)

    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    _plot_metric_compare(
        baseline_report=baseline_report,
        env_report=env_report,
        stage="student",
        output_png=figures_dir / "stage1_student_metric_compare.png",
    )
    _plot_metric_compare(
        baseline_report=baseline_report,
        env_report=env_report,
        stage="teacher",
        output_png=figures_dir / "stage1_teacher_metric_compare.png",
    )
    _plot_f1_curves(
        baseline_metrics=baseline_metrics,
        env_metrics=env_metrics,
        output_png=figures_dir / "stage1_training_val_f1_curves.png",
    )
    _plot_intermediate_feature_snapshot(
        env_csv=args.env_csv,
        output_png=figures_dir / "stage1_intermediate_feature_distributions.png",
    )
    _write_markdown_report(
        output_md=args.output_dir / "results.md",
        env_meta=env_meta,
        baseline_report=baseline_report,
        env_report=env_report,
    )
    print(f"[OK] stage-1 report written: {args.output_dir / 'results.md'}")
    print(f"[OK] charts written under: {figures_dir}")


if __name__ == "__main__":
    main()
