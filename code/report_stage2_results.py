#!/usr/bin/env python3
"""Generate stage-2 comparison charts and append markdown summary."""
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


def _plot_stage_compare(self_report: Dict, full_report: Dict, stage: str, out_png: Path) -> None:
    metrics = ["f1", "recall", "precision", "acc"]
    a = [float(self_report[stage]["test"][m]) for m in metrics]
    b = [float(full_report[stage]["test"][m]) for m in metrics]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(metrics))
    width = 0.35
    ax.bar([i - width / 2 for i in x], a, width=width, label="env_self_v1")
    ax.bar([i + width / 2 for i in x], b, width=width, label="env_full_v1")
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics)
    ax.set_ylim(0.75, 1.01)
    ax.set_title(f"{stage} test metrics: env_self_v1 vs env_full_v1")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _plot_val_f1_curves(self_metrics: pd.DataFrame, full_metrics: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for idx, stage in enumerate(["teacher", "student"]):
        ax = axes[idx]
        s = self_metrics[self_metrics["stage"] == stage].sort_values("epoch")
        f = full_metrics[full_metrics["stage"] == stage].sort_values("epoch")
        ax.plot(s["epoch"], s["val_f1"], marker="o", label="env_self_v1")
        ax.plot(f["epoch"], f["val_f1"], marker="o", label="env_full_v1")
        ax.set_title(f"{stage} val_f1 by epoch")
        ax.set_xlabel("epoch")
        ax.set_ylabel("val_f1")
        ax.set_ylim(0.82, 0.91)
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _append_stage2_markdown(results_md: Path, self_report: Dict, full_report: Dict, gate: Dict) -> None:
    s = self_report["student"]["test"]
    f = full_report["student"]["test"]
    t1 = self_report["teacher"]["test"]
    t2 = full_report["teacher"]["test"]
    gate_up = gate["uplift"]
    section = f"""

## 5) Stage 2 Results (env_full_v1)

### Stage-2增量（同群基线 + 电价环境）

- Student F1: `{s["f1"]:.6f}` -> `{f["f1"]:.6f}` (`{f["f1"] - s["f1"]:+.6f}`)
- Student Recall: `{s["recall"]:.6f}` -> `{f["recall"]:.6f}` (`{f["recall"] - s["recall"]:+.6f}`)
- Teacher F1: `{t1["f1"]:.6f}` -> `{t2["f1"]:.6f}` (`{t2["f1"] - t1["f1"]:+.6f}`)
- Teacher Recall: `{t1["recall"]:.6f}` -> `{t2["recall"]:.6f}` (`{t2["recall"] - t1["recall"]:+.6f}`)

### 门禁检查（env_full_v1 相对 env_self_v1）

- Gate-A overall_f1 uplift: `{gate_up["overall_f1"]:+.6f}`
- Gate-B overall_recall uplift: `{gate_up["overall_recall"]:+.6f}`
- Gate-C subset_recall (no-rule-hit & weak_label=1) uplift: `{gate_up["subset_recall_no_rule_and_weak_pos"]:+.6f}`
- Subset size: `{gate["env_full_v1"]["subset_size"]}` / test size `{gate["env_full_v1"]["test_size"]}`

### Stage-2 图表

![Stage2 Student metric compare](figures/stage2_student_metric_compare.png)

![Stage2 Teacher metric compare](figures/stage2_teacher_metric_compare.png)

![Stage2 val-f1 curves](figures/stage2_training_val_f1_curves.png)
"""
    content = results_md.read_text(encoding="utf-8") if results_md.exists() else ""
    if "## 5) Stage 2 Results (env_full_v1)" in content:
        content = content.split("## 5) Stage 2 Results (env_full_v1)")[0].rstrip()
    results_md.write_text(content + section, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Report stage-2 env_full_v1 comparison results.")
    parser.add_argument("--env_self_report", type=Path, default=Path("checkpoints/env_self_v1/train_report.json"))
    parser.add_argument("--env_self_metrics", type=Path, default=Path("checkpoints/env_self_v1/metrics.jsonl"))
    parser.add_argument("--env_full_report", type=Path, default=Path("checkpoints/env_full_v1/train_report.json"))
    parser.add_argument("--env_full_metrics", type=Path, default=Path("checkpoints/env_full_v1/metrics.jsonl"))
    parser.add_argument("--gate_json", type=Path, default=Path("experiments/stage2_gate_eval.json"))
    parser.add_argument("--results_md", type=Path, default=Path("experiments/results.md"))
    parser.add_argument("--figures_dir", type=Path, default=Path("experiments/figures"))
    args = parser.parse_args()

    self_report = _load_json(args.env_self_report)
    full_report = _load_json(args.env_full_report)
    self_metrics = _load_metrics(args.env_self_metrics)
    full_metrics = _load_metrics(args.env_full_metrics)
    gate = _load_json(args.gate_json)

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    _plot_stage_compare(self_report, full_report, "student", args.figures_dir / "stage2_student_metric_compare.png")
    _plot_stage_compare(self_report, full_report, "teacher", args.figures_dir / "stage2_teacher_metric_compare.png")
    _plot_val_f1_curves(self_metrics, full_metrics, args.figures_dir / "stage2_training_val_f1_curves.png")
    _append_stage2_markdown(args.results_md, self_report, full_report, gate)
    print(f"[OK] stage-2 charts written: {args.figures_dir}")
    print(f"[OK] stage-2 summary appended: {args.results_md}")


if __name__ == "__main__":
    main()
