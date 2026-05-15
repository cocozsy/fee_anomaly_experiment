#!/usr/bin/env python3
"""Live/snapshot training progress monitor based on metrics.jsonl."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _read_metrics(metrics_path: Path) -> pd.DataFrame:
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    rows: List[Dict] = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"Metrics file is empty: {metrics_path}")
    df = pd.DataFrame(rows)
    df["val_f1"] = df["val_metric"].apply(lambda x: float(x.get("f1", 0.0)))
    df["train_total"] = df["train_losses"].apply(lambda x: float(x.get("total", 0.0)))
    return df


def _latest_status(df: pd.DataFrame, stage: str) -> str:
    part = df[df["stage"] == stage].sort_values("epoch")
    if part.empty:
        return f"{stage}: no record"
    last = part.iloc[-1]
    return (
        f"{stage}: epoch {int(last['epoch'])}/{int(last['epochs'])}, "
        f"val_f1={float(last['val_f1']):.4f}, train_total={float(last['train_total']):.4f}"
    )


def _plot_progress(df: pd.DataFrame, output_png: Path, title: str) -> None:
    teacher = df[df["stage"] == "teacher"].sort_values("epoch")
    student = df[df["stage"] == "student"].sort_values("epoch")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    if not teacher.empty:
        axes[0][0].plot(teacher["epoch"], teacher["val_f1"], marker="o")
        axes[0][0].set_title("Teacher val_f1")
        axes[0][0].set_xlabel("epoch")
        axes[0][0].set_ylabel("f1")
        axes[0][0].set_ylim(0.0, 1.0)

        axes[0][1].plot(teacher["epoch"], teacher["train_total"], marker="o")
        axes[0][1].set_title("Teacher train_total")
        axes[0][1].set_xlabel("epoch")
        axes[0][1].set_ylabel("loss")

    if not student.empty:
        axes[1][0].plot(student["epoch"], student["val_f1"], marker="o")
        axes[1][0].set_title("Student val_f1")
        axes[1][0].set_xlabel("epoch")
        axes[1][0].set_ylabel("f1")
        axes[1][0].set_ylim(0.0, 1.0)

        axes[1][1].plot(student["epoch"], student["train_total"], marker="o")
        axes[1][1].set_title("Student train_total")
        axes[1][1].set_xlabel("epoch")
        axes[1][1].set_ylabel("loss")

    for row in axes:
        for ax in row:
            ax.grid(alpha=0.2)

    status_teacher = _latest_status(df, "teacher")
    status_student = _latest_status(df, "student")
    fig.suptitle(
        f"{title}\n{status_teacher} | {status_student}\n"
        f"updated_at={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training progress from metrics.jsonl.")
    parser.add_argument("--metrics", type=Path, required=True, help="Path to metrics.jsonl")
    parser.add_argument("--output_png", type=Path, required=True, help="Output PNG path")
    parser.add_argument("--title", type=str, default="Training Progress")
    parser.add_argument("--watch", action="store_true", help="Continuously refresh plot.")
    parser.add_argument("--refresh_seconds", type=int, default=10, help="Refresh interval in seconds.")
    parser.add_argument(
        "--max_updates",
        type=int,
        default=0,
        help="Max refresh count in watch mode (0 means unlimited).",
    )
    args = parser.parse_args()

    update_count = 0
    while True:
        df = _read_metrics(args.metrics)
        _plot_progress(df=df, output_png=args.output_png, title=args.title)
        teacher_status = _latest_status(df, "teacher")
        student_status = _latest_status(df, "student")
        print(f"[OK] plot updated: {args.output_png}")
        print(f"[INFO] {teacher_status}")
        print(f"[INFO] {student_status}")

        if not args.watch:
            break

        update_count += 1
        if args.max_updates > 0 and update_count >= args.max_updates:
            break
        time.sleep(max(1, args.refresh_seconds))


if __name__ == "__main__":
    main()
