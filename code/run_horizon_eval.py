#!/usr/bin/env python3
"""Run HORIZON-style evaluation across multiple split modes.

For each --split_mode in {temporal, cold_start, cross_domain_industry,
cross_domain_voltage, tariff_shift}, this script invokes train_distill.py once
and aggregates the resulting train_report.json files into experiments/horizon_eval.md.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


DEFAULT_MODES: List[str] = [
    "temporal",
    "cold_start",
    "cross_domain_industry",
    "cross_domain_voltage",
    "tariff_shift",
]


def _run_one(
    python_bin: Path,
    aligned_csv: Path,
    llm_features_csv: Path,
    output_dir: Path,
    mode: str,
    teacher_epochs: int,
    student_epochs: int,
    batch_size: int,
    seed: int,
) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(python_bin),
        "code/train_distill.py",
        "--aligned_csv",
        str(aligned_csv),
        "--llm_features_csv",
        str(llm_features_csv),
        "--output_dir",
        str(output_dir),
        "--teacher_epochs",
        str(teacher_epochs),
        "--student_epochs",
        str(student_epochs),
        "--batch_size",
        str(batch_size),
        "--seed",
        str(seed),
        "--split_mode",
        mode,
    ]
    print(f"[RUN] {' '.join(cmd)}")
    log_path = output_dir / "train_live.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        ret = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, check=False)
    if ret.returncode != 0:
        raise RuntimeError(f"split_mode={mode} failed; see log {log_path}")
    report_path = output_dir / "train_report.json"
    return json.loads(report_path.read_text(encoding="utf-8"))


def _fmt(v: float) -> str:
    return f"{v:.4f}"


def _build_markdown(results: Dict[str, Dict]) -> str:
    header = (
        "# HORIZON Evaluation Results\n\n"
        "对照 HORIZON (NeurIPS 2025) 对 in-the-wild 用户行为建模的三大主张："
        "长期时序泛化、跨域迁移、未见用户冷启动，本节列出 5 种切分下\n"
        "教师/学生模型在测试集上的 F1、Recall、Precision、AUPRC。\n\n"
    )
    cols = ["mode", "split_size(train/val/test)", "extra",
            "teacher_f1", "teacher_recall", "teacher_precision", "teacher_auprc",
            "student_f1", "student_recall", "student_precision", "student_auprc"]
    lines = [
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for mode, report in results.items():
        cfg = report["config"]
        info = cfg.get("split_info", {})
        size = f"{info.get('train_size', '-')}/{info.get('val_size', '-')}/{info.get('test_size', '-')}"
        extra = ""
        if mode == "cold_start":
            extra = (
                f"train_cust={info.get('train_customers', '-')} "
                f"val_cust={info.get('val_customers', '-')} "
                f"test_cust={info.get('test_customers', '-')}"
            )
        elif mode == "cross_domain_industry":
            extra = f"holdout_industry={info.get('holdout_industry', '-')}"
        elif mode == "cross_domain_voltage":
            extra = f"holdout_voltage={info.get('holdout_voltage', '-')}"

        t = report["teacher"]["test"]
        s = report["student"]["test"]
        row = [
            mode,
            size,
            extra,
            _fmt(t["f1"]), _fmt(t["recall"]), _fmt(t["precision"]), _fmt(t.get("auprc", 0.0)),
            _fmt(s["f1"]), _fmt(s["recall"]), _fmt(s["precision"]), _fmt(s.get("auprc", 0.0)),
        ]
        lines.append("| " + " | ".join(row) + " |")
    return header + "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HORIZON-style multi-split evaluation.")
    parser.add_argument("--python_bin", type=Path, default=Path(".venv/bin/python"))
    parser.add_argument(
        "--aligned_csv",
        type=Path,
        default=Path("data/aligned/aligned_customer_month_env_stage3.csv"),
        help="Stage-3 enriched CSV is recommended so all env_*/event_* columns are present.",
    )
    parser.add_argument(
        "--llm_features_csv",
        type=Path,
        default=Path("data/aligned/aligned_customer_month_llm_features.csv"),
    )
    parser.add_argument(
        "--checkpoints_root",
        type=Path,
        default=Path("checkpoints"),
    )
    parser.add_argument("--output_md", type=Path, default=Path("experiments/horizon_eval.md"))
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    parser.add_argument("--teacher_epochs", type=int, default=12)
    parser.add_argument("--student_epochs", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results: Dict[str, Dict] = {}
    for mode in args.modes:
        out_dir = args.checkpoints_root / f"horizon_{mode}"
        report = _run_one(
            python_bin=args.python_bin,
            aligned_csv=args.aligned_csv,
            llm_features_csv=args.llm_features_csv,
            output_dir=out_dir,
            mode=mode,
            teacher_epochs=args.teacher_epochs,
            student_epochs=args.student_epochs,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        results[mode] = report

    md = _build_markdown(results)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(md, encoding="utf-8")
    print(f"[OK] wrote HORIZON evaluation summary: {args.output_md}")


if __name__ == "__main__":
    main()
