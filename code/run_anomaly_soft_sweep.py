#!/usr/bin/env python3
"""Grid over teacher_llm_anomaly_soft_weight with fixed split; aggregates F1, AUPRC, ECE.

Each run is a separate train_distill invocation under output_parent/anomaly_soft_<weight>/.
Writes anomaly_soft_weight_sweep.json with per-run metrics and calibration.ece from train_report.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


def _ece(report: Mapping[str, Any], role: str, split: str) -> Optional[float]:
    cal = report.get("calibration")
    if not isinstance(cal, dict):
        return None
    block = cal.get(role)
    if not isinstance(block, dict):
        return None
    sp = block.get(split)
    if not isinstance(sp, dict):
        return None
    e = sp.get("ece")
    return float(e) if e is not None else None


def _run_train(code_dir: Path, cwd: Path, env: dict[str, str], argv: list[str]) -> None:
    cmd = [sys.executable, str(code_dir / "train_distill.py"), *argv]
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="teacher_llm_anomaly_soft_weight sweep + calibration.")
    parser.add_argument("--aligned_csv", type=Path, required=True)
    parser.add_argument("--llm_features_csv", type=Path, required=True)
    parser.add_argument("--output_parent", type=Path, required=True)
    parser.add_argument(
        "--anomaly_weights",
        type=str,
        default="0,0.25,0.5",
        help="Comma-separated teacher_llm_anomaly_soft_weight values.",
    )
    parser.add_argument("--split_mode", type=str, default="temporal")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--teacher_epochs", type=int, default=12)
    parser.add_argument("--student_epochs", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--teacher_lr", type=float, default=1e-3)
    parser.add_argument("--student_lr", type=float, default=1e-3)
    parser.add_argument("--teacher_llm_soft_weight", type=float, default=0.5)
    parser.add_argument("--calibration_n_bins", type=int, default=15)
    parser.add_argument("--cross_domain_industry_value", type=str, default=None)
    parser.add_argument("--cross_domain_voltage_value", type=str, default=None)
    parser.add_argument("--repo_root", type=Path, default=None)
    args = parser.parse_args()

    weights: List[float] = []
    for part in args.anomaly_weights.split(","):
        part = part.strip()
        if part:
            weights.append(float(part))
    if not weights:
        raise SystemExit("No weights parsed from --anomaly_weights")

    code_dir = Path(__file__).resolve().parent
    repo_root = args.repo_root.resolve() if args.repo_root else code_dir.parent
    out_parent = args.output_parent.resolve()
    out_parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    sep = os.pathsep
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(code_dir) + (sep + prev if prev else "")

    common_tail = [
        "--aligned_csv",
        str(args.aligned_csv.resolve()),
        "--llm_features_csv",
        str(args.llm_features_csv.resolve()),
        "--split_mode",
        args.split_mode,
        "--seed",
        str(args.seed),
        "--teacher_epochs",
        str(args.teacher_epochs),
        "--student_epochs",
        str(args.student_epochs),
        "--batch_size",
        str(args.batch_size),
        "--teacher_lr",
        str(args.teacher_lr),
        "--student_lr",
        str(args.student_lr),
        "--teacher_llm_soft_weight",
        str(args.teacher_llm_soft_weight),
        "--calibration_n_bins",
        str(args.calibration_n_bins),
    ]
    if args.cross_domain_industry_value:
        common_tail += ["--cross_domain_industry_value", args.cross_domain_industry_value]
    if args.cross_domain_voltage_value:
        common_tail += ["--cross_domain_voltage_value", args.cross_domain_voltage_value]

    runs: List[Dict[str, Any]] = []
    for w in weights:
        w_tag = str(w).replace(".", "p")
        run_dir = out_parent / f"anomaly_soft_{w_tag}"
        _run_train(
            code_dir,
            repo_root,
            env,
            [
                "--output_dir",
                str(run_dir),
                "--teacher_llm_anomaly_soft_weight",
                str(w),
                *common_tail,
            ],
        )
        report = json.loads((run_dir / "train_report.json").read_text(encoding="utf-8"))
        row: Dict[str, Any] = {
            "teacher_llm_anomaly_soft_weight": w,
            "output_dir": str(run_dir),
            "teacher": {
                "val": report.get("teacher", {}).get("val"),
                "test": report.get("teacher", {}).get("test"),
            },
            "student": {
                "val": report.get("student", {}).get("val"),
                "test": report.get("student", {}).get("test"),
            },
            "ece": {
                "teacher_val": _ece(report, "teacher", "val"),
                "teacher_test": _ece(report, "teacher", "test"),
                "student_val": _ece(report, "student", "val"),
                "student_test": _ece(report, "student", "test"),
            },
        }
        runs.append(row)

    sweep: Dict[str, Any] = {
        "schema": "anomaly_soft_weight_sweep_v1",
        "anomaly_weights": weights,
        "runs": runs,
    }
    out_path = out_parent / "anomaly_soft_weight_sweep.json"
    out_path.write_text(json.dumps(sweep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
