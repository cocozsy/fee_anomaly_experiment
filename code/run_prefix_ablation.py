#!/usr/bin/env python3
"""Prefix view ablation: M_prefix (with LLM CSV) vs M_no_prefix (same split, no LLM merge).

Runs train_distill twice with identical --aligned_csv / --split_mode / --seed, then writes
prefix_ablation_summary.json with teacher/student ΔF1 and ΔAUPRC on val/test
(Δ = M_prefix − M_no_prefix; positive means the prefix run scored higher).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


def _pick_metrics(report: Mapping[str, Any], role: str, split: str) -> Dict[str, float]:
    block = report.get(role, {})
    m = block.get(split, {})
    return {"f1": float(m.get("f1", 0.0)), "auprc": float(m.get("auprc", 0.0))}


def _delta(a: Mapping[str, float], b: Mapping[str, float]) -> Dict[str, float]:
    return {k: float(a[k]) - float(b[k]) for k in ("f1", "auprc")}


def _run_train(code_dir: Path, cwd: Path, env: dict[str, str], argv: list[str]) -> None:
    cmd = [sys.executable, str(code_dir / "train_distill.py"), *argv]
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="M_prefix vs M_no_prefix ablation (same split).")
    parser.add_argument("--aligned_csv", type=Path, required=True)
    parser.add_argument("--llm_features_csv", type=Path, required=True)
    parser.add_argument("--output_parent", type=Path, required=True, help="Directory holding M_prefix/ and M_no_prefix/.")
    parser.add_argument("--split_mode", type=str, default="temporal")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--teacher_epochs", type=int, default=12)
    parser.add_argument("--student_epochs", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--teacher_lr", type=float, default=1e-3)
    parser.add_argument("--student_lr", type=float, default=1e-3)
    parser.add_argument("--teacher_llm_soft_weight", type=float, default=0.5)
    parser.add_argument("--teacher_llm_anomaly_soft_weight", type=float, default=0.25)
    parser.add_argument("--calibration_n_bins", type=int, default=15)
    parser.add_argument("--cross_domain_industry_value", type=str, default=None)
    parser.add_argument("--cross_domain_voltage_value", type=str, default=None)
    parser.add_argument(
        "--repo_root",
        type=Path,
        default=None,
        help="Working directory for subprocess (default: parent of code/).",
    )
    args = parser.parse_args()

    code_dir = Path(__file__).resolve().parent
    repo_root = args.repo_root.resolve() if args.repo_root else code_dir.parent
    out_parent = args.output_parent.resolve()
    out_parent.mkdir(parents=True, exist_ok=True)
    dir_prefix = out_parent / "M_prefix"
    dir_no = out_parent / "M_no_prefix"

    env = os.environ.copy()
    sep = os.pathsep
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(code_dir) + (sep + prev if prev else "")

    common_tail = [
        "--aligned_csv",
        str(args.aligned_csv.resolve()),
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
        "--teacher_llm_anomaly_soft_weight",
        str(args.teacher_llm_anomaly_soft_weight),
        "--calibration_n_bins",
        str(args.calibration_n_bins),
    ]
    if args.cross_domain_industry_value:
        common_tail += ["--cross_domain_industry_value", args.cross_domain_industry_value]
    if args.cross_domain_voltage_value:
        common_tail += ["--cross_domain_voltage_value", args.cross_domain_voltage_value]

    _run_train(
        code_dir,
        repo_root,
        env,
        [
            "--output_dir",
            str(dir_prefix),
            "--llm_features_csv",
            str(args.llm_features_csv.resolve()),
            *common_tail,
        ],
    )
    _run_train(
        code_dir,
        repo_root,
        env,
        [
            "--output_dir",
            str(dir_no),
            *common_tail,
        ],
    )

    rp = json.loads((dir_prefix / "train_report.json").read_text(encoding="utf-8"))
    rn = json.loads((dir_no / "train_report.json").read_text(encoding="utf-8"))

    summary: Dict[str, Any] = {
        "schema": "prefix_ablation_v1",
        "M_prefix_dir": str(dir_prefix),
        "M_no_prefix_dir": str(dir_no),
        "delta_convention": "M_prefix minus M_no_prefix on raw val/test metrics from train_report",
        "deltas": {},
    }

    for split in ("val", "test"):
        for role in ("teacher", "student"):
            key = f"{role}_{split}"
            summary["deltas"][key] = _delta(
                _pick_metrics(rp, role, split),
                _pick_metrics(rn, role, split),
            )

    # Optional: threshold-tuned test (if both have same structure)
    def tuned_test(r: Mapping[str, Any], role: str) -> Optional[Dict[str, float]]:
        try:
            m = r["threshold_tuning"][role]["test_at_val_best_threshold"]
            return {"f1": float(m["f1"]), "auprc": float(m["auprc"])}
        except (KeyError, TypeError, ValueError):
            return None

    summary["deltas_tuned_test"] = {}
    for role in ("teacher", "student"):
        tp, tn = tuned_test(rp, role), tuned_test(rn, role)
        if tp and tn:
            summary["deltas_tuned_test"][role] = _delta(tp, tn)

    out_path = out_parent / "prefix_ablation_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
