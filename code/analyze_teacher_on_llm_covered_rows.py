#!/usr/bin/env python3
"""Evaluate teacher (and optionally student) only on rows with real LLM features.

After merge_llm_features_into_aligned.py, most rows have NaN llm_*; training uses
mask=0 for soft labels and fillna(0) for prefix. This script slices each split's
tensor bundle to rows where ``llm_anomaly_prob`` is non-NaN (same coverage rule as
soft-supervision mask), then runs the saved checkpoints with the same z-score and
categorical maps as full-split training.

Example:
  PYTHONPATH=code .venv/bin/python code/analyze_teacher_on_llm_covered_rows.py \\
    --aligned_csv data/aligned/aligned_customer_month_decoupled_env_full_d3c_merged_llm_openai_pilot2500.csv \\
    --checkpoint_dir checkpoints/horizon_stage5_openai_p2500/horizon_temporal \\
    --split_mode temporal --seed 42 --eval_student --use_tuned_threshold

  # cross_domain_industry：--cross_domain_industry_value 须与 checkpoint 的
  # train_report.json → split_info.holdout_industry 一致（未传则由 train_distill 自动选行业）
  PYTHONPATH=code .venv/bin/python code/analyze_teacher_on_llm_covered_rows.py \\
    --aligned_csv data/aligned/aligned_customer_month_decoupled_env_full_d3c_merged_llm_openai_pilot2500.csv \\
    --checkpoint_dir checkpoints/horizon_stage5_openai_p2500/horizon_cross_domain_industry \\
    --split_mode cross_domain_industry --cross_domain_industry_value 6201 \\
    --seed 42 --eval_student --use_tuned_threshold
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from models import StudentModel, TeacherModel

# Reuse split + tensor pipeline from train_distill (same preprocessing contract).
from train_distill import (  # noqa: E402
    FeeDataset,
    LLM_ANOMALY_PROB_COL,
    TensorBundle,
    binary_metrics_from_prob,
    build_tensors,
    collect_student_logits_and_labels,
    collect_teacher_logits_and_labels,
    get_llm_embedding_cols,
    infer_columns,
    set_seed,
    split_dataset,
    _llm_prob_col_for_reason_label,
)


def _llm_row_mask(df_part: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df_part.columns:
        raise ValueError(f"Missing coverage column {col!r} on split dataframe.")
    return pd.to_numeric(df_part[col], errors="coerce").notna().to_numpy()


def _slice_bundle(bundle: TensorBundle, idx: np.ndarray) -> TensorBundle:
    if idx.size == 0:
        raise ValueError("No LLM-covered rows in this split; cannot evaluate.")
    t_idx = torch.from_numpy(idx.astype(np.int64, copy=False))
    return TensorBundle(
        num_x=bundle.num_x[t_idx],
        cat_x=bundle.cat_x[t_idx],
        reason_x=bundle.reason_x[t_idx],
        llm_x=bundle.llm_x[t_idx],
        y_weak=bundle.y_weak[t_idx],
        y_reason=bundle.y_reason[t_idx],
        y_reason_soft=bundle.y_reason_soft[t_idx],
        y_reason_soft_mask=bundle.y_reason_soft_mask[t_idx],
        y_weak_soft=bundle.y_weak_soft[t_idx],
        y_weak_soft_mask=bundle.y_weak_soft_mask[t_idx],
    )


def _eval_teacher_probs(
    model: TeacherModel,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> Dict[str, float]:
    logit, y = collect_teacher_logits_and_labels(model, loader, device)
    prob = torch.sigmoid(logit)
    return binary_metrics_from_prob(prob, y, threshold=threshold)


def _eval_student_probs(
    model: StudentModel,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> Dict[str, float]:
    logit, y = collect_student_logits_and_labels(model, loader, device)
    prob = torch.sigmoid(logit)
    return binary_metrics_from_prob(prob, y, threshold=threshold)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned_csv", type=Path, required=True)
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        required=True,
        help="Directory containing teacher.pt, student.pt, train_report.json",
    )
    parser.add_argument(
        "--split_mode",
        type=str,
        required=True,
        choices=[
            "temporal",
            "cold_start",
            "cross_domain_industry",
            "cross_domain_voltage",
            "tariff_shift",
        ],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--llm_coverage_col",
        type=str,
        default=LLM_ANOMALY_PROB_COL,
        help="Row is 'LLM-covered' if this column is non-NaN (matches soft-label mask).",
    )
    parser.add_argument(
        "--use_tuned_threshold",
        action="store_true",
        help="Use teacher/student val-tuned thresholds from train_report.json if present.",
    )
    parser.add_argument("--eval_student", action="store_true")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument(
        "--cross_domain_industry_value",
        type=str,
        default=None,
        help="Must match the run that produced checkpoint_dir (default: same auto holdout as train_distill).",
    )
    parser.add_argument(
        "--cross_domain_voltage_value",
        type=str,
        default=None,
        help="Must match the run that produced checkpoint_dir.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    report_path = args.checkpoint_dir / "train_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Missing {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    thr_teacher = 0.5
    thr_student = 0.5
    if args.use_tuned_threshold:
        tt = report.get("threshold_tuning") or {}
        if "teacher" in tt and "best_threshold_from_val_f1" in tt["teacher"]:
            thr_teacher = float(tt["teacher"]["best_threshold_from_val_f1"])
        if "student" in tt and "best_threshold_from_val_f1" in tt["student"]:
            thr_student = float(tt["student"]["best_threshold_from_val_f1"])

    cfg = report.get("config") or {}
    si = cfg.get("split_info") or report.get("split_info") or {}
    inferred_industry: str | None = None
    inferred_voltage: str | None = None
    if args.split_mode == "cross_domain_industry" and args.cross_domain_industry_value is None:
        ho = si.get("holdout_industry")
        if ho is not None:
            args.cross_domain_industry_value = str(ho)
            inferred_industry = str(ho)
    if args.split_mode == "cross_domain_voltage" and args.cross_domain_voltage_value is None:
        ho = si.get("holdout_voltage")
        if ho is not None:
            args.cross_domain_voltage_value = str(ho)
            inferred_voltage = str(ho)

    df = pd.read_csv(args.aligned_csv)
    train_df, val_df, test_df, split_info = split_dataset(
        df,
        mode=args.split_mode,
        seed=args.seed,
        cross_domain_industry_value=args.cross_domain_industry_value,
        cross_domain_voltage_value=args.cross_domain_voltage_value,
    )

    reason_label_cols = [
        c for c in df.columns if c.startswith("label_reason_") or c.startswith("label_reason_rule_")
    ]
    if not reason_label_cols:
        reason_label_cols = ["weak_label"]
    num_cols, cat_cols, reason_cols = infer_columns(df)
    llm_emb_cols = get_llm_embedding_cols(df)
    llm_reason_prob_cols: List[str | None] = []
    for col in reason_label_cols:
        pc = _llm_prob_col_for_reason_label(col)
        llm_reason_prob_cols.append(pc if pc in df.columns else None)
    llm_anomaly_prob_col = LLM_ANOMALY_PROB_COL if LLM_ANOMALY_PROB_COL in df.columns else None

    train_bundle, val_bundle, test_bundle, cat_maps = build_tensors(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        num_cols=num_cols,
        cat_cols=cat_cols,
        reason_cols=reason_cols,
        reason_label_cols=reason_label_cols,
        llm_emb_cols=llm_emb_cols,
        llm_reason_prob_cols=llm_reason_prob_cols,
        llm_anomaly_prob_col=llm_anomaly_prob_col,
    )

    cat_cardinalities = [len(cat_maps[c]) for c in cat_cols]
    reason_dim = train_bundle.reason_x.size(1)
    reason_label_dim = train_bundle.y_reason.size(1)
    llm_dim = train_bundle.llm_x.size(1)

    teacher = TeacherModel(
        num_dim=train_bundle.num_x.size(1),
        cat_cardinalities=cat_cardinalities,
        reason_dim=reason_dim,
        llm_dim=llm_dim,
        reason_label_dim=reason_label_dim,
        hidden_dim=128,
        cat_emb_dim=16,
    )
    t_ckpt = args.checkpoint_dir / "teacher.pt"
    if not t_ckpt.is_file():
        raise FileNotFoundError(f"Missing {t_ckpt}")
    teacher.load_state_dict(torch.load(t_ckpt, map_location="cpu"))
    teacher.eval().to(device)

    student: StudentModel | None = None
    if args.eval_student:
        s_ckpt = args.checkpoint_dir / "student.pt"
        if not s_ckpt.is_file():
            raise FileNotFoundError(f"Missing {s_ckpt}")
        student = StudentModel(
            num_dim=train_bundle.num_x.size(1),
            hidden_dim=64,
            reason_dim=reason_label_dim,
            teacher_repr_dim=128,
        )
        student.load_state_dict(torch.load(s_ckpt, map_location="cpu"))
        student.eval().to(device)

    out: Dict[str, object] = {
        "schema": "analyze_teacher_on_llm_covered_rows_v1",
        "aligned_csv": str(args.aligned_csv.resolve()),
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "split_mode": args.split_mode,
        "seed": args.seed,
        "llm_coverage_col": args.llm_coverage_col,
        "threshold_teacher": thr_teacher,
        "threshold_student": thr_student,
        "split_info": split_info,
        "inferred_from_train_report": {
            "holdout_industry": inferred_industry,
            "holdout_voltage": inferred_voltage,
        },
        "splits": {},
    }

    for name, part_df, bundle in (
        ("train", train_df, train_bundle),
        ("val", val_df, val_bundle),
        ("test", test_df, test_bundle),
    ):
        mask = _llm_row_mask(part_df, args.llm_coverage_col)
        idx = np.flatnonzero(mask)
        n_cov = int(idx.size)
        if n_cov == 0:
            out["splits"][name] = {"n_llm_covered": 0, "skipped": True}
            continue
        y_sub = part_df["weak_label"].iloc[idx].astype(float).to_numpy()
        n_pos = int((y_sub > 0.5).sum())
        sub_bundle = _slice_bundle(bundle, idx)
        loader = DataLoader(FeeDataset(sub_bundle), batch_size=args.batch_size, shuffle=False)
        t_metrics = _eval_teacher_probs(teacher, loader, device, thr_teacher)
        block: Dict[str, object] = {
            "n_llm_covered": n_cov,
            "n_weak_pos": n_pos,
            "n_weak_neg": n_cov - n_pos,
            "teacher": t_metrics,
        }
        if student is not None:
            block["student"] = _eval_student_probs(student, loader, device, thr_student)
        out["splits"][name] = block

    out_path = args.checkpoint_dir / "llm_covered_subset_metrics.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[OK] wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
