#!/usr/bin/env python3
"""Evaluate stage-2 gate metrics including subset recall uplift."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from models import StudentModel
from train_distill import (
    FeeDataset,
    binary_metrics_from_prob,
    build_tensors,
    get_llm_embedding_cols,
    infer_columns,
    set_seed,
    split_dataset,
)


def _collect_student_prob_and_y(model: StudentModel, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_prob: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    with torch.no_grad():
        for num_x, _, _, _, y_weak, _, _, _, _, _ in loader:
            num_x = num_x.to(device)
            prob = torch.sigmoid(model(num_x)["anomaly_logit"]).detach().cpu().numpy()
            all_prob.append(prob)
            all_y.append(y_weak.numpy())
    return np.concatenate(all_prob), np.concatenate(all_y)


def _evaluate_subset_recall(
    aligned_csv: Path,
    llm_csv: Path,
    ckpt_path: Path,
    threshold: float = 0.5,
    seed: int = 42,
    split_mode: str = "temporal",
) -> Dict[str, float]:
    set_seed(seed)
    df = pd.read_csv(aligned_csv)
    llm_df = pd.read_csv(llm_csv)
    llm_df["客户ID"] = llm_df["客户ID"].astype(str)
    llm_df["month"] = llm_df["month"].astype(str)
    df["客户ID"] = df["客户ID"].astype(str)
    df["month"] = df["month"].astype(str)
    df = df.merge(llm_df, on=["客户ID", "month"], how="left")

    train_df, val_df, test_df, split_info = split_dataset(df, mode=split_mode, seed=seed)
    reason_label_cols = [c for c in df.columns if c.startswith("label_reason_") or c.startswith("label_reason_rule_")]
    num_cols, cat_cols, reason_cols = infer_columns(df)
    llm_emb_cols = get_llm_embedding_cols(df)
    llm_reason_prob_cols = [f"llm_reason_prob_{c.replace('label_', '')}" if f"llm_reason_prob_{c.replace('label_', '')}" in df.columns else None for c in reason_label_cols]
    llm_anomaly_prob_col = "llm_anomaly_prob" if "llm_anomaly_prob" in df.columns else None

    train_bundle, val_bundle, test_bundle, _ = build_tensors(
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
    test_loader = DataLoader(FeeDataset(test_bundle), batch_size=1024, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = StudentModel(
        num_dim=train_bundle.num_x.size(1),
        hidden_dim=64,
        reason_dim=train_bundle.y_reason.size(1),
        teacher_repr_dim=128,
    )
    state = torch.load(ckpt_path, map_location="cpu")
    student.load_state_dict(state)
    student.to(device)

    prob, y = _collect_student_prob_and_y(student, test_loader, device)
    pred = (prob >= threshold).astype(np.float32)

    rule_cols = [c for c in ["reason_rule_METER_INCREASE", "reason_rule_READING_MISMATCH"] if c in test_df.columns]
    if rule_cols:
        no_rule_hit = (test_df[rule_cols].fillna(0).sum(axis=1) == 0).values
    else:
        no_rule_hit = np.ones(len(test_df), dtype=bool)
    weak_pos = (test_df["weak_label"].fillna(0).astype(float).values == 1.0)
    subset_mask = no_rule_hit & weak_pos

    tp = float(((pred == 1) & (y == 1)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    overall_recall = tp / (tp + fn + 1e-6)

    subset_tp = float(((pred == 1) & (y == 1) & subset_mask).sum())
    subset_fn = float(((pred == 0) & (y == 1) & subset_mask).sum())
    subset_recall = subset_tp / (subset_tp + subset_fn + 1e-6)

    overall_metrics = binary_metrics_from_prob(torch.tensor(prob), torch.tensor(y), threshold=threshold)
    return {
        "split_mode": split_mode,
        "split_info": {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in split_info.items()},
        "overall_recall": float(overall_recall),
        "subset_recall_no_rule_and_weak_pos": float(subset_recall),
        "subset_size": int(subset_mask.sum()),
        "test_size": int(len(test_df)),
        "overall_f1": float(overall_metrics["f1"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate stage-2 gate uplift metrics.")
    parser.add_argument("--env_self_csv", type=Path, default=Path("data/aligned/aligned_customer_month_env.csv"))
    parser.add_argument("--env_full_csv", type=Path, default=Path("data/aligned/aligned_customer_month_env_full.csv"))
    parser.add_argument("--llm_csv", type=Path, default=Path("data/aligned/aligned_customer_month_llm_features.csv"))
    parser.add_argument("--env_self_ckpt", type=Path, default=Path("checkpoints/env_self_v1/student.pt"))
    parser.add_argument("--env_full_ckpt", type=Path, default=Path("checkpoints/env_full_v1/student.pt"))
    parser.add_argument("--output_json", type=Path, default=Path("experiments/stage2_gate_eval.json"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--split_mode",
        type=str,
        default="temporal",
        help="Primary split for gate metrics (must match training if comparing checkpoints).",
    )
    parser.add_argument(
        "--extra_split_modes",
        type=str,
        default="",
        help="Comma-separated extra modes, e.g. cross_domain_industry,cross_domain_voltage (evaluated on both CSVs).",
    )
    args = parser.parse_args()

    env_self_metrics = _evaluate_subset_recall(
        args.env_self_csv,
        args.llm_csv,
        args.env_self_ckpt,
        threshold=args.threshold,
        split_mode=args.split_mode,
    )
    env_full_metrics = _evaluate_subset_recall(
        args.env_full_csv,
        args.llm_csv,
        args.env_full_ckpt,
        threshold=args.threshold,
        split_mode=args.split_mode,
    )

    result: Dict[str, object] = {
        "threshold": args.threshold,
        "primary_split_mode": args.split_mode,
        "env_self_v1": env_self_metrics,
        "env_full_v1": env_full_metrics,
        "uplift": {
            "overall_f1": env_full_metrics["overall_f1"] - env_self_metrics["overall_f1"],
            "overall_recall": env_full_metrics["overall_recall"] - env_self_metrics["overall_recall"],
            "subset_recall_no_rule_and_weak_pos": env_full_metrics["subset_recall_no_rule_and_weak_pos"]
            - env_self_metrics["subset_recall_no_rule_and_weak_pos"],
        },
    }

    extras = [m.strip() for m in args.extra_split_modes.split(",") if m.strip()]
    if extras:
        by_split: Dict[str, object] = {}
        for mode in extras:
            try:
                s_self = _evaluate_subset_recall(
                    args.env_self_csv, args.llm_csv, args.env_self_ckpt, threshold=args.threshold, split_mode=mode
                )
                s_full = _evaluate_subset_recall(
                    args.env_full_csv, args.llm_csv, args.env_full_ckpt, threshold=args.threshold, split_mode=mode
                )
                by_split[mode] = {
                    "env_self_v1": s_self,
                    "env_full_v1": s_full,
                    "uplift": {
                        "overall_f1": s_full["overall_f1"] - s_self["overall_f1"],
                        "overall_recall": s_full["overall_recall"] - s_self["overall_recall"],
                        "subset_recall_no_rule_and_weak_pos": s_full["subset_recall_no_rule_and_weak_pos"]
                        - s_self["subset_recall_no_rule_and_weak_pos"],
                    },
                }
            except (ValueError, KeyError) as e:
                by_split[mode] = {"error": str(e)}
        result["by_extra_split_mode"] = by_split

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[OK] wrote gate metrics: {args.output_json}")


if __name__ == "__main__":
    main()
