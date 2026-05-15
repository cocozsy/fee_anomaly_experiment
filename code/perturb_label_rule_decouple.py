#!/usr/bin/env python3
"""Decouple weak_label from rule-hit columns (Part C).

Two perturbations (reproducible via --seed):

1) **Clear rules on weak positives** (default 7.5% of rows with weak_label=1 AND any
   reason_rule_* > 0): set all `reason_rule_*` and matching `label_reason_rule_*` to 0,
   keep weak_label=1. Creates "no rule hit but still positive weak label" when the
   row remains positive due to read_has_abn / fee_mom_ratio / residual abnormal counts.

2) **Rule false positives** (default 5% of rows with any reason_rule_* > 0): set
   weak_label=0, leave reason_rule_* unchanged. Creates "rule fired but label says normal".

Rows selected for (1) are excluded from the pool for (2) to avoid conflicting edits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _reason_rule_cols(df: pd.DataFrame) -> List[str]:
    return sorted([c for c in df.columns if c.startswith("reason_rule_")])


def _label_rule_cols(df: pd.DataFrame, reason_cols: List[str]) -> List[str]:
    out: List[str] = []
    for rc in reason_cols:
        suffix = rc.replace("reason_rule_", "label_reason_rule_", 1)
        if suffix in df.columns:
            out.append(suffix)
    return out


def apply_decouple_perturbation(
    df: pd.DataFrame,
    seed: int,
    frac_clear_rules: float,
    frac_rule_false_positive: float,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    out = df.copy()
    rng = np.random.default_rng(seed)

    reason_cols = _reason_rule_cols(out)
    if not reason_cols:
        raise ValueError("No reason_rule_* columns found; run build_aligned_dataset first.")

    rule_vals = out[reason_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    orig_rule_sum = rule_vals.sum(axis=1)
    rule_hit = orig_rule_sum > 0
    weak = pd.to_numeric(out.get("weak_label", 0), errors="coerce").fillna(0).astype(int) == 1

    pool_clear = out.index[weak & rule_hit]
    n_clear = int(np.floor(len(pool_clear) * frac_clear_rules))
    if n_clear > 0:
        idx_clear = rng.choice(pool_clear, size=n_clear, replace=False)
    else:
        idx_clear = np.array([], dtype=int)

    for rc in reason_cols:
        out.loc[idx_clear, rc] = 0.0
    for lc in _label_rule_cols(out, reason_cols):
        out.loc[idx_clear, lc] = 0

    cleared_mask = pd.Series(False, index=out.index)
    if len(idx_clear) > 0:
        cleared_mask.loc[idx_clear] = True
    # False positives: among rule-hit rows not cleared, flip weak_label to 0.
    pool_fp = out.index[rule_hit & ~cleared_mask]
    n_fp_target = int(np.floor(rule_hit.sum() * frac_rule_false_positive))
    n_fp_target = max(0, min(n_fp_target, len(pool_fp)))
    if n_fp_target > 0:
        idx_fp = rng.choice(pool_fp, size=n_fp_target, replace=False)
        out.loc[idx_fp, "weak_label"] = 0
    else:
        idx_fp = np.array([], dtype=int)

    meta: Dict[str, object] = {
        "seed": seed,
        "frac_clear_rules": frac_clear_rules,
        "frac_rule_false_positive": frac_rule_false_positive,
        "reason_rule_columns": reason_cols,
        "n_rows_rule_hit_before": int(rule_hit.sum()),
        "n_cleared_rules_on_weak_pos": int(len(idx_clear)),
        "n_rule_false_positive_weak0": int(len(idx_fp)),
        "weak_label_positive_ratio_after": float(pd.to_numeric(out["weak_label"], errors="coerce").fillna(0).mean()),
    }

    # Post-hoc diagnostics
    rule_vals_after = out[reason_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    rule_sum_after = rule_vals_after.sum(axis=1)
    weak_after = pd.to_numeric(out["weak_label"], errors="coerce").fillna(0).astype(int) == 1
    meta["n_weak_pos_no_rule_after"] = int((weak_after & (rule_sum_after == 0)).sum())
    meta["n_rule_hit_weak0_after"] = int(((rule_sum_after > 0) & ~weak_after).sum())

    return out, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Decouple weak_label from reason_rule_* (Part C).")
    parser.add_argument(
        "--input_csv",
        type=Path,
        default=Path("data/aligned/aligned_customer_month.csv"),
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("data/aligned/aligned_customer_month_decoupled.csv"),
    )
    parser.add_argument(
        "--output_meta",
        type=Path,
        default=Path("data/aligned/aligned_customer_month_decoupled_metadata.json"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--frac_clear_rules",
        type=float,
        default=0.075,
        help="Fraction of (weak_label=1 & rule-hit) rows to clear all reason_rule_* (default 7.5%%).",
    )
    parser.add_argument(
        "--frac_rule_false_positive",
        type=float,
        default=0.05,
        help="Fraction of all rule-hit rows to set weak_label=0 (default 5%%).",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    out, meta = apply_decouple_perturbation(
        df,
        seed=args.seed,
        frac_clear_rules=args.frac_clear_rules,
        frac_rule_false_positive=args.frac_rule_false_positive,
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    meta["input_csv"] = str(args.input_csv)
    meta["output_csv"] = str(args.output_csv)
    meta["row_count"] = int(len(out))
    args.output_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"[OK] wrote {args.output_csv}")
    print(f"[OK] wrote {args.output_meta}")


if __name__ == "__main__":
    main()
