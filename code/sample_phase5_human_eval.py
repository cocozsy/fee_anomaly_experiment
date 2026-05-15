#!/usr/bin/env python3
"""Sample N rows from mock vs openai LLM CSVs for human evaluation of llm_explanation.

Stratification (best effort):
- weak_label in {0, 1}
- llm_anomaly_prob bucket: low (<0.3), mid (0.3-0.6), high (>=0.6) — averaged over mock & openai.

Output CSV columns (UTF-8 BOM so Excel handles 中文):
  客户ID, month, weak_label, 行业编码, 电压等级, env_peer_fee_vs_z, event_count,
  anomaly_prob_mock, anomaly_prob_openai, explanation_mock, explanation_openai,
  score_mock, score_openai, notes
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def _bucket(v: float) -> str:
    if v < 0.3:
        return "low"
    if v < 0.6:
        return "mid"
    return "high"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock_csv", type=Path, required=True)
    parser.add_argument("--openai_csv", type=Path, required=True)
    parser.add_argument("--aligned_csv", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    m = pd.read_csv(args.mock_csv)
    o = pd.read_csv(args.openai_csv)
    for df in (m, o):
        df["客户ID"] = df["客户ID"].astype(str)
        df["month"] = df["month"].astype(str)

    a_cols = ["客户ID", "month", "weak_label", "行业编码", "电压等级", "env_peer_fee_vs_z", "event_count"]
    aligned = pd.read_csv(args.aligned_csv, usecols=lambda c: c in set(a_cols))
    aligned["客户ID"] = aligned["客户ID"].astype(str)
    aligned["month"] = aligned["month"].astype(str)

    joined = m.merge(o, on=["客户ID", "month"], suffixes=("_mock", "_openai"))
    joined = joined.merge(aligned, on=["客户ID", "month"], how="left")

    if joined.empty:
        raise SystemExit("[sample] joined frame is empty; check CSV inputs.")

    avg_prob = (joined["llm_anomaly_prob_mock"].fillna(0) + joined["llm_anomaly_prob_openai"].fillna(0)) / 2.0
    joined["_bucket"] = avg_prob.map(_bucket)
    joined["_weak"] = joined["weak_label"].fillna(0).astype(int).astype(str)
    joined["_strata"] = joined["_weak"] + "/" + joined["_bucket"]

    rng = np.random.default_rng(args.seed)
    strata = sorted(joined["_strata"].unique().tolist())
    n_per = max(1, args.n // max(1, len(strata)))
    picks: List[pd.DataFrame] = []
    for s in strata:
        sub = joined[joined["_strata"] == s]
        take = min(len(sub), n_per)
        if take > 0:
            picks.append(sub.sample(n=take, random_state=int(rng.integers(0, 10**9))))
    sampled = pd.concat(picks, ignore_index=True) if picks else joined.head(0)

    if len(sampled) < args.n:
        remaining = args.n - len(sampled)
        leftover = joined.drop(sampled.index, errors="ignore")
        if not leftover.empty:
            extra = leftover.sample(n=min(remaining, len(leftover)), random_state=int(rng.integers(0, 10**9)))
            sampled = pd.concat([sampled, extra], ignore_index=True)

    sampled = sampled.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    sampled.insert(0, "row_id", range(1, len(sampled) + 1))

    out_cols = [
        "row_id", "客户ID", "month", "weak_label", "行业编码", "电压等级",
        "env_peer_fee_vs_z", "event_count",
        "llm_anomaly_prob_mock", "llm_anomaly_prob_openai",
        "llm_explanation_mock", "llm_explanation_openai",
    ]
    final = sampled[[c for c in out_cols if c in sampled.columns]].copy()
    final["score_mock"] = ""
    final["score_openai"] = ""
    final["preferred (mock/openai/tie)"] = ""
    final["notes"] = ""

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] wrote {args.output_csv} with {len(final)} rows (seed={args.seed})")


if __name__ == "__main__":
    main()
