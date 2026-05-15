#!/usr/bin/env python3
"""Compare mock vs openai LLM-feature CSVs on the same (客户ID, month) keys.

Outputs JSON with:
- coverage stats (rows in each, joined rows)
- per-column metrics: pearson, spearman, MAE, mean diff, distribution summaries
- prefix embedding: row-wise cosine similarity (mean / std / quantiles) and column-wise pearson
- agreement at threshold 0.5 on llm_anomaly_prob
- (optional) join with aligned CSV to add weak_label coverage / stratified summaries

Usage:
  python code/compare_phase5_mock_vs_openai.py \
    --mock_csv   data/aligned/aligned_customer_month_llm_features_v2_due_sbr_pilot_mock_200.csv \
    --openai_csv data/aligned/aligned_customer_month_llm_features_v2_due_sbr_pilot_openai_200.csv \
    --aligned_csv data/aligned/aligned_customer_month_decoupled_env_full_d3c.csv \
    --output_json experiments/phase5_pilot_compare.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def _safe_corr(a: np.ndarray, b: np.ndarray, kind: str) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 3:
        return float("nan")
    aa, bb = a[mask], b[mask]
    if np.std(aa) < 1e-12 or np.std(bb) < 1e-12:
        return float("nan")
    if kind == "pearson":
        return float(np.corrcoef(aa, bb)[0, 1])
    if kind == "spearman":
        ar = pd.Series(aa).rank().to_numpy()
        br = pd.Series(bb).rank().to_numpy()
        return float(np.corrcoef(ar, br)[0, 1])
    raise ValueError(kind)


def _summary(arr: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {"count": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "p25": float(np.quantile(arr, 0.25)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "max": float(arr.max()),
    }


def _row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = (a * b).sum(axis=1)
    da = np.linalg.norm(a, axis=1)
    db = np.linalg.norm(b, axis=1)
    denom = np.maximum(da * db, 1e-12)
    return num / denom


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock_csv", type=Path, required=True)
    parser.add_argument("--openai_csv", type=Path, required=True)
    parser.add_argument("--aligned_csv", type=Path, default=None, help="Optional join for weak_label / 行业 etc.")
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    m = pd.read_csv(args.mock_csv)
    o = pd.read_csv(args.openai_csv)
    for df in (m, o):
        df["客户ID"] = df["客户ID"].astype(str)
        df["month"] = df["month"].astype(str)

    joined = m.merge(o, on=["客户ID", "month"], suffixes=("_mock", "_openai"))
    n = len(joined)
    coverage = {
        "mock_rows": int(len(m)),
        "openai_rows": int(len(o)),
        "joined_rows": int(n),
    }

    # If aligned CSV present, attach weak_label for stratified summary.
    weak_summary: Dict[str, object] | None = None
    if args.aligned_csv is not None:
        a = pd.read_csv(args.aligned_csv, usecols=lambda c: c in {"客户ID", "month", "weak_label"})
        a["客户ID"] = a["客户ID"].astype(str)
        a["month"] = a["month"].astype(str)
        joined = joined.merge(a, on=["客户ID", "month"], how="left")
        if "weak_label" in joined.columns:
            wl = joined["weak_label"].fillna(0).astype(int)
            weak_summary = {
                "weak_label_pos": int(wl.sum()),
                "weak_label_neg": int((wl == 0).sum()),
            }

    # 1) anomaly prob
    a_m = joined["llm_anomaly_prob_mock"].to_numpy()
    a_o = joined["llm_anomaly_prob_openai"].to_numpy()
    anomaly_block = {
        "summary_mock": _summary(a_m),
        "summary_openai": _summary(a_o),
        "pearson": _safe_corr(a_m, a_o, "pearson"),
        "spearman": _safe_corr(a_m, a_o, "spearman"),
        "mae": float(np.mean(np.abs(a_m - a_o))),
        "mean_diff_openai_minus_mock": float(np.mean(a_o - a_m)),
        "agreement_at_threshold": {
            "threshold": float(args.threshold),
            "both_pos": int(((a_m >= args.threshold) & (a_o >= args.threshold)).sum()),
            "both_neg": int(((a_m < args.threshold) & (a_o < args.threshold)).sum()),
            "mock_pos_only": int(((a_m >= args.threshold) & (a_o < args.threshold)).sum()),
            "openai_pos_only": int(((a_m < args.threshold) & (a_o >= args.threshold)).sum()),
        },
    }
    if weak_summary is not None and "weak_label" in joined.columns:
        wl = joined["weak_label"].fillna(0).astype(int).to_numpy()
        for label in (0, 1):
            mask = wl == label
            if mask.sum() > 0:
                anomaly_block.setdefault("by_weak_label", {})[str(label)] = {
                    "count": int(mask.sum()),
                    "mock_mean": float(np.mean(a_m[mask])),
                    "openai_mean": float(np.mean(a_o[mask])),
                    "mock_pos_rate@thr": float(np.mean(a_m[mask] >= args.threshold)),
                    "openai_pos_rate@thr": float(np.mean(a_o[mask] >= args.threshold)),
                }

    # 2) reason probs
    reason_cols = sorted(c[: -len("_mock")] for c in joined.columns
                         if c.startswith("llm_reason_prob_") and c.endswith("_mock"))
    reason_block: Dict[str, object] = {}
    for col in reason_cols:
        m_arr = joined[f"{col}_mock"].to_numpy()
        o_arr = joined[f"{col}_openai"].to_numpy()
        reason_block[col] = {
            "summary_mock": _summary(m_arr),
            "summary_openai": _summary(o_arr),
            "pearson": _safe_corr(m_arr, o_arr, "pearson"),
            "spearman": _safe_corr(m_arr, o_arr, "spearman"),
            "mae": float(np.mean(np.abs(m_arr - o_arr))),
            "mean_diff_openai_minus_mock": float(np.mean(o_arr - m_arr)),
        }

    # 3) prefix embedding cosine
    emb_cols = sorted(c[: -len("_mock")] for c in joined.columns
                      if c.startswith("llm_prefix_emb_") and c.endswith("_mock"))
    prefix_block: Dict[str, object] = {}
    if emb_cols:
        em_m = joined[[f"{c}_mock" for c in emb_cols]].to_numpy(dtype=np.float64)
        em_o = joined[[f"{c}_openai" for c in emb_cols]].to_numpy(dtype=np.float64)
        cos = _row_cosine(em_m, em_o)
        prefix_block["dim"] = len(emb_cols)
        prefix_block["row_cosine"] = _summary(cos)
        per_dim = []
        for i, c in enumerate(emb_cols):
            per_dim.append({
                "dim": c,
                "pearson": _safe_corr(em_m[:, i], em_o[:, i], "pearson"),
                "mae": float(np.mean(np.abs(em_m[:, i] - em_o[:, i]))),
            })
        prefix_block["per_dim"] = per_dim

    # 4) explanation length / non-empty rate
    expl_block: Dict[str, object] = {}
    if "llm_explanation_mock" in joined.columns and "llm_explanation_openai" in joined.columns:
        m_len = joined["llm_explanation_mock"].fillna("").astype(str).str.len().to_numpy()
        o_len = joined["llm_explanation_openai"].fillna("").astype(str).str.len().to_numpy()
        expl_block = {
            "mock_len": _summary(m_len),
            "openai_len": _summary(o_len),
            "mock_nonempty": int((m_len > 0).sum()),
            "openai_nonempty": int((o_len > 0).sum()),
        }

    out = {
        "schema": "phase5_mock_vs_openai_compare_v1",
        "inputs": {
            "mock_csv": str(args.mock_csv.resolve()),
            "openai_csv": str(args.openai_csv.resolve()),
            "aligned_csv": str(args.aligned_csv.resolve()) if args.aligned_csv is not None else None,
        },
        "coverage": coverage,
        "anomaly_prob": anomaly_block,
        "reason_prob": reason_block,
        "prefix_embedding": prefix_block,
        "explanation": expl_block,
    }
    if weak_summary is not None:
        out["weak_label_coverage"] = weak_summary

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] wrote {args.output_json}")


if __name__ == "__main__":
    main()
