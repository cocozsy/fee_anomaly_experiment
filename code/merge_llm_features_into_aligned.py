#!/usr/bin/env python3
"""Left-merge a partial LLM feature CSV into a full aligned customer-month table.

Use case: pilot OpenAI/mock features exist only for a subset of (客户ID, month);
`train_distill.py` merges on keys and uses NaN masks for soft LLM supervision
and fillna(0) for prefix columns on rows without LLM rows.

Outputs a new aligned CSV plus a small merge provenance JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pandas as pd


def _llm_side_columns(df: pd.DataFrame) -> List[str]:
    keys = {"客户ID", "month"}
    out: List[str] = []
    for c in df.columns:
        if c in keys:
            continue
        if c.startswith("llm_"):
            out.append(c)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned_csv", type=Path, required=True, help="Full aligned table.")
    parser.add_argument(
        "--llm_features_csv",
        type=Path,
        required=True,
        help="LLM feature table keyed by 客户ID, month (subset or full).",
    )
    parser.add_argument("--output_csv", type=Path, required=True, help="Merged output path.")
    args = parser.parse_args()

    base = pd.read_csv(args.aligned_csv)
    llm = pd.read_csv(args.llm_features_csv)
    for col in ("客户ID", "month"):
        if col not in base.columns or col not in llm.columns:
            raise ValueError(f"Both CSVs must contain column {col!r}")
    base["客户ID"] = base["客户ID"].astype(str)
    base["month"] = base["month"].astype(str)
    llm["客户ID"] = llm["客户ID"].astype(str)
    llm["month"] = llm["month"].astype(str)

    llm_cols = _llm_side_columns(llm)
    if not llm_cols:
        raise ValueError("llm_features_csv has no llm_* columns to merge.")

    overlap = [c for c in llm_cols if c in base.columns]
    if overlap:
        raise ValueError(
            "aligned_csv already contains LLM columns; drop them first or use a clean aligned file. "
            f"Overlap: {overlap[:20]}"
        )

    llm_sub = llm[["客户ID", "month"] + llm_cols].copy()
    merged = base.merge(llm_sub, on=["客户ID", "month"], how="left")

    key = merged["客户ID"].astype(str) + "|" + merged["month"].astype(str)
    llm_key = set(llm_sub["客户ID"].astype(str) + "|" + llm_sub["month"].astype(str))
    covered = merged.apply(lambda r: f"{r['客户ID']}|{r['month']}" in llm_key, axis=1)
    n_covered = int(covered.sum())

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_csv, index=False)

    prov = {
        "schema": "merge_llm_features_into_aligned_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv.copy(),
        "aligned_csv": str(args.aligned_csv.resolve()),
        "llm_features_csv": str(args.llm_features_csv.resolve()),
        "output_csv": str(args.output_csv.resolve()),
        "aligned_rows": int(len(base)),
        "llm_feature_rows": int(len(llm)),
        "merged_rows": int(len(merged)),
        "llm_columns_merged": llm_cols,
        "rows_with_llm_match": n_covered,
        "fraction_with_llm": round(n_covered / max(len(merged), 1), 6),
    }
    prov_path = args.output_csv.parent / f"{args.output_csv.stem}_merge_provenance.json"
    prov_path.write_text(json.dumps(prov, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] wrote {args.output_csv} ({len(merged)} rows, {n_covered} with LLM keys)")
    print(f"[OK] provenance {prov_path}")


if __name__ == "__main__":
    main()
