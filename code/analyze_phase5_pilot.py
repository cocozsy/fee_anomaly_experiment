#!/usr/bin/env python3
"""Deeper analysis on the N=200 mock vs openai pilot to decide whether scaling is worth it.

Pure numpy + pandas (no sklearn/scipy). Adds the discriminative power not covered by
compare_phase5_mock_vs_openai.py:

1. AUROC / AUPRC / ECE of llm_anomaly_prob vs weak_label (mock & openai).
2. Per-reason AUROC / AUPRC of llm_reason_prob_* vs the matching label_reason_*.
3. Prefix embedding: row-wise cosine to weak=1 centroid - weak=0 centroid; AUROC of that score.
4. Bootstrap 95% CI for openai AUPRC and mean cosine, plus P(openai_AUPRC > mock_AUPRC).
5. A printed verdict: whether scaling to a larger N is worth the cost.

Outputs JSON to --output_json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibration import calibration_report  # noqa: E402


# ---------- pure-numpy metrics ----------

def auroc(prob: np.ndarray, y: np.ndarray) -> float:
    p = np.asarray(prob, dtype=np.float64).reshape(-1)
    t = np.asarray(y, dtype=np.int64).reshape(-1)
    pos = int((t == 1).sum())
    neg = int((t == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(-p, kind="mergesort")
    cum_pos = 0
    cum_neg = 0
    fpr = [0.0]
    tpr = [0.0]
    i = 0
    n = len(p)
    while i < n:
        j = i
        while j + 1 < n and p[order[j + 1]] == p[order[i]]:
            j += 1
        seg = t[order[i : j + 1]]
        cum_pos += int(seg.sum())
        cum_neg += int((seg == 0).sum())
        fpr.append(cum_neg / neg)
        tpr.append(cum_pos / pos)
        i = j + 1
    return float(np.trapezoid(np.array(tpr), np.array(fpr)))


def auprc(prob: np.ndarray, y: np.ndarray) -> float:
    p = np.asarray(prob, dtype=np.float64).reshape(-1)
    t = np.asarray(y, dtype=np.int64).reshape(-1)
    if t.sum() == 0 or len(t) == 0:
        return 0.0
    order = np.argsort(-p, kind="mergesort")
    t_sorted = t[order]
    tp = np.cumsum(t_sorted)
    fp = np.cumsum(1 - t_sorted)
    precisions = tp / np.maximum(tp + fp, 1e-9)
    recalls = tp / max(int(t.sum()), 1)
    recalls_prev = np.concatenate([[0.0], recalls[:-1]])
    return float(np.sum(precisions * (recalls - recalls_prev)))


def safe_corr(a: np.ndarray, b: np.ndarray, kind: str = "pearson") -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if a.size < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    if kind == "spearman":
        a = pd.Series(a).rank().to_numpy()
        b = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(a, b)[0, 1])


def bootstrap(stat_fn, prob: np.ndarray, y: np.ndarray, n_iter: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(prob)
    vals = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        vals[i] = stat_fn(prob[idx], y[idx])
    return vals


def positive_rate_at(prob: np.ndarray, y: np.ndarray, thr: float) -> Dict[str, float]:
    p = np.asarray(prob, dtype=np.float64)
    t = np.asarray(y, dtype=np.int64)
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (t == 1)).sum())
    fp = int(((pred == 1) & (t == 0)).sum())
    fn = int(((pred == 0) & (t == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


# ---------- main analysis ----------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock_csv", type=Path, required=True)
    parser.add_argument("--openai_csv", type=Path, required=True)
    parser.add_argument("--aligned_csv", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    m = pd.read_csv(args.mock_csv)
    o = pd.read_csv(args.openai_csv)
    for df in (m, o):
        df["客户ID"] = df["客户ID"].astype(str)
        df["month"] = df["month"].astype(str)

    reason_label_cols = sorted(
        c for c in pd.read_csv(args.aligned_csv, nrows=0).columns
        if c.startswith("label_reason_") or c.startswith("label_reason_rule_")
    )
    use_cols = ["客户ID", "month", "weak_label"] + reason_label_cols
    a = pd.read_csv(args.aligned_csv, usecols=lambda c: c in set(use_cols))
    a["客户ID"] = a["客户ID"].astype(str)
    a["month"] = a["month"].astype(str)

    j = m.merge(o, on=["客户ID", "month"], suffixes=("_mock", "_openai"))
    j = j.merge(a, on=["客户ID", "month"], how="left")

    n = len(j)
    y = j["weak_label"].fillna(0).astype(int).to_numpy()
    pos_count = int(y.sum())
    neg_count = int((y == 0).sum())

    a_m = j["llm_anomaly_prob_mock"].to_numpy(dtype=np.float64)
    a_o = j["llm_anomaly_prob_openai"].to_numpy(dtype=np.float64)

    rng = np.random.default_rng(args.seed)

    # --- 1. Discrimination of anomaly_prob vs weak_label
    block_anom: Dict[str, object] = {
        "n_total": int(n),
        "weak_pos": pos_count,
        "weak_neg": neg_count,
        "mock":   {"auroc": auroc(a_m, y), "auprc": auprc(a_m, y)},
        "openai": {"auroc": auroc(a_o, y), "auprc": auprc(a_o, y)},
    }
    # Bootstrap AUPRC
    if pos_count >= 3 and args.n_bootstrap > 0:
        boot_m = bootstrap(auprc, a_m, y, n_iter=args.n_bootstrap, seed=args.seed)
        boot_o = bootstrap(auprc, a_o, y, n_iter=args.n_bootstrap, seed=args.seed + 1)
        block_anom["bootstrap"] = {
            "n_iter": int(args.n_bootstrap),
            "mock_auprc_p2.5_p97.5": [float(np.quantile(boot_m, 0.025)), float(np.quantile(boot_m, 0.975))],
            "openai_auprc_p2.5_p97.5": [float(np.quantile(boot_o, 0.025)), float(np.quantile(boot_o, 0.975))],
            "P(openai>mock)": float(np.mean(boot_o > boot_m)),
            "median_diff_openai_minus_mock": float(np.median(boot_o - boot_m)),
        }

    # ECE & reliability
    block_anom["calibration_mock"] = calibration_report(a_m, y, n_bins=10)
    block_anom["calibration_openai"] = calibration_report(a_o, y, n_bins=10)

    # operating points
    for thr in (0.3, 0.5):
        block_anom.setdefault("at_thr", {})[str(thr)] = {
            "mock": positive_rate_at(a_m, y, thr),
            "openai": positive_rate_at(a_o, y, thr),
        }

    # --- 2. Per-reason discrimination
    reason_block: Dict[str, object] = {}
    for label_col in reason_label_cols:
        reason_name = label_col.replace("label_", "")
        prob_col = f"llm_reason_prob_{reason_name}"
        if prob_col + "_mock" not in j.columns or prob_col + "_openai" not in j.columns:
            continue
        y_r = j[label_col].fillna(0).astype(int).to_numpy()
        if int(y_r.sum()) == 0:
            reason_block[label_col] = {"skipped": "no positives in pilot subset"}
            continue
        p_m = j[prob_col + "_mock"].to_numpy(dtype=np.float64)
        p_o = j[prob_col + "_openai"].to_numpy(dtype=np.float64)
        reason_block[label_col] = {
            "pos_count": int(y_r.sum()),
            "mock": {"auroc": auroc(p_m, y_r), "auprc": auprc(p_m, y_r)},
            "openai": {"auroc": auroc(p_o, y_r), "auprc": auprc(p_o, y_r)},
        }

    # --- 3. Prefix embedding centroid score
    emb_cols = sorted(
        c[: -len("_mock")] for c in j.columns
        if c.startswith("llm_prefix_emb_") and c.endswith("_mock")
    )
    prefix_block: Dict[str, object] = {}
    if emb_cols:
        em_m = j[[f"{c}_mock" for c in emb_cols]].to_numpy(dtype=np.float64)
        em_o = j[[f"{c}_openai" for c in emb_cols]].to_numpy(dtype=np.float64)

        def centroid_score(em: np.ndarray, y: np.ndarray) -> Dict[str, float]:
            em = np.asarray(em, dtype=np.float64)
            y = np.asarray(y, dtype=np.int64)
            pos_mask = y == 1
            neg_mask = y == 0
            mu_pos = em[pos_mask].mean(axis=0) if pos_mask.any() else np.zeros(em.shape[1])
            mu_neg = em[neg_mask].mean(axis=0) if neg_mask.any() else np.zeros(em.shape[1])
            direction = mu_pos - mu_neg
            direction = np.nan_to_num(direction, nan=0.0, posinf=0.0, neginf=0.0)
            norm = float(np.linalg.norm(direction))
            if norm < 1e-12:
                return {"auroc": float("nan"), "auprc": float("nan"), "direction_norm": norm}
            em_c = np.nan_to_num(em, nan=0.0, posinf=0.0, neginf=0.0)
            # float64 dot product: float32 matmul can overflow on wide dynamic-range embeddings.
            scores = (em_c.astype(np.float64) @ direction.astype(np.float64)) / norm
            scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
            return {
                "auroc": auroc(scores, y),
                "auprc": auprc(scores, y),
                "direction_norm": norm,
            }

        prefix_block = {
            "dim": len(emb_cols),
            "mock_centroid": centroid_score(em_m, y),
            "openai_centroid": centroid_score(em_o, y),
        }

    out = {
        "schema": "phase5_pilot_analysis_v1",
        "inputs": {
            "mock_csv": str(args.mock_csv.resolve()),
            "openai_csv": str(args.openai_csv.resolve()),
            "aligned_csv": str(args.aligned_csv.resolve()),
        },
        "anomaly_prob_vs_weak_label": block_anom,
        "reason_prob_vs_reason_label": reason_block,
        "prefix_embedding_centroid_score": prefix_block,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- printed verdict ----
    bm = block_anom["mock"]; bo = block_anom["openai"]
    print(f"[verdict] N={n}  weak_pos={pos_count}  weak_neg={neg_count}")
    print(f"[verdict] anomaly_prob → weak_label  AUROC: mock={bm['auroc']:.3f}  openai={bo['auroc']:.3f}")
    print(f"[verdict] anomaly_prob → weak_label  AUPRC: mock={bm['auprc']:.3f}  openai={bo['auprc']:.3f}")
    boot = block_anom.get("bootstrap")
    if boot is not None:
        print(f"[verdict] bootstrap P(openai>mock AUPRC) = {boot['P(openai>mock)']:.3f}")
        print(f"[verdict] mock 95%CI {boot['mock_auprc_p2.5_p97.5']}  openai 95%CI {boot['openai_auprc_p2.5_p97.5']}")
    print(f"[verdict] ECE: mock={block_anom['calibration_mock']['ece']:.3f}  openai={block_anom['calibration_openai']['ece']:.3f}")
    if prefix_block:
        cm = prefix_block["mock_centroid"]; co = prefix_block["openai_centroid"]
        print(f"[verdict] prefix-centroid AUROC: mock={cm['auroc']:.3f}  openai={co['auroc']:.3f}")
    print(f"[OK] wrote {args.output_json}")


if __name__ == "__main__":
    main()
