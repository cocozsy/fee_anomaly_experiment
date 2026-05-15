#!/usr/bin/env python3
"""Build environment features on aligned customer-month data.

Current scope:
- env_self_*  : self-drift rolling features (stage-1)
- env_peer_*  : previous-month peer baseline features (stage-2 + D3-C)
- env_tariff_*: tariff/environment features (stage-2)

D3-C note: peer aggregates (median/p25/p75/p90/std) are computed on the
*previous month's* same-group slice, then joined back to the current row.
The current customer never appears in the peer slice (it would be in last
month's group, but its label of *this* month cannot leak through last
month's median). Rows with no previous-month peer (first observed month
for that group) get zeros, treated as cold-start.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


SELF_DRIFT_TARGETS: Dict[str, str] = {
    "总电费": "fee",
    "周期结算电量": "energy",
    "unit_price_dev": "unit_price_dev",
    "abnormal_event_count": "abnormal_event",
}
PEER_TARGETS: Dict[str, str] = {
    "总电费": "fee",
    "周期结算电量": "energy",
    "unit_price_dev": "unit_price_dev",
}
PEER_MIN_GROUP_SIZE = 30
PEER_LAG_MONTHS = 1  # D3-C: peer slice = previous month's same-group rows
EVENT_DOMINANT_TYPE_MAP = {
    "none": 0,
    "fee_spike": 1,
    "meter_increase": 2,
    "reading_mismatch": 3,
    "read_abn": 4,
    "unit_price_dev_high": 5,
    "energy_spike": 6,
}


def _calc_window_slope(values: np.ndarray) -> float:
    """Linear slope on valid values only."""
    mask = ~np.isnan(values)
    if int(mask.sum()) < 2:
        return 0.0
    y = values[mask]
    x = np.arange(len(values), dtype=np.float64)[mask]
    x_center = x - x.mean()
    denom = float((x_center ** 2).sum())
    if denom <= 1e-12:
        return 0.0
    y_center = y - y.mean()
    return float((x_center * y_center).sum() / denom)


def _consecutive_increase_until_prev(series: pd.Series) -> pd.Series:
    """Count consecutive monthly increases up to previous month."""
    diff = series.diff()
    increases = (diff > 0).astype(int)
    running: List[int] = []
    count = 0
    for value in increases.tolist():
        if value == 1:
            count += 1
        else:
            count = 0
        running.append(count)
    return pd.Series(running, index=series.index).shift(1).fillna(0.0)


def _safe_float(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    return df


def _prepare_base_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = {"客户ID", "month"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    missing_targets = [c for c in SELF_DRIFT_TARGETS if c not in df.columns]
    if missing_targets:
        raise ValueError(f"Input CSV missing self-drift target columns: {missing_targets}")

    out = df.copy()
    out = _safe_float(out, SELF_DRIFT_TARGETS.keys())
    if "month_dt" not in out.columns:
        out["month_dt"] = pd.to_datetime(out["month"].astype(str) + "-01", errors="coerce")
    else:
        out["month_dt"] = pd.to_datetime(out["month_dt"], errors="coerce")

    out = out.sort_values(["客户ID", "month_dt"]).reset_index(drop=True)
    out["history_len"] = out.groupby("客户ID").cumcount().astype(float)
    return out


def build_env_self_features(out: pd.DataFrame) -> pd.DataFrame:
    grouped = out.groupby("客户ID", group_keys=False)
    for source_col, alias in SELF_DRIFT_TARGETS.items():
        hist = grouped[source_col].shift(1)
        roll3_mean = hist.groupby(out["客户ID"]).rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
        roll3_std = (
            hist.groupby(out["客户ID"]).rolling(window=3, min_periods=2).std().reset_index(level=0, drop=True).fillna(0.0)
        )
        roll6_slope = (
            hist.groupby(out["客户ID"])
            .rolling(window=6, min_periods=2)
            .apply(_calc_window_slope, raw=True)
            .reset_index(level=0, drop=True)
            .fillna(0.0)
        )
        z = (out[source_col] - roll3_mean) / (roll3_std + 1e-6)
        z = z.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        consec = grouped[source_col].apply(_consecutive_increase_until_prev).reset_index(level=0, drop=True)

        out[f"env_self_{alias}_roll3_mean"] = roll3_mean.fillna(0.0)
        out[f"env_self_{alias}_roll3_std"] = roll3_std.fillna(0.0)
        out[f"env_self_{alias}_vs_roll3_z"] = z
        out[f"env_self_{alias}_roll6_slope"] = roll6_slope.fillna(0.0)
        out[f"env_self_{alias}_consec_increase_months"] = consec.fillna(0.0)

    return out


def _add_month_numeric(out: pd.DataFrame) -> pd.DataFrame:
    month_num = pd.to_datetime(out["month"].astype(str) + "-01", errors="coerce").dt.month.fillna(1).astype(int)
    out["env_season_sin"] = np.sin(2.0 * np.pi * month_num / 12.0)
    out["env_season_cos"] = np.cos(2.0 * np.pi * month_num / 12.0)
    return out


def _build_peer_stats(df: pd.DataFrame, group_cols: List[str], value_col: str, alias: str) -> pd.DataFrame:
    grouped = df.groupby(group_cols, dropna=False)[value_col]
    stats = grouped.agg(
        count="count",
        median="median",
        p25=lambda x: np.percentile(x, 25),
        p75=lambda x: np.percentile(x, 75),
        p90=lambda x: np.percentile(x, 90),
        std=lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
    ).reset_index()
    stats = stats.rename(
        columns={
            "count": f"env_peer_{alias}_count",
            "median": f"env_peer_{alias}_median",
            "p25": f"env_peer_{alias}_p25",
            "p75": f"env_peer_{alias}_p75",
            "p90": f"env_peer_{alias}_p90",
            "std": f"env_peer_{alias}_std",
        }
    )
    return stats


def _shift_month_str(month: pd.Series, months: int) -> pd.Series:
    """Shift YYYY-MM strings by `months` (negative = past)."""
    dt = pd.to_datetime(month.astype(str) + "-01", errors="coerce")
    shifted = dt + pd.DateOffset(months=months)
    return shifted.dt.strftime("%Y-%m")


def build_env_peer_features(out: pd.DataFrame) -> pd.DataFrame:
    """D3-C: peer slice uses *previous month's* same-group rows.

    Implementation: compute peer stats keyed by (`month`, group_keys) on the
    full table, then join into the current row using `_peer_month = month -
    PEER_LAG_MONTHS`. This avoids contemporaneous label contamination —
    peer median/p90 of last month cannot reflect this month's anomaly.
    """
    primary_keys = [c for c in ["month", "客户类型", "电压等级", "行业编码"] if c in out.columns]
    if "month" not in primary_keys:
        primary_keys = ["month"]
    fallback_keys = [c for c in ["month", "电压等级"] if c in out.columns]
    if "month" not in fallback_keys:
        fallback_keys = ["month"]

    out["_peer_month"] = _shift_month_str(out["month"], months=-PEER_LAG_MONTHS)

    primary_keys_join = ["_peer_month"] + [c for c in primary_keys if c != "month"]
    fallback_keys_join = ["_peer_month"] + [c for c in fallback_keys if c != "month"]

    for source_col, alias in PEER_TARGETS.items():
        primary_stats = _build_peer_stats(out, primary_keys, source_col, alias)
        primary_stats = primary_stats.rename(columns={"month": "_peer_month"})
        primary_stats = primary_stats.rename(
            columns=lambda c: f"{c}_primary" if c.startswith(f"env_peer_{alias}_") else c
        )
        out = out.merge(primary_stats, on=primary_keys_join, how="left")

        fallback_stats = _build_peer_stats(out, fallback_keys, source_col, alias)
        fallback_stats = fallback_stats.rename(columns={"month": "_peer_month"})
        fallback_stats = fallback_stats.rename(
            columns=lambda c: f"{c}_fallback" if c.startswith(f"env_peer_{alias}_") else c
        )
        out = out.merge(fallback_stats, on=fallback_keys_join, how="left")

        primary_count = pd.to_numeric(
            out.get(f"env_peer_{alias}_count_primary", 0), errors="coerce"
        ).fillna(0)
        use_primary = primary_count >= PEER_MIN_GROUP_SIZE
        for metric in ["count", "median", "p25", "p75", "p90", "std"]:
            primary_col = f"env_peer_{alias}_{metric}_primary"
            fallback_col = f"env_peer_{alias}_{metric}_fallback"
            final_col = f"env_peer_{alias}_{metric}"
            out[final_col] = np.where(
                use_primary,
                pd.to_numeric(out.get(primary_col, 0), errors="coerce").fillna(0).values,
                pd.to_numeric(out.get(fallback_col, 0), errors="coerce").fillna(0).values,
            )

        drop_cols = [
            f"env_peer_{alias}_{m}_primary" for m in ["count", "median", "p25", "p75", "p90", "std"]
        ] + [f"env_peer_{alias}_{m}_fallback" for m in ["count", "median", "p25", "p75", "p90", "std"]]
        out = out.drop(columns=[c for c in drop_cols if c in out.columns])

    # Cold-start guard: rows whose previous-month group is missing have
    # count=0/median=0/std=0; computing z/ratio there yields huge values
    # (numerator divided by 1e-6 floor). Mask them to 0 explicitly.
    cold_fee = pd.to_numeric(out["env_peer_fee_count"], errors="coerce").fillna(0) == 0
    cold_energy = pd.to_numeric(out["env_peer_energy_count"], errors="coerce").fillna(0) == 0
    cold_unit = pd.to_numeric(out["env_peer_unit_price_dev_count"], errors="coerce").fillna(0) == 0

    out["env_peer_fee_vs_peer_ratio"] = np.where(
        cold_fee,
        0.0,
        out["总电费"] / (out["env_peer_fee_median"] + 1e-6),
    )
    out["env_peer_fee_vs_peer_z"] = np.where(
        cold_fee,
        0.0,
        (out["总电费"] - out["env_peer_fee_median"]) / (out["env_peer_fee_std"] + 1e-6),
    )
    out["env_peer_energy_vs_peer_z"] = np.where(
        cold_energy,
        0.0,
        (out["周期结算电量"] - out["env_peer_energy_median"])
        / (out["env_peer_energy_std"] + 1e-6),
    )
    out["env_peer_unit_price_dev_vs_peer_z"] = np.where(
        cold_unit,
        0.0,
        (out["unit_price_dev"] - out["env_peer_unit_price_dev_median"])
        / (out["env_peer_unit_price_dev_std"] + 1e-6),
    )

    # Alias names kept for stage-2 spec compatibility.
    out["env_peer_fee_vs_ratio"] = out["env_peer_fee_vs_peer_ratio"]
    out["env_peer_fee_vs_z"] = out["env_peer_fee_vs_peer_z"]
    out["env_peer_energy_vs_z"] = out["env_peer_energy_vs_peer_z"]
    out["env_peer_unit_price_dev_vs_z"] = out["env_peer_unit_price_dev_vs_peer_z"]

    for col in [
        "env_peer_fee_vs_peer_ratio",
        "env_peer_fee_vs_peer_z",
        "env_peer_energy_vs_peer_z",
        "env_peer_unit_price_dev_vs_peer_z",
        "env_peer_fee_vs_ratio",
        "env_peer_fee_vs_z",
        "env_peer_energy_vs_z",
        "env_peer_unit_price_dev_vs_z",
    ]:
        series = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], 0.0).fillna(0.0)
        # Clip to a sane band so a few outliers don't dominate gradients.
        out[col] = series.clip(lower=-50.0, upper=50.0)

    if "_peer_month" in out.columns:
        out = out.drop(columns=["_peer_month"])
    return out


def build_env_tariff_features(out: pd.DataFrame) -> pd.DataFrame:
    out = _add_month_numeric(out)
    grouped = out.groupby("客户ID", group_keys=False)

    curr_price_code = out["电价码"].astype(str) if "电价码" in out.columns else pd.Series([""] * len(out), index=out.index)
    curr_tariff_type = out["费率类型"].astype(str) if "费率类型" in out.columns else pd.Series([""] * len(out), index=out.index)
    prev_price_code = grouped["电价码"].shift(1) if "电价码" in out.columns else pd.Series(index=out.index, dtype=object)
    prev_tariff_type = grouped["费率类型"].shift(1) if "费率类型" in out.columns else pd.Series(index=out.index, dtype=object)
    prev_base = grouped["基准费率"].shift(1) if "基准费率" in out.columns else pd.Series(index=out.index, dtype=float)
    prev_extra = grouped["附加费率"].shift(1) if "附加费率" in out.columns else pd.Series(index=out.index, dtype=float)

    price_changed = (curr_price_code != prev_price_code.astype(str))
    tariff_type_changed = (curr_tariff_type != prev_tariff_type.astype(str))
    base_changed = (pd.to_numeric(out.get("基准费率", 0), errors="coerce").fillna(0) != prev_base.fillna(0))
    extra_changed = (pd.to_numeric(out.get("附加费率", 0), errors="coerce").fillna(0) != prev_extra.fillna(0))
    out["env_tariff_change_flag"] = (price_changed | tariff_type_changed | base_changed | extra_changed).astype(float)

    out["env_tariff_expected_unit_price"] = pd.to_numeric(out.get("expected_unit_price", 0), errors="coerce").fillna(0.0)
    out["env_tariff_base_rate"] = pd.to_numeric(out.get("基准费率", 0), errors="coerce").fillna(0.0)
    out["env_tariff_extra_rate"] = pd.to_numeric(out.get("附加费率", 0), errors="coerce").fillna(0.0)

    hist_unit_dev = grouped["unit_price_dev"].shift(1)
    hist_unit_dev_mean = (
        hist_unit_dev.groupby(out["客户ID"]).expanding(min_periods=1).mean().reset_index(level=0, drop=True)
    )
    out["env_unit_price_dev_vs_self_history"] = (
        out["unit_price_dev"] - hist_unit_dev_mean.fillna(0.0)
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return out


def _safe_binary(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower=0.0)


def build_sparse_event_features(out: pd.DataFrame) -> pd.DataFrame:
    out["event_fee_spike"] = _safe_binary(out.get("reason_fee_spike", 0)).clip(upper=1.0)
    out["event_meter_increase"] = _safe_binary(out.get("reason_rule_METER_INCREASE", 0)).clip(upper=1.0)
    out["event_reading_mismatch"] = _safe_binary(out.get("reason_rule_READING_MISMATCH", 0)).clip(upper=1.0)
    out["event_read_abn"] = _safe_binary(out.get("read_has_abn", 0)).clip(upper=1.0)

    # 与 env_peer_* 一致：p90 来自 D3-C 上月同群截面，再与当月 unit_price_dev 比较。
    unit_peer_p90 = out.get("env_peer_unit_price_dev_p90", 0)
    out["event_unit_price_dev_high"] = (
        pd.to_numeric(out.get("unit_price_dev", 0), errors="coerce").fillna(0.0)
        > pd.to_numeric(unit_peer_p90, errors="coerce").fillna(0.0)
    ).astype(float)

    # energy_mom_ratio 原表已是历史对比值，这里只做阈值触发重组，不新增派生量。
    out["event_energy_spike"] = (
        pd.to_numeric(out.get("energy_mom_ratio", 0), errors="coerce").fillna(0.0) > 0.25
    ).astype(float)

    event_cols = [
        "event_fee_spike",
        "event_meter_increase",
        "event_reading_mismatch",
        "event_read_abn",
        "event_unit_price_dev_high",
        "event_energy_spike",
    ]
    out["event_count"] = out[event_cols].sum(axis=1).astype(float)

    severity_raw = np.maximum(
        pd.to_numeric(out.get("abn_change_ratio_max", 0), errors="coerce").fillna(0.0).values,
        pd.to_numeric(out.get("fee_mom_ratio", 0), errors="coerce").fillna(0.0).values,
    )
    severity_raw = np.clip(severity_raw, a_min=0.0, a_max=None)
    severity_p95 = float(np.percentile(severity_raw, 95)) if len(severity_raw) else 1.0
    severity_scale = max(severity_p95, 1e-6)
    out["event_severity"] = np.clip(severity_raw / severity_scale, 0.0, 1.0)

    amplitude = pd.DataFrame(
        {
            "fee_spike": out["event_fee_spike"].values
            * np.abs(pd.to_numeric(out.get("fee_mom_ratio", 0), errors="coerce").fillna(0.0).values),
            "meter_increase": out["event_meter_increase"].values
            * np.abs(pd.to_numeric(out.get("abn_change_ratio_max", 0), errors="coerce").fillna(0.0).values),
            "reading_mismatch": out["event_reading_mismatch"].values
            * np.abs(pd.to_numeric(out.get("abn_change_ratio_max", 0), errors="coerce").fillna(0.0).values),
            "read_abn": out["event_read_abn"].values
            * np.abs(pd.to_numeric(out.get("abn_change_ratio_mean", 0), errors="coerce").fillna(0.0).values),
            "unit_price_dev_high": out["event_unit_price_dev_high"].values
            * np.abs(pd.to_numeric(out.get("unit_price_dev", 0), errors="coerce").fillna(0.0).values),
            "energy_spike": out["event_energy_spike"].values
            * np.abs(pd.to_numeric(out.get("energy_mom_ratio", 0), errors="coerce").fillna(0.0).values),
        },
        index=out.index,
    )
    dominant = amplitude.idxmax(axis=1)
    dominant_max = amplitude.max(axis=1)
    dominant = np.where(dominant_max > 0, dominant, "none")
    out["event_dominant_type"] = pd.Series(dominant, index=out.index).map(EVENT_DOMINANT_TYPE_MAP).fillna(0).astype(float)

    out["sparse_event_vec"] = out[event_cols].round(6).astype(str).agg("|".join, axis=1)
    return out


def build_environment_features(
    df: pd.DataFrame,
    feature_scope: str = "full",
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    out = _prepare_base_frame(df)
    out = build_env_self_features(out)
    if feature_scope == "stage1_self":
        groups = {
            "env_self": sorted([c for c in out.columns if c.startswith("env_self_")]),
            "env_peer": [],
            "env_tariff": [],
            "event": [],
            "other": ["history_len"],
        }
        return out, groups
    if feature_scope != "full":
        raise ValueError(f"Unknown feature_scope: {feature_scope} (use 'full' or 'stage1_self').")
    out = build_env_peer_features(out)
    out = build_env_tariff_features(out)
    out = build_sparse_event_features(out)
    groups = {
        "env_self": sorted([c for c in out.columns if c.startswith("env_self_")]),
        "env_peer": sorted([c for c in out.columns if c.startswith("env_peer_")]),
        "env_tariff": sorted([c for c in out.columns if c.startswith("env_tariff_") or c.startswith("env_season_")]),
        "event": sorted([c for c in out.columns if c.startswith("event_")] + ["sparse_event_vec"]),
        "other": ["history_len", "env_unit_price_dev_vs_self_history"],
    }
    return out, groups


def main() -> None:
    parser = argparse.ArgumentParser(description="Build environment feature set (self + peer + tariff).")
    parser.add_argument(
        "--input_csv",
        type=Path,
        default=Path("data/aligned/aligned_customer_month.csv"),
        help="Input aligned customer-month CSV.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("data/aligned/aligned_customer_month_env.csv"),
        help="Output CSV with env_self_* columns.",
    )
    parser.add_argument(
        "--output_meta",
        type=Path,
        default=Path("data/aligned/aligned_customer_month_env_metadata.json"),
        help="Output metadata JSON path.",
    )
    parser.add_argument(
        "--feature_scope",
        type=str,
        default="full",
        choices=["full", "stage1_self"],
        help="stage1_self: only env_self_* + history_len; full: peer + tariff + sparse events.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    out, groups = build_environment_features(df, feature_scope=args.feature_scope)
    if args.feature_scope == "stage1_self":
        new_cols = sorted(
            [c for c in out.columns if c.startswith("env_self_")]
            + ["history_len"]
        )
    else:
        new_cols = sorted(
            [
                c
                for c in out.columns
                if c.startswith("env_self_")
                or c.startswith("env_peer_")
                or c.startswith("env_tariff_")
                or c.startswith("event_")
            ]
            + ["history_len", "env_unit_price_dev_vs_self_history", "env_season_sin", "env_season_cos", "sparse_event_vec"]
        )
    meta = {
        "input_csv": str(args.input_csv),
        "output_csv": str(args.output_csv),
        "feature_scope": args.feature_scope,
        "row_count": int(len(out)),
        "customer_count": int(out["客户ID"].nunique()),
        "month_count": int(out["month"].nunique()),
        "new_feature_count": int(len(new_cols)),
        "new_features": new_cols,
        "feature_groups": groups,
        "peer_group_min_size": PEER_MIN_GROUP_SIZE,
        "peer_primary_keys": [c for c in ["month", "客户类型", "电压等级", "行业编码"] if c in out.columns],
        "peer_fallback_keys": [c for c in ["month", "电压等级"] if c in out.columns],
        "peer_lag_months": PEER_LAG_MONTHS,
        "peer_cold_start_fallback": "zero",
    }

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    args.output_meta.parent.mkdir(parents=True, exist_ok=True)
    args.output_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] wrote env dataset: {args.output_csv} ({len(out)} rows)")
    print(f"[OK] wrote env metadata: {args.output_meta}")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
