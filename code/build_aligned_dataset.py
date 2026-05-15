#!/usr/bin/env python3
"""Build customer-month aligned dataset from 7 business tables.

Output:
1) aligned_customer_month.csv
2) aligned_metadata.json

The script aligns and aggregates data at customer-month granularity,
builds weak labels and reason multi-labels for downstream teacher/student training.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def read_csv_auto(path: Path) -> pd.DataFrame:
    encodings = ("utf-8-sig", "utf-8", "gbk", "gb18030")
    last_error: Optional[Exception] = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as exc:  # pragma: no cover
            last_error = exc
    raise RuntimeError(f"Cannot read {path}: {last_error}")


def to_month(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.dt.strftime("%Y-%m")


def parse_abnormal_day_to_month(row: pd.Series) -> Optional[str]:
    record_id = str(row.get("异常记录ID", ""))
    matched = re.search(r"(20\d{2}\d{2}\d{2})", record_id)
    if matched:
        dt = pd.to_datetime(matched.group(1), format="%Y%m%d", errors="coerce")
        if pd.notna(dt):
            return dt.strftime("%Y-%m")
    check_time = row.get("检查时间")
    dt = pd.to_datetime(check_time, errors="coerce")
    if pd.notna(dt):
        return dt.strftime("%Y-%m")
    return None


def safe_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_meter_month_features(meter_reading: pd.DataFrame) -> pd.DataFrame:
    meter_reading = meter_reading.copy()
    meter_reading["month"] = to_month(meter_reading["数据时间"])
    meter_reading = safe_numeric(
        meter_reading,
        cols=["抄见电量", "结算电量", "峰电量", "谷电量", "平电量", "是否异常"],
    )

    agg_base = (
        meter_reading.groupby(["客户ID", "month"], as_index=False)
        .agg(
            read_count=("电表ID", "count"),
            meter_count_in_month=("电表ID", "nunique"),
            read_energy_sum=("抄见电量", "sum"),
            settle_energy_sum=("结算电量", "sum"),
            peak_energy_sum=("峰电量", "sum"),
            valley_energy_sum=("谷电量", "sum"),
            flat_energy_sum=("平电量", "sum"),
            read_has_abn=("是否异常", "max"),
        )
        .fillna(0.0)
    )

    if "抄表方式" in meter_reading.columns:
        method_ratio = (
            meter_reading.assign(cnt=1)
            .pivot_table(
                index=["客户ID", "month"],
                columns="抄表方式",
                values="cnt",
                aggfunc="sum",
                fill_value=0,
            )
            .astype(float)
        )
        method_ratio = method_ratio.div(method_ratio.sum(axis=1).replace(0, 1), axis=0)
        method_ratio.columns = [f"method_ratio_{str(c)}" for c in method_ratio.columns]
        method_ratio = method_ratio.reset_index()
        agg_base = agg_base.merge(method_ratio, on=["客户ID", "month"], how="left")

    return agg_base.fillna(0.0)


def build_abnormal_month_features(abnormal: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    abnormal = abnormal.copy()
    abnormal["month"] = abnormal.apply(parse_abnormal_day_to_month, axis=1)
    abnormal = abnormal.dropna(subset=["客户ID", "month"])

    abnormal_base = (
        abnormal.groupby(["客户ID", "month"], as_index=False)
        .agg(
            abnormal_event_count=("异常记录ID", "count"),
            abnormal_meter_count=("电表ID", "nunique"),
        )
        .fillna(0)
    )

    reason_cols: List[str] = []
    if "触发规则ID" in abnormal.columns:
        by_rule = (
            abnormal.assign(v=1)
            .pivot_table(
                index=["客户ID", "month"],
                columns="触发规则ID",
                values="v",
                aggfunc="sum",
                fill_value=0,
            )
            .reset_index()
        )
        rename_map = {}
        for col in by_rule.columns:
            if col in ("客户ID", "month"):
                continue
            new_col = f"reason_rule_{col}"
            rename_map[col] = new_col
            reason_cols.append(new_col)
        by_rule = by_rule.rename(columns=rename_map)
        abnormal_base = abnormal_base.merge(by_rule, on=["客户ID", "month"], how="left")

    # Parse "变化率为 x.xx" as a soft severity signal.
    if "异常描述" in abnormal.columns:
        ratio_series = abnormal["异常描述"].astype(str).str.extract(r"变化率为\s*([0-9.]+)")[0]
        ratio_series = pd.to_numeric(ratio_series, errors="coerce")
        abnormal["abn_change_ratio"] = ratio_series
        ratio_agg = (
            abnormal.groupby(["客户ID", "month"], as_index=False)
            .agg(
                abn_change_ratio_mean=("abn_change_ratio", "mean"),
                abn_change_ratio_max=("abn_change_ratio", "max"),
            )
            .fillna(0.0)
        )
        abnormal_base = abnormal_base.merge(ratio_agg, on=["客户ID", "month"], how="left")

    return abnormal_base.fillna(0.0), sorted(reason_cols)


def build_meter_static_features(meter_archive: pd.DataFrame) -> pd.DataFrame:
    meter_archive = meter_archive.copy()
    meter_archive = safe_numeric(meter_archive, cols=["计量精度", "综合倍率", "运行状态"])
    if "安装日期" in meter_archive.columns:
        install_date = pd.to_datetime(meter_archive["安装日期"], errors="coerce")
        meter_archive["meter_age_days"] = (
            pd.Timestamp("2026-01-01") - install_date
        ).dt.days.clip(lower=0)
    else:
        meter_archive["meter_age_days"] = 0

    static = (
        meter_archive.groupby("客户ID", as_index=False)
        .agg(
            meter_count_total=("电表ID", "nunique"),
            meter_precision_mean=("计量精度", "mean"),
            meter_ratio_mean=("综合倍率", "mean"),
            meter_running_ratio=("运行状态", "mean"),
            meter_age_days_mean=("meter_age_days", "mean"),
        )
        .fillna(0.0)
    )
    return static


def build_price_table(price: pd.DataFrame) -> pd.DataFrame:
    keep_cols = ["电价码", "基准费率", "附加费率", "费率类型", "基本电费", "价格分类"]
    existing = [c for c in keep_cols if c in price.columns]
    out = price[existing].copy()
    out = safe_numeric(out, cols=["基准费率", "附加费率", "基本电费"])
    if "生效日期" in price.columns:
        out["生效日期"] = pd.to_datetime(price["生效日期"], errors="coerce")
        out = out.sort_values("生效日期")
    out = out.drop_duplicates(subset=["电价码"], keep="last")
    return out


def build_dataset(input_dir: Path, output_path: Path, metadata_path: Path) -> None:
    customer = read_csv_auto(input_dir / "客户档案表.csv")
    meter_archive = read_csv_auto(input_dir / "电表档案表.csv")
    meter_reading = read_csv_auto(input_dir / "抄表数据表.csv")
    usage = read_csv_auto(input_dir / "客户用电量表.csv")
    bill = read_csv_auto(input_dir / "客户电费结果表.csv")
    abnormal = read_csv_auto(input_dir / "异常结果表.csv")
    price = read_csv_auto(input_dir / "电价表.csv")

    # Base customer-month table from bill + usage.
    usage = usage.copy()
    bill = bill.copy()
    usage["month"] = usage["统计周期"].astype(str)
    bill["month"] = bill["统计周期"].astype(str)

    num_usage_cols = [
        "周期总电量",
        "周期峰电量",
        "周期谷电量",
        "周期平电量",
        "周期结算电量",
        "上月结转电量",
        "上月结转金额",
    ]
    num_bill_cols = ["基础电费", "电量电费", "附加费", "总电费", "是否异常"]
    usage = safe_numeric(usage, num_usage_cols)
    bill = safe_numeric(bill, num_bill_cols)

    aligned = bill.merge(
        usage[
            [
                "客户ID",
                "month",
                "周期总电量",
                "周期峰电量",
                "周期谷电量",
                "周期平电量",
                "周期结算电量",
                "上月结转电量",
                "上月结转金额",
            ]
        ],
        on=["客户ID", "month"],
        how="left",
    )

    meter_month = build_meter_month_features(meter_reading)
    aligned = aligned.merge(meter_month, on=["客户ID", "month"], how="left")

    abnormal_month, reason_cols = build_abnormal_month_features(abnormal)
    aligned = aligned.merge(abnormal_month, on=["客户ID", "month"], how="left")

    meter_static = build_meter_static_features(meter_archive)
    aligned = aligned.merge(meter_static, on="客户ID", how="left")

    customer_keep = [
        "客户ID",
        "客户类型",
        "客户子类型",
        "用户分类",
        "用能类别",
        "价格分类",
        "电压等级",
        "地区编码",
        "行业编码",
        "合同容量",
        "运行容量",
        "是否异常",
        "异常类型",
    ]
    customer_keep = [c for c in customer_keep if c in customer.columns]
    aligned = aligned.merge(customer[customer_keep], on="客户ID", how="left", suffixes=("", "_cust"))

    price_info = build_price_table(price)
    aligned = aligned.merge(price_info, on="电价码", how="left", suffixes=("", "_price"))

    aligned = aligned.fillna(0.0)

    # Sort for temporal features.
    aligned["month_dt"] = pd.to_datetime(aligned["month"] + "-01", errors="coerce")
    aligned = aligned.sort_values(["客户ID", "month_dt"]).reset_index(drop=True)

    aligned["fee_log1p"] = np.log1p(pd.to_numeric(aligned["总电费"], errors="coerce").clip(lower=0))
    aligned["energy_log1p"] = np.log1p(pd.to_numeric(aligned["周期结算电量"], errors="coerce").clip(lower=0))

    aligned["fee_prev"] = aligned.groupby("客户ID")["总电费"].shift(1)
    aligned["energy_prev"] = aligned.groupby("客户ID")["周期结算电量"].shift(1)
    aligned["fee_mom_ratio"] = aligned["总电费"] / (aligned["fee_prev"] + 1e-6)
    aligned["energy_mom_ratio"] = aligned["周期结算电量"] / (aligned["energy_prev"] + 1e-6)
    aligned["fee_mom_diff"] = aligned["总电费"] - aligned["fee_prev"].fillna(0)
    aligned["energy_mom_diff"] = aligned["周期结算电量"] - aligned["energy_prev"].fillna(0)

    aligned["unit_price_est"] = aligned["电量电费"] / (aligned["周期结算电量"] + 1e-6)
    aligned["expected_unit_price"] = (
        pd.to_numeric(aligned.get("基准费率", 0), errors="coerce").fillna(0)
        + pd.to_numeric(aligned.get("附加费率", 0), errors="coerce").fillna(0)
    )
    aligned["unit_price_dev"] = aligned["unit_price_est"] - aligned["expected_unit_price"]

    # Weak label for training.
    abn_count = pd.to_numeric(aligned.get("abnormal_event_count", 0), errors="coerce").fillna(0)
    read_abn = pd.to_numeric(aligned.get("read_has_abn", 0), errors="coerce").fillna(0)
    fee_spike = (aligned["fee_mom_ratio"] >= 2.0).astype(int)
    aligned["weak_label"] = ((abn_count > 0) | (read_abn > 0) | (fee_spike > 0)).astype(int)

    # Reason labels for multi-label distillation.
    aligned["reason_fee_spike"] = fee_spike
    for reason_col in reason_cols:
        aligned[f"label_{reason_col}"] = (pd.to_numeric(aligned[reason_col], errors="coerce").fillna(0) > 0).astype(int)
    aligned["label_reason_fee_spike"] = aligned["reason_fee_spike"]

    reason_label_cols = sorted([c for c in aligned.columns if c.startswith("label_reason_") or c.startswith("label_reason_rule_")])

    # Keep all rows with valid customer-month.
    aligned = aligned[aligned["month_dt"].notna()].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(output_path, index=False)

    metadata = {
        "row_count": int(len(aligned)),
        "customer_count": int(aligned["客户ID"].nunique()),
        "month_count": int(aligned["month"].nunique()),
        "weak_label_positive_ratio": float(aligned["weak_label"].mean()),
        "reason_label_columns": reason_label_cols,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] wrote dataset: {output_path} ({len(aligned)} rows)")
    print(f"[OK] wrote metadata: {metadata_path}")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build aligned customer-month dataset for fee anomaly distillation.")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "output",
        help="Directory containing the 7 CSV tables.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "dataset" / "aligned_customer_month.csv",
        help="Output aligned dataset CSV path.",
    )
    parser.add_argument(
        "--output_meta",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "dataset" / "aligned_metadata.json",
        help="Output metadata JSON path.",
    )
    args = parser.parse_args()

    build_dataset(args.input_dir, args.output_csv, args.output_meta)


if __name__ == "__main__":
    main()
