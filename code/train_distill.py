#!/usr/bin/env python3
"""Train teacher and student models on aligned customer-month dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from calibration import calibration_report
from models import ModelBatch, StudentModel, TeacherModel, distill_loss

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None


LEAKAGE_EXACT_COLS = {
    "abnormal_event_count",
    "read_has_abn",
    "fee_mom_ratio",
    "reason_fee_spike",
    "是否异常",
    "是否异常_cust",
    # D1: 阶段 3 设计为"弱化稀疏事件给学生看"的两个汇总列，但当前实现把 6 个事件
    # 与 abn_change_ratio_max / fee_mom_ratio 几乎无损地汇总过来，是当前学生虚高的
    # 主要捷径之一。屏蔽掉，让学生只能从 env_peer_* / env_tariff_* / env_self_* 里
    # 学到环境信号。教师视图通过 reason_cols 显式列表仍然吃 7 个 event_* 原始列。
    "event_count",
    "event_severity",
    # D1: 直接驱动 reason_rule_* 与 weak_label 的当月底层连续量。这些是合成数据里
    # "规则 → 标签"链路的真正源头，留给学生等于让它绕开 reason_rule_ 屏蔽。环境
    # 派生列（env_self_* / env_peer_* / env_unit_price_dev_vs_self_history 等）
    # 均为 shift(1) 自身历史或 D3-C 上月同群基线（非当月同群），依然保留。
    "unit_price_dev",
    "abn_change_ratio_max",
    "abn_change_ratio_mean",
    "abnormal_meter_count",
    "fee_mom_diff",
    "energy_mom_ratio",
    "energy_mom_diff",
}
LEAKAGE_PREFIXES = (
    "reason_rule_",
    "event_meter_",
    "event_reading_",
    "event_fee_spike",
    "event_read_abn",
    "event_unit_price_dev_high",
    "event_energy_spike",
    "event_dominant_type",
)
LLM_EMB_PREFIXES = ("llm_prefix_emb_", "llm_risk_emb_")
LLM_REASON_PROB_PREFIX = "llm_reason_prob_"
LLM_ANOMALY_PROB_COL = "llm_anomaly_prob"
# Phase-6: prefix used as the alignment target for the student latent. We
# always pin to the first prefix family that exists, in this order, so the
# choice mirrors `get_llm_embedding_cols` (the teacher 4th view) and there is
# only one source of truth across the run.
LLM_PREFIX_TARGET_PREFIXES = ("llm_prefix_emb_", "llm_risk_emb_")
# Phase-6: column-name prefixes treated as the dedicated `env_x` view fed to
# the student's EnvironmentEncoder. These columns are removed from `num_cols`
# at training time so the main numeric trunk no longer sees them; the teacher
# view is unchanged because the teacher consumes (num + env) jointly anyway.
ENV_VIEW_PREFIXES = ("env_self_", "env_peer_", "env_tariff_", "env_season_")
# These two columns are env-derived but currently named without an env_*
# prefix; keep them in env view too, to avoid leaking back into num.
ENV_VIEW_EXACT = {"env_unit_price_dev_vs_self_history"}


def get_llm_embedding_cols(df: pd.DataFrame) -> List[str]:
    for prefix in LLM_EMB_PREFIXES:
        cols = sorted([c for c in df.columns if c.startswith(prefix)])
        if cols:
            return cols
    return []


def get_llm_prefix_target_cols(df: pd.DataFrame) -> List[str]:
    """Phase-6: choose the prefix family used as the student-alignment target.

    Pinning a single deterministic prefix avoids the case where teacher uses
    `llm_prefix_emb_*` while student is silently asked to align against
    `llm_risk_emb_*` (or vice versa) just because of column ordering changes.
    """
    for prefix in LLM_PREFIX_TARGET_PREFIXES:
        cols = sorted([c for c in df.columns if c.startswith(prefix)])
        if cols:
            return cols
    return []


class RunMonitor:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.metrics_path = output_dir / "metrics.jsonl"
        self.metrics_path.write_text("", encoding="utf-8")
        self.history: Dict[str, List[Dict[str, float]]] = {"teacher": [], "student": []}
        self.writer = SummaryWriter(log_dir=str(output_dir / "tb_logs")) if SummaryWriter is not None else None
        if self.writer is None:
            print("[INFO] tensorboard not available, skip tb_logs.")

    def log_epoch(
        self,
        stage: str,
        epoch: int,
        epochs: int,
        train_losses: Dict[str, float],
        val_metric: Dict[str, float],
        lr: float,
    ) -> None:
        payload = {
            "stage": stage,
            "epoch": epoch,
            "epochs": epochs,
            "lr": float(lr),
            "train_losses": {k: float(v) for k, v in train_losses.items()},
            "val_metric": {k: float(v) for k, v in val_metric.items()},
        }
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.history[stage].append(payload)
        if self.writer is not None:
            for k, v in payload["train_losses"].items():
                self.writer.add_scalar(f"{stage}/train_{k}", v, epoch)
            for k, v in payload["val_metric"].items():
                self.writer.add_scalar(f"{stage}/val_{k}", v, epoch)
            self.writer.add_scalar(f"{stage}/lr", payload["lr"], epoch)
        print(
            f"[{stage}] epoch {epoch}/{epochs} "
            f"train_total={payload['train_losses'].get('total', 0.0):.4f} "
            f"val_f1={payload['val_metric']['f1']:.4f} "
            f"val_precision={payload['val_metric']['precision']:.4f} "
            f"val_recall={payload['val_metric']['recall']:.4f}"
        )

    def save_curves(self) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:  # pragma: no cover
            print("[INFO] matplotlib not available, skip curve png export.")
            return
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for row, stage in enumerate(["teacher", "student"]):
            records = self.history[stage]
            if not records:
                continue
            xs = [r["epoch"] for r in records]
            total_loss = [r["train_losses"].get("total", 0.0) for r in records]
            val_f1 = [r["val_metric"]["f1"] for r in records]
            axes[row][0].plot(xs, total_loss, marker="o")
            axes[row][0].set_title(f"{stage} train total loss")
            axes[row][0].set_xlabel("epoch")
            axes[row][0].set_ylabel("loss")
            axes[row][1].plot(xs, val_f1, marker="o")
            axes[row][1].set_title(f"{stage} val f1")
            axes[row][1].set_xlabel("epoch")
            axes[row][1].set_ylabel("f1")
            axes[row][1].set_ylim(0.0, 1.0)
        fig.tight_layout()
        out_path = self.output_dir / "training_curves.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[OK] curves saved: {out_path}")

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
        self.save_curves()


def maybe_tqdm(loader: DataLoader, desc: str):
    if tqdm is None:
        return loader
    return tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode_categorical(series: pd.Series) -> Tuple[np.ndarray, Dict[str, int]]:
    values = series.fillna("UNK").astype(str)
    uniq = sorted(values.unique().tolist())
    mapping = {v: i for i, v in enumerate(uniq)}
    encoded = values.map(mapping).fillna(0).astype(int).values
    return encoded, mapping


def zscore_fit_transform(train: np.ndarray, val: np.ndarray, test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(train, axis=0)
    std = np.nanstd(train, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return (train - mean) / std, (val - mean) / std, (test - mean) / std


@dataclass
class TensorBundle:
    num_x: torch.Tensor
    cat_x: torch.Tensor
    reason_x: torch.Tensor
    llm_x: torch.Tensor
    y_weak: torch.Tensor
    y_reason: torch.Tensor
    y_reason_soft: torch.Tensor
    y_reason_soft_mask: torch.Tensor
    y_weak_soft: torch.Tensor
    y_weak_soft_mask: torch.Tensor
    # Phase-6 additions ---------------------------------------------------
    # student-only environment view; teacher still sees env via its num view.
    env_x: torch.Tensor
    # prefix target + presence mask used by `prefix_alignment_loss`.
    llm_prefix_target: torch.Tensor
    llm_prefix_mask: torch.Tensor


class FeeDataset(Dataset):
    def __init__(self, bundle: TensorBundle) -> None:
        self.bundle = bundle

    def __len__(self) -> int:
        return self.bundle.num_x.size(0)

    def __getitem__(self, idx: int):
        return (
            self.bundle.num_x[idx],
            self.bundle.cat_x[idx],
            self.bundle.reason_x[idx],
            self.bundle.llm_x[idx],
            self.bundle.y_weak[idx],
            self.bundle.y_reason[idx],
            self.bundle.y_reason_soft[idx],
            self.bundle.y_reason_soft_mask[idx],
            self.bundle.y_weak_soft[idx],
            self.bundle.y_weak_soft_mask[idx],
            self.bundle.env_x[idx],
            self.bundle.llm_prefix_target[idx],
            self.bundle.llm_prefix_mask[idx],
        )


def _is_env_view_col(col: str) -> bool:
    return col in ENV_VIEW_EXACT or any(col.startswith(p) for p in ENV_VIEW_PREFIXES)


def infer_columns(
    df: pd.DataFrame,
    *,
    use_env_view: bool = False,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Infer (num_cols, cat_cols, reason_cols, env_cols).

    When `use_env_view=True` (Phase-6), columns whose name starts with any of
    `ENV_VIEW_PREFIXES` (or matches `ENV_VIEW_EXACT`) are *removed* from
    `num_cols` and returned as a separate `env_cols` list, so the student can
    route them through its EnvironmentEncoder. With `use_env_view=False` the
    returned `env_cols` is empty and `num_cols` reproduces the pre-phase-6
    column set byte-for-byte (no behaviour change for the teacher path).
    """
    cat_cols = [
        c
        for c in ["客户类型", "客户子类型", "用户分类", "用能类别", "价格分类", "电压等级", "费率类型"]
        if c in df.columns
    ]
    # reason_x is teacher-only view: feed structured sparse events there.
    reason_cols = [
        c
        for c in [
            "event_fee_spike",
            "event_meter_increase",
            "event_reading_mismatch",
            "event_read_abn",
            "event_unit_price_dev_high",
            "event_energy_spike",
            "event_dominant_type",
        ]
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    reason_label_cols = [c for c in df.columns if c.startswith("label_reason_") or c.startswith("label_reason_rule_")]

    blocklist = {
        "客户ID",
        "month",
        "month_dt",
        "统计周期",
        "电费记录ID",
        "模拟批次ID",
        "weak_label",
    }
    blocklist.update(LEAKAGE_EXACT_COLS)
    blocklist.update(cat_cols)
    blocklist.update(reason_label_cols)
    all_num_cols = [
        c
        for c in df.columns
        if c not in blocklist
        and not any(c.startswith(prefix) for prefix in LEAKAGE_PREFIXES)
        and not any(c.startswith(prefix) for prefix in LLM_EMB_PREFIXES)
        and not c.startswith(LLM_REASON_PROB_PREFIX)
        and c != LLM_ANOMALY_PROB_COL
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    if use_env_view:
        env_cols = [c for c in all_num_cols if _is_env_view_col(c)]
        num_cols = [c for c in all_num_cols if not _is_env_view_col(c)]
    else:
        env_cols = []
        num_cols = all_num_cols
    return num_cols, cat_cols, reason_cols, env_cols


def split_by_month(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    months = sorted(df["month"].dropna().astype(str).unique().tolist())
    if len(months) < 3:
        raise ValueError("Need at least 3 months for train/val/test split.")
    train_months = set(months[:-3])
    val_months = {months[-3], months[-2]}
    test_months = {months[-1]}
    train_df = df[df["month"].isin(train_months)].copy()
    val_df = df[df["month"].isin(val_months)].copy()
    test_df = df[df["month"].isin(test_months)].copy()
    return train_df, val_df, test_df


def _temporal_split_inside(remain_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split remaining rows into train/val along the time axis."""
    months = sorted(remain_df["month"].dropna().astype(str).unique().tolist())
    if len(months) < 2:
        n = len(remain_df)
        cut = max(1, int(n * 0.85))
        return remain_df.iloc[:cut].copy(), remain_df.iloc[cut:].copy()
    val_months = {months[-1]}
    train_months = set(months[:-1])
    train_df = remain_df[remain_df["month"].isin(train_months)].copy()
    val_df = remain_df[remain_df["month"].isin(val_months)].copy()
    return train_df, val_df


def _select_holdout_value(df: pd.DataFrame, key: str, target_test_ratio: float = 0.2) -> str:
    """Pick a holdout value whose share is closest to target_test_ratio (default 20%)."""
    series = df[key].astype(str)
    counts = series.value_counts(dropna=False)
    total = float(counts.sum())
    ratios = (counts / total - target_test_ratio).abs().sort_values()
    return str(ratios.index[0])


def split_cold_start(
    df: pd.DataFrame,
    seed: int,
    train_n: int = 7000,
    val_n: int = 1000,
    test_n: int = 2000,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    customers = df["客户ID"].astype(str).unique().tolist()
    rng.shuffle(customers)
    if len(customers) < train_n + val_n + test_n:
        # Fallback to fractional split if customer count is smaller.
        a = int(len(customers) * 0.7)
        b = int(len(customers) * 0.8)
        train_ids = set(customers[:a])
        val_ids = set(customers[a:b])
        test_ids = set(customers[b:])
    else:
        train_ids = set(customers[:train_n])
        val_ids = set(customers[train_n:train_n + val_n])
        test_ids = set(customers[train_n + val_n:train_n + val_n + test_n])
    cust = df["客户ID"].astype(str)
    train_df = df[cust.isin(train_ids)].copy()
    val_df = df[cust.isin(val_ids)].copy()
    test_df = df[cust.isin(test_ids)].copy()
    return train_df, val_df, test_df


def split_cross_domain(
    df: pd.DataFrame,
    key: str,
    holdout_value: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    if key not in df.columns:
        raise ValueError(f"Cross-domain split needs column {key} in dataframe.")
    if holdout_value is None:
        holdout_value = _select_holdout_value(df, key)
    series = df[key].astype(str)
    test_df = df[series == holdout_value].copy()
    remain_df = df[series != holdout_value].copy()
    train_df, val_df = _temporal_split_inside(remain_df)
    return train_df, val_df, test_df, holdout_value


def split_tariff_shift(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "env_tariff_change_flag" not in df.columns:
        raise ValueError("Tariff-shift split requires env_tariff_change_flag column.")
    df_sorted = df.sort_values(["客户ID", "month"]).reset_index(drop=True)
    flag_sorted = pd.to_numeric(df_sorted["env_tariff_change_flag"], errors="coerce").fillna(0)
    cumax = flag_sorted.groupby(df_sorted["客户ID"]).cummax().astype(int)
    # The very month a tariff change first occurs is left in train/val so the
    # model can observe the boundary. Strictly *after* the first change is test.
    after_first_change = cumax.groupby(df_sorted["客户ID"]).shift(1).fillna(0).astype(int)
    test_mask = after_first_change > 0
    test_df = df_sorted[test_mask].copy()
    remain_df = df_sorted[~test_mask].copy()
    if len(test_df) == 0 or len(remain_df) == 0:
        return split_by_month(df_sorted)
    train_df, val_df = _temporal_split_inside(remain_df)
    return train_df, val_df, test_df


def split_dataset(
    df: pd.DataFrame,
    mode: str,
    seed: int,
    cross_domain_industry_value: str | None = None,
    cross_domain_voltage_value: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    info: Dict[str, object] = {"mode": mode}
    if mode == "temporal":
        train_df, val_df, test_df = split_by_month(df)
    elif mode == "cold_start":
        train_df, val_df, test_df = split_cold_start(df, seed=seed)
        info["train_customers"] = int(train_df["客户ID"].nunique())
        info["val_customers"] = int(val_df["客户ID"].nunique())
        info["test_customers"] = int(test_df["客户ID"].nunique())
    elif mode == "cross_domain_industry":
        train_df, val_df, test_df, holdout = split_cross_domain(df, "行业编码", cross_domain_industry_value)
        info["holdout_industry"] = holdout
    elif mode == "cross_domain_voltage":
        train_df, val_df, test_df, holdout = split_cross_domain(df, "电压等级", cross_domain_voltage_value)
        info["holdout_voltage"] = holdout
    elif mode == "tariff_shift":
        train_df, val_df, test_df = split_tariff_shift(df)
    else:
        raise ValueError(f"Unknown split mode: {mode}")

    info["train_size"] = int(len(train_df))
    info["val_size"] = int(len(val_df))
    info["test_size"] = int(len(test_df))
    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError(f"split_mode={mode} produced empty partition: {info}")
    return train_df, val_df, test_df, info


def _llm_prob_col_for_reason_label(label_col: str) -> str:
    return f"{LLM_REASON_PROB_PREFIX}{label_col.replace('label_', '')}"


def build_tensors(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_cols: Sequence[str],
    cat_cols: Sequence[str],
    reason_cols: Sequence[str],
    reason_label_cols: Sequence[str],
    llm_emb_cols: Sequence[str],
    llm_reason_prob_cols: Sequence[str | None],
    llm_anomaly_prob_col: str | None,
    env_cols: Sequence[str] = (),
    llm_prefix_target_cols: Sequence[str] = (),
) -> Tuple[TensorBundle, TensorBundle, TensorBundle, Dict[str, Dict[str, int]]]:
    cat_maps: Dict[str, Dict[str, int]] = {}
    train_cat_list, val_cat_list, test_cat_list = [], [], []
    for col in cat_cols:
        train_enc, mapping = encode_categorical(train_df[col])
        cat_maps[col] = mapping
        val_enc = val_df[col].fillna("UNK").astype(str).map(mapping).fillna(0).astype(int).values
        test_enc = test_df[col].fillna("UNK").astype(str).map(mapping).fillna(0).astype(int).values
        train_cat_list.append(train_enc)
        val_cat_list.append(val_enc)
        test_cat_list.append(test_enc)

    train_num = train_df[num_cols].fillna(0).astype(float).values
    val_num = val_df[num_cols].fillna(0).astype(float).values
    test_num = test_df[num_cols].fillna(0).astype(float).values
    train_num, val_num, test_num = zscore_fit_transform(train_num, val_num, test_num)

    if env_cols:
        train_env_raw = train_df[list(env_cols)].fillna(0).astype(float).values
        val_env_raw = val_df[list(env_cols)].fillna(0).astype(float).values
        test_env_raw = test_df[list(env_cols)].fillna(0).astype(float).values
        train_env, val_env, test_env = zscore_fit_transform(train_env_raw, val_env_raw, test_env_raw)
    else:
        train_env = np.zeros((len(train_df), 0), dtype=np.float32)
        val_env = np.zeros((len(val_df), 0), dtype=np.float32)
        test_env = np.zeros((len(test_df), 0), dtype=np.float32)

    def to_bundle(
        part_df: pd.DataFrame,
        num: np.ndarray,
        cat_list: List[np.ndarray],
        env: np.ndarray,
    ) -> TensorBundle:
        cat = np.stack(cat_list, axis=1) if cat_list else np.zeros((len(part_df), 0), dtype=np.int64)
        reason_x = (
            part_df[list(reason_cols)].fillna(0).astype(float).values
            if reason_cols
            else np.zeros((len(part_df), 1), dtype=np.float32)
        )
        y_reason = (
            part_df[list(reason_label_cols)].fillna(0).astype(float).values
            if reason_label_cols
            else np.zeros((len(part_df), 1), dtype=np.float32)
        )
        llm_x = (
            part_df[list(llm_emb_cols)].fillna(0).astype(float).values
            if llm_emb_cols
            else np.zeros((len(part_df), 0), dtype=np.float32)
        )
        y_reason_soft = np.zeros_like(y_reason, dtype=np.float32)
        y_reason_soft_mask = np.zeros_like(y_reason, dtype=np.float32)
        for idx, col in enumerate(llm_reason_prob_cols):
            if col is None or col not in part_df.columns:
                continue
            col_raw = pd.to_numeric(part_df[col], errors="coerce")
            valid_mask = col_raw.notna().astype(np.float32).values
            col_values = col_raw.fillna(0.0).astype(float).values
            y_reason_soft[:, idx] = col_values
            y_reason_soft_mask[:, idx] = valid_mask
        if llm_anomaly_prob_col is not None and llm_anomaly_prob_col in part_df.columns:
            weak_soft_raw = pd.to_numeric(part_df[llm_anomaly_prob_col], errors="coerce")
            weak_soft_mask = weak_soft_raw.notna().astype(np.float32).values
            weak_soft = weak_soft_raw.fillna(0.0).astype(float).values
        else:
            weak_soft_mask = np.zeros((len(part_df),), dtype=np.float32)
            weak_soft = np.zeros((len(part_df),), dtype=np.float32)
        if llm_prefix_target_cols:
            prefix_raw = part_df[list(llm_prefix_target_cols)]
            prefix_present_mask = prefix_raw.notna().all(axis=1).astype(np.float32).values
            prefix_target = prefix_raw.fillna(0.0).astype(float).values
        else:
            prefix_present_mask = np.zeros((len(part_df),), dtype=np.float32)
            prefix_target = np.zeros((len(part_df), 0), dtype=np.float32)
        y_weak = part_df["weak_label"].fillna(0).astype(float).values
        return TensorBundle(
            num_x=torch.tensor(num, dtype=torch.float32),
            cat_x=torch.tensor(cat, dtype=torch.long),
            reason_x=torch.tensor(reason_x, dtype=torch.float32),
            llm_x=torch.tensor(llm_x, dtype=torch.float32),
            y_weak=torch.tensor(y_weak, dtype=torch.float32),
            y_reason=torch.tensor(y_reason, dtype=torch.float32),
            y_reason_soft=torch.tensor(y_reason_soft, dtype=torch.float32),
            y_reason_soft_mask=torch.tensor(y_reason_soft_mask, dtype=torch.float32),
            y_weak_soft=torch.tensor(weak_soft, dtype=torch.float32),
            y_weak_soft_mask=torch.tensor(weak_soft_mask, dtype=torch.float32),
            env_x=torch.tensor(env, dtype=torch.float32),
            llm_prefix_target=torch.tensor(prefix_target, dtype=torch.float32),
            llm_prefix_mask=torch.tensor(prefix_present_mask, dtype=torch.float32),
        )

    return (
        to_bundle(train_df, train_num, train_cat_list, train_env),
        to_bundle(val_df, val_num, val_cat_list, val_env),
        to_bundle(test_df, test_num, test_cat_list, test_env),
        cat_maps,
    )


def binary_metrics(logit: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    prob = torch.sigmoid(logit)
    return binary_metrics_from_prob(prob, y, threshold=0.5)


def compute_auprc(prob: torch.Tensor, y: torch.Tensor) -> float:
    """Average-precision style AUPRC; returns 0.0 when no positive samples."""
    p = prob.detach().cpu().numpy().reshape(-1)
    t = y.detach().cpu().numpy().reshape(-1)
    if t.sum() <= 0 or len(t) == 0:
        return 0.0
    order = np.argsort(-p)
    t_sorted = t[order]
    tp_cum = np.cumsum(t_sorted)
    fp_cum = np.cumsum(1.0 - t_sorted)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    recalls = tp_cum / max(t.sum(), 1e-9)
    recalls_prev = np.concatenate([[0.0], recalls[:-1]])
    delta_recall = recalls - recalls_prev
    return float(np.sum(precisions * delta_recall))


def binary_metrics_from_prob(prob: torch.Tensor, y: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    pred = (prob >= threshold).float()
    tp = ((pred == 1) & (y == 1)).sum().item()
    fp = ((pred == 1) & (y == 0)).sum().item()
    fn = ((pred == 0) & (y == 1)).sum().item()
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    acc = (pred == y).float().mean().item()
    auprc = compute_auprc(prob, y)
    return {
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auprc": float(auprc),
    }


def _find_best_threshold(prob: torch.Tensor, y: torch.Tensor) -> Tuple[float, Dict[str, float]]:
    best_thr = 0.5
    best_metric = binary_metrics_from_prob(prob, y, threshold=best_thr)
    for thr in np.linspace(0.05, 0.95, 19):
        metric = binary_metrics_from_prob(prob, y, threshold=float(thr))
        if metric["f1"] > best_metric["f1"]:
            best_metric = metric
            best_thr = float(thr)
    return best_thr, best_metric


def collect_teacher_logits_and_labels(
    model: TeacherModel,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_logit, all_y = [], []
    with torch.no_grad():
        for num_x, cat_x, reason_x, llm_x, y_weak, _, _, _, _, _, env_x, _, _ in loader:
            # Phase-6: teacher consumes a flat numeric view; if the env view is
            # split out for the student we re-concatenate it here so the
            # teacher trunk keeps its phase-5 input shape.
            full_num = torch.cat([num_x, env_x], dim=1) if env_x.size(1) > 0 else num_x
            batch = ModelBatch(
                num_x=full_num.to(device),
                cat_x=cat_x.to(device),
                reason_x=reason_x.to(device),
                llm_x=llm_x.to(device),
            )
            out = model(batch)
            all_logit.append(out["anomaly_logit"].cpu())
            all_y.append(y_weak.cpu())
    logit = torch.cat(all_logit, dim=0)
    y = torch.cat(all_y, dim=0)
    return logit, y


def evaluate_teacher(model: TeacherModel, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    logit, y = collect_teacher_logits_and_labels(model, loader, device)
    return binary_metrics(logit, y)


def collect_student_logits_and_labels(
    model: StudentModel,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_logit, all_y = [], []
    with torch.no_grad():
        for num_x, _, _, _, y_weak, _, _, _, _, _, env_x, _, _ in loader:
            out = model(num_x.to(device), env_x.to(device) if env_x.size(1) > 0 else None)
            all_logit.append(out["anomaly_logit"].cpu())
            all_y.append(y_weak.cpu())
    logit = torch.cat(all_logit, dim=0)
    y = torch.cat(all_y, dim=0)
    return logit, y


def evaluate_student(model: StudentModel, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    logit, y = collect_student_logits_and_labels(model, loader, device)
    return binary_metrics(logit, y)


def train_teacher(
    model: TeacherModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    monitor: RunMonitor | None = None,
    alpha_reason_soft: float = 0.5,
    alpha_anomaly_soft: float = 0.25,
) -> Dict[str, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_val_f1 = -1.0
    best_state: Dict[str, torch.Tensor] = {}
    model.to(device)

    for epoch_idx in range(1, epochs + 1):
        model.train()
        loss_sums = {"weak": 0.0, "weak_soft": 0.0, "reason": 0.0, "reason_soft": 0.0, "total": 0.0}
        batch_count = 0
        for (
            num_x,
            cat_x,
            reason_x,
            llm_x,
            y_weak,
            y_reason,
            y_reason_soft,
            y_reason_soft_mask,
            y_weak_soft,
            y_weak_soft_mask,
            env_x,
            _llm_prefix_target,
            _llm_prefix_mask,
        ) in maybe_tqdm(
            train_loader,
            desc=f"teacher {epoch_idx}/{epochs}",
        ):
            full_num = torch.cat([num_x, env_x], dim=1) if env_x.size(1) > 0 else num_x
            batch = ModelBatch(
                num_x=full_num.to(device),
                cat_x=cat_x.to(device),
                reason_x=reason_x.to(device),
                llm_x=llm_x.to(device),
            )
            y_weak = y_weak.to(device)
            y_reason = y_reason.to(device)
            y_reason_soft = y_reason_soft.to(device)
            y_reason_soft_mask = y_reason_soft_mask.to(device)
            y_weak_soft = y_weak_soft.to(device)
            y_weak_soft_mask = y_weak_soft_mask.to(device)
            out = model(batch)
            loss_weak = nn.functional.binary_cross_entropy_with_logits(out["anomaly_logit"], y_weak)
            weak_soft_prob = torch.sigmoid(out["anomaly_logit"])
            weak_mask_sum = y_weak_soft_mask.sum()
            if weak_mask_sum.item() > 0:
                loss_weak_soft = (((weak_soft_prob - y_weak_soft) ** 2) * y_weak_soft_mask).sum() / weak_mask_sum
            else:
                loss_weak_soft = torch.zeros((), device=device, dtype=weak_soft_prob.dtype)
            loss_reason = nn.functional.binary_cross_entropy_with_logits(out["reason_logit"], y_reason)
            soft_prob = torch.sigmoid(out["reason_logit"])
            mask_sum = y_reason_soft_mask.sum()
            if mask_sum.item() > 0:
                loss_reason_soft = (((soft_prob - y_reason_soft) ** 2) * y_reason_soft_mask).sum() / mask_sum
            else:
                loss_reason_soft = torch.zeros((), device=device, dtype=soft_prob.dtype)
            loss = loss_weak + loss_reason + alpha_reason_soft * loss_reason_soft + alpha_anomaly_soft * loss_weak_soft
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sums["weak"] += float(loss_weak.item())
            loss_sums["weak_soft"] += float(loss_weak_soft.item())
            loss_sums["reason"] += float(loss_reason.item())
            loss_sums["reason_soft"] += float(loss_reason_soft.item())
            loss_sums["total"] += float(loss.item())
            batch_count += 1
        val_metric = evaluate_teacher(model, val_loader, device)
        avg_losses = {k: v / max(1, batch_count) for k, v in loss_sums.items()}
        if monitor is not None:
            monitor.log_epoch(
                stage="teacher",
                epoch=epoch_idx,
                epochs=epochs,
                train_losses=avg_losses,
                val_metric=val_metric,
                lr=optimizer.param_groups[0]["lr"],
            )
        if val_metric["f1"] > best_val_f1:
            best_val_f1 = val_metric["f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return evaluate_teacher(model, val_loader, device)


def train_student(
    teacher: TeacherModel,
    student: StudentModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    sparse_lambda: float = 1e-3,
    monitor: RunMonitor | None = None,
    *,
    alpha_kl: float = 0.0,
    alpha_prefix: float = 0.0,
    prefix_mode: str = "mse",
) -> Dict[str, float]:
    teacher.eval().to(device)
    student.to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)

    best_val_f1 = -1.0
    best_state: Dict[str, torch.Tensor] = {}

    for epoch_idx in range(1, epochs + 1):
        student.train()
        loss_sums = {
            "sup": 0.0,
            "prob": 0.0,
            "reason": 0.0,
            "repr": 0.0,
            "kl": 0.0,
            "prefix": 0.0,
            "prefix_mask_frac": 0.0,
            "sparse": 0.0,
            "total": 0.0,
        }
        batch_count = 0
        for (
            num_x,
            cat_x,
            reason_x,
            llm_x,
            y_weak,
            y_reason,
            _y_reason_soft,
            _y_reason_soft_mask,
            _y_weak_soft,
            _y_weak_soft_mask,
            env_x,
            llm_prefix_target,
            llm_prefix_mask,
        ) in maybe_tqdm(
            train_loader,
            desc=f"student {epoch_idx}/{epochs}",
        ):
            num_x = num_x.to(device)
            cat_x = cat_x.to(device)
            reason_x = reason_x.to(device)
            llm_x = llm_x.to(device)
            env_x = env_x.to(device)
            llm_prefix_target = llm_prefix_target.to(device)
            llm_prefix_mask = llm_prefix_mask.to(device)
            y_weak = y_weak.to(device)
            y_reason = y_reason.to(device)

            full_num = torch.cat([num_x, env_x], dim=1) if env_x.size(1) > 0 else num_x
            with torch.no_grad():
                # Teacher trunk consumes flat numeric (num + env) view; LLM
                # view stays optional and is gated by `teacher.use_llm_view`.
                t_out = teacher(ModelBatch(num_x=full_num, cat_x=cat_x, reason_x=reason_x, llm_x=llm_x))
            s_out = student(num_x, env_x if env_x.size(1) > 0 else None)
            losses = distill_loss(
                s_out,
                t_out,
                y_weak,
                y_reason,
                llm_prefix_target=llm_prefix_target if llm_prefix_target.size(1) > 0 else None,
                llm_prefix_mask=llm_prefix_mask if llm_prefix_target.size(1) > 0 else None,
                alpha_kl=alpha_kl,
                alpha_prefix=alpha_prefix,
                prefix_mode=prefix_mode,
            )
            sparse_reg = student.sparse_gate.l1_regularization()
            total = losses["total"] + sparse_lambda * sparse_reg
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            loss_sums["sup"] += float(losses["l_sup"].item())
            loss_sums["prob"] += float(losses["l_prob"].item())
            loss_sums["reason"] += float(losses["l_reason"].item())
            loss_sums["repr"] += float(losses["l_repr"].item())
            loss_sums["kl"] += float(losses["l_kl"].item())
            loss_sums["prefix"] += float(losses["l_prefix"].item())
            loss_sums["prefix_mask_frac"] += float(llm_prefix_mask.mean().item())
            loss_sums["sparse"] += float(sparse_reg.item())
            loss_sums["total"] += float(total.item())
            batch_count += 1

        val_metric = evaluate_student(student, val_loader, device)
        avg_losses = {k: v / max(1, batch_count) for k, v in loss_sums.items()}
        if monitor is not None:
            monitor.log_epoch(
                stage="student",
                epoch=epoch_idx,
                epochs=epochs,
                train_losses=avg_losses,
                val_metric=val_metric,
                lr=optimizer.param_groups[0]["lr"],
            )
        if val_metric["f1"] > best_val_f1:
            best_val_f1 = val_metric["f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}

    student.load_state_dict(best_state)
    return evaluate_student(student, val_loader, device)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_data_lineage(aligned_csv: Path, llm_features_csv: Path | None) -> Dict[str, object]:
    """Record training CSV + optional LLM feature sidecar provenance for reproducibility."""
    try:
        train_generation_command = shlex.join(sys.argv)
    except (TypeError, ValueError):  # pragma: no cover
        train_generation_command = " ".join(sys.argv)
    aligned_res = aligned_csv.resolve()
    training = {
        "aligned_csv": str(aligned_res),
        "aligned_csv_stem": aligned_csv.stem,
        "aligned_csv_basename": aligned_csv.name,
        "aligned_csv_sha256": _sha256_file(aligned_res),
    }
    llm_block: Dict[str, object] = {
        "llm_features_csv": None,
        "provenance_sidecar": None,
        "provenance_sidecar_found": False,
        "build_provenance": None,
    }
    if llm_features_csv is not None:
        llm_res = llm_features_csv.resolve()
        sidecar = (llm_res.parent / f"{llm_res.stem}.provenance.json").resolve()
        llm_block["llm_features_csv"] = str(llm_res)
        llm_block["provenance_sidecar"] = str(sidecar)
        if sidecar.is_file():
            record = json.loads(sidecar.read_text(encoding="utf-8"))
            llm_block["build_provenance"] = record
            llm_block["provenance_sidecar_found"] = True
            build_aligned = record.get("aligned_csv") if isinstance(record, dict) else None
            if isinstance(build_aligned, str) and build_aligned != str(aligned_res):
                llm_block["warning_aligned_csv_mismatch_vs_training"] = {
                    "training_aligned_csv": str(aligned_res),
                    "llm_build_aligned_csv": build_aligned,
                    "hint": "LLM CSV was generated from a different aligned table than this train_distill run.",
                }
        else:
            llm_block["missing_sidecar_help"] = (
                "Run code/build_llm_teacher_features.py with the same --output_csv stem to write "
                f"{sidecar.name} next to the LLM features CSV."
            )
    return {
        "training_run": {
            "argv": sys.argv.copy(),
            "generation_command": train_generation_command,
        },
        "training_input": training,
        "llm_features": llm_block,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train teacher and distill tiny student for fee anomaly analysis.")
    parser.add_argument(
        "--aligned_csv",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "dataset" / "aligned_customer_month.csv",
        help="Path to aligned customer-month CSV.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "checkpoints" / "fee_anomaly",
        help="Directory to save checkpoints and report.",
    )
    parser.add_argument("--teacher_epochs", type=int, default=12)
    parser.add_argument("--student_epochs", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--teacher_lr", type=float, default=1e-3)
    parser.add_argument("--student_lr", type=float, default=1e-3)
    parser.add_argument(
        "--llm_features_csv",
        type=Path,
        default=None,
        help="Optional offline LLM feature CSV keyed by 客户ID,month.",
    )
    parser.add_argument(
        "--teacher_llm_soft_weight",
        type=float,
        default=0.5,
        help="Weight for teacher LLM soft reason supervision loss.",
    )
    parser.add_argument(
        "--teacher_llm_anomaly_soft_weight",
        type=float,
        default=0.25,
        help="Weight for teacher LLM anomaly-prob soft supervision loss.",
    )
    parser.add_argument(
        "--split_mode",
        type=str,
        default="temporal",
        choices=["temporal", "cold_start", "cross_domain_industry", "cross_domain_voltage", "tariff_shift"],
        help="Train/val/test split mode (HORIZON-style evaluation).",
    )
    parser.add_argument("--cross_domain_industry_value", type=str, default=None)
    parser.add_argument("--cross_domain_voltage_value", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--calibration_n_bins",
        type=int,
        default=15,
        help="Bins for ECE / reliability in train_report.calibration (0 to disable).",
    )
    # ----- Phase-6 (D module v1) options ----------------------------------
    parser.add_argument(
        "--use_env_view",
        action="store_true",
        help=(
            "Phase-6: route env_self_*/env_peer_*/env_tariff_*/env_season_* "
            "to a dedicated student EnvironmentEncoder branch. The teacher "
            "still consumes (num + env) jointly, so its phase-5 input shape "
            "is unchanged."
        ),
    )
    parser.add_argument(
        "--student_latent_dim",
        type=int,
        default=0,
        help=(
            "Phase-6: when > 0 the student exposes a (mu, logvar) latent "
            "head reparameterized as z = mu + sigma*eps (training time); "
            "anomaly/reason heads consume z. Set to 0 to disable (phase-5 "
            "behaviour)."
        ),
    )
    parser.add_argument(
        "--student_alpha_kl",
        type=float,
        default=0.0,
        help=(
            "Phase-6: weight for the KL(N(mu,sigma)||N(0,I)) regularizer on "
            "the student latent. Effective only when --student_latent_dim>0."
        ),
    )
    parser.add_argument(
        "--student_alpha_prefix",
        type=float,
        default=0.0,
        help=(
            "Phase-6: weight for prefix_alignment_loss between the student "
            "latent (projected) and the LLM teacher's llm_prefix_emb_*. The "
            "loss is mask-averaged over LLM-covered rows only."
        ),
    )
    parser.add_argument(
        "--student_prefix_mode",
        type=str,
        default="mse",
        choices=["mse", "cosine"],
        help="Phase-6: prefix alignment metric (MSE or 1-cosine).",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    monitor = RunMonitor(args.output_dir)

    set_seed(args.seed)
    df = pd.read_csv(args.aligned_csv)
    if args.llm_features_csv is not None:
        llm_df = pd.read_csv(args.llm_features_csv)
        required_keys = {"客户ID", "month"}
        if not required_keys.issubset(set(llm_df.columns)):
            raise ValueError(f"LLM feature CSV must contain keys: {required_keys}")
        llm_df["客户ID"] = llm_df["客户ID"].astype(str)
        llm_df["month"] = llm_df["month"].astype(str)
        df["客户ID"] = df["客户ID"].astype(str)
        df["month"] = df["month"].astype(str)
        df = df.merge(llm_df, on=["客户ID", "month"], how="left")
        print(f"[INFO] merged llm features from {args.llm_features_csv}")
        _side = args.llm_features_csv.resolve().parent / f"{args.llm_features_csv.resolve().stem}.provenance.json"
        if not _side.is_file():
            print(
                f"[WARN] LLM provenance sidecar missing ({_side.name}); "
                "train_report will record missing_sidecar_help under data_lineage.llm_features."
            )
    train_df, val_df, test_df, split_info = split_dataset(
        df,
        mode=args.split_mode,
        seed=args.seed,
        cross_domain_industry_value=args.cross_domain_industry_value,
        cross_domain_voltage_value=args.cross_domain_voltage_value,
    )
    print(f"[INFO] split_info: {json.dumps(split_info, ensure_ascii=False)}")

    reason_label_cols = [c for c in df.columns if c.startswith("label_reason_") or c.startswith("label_reason_rule_")]
    num_cols, cat_cols, reason_cols, env_cols = infer_columns(df, use_env_view=args.use_env_view)
    if not reason_cols:
        print("[INFO] reason view inputs disabled by anti-leakage filter.")
    if not reason_label_cols:
        reason_label_cols = ["weak_label"]
    llm_emb_cols = get_llm_embedding_cols(df)
    llm_prefix_target_cols = get_llm_prefix_target_cols(df)
    llm_anomaly_prob_col = LLM_ANOMALY_PROB_COL if LLM_ANOMALY_PROB_COL in df.columns else None
    llm_reason_prob_cols: List[str | None] = []
    for col in reason_label_cols:
        llm_prob_col = _llm_prob_col_for_reason_label(col)
        llm_reason_prob_cols.append(llm_prob_col if llm_prob_col in df.columns else None)
    llm_reason_soft_dim = int(sum(col is not None for col in llm_reason_prob_cols))
    print(
        f"[INFO] llm embedding dims: {len(llm_emb_cols)}, "
        f"llm soft reason dims: {llm_reason_soft_dim}, "
        f"llm anomaly soft: {llm_anomaly_prob_col is not None}, "
        f"llm prefix target cols: {len(llm_prefix_target_cols)}, "
        f"env view cols: {len(env_cols)} (use_env_view={args.use_env_view})"
    )

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
        env_cols=env_cols,
        llm_prefix_target_cols=llm_prefix_target_cols,
    )

    train_loader = DataLoader(FeeDataset(train_bundle), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(FeeDataset(val_bundle), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(FeeDataset(test_bundle), batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cat_cardinalities = [len(cat_maps[c]) for c in cat_cols]
    reason_dim = train_bundle.reason_x.size(1)
    reason_label_dim = train_bundle.y_reason.size(1)
    llm_dim = train_bundle.llm_x.size(1)
    env_dim = train_bundle.env_x.size(1)
    student_num_dim = train_bundle.num_x.size(1)
    teacher_num_dim = student_num_dim + env_dim
    prefix_target_dim = train_bundle.llm_prefix_target.size(1)
    train_prefix_coverage = float(train_bundle.llm_prefix_mask.mean().item()) if prefix_target_dim > 0 else 0.0
    val_prefix_coverage = float(val_bundle.llm_prefix_mask.mean().item()) if prefix_target_dim > 0 else 0.0
    test_prefix_coverage = float(test_bundle.llm_prefix_mask.mean().item()) if prefix_target_dim > 0 else 0.0
    print(
        f"[INFO] env_dim={env_dim}, student_num_dim={student_num_dim}, "
        f"teacher_num_dim={teacher_num_dim}, prefix_target_dim={prefix_target_dim}, "
        f"prefix_coverage train/val/test = {train_prefix_coverage:.4f}/"
        f"{val_prefix_coverage:.4f}/{test_prefix_coverage:.4f}"
    )

    teacher = TeacherModel(
        num_dim=teacher_num_dim,
        cat_cardinalities=cat_cardinalities,
        reason_dim=reason_dim,
        llm_dim=llm_dim,
        reason_label_dim=reason_label_dim,
        hidden_dim=128,
        cat_emb_dim=16,
    )
    teacher_val_metric = train_teacher(
        model=teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.teacher_epochs,
        lr=args.teacher_lr,
        monitor=monitor,
        alpha_reason_soft=args.teacher_llm_soft_weight,
        alpha_anomaly_soft=args.teacher_llm_anomaly_soft_weight,
    )
    teacher_test_metric = evaluate_teacher(teacher, test_loader, device)
    teacher_val_logit, teacher_val_y = collect_teacher_logits_and_labels(teacher, val_loader, device)
    teacher_test_logit, teacher_test_y = collect_teacher_logits_and_labels(teacher, test_loader, device)
    teacher_best_thr, teacher_val_metric_best_thr = _find_best_threshold(torch.sigmoid(teacher_val_logit), teacher_val_y)
    teacher_test_metric_best_thr = binary_metrics_from_prob(
        torch.sigmoid(teacher_test_logit),
        teacher_test_y,
        threshold=teacher_best_thr,
    )

    student_prefix_dim = prefix_target_dim if args.student_alpha_prefix > 0 else 0
    if args.student_alpha_prefix > 0 and prefix_target_dim == 0:
        print(
            "[WARN] --student_alpha_prefix > 0 but no llm prefix target columns found; "
            "prefix alignment loss will be disabled at runtime."
        )
    student = StudentModel(
        num_dim=student_num_dim,
        hidden_dim=64,
        reason_dim=reason_label_dim,
        teacher_repr_dim=128,
        env_dim=env_dim,
        latent_dim=int(args.student_latent_dim),
        prefix_dim=int(student_prefix_dim),
    )
    student_val_metric = train_student(
        teacher=teacher,
        student=student,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.student_epochs,
        lr=args.student_lr,
        monitor=monitor,
        alpha_kl=float(args.student_alpha_kl),
        alpha_prefix=float(args.student_alpha_prefix),
        prefix_mode=args.student_prefix_mode,
    )
    student_test_metric = evaluate_student(student, test_loader, device)
    student_val_logit, student_val_y = collect_student_logits_and_labels(student, val_loader, device)
    student_test_logit, student_test_y = collect_student_logits_and_labels(student, test_loader, device)
    student_best_thr, student_val_metric_best_thr = _find_best_threshold(torch.sigmoid(student_val_logit), student_val_y)
    student_test_metric_best_thr = binary_metrics_from_prob(
        torch.sigmoid(student_test_logit),
        student_test_y,
        threshold=student_best_thr,
    )

    calib_block: Dict[str, object] | None = None
    if args.calibration_n_bins > 0:
        nb = int(args.calibration_n_bins)
        calib_block = {
            "n_bins": nb,
            "teacher": {
                "val": calibration_report(
                    torch.sigmoid(teacher_val_logit).detach().cpu().numpy(),
                    teacher_val_y.detach().cpu().numpy(),
                    n_bins=nb,
                ),
                "test": calibration_report(
                    torch.sigmoid(teacher_test_logit).detach().cpu().numpy(),
                    teacher_test_y.detach().cpu().numpy(),
                    n_bins=nb,
                ),
            },
            "student": {
                "val": calibration_report(
                    torch.sigmoid(student_val_logit).detach().cpu().numpy(),
                    student_val_y.detach().cpu().numpy(),
                    n_bins=nb,
                ),
                "test": calibration_report(
                    torch.sigmoid(student_test_logit).detach().cpu().numpy(),
                    student_test_y.detach().cpu().numpy(),
                    n_bins=nb,
                ),
            },
        }

    teacher_ckpt = args.output_dir / "teacher.pt"
    student_ckpt = args.output_dir / "student.pt"
    torch.save(teacher.state_dict(), teacher_ckpt)
    torch.save(student.state_dict(), student_ckpt)

    report = {
        "data_lineage": build_data_lineage(args.aligned_csv, args.llm_features_csv),
        "config": {
            "teacher_epochs": args.teacher_epochs,
            "student_epochs": args.student_epochs,
            "batch_size": args.batch_size,
            "teacher_lr": args.teacher_lr,
            "student_lr": args.student_lr,
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "reason_cols": reason_cols,
            "reason_label_cols": reason_label_cols,
            "llm_features_csv": str(args.llm_features_csv) if args.llm_features_csv is not None else None,
            "llm_emb_cols": llm_emb_cols,
            "llm_reason_prob_cols": llm_reason_prob_cols,
            "llm_anomaly_prob_col": llm_anomaly_prob_col,
            "teacher_llm_soft_weight": args.teacher_llm_soft_weight,
            "teacher_llm_anomaly_soft_weight": args.teacher_llm_anomaly_soft_weight,
            "split_mode": args.split_mode,
            "split_info": split_info,
            "aligned_csv": str(args.aligned_csv.resolve()),
            "aligned_csv_stem": args.aligned_csv.stem,
            "llm_dim": int(llm_dim),
            "calibration_n_bins": int(args.calibration_n_bins),
            "phase6": {
                "use_env_view": bool(args.use_env_view),
                "env_cols": list(env_cols),
                "env_dim": int(env_dim),
                "student_num_dim": int(student_num_dim),
                "teacher_num_dim": int(teacher_num_dim),
                "student_latent_dim": int(args.student_latent_dim),
                "student_alpha_kl": float(args.student_alpha_kl),
                "student_alpha_prefix": float(args.student_alpha_prefix),
                "student_prefix_mode": str(args.student_prefix_mode),
                "student_prefix_dim": int(student_prefix_dim),
                "llm_prefix_target_cols": list(llm_prefix_target_cols),
                "prefix_coverage": {
                    "train": train_prefix_coverage,
                    "val": val_prefix_coverage,
                    "test": test_prefix_coverage,
                },
            },
        },
        "teacher": {"val": teacher_val_metric, "test": teacher_test_metric},
        "student": {"val": student_val_metric, "test": student_test_metric},
        "threshold_tuning": {
            "teacher": {
                "best_threshold_from_val_f1": teacher_best_thr,
                "val_at_best_threshold": teacher_val_metric_best_thr,
                "test_at_val_best_threshold": teacher_test_metric_best_thr,
            },
            "student": {
                "best_threshold_from_val_f1": student_best_thr,
                "val_at_best_threshold": student_val_metric_best_thr,
                "test_at_val_best_threshold": student_test_metric_best_thr,
            },
        },
        "monitoring": {
            "metrics_jsonl": str(args.output_dir / "metrics.jsonl"),
            "tensorboard_dir": str(args.output_dir / "tb_logs"),
            "curves_png": str(args.output_dir / "training_curves.png"),
        },
    }
    if calib_block is not None:
        report["calibration"] = calib_block
    (args.output_dir / "train_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    monitor.close()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[OK] teacher ckpt: {teacher_ckpt}")
    print(f"[OK] student ckpt: {student_ckpt}")


if __name__ == "__main__":
    main()
