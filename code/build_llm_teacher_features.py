#!/usr/bin/env python3
"""Build LLM-enhanced teacher features for customer-month samples.

Providers:
- mock: deterministic local heuristic (default)
- openai: real API call with cache + retry

E2P-style prompting (default ``--prompt_version v3_e2p_natural``):
three blocks [ENVIRONMENT] / [SPARSE_EVENTS] / [TASK] compile tabular
context into an LLM-readable user prefix; JSON output supplies
``llm_anomaly_prob``, ``llm_reason_prob_*``, and ``llm_prefix_emb_*`` for
the teacher's fourth view (see ``TeacherModel`` in ``models.py``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _stable_rng(customer_id: str, month: str, seed: int) -> np.random.RandomState:
    key = f"{customer_id}|{month}|{seed}".encode("utf-8")
    digest = hashlib.md5(key).hexdigest()[:8]
    return np.random.RandomState(int(digest, 16))


def _reason_label_to_name(label_col: str) -> str:
    return label_col.replace("label_", "")


def _make_cache_key(
    customer_id: str,
    month: str,
    model: str,
    embedding_dim: int,
    reason_names: List[str],
    prompt_version: str,
    env_hash: str,
    event_hash: str,
) -> str:
    raw = json.dumps(
        {
            "customer_id": customer_id,
            "month": month,
            "model": model,
            "embedding_dim": embedding_dim,
            "reason_names": reason_names,
            "prompt_version": prompt_version,
            "env_hash": env_hash,
            "event_hash": event_hash,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _to_float(v: object, default: float = 0.0) -> float:
    parsed = pd.to_numeric(v, errors="coerce")
    if pd.isna(parsed):
        return float(default)
    return float(parsed)


def _to_str(v: object, default: str = "未知") -> str:
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default


def _season_name(month_raw: object) -> str:
    month = 1
    dt = pd.to_datetime(month_raw, errors="coerce")
    if not pd.isna(dt):
        month = int(dt.month)
    else:
        raw = _to_str(month_raw, "1")
        if "-" in raw:
            parts = raw.split("-")
            month = int(max(1, min(12, _to_float(parts[-1], 1))))
        else:
            month = int(max(1, min(12, _to_float(raw, 1))))
    if month in (12, 1, 2):
        return "冬季"
    if month in (3, 4, 5):
        return "春季"
    if month in (6, 7, 8):
        return "夏季"
    return "秋季"


def _collect_env_values(row: pd.Series) -> Dict[str, object]:
    out: Dict[str, object] = {
        "客户类型": _to_str(row.get("客户类型")),
        "行业编码": _to_str(row.get("行业编码")),
        "电压等级": _to_str(row.get("电压等级")),
        "合同容量": round(_to_float(row.get("合同容量", 0.0)), 4),
        "基准费率": round(_to_float(row.get("env_tariff_base_rate", row.get("基准费率", 0.0))), 6),
        "附加费率": round(_to_float(row.get("env_tariff_extra_rate", row.get("附加费率", 0.0))), 6),
        "费率类型": _to_str(row.get("费率类型")),
        "当月是否变价": int(_to_float(row.get("env_tariff_change_flag", 0.0)) > 0.5),
        "季节": _season_name(row.get("month_dt", row.get("month"))),
        "同群费中位数": round(_to_float(row.get("env_peer_fee_median", 0.0)), 4),
        "同群费p90": round(_to_float(row.get("env_peer_fee_p90", 0.0)), 4),
        "自身3月费均值": round(_to_float(row.get("env_self_fee_roll3_mean", 0.0)), 4),
        "自身3月费波动": round(_to_float(row.get("env_self_fee_roll3_std", 0.0)), 4),
        "单价历史偏离": round(_to_float(row.get("env_unit_price_dev_vs_self_history", 0.0)), 4),
    }
    slope = _to_float(row.get("env_self_fee_roll6_slope", float("nan")))
    if slope == slope:  # not NaN
        out["自身6月电费斜率"] = round(slope, 6)
    return out


def _collect_event_values(row: pd.Series) -> Dict[str, object]:
    event_cols = [
        "event_fee_spike",
        "event_meter_increase",
        "event_reading_mismatch",
        "event_read_abn",
        "event_unit_price_dev_high",
        "event_energy_spike",
    ]
    return {
        "event_count": round(_to_float(row.get("event_count", 0.0)), 4),
        "event_severity": round(_to_float(row.get("event_severity", 0.0)), 4),
        "event_dominant_type": int(_to_float(row.get("event_dominant_type", 0.0))),
        "fee_vs_peer_z": round(_to_float(row.get("env_peer_fee_vs_z", 0.0)), 4),
        "unit_price_dev_vs_peer_z": round(_to_float(row.get("env_peer_unit_price_dev_vs_z", 0.0)), 4),
        "events": {
            c: int(_to_float(row.get(c, 0.0)) > 0.5)
            for c in event_cols
        },
    }


def _hash_feature_dict(d: Dict[str, object]) -> str:
    raw = json.dumps(d, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _format_environment_nl(row: pd.Series, env_values: Dict[str, object]) -> str:
    """自然语言环境段：与 E2P 中「user prefix 的环境上下文」对齐，便于 LLM 形成稳定语义前缀。"""
    change = "是" if int(env_values.get("当月是否变价", 0)) > 0 else "否"
    extra_slope = ""
    if "自身6月电费斜率" in env_values:
        extra_slope = f"\n长期电费趋势(约6月斜率): {env_values['自身6月电费斜率']}"
    return (
        f"客户类型: {env_values['客户类型']}\n"
        f"行业编码: {env_values['行业编码']}\n"
        f"电压等级: {env_values['电压等级']}\n"
        f"合同容量: {env_values['合同容量']} kVA\n"
        f"统计月份: {_to_str(row.get('month'))}\n"
        f"当前季节: {env_values['季节']}\n"
        f"本月电价: 基准 {env_values['基准费率']}, 附加 {env_values['附加费率']}, "
        f"费率类型「{env_values['费率类型']}」, 是否阶梯/复合以费率类型为准\n"
        f"本月是否变价: {change}\n"
        f"同群基线: 当月同群总电费中位数 {env_values['同群费中位数']} 元, p90 {env_values['同群费p90']} 元\n"
        f"自身基线: 过去约3个月电费均值 {env_values['自身3月费均值']} 元, 波动(std) {env_values['自身3月费波动']} 元"
        f"{extra_slope}\n"
        f"单价相对自身历史偏离: {env_values['单价历史偏离']}"
    )


_EVENT_FLAG_CN = {
    "event_fee_spike": "费用突增",
    "event_meter_increase": "示数突增",
    "event_reading_mismatch": "读数不匹配",
    "event_read_abn": "抄表异常",
    "event_unit_price_dev_high": "单价偏高",
    "event_energy_spike": "电量突增",
}


def _format_sparse_events_nl(row: pd.Series, event_values: Dict[str, object]) -> str:
    ev = event_values.get("events") or {}
    fired = [_EVENT_FLAG_CN[k] for k, v in ev.items() if int(v) > 0 and k in _EVENT_FLAG_CN]
    trig = "、".join(fired) if fired else "无显式规则事件触发(仅连续量)"
    fee_mom = _to_float(row.get("fee_mom_ratio", float("nan")))
    fee_mom_s = f"{fee_mom:.2f}倍" if fee_mom == fee_mom else "未知"
    read_abn = "是" if _to_float(row.get("read_has_abn", 0.0)) > 0.5 else "否"
    peer_z = event_values.get("fee_vs_peer_z", 0.0)
    upz = event_values.get("unit_price_dev_vs_peer_z", 0.0)
    return (
        f"本月触发事件摘要: {trig}\n"
        f"事件计数: {event_values.get('event_count', 0)}\n"
        f"最严重幅度(聚合): {event_values.get('event_severity', 0)}\n"
        f"费用环比(对上月): 约 {fee_mom_s}\n"
        f"读数/抄表异常标记: {read_abn}\n"
        f"电费相对同群 z: {peer_z}, 单价相对同群 z: {upz}"
    )


def _build_prompt_v2_json_sections(
    row: pd.Series,
    reason_names: List[str],
    embedding_dim: int,
    env_values: Dict[str, object],
    event_values: Dict[str, object],
) -> str:
    """旧版 v2：三段标题下仍为 JSON 块，便于与历史 pilot 缓存对齐。"""
    reason_json_hint = {name: 0.0 for name in reason_names}
    return (
        "你是电费异常原因分析助手。请基于给定信息输出严格 JSON，不要输出其他文本。\n"
        f"客户ID: {row['客户ID']}\n"
        f"月份: {row['month']}\n"
        "\n[ENVIRONMENT]\n"
        f"{json.dumps(env_values, ensure_ascii=False)}\n"
        "\n[SPARSE_EVENTS]\n"
        f"{json.dumps(event_values, ensure_ascii=False)}\n"
        "\n[TASK]\n"
        "请给出异常概率、原因概率、风险prefix向量和一句解释。\n"
        "输出JSON结构如下：\n"
        "{\n"
        '  "anomaly_prob": 0.0,\n'
        f'  "reason_probs": {json.dumps(reason_json_hint, ensure_ascii=False)},\n'
        f'  "prefix_embedding": [长度为{embedding_dim}的浮点数组],\n'
        '  "explanation": "一句话中文解释"\n'
        "}\n"
        "约束：anomaly_prob与reason_probs值均在[0,1]；prefix_embedding每个值在[-1,1]。"
    )


def _build_prompt_v3_e2p_natural(
    row: pd.Series,
    reason_names: List[str],
    embedding_dim: int,
    env_values: Dict[str, object],
    event_values: Dict[str, object],
) -> str:
    """E2P 风格三段式 user 前缀：环境叙述 | 稀疏事件叙述 | 任务与 JSON 契约。

    与 v2 的差异：前两段为自然语言，模拟「把结构化表特征编译成 LLM 可读 prefix」，
    输出仍为严格 JSON，便于解析与下游 `llm_prefix_emb_*` 注入教师第四视图。
    """
    reason_json_hint = {name: 0.0 for name in reason_names}
    env_nl = _format_environment_nl(row, env_values)
    sparse_nl = _format_sparse_events_nl(row, event_values)
    return (
        "你是电力计费与异常分析助手。下面分三段给出上下文，最后一段是输出要求。\n"
        "除最后一行 JSON 外不要使用代码块；最终回答必须是**单行或可解析的 JSON 对象**。\n"
        f"客户ID: {row['客户ID']}\n\n"
        "[ENVIRONMENT]\n"
        f"{env_nl}\n\n"
        "[SPARSE_EVENTS]\n"
        f"{sparse_nl}\n\n"
        "[TASK]\n"
        "请基于以上电力环境与本月稀疏事件，输出：\n"
        "- 异常概率 anomaly_prob\n"
        f"- 各原因概率 reason_probs（键与下列一致）：{json.dumps(list(reason_json_hint.keys()), ensure_ascii=False)}\n"
        f"- 风险表征向量 prefix_embedding（长度 {embedding_dim}，模拟 E2P 的 user-prefix 嵌入，供下游模型作 condition）\n"
        "- 简明风险解释 explanation（**不超过30个汉字**）\n\n"
        "输出 JSON 模板：\n"
        "{\n"
        '  "anomaly_prob": 0.0,\n'
        f'  "reason_probs": {json.dumps(reason_json_hint, ensure_ascii=False)},\n'
        f'  "prefix_embedding": [长度为{embedding_dim}的浮点数组],\n'
        '  "explanation": "不超过30字"\n'
        "}\n"
        "约束：anomaly_prob 与 reason_probs 均在 [0,1]；prefix_embedding 每个元素在 [-1,1]。"
    )


def build_user_prompt(
    row: pd.Series,
    reason_names: List[str],
    embedding_dim: int,
    env_values: Dict[str, object],
    event_values: Dict[str, object],
    prompt_version: str,
) -> str:
    if prompt_version in ("v2_due_sbr", "v2"):
        return _build_prompt_v2_json_sections(
            row, reason_names, embedding_dim, env_values, event_values
        )
    # 默认：v3_e2p_natural 及后续以 v3_ 前缀的变体
    return _build_prompt_v3_e2p_natural(row, reason_names, embedding_dim, env_values, event_values)


def _load_cache(cache_file: Path) -> Dict[str, Dict[str, object]]:
    if not cache_file.exists():
        return {}
    cache: Dict[str, Dict[str, object]] = {}
    for line in cache_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        cache[item["cache_key"]] = item["payload"]
    return cache


def _append_cache(cache_file: Path, cache_key: str, payload: Dict[str, object]) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with cache_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"cache_key": cache_key, "payload": payload}, ensure_ascii=False) + "\n")


def _mock_payload(
    row: pd.Series,
    reason_names: List[str],
    embedding_dim: int,
    seed: int,
) -> Dict[str, object]:
    # Deliberately do NOT use weak_label — mock should mirror OpenAI: same env + events only.
    fee_peer_z = _to_float(row.get("env_peer_fee_vs_z", 0.0))
    unit_peer_z = _to_float(row.get("env_peer_unit_price_dev_vs_z", 0.0))
    event_count = _to_float(row.get("event_count", 0.0))
    event_severity = _to_float(row.get("event_severity", 0.0))
    anomaly_linear = (
        0.35 * np.tanh(fee_peer_z / 2.0)
        + 0.25 * np.tanh(unit_peer_z / 2.0)
        + 0.25 * event_severity
        + 0.12 * event_count
        - 0.45
    )
    anomaly_prob = float(np.clip(_sigmoid(np.array([anomaly_linear]))[0], 0.01, 0.99))
    reason_probs: Dict[str, float] = {}
    for reason in reason_names:
        label_key = reason if reason.startswith("label_") else f"label_{reason}"
        src = _to_float(row.get(label_key, 0.0))
        prob = float(_sigmoid(np.array([1.2 * src + 0.6 * anomaly_prob - 0.55]))[0])
        reason_probs[reason] = float(np.clip(prob, 0.02, 0.98))

    rng = _stable_rng(str(row["客户ID"]), str(row["month"]), seed)
    vec = rng.normal(loc=0.0, scale=1.0, size=embedding_dim).astype(np.float32)
    vec[0] += np.float32(anomaly_prob * 0.75)
    vec[1] += np.float32(event_severity * 0.5)
    vec = vec / (np.linalg.norm(vec) + 1e-6)
    top2 = sorted(reason_probs.items(), key=lambda x: x[1], reverse=True)[:2]
    explanation = f"Top{top2};同群与事件驱动风险"
    if len(explanation) > 30:
        explanation = explanation[:30]
    return {
        "anomaly_prob": anomaly_prob,
        "reason_probs": reason_probs,
        "prefix_embedding": vec.tolist(),
        "explanation": explanation,
    }


def _accumulate_payload(
    idx: int,
    payload: Dict[str, object],
    reason_names: List[str],
    embedding_dim: int,
    emb_matrix: np.ndarray,
    llm_anomaly_prob: np.ndarray,
    llm_reason_values: Dict[str, np.ndarray],
    explanations: List[str],
) -> None:
    reason_probs = payload.get("reason_probs", {}) or {}
    llm_anomaly_prob[idx] = float(np.clip(_to_float(payload.get("anomaly_prob", 0.0)), 0.0, 1.0))
    for name in reason_names:
        prob = float(reason_probs.get(name, 0.0)) if isinstance(reason_probs, dict) else 0.0
        llm_reason_values[f"llm_reason_prob_{name}"][idx] = float(np.clip(prob, 0.0, 1.0))
    emb = payload.get("prefix_embedding", payload.get("risk_embedding", []))
    emb_np = np.array(emb, dtype=np.float32) if isinstance(emb, list) else np.zeros((0,), dtype=np.float32)
    if emb_np.size < embedding_dim:
        padded = np.zeros((embedding_dim,), dtype=np.float32)
        padded[: emb_np.size] = emb_np[:embedding_dim]
        emb_np = padded
    else:
        emb_np = emb_np[:embedding_dim]
    emb_matrix[idx] = emb_np
    explanations[idx] = str(payload.get("explanation", ""))


def _strip_json_fences(text: str) -> str:
    """Best-effort cleanup for OpenAI-compatible endpoints (e.g. GLM/智谱) that may wrap JSON
    in ```json ... ``` fences or add minor leading/trailing chatter."""
    s = (text or "").strip()
    if s.startswith("```"):
        s = s[3:]
        if s[:4].lower() == "json":
            s = s[4:]
        s = s.lstrip("\n").lstrip()
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3].rstrip()
    if not s:
        return s
    first = s.find("{")
    last = s.rfind("}")
    if first != -1 and last != -1 and last > first:
        return s[first : last + 1]
    return s


def _call_openai_payload(
    row: pd.Series,
    reason_names: List[str],
    embedding_dim: int,
    model: str,
    max_retries: int,
    retry_wait_seconds: float,
    prompt_version: str,
    base_url: str | None = None,
) -> Dict[str, object]:
    from openai import OpenAI  # lazy import

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")
    client_kwargs: Dict[str, object] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)  # type: ignore[arg-type]
    env_values = _collect_env_values(row)
    event_values = _collect_event_values(row)
    prompt = build_user_prompt(
        row,
        reason_names=reason_names,
        embedding_dim=embedding_dim,
        env_values=env_values,
        event_values=event_values,
        prompt_version=prompt_version,
    )

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是电费异常原因分析助手，必须只输出严格 JSON，禁止输出任何其他文本。",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            text_raw = resp.choices[0].message.content if resp.choices else ""
            text = _strip_json_fences(text_raw or "")
            data = json.loads(text)
            anomaly_prob = float(np.clip(_to_float(data.get("anomaly_prob", 0.0)), 0.0, 1.0))
            return {
                "anomaly_prob": anomaly_prob,
                "reason_probs": data.get("reason_probs", {}),
                "prefix_embedding": data.get("prefix_embedding", data.get("risk_embedding", [])),
                "explanation": str(data.get("explanation", "")),
            }
        except Exception as exc:  # pragma: no cover
            last_error = exc
            if attempt == max_retries:
                break
            time.sleep(retry_wait_seconds * (attempt + 1))
    raise RuntimeError(f"OpenAI-compatible call failed after retries: {last_error}")


def build_llm_features(
    df: pd.DataFrame,
    provider: str,
    model: str,
    embedding_dim: int,
    seed: int,
    cache_file: Path,
    prompt_version: str,
    max_retries: int,
    retry_wait_seconds: float,
    limit_rows: int,
    openai_base_url: str | None = None,
    concurrency: int = 1,
    qps_sleep: float = 0.0,
) -> pd.DataFrame:
    if "客户ID" not in df.columns or "month" not in df.columns:
        raise ValueError("Input CSV must contain 客户ID and month columns.")
    if limit_rows > 0:
        df = df.head(limit_rows).copy()

    reason_label_cols: List[str] = sorted(
        [c for c in df.columns if c.startswith("label_reason_") or c.startswith("label_reason_rule_")]
    )
    reason_names = [_reason_label_to_name(c) for c in reason_label_cols]
    out = df[["客户ID", "month"]].copy()
    cache = _load_cache(cache_file)
    cache_lock = threading.Lock()

    emb_matrix = np.zeros((len(df), embedding_dim), dtype=np.float32)
    llm_anomaly_prob = np.zeros((len(df),), dtype=np.float32)
    llm_reason_values = {f"llm_reason_prob_{name}": np.zeros(len(df), dtype=np.float32) for name in reason_names}
    explanations: List[str] = ["" for _ in range(len(df))]

    df_indexed = df.reset_index(drop=True)

    def _resolve_one(idx: int, row: pd.Series) -> Tuple[int, Dict[str, object]]:
        cache_key = _make_cache_key(
            customer_id=str(row["客户ID"]),
            month=str(row["month"]),
            model=model,
            embedding_dim=embedding_dim,
            reason_names=reason_names,
            prompt_version=prompt_version,
            env_hash=_hash_feature_dict(_collect_env_values(row)),
            event_hash=_hash_feature_dict(_collect_event_values(row)),
        )
        with cache_lock:
            payload = cache.get(cache_key)
        if payload is None:
            if provider == "openai":
                payload = _call_openai_payload(
                    row=row,
                    reason_names=reason_names,
                    embedding_dim=embedding_dim,
                    model=model,
                    max_retries=max_retries,
                    retry_wait_seconds=retry_wait_seconds,
                    prompt_version=prompt_version,
                    base_url=openai_base_url,
                )
            else:
                payload = _mock_payload(row=row, reason_names=reason_names, embedding_dim=embedding_dim, seed=seed)
            with cache_lock:
                cache[cache_key] = payload
                _append_cache(cache_file, cache_key, payload)
            if qps_sleep > 0.0:
                time.sleep(qps_sleep)
        return idx, payload

    workers = max(1, int(concurrency))
    if workers > 1 and provider != "openai":
        # Mock provider is CPU-bound + cheap; threads add overhead without benefit.
        workers = 1

    if workers == 1:
        rows_iter = df_indexed.iterrows()
        if tqdm is not None:
            rows_iter = tqdm(rows_iter, total=len(df_indexed), desc=f"llm-features({provider})", dynamic_ncols=True)
        for idx, row in rows_iter:
            _, payload = _resolve_one(int(idx), row)
            _accumulate_payload(
                idx=int(idx),
                payload=payload,
                reason_names=reason_names,
                embedding_dim=embedding_dim,
                emb_matrix=emb_matrix,
                llm_anomaly_prob=llm_anomaly_prob,
                llm_reason_values=llm_reason_values,
                explanations=explanations,
            )
    else:
        progress = tqdm(total=len(df_indexed), desc=f"llm-features({provider}, k={workers})", dynamic_ncols=True) if tqdm is not None else None
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_resolve_one, int(idx), row)
                for idx, row in df_indexed.iterrows()
            ]
            for fut in as_completed(futures):
                idx, payload = fut.result()
                _accumulate_payload(
                    idx=idx,
                    payload=payload,
                    reason_names=reason_names,
                    embedding_dim=embedding_dim,
                    emb_matrix=emb_matrix,
                    llm_anomaly_prob=llm_anomaly_prob,
                    llm_reason_values=llm_reason_values,
                    explanations=explanations,
                )
                if progress is not None:
                    progress.update(1)
        if progress is not None:
            progress.close()

    for col, values in llm_reason_values.items():
        out[col] = values
    out["llm_anomaly_prob"] = llm_anomaly_prob
    for i in range(embedding_dim):
        out[f"llm_prefix_emb_{i:02d}"] = emb_matrix[:, i]
        # Backward-compatibility: keep legacy column names to avoid breaking old checkpoints/tools.
        out[f"llm_risk_emb_{i:02d}"] = emb_matrix[:, i]
    out["llm_explanation"] = explanations
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LLM-enhanced teacher features CSV.")
    parser.add_argument("--aligned_csv", type=Path, required=True, help="Input aligned customer-month CSV.")
    parser.add_argument("--output_csv", type=Path, required=True, help="Output LLM features CSV.")
    parser.add_argument("--provider", type=str, default="mock", choices=["mock", "openai"], help="Feature provider.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM model name when provider=openai.")
    parser.add_argument("--cache_file", type=Path, default=Path("data/aligned/llm_api_cache.jsonl"), help="Cache file path.")
    parser.add_argument(
        "--prompt_version",
        type=str,
        default="v3_e2p_natural",
        help="Prompt/cache version: v3_e2p_natural=E2P三段自然语言前缀(默认); v2_due_sbr=历史JSON段内嵌(复现旧pilot).",
    )
    parser.add_argument("--max_retries", type=int, default=3, help="API retry count.")
    parser.add_argument("--retry_wait_seconds", type=float, default=1.5, help="Base retry wait in seconds.")
    parser.add_argument("--embedding_dim", type=int, default=16, help="Risk embedding dimension.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used by mock provider.")
    parser.add_argument("--limit_rows", type=int, default=0, help="Optional row limit for pilot run (0=all).")
    parser.add_argument(
        "--openai_base_url",
        type=str,
        default=os.environ.get("OPENAI_BASE_URL"),
        help=(
            "Override the OpenAI-compatible endpoint base URL (e.g. https://open.bigmodel.cn/api/paas/v4). "
            "Defaults to env OPENAI_BASE_URL; empty => official api.openai.com."
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.environ.get("PHASE5_CONCURRENCY", "1")),
        help=(
            "Concurrent worker threads for the openai provider (1 = sequential). "
            "Cache writes are mutex-protected. Default reads PHASE5_CONCURRENCY env."
        ),
    )
    parser.add_argument(
        "--qps_sleep",
        type=float,
        default=float(os.environ.get("PHASE5_QPS_SLEEP", "0")),
        help="Optional per-call sleep seconds (after each successful API call) to soften QPS.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.aligned_csv)
    out = build_llm_features(
        df=df,
        provider=args.provider,
        model=args.model,
        embedding_dim=args.embedding_dim,
        seed=args.seed,
        cache_file=args.cache_file,
        prompt_version=args.prompt_version,
        max_retries=args.max_retries,
        openai_base_url=args.openai_base_url,
        retry_wait_seconds=args.retry_wait_seconds,
        limit_rows=args.limit_rows,
        concurrency=args.concurrency,
        qps_sleep=args.qps_sleep,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    prov_path = args.output_csv.parent / f"{args.output_csv.stem}.provenance.json"
    try:
        generation_command = shlex.join(sys.argv)
    except (TypeError, ValueError):  # pragma: no cover
        generation_command = " ".join(sys.argv)
    provenance: Dict[str, object] = {
        "schema": "llm_features_build_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv.copy(),
        "generation_command": generation_command,
        "aligned_csv": str(args.aligned_csv.resolve()),
        "aligned_csv_sha256": _sha256_file(args.aligned_csv),
        "aligned_csv_stem": args.aligned_csv.stem,
        "aligned_csv_basename": args.aligned_csv.name,
        "output_csv": str(args.output_csv.resolve()),
        "output_csv_sha256": _sha256_file(args.output_csv),
        "row_count": int(len(out)),
        "provider": args.provider,
        "model": args.model,
        "prompt_version": args.prompt_version,
        "embedding_dim": int(args.embedding_dim),
        "seed": int(args.seed),
        "cache_file": str(Path(args.cache_file).resolve()),
        "limit_rows": int(args.limit_rows),
        "max_retries": int(args.max_retries),
        "retry_wait_seconds": float(args.retry_wait_seconds),
        "openai_base_url": args.openai_base_url or None,
        "concurrency": int(args.concurrency),
        "qps_sleep": float(args.qps_sleep),
    }
    prov_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] wrote llm features: {args.output_csv} ({len(out)} rows)")
    print(f"[OK] provider={args.provider} model={args.model}")
    print(f"[OK] cache_file={args.cache_file}")
    print(f"[OK] provenance: {prov_path}")


if __name__ == "__main__":
    main()
