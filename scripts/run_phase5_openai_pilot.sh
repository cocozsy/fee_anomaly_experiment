#!/usr/bin/env bash
# One-shot helper for the Phase 5 OpenAI pilot.
#
# Prereqs:
#   1)  Edit .env and paste your real OPENAI_API_KEY.
#   2)  source scripts/load_env.sh
#   3)  ./scripts/run_phase5_openai_pilot.sh [N]
#
# Defaults are read from .env (PHASE5_*); CLI arg N overrides PHASE5_LIMIT_ROWS.
#
# Outputs:
#   data/aligned/aligned_customer_month_llm_features_${PROMPT_VERSION}_pilot_openai_N.csv
#   data/aligned/llm_api_cache_pilot_openai.jsonl
#   (optional, if --train) checkpoints/phase5_pilot_openai_N_temporal/

set -euo pipefail

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "[pilot] ERROR: OPENAI_API_KEY not set. Run 'source scripts/load_env.sh' first." >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY_BIN="${PY_BIN:-.venv/bin/python}"
N="${1:-${PHASE5_LIMIT_ROWS:-300}}"
PROMPT_VERSION="${PHASE5_PROMPT_VERSION:-v2_due_sbr}"
EMB_DIM="${PHASE5_EMBEDDING_DIM:-16}"
ALIGNED_CSV="${PHASE5_ALIGNED_CSV:-data/aligned/aligned_customer_month_decoupled_env_full_d3c.csv}"
MODEL="${OPENAI_MODEL:-gpt-4o-mini}"

OUT_CSV="data/aligned/aligned_customer_month_llm_features_${PROMPT_VERSION}_pilot_openai_${N}.csv"
CACHE_FILE="data/aligned/llm_api_cache_pilot_openai.jsonl"

if [ ! -f "$ALIGNED_CSV" ]; then
  echo "[pilot] ERROR: aligned CSV not found: $ALIGNED_CSV" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT_CSV")"

EXTRA_ARGS=()
if [ -n "${OPENAI_BASE_URL:-}" ]; then
  EXTRA_ARGS+=(--openai_base_url "$OPENAI_BASE_URL")
fi

echo "[pilot] aligned_csv=$ALIGNED_CSV  N=$N  model=$MODEL  prompt=$PROMPT_VERSION  base_url=${OPENAI_BASE_URL:-<official OpenAI>}"
PYTHONPATH=code "$PY_BIN" code/build_llm_teacher_features.py \
  --aligned_csv  "$ALIGNED_CSV" \
  --output_csv   "$OUT_CSV" \
  --provider     openai \
  --model        "$MODEL" \
  --prompt_version "$PROMPT_VERSION" \
  --embedding_dim  "$EMB_DIM" \
  --limit_rows     "$N" \
  --cache_file     "$CACHE_FILE" \
  "${EXTRA_ARGS[@]}"

echo "[pilot] done. CSV: $OUT_CSV"
echo "[pilot] provenance: ${OUT_CSV%.csv}.provenance.json"
