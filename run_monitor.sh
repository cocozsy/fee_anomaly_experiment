#!/usr/bin/env bash
set -euo pipefail

# One-click launcher:
# 1) train_distill.py
# 2) metrics-based live progress plot
# 3) tensorboard
#
# Usage:
#   bash run_monitor.sh
#   bash run_monitor.sh --output-dir checkpoints/env_self_v1 --tensorboard-port 6006

OUTPUT_DIR="checkpoints/env_self_v1"
ALIGNED_CSV="data/aligned/aligned_customer_month_env.csv"
LLM_CSV="data/aligned/aligned_customer_month_llm_features.csv"
TEACHER_EPOCHS=12
STUDENT_EPOCHS=16
BATCH_SIZE=512
TENSORBOARD_PORT=6006
REFRESH_SECONDS=10

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --aligned-csv)
      ALIGNED_CSV="$2"
      shift 2
      ;;
    --llm-csv)
      LLM_CSV="$2"
      shift 2
      ;;
    --teacher-epochs)
      TEACHER_EPOCHS="$2"
      shift 2
      ;;
    --student-epochs)
      STUDENT_EPOCHS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --tensorboard-port)
      TENSORBOARD_PORT="$2"
      shift 2
      ;;
    --refresh-seconds)
      REFRESH_SECONDS="$2"
      shift 2
      ;;
    *)
      echo "[ERR] Unknown arg: $1"
      exit 1
      ;;
  esac
done

if [[ ! -x ".venv/bin/python" ]]; then
  echo "[ERR] .venv not found. Please create venv and install dependencies first."
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "experiments/figures" ".matplotlib_cache"

METRICS_FILE="$OUTPUT_DIR/metrics.jsonl"
LIVE_PLOT="experiments/figures/$(basename "$OUTPUT_DIR")_progress_live.png"
TRAIN_LOG="$OUTPUT_DIR/train_live.log"
MONITOR_LOG="$OUTPUT_DIR/monitor_live.log"
TB_LOG="$OUTPUT_DIR/tensorboard_live.log"

echo "[INFO] output_dir: $OUTPUT_DIR"
echo "[INFO] aligned_csv: $ALIGNED_CSV"
echo "[INFO] llm_csv: $LLM_CSV"
echo "[INFO] live_plot: $LIVE_PLOT"
echo "[INFO] tensorboard: http://localhost:$TENSORBOARD_PORT"

cleanup() {
  echo ""
  echo "[INFO] stopping background processes..."
  for pid in ${TRAIN_PID:-} ${MONITOR_PID:-} ${TB_PID:-}; do
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

# Start training
MPLCONFIGDIR=".matplotlib_cache" .venv/bin/python code/train_distill.py \
  --aligned_csv "$ALIGNED_CSV" \
  --llm_features_csv "$LLM_CSV" \
  --output_dir "$OUTPUT_DIR" \
  --teacher_epochs "$TEACHER_EPOCHS" \
  --student_epochs "$STUDENT_EPOCHS" \
  --batch_size "$BATCH_SIZE" >"$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!
echo "[INFO] training pid=$TRAIN_PID log=$TRAIN_LOG"

# Wait until metrics file exists
for _ in {1..120}; do
  if [[ -f "$METRICS_FILE" ]]; then
    break
  fi
  sleep 1
done

if [[ ! -f "$METRICS_FILE" ]]; then
  echo "[ERR] metrics file not created in time: $METRICS_FILE"
  exit 1
fi

# Start live plot monitor
MPLCONFIGDIR=".matplotlib_cache" .venv/bin/python code/monitor_training_progress.py \
  --metrics "$METRICS_FILE" \
  --output_png "$LIVE_PLOT" \
  --title "$(basename "$OUTPUT_DIR") live progress" \
  --watch \
  --refresh_seconds "$REFRESH_SECONDS" >"$MONITOR_LOG" 2>&1 &
MONITOR_PID=$!
echo "[INFO] monitor pid=$MONITOR_PID log=$MONITOR_LOG"

# Start tensorboard
.venv/bin/tensorboard --logdir "$OUTPUT_DIR/tb_logs" --port "$TENSORBOARD_PORT" >"$TB_LOG" 2>&1 &
TB_PID=$!
echo "[INFO] tensorboard pid=$TB_PID log=$TB_LOG"

echo "[INFO] running... press Ctrl+C to stop all"
echo "[INFO] quick check:"
echo "       tail -f \"$TRAIN_LOG\""
echo "       open \"$LIVE_PLOT\""

# Keep script alive while training is running
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  sleep 2
done

echo "[INFO] training finished. logs:"
echo "       $TRAIN_LOG"
echo "       $MONITOR_LOG"
echo "       $TB_LOG"
