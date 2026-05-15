#!/usr/bin/env bash
# Source (do not execute) this file from the project root:
#     source scripts/load_env.sh           # loads ./.env
#     source scripts/load_env.sh path/to/other.env
#
# Lines must be plain KEY=VALUE (shell syntax). Comment lines starting with '#' are ignored.
# This script:
#   1) Refuses to load if .env has world-readable permissions on POSIX systems.
#   2) Auto-exports every variable defined in the file via `set -a`.
#   3) Masks sensitive values when printing a summary.

ENV_FILE="${1-}"
if [ -z "$ENV_FILE" ] || [ ! -f "$ENV_FILE" ]; then
  ENV_FILE="$(pwd)/.env"
fi

if [ ! -f "$ENV_FILE" ]; then
  echo "[load_env] ERROR: '$ENV_FILE' not found. Copy .env.example to .env first." >&2
  return 1 2>/dev/null || exit 1
fi

# zsh's `.` builtin does not search the current directory; use the absolute path.
case "$ENV_FILE" in
  /*) ;;
  *) ENV_FILE="$(pwd)/$ENV_FILE" ;;
esac

# Permission check (best-effort; works on macOS / Linux).
if command -v stat >/dev/null 2>&1; then
  if stat -f "%Lp" "$ENV_FILE" >/dev/null 2>&1; then
    PERM="$(stat -f "%Lp" "$ENV_FILE")"
  else
    PERM="$(stat -c "%a" "$ENV_FILE" 2>/dev/null || echo "")"
  fi
  case "$PERM" in
    *4|*5|*6|*7)
      echo "[load_env] WARN: '$ENV_FILE' is world/group readable (mode=$PERM). Consider:  chmod 600 $ENV_FILE" >&2
      ;;
  esac
fi

set -a
# shellcheck disable=SC1090
. "$ENV_FILE"
set +a

if [ -n "${OPENAI_API_KEY:-}" ]; then
  MASK="${OPENAI_API_KEY:0:6}…${OPENAI_API_KEY: -4}"
  echo "[load_env] loaded $ENV_FILE  (OPENAI_API_KEY=$MASK, OPENAI_MODEL=${OPENAI_MODEL:-<unset>}, OPENAI_BASE_URL=${OPENAI_BASE_URL:-<official>})"
else
  echo "[load_env] loaded $ENV_FILE  (OPENAI_API_KEY=<empty>, edit $ENV_FILE and reload)"
fi
