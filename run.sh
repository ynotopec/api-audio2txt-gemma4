#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Utilisation: source run.sh [IP] [PORT]"
  exit 1
fi

set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_BASENAME="$(basename "$PROJECT_DIR")"
VENV_DIR="${HOME}/venv/${PROJECT_BASENAME}"

IP="${1:-${HOST:-0.0.0.0}}"
PORT="${2:-${PORT:-8000}}"

cd "$PROJECT_DIR"

if [ ! -d "$VENV_DIR" ]; then
  echo "[run] venv absent: $VENV_DIR"
  echo "[run] lance d'abord: ./install.sh"
  return 1
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source ./.env
  set +a
fi

export HOST="${IP}"
export PORT="${PORT}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

uvicorn app:app --host "${HOST}" --port "${PORT}"
