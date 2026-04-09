#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_BASENAME="$(basename "$PROJECT_DIR")"
VENV_DIR="${HOME}/venv/${PROJECT_BASENAME}"

cd "$PROJECT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "[install] uv introuvable"
  echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

mkdir -p "$(dirname "$VENV_DIR")"

if [ ! -d "$VENV_DIR" ]; then
  echo "[install] création du venv: $VENV_DIR"
  uv venv "$VENV_DIR"
else
  echo "[install] venv déjà présent: $VENV_DIR"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "[install] upgrade outils de base"
uv pip install -U pip setuptools wheel

# Torch GPU:
# - PyTorch stable affiche encore cu126/cu128 sur la page d'install.
# - Côté dev PyTorch, CUDA 12.8 est en cours de retrait et CUDA 13.0 reste la variante stable publiée.
# On tente donc dans cet ordre: cu130 -> cu128 -> PyPI standard.
# Cela évite de casser DGX Spark / H100 selon la machine et l'archi.
install_torch() {
  local ok=0

  if uv pip install -U \
      --index-url https://download.pytorch.org/whl/cu130 \
      torch torchvision torchaudio; then
    ok=1
  elif uv pip install -U \
      --index-url https://download.pytorch.org/whl/cu128 \
      torch torchvision torchaudio; then
    ok=1
  elif uv pip install -U torch torchvision torchaudio; then
    ok=1
  fi

  [ "$ok" -eq 1 ]
}

echo "[install] installation PyTorch"
install_torch || {
  echo "[install] échec installation torch"
  exit 1
}

echo "[install] installation dépendances Python"
uv pip install -U -r requirements.txt

if [ ! -f .env ] && [ -f .env.example ]; then
  echo "[install] création .env depuis .env.example"
  cp .env.example .env
fi

echo "[install] smoke test"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device_count:", torch.cuda.device_count())
    print("device_0:", torch.cuda.get_device_name(0))
PY

echo "[install] OK"
