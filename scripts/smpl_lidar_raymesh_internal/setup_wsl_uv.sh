#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1090
  source "$HOME/.local/bin/env"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv was installed but is not on PATH. Run: source ~/.local/bin/env" >&2
  exit 1
fi

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"

uv venv --python "$PYTHON_VERSION" .venv
# shellcheck disable=SC1091
source .venv/bin/activate

uv pip install torch --index-url "$TORCH_INDEX_URL"
uv pip install pip setuptools
uv pip install -r requirements.txt --no-build-isolation
uv pip install -e human2humanoid/phc

python - <<'PY'
import numpy as np
import torch
import smplx
from phc.utils.pc_anomaly import SmplToPointCloud

print("Environment OK")
print("numpy", np.__version__)
print("torch", torch.__version__)
print("smplx", getattr(smplx, "__version__", "ok"))
print("SmplToPointCloud", SmplToPointCloud.__name__)
PY

echo
echo "Setup complete."
echo "Activate with: source .venv/bin/activate"
