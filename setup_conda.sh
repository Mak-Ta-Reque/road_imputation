#!/usr/bin/env bash
set -euo pipefail

# Usage: ./setup_conda.sh [env_name] [python_version]
# Example: ./setup_conda.sh road_imputation 3.10

ENV_NAME="${1:-road_imputation}"
PYTHON_VERSION="${2:-3.10}"
REQ1="requirements.txt"
REQ2="requirements_running.txt"

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda not found. Please install Miniconda or Anaconda and try again." >&2
  exit 1
fi

if command -v mamba >/dev/null 2>&1; then
  CREATE_CMD=(mamba create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y)
else
  CREATE_CMD=(conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y)
fi

echo "Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
"${CREATE_CMD[@]}"

echo "To activate the environment, run: conda activate $ENV_NAME"

echo "Installing pip requirements inside the environment..."

if [ -f "$REQ1" ]; then
  conda run -n "$ENV_NAME" python -m pip install --upgrade pip setuptools wheel
  conda run -n "$ENV_NAME" python -m pip install -r "$REQ1"
else
  echo "Warning: $REQ1 not found; skipping." >&2
fi

if [ -f "$REQ2" ]; then
  conda run -n "$ENV_NAME" python -m pip install -r "$REQ2"
else
  echo "Note: $REQ2 not found; if you need runtime-only requirements, add $REQ2 to the repo." >&2
fi

# Install a jupyter kernel for easy notebook usage
if command -v jupyter >/dev/null 2>&1 || conda run -n "$ENV_NAME" python -c "import importlib; importlib.import_module('ipykernel')" 2>/dev/null; then
  echo "Installing ipykernel for Jupyter..."
  conda run -n "$ENV_NAME" python -m ipykernel install --user --name="$ENV_NAME" --display-name "road_imputation ($ENV_NAME)"
fi

echo "Done. To start using the environment run: conda activate $ENV_NAME"
