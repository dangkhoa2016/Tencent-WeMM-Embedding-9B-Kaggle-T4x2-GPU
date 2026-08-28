#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/kaggle/runtime-env.sh"
VENV_DIR="${VENV_DIR:-/opt/wemm-embedding-venv}"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
python "$ROOT/scripts/preflight.py" "$@"
