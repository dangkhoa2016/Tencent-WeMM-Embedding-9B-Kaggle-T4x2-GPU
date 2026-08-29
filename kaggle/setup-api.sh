#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/kaggle/runtime-env.sh"
VENV_DIR="${VENV_DIR:-/opt/wemm-embedding-venv}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GUARD="$ROOT/scripts/setup_guard.py"
REQ="$ROOT/requirements-api.txt"
EVIDENCE_DIR="${WEMM_API_SETUP_EVIDENCE_DIR:-/kaggle/working/wemm-embedding-9b-t4x2-api-setup}"
mkdir -p "$EVIDENCE_DIR"

[[ -x "$VENV_DIR/bin/python" ]] || { echo "WEMM_API_SETUP_FAIL: base venv missing" >&2; exit 1; }

if grep -Eiv '^[[:space:]]*(#|$)' "$REQ" \
  | grep -Ei '(^|[[:space:]])(torch|triton|nvidia[-_]|cuda[-_])' >/dev/null; then
  echo "WEMM_API_SETUP_FAIL: forbidden Torch/CUDA/NVIDIA dependency" >&2
  exit 1
fi

"$VENV_DIR/bin/python" "$GUARD" capture --require-cuda --min-gpus 2 \
  --output "$EVIDENCE_DIR/torch-before-api-install.json"

"$PYTHON_BIN" -m pip --python "$VENV_DIR/bin/python" install \
  --disable-pip-version-check --no-cache-dir -r "$REQ"

"$VENV_DIR/bin/python" "$GUARD" capture --require-cuda --min-gpus 2 \
  --output "$EVIDENCE_DIR/torch-after-api-install.json"

"$PYTHON_BIN" "$GUARD" verify-reuse \
  --expected "$EVIDENCE_DIR/torch-before-api-install.json" \
  --actual "$EVIDENCE_DIR/torch-after-api-install.json" \
  --venv-dir "$VENV_DIR" \
  --output "$EVIDENCE_DIR/torch-reuse-api-install.json"

"$VENV_DIR/bin/python" "$GUARD" scan-local \
  --output "$EVIDENCE_DIR/venv-local-forbidden-api-scan.json"

"$VENV_DIR/bin/python" - <<'PY'
import importlib.metadata as md
expected = {
    "fastapi": "0.141.1",
    "uvicorn": "0.52.4",
    "python-multipart": "0.0.32",
    "httpx": "0.28.1",
}
for name, want in expected.items():
    got = md.version(name)
    if got != want:
        raise SystemExit(f"{name}: expected {want}, got {got}")
print("WEMM_API_SETUP_PASS")
PY