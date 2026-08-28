#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/kaggle/runtime-env.sh"
VENV_DIR="${VENV_DIR:-/opt/wemm-embedding-venv}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EVIDENCE_ROOT="${WEMM_SETUP_EVIDENCE_ROOT:-/kaggle/working/wemm-embedding-9b-t4x2-setup-runs}"
SETUP_RUN_ID="${WEMM_SETUP_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
EVIDENCE_DIR="${WEMM_SETUP_EVIDENCE_DIR:-$EVIDENCE_ROOT/$SETUP_RUN_ID}"
GUARD="$ROOT/scripts/setup_guard.py"
REQ="$ROOT/requirements-kaggle.txt"

mkdir -p "$EVIDENCE_DIR"
SYSTEM_ID="$EVIDENCE_DIR/system-torch.json"
VENV_BEFORE_ID="$EVIDENCE_DIR/venv-torch-before-install.json"
VENV_AFTER_ID="$EVIDENCE_DIR/venv-torch-after-install.json"
VERIFY_BEFORE="$EVIDENCE_DIR/torch-reuse-before-install.json"
VERIFY_AFTER="$EVIDENCE_DIR/torch-reuse-after-install.json"
LOCAL_SCAN="$EVIDENCE_DIR/venv-local-forbidden-scan.json"
PYVENV_COPY="$EVIDENCE_DIR/pyvenv.cfg"

fail() {
  echo "WEMM_SETUP_FAIL: $*" >&2
  exit 1
}

[[ -f "$GUARD" ]] || fail "missing $GUARD"
[[ -f "$REQ" ]] || fail "missing $REQ"
case "$VENV_DIR" in
  /|/opt|/kaggle|/kaggle/working) fail "refusing unsafe VENV_DIR=$VENV_DIR" ;;
esac
echo "WEMM_SETUP_RUN_ID=$SETUP_RUN_ID"
echo "WEMM_SETUP_EVIDENCE_DIR=$EVIDENCE_DIR"

# Reject explicit forbidden runtime specs before touching the environment.
if grep -Eiv '^[[:space:]]*(#|$)' "$REQ" \
  | grep -Ei '(^|[[:space:]])(torch|triton|nvidia[-_]|cuda[-_])' >/dev/null; then
  fail "requirements-kaggle.txt contains a forbidden Torch/CUDA/NVIDIA runtime spec"
fi

ACCELERATE_SPEC="$(grep -E '^[[:space:]]*accelerate==' "$REQ" | tr -d '[:space:]')"
[[ "$ACCELERATE_SPEC" == "accelerate==1.14.0" ]] \
  || fail "accelerate must be pinned exactly to accelerate==1.14.0 for the qualified Kaggle runtime"

# 1) Capture the Kaggle system Torch identity before venv creation.
"$PYTHON_BIN" "$GUARD" capture \
  --require-cuda --min-gpus 2 --output "$SYSTEM_ID"

# 2) Recreate the project-dedicated venv deterministically.
# --without-pip avoids Kaggle's observed ensurepip failure.
rm -rf "$VENV_DIR"
"$PYTHON_BIN" -m venv --without-pip --system-site-packages "$VENV_DIR"

[[ -x "$VENV_DIR/bin/python" ]] || fail "venv python was not created"
[[ -f "$VENV_DIR/pyvenv.cfg" ]] || fail "pyvenv.cfg missing"
grep -Eq '^include-system-site-packages[[:space:]]*=[[:space:]]*true$' "$VENV_DIR/pyvenv.cfg" \
  || fail "venv does not expose system site packages"
cp "$VENV_DIR/pyvenv.cfg" "$PYVENV_COPY"

# 3) Prove system Torch is visible from the venv BEFORE pip can install anything.
"$VENV_DIR/bin/python" "$GUARD" capture \
  --require-cuda --min-gpus 2 --output "$VENV_BEFORE_ID"
"$PYTHON_BIN" "$GUARD" verify-reuse \
  --expected "$SYSTEM_ID" \
  --actual "$VENV_BEFORE_ID" \
  --venv-dir "$VENV_DIR" \
  --output "$VERIFY_BEFORE"

# The venv has no private pip bootstrap, but system-site-packages should expose Kaggle's pip.
"$VENV_DIR/bin/python" -m pip --version \
  || fail "Kaggle system pip is not visible from the system-site-packages venv"

# 4) Use system pip to manage the target venv. No ensurepip/bootstrap is needed.
"$PYTHON_BIN" -m pip --version
"$PYTHON_BIN" -m pip --python "$VENV_DIR/bin/python" --version

SAFE_REQ="$(mktemp)"
cleanup() { rm -f "$SAFE_REQ"; }
trap cleanup EXIT
awk '!/^[[:space:]]*accelerate==/' "$REQ" > "$SAFE_REQ"

# transformers base + qwen-vl-utils do not require torch. Install their deps normally.
"$PYTHON_BIN" -m pip --python "$VENV_DIR/bin/python" install \
  --disable-pip-version-check --no-cache-dir -r "$SAFE_REQ"

# accelerate declares torch as a mandatory dependency. Install only accelerate itself so pip
# cannot resolve/download/install a second Torch/CUDA/NVIDIA stack.
"$PYTHON_BIN" -m pip --python "$VENV_DIR/bin/python" install \
  --disable-pip-version-check --no-cache-dir --no-deps "$ACCELERATE_SPEC"

# 5) Verify Accelerate's declared runtime requirements against the target interpreter.
# Avoid a global `pip check`: Kaggle base images can contain unrelated package conflicts.
"$VENV_DIR/bin/python" "$GUARD" check-requires \
  --distribution accelerate --output "$EVIDENCE_DIR/accelerate-requirements.json"

# 6) Prove Torch identity remained unchanged after all installs.
"$VENV_DIR/bin/python" "$GUARD" capture \
  --require-cuda --min-gpus 2 --output "$VENV_AFTER_ID"
"$PYTHON_BIN" "$GUARD" verify-reuse \
  --expected "$SYSTEM_ID" \
  --actual "$VENV_AFTER_ID" \
  --venv-dir "$VENV_DIR" \
  --output "$VERIFY_AFTER"

# 7) Fail if any forbidden runtime distribution was installed into the venv's own purelib.
"$VENV_DIR/bin/python" "$GUARD" scan-local --output "$LOCAL_SCAN"

# 8) Exact runtime version gate for the already-proven software stack.
"$VENV_DIR/bin/python" - <<'PY'
import importlib.metadata as md

expected = {
    "transformers": "5.2.0",
    "accelerate": "1.14.0",
    "qwen-vl-utils": "0.0.14",
}
for name, want in expected.items():
    got = md.version(name)
    if got != want:
        raise SystemExit(f"{name}: expected {want}, got {got}")

for name in ("torch", "transformers", "accelerate", "qwen-vl-utils", "safetensors", "Pillow"):
    try:
        print(f"{name}={md.version(name)}")
    except md.PackageNotFoundError:
        print(f"{name}=MISSING")
        raise

print("WEMM_SETUP_PASS")
PY
