#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/kaggle/runtime-env.sh"

VENV_DIR="${VENV_DIR:-/opt/wemm-embedding-venv}"
RUN_DIR="${WEMM_SEARCH_RUN_DIR:-/kaggle/working/wemm-v030/run}"
TOKEN_ENV="${WEMM_TOKEN_ENV:-/kaggle/working/wemm-v030/token.env}"
EVIDENCE_DIR="${WEMM_SEARCH_EVIDENCE_DIR:-/kaggle/working/wemm-v030/evidence}"
mkdir -p "$RUN_DIR" "$EVIDENCE_DIR"

# Forward the frozen API token to the search gateway.
if [[ -f "$TOKEN_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$TOKEN_ENV"
fi
export WEMM_API_TOKEN WEMM_EMBEDDING_URL="${WEMM_EMBEDDING_URL:-http://127.0.0.1:8090}" \
  WEMM_QDRANT_HOST="127.0.0.1" WEMM_QDRANT_PORT="${WEMM_QDRANT_PORT:-6333}" \
  WEMM_SEARCH_HOST="127.0.0.1" WEMM_SEARCH_PORT="${WEMM_SEARCH_PORT:-8091}"

"$VENV_DIR/bin/python" "$ROOT/scripts/serve_search.py" \
  > "$RUN_DIR/search-gateway.log" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$RUN_DIR/search.pid"

TIMEOUT=120
for _ in $(seq 1 "$TIMEOUT"); do
  if curl -fsS "http://127.0.0.1:8091/readyz" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "WEMM_SEARCH_FAIL: gateway exited early" >&2
    cat "$RUN_DIR/search-gateway.log" >&2 || true
    exit 1
  fi
  sleep 1
done

curl -fsS "http://127.0.0.1:8091/readyz" > "$EVIDENCE_DIR/search-readyz.json"
echo "WEMM_SEARCH_START_PASS pid=$SERVER_PID port=8091"
