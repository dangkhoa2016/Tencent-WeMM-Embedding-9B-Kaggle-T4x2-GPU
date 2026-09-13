#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/kaggle/runtime-env.sh"

QDRANT_VERSION="1.19.0"
QDRANT_DIR="/kaggle/working/wemm-qdrant"
BIN_PATH="$QDRANT_DIR/bin/qdrant"
CONFIG_PATH="$QDRANT_DIR/run/config.yaml"
RUN_DIR="$QDRANT_DIR/run"
LOG_DIR="$QDRANT_DIR/logs"
EVIDENCE_DIR="${WEMM_QDRANT_EVIDENCE_DIR:-/kaggle/working/wemm-v030/evidence}"
PORT="${WEMM_QDRANT_PORT:-6333}"

mkdir -p "$RUN_DIR" "$LOG_DIR" "$EVIDENCE_DIR"

# Verify exact binary version.
"$BIN_PATH" --version 2>&1 | grep -q "$QDRANT_VERSION" \
  || { echo "WEMM_QDRANT_FAIL: unexpected qdrant version" >&2; exit 1; }

cat > "$CONFIG_PATH" <<YAML
log_level: INFO

storage:
  storage_path: $QDRANT_DIR/storage
  snapshots_path: $QDRANT_DIR/snapshots

service:
  host: 127.0.0.1
  http_port: $PORT
  grpc_port: null
  enable_cors: false
  enable_tls: false
YAML

# Refuse if port already owned.
if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
  echo "WEMM_QDRANT_FAIL: port $PORT already in use" >&2
  exit 1
fi

"$BIN_PATH" --config-path "$CONFIG_PATH" \
  > "$LOG_DIR/qdrant.log" 2>&1 &
QDRANT_PID=$!
echo "$QDRANT_PID" > "$RUN_DIR/qdrant.pid"

TIMEOUT=120
for _ in $(seq 1 "$TIMEOUT"); do
  if curl -fsS "http://127.0.0.1:$PORT/readyz" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$QDRANT_PID" 2>/dev/null; then
    echo "WEMM_QDRANT_FAIL: qdrant process exited early" >&2
    cat "$LOG_DIR/qdrant.log" >&2 || true
    exit 1
  fi
  sleep 1
done

if ! curl -fsS "http://127.0.0.1:$PORT/readyz" >/dev/null 2>&1; then
  echo "WEMM_QDRANT_FAIL: readiness timeout" >&2
  exit 1
fi

curl -fsS "http://127.0.0.1:$PORT/readyz" | head -c 200 > "$EVIDENCE_DIR/qdrant-readyz.txt"
curl -fsS "http://127.0.0.1:$PORT/" > "$EVIDENCE_DIR/qdrant-root.json" 2>/dev/null || true
"$BIN_PATH" --version > "$EVIDENCE_DIR/qdrant-version.txt" 2>&1

ps -fp "$QDRANT_PID" > "$EVIDENCE_DIR/qdrant-process.txt" 2>&1 || true
ss -ltnp 2>/dev/null | grep ":$PORT " > "$EVIDENCE_DIR/qdrant-listener.txt" || \
  { echo "WEMM_QDRANT_FAIL: no listener on $PORT" >&2; exit 1; }

if grep -q "0.0.0.0:$PORT" "$EVIDENCE_DIR/qdrant-listener.txt"; then
  echo "WEMM_QDRANT_FAIL: listener bound to 0.0.0.0" >&2
  exit 1
fi

cat > "$EVIDENCE_DIR/qdrant-startup.json" <<JSON
{
  "pid": $QDRANT_PID,
  "version": "$QDRANT_VERSION",
  "config": "$CONFIG_PATH",
  "listener": "$(cat "$EVIDENCE_DIR/qdrant-listener.txt")",
  "status": "PASS"
}
JSON

echo "WEMM_QDRANT_START_PASS pid=$QDRANT_PID port=$PORT"
