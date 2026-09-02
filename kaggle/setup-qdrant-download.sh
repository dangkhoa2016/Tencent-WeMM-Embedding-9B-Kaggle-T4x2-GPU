#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/kaggle/runtime-env.sh"

QDRANT_VERSION="1.19.0"
MUSL_ARCHIVE="qdrant-x86_64-unknown-linux-musl.tar.gz"
MUSL_URL="https://github.com/qdrant/qdrant/releases/download/v1.19.0/qdrant-x86_64-unknown-linux-musl.tar.gz"
MUSL_SHA256="9ec667456443463eee390e43cd36988af6b730c6db807b4e39f57c303d0264a3"
GNU_URL="https://github.com/qdrant/qdrant/releases/download/v1.19.0/qdrant-x86_64-unknown-linux-gnu.tar.gz"
GNU_SHA256="e4405091f67d02f96fb941695ef8a6974e677632507ff7b04a3fcbb332ad9c19"

QDRANT_DIR="/kaggle/working/wemm-qdrant"
EVIDENCE_DIR="${WEMM_QDRANT_EVIDENCE_DIR:-/kaggle/working/wemm-v030/evidence}"
mkdir -p "$QDRANT_DIR/downloads" "$QDRANT_DIR/bin" "$EVIDENCE_DIR"

ARCHIVE_PATH="$QDRANT_DIR/downloads/$MUSL_ARCHIVE"
if [[ ! -f "$ARCHIVE_PATH" ]]; then
  curl -fL --retry 3 -o "$ARCHIVE_PATH" "$MUSL_URL"
fi
echo "$MUSL_SHA256  $ARCHIVE_PATH" > "$ARCHIVE_PATH.sha256"
(cd "$QDRANT_DIR/downloads" && sha256sum -c "$(basename "$ARCHIVE_PATH").sha256")

BIN_PATH="$QDRANT_DIR/bin/qdrant"
if [[ ! -x "$BIN_PATH" ]] || ! "$BIN_PATH" --version 2>/dev/null | grep -q "$QDRANT_VERSION"; then
  mkdir -p "$QDRANT_DIR/downloads/extract"
  rm -rf "$QDRANT_DIR/downloads/extract"/*
  tar -xzf "$ARCHIVE_PATH" -C "$QDRANT_DIR/downloads/extract"
  find "$QDRANT_DIR/downloads/extract" -type f -name qdrant -exec cp {} "$BIN_PATH" \;
  chmod +x "$BIN_PATH"
  rm -rf "$QDRANT_DIR/downloads/extract"
fi

"$BIN_PATH" --version > "$QDRANT_DIR/downloads/qdrant-version.txt" 2>&1
BIN_SHA256="$(sha256sum "$BIN_PATH" | awk '{print $1}')"
QDRANT_VERSION_OUT="$(cat "$QDRANT_DIR/downloads/qdrant-version.txt")"

cat > "$EVIDENCE_DIR/qdrant-setup.json" <<JSON
{
  "qdrant_version": "$QDRANT_VERSION",
  "deviation_url": "$MUSL_URL",
  "deviation_original_url": "$GNU_URL",
  "deviation_original_sha256_expected": "$GNU_SHA256",
  "deviation_sha256": "$MUSL_SHA256",
  "deviation_rationale": "Official GNU Qdrant 1.19.0 build requires glibc >= 2.38 but this Kaggle host is glibc 2.35; the official musl static build is used instead and verified by SHA256.",
  "archive_url": "$MUSL_URL",
  "archive_sha256": "$MUSL_SHA256",
  "binary_sha256": "$BIN_SHA256",
  "binary_version": "$QDRANT_VERSION_OUT",
  "status": "PASS"
}
JSON

echo "WEMM_SEARCH_SETUP_PASS"
