#!/usr/bin/env bash
set -Eeuo pipefail

NVIDIA_LIB_DIR=/usr/local/nvidia/lib64
NVIDIA_SMI_DIR=/opt/bin

if [[ -e "$NVIDIA_LIB_DIR/libcuda.so.1" ]]; then
  case ":${LD_LIBRARY_PATH:-}:" in
    *":$NVIDIA_LIB_DIR:"*) ;;
    *)
      export LD_LIBRARY_PATH="$NVIDIA_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
      ;;
  esac
fi

if ! command -v nvidia-smi >/dev/null 2>&1 \
   && [[ -x "$NVIDIA_SMI_DIR/nvidia-smi" ]]; then
  export PATH="$NVIDIA_SMI_DIR:$PATH"
fi