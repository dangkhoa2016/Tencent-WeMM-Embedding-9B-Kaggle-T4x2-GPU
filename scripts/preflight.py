#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wemm_kaggle.gpu import cuda_preflight
from wemm_kaggle.model_resolver import resolve_model_dir, validate_model_dir
from wemm_kaggle.offline import enforce_offline


def package_versions() -> dict[str, str]:
    names = ("transformers", "accelerate", "qwen-vl-utils", "safetensors", "Pillow")
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "MISSING"
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-non-t4", action="store_true")
    args = parser.parse_args()
    enforce_offline()

    versions = package_versions()
    if versions["transformers"] != "5.2.0":
        raise RuntimeError(f"transformers==5.2.0 required; found {versions['transformers']}")
    if versions["qwen-vl-utils"] != "0.0.14":
        raise RuntimeError(f"qwen-vl-utils==0.0.14 required; found {versions['qwen-vl-utils']}")

    gpu = cuda_preflight(2)
    names = [row["name"] for row in gpu["devices"][:2]]
    allow_non_t4 = args.allow_non_t4 or os.environ.get("ALLOW_NON_T4") == "1"
    if not allow_non_t4 and not all("T4" in name.upper() for name in names):
        raise RuntimeError(f"Expected Kaggle T4 x2; observed {names}. Set ALLOW_NON_T4=1 only for deliberate testing.")

    model_dir = resolve_model_dir(Path("/kaggle/input"), os.environ.get("KAGGLE_MODEL_DIR"))
    identity = validate_model_dir(model_dir)
    payload = {
        "status": "PASS",
        "model_source": "kaggle_input",
        "model": identity.to_dict(),
        "packages": versions,
        "gpu": gpu,
        "offline": {
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
            "HF_DATASETS_OFFLINE": os.environ.get("HF_DATASETS_OFFLINE"),
        },
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
