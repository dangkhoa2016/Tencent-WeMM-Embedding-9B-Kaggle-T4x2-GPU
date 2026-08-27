from __future__ import annotations

import os
from pathlib import Path

KAGGLE_INPUT_ROOT = Path("/kaggle/input")


def enforce_offline() -> None:
    """Force Hugging Face/data libraries into offline mode for model execution."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def assert_kaggle_input_path(path: Path) -> Path:
    """Return a resolved path only when it lives under the Kaggle input root."""
    resolved = Path(path).expanduser().resolve()
    root = KAGGLE_INPUT_ROOT.expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Model path must be under /kaggle/input: {resolved}") from exc
    return resolved
