"""Structural guards for the modular one-cell public Kaggle notebook."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "kaggle-production-demo-thin.ipynb"
MODULE_DIR = ROOT / "wemm_notebook"


def _notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_public_notebook_is_one_short_atomic_code_cell():
    nb = _notebook()
    code_cells = [cell for cell in nb["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) == 1

    source = "".join(code_cells[0]["source"])
    assert len(source.splitlines()) <= 50
    assert "run_public_notebook()" in source
    assert 'PUBLIC_RELEASE_REF = "v1.0.0"' in source
    assert code_cells[0]["outputs"] == []
    assert code_cells[0]["execution_count"] is None


def test_modular_phase_files_exist_and_parse():
    expected = {
        "__init__.py",
        "bootstrap.py",
        "runner.py",
        "text_showcase.py",
        "image_showcase.py",
        "visual_showcase.py",
        "closeout.py",
    }
    assert expected.issubset({path.name for path in MODULE_DIR.glob("*.py")})

    for name in expected:
        source = (MODULE_DIR / name).read_text(encoding="utf-8")
        ast.parse(source, filename=name)


def test_runner_preserves_atomic_phase_order_and_fail_closed_cleanup():
    source = (MODULE_DIR / "runner.py").read_text(encoding="utf-8")
    order = [
        "bootstrap_runtime()",
        "start_public_demo()",
        "run_text_showcase(demo)",
        "run_image_showcase(demo)",
        "run_visual_showcase(demo)",
        "run_closeout(demo, visual_results)",
    ]
    offsets = [source.index(marker) for marker in order]
    assert offsets == sorted(offsets)
    assert "except BaseException:" in source
    assert "demo.abort()" in source
    assert "ATOMIC_FAILURE_DEMO_ABORT=PASS" in source


def test_text_showcase_is_full_text_and_separates_winner_competitor():
    source = (MODULE_DIR / "text_showcase.py").read_text(encoding="utf-8")
    assert "_clip(" not in source
    assert "TOP-1 WINNER / KẾT QUẢ #1" in source
    assert "NEAREST COMPETITOR / ĐỐI THỦ GẦN NHẤT" in source
    assert "full embedded text" in source
    assert "Full candidate text" in source
    assert "STEP_6_PRESENTATION_LAYER=HUMAN_FIRST_FULL_TEXT" in source
    assert "confidence percentage" in source


def test_frozen_runtime_and_visual_contracts_are_not_reopened():
    bootstrap = (MODULE_DIR / "bootstrap.py").read_text(encoding="utf-8")
    visual = (MODULE_DIR / "visual_showcase.py").read_text(encoding="utf-8")
    closeout = (MODULE_DIR / "closeout.py").read_text(encoding="utf-8")

    assert "d04bcd3e601b449b67d09ff1132cab965619d858" in bootstrap
    assert "VISUAL_THRESHOLD = 0.90" in visual
    for qid in ("Q19217", "Q10489198", "Q168751", "Q51756"):
        assert qid in visual
    for transform in (
        "resize_80pct",
        "jpeg_q90",
        "center_crop_96pct",
        "brightness_103pct",
    ):
        assert transform in visual
    assert "SEMANTIC_CORPUS_RETRIEVAL_PATHS_TOP1=36/36" in closeout
    assert "VISUAL_ROBUSTNESS_RETRIEVAL_PATHS_TOP1=32/32" in closeout
    assert "PUBLIC_DEMO_TOTAL_EXECUTED_RETRIEVAL_CHECKS=68/68" in closeout
