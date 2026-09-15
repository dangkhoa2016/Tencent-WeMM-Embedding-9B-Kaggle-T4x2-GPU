"""Structural guards for the modular sectioned public Kaggle notebook."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "kaggle-production-demo-thin.ipynb"
MODULE_DIR = ROOT / "wemm_notebook"


def _notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_public_notebook_has_five_short_sectioned_code_cells():
    nb = _notebook()
    code_cells = [cell for cell in nb["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) == 5

    sources = ["".join(cell["source"]) for cell in code_cells]
    assert all(len(source.splitlines()) <= 40 for source in sources)
    assert 'PUBLIC_RELEASE_REF = "v1.0.0"' in sources[0]
    assert "demo = start_public_session()" in sources[0]
    assert sources[1].strip() == "text_results = run_step6(demo)"
    assert sources[2].strip() == "image_results = run_step7a(demo)"
    assert sources[3].strip() == "visual_results = run_step7b(demo)"
    assert sources[4].strip() == "final_summary = run_step8(demo, visual_results)"

    for cell in code_cells:
        assert cell["outputs"] == []
        assert cell["execution_count"] is None


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


def test_runner_exposes_sectioned_phase_order_and_fail_closed_cleanup():
    source = (MODULE_DIR / "runner.py").read_text(encoding="utf-8")
    order = [
        "def start_public_session():",
        "def run_step6(demo):",
        "def run_step7a(demo):",
        "def run_step7b(demo):",
        "def run_step8(demo, visual_results=None):",
    ]
    offsets = [source.index(marker) for marker in order]
    assert offsets == sorted(offsets)

    assert '_require_phase(demo, "setup")' in source
    assert '_require_phase(demo, "step6")' in source
    assert '_require_phase(demo, "step7a")' in source
    assert '_require_phase(demo, "step7b")' in source
    assert "SECTIONED_STEP6_FAILURE_ABORT" in source
    assert "SECTIONED_STEP7A_FAILURE_ABORT" in source
    assert "SECTIONED_STEP7B_FAILURE_ABORT" in source
    assert "SECTIONED_STEP8_FAILURE_ABORT" in source
    assert "STEP_7A_7B_SEPARATE_CELLS=PASS" in source
    assert "NOTEBOOK_EXECUTABLE_CELLS=5" in source


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


def test_visual_showcase_defers_frozen_runtime_imports():
    source = (MODULE_DIR / "visual_showcase.py").read_text(encoding="utf-8")
    module = ast.parse(source, filename="visual_showcase.py")
    top_level_imports = []
    for node in module.body:
        if isinstance(node, ast.Import):
            top_level_imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            top_level_imports.append(node.module or "")

    assert not any(name.startswith("wemm_kaggle") for name in top_level_imports)
    assert "qdrant_client" not in top_level_imports
    assert "from wemm_kaggle.demo_config import EVID, RUN_ROOT" in source


def test_public_notebook_pairs_five_markdown_sections_with_five_code_cells():
    nb = _notebook()
    assert len(nb["cells"]) == 10
    assert [cell["cell_type"] for cell in nb["cells"]] == [
        "markdown", "code",
        "markdown", "code",
        "markdown", "code",
        "markdown", "code",
        "markdown", "code",
    ]

    markdown_cells = [cell for cell in nb["cells"] if cell["cell_type"] == "markdown"]
    rendered = ["".join(cell["source"]) for cell in markdown_cells]
    expected_headings = [
        "# Tencent WeMM-Embedding-9B + Qdrant — Kaggle T4×2 Production Demo",
        "## Step 6/8 — Truy xuất văn bản song ngữ / Bilingual text retrieval",
        "## Step 7A/8 — Truy xuất semantic ảnh→văn bản / Semantic image→text retrieval",
        "## Step 7B/8 — Độ bền truy xuất hình ảnh / Visual robustness retrieval",
        "## Step 8/8 — Đóng phiên + nghiệm thu / Closeout + acceptance",
    ]
    assert [text.splitlines()[0] for text in rendered] == expected_headings

    metadata = nb["metadata"]["wemm_public_demo"]
    assert metadata["presentation_markdown_cells"] == 5
    assert metadata["executable_code_cells"] == 5
    assert metadata["sectioned_five_code_cell_runner"] is True
    assert metadata["atomic_one_code_cell_runner"] is False
    assert metadata["step_7a_7b_separate_code_cells"] is True


def test_step7a_and_step7b_modules_do_not_duplicate_notebook_headings():
    image = (MODULE_DIR / "image_showcase.py").read_text(encoding="utf-8")
    visual = (MODULE_DIR / "visual_showcase.py").read_text(encoding="utf-8")
    closeout = (MODULE_DIR / "closeout.py").read_text(encoding="utf-8")

    assert "## Step 7A/8" not in image
    assert "## Step 7B/8" not in visual
    assert "## Step 8/8" not in closeout
