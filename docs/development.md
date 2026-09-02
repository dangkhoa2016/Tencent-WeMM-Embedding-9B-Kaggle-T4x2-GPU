# Development

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](development.vi.md)

This document is for contributors working on the repository source.

## Repository boundaries

Core reusable code:

```text
wemm_runtime/
```

Public-notebook presentation and Kaggle integration:

```text
wemm_notebook/
wemm_kaggle/
kaggle/
notebooks/
```

Repository validation:

```text
tests/
scripts/
.github/workflows/
```

## Local CPU checks

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python -m pip install pytest packaging pillow -r requirements-api.txt -r requirements-demo.txt -r requirements-search.txt

python scripts/check_bilingual_docs.py --history
python scripts/check_doc_links.py
python scripts/check_publication_policy.py
python -m pytest -q
python -m compileall -q wemm_runtime wemm_kaggle wemm_notebook scripts
bash -n kaggle/*.sh
git diff --check
```

## Bilingual documentation policy

Every public Markdown document must have an English/Vietnamese counterpart.

Example:

```text
docs/api.md
docs/api.vi.md
```

Both files must change in the same commit. CI enforces the pair and exact language-navigation header.

## Documentation link policy

Relative Markdown links must resolve to files that exist in the repository. External HTTP(S), mailto, and fragment-only links are outside the local target check.

## Qualification-sensitive changes

Do not treat changes to model/runtime science as ordinary documentation or maintenance.

A new qualification phase is required for changes to:

- model selection/weights;
- precision;
- GPU placement;
- Qdrant corpus semantics;
- public retrieval examples;
- visual transforms;
- acceptance threshold;
- scorecard methodology.

## Kaggle system Torch policy

`requirements-kaggle.txt` intentionally does not own PyTorch/CUDA. The project venv reuses the Kaggle system runtime and setup guards verify that identity before and after dependency installation.

## Pull requests

Before opening a PR:

1. keep scope narrow;
2. run local checks;
3. update both language variants of public docs;
4. state whether qualification boundaries are affected;
5. include sanitized evidence/logs when relevant.

See [CONTRIBUTING.md](../.github/CONTRIBUTING.md).
