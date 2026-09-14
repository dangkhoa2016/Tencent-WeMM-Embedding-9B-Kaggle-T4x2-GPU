# Contributing

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](CONTRIBUTING.vi.md)

Thank you for helping improve this project. Please keep changes focused, reproducible, and explicit about whether they affect runtime qualification.

## Development checks

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python -m pip install pytest packaging pillow -r requirements-api.txt -r requirements-demo.txt
python scripts/check_bilingual_docs.py --history
python scripts/check_publication_policy.py
python -m pytest -q
python -m compileall -q wemm_runtime wemm_kaggle scripts
bash -n kaggle/*.sh
git diff --check
```

Hardware qualification remains on Kaggle T4 ×2.

## Documentation

Every public Markdown document must have an English/Vietnamese counterpart and both files must change in the same commit.

## Qualification boundary

Do not silently alter frozen examples, the raw-cosine threshold, Qdrant corpus semantics, model weights, or dual-T4 placement. Changes to those boundaries require a new qualification phase and evidence.
