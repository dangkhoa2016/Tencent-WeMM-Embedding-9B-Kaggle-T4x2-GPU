# Development

> 🌐 Language / Ngôn ngữ: [English](development.md) | **Tiếng Việt**

Tài liệu này dành cho contributor làm việc với source repository.

## Repository boundaries

Core reusable code:

```text
wemm_runtime/
```

Presentation của public notebook và tích hợp Kaggle:

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

## Chính sách tài liệu song ngữ

Mọi public Markdown document phải có counterpart Anh/Việt.

Ví dụ:

```text
docs/api.md
docs/api.vi.md
```

Hai file phải thay đổi trong cùng một commit. CI enforce pairing và exact language-navigation header.

## Documentation link policy

Relative Markdown links phải resolve tới file tồn tại trong repository. External HTTP(S), mailto và fragment-only links không thuộc local target check.

## Qualification-sensitive changes

Không xử lý thay đổi model/runtime science như documentation/maintenance thông thường.

Cần qualification phase mới khi thay đổi:

- model selection/weights;
- precision;
- GPU placement;
- Qdrant corpus semantics;
- public retrieval examples;
- visual transforms;
- acceptance threshold;
- scorecard methodology.

## Kaggle system Torch policy

`requirements-kaggle.txt` chủ ý không sở hữu PyTorch/CUDA. Project venv reuse Kaggle system runtime và setup guards verify identity trước/sau dependency installation.

## Pull requests

Trước khi mở PR:

1. giữ scope hẹp;
2. chạy local checks;
3. update cả hai language variants của public docs;
4. nêu rõ qualification boundaries có bị ảnh hưởng không;
5. kèm sanitized evidence/logs nếu liên quan.

Xem [CONTRIBUTING.md](../.github/CONTRIBUTING.vi.md).
