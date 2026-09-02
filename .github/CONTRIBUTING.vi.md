# Đóng góp

> 🌐 Language / Ngôn ngữ: [English](https://github.com/dangkhoa2016/Tencent-WeMM-Embedding-9B-Kaggle-T4x2-GPU/blob/main/.github/CONTRIBUTING.md) | **Tiếng Việt**

Cảm ơn bạn đã giúp cải thiện dự án. Hãy giữ thay đổi có phạm vi rõ ràng, có thể tái lập và nêu chính xác việc thay đổi có ảnh hưởng runtime qualification hay không.

## Development checks

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python -m pip install pytest packaging pillow -r requirements-api.txt -r requirements-demo.txt -r requirements-search.txt
python scripts/check_bilingual_docs.py --history
python scripts/check_doc_links.py
python scripts/check_publication_policy.py
python -m pytest -q
python -m compileall -q wemm_runtime wemm_kaggle scripts
bash -n kaggle/*.sh
git diff --check
```

Hardware qualification vẫn chạy trên Kaggle T4 ×2.

## Tài liệu

Mọi public Markdown document bắt buộc có counterpart Anh/Việt và hai file phải thay đổi trong cùng một commit.

## Qualification boundary

Không âm thầm thay đổi frozen examples, raw-cosine threshold, Qdrant corpus semantics, model weights hoặc dual-T4 placement. Những thay đổi đó cần qualification phase và evidence mới.
