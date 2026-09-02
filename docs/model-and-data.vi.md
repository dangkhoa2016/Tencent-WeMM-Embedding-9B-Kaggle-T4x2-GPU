# Model và Data

> 🌐 Language / Ngôn ngữ: [English](model-and-data.md) | **Tiếng Việt**

Repository này chứa integration code và tài liệu. Nó **không** relicense model upstream **WeMM-Embedding-9B**, được phát triển bởi **đội ngũ WeChat Vision tại Tencent**, hoặc public data dùng trong demo.

## Model

Public Kaggle input:

```text
dangkhoa2016/tencent-wemm-embedding-9b
version 1
```

Runtime yêu cầu local **WeMM-Embedding-9B** model directory đầy đủ gồm upstream custom model code, processor/tokenizer configuration và safetensors weights.

Repository không nhúng model weights 9B vào Git hoặc source release archive.

## Qdrant dataset

Public snapshot dataset:

```text
dangkhoa2016/wemm-embedding-9b-v1-qdrant-snapshots
version 1
```

Dataset cung cấp verified Qdrant storage/snapshots dùng trong public semantic retrieval demo.

## Semantic corpus

Qualified corpus có **99.967 Wikidata entities**. Mỗi qualified Qdrant collection expose named vector tiếng Anh và tiếng Việt.

Dự án sử dụng corpus cho bilingual/cross-modal retrieval demo có thể tái lập; không tuyên bố ownership với Wikidata content.

## Visual inputs

Visual robustness dùng controlled four-image gallery riêng. Các ảnh này là demo input, không phải semantic corpus và không được phân phối như v1.0.0 release artifacts.

## Licensing boundaries

Repository source theo MIT.

License/terms riêng áp dụng cho:

- WeMM-Embedding-9B model weights và upstream model code từ dự án Tencent/WeMM-Embedding;
- Wikidata-derived corpus content;
- Kaggle services và hosted inputs;
- Qdrant;
- PyTorch, Transformers, FastAPI và dependencies khác.

User cần tự review upstream terms trước khi redistribute model/data artifacts.

## Data integrity

Public workflow ưu tiên verified read-only input và snapshot reuse. Mutable working copy nên được coi là disposable runtime state, không phải source authority.

Xem [Qdrant](qdrant.vi.md), [Reproducibility](reproducibility.vi.md) và [License](../LICENSE).
