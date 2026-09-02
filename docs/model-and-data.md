# Model and Data

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](model-and-data.vi.md)

This repository contains integration code and documentation. It does **not** relicense the upstream **WeMM-Embedding-9B** model, developed by the **WeChat Vision Team at Tencent**, or the public data used by the demo.

## Model

Public Kaggle input:

```text
dangkhoa2016/tencent-wemm-embedding-9b
version 1
```

The runtime expects a complete local **WeMM-Embedding-9B** model directory containing the upstream custom model code, processor/tokenizer configuration, and safetensors weights.

The repository does not embed the 9B model weights in Git or in the source release archive.

## Qdrant dataset

Public snapshot dataset:

```text
dangkhoa2016/wemm-embedding-9b-v1-qdrant-snapshots
version 1
```

It provides verified Qdrant storage/snapshots used by the public semantic retrieval demo.

## Semantic corpus

The qualified corpus contains **99,967 Wikidata entities**. Each qualified Qdrant collection exposes named English and Vietnamese vectors.

The project uses the corpus for reproducible bilingual and cross-modal retrieval demonstrations. It does not claim ownership of Wikidata content.

## Visual inputs

Visual robustness uses a separate controlled four-image gallery. Those images are demo inputs, not the semantic corpus, and they are not distributed as v1.0.0 release artifacts.

## Licensing boundaries

The repository source is MIT-licensed.

Separate licensing/terms apply to:

- WeMM-Embedding-9B model weights and upstream model code from the Tencent/WeMM-Embedding project;
- Wikidata-derived corpus content;
- Kaggle services and hosted inputs;
- Qdrant;
- PyTorch, Transformers, FastAPI, and other dependencies.

Users are responsible for reviewing upstream terms before redistributing model/data artifacts.

## Data integrity

The public workflow prefers verified read-only inputs and snapshot reuse. Mutable working copies should be treated as disposable runtime state, not source authority.

See [Qdrant](qdrant.md), [Reproducibility](reproducibility.md), and [License](../LICENSE).
