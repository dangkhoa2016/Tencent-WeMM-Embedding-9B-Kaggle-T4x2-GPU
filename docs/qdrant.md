# Qdrant

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](qdrant.vi.md)

The public demo uses Qdrant 1.19.0 for reproducible vector retrieval.

## Qualified collections

```text
wikidata_en_vi_wemm9b_4096_v030_rc2_d819dc7_v1
wikidata_en_vi_wemm9b_1024_v030_rc2_d819dc7_v1
```

Each collection contains **99,967 points**.

## Vector schema

The semantic collections expose named English and Vietnamese vectors and use cosine distance.

Two public embedding dimensions are represented:

- 4096d;
- 1024d.

## Snapshot reuse

The public Kaggle workflow reuses/restores verified snapshots rather than rebuilding the entire corpus in every session.

This improves:

- reproducibility;
- startup time;
- disk hygiene;
- consistency between public demonstrations.

## Read-only authority vs working state

The Kaggle dataset is read-only input authority. A restored Qdrant working directory under the notebook runtime is mutable and disposable.

If the working copy becomes dirty, the safer recovery path is to discard it and restore again from the verified source rather than layering another copy on top.

## Semantic vs visual collections

Do not confuse the production semantic collections with the visual robustness gallery.

Semantic retrieval:

```text
99,967 entities
persistent verified snapshots
English/Vietnamese named vectors
```

Visual robustness:

```text
4 original images
temporary isolated gallery
not persisted into semantic storage
```

## Safety boundary

The public demo checks expected collection names and point counts before treating storage as qualified.

See [Retrieval and Evaluation](retrieval-and-evaluation.md) for how the collections are used.
