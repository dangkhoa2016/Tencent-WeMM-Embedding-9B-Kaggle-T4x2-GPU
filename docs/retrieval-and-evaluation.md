# Retrieval and Evaluation

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](retrieval-and-evaluation.vi.md)

This document explains what the published retrieval numbers mean and, equally importantly, what they do **not** mean.

## Two independent evaluation spaces

The public demo reports two separate retrieval spaces.

| Evaluation | Search space | Result |
| --- | ---: | ---: |
| Bilingual + cross-modal semantic retrieval | 99,967 entities per Qdrant collection | **36/36 TOP-1** |
| Visual robustness retrieval | isolated gallery of 4 original images | **32/32 TOP-1** |
| Total executed checks | two distinct spaces above | **68/68 PASS** |

The total 68/68 is a count of executed checks. It is **not** a single 68-query benchmark over one shared index.

## Semantic retrieval

The semantic portion uses the verified 4096d and 1024d Qdrant collections.

The frozen public showcase includes:

- English → Vietnamese text retrieval;
- Vietnamese → English text retrieval;
- both public vector dimensions;
- image → English text retrieval;
- image → Vietnamese text retrieval.

For these explicitly defined public paths, the expected entity is ranked #1 in **36/36 checks**.

This is reproducibility evidence for the published examples and corpus configuration. It is not a universal accuracy claim for every possible query.

## Visual robustness

The visual section is deliberately isolated from the 99,967-entity corpus.

It uses four original images and four deterministic transformations:

- resize to 80%;
- JPEG quality 90;
- center crop to 96%;
- brightness 103%.

Each transformed query is evaluated at both dimensions:

```text
4 originals × 4 transforms × 2 dimensions = 32 checks
```

Acceptance requires:

1. the correct original image is ranked #1;
2. raw cosine similarity is at least 0.90.

All published paths pass: **32/32**.

## Score interpretation

Raw cosine similarity is reported directly.

Do not interpret:

```text
0.93 cosine
```

as:

```text
93% confidence
```

The project does not rescale cosine values into percentages.

## Why the visual gallery is small

The visual robustness gallery is a controlled invariance check, not an image benchmark. Its purpose is to confirm that realistic image transforms preserve nearest-neighbor identity under the published embedding path.

The gallery is temporary and contains only the four originals. It is never merged into the semantic Wikidata corpus.

## Reproducing results

Use the public Kaggle notebook and the exact named Qdrant inputs. See:

- [Kaggle Guide](kaggle.md)
- [Qdrant](qdrant.md)
- [Reproducibility](reproducibility.md)
