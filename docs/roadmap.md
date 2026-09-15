# Roadmap

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](roadmap.vi.md)

This roadmap describes directions, not promises or delivery dates. The current stable contract remains **v1.0.0**.

## Current stable scope

v1.0.0 focuses on:

- WeMM-Embedding-9B on Kaggle T4 ×2;
- FP16 dual-GPU runtime;
- text, image, and image+text embeddings;
- 4096d and 1024d Matryoshka vectors;
- bilingual/cross-modal Qdrant retrieval;
- loopback FastAPI;
- reproducible public notebook;
- bilingual public documentation.

## Near-term maintenance

Candidate maintenance directions:

- keep documentation synchronized with source interfaces;
- improve user-facing troubleshooting and examples as real issues appear;
- keep CI dependencies and action pins current;
- maintain clean release packaging;
- keep the Kaggle notebook compatible with platform/runtime changes without weakening qualification gates.

## Candidate feature directions

Possible future work, subject to independent qualification:

- additional supported accelerators;
- additional serving/deployment adapters;
- broader embedding batch/performance profiling;
- richer retrieval examples;
- improved packaging around reusable runtime integration.

These are not part of the v1.0.0 support claim.

## Research directions

Potential research-only exploration:

- alternative precision/quantization configurations;
- larger retrieval evaluation sets;
- broader image robustness suites;
- multilingual retrieval beyond the current English/Vietnamese showcase;
- performance comparison across accelerator classes.

Any results would need separate methodology and should not be merged into v1.0.0 claims retroactively.

## Explicit non-goals for v1.0.0

- Internet-facing managed inference;
- HA/autoscaling platform;
- universal model benchmark claims;
- automatic support for arbitrary hardware;
- relicensing upstream model/data artifacts.

## Versioning principle

New public behavior should be reflected in a new release when it changes the stable user contract or qualification boundary. Documentation corrections that do not alter behavior may remain maintenance updates to the current public source while preserving clear release records.
