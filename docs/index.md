# Documentation

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](index.vi.md)

This documentation set is the long-form manual for the independent **Tencent-WeMM-Embedding-9B-Kaggle-T4x2-GPU** project. It integrates the upstream **WeMM-Embedding-9B** model, developed by the **WeChat Vision Team at Tencent**, with a qualified Kaggle T4×2 workflow. The root README is the project landing page; the documents below explain setup, architecture, interfaces, evaluation, reproducibility, limits, and maintenance in more depth. This repository is independent and is not an official Tencent or WeChat release.

## Choose your path

| Goal | Start here |
| --- | --- |
| Run the public demo on Kaggle | [Kaggle Guide](kaggle.md) |
| Understand the system design | [Architecture](architecture.md) |
| See what v1.0.0 supports | [Features](features.md) |
| Integrate the reusable runtime | [Python Runtime](python-runtime.md) |
| Call the local service | [REST API](api.md) |
| Understand model/data ownership | [Model and Data](model-and-data.md) |
| Understand Qdrant collections and snapshots | [Qdrant](qdrant.md) |
| Review retrieval methodology and results | [Retrieval and Evaluation](retrieval-and-evaluation.md) |
| Reproduce the published configuration | [Reproducibility](reproducibility.md) |
| Understand known boundaries | [Limitations](limitations.md) |
| Diagnose common failures | [Troubleshooting](troubleshooting.md) |
| Work on the repository | [Development](development.md) |
| See planned directions | [Roadmap](roadmap.md) |
| Review the stable release record | [v1.0.0 Release Notes](releases/v1.0.0.md) |
| Review version history | [CHANGELOG](../CHANGELOG.md) |

## Documentation principles

- **Public-first:** documents explain the project for users and reviewers, not only maintainers.
- **Evidence-aware:** verified results are separated from general capability descriptions.
- **Bilingual:** every public Markdown document has an English/Vietnamese counterpart.
- **Fail-closed:** setup and runtime behavior should stop on unsupported configurations rather than silently downgrade.
- **Source-linked:** long-form documentation reflects the current public source tree and its tested interfaces.

## Stable release

The current stable release is **v1.0.0**. Treat the annotated tag as the stable source identity for the release rather than hard-coding a mutable publication commit inside the documentation.

The public notebook pins the reusable runtime authority at:

```text
224f07cd1d6eb174d3532c9eaeeb9abd606a857f
```

See [Reproducibility](reproducibility.md) for the distinction between release source, runtime authority, and execution evidence.
