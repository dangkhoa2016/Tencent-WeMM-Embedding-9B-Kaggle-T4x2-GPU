# Features

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](features.vi.md)

This page describes the supported public feature surface of **v1.0.0**.

## Embedding capabilities

| Feature | v1.0.0 |
| --- | --- |
| Text embeddings | Supported |
| Image embeddings | Supported |
| Image + text embeddings | Supported |
| English input | Supported |
| Vietnamese input | Supported |
| 4096d vectors | Supported |
| 1024d vectors | Supported |
| Normalized output vectors | Supported |
| Batch REST requests | Supported, bounded |
| Local Python runtime | Supported |
| Loopback REST API | Supported |
| Public Internet serving | Not claimed |

## Kaggle runtime

The verified execution profile supports:

- NVIDIA T4 ×2;
- FP16;
- two-GPU model placement;
- no CPU offload;
- no disk offload;
- reuse of Kaggle's existing CUDA-enabled PyTorch;
- local model discovery under `/kaggle/input`.

## Retrieval

The public demo includes:

- English → Vietnamese semantic retrieval;
- Vietnamese → English semantic retrieval;
- image → English text retrieval;
- image → Vietnamese text retrieval;
- 4096d and 1024d paths;
- controlled transformed-image → original-image robustness retrieval.

## Qdrant reuse

The project can restore/reuse verified Qdrant snapshots for the semantic corpus. This avoids rebuilding the 99,967-entity corpus during every public demo run.

## REST API

The local service provides:

- health and readiness endpoints;
- bearer-token authentication;
- text embeddings;
- image embeddings;
- image+text embeddings;
- bounded queue and batch configuration;
- image safety validation;
- request IDs and structured timing;
- explicit error mapping for not-ready, queue-full, timeout, inference failure, CUDA OOM, and invalid/oversized images.

See [REST API](api.md) for exact limits and examples.

## Repository quality

v1.0.0 also includes:

- English/Vietnamese public documentation pairing;
- MIT source license;
- GitHub Actions CI;
- community health files;
- issue and pull-request templates;
- publication-history policy checks;
- Markdown link validation.

## Explicit non-features

The release does not claim:

- support for arbitrary GPU types;
- production Internet-facing inference hardening;
- high availability;
- autoscaling;
- multi-tenant isolation;
- an SLA;
- universal benchmark accuracy from the public showcase scorecards.

See [Limitations](limitations.md).
