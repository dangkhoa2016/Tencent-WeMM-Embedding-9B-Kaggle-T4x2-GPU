# Tính năng

> 🌐 Language / Ngôn ngữ: [English](features.md) | **Tiếng Việt**

Tài liệu này mô tả public feature surface được hỗ trợ trong **v1.0.0**.

## Khả năng embedding

| Tính năng | v1.0.0 |
| --- | --- |
| Text embeddings | Hỗ trợ |
| Image embeddings | Hỗ trợ |
| Image + text embeddings | Hỗ trợ |
| Input tiếng Anh | Hỗ trợ |
| Input tiếng Việt | Hỗ trợ |
| Vector 4096d | Hỗ trợ |
| Vector 1024d | Hỗ trợ |
| Vector output đã normalize | Hỗ trợ |
| Batch REST request | Hỗ trợ, có giới hạn |
| Local Python runtime | Hỗ trợ |
| Loopback REST API | Hỗ trợ |
| Public Internet serving | Không tuyên bố |

## Kaggle runtime

Execution profile đã xác minh hỗ trợ:

- NVIDIA T4 ×2;
- FP16;
- placement model trên hai GPU;
- không CPU offload;
- không disk offload;
- tái sử dụng CUDA-enabled PyTorch có sẵn của Kaggle;
- local model discovery dưới `/kaggle/input`.

## Retrieval

Public demo gồm:

- semantic retrieval English → Vietnamese;
- semantic retrieval Vietnamese → English;
- image → English text;
- image → Vietnamese text;
- path 4096d và 1024d;
- transformed-image → original-image robustness retrieval có kiểm soát.

## Tái sử dụng Qdrant

Dự án có thể restore/reuse verified Qdrant snapshots cho semantic corpus, tránh rebuild corpus 99.967 thực thể ở mỗi lần public demo.

## REST API

Local service cung cấp:

- health/readiness endpoints;
- bearer-token authentication;
- text embeddings;
- image embeddings;
- image+text embeddings;
- bounded queue và batch;
- image safety validation;
- request ID và structured timing;
- error mapping rõ ràng cho not-ready, queue-full, timeout, inference failure, CUDA OOM và image input không hợp lệ/quá lớn.

Xem [REST API](api.vi.md) để biết limits và ví dụ chính xác.

## Chất lượng repository

v1.0.0 còn có:

- public documentation pairing Anh/Việt;
- MIT source license;
- GitHub Actions CI;
- community health files;
- issue và pull-request templates;
- publication-history policy checks;
- Markdown link validation.

## Những gì release không tuyên bố

Release không tuyên bố:

- hỗ trợ mọi loại GPU;
- production Internet-facing inference hardening;
- high availability;
- autoscaling;
- multi-tenant isolation;
- SLA;
- universal benchmark accuracy từ public showcase scorecards.

Xem [Limitations](limitations.vi.md).
