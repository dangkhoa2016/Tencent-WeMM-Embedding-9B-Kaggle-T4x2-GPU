# Giới hạn

> 🌐 Language / Ngôn ngữ: [English](limitations.md) | **Tiếng Việt**

Tài liệu này nêu rõ boundaries của public v1.0.0 để user phân biệt verified behavior với assumption chưa được support.

## Hardware scope

Accelerator target đã xác minh:

```text
NVIDIA T4 ×2
```

Accelerator khác có thể được hỗ trợ trong tương lai nhưng không thuộc qualified claim của v1.0.0.

## Precision và placement

Cấu hình đã xác minh dùng FP16 trên hai T4, không CPU hoặc disk offload.

Dự án không tuyên bố equivalence cho quantization, precision hoặc placement khác.

## Public path riêng cho Kaggle

Public notebook kỳ vọng named Kaggle model và Qdrant snapshot inputs. General runtime tái sử dụng được, nhưng published end-to-end evidence là Kaggle-specific.

## API scope

FastAPI service:

- chỉ bind loopback hosts;
- dùng bearer-token authentication;
- có bounded queue/batch;
- không cung cấp Internet-facing hardening;
- không tuyên bố HA, autoscaling, SLA hoặc multi-tenant isolation.

## Evaluation scope

Semantic 36/36 áp dụng cho explicit frozen public retrieval paths trên corpus configuration đã document.

Visual 32/32 áp dụng cho controlled four-image gallery với transforms đã document.

Không con số nào là universal model-accuracy benchmark.

## Dataset/model ownership

MIT chỉ áp dụng cho repository source code. Model weights và data giữ upstream licensing/terms.

## Resource variability

Kaggle runtime capacity, base image và platform availability có thể thay đổi. Strict preflight giúp phát hiện incompatible environment thay vì âm thầm làm yếu run.

## Internet behavior

Public notebook hiện cần Internet ON tổng thể, trong khi model runtime sau setup bị ép local/offline loading.

## Support model

Đây là community engineering project, support theo best-effort và không có commercial SLA.

## Tài liệu liên quan

- [Tính năng](features.vi.md)
- [Hướng dẫn Kaggle](kaggle.vi.md)
- [REST API](api.vi.md)
- [Retrieval và Evaluation](retrieval-and-evaluation.vi.md)
