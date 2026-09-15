# Tài liệu

> 🌐 Language / Ngôn ngữ: [English](index.md) | **Tiếng Việt**

Đây là bộ tài liệu dài-form cho dự án **Tencent WeMM-Embedding-9B trên Kaggle T4×2**. README ở root đóng vai trò landing page; các tài liệu bên dưới giải thích chi tiết hơn về setup, kiến trúc, giao diện sử dụng, đánh giá, khả năng tái lập, giới hạn và cách phát triển repository.

## Chọn tài liệu theo nhu cầu

| Mục tiêu | Bắt đầu tại |
| --- | --- |
| Chạy public demo trên Kaggle | [Hướng dẫn Kaggle](kaggle.vi.md) |
| Hiểu kiến trúc hệ thống | [Kiến trúc](architecture.vi.md) |
| Xem v1.0.0 hỗ trợ gì | [Tính năng](features.vi.md) |
| Tích hợp reusable runtime | [Python Runtime](python-runtime.vi.md) |
| Gọi local service | [REST API](api.vi.md) |
| Hiểu model/data và quyền sở hữu | [Model và Data](model-and-data.vi.md) |
| Hiểu Qdrant collections và snapshots | [Qdrant](qdrant.vi.md) |
| Review methodology và kết quả retrieval | [Retrieval và Evaluation](retrieval-and-evaluation.vi.md) |
| Tái lập cấu hình đã công bố | [Reproducibility](reproducibility.vi.md) |
| Hiểu rõ giới hạn | [Limitations](limitations.vi.md) |
| Chẩn đoán lỗi thường gặp | [Troubleshooting](troubleshooting.vi.md) |
| Phát triển repository | [Development](development.vi.md) |
| Xem định hướng | [Roadmap](roadmap.vi.md) |
| Xem release record ổn định | [Release Notes v1.0.0](releases/v1.0.0.vi.md) |
| Xem lịch sử phiên bản | [CHANGELOG](../CHANGELOG.vi.md) |

## Nguyên tắc tài liệu

- **Ưu tiên cộng đồng:** tài liệu hướng tới user/reviewer, không chỉ maintainer.
- **Dựa trên evidence:** kết quả đã xác minh được tách biệt với mô tả capability tổng quát.
- **Song ngữ:** mọi public Markdown đều có cặp Anh/Việt.
- **Fail-closed:** setup/runtime dừng trên cấu hình không hỗ trợ thay vì âm thầm downgrade.
- **Bám source:** tài liệu dài-form phản ánh public source tree và interface hiện đang được test.

## Stable release

Stable release hiện tại là **v1.0.0**. Annotated tag được dùng làm stable source identity của release, thay vì hard-code một publication commit có thể thay đổi trong chính tài liệu.

Public notebook pin reusable runtime authority tại:

```text
d04bcd3e601b449b67d09ff1132cab965619d858
```

Xem [Reproducibility](reproducibility.vi.md) để hiểu sự khác nhau giữa release source, runtime authority và execution evidence.
