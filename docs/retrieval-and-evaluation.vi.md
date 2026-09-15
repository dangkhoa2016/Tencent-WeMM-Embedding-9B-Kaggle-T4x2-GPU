# Retrieval và Evaluation

> 🌐 Language / Ngôn ngữ: [English](retrieval-and-evaluation.md) | **Tiếng Việt**

Tài liệu này giải thích các con số retrieval đã công bố có ý nghĩa gì và quan trọng không kém: chúng **không** có ý nghĩa gì.

## Hai không gian đánh giá độc lập

Public demo báo cáo hai retrieval space riêng biệt.

| Đánh giá | Search space | Kết quả |
| --- | ---: | ---: |
| Semantic retrieval song ngữ + cross-modal | 99.967 thực thể trên mỗi Qdrant collection | **36/36 TOP-1** |
| Visual robustness retrieval | gallery tách biệt gồm 4 ảnh gốc | **32/32 TOP-1** |
| Tổng số check đã thực thi | hai không gian riêng ở trên | **68/68 PASS** |

68/68 là tổng số check đã chạy. Nó **không** phải một benchmark 68 query trên cùng một index.

## Semantic retrieval

Phần semantic dùng verified Qdrant collections 4096d và 1024d.

Frozen public showcase gồm:

- English → Vietnamese text retrieval;
- Vietnamese → English text retrieval;
- cả hai public vector dimensions;
- image → English text retrieval;
- image → Vietnamese text retrieval.

Trong các public path được định nghĩa rõ này, expected entity đứng #1 ở **36/36 checks**.

Đây là reproducibility evidence cho examples/corpus configuration đã công bố, không phải universal accuracy claim cho mọi query.

## Visual robustness

Phần visual chủ ý tách biệt khỏi corpus 99.967 thực thể.

Nó sử dụng bốn ảnh gốc và bốn deterministic transforms:

- resize còn 80%;
- JPEG quality 90;
- center crop còn 96%;
- brightness 103%.

Mỗi transformed query được đánh giá ở cả hai dimensions:

```text
4 ảnh gốc × 4 transforms × 2 dimensions = 32 checks
```

Acceptance yêu cầu:

1. ảnh gốc đúng đứng #1;
2. raw cosine similarity tối thiểu 0.90.

Toàn bộ public path PASS: **32/32**.

## Cách hiểu score

Raw cosine similarity được báo cáo trực tiếp.

Không được hiểu:

```text
0.93 cosine
```

thành:

```text
93% confidence
```

Dự án không rescale cosine thành phần trăm.

## Vì sao visual gallery nhỏ?

Visual robustness gallery là controlled invariance check, không phải image benchmark. Mục tiêu là xác nhận các biến đổi ảnh thực tế vẫn giữ nearest-neighbor identity theo embedding path đã công bố.

Gallery tồn tại tạm và chỉ chứa bốn ảnh gốc. Nó không bao giờ được merge vào semantic Wikidata corpus.

## Tái lập kết quả

Dùng public Kaggle notebook và exact named Qdrant inputs. Xem:

- [Hướng dẫn Kaggle](kaggle.vi.md)
- [Qdrant](qdrant.vi.md)
- [Reproducibility](reproducibility.vi.md)
