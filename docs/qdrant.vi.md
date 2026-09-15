# Qdrant

> 🌐 Language / Ngôn ngữ: [English](qdrant.md) | **Tiếng Việt**

Public demo dùng Qdrant 1.19.0 để thực hiện vector retrieval có thể tái lập.

## Qualified collections

```text
wikidata_en_vi_wemm9b_4096_v030_rc2_d819dc7_v1
wikidata_en_vi_wemm9b_1024_v030_rc2_d819dc7_v1
```

Mỗi collection có **99.967 points**.

## Vector schema

Semantic collections expose named vector tiếng Anh và tiếng Việt, dùng cosine distance.

Hai public embedding dimensions:

- 4096d;
- 1024d.

## Snapshot reuse

Public Kaggle workflow reuse/restore verified snapshots thay vì rebuild toàn bộ corpus mỗi session.

Điều này cải thiện:

- reproducibility;
- startup time;
- disk hygiene;
- consistency giữa các public demo.

## Read-only authority và working state

Kaggle dataset là read-only input authority. Qdrant working directory sau restore là mutable/disposable runtime state.

Nếu working copy bị dirty, recovery an toàn hơn là bỏ nó và restore lại từ verified source thay vì tạo thêm copy mới.

## Semantic và visual collections

Không được nhầm production semantic collections với visual robustness gallery.

Semantic retrieval:

```text
99.967 entities
persistent verified snapshots
named vector Anh/Việt
```

Visual robustness:

```text
4 ảnh gốc
temporary isolated gallery
không persist vào semantic storage
```

## Safety boundary

Public demo kiểm tra expected collection names và point counts trước khi coi storage là qualified.

Xem [Retrieval và Evaluation](retrieval-and-evaluation.vi.md) để hiểu các collection được dùng thế nào.
