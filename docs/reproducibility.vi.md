# Reproducibility

> 🌐 Language / Ngôn ngữ: [English](reproducibility.md) | **Tiếng Việt**

Dự án phân biệt source authority, runtime authority, execution evidence và release presentation.

## Source authority

Stable source identity của v1.0.0 là annotated release tag cùng Git commit history của nó.

Documentation suite không thay đổi scientific/runtime qualification boundaries.

## Runtime authority

Public notebook pin reusable runtime commit:

```text
33417cee59bd4eb03bfa1e5e0fdc605ae558cf02
```

Commit này vẫn reachable trong public history.

## Execution evidence

Kaggle Saved Version chứng minh notebook đã chạy trong một môi trường Kaggle cụ thể. Nó là execution evidence, không phải canonical source copy.

Git notebook chủ ý giữ clean/unexecuted.

## Hardware/runtime đã xác minh

| Thuộc tính | Giá trị |
| --- | --- |
| Accelerator | NVIDIA T4 ×2 |
| Precision | FP16 |
| GPU0 modules | 13 |
| GPU1 modules | 24 |
| CPU offload | không |
| Disk offload | không |
| Qdrant | 1.19.0 |
| Public dimensions | 4096, 1024 |

## Runtime dependencies đã pin

Qualified Kaggle setup pin:

```text
transformers==5.2.0
accelerate==1.14.0
qwen-vl-utils[decord]==0.0.14
```

System PyTorch của Kaggle được chủ ý tái sử dụng. Setup guard từ chối Torch/CUDA/NVIDIA stack thứ hai trong project venv.

Public notebook cũng tách dependency authority rõ ràng: `requirements-kaggle.txt` được lấy từ frozen runtime commit ở trên, còn `requirements-demo.txt` thuộc current public presentation source. Không được tìm presentation-only requirements file bên trong frozen runtime checkout.

## CI

Repository CI chạy CPU-only và verify source-level contracts, documentation policy, local Markdown links và tests.

Current public test qualification:

```text
142 passed
3 skipped
2 warnings
```

CPU CI không thay thế hardware execution trên T4 ×2.

## Source downloads

GitHub tự động cung cấp **Source code (zip)** và **Source code (tar.gz)** cho release tag.

Dự án không upload custom source archive thứ hai hoặc custom archive checksum vì chúng sẽ trùng chức năng với source download dựa trên tag của GitHub. Git commit/tag identity là canonical source reference.

## Checklist tái lập

1. Dùng published `v1.0.0` tag/source.
2. Chọn Kaggle T4 ×2.
3. Attach exact model và snapshot dataset.
4. Dùng public notebook.
5. Để strict preflight validate environment/storage.
6. So sánh scorecard output với retrieval boundaries đã document.
7. Giữ semantic và visual evaluation spaces riêng biệt.

## Thay đổi nào được coi là qualification change?

Thay đổi các mục sau cần qualification phase mới thay vì documentation-only update:

- model weights/model selection;
- dual-GPU placement;
- precision;
- Qdrant corpus semantics;
- frozen semantic examples;
- visual robustness examples/transforms;
- raw cosine threshold;
- public scorecard methodology.

Xem [Retrieval và Evaluation](retrieval-and-evaluation.vi.md).
