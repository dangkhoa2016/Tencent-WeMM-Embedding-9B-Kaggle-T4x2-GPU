# Roadmap

> 🌐 Language / Ngôn ngữ: [English](roadmap.md) | **Tiếng Việt**

Roadmap này mô tả hướng đi, không phải lời hứa hoặc delivery date. Stable contract hiện tại vẫn là **v1.0.0**.

## Current stable scope

v1.0.0 tập trung vào:

- WeMM-Embedding-9B trên Kaggle T4 ×2;
- FP16 dual-GPU runtime;
- text, image và image+text embeddings;
- Matryoshka vectors 4096d và 1024d;
- bilingual/cross-modal Qdrant retrieval;
- loopback FastAPI;
- public notebook có thể tái lập;
- public documentation song ngữ.

## Near-term maintenance

Các hướng maintenance khả thi:

- giữ docs đồng bộ với source interfaces;
- cải thiện troubleshooting/examples theo issue thực tế;
- giữ CI dependencies và action pins cập nhật;
- duy trì release package sạch;
- giữ Kaggle notebook tương thích khi platform/runtime thay đổi mà không làm yếu qualification gates.

## Candidate feature directions

Một số hướng tương lai, cần qualification độc lập:

- hỗ trợ thêm accelerators;
- thêm serving/deployment adapters;
- profiling batch/performance rộng hơn;
- thêm retrieval examples;
- cải thiện packaging cho reusable runtime integration.

Các mục này không thuộc support claim của v1.0.0.

## Research directions

Các hướng research-only có thể xem xét:

- precision/quantization khác;
- retrieval evaluation set lớn hơn;
- image robustness suite rộng hơn;
- multilingual retrieval ngoài showcase Anh/Việt hiện tại;
- performance comparison giữa accelerator classes.

Kết quả tương lai cần methodology riêng và không được retroactively nhập vào v1.0.0 claims.

## Explicit non-goals của v1.0.0

- Internet-facing managed inference;
- HA/autoscaling platform;
- universal model benchmark claims;
- tự động support arbitrary hardware;
- relicense upstream model/data artifacts.

## Nguyên tắc versioning

Public behavior mới nên đi cùng release mới khi nó thay đổi stable user contract hoặc qualification boundary. Documentation corrections không đổi behavior có thể là maintenance update của current public source, nhưng release record phải luôn rõ ràng.
