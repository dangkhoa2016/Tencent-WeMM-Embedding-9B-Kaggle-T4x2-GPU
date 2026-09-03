# Reproducibility

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](reproducibility.vi.md)

The project distinguishes source authority, runtime authority, execution evidence, and release presentation.

## Source authority

The stable source identity for v1.0.0 is the annotated release tag together with its Git commit history.

The documentation suite does not change the scientific/runtime qualification boundaries.

## Runtime authority

The public notebook pins the reusable runtime commit:

```text
224f07cd1d6eb174d3532c9eaeeb9abd606a857f
```

That commit remains reachable in public history.

## Execution evidence

A Kaggle Saved Version proves that the notebook executed in a concrete Kaggle environment. It is evidence of execution, not the canonical source copy.

The Git notebook intentionally remains clean/unexecuted.

## Verified hardware/runtime

| Property | Value |
| --- | --- |
| Accelerator | NVIDIA T4 ×2 |
| Precision | FP16 |
| GPU0 modules | 13 |
| GPU1 modules | 24 |
| CPU offload | none |
| Disk offload | none |
| Qdrant | 1.19.0 |
| Public dimensions | 4096, 1024 |

## Pinned runtime dependencies

The qualified Kaggle setup pins:

```text
transformers==5.2.0
accelerate==1.14.0
qwen-vl-utils[decord]==0.0.14
```

Kaggle system PyTorch is intentionally reused. The setup guard rejects a second local Torch/CUDA/NVIDIA stack in the project venv.

The public notebook keeps dependency authority split explicitly: `requirements-kaggle.txt` comes from the frozen runtime commit above, while `requirements-demo.txt` comes from the current public presentation source. The presentation-only requirements file must not be looked up under the frozen runtime checkout.

## CI

Repository CI is CPU-only and verifies source-level contracts, documentation policy, local Markdown links, and tests.

v1.0.0 release qualification baseline:

```text
470 passed
3 skipped
234 Markdown link targets
```

CPU CI is not a replacement for the T4 ×2 hardware execution.

## Source downloads

GitHub automatically provides **Source code (zip)** and **Source code (tar.gz)** for the release tag.

The project does not upload a second custom source archive or a custom archive checksum because those would duplicate GitHub's tag-based source downloads. Git commit/tag identity is the canonical source reference.

## Reproduction checklist

1. Use the published `v1.0.0` tag/source.
2. Select Kaggle T4 ×2.
3. Attach the exact model and snapshot dataset.
4. Use the public notebook.
5. Allow strict preflight to validate environment and storage.
6. Compare scorecard outputs with the documented retrieval boundaries.
7. Keep semantic and visual evaluation spaces separate.

## What counts as a qualification change?

Changes to any of the following should trigger a new qualification phase rather than a documentation-only update:

- model weights/model selection;
- dual-GPU placement;
- precision;
- Qdrant corpus semantics;
- frozen semantic examples;
- visual robustness examples/transforms;
- raw cosine threshold;
- public scorecard methodology.

See [Retrieval and Evaluation](retrieval-and-evaluation.md).
