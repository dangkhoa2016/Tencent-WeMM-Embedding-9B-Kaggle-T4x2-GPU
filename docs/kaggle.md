# Kaggle Guide

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](kaggle.vi.md)

This guide is the recommended end-user path for reproducing the public demo on Kaggle.

## Requirements

| Requirement | Qualified value |
| --- | --- |
| Platform | Kaggle Notebook |
| Accelerator | **GPU T4 ×2** |
| Internet | **ON** for the current public demo workflow |
| Model input | `dangkhoa2016/tencent-wemm-embedding-9b`, version 1 |
| Dataset input | `dangkhoa2016/wemm-embedding-9b-v1-qdrant-snapshots`, version 1 |
| Precision | FP16 |
| Qdrant | 1.19.0 |
| Public vector sizes | 4096d, 1024d |

## Recommended entry point

Use:

[`notebooks/kaggle-production-demo-thin.ipynb`](../notebooks/kaggle-production-demo-thin.ipynb)

The Git copy is intentionally clean and unexecuted. A Kaggle Saved Version is execution evidence, not source authority.

## Setup sequence

1. Create or open a Kaggle Notebook.
2. Select **GPU T4 ×2**.
3. Turn Internet **ON** for the current public workflow.
4. Attach the model input and Qdrant snapshot dataset listed above.
5. Import the public notebook.
6. Run the notebook from top to bottom.

The workflow performs strict checks and should stop if required inputs or hardware do not match.

## Lower-level setup scripts

For direct script-based setup:

```bash
bash kaggle/setup.sh
bash kaggle/run-preflight.sh
bash kaggle/run-acceptance.sh
```

The setup creates a dedicated virtual environment with `--system-site-packages` and deliberately reuses Kaggle's system PyTorch. `requirements-kaggle.txt` is not allowed to install another Torch/CUDA/NVIDIA runtime stack.

For the public notebook bootstrap, dependency ownership is intentionally split: `requirements-kaggle.txt` is read from the frozen runtime checkout, while `requirements-demo.txt` is read from the public presentation source checked out from `v1.0.0`. Do not copy `requirements-demo.txt` into the frozen runtime tree as a workaround.

## Offline runtime behavior

After setup, runtime scripts export:

```text
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1
```

This means the model execution path resolves local Kaggle input files rather than fetching weights at inference time. The overall public notebook still requires Internet ON for its current complete demo workflow.

## Model discovery

The resolver expects model content below:

```text
/kaggle/input
```

`KAGGLE_MODEL_DIR` is only needed if automatic discovery is ambiguous. Paths outside the Kaggle input boundary are rejected by the public integration.

## What the notebook runs

The production-style notebook covers:

1. host/input preflight;
2. Qdrant availability and snapshot reuse/restore;
3. model worker startup;
4. verified dual-T4 placement;
5. semantic text retrieval;
6. semantic image→text retrieval;
7. visual robustness retrieval;
8. scorecard summary;
9. worker/Qdrant shutdown.

## Tokenizer padding evidence

The qualified runtime loads the real processor from `/kaggle/input`. At model load it emits a deterministic evidence marker:

```text
WEMM_TOKENIZER_PADDING_SIDE=right
WEMM_TOKENIZER_PADDING_CONTRACT=PASS
```

The right-padding contract is required for EOS pooling. If the loaded tokenizer does not expose `padding_side`, the runtime emits:

```text
WEMM_TOKENIZER_PADDING_SIDE=UNKNOWN
WEMM_TOKENIZER_PADDING_CONTRACT=NOT_PROVEN
```

A not-proven marker is not treated as qualified. A `left` value (or batch behavior that contradicts right padding) requires a new T4×2 qualification phase; the embedding-science change must not be silently patched.

## Disk hygiene

The public workflow is designed to reuse verified data rather than continually rebuilding persistent copies. If a working copy becomes dirty or a restore is required, the notebook should restore from the read-only source instead of accumulating multiple mutable copies.

## Failure policy

The public path does not silently fall back to CPU when the qualified GPU configuration is expected. Unsupported accelerator, missing model files, invalid snapshots, device-map violations, or unsupported dimensions should fail clearly.

## Next

- [Troubleshooting](troubleshooting.md)
- [Reproducibility](reproducibility.md)
- [Qdrant](qdrant.md)
