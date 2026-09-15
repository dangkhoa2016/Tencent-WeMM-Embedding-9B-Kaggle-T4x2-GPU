# Troubleshooting

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](troubleshooting.vi.md)

This guide covers common failures in the public Kaggle/runtime/API path.

## Kaggle does not provide T4 ×2

The verified public path requires **GPU T4 ×2**. If Kaggle assigns another accelerator or no GPU, do not treat that run as equivalent qualification evidence.

Action:

- stop the run;
- select T4 ×2 when available;
- rerun preflight.

## Model input is not found

The public integration resolves model files below:

```text
/kaggle/input
```

Check that the model input is attached and contains the required WeMM files. Use `KAGGLE_MODEL_DIR` only when automatic discovery is ambiguous.

Do not point the public runtime at an arbitrary writable directory outside the intended Kaggle input boundary.

## Missing or incomplete safetensors

The runtime rejects incomplete model directories.

Verify that the model contains complete safetensors weights and the required configuration/tokenizer files described in [Model and Data](model-and-data.md).

## Setup tries to install another Torch/CUDA stack

The qualified setup intentionally reuses Kaggle's system PyTorch.

If `requirements-kaggle.txt` contains Torch, Triton, CUDA, or NVIDIA runtime packages, setup should fail. Remove those runtime specifications rather than forcing installation.

## Preflight reports GPU placement failure

The verified path requires both GPU IDs and forbids CPU/disk offload.

A placement mismatch can indicate:

- wrong accelerator;
- insufficient available GPU memory;
- model/runtime version drift;
- unexpected device-map behavior.

Do not bypass the validator.

## Unsupported embedding dimension

Public supported dimensions are:

```text
4096
1024
```

Other dimensions should be rejected.

## Qdrant snapshot restore fails

Check:

- the snapshot dataset is attached;
- the expected collection/snapshot files exist;
- writable disk has enough free space;
- a stale mutable copy is not conflicting with restore.

If the working storage is dirty, remove the disposable working copy and restore again from the verified read-only input.

## Qdrant point count or collection name mismatch

The public semantic authority expects the exact documented collection names and **99,967 points** per collection.

Treat mismatches as a failed qualification check rather than continuing with a different corpus.

## API refuses to start

Common causes:

- `WEMM_API_TOKEN` shorter than 32 characters;
- non-loopback `WEMM_API_HOST`;
- base Kaggle venv not created;
- model runtime not loadable;
- requested port already in use.

Default:

```text
127.0.0.1:8090
```

## API returns 503 NOT_READY

The runtime worker may still be initializing. Check `/readyz` and server logs.

A persistent NOT_READY state indicates initialization did not complete successfully.

## API returns QUEUE_FULL or REQUEST_TIMEOUT

The API has bounded concurrency. Default queue size is 16 items and request timeout is 30 seconds.

Reduce client concurrency or wait and retry.

## CUDA OOM

Do not reduce correctness checks or silently switch to CPU in the qualified path.

Check for:

- other GPU processes;
- stale model workers;
- changed model/runtime versions;
- changed per-GPU memory configuration.

Restart from a clean Kaggle session if needed.

## Disk usage keeps growing

The demo should reuse/restore verified storage, not create a new durable corpus copy on every run.

Inspect writable working directories for stale Qdrant restores or duplicated temporary data. Preserve read-only input authority and remove disposable dirty copies before restoring again.

## Where to report problems

Use GitHub Issues for reproducible non-security bugs. For vulnerabilities, follow [SECURITY.md](../.github/SECURITY.md).
