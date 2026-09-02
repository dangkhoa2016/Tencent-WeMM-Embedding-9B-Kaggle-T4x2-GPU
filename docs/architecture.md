# Architecture

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](architecture.vi.md)

The project integrates the upstream **WeMM-Embedding-9B** model, developed by the WeChat Vision Team at Tencent, and separates reusable model runtime code from public-notebook presentation and Kaggle-specific orchestration.

## High-level flow

```mermaid
flowchart LR
    A["Text / Image / Image + Text"] --> B["wemm_runtime"]
    B --> C["WeMM-Embedding-9B"]
    G0["NVIDIA T4 GPU 0"] --> C
    G1["NVIDIA T4 GPU 1"] --> C
    C --> D["4096d / 1024d normalized vectors"]
    D --> E["wemm_kaggle"]
    E --> F["Qdrant retrieval"]
    E --> H["Loopback FastAPI"]
    E --> I["Kaggle public demo"]
```

## Layers

### `wemm_runtime/`

Reusable WeMM-Embedding-9B runtime primitives live here. Responsibilities include:

- validating the local model directory and required safetensors;
- enforcing offline/local model loading once the runtime begins;
- loading the processor and upstream WeMM-Embedding-9B custom model code;
- validating the Hugging Face device map;
- enforcing supported Matryoshka dimensions;
- generating normalized text, image, and image+text embeddings.

### `wemm_notebook/`

Public-notebook presentation and phase orchestration live here. It owns the release-source bootstrap, the five-cell phase runner, human-readable showcase presentation, and closeout presentation while keeping the frozen science/runtime authority separate.

### `wemm_kaggle/`

Kaggle integration lives here. It handles:

- resolving model files under `/kaggle/input`;
- preflight and environment checks;
- subprocess runtime lifecycle;
- Qdrant snapshot discovery/reuse;
- public demo orchestration;
- REST API configuration, validation, scheduling, and observability.

### `kaggle/`

Shell entry points create and validate the qualified runtime environment. The setup deliberately reuses Kaggle's existing CUDA-enabled PyTorch rather than installing a second Torch/CUDA stack.

## Dual-T4 placement

The verified FP16 configuration maps **37 model modules**:

| GPU | Modules |
| --- | ---: |
| GPU 0 | 13 |
| GPU 1 | 24 |

The qualified path allows neither CPU nor disk offload. Device-map validation rejects unsupported placement.

## Runtime isolation

The public Kaggle notebook/demo executes model workloads through the isolated `SubprocessEmbeddingWorker` so GPU state can be reclaimed predictably at demo closeout.

The loopback FastAPI service owns an in-process model runtime behind a single-threaded `InferenceScheduler` (`ThreadPoolExecutor(max_workers=1)` with `load_local_runtime(...)`). Its service lifecycle is separate from the notebook/demo subprocess worker.

## Embedding path

The runtime constructs multimodal chat-style inputs, runs the WeMM-Embedding-9B embedding path, pools the final hidden representation, and L2-normalizes the vector. For Matryoshka output, the vector is truncated to the requested supported dimension and normalized again.

Supported public dimensions:

- 4096
- 1024

## Retrieval path

Semantic retrieval uses verified Qdrant collections with named English/Vietnamese vectors. Visual robustness uses a separate temporary four-image gallery. The two spaces intentionally remain distinct; see [Retrieval and Evaluation](retrieval-and-evaluation.md).

## API path

The loopback API wraps the same runtime behind a bounded scheduler (in-process, not the notebook/demo subprocess worker):

```text
HTTP request
  -> authentication / validation
  -> bounded queue
  -> in-process scheduler worker
  -> normalized embedding
  -> structured response + timing
```

The service is local/notebook-oriented, not an Internet-facing multi-tenant serving architecture.

## Related documents

- [Python Runtime](python-runtime.md)
- [REST API](api.md)
- [Qdrant](qdrant.md)
- [Reproducibility](reproducibility.md)
- [Limitations](limitations.md)
