# Python Runtime

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](python-runtime.vi.md)

The reusable embedding runtime lives in `wemm_runtime/`. It contains the model validation, loading, embedding, normalization, and device-map logic used by the Kaggle integration.

## Supported workloads

```python
vector = runtime.embed_text(
    "Hồ Gươm nằm ở trung tâm Hà Nội.",
    dimension=1024,
)

image_vector = runtime.embed_image(
    image,
    dimension=4096,
)

multimodal_vector = runtime.embed_image_texts(
    [(image, "A historic building beside a lake")],
    dimension=1024,
)
```

Batch methods are also available:

- `embed_texts(...)`
- `embed_images(...)`
- `embed_image_texts(...)`

## Model validation

Before loading, the runtime requires:

- `config.json`;
- `modeling_wemm_embedding.py`;
- `processor_config.json`;
- `tokenizer.json`;
- `tokenizer_config.json`;
- complete safetensors weights.

The WeMM-Embedding-9B model configuration must expose the required upstream custom model code and advertise Matryoshka dimension 4096.

## Offline loading

When offline mode is enabled, the runtime sets:

```text
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1
```

The processor and model are then loaded with `local_files_only=True`.

## Device map

The loaded Hugging Face device map is validated against the expected GPU IDs. The verified public path requires both GPUs and rejects CPU/offload targets.

The preferred input device is derived from the embedding-token placement in the device map.

## Embedding normalization

The runtime pools the final hidden representation at the effective end-of-sequence position and L2-normalizes it.

For a requested Matryoshka dimension, the embedding is truncated and normalized again. The public release supports:

```text
4096
1024
```

Unsupported dimensions fail explicitly.

## Dual-GPU compatibility

When a multi-GPU device map is detected, the runtime installs a small embedding dispatch compatibility path around the loaded WeMM-Embedding-9B model so the embedding call follows the validated device placement.

## Resource lifecycle

The public notebook/demo integration wraps the runtime in a `SubprocessEmbeddingWorker` so GPU state can be reclaimed predictably at demo closeout. Direct users of `wemm_runtime/` are responsible for their own model/process lifecycle if they bypass that integration.

The loopback FastAPI service currently owns an in-process model runtime behind a single-threaded `InferenceScheduler`; it does not route API inference through the notebook/demo subprocess worker.

## Related documents

- [Architecture](architecture.md)
- [Kaggle Guide](kaggle.md)
- [REST API](api.md)
- [Reproducibility](reproducibility.md)
