# Limitations

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](limitations.vi.md)

This page states the boundaries of the public v1.0.0 release so users can distinguish verified behavior from unsupported assumptions.

## Hardware scope

The verified accelerator target is:

```text
NVIDIA T4 ×2
```

Other accelerators may work with future changes, but they are not part of the v1.0.0 qualified claim.

## Precision and placement

The verified configuration uses FP16 across two T4 GPUs with no CPU or disk offload.

The project does not claim equivalence for alternative quantization, precision, or placement strategies.

## Kaggle-specific public path

The public notebook expects the named Kaggle model and Qdrant snapshot inputs. The general runtime is reusable, but the published end-to-end evidence is Kaggle-specific.

## API scope

The FastAPI service:

- binds only to loopback hosts;
- uses bearer-token authentication;
- has bounded queues/batches;
- does not provide Internet-facing hardening;
- does not claim HA, autoscaling, SLA, or multi-tenant isolation.

## Evaluation scope

The semantic 36/36 result applies to explicitly frozen public retrieval paths over the documented corpus configuration.

The visual 32/32 result applies to a controlled four-image gallery with documented transforms.

Neither number is a universal model-accuracy benchmark.

## Dataset/model ownership

MIT licensing covers repository source code only. Model weights and data remain under upstream licensing/terms.

## Resource variability

Kaggle runtime capacity, package base images, and platform availability can change. Strict preflight exists to catch incompatible environments instead of silently weakening the run.

## Internet behavior

The current public notebook workflow requires Internet ON overall, while the model runtime itself is forced into local/offline loading after setup.

## Support model

This is a community engineering project. Support is best-effort; there is no commercial SLA.

## Related documents

- [Features](features.md)
- [Kaggle Guide](kaggle.md)
- [REST API](api.md)
- [Retrieval and Evaluation](retrieval-and-evaluation.md)
