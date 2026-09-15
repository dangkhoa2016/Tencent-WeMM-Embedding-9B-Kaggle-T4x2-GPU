# REST API

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](api.vi.md)

The project includes a FastAPI embedding service for **loopback/local notebook use**.

## Start the service

```bash
export WEMM_API_TOKEN='replace-with-a-random-token-at-least-32-characters'
bash kaggle/setup-api.sh
bash kaggle/run-api.sh
```

Default address:

```text
http://127.0.0.1:8090
```

The configured host must be one of `127.0.0.1`, `localhost`, or `::1`.

## Authentication

Embedding endpoints require:

```text
Authorization: Bearer <token>
```

`WEMM_API_TOKEN` must contain at least **32 characters**.

Health/readiness endpoints do not require the embedding bearer dependency.

## Endpoints

### `GET /healthz`

Returns service health and current service phase.

### `GET /readyz`

Returns HTTP 200 only when the scheduler/runtime is ready. Otherwise it returns HTTP 503.

### `POST /v1/embeddings/text`

JSON body:

```json
{
  "inputs": [
    "Hồ Gươm nằm ở trung tâm Hà Nội.",
    "Hoan Kiem Lake is in central Hanoi."
  ],
  "dimension": 1024
}
```

### `POST /v1/embeddings/image`

Multipart form fields:

- `images`: one or more uploaded images;
- `dimension`: 4096 or 1024.

### `POST /v1/embeddings/image-text`

Multipart form fields:

- `images`;
- `texts`;
- `dimension`.

The number of images must equal the number of texts.

## Limits

Defaults from the public API configuration:

| Setting | Default |
| --- | ---: |
| Max batch | 4 |
| Queue max items | 16 |
| Request timeout | 30 s |
| Max text length | 8192 chars/item |
| Max image bytes | 8 MiB |
| Max image edge | 4096 px |
| Max image pixels | 16,777,216 |
| Per-GPU memory target | 14,200 MiB |

## Response shape

Embedding responses include:

- `request_id`;
- model name;
- workload;
- dimension;
- item count;
- embedding vectors;
- queue, inference, and total timing.

## Error model

The service maps common runtime conditions to explicit responses:

| Condition | HTTP |
| --- | ---: |
| Not ready | 503 |
| Queue full | 503 |
| Request timeout | 504 |
| Inference failed | 500 |
| CUDA OOM | 503 |
| Image too large | 413 |
| Unsupported image | 415 |
| Invalid image | 422 |
| Invalid dimension/batch/input | 422 |

Queue-full responses include a `Retry-After` header.

## Example: text request

```bash
curl -X POST http://127.0.0.1:8090/v1/embeddings/text \
  -H "Authorization: Bearer $WEMM_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"inputs":["xin chào","hello"],"dimension":1024}'
```

## Security boundary

The service is intentionally loopback-bound. It is not claimed to provide Internet-facing hardening, HA, multi-tenant isolation, or an SLA.

See [Security Policy](../.github/SECURITY.md) and [Limitations](limitations.md).
