# REST API

> 🌐 Language / Ngôn ngữ: [English](api.md) | **Tiếng Việt**

Dự án có FastAPI embedding service dành cho **loopback/local notebook use**.

## Khởi động service

```bash
export WEMM_API_TOKEN='thay-bang-token-ngau-nhien-it-nhat-32-ky-tu'
bash kaggle/setup-api.sh
bash kaggle/run-api.sh
```

Địa chỉ mặc định:

```text
http://127.0.0.1:8090
```

Configured host phải thuộc `127.0.0.1`, `localhost` hoặc `::1`.

## Authentication

Embedding endpoints yêu cầu:

```text
Authorization: Bearer <token>
```

`WEMM_API_TOKEN` phải dài ít nhất **32 ký tự**.

Health/readiness endpoints không dùng embedding bearer dependency.

## Endpoints

### `GET /healthz`

Trả về health và service phase hiện tại.

### `GET /readyz`

Chỉ trả HTTP 200 khi scheduler/runtime đã ready. Trường hợp khác trả HTTP 503.

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

Multipart form:

- `images`: một hoặc nhiều image uploads;
- `dimension`: 4096 hoặc 1024.

### `POST /v1/embeddings/image-text`

Multipart form:

- `images`;
- `texts`;
- `dimension`.

Số image phải bằng số text.

## Giới hạn

Default của public API configuration:

| Setting | Default |
| --- | ---: |
| Max batch | 4 |
| Queue max items | 16 |
| Request timeout | 30 s |
| Max text length | 8192 chars/item |
| Max image bytes | 8 MiB |
| Max image edge | 4096 px |
| Max image pixels | 16.777.216 |
| Per-GPU memory target | 14.200 MiB |

## Response shape

Embedding response gồm:

- `request_id`;
- model name;
- workload;
- dimension;
- item count;
- embedding vectors;
- queue, inference và total timing.

## Error model

Service map runtime condition sang response rõ ràng:

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

Queue-full response có `Retry-After` header.

## Ví dụ text request

```bash
curl -X POST http://127.0.0.1:8090/v1/embeddings/text \
  -H "Authorization: Bearer $WEMM_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"inputs":["xin chào","hello"],"dimension":1024}'
```

## Security boundary

Service chủ ý bind loopback. Dự án không tuyên bố Internet-facing hardening, HA, multi-tenant isolation hoặc SLA.

Xem [Security Policy](../.github/SECURITY.vi.md) và [Limitations](limitations.vi.md).
