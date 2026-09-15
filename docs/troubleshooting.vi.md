# Troubleshooting

> 🌐 Language / Ngôn ngữ: [English](troubleshooting.md) | **Tiếng Việt**

Tài liệu này tổng hợp các lỗi thường gặp trên public Kaggle/runtime/API path.

## Kaggle không có T4 ×2

Verified public path yêu cầu **GPU T4 ×2**. Nếu Kaggle cấp accelerator khác hoặc không có GPU, không coi run đó là qualification evidence tương đương.

Cách xử lý:

- dừng run;
- chọn T4 ×2 khi có;
- chạy lại preflight.

## Không tìm thấy model input

Public integration resolve model dưới:

```text
/kaggle/input
```

Kiểm tra model input đã attach và có đủ WeMM files. Chỉ dùng `KAGGLE_MODEL_DIR` khi automatic discovery bị mơ hồ.

Không trỏ public runtime tới writable directory tùy ý ngoài Kaggle input boundary.

## Thiếu hoặc không đủ safetensors

Runtime từ chối incomplete model directory.

Kiểm tra model có complete safetensors weights và các configuration/tokenizer files được mô tả trong [Model và Data](model-and-data.vi.md).

## Setup cố cài Torch/CUDA stack khác

Qualified setup chủ ý tái sử dụng system PyTorch của Kaggle.

Nếu `requirements-kaggle.txt` chứa Torch, Triton, CUDA hoặc NVIDIA runtime packages, setup phải fail. Hãy bỏ các runtime specification đó thay vì ép cài.

## Preflight báo GPU placement failure

Verified path yêu cầu đủ hai GPU IDs và cấm CPU/disk offload.

Placement mismatch có thể do:

- accelerator sai;
- GPU memory không đủ;
- model/runtime version drift;
- device-map behavior thay đổi.

Không bypass validator.

## Embedding dimension không hỗ trợ

Public supported dimensions:

```text
4096
1024
```

Dimension khác phải bị reject.

## Qdrant snapshot restore thất bại

Kiểm tra:

- snapshot dataset đã attach;
- expected collection/snapshot files tồn tại;
- writable disk đủ dung lượng;
- stale mutable copy không conflict với restore.

Nếu working storage bị dirty, xóa disposable working copy rồi restore lại từ verified read-only input.

## Qdrant point count hoặc collection name mismatch

Public semantic authority kỳ vọng exact collection names đã document và **99.967 points** mỗi collection.

Mismatch phải được coi là failed qualification check, không tiếp tục bằng corpus khác.

## API không start

Nguyên nhân thường gặp:

- `WEMM_API_TOKEN` ngắn hơn 32 ký tự;
- `WEMM_API_HOST` không phải loopback;
- base Kaggle venv chưa được tạo;
- model runtime không load được;
- port đang bị chiếm.

Default:

```text
127.0.0.1:8090
```

## API trả 503 NOT_READY

Runtime worker có thể vẫn đang initialize. Kiểm tra `/readyz` và server logs.

Nếu NOT_READY kéo dài, initialization chưa hoàn tất thành công.

## API trả QUEUE_FULL hoặc REQUEST_TIMEOUT

API có bounded concurrency. Queue mặc định 16 items, request timeout 30 giây.

Giảm client concurrency hoặc đợi rồi retry.

## CUDA OOM

Không giảm correctness checks hoặc âm thầm chuyển CPU trong qualified path.

Kiểm tra:

- GPU process khác;
- stale model workers;
- model/runtime versions thay đổi;
- per-GPU memory config thay đổi.

Có thể restart từ fresh Kaggle session.

## Disk usage tăng liên tục

Demo phải reuse/restore verified storage, không tạo durable corpus copy mới ở mỗi run.

Kiểm tra writable working directories xem có stale Qdrant restore hoặc duplicated temporary data. Giữ read-only input authority và xóa disposable dirty copy trước khi restore lại.

## Báo lỗi ở đâu?

Dùng GitHub Issues cho reproducible non-security bugs. Với vulnerability, xem [SECURITY.md](../.github/SECURITY.vi.md).
