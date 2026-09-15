# Hướng dẫn Kaggle

> 🌐 Language / Ngôn ngữ: [English](kaggle.md) | **Tiếng Việt**

Đây là đường chạy được khuyến nghị để end user tái lập public demo trên Kaggle.

## Yêu cầu

| Thành phần | Giá trị đã qualification |
| --- | --- |
| Platform | Kaggle Notebook |
| Accelerator | **GPU T4 ×2** |
| Internet | **ON** cho public demo hiện tại |
| Model input | `dangkhoa2016/tencent-wemm-embedding-9b`, version 1 |
| Dataset input | `dangkhoa2016/wemm-embedding-9b-v1-qdrant-snapshots`, version 1 |
| Precision | FP16 |
| Qdrant | 1.19.0 |
| Public vector sizes | 4096d, 1024d |

## Điểm bắt đầu khuyến nghị

Dùng:

[`notebooks/kaggle-production-demo-thin.ipynb`](../notebooks/kaggle-production-demo-thin.ipynb)

Bản Git chủ ý sạch và chưa executed. Kaggle Saved Version là execution evidence, không phải source authority.

## Trình tự setup

1. Tạo hoặc mở Kaggle Notebook.
2. Chọn **GPU T4 ×2**.
3. Bật Internet **ON** cho workflow công khai hiện tại.
4. Attach model input và Qdrant snapshot dataset ở bảng trên.
5. Import public notebook.
6. Chạy notebook từ trên xuống dưới.

Workflow thực hiện strict checks và phải dừng nếu hardware/input không đúng cấu hình.

## Lower-level setup scripts

Nếu muốn chạy trực tiếp bằng script:

```bash
bash kaggle/setup.sh
bash kaggle/run-preflight.sh
bash kaggle/run-acceptance.sh
```

Setup tạo dedicated virtual environment bằng `--system-site-packages` và chủ ý tái sử dụng system PyTorch của Kaggle. `requirements-kaggle.txt` không được phép cài thêm Torch/CUDA/NVIDIA runtime stack thứ hai.

## Offline runtime behavior

Sau setup, runtime scripts export:

```text
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1
```

Điều này có nghĩa model execution path dùng local Kaggle input files thay vì tải weights lúc inference. Tuy nhiên toàn bộ public notebook hiện vẫn cần Internet ON cho workflow demo hoàn chỉnh.

## Model discovery

Resolver kỳ vọng model content nằm dưới:

```text
/kaggle/input
```

Chỉ cần `KAGGLE_MODEL_DIR` khi automatic discovery bị mơ hồ. Public integration từ chối path nằm ngoài Kaggle input boundary.

## Notebook thực hiện gì?

Production-style notebook gồm:

1. host/input preflight;
2. kiểm tra Qdrant và snapshot reuse/restore;
3. khởi động model worker;
4. xác minh dual-T4 placement;
5. semantic text retrieval;
6. semantic image→text retrieval;
7. visual robustness retrieval;
8. scorecard summary;
9. shutdown worker/Qdrant.

## Disk hygiene

Public workflow được thiết kế để reuse verified data thay vì liên tục tạo persistent copy mới. Nếu working copy bị dirty hoặc cần restore, nên restore lại từ read-only source thay vì tích lũy nhiều mutable copy.

## Failure policy

Public path không âm thầm fallback sang CPU khi qualified GPU configuration được yêu cầu. Unsupported accelerator, thiếu model file, snapshot sai, device-map violation hoặc dimension không hỗ trợ phải fail rõ ràng.

## Tiếp theo

- [Troubleshooting](troubleshooting.vi.md)
- [Reproducibility](reproducibility.vi.md)
- [Qdrant](qdrant.vi.md)
