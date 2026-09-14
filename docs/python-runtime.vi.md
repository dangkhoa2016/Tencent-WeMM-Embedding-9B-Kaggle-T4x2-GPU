# Python Runtime

> 🌐 Language / Ngôn ngữ: [English](python-runtime.md) | **Tiếng Việt**

Reusable embedding runtime nằm trong `wemm_runtime/`. Đây là nơi chứa logic validate model, load model, tạo embedding, normalize vector và kiểm tra device map mà Kaggle integration sử dụng.

## Workload được hỗ trợ

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
    [(image, "Một công trình lịch sử bên hồ")],
    dimension=1024,
)
```

Các batch method:

- `embed_texts(...)`
- `embed_images(...)`
- `embed_image_texts(...)`

## Model validation

Trước khi load, runtime yêu cầu:

- `config.json`;
- `modeling_wemm_embedding.py`;
- `processor_config.json`;
- `tokenizer.json`;
- `tokenizer_config.json`;
- safetensors weights đầy đủ.

Model config phải expose WeMM custom code và advertise Matryoshka dimension 4096.

## Offline loading

Khi offline mode bật, runtime đặt:

```text
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1
```

Processor/model được load bằng `local_files_only=True`.

## Device map

Hugging Face device map sau khi load được validate theo GPU IDs mong đợi. Public path đã xác minh yêu cầu đủ hai GPU và từ chối CPU/offload targets.

Input device ưu tiên được suy ra từ placement của embedding token trong device map.

## Chuẩn hóa embedding

Runtime pool representation cuối tại effective end-of-sequence position rồi L2-normalize.

Với Matryoshka dimension yêu cầu, embedding được truncate và normalize lại. Public release hỗ trợ:

```text
4096
1024
```

Dimension không hỗ trợ sẽ fail rõ ràng.

## Tương thích dual-GPU

Khi phát hiện multi-GPU device map, runtime cài một embedding-dispatch compatibility path nhỏ quanh WeMM model để embedding call đi theo placement đã validate.

## Resource lifecycle

Kaggle integration bọc runtime trong subprocess worker. Nếu dùng trực tiếp `wemm_runtime/`, người tích hợp phải tự quản lý lifecycle model/process.

## Tài liệu liên quan

- [Architecture](architecture.vi.md)
- [Hướng dẫn Kaggle](kaggle.vi.md)
- [REST API](api.vi.md)
- [Reproducibility](reproducibility.vi.md)
