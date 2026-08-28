from types import SimpleNamespace

import pytest
import torch


class FakeBatch(dict):
    def __init__(self, batch_size):
        super().__init__(
            input_ids=torch.ones((batch_size, 3), dtype=torch.int64),
            attention_mask=torch.ones((batch_size, 3), dtype=torch.int64),
        )
        self.to_calls = []

    def to(self, device):
        self.to_calls.append(device)
        return self


class FakeProcessor:
    def __init__(self):
        self.template_calls = []
        self.processor_calls = []
        self.last_batch = None

    def apply_chat_template(self, conversation, **kwargs):
        self.template_calls.append((conversation, kwargs))
        return f"rendered-{len(self.template_calls)}"

    def __call__(self, **kwargs):
        self.processor_calls.append(kwargs)
        text = kwargs["text"]
        batch_size = len(text) if isinstance(text, list) else 1
        if batch_size > 1 and kwargs.get("padding") is not True:
            raise ValueError(
                "Unable to convert output 'input_ids' (type: list) to tensor: "
                "expected all sequences to share a length; pass padding=True"
            )
        self.last_batch = FakeBatch(batch_size)
        return self.last_batch


class FakeModel:
    def __init__(self):
        self.embedding_calls = []

    def embedding(self, **inputs):
        self.embedding_calls.append(inputs)
        batch = inputs["input_ids"].shape[0]
        values = torch.arange(1, 4097, dtype=torch.float32).repeat(batch, 1)
        return torch.nn.functional.normalize(values, dim=-1)


def make_runtime():
    from wemm_kaggle.runtime import WeMMRuntime

    processor = FakeProcessor()
    model = FakeModel()
    vision_calls = []

    def process_vision_info(conversations, **kwargs):
        vision_calls.append((conversations, kwargs))
        images = []
        for conversation in conversations:
            for item in conversation[0]["content"]:
                if item["type"] == "image":
                    images.append(item["image"])
        return (images or None), None, {}

    runtime = WeMMRuntime(
        model_dir=None,
        identity=None,
        processor=processor,
        model=model,
        process_vision_info=process_vision_info,
        torch=torch,
        device_map_report=SimpleNamespace(),
        input_device="cpu",
        allowed_dimensions=(64, 128, 256, 512, 1024, 2048, 4096),
    )
    return runtime, processor, model, vision_calls


def test_embed_texts_batches_conversations_into_one_model_call():
    runtime, processor, model, vision_calls = make_runtime()

    out = runtime.embed_texts(["hello", "xin chao"])

    assert out.shape == (2, 4096)
    assert len(processor.template_calls) == 2
    assert processor.template_calls[0][0][0]["content"] == [{"type": "text", "text": "hello"}]
    assert processor.template_calls[1][0][0]["content"] == [{"type": "text", "text": "xin chao"}]
    assert len(vision_calls) == 1
    assert len(vision_calls[0][0]) == 2
    assert processor.processor_calls[-1]["text"] == ["rendered-1", "rendered-2"]
    assert processor.processor_calls[-1]["return_tensors"] == "pt"
    assert processor.last_batch.to_calls == ["cpu"]
    assert len(model.embedding_calls) == 1


def test_batched_runtime_enables_dynamic_padding_for_variable_length_inputs():
    runtime, processor, model, _ = make_runtime()

    out = runtime.embed_texts(["first fixture", "a much longer second fixture"])

    assert out.shape == (2, 4096)
    call = processor.processor_calls[-1]
    assert call["padding"] is True
    assert call["return_tensors"] == "pt"
    assert call["text"] == ["rendered-1", "rendered-2"]
    assert len(model.embedding_calls) == 1


def test_embed_images_and_image_texts_preserve_image_before_text():
    runtime, processor, model, vision_calls = make_runtime()
    image_a = object()
    image_b = object()

    image_out = runtime.embed_images([image_a, image_b], dimension=1024)
    assert image_out.shape == (2, 1024)
    assert vision_calls[-1][0][0][0]["content"] == [{"type": "image", "image": image_a}]
    assert vision_calls[-1][0][1][0]["content"] == [{"type": "image", "image": image_b}]

    multimodal_out = runtime.embed_image_texts([(image_a, "first"), (image_b, "second")])
    assert multimodal_out.shape == (2, 4096)
    first_content = vision_calls[-1][0][0][0]["content"]
    assert first_content == [
        {"type": "image", "image": image_a},
        {"type": "text", "text": "first"},
    ]
    assert len(model.embedding_calls) == 2


def test_batch_apis_reject_empty_batches():
    runtime, _, _, _ = make_runtime()

    with pytest.raises(ValueError, match="batch"):
        runtime.embed_texts([])
    with pytest.raises(ValueError, match="batch"):
        runtime.embed_images([])
    with pytest.raises(ValueError, match="batch"):
        runtime.embed_image_texts([])


def test_single_item_apis_delegate_to_batch_methods(monkeypatch):
    runtime, _, _, _ = make_runtime()
    calls = []
    sentinel = object()

    def fake_texts(items, dimension=None):
        calls.append(("text", items, dimension))
        return sentinel

    def fake_images(items, dimension=None):
        calls.append(("image", items, dimension))
        return sentinel

    def fake_image_texts(items, dimension=None):
        calls.append(("image_text", items, dimension))
        return sentinel

    monkeypatch.setattr(runtime, "embed_texts", fake_texts)
    monkeypatch.setattr(runtime, "embed_images", fake_images)
    monkeypatch.setattr(runtime, "embed_image_texts", fake_image_texts)

    image = object()
    assert runtime.embed_text("x", 1024) is sentinel
    assert runtime.embed_image(image, 4096) is sentinel
    assert runtime.embed_image_text(image, "describe", 1024) is sentinel
    assert calls == [
        ("text", ["x"], 1024),
        ("image", [image], 4096),
        ("image_text", [(image, "describe")], 1024),
    ]
