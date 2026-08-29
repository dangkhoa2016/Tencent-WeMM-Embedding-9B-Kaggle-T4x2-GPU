import asyncio
import io

import pytest
from PIL import Image
from starlette.datastructures import UploadFile

from wemm_kaggle.api.schemas import TextEmbeddingRequest
from wemm_kaggle.api.validation import (
    ImageTooLargeError,
    InvalidImageError,
    UnsupportedImageError,
    decode_upload_image,
)

MAX_BYTES = 8 * 1024 * 1024


def image_bytes(fmt="PNG", size=(64, 64), color="red"):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format=fmt)
    return buf.getvalue()


def run(coro):
    return asyncio.run(coro)


def decode(data, filename="image.png", mime="image/png"):
    upload = UploadFile(file=io.BytesIO(data), filename=filename)
    return run(decode_upload_image(upload))


# --- text request schema ---


def test_text_batches_b1_b2_b4_accepted():
    for n in (1, 2, 4):
        model = TextEmbeddingRequest(inputs=[f"text {i}" for i in range(n)], dimension=4096)
        assert len(model.inputs) == n


def test_empty_text_batch_rejected():
    with pytest.raises(Exception):
        TextEmbeddingRequest(inputs=[], dimension=4096)


def test_b5_text_batch_rejected():
    with pytest.raises(Exception):
        TextEmbeddingRequest(inputs=[f"text {i}" for i in range(5)], dimension=4096)


def test_empty_string_text_rejected():
    with pytest.raises(Exception):
        TextEmbeddingRequest(inputs=[""], dimension=4096)


def test_overlong_8193_char_text_rejected():
    with pytest.raises(Exception):
        TextEmbeddingRequest(inputs=["x" * 8193], dimension=4096)


def test_8192_char_text_accepted():
    model = TextEmbeddingRequest(inputs=["x" * 8192], dimension=4096)
    assert len(model.inputs[0]) == 8192


def test_dimension_4096_and_1024_accepted():
    assert TextEmbeddingRequest(inputs=["a"], dimension=4096).dimension == 4096
    assert TextEmbeddingRequest(inputs=["a"], dimension=1024).dimension == 1024


@pytest.mark.parametrize("dimension", [0, 512, 2048, 8192])
def test_disallowed_dimension_rejected(dimension):
    with pytest.raises(Exception):
        TextEmbeddingRequest(inputs=["a"], dimension=dimension)


# --- image decoding ---


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP"])
def test_valid_png_jpeg_webp_accepted(fmt):
    mime = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}[fmt]
    decoded = decode(image_bytes(fmt=fmt), filename=f"a.{fmt.lower()}", mime=mime)
    assert decoded.size == (64, 64)


def test_malformed_bytes_rejected():
    with pytest.raises(InvalidImageError):
        decode(b"\x00\x01\x02\xffnot-an-image-at-all")


def test_oversize_bytes_rejected():
    data = b"\x00" * (MAX_BYTES + 1)
    with pytest.raises(ImageTooLargeError):
        decode(data)


def test_oversize_edge_rejected():
    data = image_bytes(size=(4097, 32))
    with pytest.raises(ImageTooLargeError):
        decode(data)


def test_oversize_pixel_count_rejected():
    data = image_bytes(size=(4800, 3500))
    assert 4800 * 3500 > 16_777_216
    with pytest.raises(ImageTooLargeError):
        decode(data)


def test_disallowed_image_format_rejected():
    data = image_bytes(fmt="GIF", size=(64, 64))
    with pytest.raises(UnsupportedImageError):
        decode(data)