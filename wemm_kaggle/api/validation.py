from __future__ import annotations

import io

from fastapi import UploadFile
from PIL import Image

ALLOWED_IMAGE_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})

Image.MAX_IMAGE_PIXELS = None


class ImageValidationError(Exception):
    error_code = "INVALID_IMAGE"


class ImageTooLargeError(ImageValidationError):
    error_code = "IMAGE_TOO_LARGE"


class UnsupportedImageError(ImageValidationError):
    error_code = "UNSUPPORTED_IMAGE"


class InvalidImageError(ImageValidationError):
    error_code = "INVALID_IMAGE"


async def decode_upload_image(
    upload: UploadFile,
    *,
    max_image_bytes: int = 8 * 1024 * 1024,
    max_image_edge: int = 4096,
    max_image_pixels: int = 16_777_216,
) -> Image.Image:
    data = await upload.read(max_image_bytes + 1)
    if len(data) > max_image_bytes:
        raise ImageTooLargeError(
            f"image exceeds the {max_image_bytes}-byte limit"
        )
    try:
        image = Image.open(io.BytesIO(data))
    except Exception as exc:
        raise InvalidImageError("image bytes could not be opened") from exc
    fmt = (image.format or "").upper()
    if fmt not in ALLOWED_IMAGE_FORMATS:
        raise UnsupportedImageError(f"image format {fmt or 'unknown'} is not allowed")
    width, height = image.size
    if width > max_image_edge or height > max_image_edge:
        raise ImageTooLargeError(
            f"image geometry {width}x{height} exceeds the {max_image_edge}-pixel edge limit"
        )
    if width * height > max_image_pixels:
        raise ImageTooLargeError(
            f"image pixel count {width * height} exceeds the {max_image_pixels}-pixel limit"
        )
    try:
        image.load()
    except Exception as exc:
        raise InvalidImageError("image content could not be decoded") from exc
    return image.convert("RGB").copy()