from __future__ import annotations

import asyncio
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from ..model_resolver import resolve_model_dir
from ..runtime import load_local_runtime
from .auth import build_auth_dependency
from .config import ApiSettings
from .logging import emit_event
from .scheduler import (
    InferenceFailedError,
    InferenceOomError,
    InferenceResult,
    InferenceScheduler,
    QueueFullError,
    RequestTimeoutError,
    ServiceNotReadyError,
)
from .schemas import EmbeddingItem, EmbeddingResponse, TextEmbeddingRequest, TimingMs
from .state import ServicePhase
from .validation import (
    ImageTooLargeError,
    InvalidImageError,
    UnsupportedImageError,
    decode_upload_image,
)

MODEL_NAME = "Tencent/WeMM-Embedding-9B"
ALLOWED_DIMENSIONS = (4096, 1024)


def _default_runtime_loader(settings: ApiSettings) -> Callable[[], Any]:
    def load() -> Any:
        model_dir = resolve_model_dir(
            Path("/kaggle/input"),
            os.environ.get("KAGGLE_MODEL_DIR"),
        )
        return load_local_runtime(model_dir, per_gpu_mib=settings.per_gpu_mib)

    return load


def _error_handler(code: str, status: int, *, retry_after: str | None = None):
    async def handler(request: Request, exc: Exception) -> JSONResponse:
        headers = {}
        if retry_after is not None:
            headers["Retry-After"] = retry_after
        payload = {"error": code, "message": str(exc) or code}
        request_id = getattr(request.state, "request_id", None)
        if request_id:
            payload["request_id"] = request_id
        return JSONResponse(status_code=status, content=payload, headers=headers)

    return handler


def build_response(
    request_id: str,
    workload: str,
    dimension: int,
    result: InferenceResult,
) -> EmbeddingResponse:
    data = [
        EmbeddingItem(index=index, embedding=list(vec))
        for index, vec in enumerate(result.embeddings)
    ]
    return EmbeddingResponse(
        request_id=request_id,
        model=MODEL_NAME,
        workload=workload,
        dimension=dimension,
        count=len(data),
        data=data,
        timing_ms=TimingMs(
            queue=round(result.queue_ms, 3),
            inference=round(result.inference_ms, 3),
            total=round(result.queue_ms + result.inference_ms, 3),
        ),
    )


def _check_batch_size(count: int, limit: int) -> None:
    if count < 1 or count > limit:
        raise HTTPException(status_code=422, detail=f"batch size must be in 1..{limit}; got {count}")


def _check_text_limits(texts: list[str], limit: int) -> None:
    if any((not text) or len(text) > limit for text in texts):
        raise HTTPException(status_code=422, detail=f"each text must contain 1..{limit} characters")


def create_app(
    settings: ApiSettings,
    runtime_loader: Callable[[], Any] | None = None,
) -> FastAPI:
    if runtime_loader is None:
        runtime_loader = _default_runtime_loader(settings)

    service = InferenceScheduler(
        runtime_loader,
        queue_max_items=settings.queue_max_items,
        request_timeout_s=settings.request_timeout_s,
        evidence_dir=settings.evidence_dir,
    )
    require_auth = build_auth_dependency(settings.api_token)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        emit_event("service_start")
        init_task = asyncio.create_task(service.initialize())
        app.state.init_task = init_task
        yield
        await service.shutdown()

    app = FastAPI(title="Tencent WeMM-Embedding-9B REST API", lifespan=lifespan)
    app.state.settings = settings
    app.state.scheduler = service

    app.add_exception_handler(ServiceNotReadyError, _error_handler("NOT_READY", 503))
    app.add_exception_handler(QueueFullError, _error_handler("QUEUE_FULL", 503, retry_after="1"))
    app.add_exception_handler(RequestTimeoutError, _error_handler("REQUEST_TIMEOUT", 504))
    app.add_exception_handler(InferenceFailedError, _error_handler("INFERENCE_FAILED", 500))
    app.add_exception_handler(InferenceOomError, _error_handler("CUDA_OOM", 503))
    app.add_exception_handler(ImageTooLargeError, _error_handler("IMAGE_TOO_LARGE", 413))
    app.add_exception_handler(UnsupportedImageError, _error_handler("UNSUPPORTED_IMAGE", 415))
    app.add_exception_handler(InvalidImageError, _error_handler("INVALID_IMAGE", 422))

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        emit_event(
            "request_received",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )
        service.note_request()
        try:
            response = await call_next(request)
        except Exception as exc:
            service.note_rejected()
            emit_event("request_rejected", request_id=request_id, error_type=type(exc).__name__)
            raise
        response.headers["X-Request-ID"] = request_id
        emit_event("request_complete", request_id=request_id, status=response.status_code)
        return response

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "state": service.phase.value}

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        if service.phase != ServicePhase.READY:
            payload = {"status": "not_ready", "state": service.phase.value}
            return JSONResponse(status_code=503, content=payload)
        return JSONResponse(status_code=200, content={"status": "ready", "state": "READY"})

    @app.post("/v1/embeddings/text", response_model=EmbeddingResponse)
    async def embed_text(
        request: Request,
        body: TextEmbeddingRequest,
        _auth: None = Depends(require_auth),
    ) -> EmbeddingResponse:
        _check_batch_size(len(body.inputs), settings.max_batch)
        _check_text_limits(body.inputs, settings.max_text_chars)
        result = await service.submit(
            "text",
            body.inputs,
            body.dimension,
            timeout_s=settings.request_timeout_s,
        )
        return build_response(request.state.request_id, "text", body.dimension, result)

    @app.post("/v1/embeddings/image", response_model=EmbeddingResponse)
    async def embed_image(
        request: Request,
        images: list[UploadFile] = File(...),
        dimension: int = Form(4096),
        _auth: None = Depends(require_auth),
    ) -> EmbeddingResponse:
        _check_dimension(dimension)
        _check_batch_size(len(images), settings.max_batch)
        decoded = [
            await decode_upload_image(
                upload,
                max_image_bytes=settings.max_image_bytes,
                max_image_edge=settings.max_image_edge,
                max_image_pixels=settings.max_image_pixels,
            )
            for upload in images
        ]
        result = await service.submit(
            "image",
            decoded,
            dimension,
            timeout_s=settings.request_timeout_s,
        )
        return build_response(request.state.request_id, "image", dimension, result)

    @app.post("/v1/embeddings/image-text", response_model=EmbeddingResponse)
    async def embed_image_text(
        request: Request,
        images: list[UploadFile] = File(...),
        texts: list[str] = Form(...),
        dimension: int = Form(4096),
        _auth: None = Depends(require_auth),
    ) -> EmbeddingResponse:
        _check_dimension(dimension)
        _check_batch_size(len(images), settings.max_batch)
        if len(images) != len(texts):
            raise HTTPException(
                status_code=422,
                detail="the number of images must equal the number of texts",
            )
        _check_text_limits(texts, settings.max_text_chars)
        decoded = [
            await decode_upload_image(
                upload,
                max_image_bytes=settings.max_image_bytes,
                max_image_edge=settings.max_image_edge,
                max_image_pixels=settings.max_image_pixels,
            )
            for upload in images
        ]
        items = [(image, text) for image, text in zip(decoded, texts)]
        result = await service.submit(
            "image_text",
            items,
            dimension,
            timeout_s=settings.request_timeout_s,
        )
        return build_response(request.state.request_id, "image_text", dimension, result)

    return app


def _check_dimension(dimension: int) -> None:
    if dimension not in ALLOWED_DIMENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"dimension must be one of {list(ALLOWED_DIMENSIONS)}",
        )