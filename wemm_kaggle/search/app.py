from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .schemas import SearchRequest, SearchResponse
from .service import SearchDependencyError, SearchService


def build_app(service: SearchService) -> FastAPI:
    app = FastAPI(title="wemm-search-gateway", version="0.3.0-rc1")

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> dict:
        result = service.ready()
        if not result["ready"]:
            return JSONResponse(status_code=503, content=result)
        return result

    @app.post("/v1/search", response_model=SearchResponse)
    def search(payload: SearchRequest, request: Request) -> SearchResponse:
        request_id = request.headers.get("x-request-id", "local")
        try:
            return service.search(payload, request_id=request_id)
        except SearchDependencyError as exc:
            raise HTTPException(status_code=503, detail=str(exc))

    return app
