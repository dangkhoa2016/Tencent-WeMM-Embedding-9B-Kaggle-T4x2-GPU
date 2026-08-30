#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import uvicorn

from wemm_kaggle.search.app import build_app
from wemm_kaggle.search.config import SearchSettings
from wemm_kaggle.search.service import SearchService


def main() -> int:
    settings = SearchSettings.from_mapping(dict(os.environ))
    token = settings.embedding_token or os.environ.get("WEMM_API_TOKEN", "")
    service = SearchService.from_env(
        api_token=token,
        embedding_url=settings.embedding_url,
        qdrant_host=settings.qdrant_host,
        qdrant_port=settings.qdrant_port,
        collection_4096=settings.collections_4096,
        collection_1024=settings.collections_1024,
    )
    app = build_app(service)
    uvicorn.run(
        app,
        host=settings.search_host,
        port=settings.search_port,
        workers=1,
        reload=False,
        access_log=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
