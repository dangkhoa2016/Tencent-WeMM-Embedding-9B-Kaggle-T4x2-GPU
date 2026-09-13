from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORK = Path("/kaggle/working")
RUN_ROOT = WORK / "tencent-wemm-embedding-9b-public-demo-run"
EVID = RUN_ROOT / "evidence"
QDRANT_ROOT = WORK / "wemm-qdrant"
QDRANT_SEAL = WORK / "tencent-wemm-embedding-9b-public-demo-qdrant-cache-seal.json"

DATASET = Path(
    "/kaggle/input/datasets/dangkhoa2016/"
    "wemm-embedding-9b-v1-qdrant-snapshots"
)
MODEL_DIR = Path(
    "/kaggle/input/models/dangkhoa2016/"
    "tencent-wemm-embedding-9b/transformers/default/1"
)

COLLECTIONS = {
    4096: "wikidata_en_vi_wemm9b_4096_v030_rc2_d819dc7_v1",
    1024: "wikidata_en_vi_wemm9b_1024_v030_rc2_d819dc7_v1",
}
SNAPSHOTS = {
    4096: {
        "filename": (
            "wikidata_en_vi_wemm9b_4096_v030_rc2_d819dc7_v1-"
            "5812074853379615-2026-09-10-10-42-46.snapshot"
        ),
        "bytes": 3351233024,
        "sha256": "7d2f7324a4d85dfcbfd4cff980115d381a7edc6771cb2d3042c10130e1786b4f",
    },
    1024: {
        "filename": (
            "wikidata_en_vi_wemm9b_1024_v030_rc2_d819dc7_v1-"
            "5812074853379615-2026-09-10-10-42-47.snapshot"
        ),
        "bytes": 891202560,
        "sha256": "da59ce4f8d7fef70488a386390ae84de1cb72a64a52dccd5344ba1c01dad75cc",
    },
}
EXPECTED_POINTS = 99967
TOP_K = 20
IMAGE_MAX_EDGE = 1600
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024
USER_AGENT = (
    "Tencent-WeMM-Embedding-9B/1.0 "
    "(https://github.com/dangkhoa2016/"
    "Tencent-WeMM-Embedding-9B-Kaggle-T4x2-GPU; public demo)"
)
