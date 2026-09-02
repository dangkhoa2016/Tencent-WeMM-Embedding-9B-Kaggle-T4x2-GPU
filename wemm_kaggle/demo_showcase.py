from __future__ import annotations

import hashlib
import io
from pathlib import Path
import time
from urllib.parse import quote
import urllib.request

from PIL import Image
from IPython.display import display
from qdrant_client import QdrantClient

from wemm_runtime import (
    FROZEN_IMAGE_SHOWCASE,
    FROZEN_TEXT_SHOWCASE,
    QdrantConfig,
    QdrantRetriever,
    evaluate_four_paths,
)

from .demo_config import (
    COLLECTIONS,
    DATASET,
    IMAGE_MAX_EDGE,
    MAX_DOWNLOAD_BYTES,
    RUN_ROOT,
    SNAPSHOTS,
    TOP_K,
    USER_AGENT,
)
from .demo_io import kv, run, sha256_file, truncate_vector


def verify_dataset() -> dict:
    entries = sorted(p.name for p in DATASET.iterdir() if p.is_file())
    if len(entries) != 9:
        raise RuntimeError(
            f"Expected exactly 9 public Dataset files, found {len(entries)}: {entries}"
        )
    manifest = DATASET / "PUBLIC-MANIFEST.sha256"
    if not manifest.is_file():
        raise RuntimeError("PUBLIC-MANIFEST.sha256 is missing")
    check = run(
        ["bash", "-lc", "sha256sum -c PUBLIC-MANIFEST.sha256"],
        cwd=DATASET,
        capture=True,
    )
    lines = [line.strip() for line in check.stdout.splitlines() if line.strip()]
    if len(lines) != 8 or not all(line.endswith(": OK") for line in lines):
        raise RuntimeError(f"Dataset checksum manifest failed: {lines}")
    verified = {line.rsplit(": OK", 1)[0] for line in lines}
    expected = set(entries) - {"PUBLIC-MANIFEST.sha256"}
    if verified != expected:
        raise RuntimeError(
            f"Dataset manifest coverage mismatch: verified={verified} expected={expected}"
        )
    snapshots = {}
    for dimension, spec in SNAPSHOTS.items():
        path = DATASET / spec["filename"]
        if not path.is_file() or path.stat().st_size != spec["bytes"]:
            raise RuntimeError(f"Snapshot size contract failed: {path}")
        digest = sha256_file(path)
        if digest != spec["sha256"]:
            raise RuntimeError(
                f"Snapshot SHA mismatch {dimension}: {digest} != {spec['sha256']}"
            )
        snapshots[str(dimension)] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": digest,
            "status": "PASS",
        }
    return {
        "file_count": len(entries),
        "files": entries,
        "manifest_verify": "PASS",
        "snapshots": snapshots,
        "status": "PASS",
    }


def _retriever(client: QdrantClient) -> QdrantRetriever:
    return QdrantRetriever(
        QdrantConfig(
            url="http://127.0.0.1:6333",
            collections=COLLECTIONS,
            timeout_s=3600,
        ),
        client=client,
    )


def _search_fn(retriever: QdrantRetriever):
    def search(dimension: int, vector, using: str):
        return retriever.query(
            dimension=int(dimension),
            vector=vector,
            using=using,
            limit=TOP_K,
        )

    return search


CATEGORY_VI = {
    "Q105076326": "địa danh / Việt Nam / lịch sử đô thị",
    "Q80398": "nhân vật / lịch sử cổ đại",
    "Q336": "khái niệm khoa học trừu tượng",
    "Q826858": "tổ chức / ngành công nghiệp âm nhạc",
    "Q482539": "công ty / thương hiệu tiêu dùng",
    "Q43304": "nhân vật",
    "Q546": "địa danh",
    "Q1416632": "tổ chức / tòa nhà",
    "Q1130757": "tổ chức / cảnh thực tế",
}


def _clip(value: str, limit: int = 104) -> str:
    text = str(value or "").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _direction_vi(direction: str) -> str:
    return {
        "EN→VI": "Anh→Việt",
        "VI→EN": "Việt→Anh",
        "IMG→TXT": "Ảnh→Văn bản",
    }.get(direction, direction)


def _path_label(direction: str, path: dict) -> str:
    return (
        f"{_direction_vi(direction)} / {direction} | "
        f"{path['dimension']}d → {path['vector_name']}"
    )


def _print_retrieval_paths(direction: str, paths: list[dict]) -> None:
    for path in paths:
        expected_score = path.get("expected_score")
        score_text = "n/a" if expected_score is None else f"{expected_score:.6f}"
        print(f"  {_path_label(direction, path)}", flush=True)
        print(
            f"    Hạng mong đợi / Expected rank : #{path['rank']}",
            flush=True,
        )
        print(
            f"    Độ tương đồng / Similarity    : {score_text}",
            flush=True,
        )
        print("    3 kết quả đầu / Top-3 results:", flush=True)
        for hit in path.get("top3", []):
            print(
                f"      #{hit['rank']}  {hit['qid']:<12} "
                f"score={hit['score']:.6f}",
                flush=True,
            )
            label_en = _clip(hit.get("label_en", ""))
            label_vi = _clip(hit.get("label_vi", ""))
            if label_vi:
                print(f"         Việt / VI: {label_vi}", flush=True)
            if label_en:
                print(f"         Anh / EN : {label_en}", flush=True)


def run_text_showcase(worker, client: QdrantClient) -> list[dict]:
    retriever = _retriever(client)
    search = _search_fn(retriever)
    results = []
    embedding_requests = 0
    query_executions = 0

    for index, example in enumerate(FROZEN_TEXT_SHOWCASE, 1):
        example_t0 = time.perf_counter()
        print("\n" + "-" * 92, flush=True)
        print(
            f"[VĂN BẢN / TEXT {index}/5] {example.name} — {example.qid}",
            flush=True,
        )
        print(
            f"  Nhóm / Category: {CATEGORY_VI.get(example.qid, example.category)} / "
            f"{example.category}",
            flush=True,
        )
        print(f"  Truy vấn tiếng Việt / Vietnamese query:", flush=True)
        print(f"    {example.text_vi}", flush=True)
        print(f"  Truy vấn tiếng Anh / English query:", flush=True)
        print(f"    {example.text_en}", flush=True)

        embed_t0 = time.perf_counter()
        en4096 = worker.embed_text(example.text_en, 4096)
        vi4096 = worker.embed_text(example.text_vi, 4096)
        embedding_seconds = time.perf_counter() - embed_t0
        embedding_requests += 2

        en_vectors = {4096: en4096, 1024: truncate_vector(en4096, 1024)}
        vi_vectors = {4096: vi4096, 1024: truncate_vector(vi4096, 1024)}

        en_to_vi = evaluate_four_paths(
            expected_qid=example.qid,
            vectors=en_vectors,
            search=search,
            paths=((4096, "vi"), (1024, "vi")),
        )
        vi_to_en = evaluate_four_paths(
            expected_qid=example.qid,
            vectors=vi_vectors,
            search=search,
            paths=((4096, "en"), (1024, "en")),
        )
        query_executions += 4
        paths = en_to_vi["paths"] + vi_to_en["paths"]
        _print_retrieval_paths("EN→VI", en_to_vi["paths"])
        _print_retrieval_paths("VI→EN", vi_to_en["paths"])
        ranks = [row["rank"] for row in paths]
        all_top1 = all(rank == 1 for rank in ranks)
        strict = all(rank is not None and int(rank) <= 3 for rank in ranks)
        if not strict or not all_top1:
            raise RuntimeError(
                f"Frozen text example failed {example.qid}: ranks={ranks}"
            )
        elapsed_seconds = time.perf_counter() - example_t0
        kv(
            "Hạng runtime / Runtime ranks",
            " / ".join(f"#{rank}" for rank in ranks),
        )
        kv(
            "2 lần embedding / 2 embedding requests",
            f"{embedding_seconds:.3f} s",
        )
        kv(
            "Tổng thời gian ví dụ / Example total",
            f"{elapsed_seconds:.3f} s",
        )
        kv("Kết quả / Verdict", "ALL TOP-1 — PASS")
        results.append(
            {
                "qid": example.qid,
                "name": example.name,
                "category": example.category,
                "text_en": example.text_en,
                "text_vi": example.text_vi,
                "authority_ranks": list(example.authority_ranks),
                "runtime_paths": paths,
                "runtime_ranks": ranks,
                "all_top1": all_top1,
                "strict_all_top3": strict,
                "embedding_seconds": round(embedding_seconds, 3),
                "elapsed_seconds": round(elapsed_seconds, 3),
            }
        )

    if embedding_requests != 10 or query_executions != 20:
        raise RuntimeError(
            f"Text execution contract mismatch: embeddings={embedding_requests}, "
            f"queries={query_executions}"
        )
    return results


def _download_normalized_image(example) -> tuple[Path, dict]:
    out_dir = RUN_ROOT / "frozen-images"
    out_dir.mkdir(parents=True, exist_ok=True)
    redirect = (
        "https://commons.wikimedia.org/wiki/Special:Redirect/file/"
        + quote(example.p18_filename, safe="")
        + f"?width={IMAGE_MAX_EDGE}"
    )
    last_error = None
    for attempt in range(1, 4):
        try:
            request = urllib.request.Request(
                redirect,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "image/*,*/*;q=0.8",
                },
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                resolved = response.geturl()
                length = response.headers.get("Content-Length")
                if length is not None and int(length) > MAX_DOWNLOAD_BYTES:
                    raise RuntimeError("Frozen image exceeds download byte bound")
                raw = response.read(MAX_DOWNLOAD_BYTES + 1)
            if len(raw) > MAX_DOWNLOAD_BYTES:
                raise RuntimeError("Frozen image exceeds download byte bound")

            with Image.open(io.BytesIO(raw)) as source:
                source.load()
                downloaded_size = list(map(int, source.size))
                image = source.convert("RGB")
            image.thumbnail(
                (IMAGE_MAX_EDGE, IMAGE_MAX_EDGE),
                Image.Resampling.LANCZOS,
            )
            path = out_dir / f"{example.qid}.png"
            image.save(path, format="PNG", optimize=True)
            digest = sha256_file(path)
            if digest != example.normalized_png_sha256:
                raise RuntimeError(
                    f"Frozen image SHA mismatch {example.qid}: "
                    f"{digest} != {example.normalized_png_sha256}"
                )
            return path, {
                "qid": example.qid,
                "wikidata_p18": example.p18_filename,
                "download_url": redirect,
                "resolved_url": resolved,
                "downloaded_bytes": len(raw),
                "downloaded_size": downloaded_size,
                "normalized_size": list(map(int, image.size)),
                "derived_png": str(path),
                "derived_png_sha256": digest,
                "status": "PASS",
            }
        except Exception as exc:
            last_error = exc
            if attempt == 3:
                raise
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"Frozen image download failed: {last_error}")


def run_image_showcase(worker, client: QdrantClient) -> list[dict]:
    retriever = _retriever(client)
    search = _search_fn(retriever)
    results = []
    embeddings = 0
    queries = 0

    for index, example in enumerate(FROZEN_IMAGE_SHOWCASE, 1):
        example_t0 = time.perf_counter()
        print("\n" + "-" * 92, flush=True)
        print(
            f"[HÌNH ẢNH / IMAGE {index}/4] {example.name} — {example.qid}",
            flush=True,
        )
        print(
            f"  Category / Nhóm: {example.category} / "
            f"{CATEGORY_VI.get(example.qid, example.category)}",
            flush=True,
        )
        path, asset = _download_normalized_image(example)
        print(
            f"  Ảnh nguồn / Source image (Wikimedia P18): "
            f"{asset['wikidata_p18']}",
            flush=True,
        )
        print(
            f"  Đã tải / Downloaded: {asset['downloaded_bytes']:,} bytes",
            flush=True,
        )
        print(
            f"  Kích thước nguồn / Source size: {asset['downloaded_size']}",
            flush=True,
        )
        print(
            f"  Kích thước chuẩn hóa / Normalized size: "
            f"{asset['normalized_size']}",
            flush=True,
        )
        print(
            f"  SHA-256 ảnh / Image SHA-256: "
            f"{asset['derived_png_sha256']}",
            flush=True,
        )
        print("  Xem trước / Preview:", flush=True)
        with Image.open(path) as shown:
            thumb = shown.copy()
            thumb.thumbnail((560, 380), Image.Resampling.LANCZOS)
            display(thumb)

        embed_t0 = time.perf_counter()
        vector4096 = worker.embed_image(path, 4096)
        embedding_seconds = time.perf_counter() - embed_t0
        embeddings += 1
        vectors = {
            4096: vector4096,
            1024: truncate_vector(vector4096, 1024),
        }
        outcome = evaluate_four_paths(
            expected_qid=example.qid,
            vectors=vectors,
            search=search,
            paths=(
                (4096, "en"),
                (1024, "en"),
                (4096, "vi"),
                (1024, "vi"),
            ),
        )
        queries += 4
        _print_retrieval_paths("IMG→TXT", outcome["paths"])
        ranks = [row["rank"] for row in outcome["paths"]]
        if not outcome["strict_all_top3"] or not outcome["all_top1"]:
            raise RuntimeError(
                f"Frozen image example failed {example.qid}: ranks={ranks}"
            )
        elapsed_seconds = time.perf_counter() - example_t0
        kv(
            "Runtime ranks / Hạng runtime",
            " / ".join(f"#{rank}" for rank in ranks),
        )
        kv(
            "1 lần embedding ảnh / 1 image embedding",
            f"{embedding_seconds:.3f} s",
        )
        kv(
            "Example total / Tổng thời gian ví dụ",
            f"{elapsed_seconds:.3f} s",
        )
        kv("Verdict / Kết quả", "ALL TOP-1 — PASS")
        results.append(
            {
                "qid": example.qid,
                "name": example.name,
                "category": example.category,
                "authority_ranks": list(example.authority_ranks),
                "runtime_paths": outcome["paths"],
                "runtime_ranks": ranks,
                "asset": asset,
                "all_top1": outcome["all_top1"],
                "strict_all_top3": outcome["strict_all_top3"],
                "embedding_seconds": round(embedding_seconds, 3),
                "elapsed_seconds": round(elapsed_seconds, 3),
            }
        )

    if embeddings != 4 or queries != 16:
        raise RuntimeError(
            f"Image execution contract mismatch: embeddings={embeddings}, queries={queries}"
        )
    return results
