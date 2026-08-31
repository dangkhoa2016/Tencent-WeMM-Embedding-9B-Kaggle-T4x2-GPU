#!/usr/bin/env python3
"""Qdrant vector-search benchmark.

Measures recall, MRR, and e2e latency for stored-vector retrieval.
No model loading or GPU usage; embeddings are pre-ingested.

Schema: wemm-search-benchmark-v3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx
from qdrant_client import QdrantClient

from wemm_kaggle.search.qualification import (
    CanonicalAuditRow,
    deterministic_qid_sample,
    load_canonical_rows_from_parquet,
)

SCHEMA_VERSION = "wemm-search-benchmark-v3"

QUALITY_SIZE = 500
LATENCY_SIZE = 50

QUALITY_COMBOS = [
    ("4096", "en", "vi"),
    ("4096", "vi", "en"),
    ("1024", "en", "vi"),
    ("1024", "vi", "en"),
]

COLLECTION_MAP = {
    "4096": "wikidata_en_vi_wemm9b_4096_v030_rc2_d819dc7_v1",
    "1024": "wikidata_en_vi_wemm9b_1024_v030_rc2_d819dc7_v1",
}

GATEWAY_URL = "http://127.0.0.1:8091"
JSON_HEADERS = {"content-type": "application/json"}


def gateway_search_request(
    gateway_url: str,
    query: str,
    src_lang: str,
    tgt_lang: str,
    dimension: int,
    limit: int = 10,
    timeout: float = 120.0,
) -> dict:
    """Perform one real E2E HTTP request to the local search gateway.

    POSTs ``{query, query_language, target_language, dimension, limit}`` to
    ``<gateway_url>/v1/search`` and returns:

        {"ok": bool, "elapsed_ms": float, "http_status": int|None,
         "error": str|None}

    ``elapsed_ms`` wraps the full request/response round trip.  Any transport
    failure or non-200 response fails closed (``ok=False``).
    """
    start = time.monotonic()
    ok = False
    http_status = None
    error = None
    try:
        resp = httpx.post(
            f"{gateway_url}/v1/search",
            headers=JSON_HEADERS,
            json={
                "query": query,
                "query_language": src_lang,
                "target_language": tgt_lang,
                "dimension": dimension,
                "limit": limit,
            },
            timeout=timeout,
        )
        http_status = resp.status_code
        ok = resp.status_code == 200
        if not ok:
            error = f"http_status={resp.status_code}"
    except httpx.HTTPError as exc:
        error = str(exc)
    elapsed_ms = (time.monotonic() - start) * 1000.0
    return {
        "ok": ok,
        "elapsed_ms": round(elapsed_ms, 3),
        "http_status": http_status,
        "error": error,
    }


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int((pct / 100.0) * (len(s) - 1)))
    return s[idx]


# ---------------------------------------------------------------------------
# Public functions (importable for tests)
# ---------------------------------------------------------------------------

def load_benchmark_canonical_parquet(source_parquet: Path) -> list[CanonicalAuditRow]:
    """Load canonical rows from a source parquet file."""
    return load_canonical_rows_from_parquet(source_parquet)


def sample_quality_qids(rows: list[CanonicalAuditRow]) -> list[CanonicalAuditRow]:
    """Deterministic sample of 500 QIDs by SHA256 ascending."""
    return deterministic_qid_sample(rows, QUALITY_SIZE)


def sample_latency_qids(rows: list[CanonicalAuditRow]) -> list[CanonicalAuditRow]:
    """Deterministic sample of 50 QIDs by SHA256 ascending."""
    return deterministic_qid_sample(rows, LATENCY_SIZE)


def _retrieve_source_vector(
    client: QdrantClient,
    collection: str,
    point_id: int,
    src_lang: str,
) -> list[float] | None:
    """Retrieve the source-language vector for a given point_id.

    Uses the real qdrant-client 1.19.0 ``Record.vector`` field (singular), which
    holds the named-vector dict ``{"en": [...], "vi": [...]}`` for named-vector
    collections.  A missing vector or a vector missing *src_lang* fails closed
    (returns None, never silently substituted).
    """
    results = client.retrieve(
        collection_name=collection,
        ids=[point_id],
        with_payload=False,
        with_vectors=[src_lang],
    )
    if not results:
        return None
    point = results[0]
    vector = getattr(point, "vector", None)
    if isinstance(vector, dict):
        return vector.get(src_lang)
    return None


def evaluate_combination(
    client: QdrantClient,
    collection: str,
    src_lang: str,
    tgt_lang: str,
    sampled_qids: list[CanonicalAuditRow],
    top_k: int = 10,
) -> dict[str, Any]:
    """Evaluate a single dimension/language combination.

    For each query point, retrieves the source-language vector from the
    collection, then searches the target collection for top-k results.
    Ground truth: the QID's target-language point is the correct match.

    Returns dict with: queries, Recall@1, Recall@5, Recall@10, MRR@10, http_errors.
    """
    correct_at_1 = 0
    correct_at_5 = 0
    correct_at_10 = 0
    rr10_sum = 0.0
    http_errors = 0
    total = len(sampled_qids)

    for row in sampled_qids:
        try:
            query_vector = _retrieve_source_vector(client, collection, row.point_id, src_lang)
            if query_vector is None:
                http_errors += 1
                continue

            search_results = client.query_points(
                collection_name=collection,
                query=query_vector,
                using=tgt_lang,
                limit=top_k,
                query_filter=None,
            )
            points = search_results.points if hasattr(search_results, "points") else search_results
            top_qids = []
            for p in points:
                payload = getattr(p, "payload", None) or {}
                qid_val = payload.get("qid")
                if qid_val is None:
                    pid = getattr(p, "id", None)
                    if pid is not None:
                        qid_val = f"Q{pid}"
                if qid_val is not None:
                    top_qids.append(str(qid_val))

            rank = (top_qids.index(row.qid) + 1) if row.qid in top_qids else 0
            if rank == 1:
                correct_at_1 += 1
            if 1 <= rank <= 5:
                correct_at_5 += 1
            if 1 <= rank <= 10:
                correct_at_10 += 1
            if rank > 0:
                rr10_sum += 1.0 / rank
        except Exception:
            http_errors += 1

    return {
        "queries": total,
        "Recall@1": round(correct_at_1 / total, 4) if total else 0.0,
        "Recall@5": round(correct_at_5 / total, 4) if total else 0.0,
        "Recall@10": round(correct_at_10 / total, 4) if total else 0.0,
        "MRR@10": round(rr10_sum / total, 4) if total else 0.0,
        "http_errors": http_errors,
    }


def evaluate_latency(
    gateway_url: str,
    sampled_latency_qids: list[CanonicalAuditRow],
    src_lang: str,
    tgt_lang: str,
    dimension: int = 4096,
) -> dict[str, Any]:
    """Measure real gateway E2E search latency over the 50-query latency sample.

    Each query issues an actual HTTP request to the local loopback search
    gateway (``<gateway_url>/v1/search``); the source text of each sampled QID
    is the query.  End-to-end elapsed time wraps the HTTP request/response.
    Any transport failure or non-200 response increments ``http_errors``.

    Returns dict with: queries, http_errors, min_ms, p50_ms, p95_ms, p99_ms,
    max_ms, mean_ms.
    """
    latencies: list[float] = []
    http_errors = 0
    total = len(sampled_latency_qids)

    for row in sampled_latency_qids:
        query_text = getattr(row, f"text_{src_lang}", "") if hasattr(row, f"text_{src_lang}") else ""
        if not query_text:
            http_errors += 1
            continue
        result = gateway_search_request(
            gateway_url=gateway_url,
            query=query_text,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            dimension=dimension,
        )
        if result["ok"]:
            latencies.append(result["elapsed_ms"])
        else:
            http_errors += 1

    if not latencies:
        return {
            "queries": total,
            "http_errors": http_errors,
            "min_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "max_ms": 0.0,
            "mean_ms": 0.0,
        }

    return {
        "queries": total,
        "http_errors": http_errors,
        "min_ms": round(min(latencies), 3),
        "p50_ms": round(_percentile(latencies, 50), 3),
        "p95_ms": round(_percentile(latencies, 95), 3),
        "p99_ms": round(_percentile(latencies, 99), 3),
        "max_ms": round(max(latencies), 3),
        "mean_ms": round(statistics.mean(latencies), 3),
    }


def build_benchmark_v3_schema(
    bench_name: str,
    quality: dict[str, dict[str, Any]],
    latency_e2e_ms: dict[str, Any],
    selection: dict[str, Any],
    verdict: str,
) -> dict[str, Any]:
    """Build the top-level v3 benchmark output dict."""
    return {
        "schema_version": SCHEMA_VERSION,
        "bench_name": bench_name,
        "selection": selection,
        "quality": quality,
        "latency_e2e_ms": latency_e2e_ms,
        "execution_verdict": verdict,
    }


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Qdrant search benchmark")
    parser.add_argument("--bench-name", required=True)
    parser.add_argument("--source-parquet", required=True)
    parser.add_argument("--collection-1024", default=COLLECTION_MAP["1024"])
    parser.add_argument("--collection-4096", default=COLLECTION_MAP["4096"])
    parser.add_argument("--qdrant-host", default="127.0.0.1")
    parser.add_argument("--qdrant-port", type=int, default=6333)
    parser.add_argument("--gateway-url", default=GATEWAY_URL)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    # --- Load canonical rows ---
    canonical_rows = load_benchmark_canonical_parquet(Path(args.source_parquet))

    quality_rows = sample_quality_qids(canonical_rows)
    latency_rows = sample_latency_qids(canonical_rows)

    selection = {
        "method": "sha256_qid_ascending",
        "quality_queries": len(quality_rows),
        "latency_queries": len(latency_rows),
    }

    # --- Qdrant client ---
    client = QdrantClient(host=args.qdrant_host, port=args.qdrant_port, timeout=120.0)

    # --- Evaluate four quality combos ---
    quality: dict[str, dict[str, Any]] = {}
    collection_map = {"1024": args.collection_1024, "4096": args.collection_4096}
    for dim_key, src, tgt in QUALITY_COMBOS:
        collection = collection_map[dim_key]
        block = evaluate_combination(client, collection, src, tgt, quality_rows, top_k=10)
        combo_name = f"{dim_key}_{src}_{tgt}"
        quality[combo_name] = block

    # --- E2E latency: real gateway HTTP, loopback only (4096, en->vi representative) ---
    latency_e2e_ms = evaluate_latency(
        gateway_url=args.gateway_url,
        sampled_latency_qids=latency_rows,
        src_lang="en",
        tgt_lang="vi",
        dimension=4096,
    )

    # --- Verdict: PASS iff benchmark completed, schema valid, no required request failed ---
    any_http_error = any(b["http_errors"] > 0 for b in quality.values())
    latency_http_error = latency_e2e_ms["http_errors"] > 0
    verdict = "PASS" if (not any_http_error and not latency_http_error) else "FAIL"

    output = build_benchmark_v3_schema(
        bench_name=args.bench_name,
        quality=quality,
        latency_e2e_ms=latency_e2e_ms,
        selection=selection,
        verdict=verdict,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    sys.exit(main())
