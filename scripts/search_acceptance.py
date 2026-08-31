#!/usr/bin/env python3
"""Raw 5K semantic-search acceptance measurement (Recall/MRR/latency/ranked hits)."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx
from qdrant_client import QdrantClient

from wemm_kaggle.search.dataset import resolve_contract_paths
from wemm_kaggle.search.qdrant_store import COLLECTION_4096, COLLECTION_1024


JSON_HEADERS = {"content-type": "application/json"}
DIMENSIONS = (4096, 1024)
DIRECTIONS = (("en", "vi"), ("vi", "en"), ("en", "en"), ("vi", "vi"))
CROSS = {("en", "vi"), ("vi", "en")}


def _p50_p95(values: list[float]) -> dict:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0, "max": 0.0, "samples": 0}
    s = sorted(values)
    def q(i):
        idx = min(len(s) - 1, int((i / 100.0) * (len(s) - 1)))
        return s[idx]
    return {
        "p50": round(q(50), 3),
        "p95": round(q(95), 3),
        "mean": round(statistics.mean(s), 3),
        "max": round(max(s), 3),
        "samples": len(s),
    }


def _load_canonical_qids(contract: dict, expected_points: int, max_queries: int) -> list[dict]:
    rows = [json.loads(l) for l in open(contract["canonical_eligible_path"]) if l.strip()]
    rows = rows[:expected_points]
    selected = []
    for index, row in enumerate(rows):
        if index % 50 == 0:
            selected.append(row)
            if len(selected) >= max_queries:
                break
    return selected


def _search_once(gateway: str, query: str, qlng: str, tlng: str, dim: int, limit: int):
    start = time.monotonic()
    try:
        resp = httpx.post(
            f"{gateway}/v1/search",
            headers=JSON_HEADERS,
            json={
                "query": query,
                "query_language": qlng,
                "target_language": tlng,
                "dimension": dim,
                "limit": limit,
            },
            timeout=120.0,
        )
    except httpx.HTTPError as exc:
        return None, 503, {"error": str(exc)}, time.monotonic() - start
    return resp, resp.status_code, resp.json(), time.monotonic() - start


def run_acceptance(
    gateway: str,
    qdrant_host: str,
    qdrant_port: int,
    contract: dict,
    expected_points: int,
    out_dir: Path,
) -> dict:
    client = QdrantClient(host=qdrant_host, port=qdrant_port, timeout=30.0)
    counts = {
        COLLECTION_4096: client.count(COLLECTION_4096, exact=True).count,
        COLLECTION_1024: client.count(COLLECTION_1024, exact=True).count,
    }

    rows = _load_canonical_qids(contract, expected_points, 100)
    ranked_path = out_dir / "ranked-hits.jsonl"
    ranked_path.parent.mkdir(parents=True, exist_ok=True)
    ranked_f = ranked_path.open("w", encoding="utf-8")

    per_combo_recall = {}
    per_combo_mrr = {}
    per_combo_lat = {}
    http_errors = 0
    total_queries = 0
    cross_total = 0
    cross_misses = 0
    self_total = 0
    self_misses = 0

    for dimension in DIMENSIONS:
        for qlng, tlng in DIRECTIONS:
            key = f"{dimension}:{qlng}_{tlng}"
            correct_at_1 = 0
            correct_at_5 = 0
            correct_at_10 = 0
            rr10_sum = 0.0
            emb_lat = []
            qdr_lat = []
            tot_lat = []
            is_cross = (qlng, tlng) in CROSS

            for row in rows:
                query = row["text_en"] if qlng == "en" else row["text_vi"]
                resp, code, body, _ = _search_once(gateway, query, qlng, tlng, dimension, 10)
                total_queries += 1
                if is_cross:
                    cross_total += 1
                else:
                    self_total += 1
                if resp is None:
                    http_errors += 1
                    ranked_f.write(json.dumps({"query_qid": row["qid"], "combo": key, "error": code}) + "\n")
                    continue
                if code != 200:
                    http_errors += 1
                    ranked_f.write(json.dumps({"query_qid": row["qid"], "combo": key, "error": code}) + "\n")
                    continue
                hits = body.get("hits", [])
                top_qids = [h["qid"] for h in hits]
                qid = row["qid"]
                rank = (top_qids.index(qid) + 1) if qid in top_qids else 0
                if rank == 0:
                    if is_cross:
                        cross_misses += 1
                    else:
                        self_misses += 1
                if 1 <= rank <= 1:
                    correct_at_1 += 1
                if 1 <= rank <= 5:
                    correct_at_5 += 1
                if 1 <= rank <= 10:
                    correct_at_10 += 1
                if rank > 0:
                    rr10_sum += 1.0 / rank
                emb_lat.append(body.get("timing_ms", {}).get("embedding", 0.0))
                qdr_lat.append(body.get("timing_ms", {}).get("qdrant", 0.0))
                tot_lat.append(body.get("timing_ms", {}).get("total", 0.0))
                ranked_f.write(
                    json.dumps(
                        {
                            "query_qid": qid,
                            "combo": key,
                            "top_qids": top_qids,
                            "expected_rank": rank,
                            "scores": [h["score"] for h in hits],
                        }
                    )
                    + "\n"
                )

            n = len(rows)
            per_combo_recall[key] = {
                "queries": n,
                "recall_at_1": round(correct_at_1 / n, 4) if n else 0.0,
                "recall_at_5": round(correct_at_5 / n, 4) if n else 0.0,
                "recall_at_10": round(correct_at_10 / n, 4) if n else 0.0,
                "mrr_at_10": round(rr10_sum / n, 4) if n else 0.0,
            }
            per_combo_mrr[key] = per_combo_recall[key]
            per_combo_lat[key] = {
                "embedding": _p50_p95(emb_lat),
                "qdrant": _p50_p95(qdr_lat),
                "total": _p50_p95(tot_lat),
            }

    ranked_f.close()

    cross_lingual = {
        k: v
        for k, v in per_combo_recall.items()
        if tuple(k.split(":")[1].split("_")) in CROSS
    }
    self_retrieval = {
        k: v
        for k, v in per_combo_recall.items()
        if tuple(k.split(":")[1].split("_")) not in CROSS
    }

    verdict = "PASS"
    failure_reasons = []
    for key, rec in cross_lingual.items():
        if rec["recall_at_10"] < 1.0:
            verdict = "FAIL"
            failure_reasons.append(f"{key} expected QID not in top-10 for every query")
    if counts[COLLECTION_4096] != expected_points or counts[COLLECTION_1024] != expected_points:
        verdict = "FAIL"
        failure_reasons.append("collection count mismatch")
    if http_errors != 0:
        verdict = "FAIL"
        failure_reasons.append("http errors present")
    if cross_total == 0:
        verdict = "FAIL"
        failure_reasons.append("no cross-lingual queries")

    report = {
        "collection_counts": counts,
        "expected_points": expected_points,
        "cross_lingual": cross_lingual,
        "self_retrieval": self_retrieval,
        "latency_ms": per_combo_lat,
        "http_error_count": http_errors,
        "total_queries": total_queries,
        "cross_lingual_queries": cross_total,
        "self_retrieval_queries": self_total,
        "cross_lingual_misses_top10": cross_misses,
        "self_retrieval_misses_top10": self_misses,
        "verdict": verdict,
        "failure_reasons": failure_reasons,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway-url", default="http://127.0.0.1:8091")
    parser.add_argument("--qdrant-host", default="127.0.0.1")
    parser.add_argument("--qdrant-port", type=int, default=6333)
    parser.add_argument("--dataset-contract", required=True)
    parser.add_argument("--expected-points", type=int, default=5000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    contract_file = Path(args.dataset_contract)
    contract = resolve_contract_paths(json.loads(contract_file.read_text()), contract_file)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = run_acceptance(
        args.gateway_url,
        args.qdrant_host,
        args.qdrant_port,
        contract,
        args.expected_points,
        out_dir,
    )
    (out_dir / "search-acceptance.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
