"""Focused tests for the corrected search benchmark CLI and output schema."""

from __future__ import annotations

from scripts.benchmark_search import COLLECTION_MAP, build_benchmark_v3_schema, main

CUSTOM_1024 = "custom_1024_collection"
CUSTOM_4096 = "custom_4096_collection"


def _install_fakes(monkeypatch, used):
    monkeypatch.setattr("scripts.benchmark_search.QdrantClient", lambda *a, **k: object())

    def fake_load(path):
        return []

    def fake_sample(rows, *args, **kwargs):
        return []

    def fake_eval(client, collection, src, tgt, rows, top_k=10):
        used.append(collection)
        return {
            "queries": 0,
            "Recall@1": 0.0,
            "Recall@5": 0.0,
            "Recall@10": 0.0,
            "MRR@10": 0.0,
            "http_errors": 0,
        }

    def fake_latency(*args, **kwargs):
        return {
            "queries": 0,
            "http_errors": 0,
            "min_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "max_ms": 0.0,
            "mean_ms": 0.0,
        }

    monkeypatch.setattr("scripts.benchmark_search.load_benchmark_canonical_parquet", fake_load)
    monkeypatch.setattr("scripts.benchmark_search.sample_quality_qids", fake_sample)
    monkeypatch.setattr("scripts.benchmark_search.sample_latency_qids", fake_sample)
    monkeypatch.setattr("scripts.benchmark_search.evaluate_combination", fake_eval)
    monkeypatch.setattr("scripts.benchmark_search.evaluate_latency", fake_latency)


def test_custom_collection_overrides_are_used(monkeypatch, tmp_path):
    used = []
    _install_fakes(monkeypatch, used)
    out = tmp_path / "bench.json"
    code = main([
        "--bench-name", "custom-collections",
        "--source-parquet", str(tmp_path / "rows.parquet"),
        "--collection-1024", CUSTOM_1024,
        "--collection-4096", CUSTOM_4096,
        "--out", str(out),
    ])
    assert code == 0
    assert set(used) == {CUSTOM_1024, CUSTOM_4096}


def test_default_collections_remain_canonical(monkeypatch, tmp_path):
    used = []
    _install_fakes(monkeypatch, used)
    out = tmp_path / "bench.json"
    code = main([
        "--bench-name", "defaults",
        "--source-parquet", str(tmp_path / "rows.parquet"),
        "--out", str(out),
    ])
    assert code == 0
    assert set(used) == set(COLLECTION_MAP.values())


def test_bench_name_emitted_in_output_schema():
    output = build_benchmark_v3_schema(
        bench_name="bilingual-probe",
        quality={},
        latency_e2e_ms={"queries": 0},
        selection={"method": "sha256_qid_ascending"},
        verdict="PASS",
    )
    assert output["bench_name"] == "bilingual-probe"
    assert output["schema_version"] == "wemm-search-benchmark-v3"
