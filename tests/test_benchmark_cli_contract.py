from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_benchmark_cli_loads_local_runtime_once_and_runs_matrix():
    text = read("scripts/benchmark.py")
    for required in (
        "BenchmarkConfig",
        "run_matrix",
        "resolve_model_dir",
        "validate_model_dir",
        "load_local_runtime",
        "cuda_preflight",
        "generated_benchmark_image",
        "benchmark.json",
        "operator.log",
    ):
        assert required in text
    assert text.count("load_local_runtime(") == 1
    for forbidden in (
        "snapshot_download(",
        "hf_hub_download(",
        "kagglehub.model_download(",
        'tencent/WeMM-Embedding-9B',
    ):
        assert forbidden not in text


def test_benchmark_cli_defaults_match_frozen_v013_contract():
    text = read("scripts/benchmark.py")
    assert 'default="1,2,4"' in text
    assert 'default="4096,1024"' in text
    assert 'default=2' in text
    assert 'default=20' in text
    assert '"PASS"' in text
    assert '"PARTIAL"' in text
    assert '"FAIL"' in text
    assert "WEMM_T4X2_BENCHMARK_PASS" in text
    assert "WEMM_T4X2_BENCHMARK_PARTIAL" in text


def test_benchmark_cli_packages_run_evidence_and_sha256():
    text = read("scripts/benchmark.py")
    assert "package_evidence_zip" in text
    assert ".zip.sha256" in text or "sha256_path" in text


def test_benchmark_cli_persists_series_records_incrementally():
    text = read("scripts/benchmark.py")
    assert "benchmark_series_complete" in text
    assert "on_record=" in text
    assert 'evidence["records"].append(record)' in text
    assert "atomic_write_json(evidence_path, evidence)" in text
