import json
from pathlib import Path


def test_atomic_write_json_replaces_target(tmp_path):
    from wemm_kaggle.evidence import atomic_write_json

    target = tmp_path / "evidence.json"
    atomic_write_json(target, {"status": "first"})
    atomic_write_json(target, {"status": "pass", "value": 2})
    assert json.loads(target.read_text()) == {"status": "pass", "value": 2}
    assert not list(tmp_path.glob("*.tmp"))


def test_process_snapshot_has_rss_and_host_memory():
    from wemm_kaggle.evidence import process_snapshot

    snap = process_snapshot()
    assert snap["pid"] > 0
    assert snap["rss_bytes"] > 0
    assert snap["host_mem_total_bytes"] > 0
    assert snap["host_mem_available_bytes"] > 0


def test_package_evidence_zip_preserves_run_directory_and_writes_sha256(tmp_path):
    import hashlib
    import zipfile

    from wemm_kaggle.evidence import package_evidence_zip

    run_dir = tmp_path / "20260828T000000Z"
    run_dir.mkdir()
    (run_dir / "benchmark.json").write_text('{"status":"PASS"}\n', encoding="utf-8")
    (run_dir / "operator.log").write_text("done\n", encoding="utf-8")
    zip_path = tmp_path / "wemm-embedding-9b-t4x2-benchmark-20260828T000000Z.zip"

    result = package_evidence_zip(run_dir, zip_path)

    assert result["zip_path"] == str(zip_path)
    assert result["sha256_path"] == str(zip_path) + ".sha256"
    with zipfile.ZipFile(zip_path) as archive:
        assert sorted(archive.namelist()) == [
            "20260828T000000Z/benchmark.json",
            "20260828T000000Z/operator.log",
        ]
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    sha_text = Path(result["sha256_path"]).read_text(encoding="utf-8").strip()
    assert sha_text == f"{digest}  {zip_path.name}"
