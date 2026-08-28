import pytest


def test_batch_progression_stops_after_oom_and_materializes_skips():
    from wemm_kaggle.benchmark import run_batch_progression

    calls = []

    def run_one(batch):
        calls.append(batch)
        if batch == 2:
            return {"workload": "text", "dimension": 4096, "batch": batch, "status": "OOM"}
        return {"workload": "text", "dimension": 4096, "batch": batch, "status": "PASS"}

    records = run_batch_progression([1, 2, 4], run_one)

    assert calls == [1, 2]
    assert [row["batch"] for row in records] == [1, 2, 4]
    assert [row["status"] for row in records] == ["PASS", "OOM", "SKIPPED_AFTER_OOM"]
    assert records[-1]["workload"] == "text"
    assert records[-1]["dimension"] == 4096
    assert records[-1]["skipped_because_batch"] == 2


def test_batch_progression_propagates_non_oom_failures():
    from wemm_kaggle.benchmark import run_batch_progression

    def run_one(batch):
        raise ValueError(f"bad batch {batch}")

    with pytest.raises(ValueError, match="bad batch 1"):
        run_batch_progression([1, 2, 4], run_one)


def test_cuda_oom_classifier_accepts_cuda_oom_messages_only():
    from wemm_kaggle.benchmark import is_cuda_oom

    assert is_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB")) is True
    assert is_cuda_oom(RuntimeError("cuda runtime error: out of memory")) is True
    assert is_cuda_oom(RuntimeError("host out of memory")) is False
    assert is_cuda_oom(ValueError("CUDA out of memory")) is True


def test_safe_envelope_reports_highest_pass_and_first_oom_per_group():
    from wemm_kaggle.benchmark import build_safe_envelope

    records = [
        {"workload": "text", "dimension": 4096, "batch": 1, "status": "PASS"},
        {"workload": "text", "dimension": 4096, "batch": 2, "status": "OOM"},
        {"workload": "text", "dimension": 4096, "batch": 4, "status": "SKIPPED_AFTER_OOM"},
        {"workload": "image", "dimension": 1024, "batch": 1, "status": "PASS"},
        {"workload": "image", "dimension": 1024, "batch": 2, "status": "PASS"},
        {"workload": "image", "dimension": 1024, "batch": 4, "status": "PASS"},
    ]

    envelope = build_safe_envelope(records)
    by_key = {(row["workload"], row["dimension"]): row for row in envelope}

    text = by_key[("text", 4096)]
    assert text["planned_batches"] == [1, 2, 4]
    assert text["passed_batches"] == [1]
    assert text["first_oom_batch"] == 2
    assert text["max_tested_passing_batch"] == 1
    assert text["all_planned_batches_passed"] is False
    assert text["oom_boundary_observed"] is True

    image = by_key[("image", 1024)]
    assert image["passed_batches"] == [1, 2, 4]
    assert image["first_oom_batch"] is None
    assert image["max_tested_passing_batch"] == 4
    assert image["all_planned_batches_passed"] is True
    assert image["oom_boundary_observed"] is False


def test_batch_progression_emits_each_record_to_callback_immediately():
    from wemm_kaggle.benchmark import run_batch_progression

    emitted = []

    def run_one(batch):
        return {"workload": "text", "dimension": 1024, "batch": batch, "status": "OOM" if batch == 2 else "PASS"}

    records = run_batch_progression([1, 2, 4], run_one, on_record=emitted.append)

    assert emitted == records
    assert [row["status"] for row in emitted] == ["PASS", "OOM", "SKIPPED_AFTER_OOM"]
