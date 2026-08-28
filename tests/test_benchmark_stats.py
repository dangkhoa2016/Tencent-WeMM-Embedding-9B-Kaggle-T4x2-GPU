import math

import pytest


def test_percentile_uses_linear_interpolation():
    from wemm_kaggle.benchmark import percentile

    samples = [4.0, 1.0, 3.0, 2.0]
    assert percentile(samples, 0.50) == pytest.approx(2.5)
    assert percentile(samples, 0.95) == pytest.approx(3.85)
    assert percentile([7.0], 0.95) == pytest.approx(7.0)


def test_summarize_latencies_reports_repeatability_and_total_throughput():
    from wemm_kaggle.benchmark import summarize_latencies

    result = summarize_latencies([0.1, 0.2, 0.4], batch_size=2)

    assert result["min"] == pytest.approx(0.1)
    assert result["mean"] == pytest.approx(0.7 / 3)
    assert result["p50"] == pytest.approx(0.2)
    assert result["p95"] == pytest.approx(0.38)
    assert result["max"] == pytest.approx(0.4)
    assert result["total_seconds"] == pytest.approx(0.7)
    assert result["total_items"] == 6
    assert result["items_per_second"] == pytest.approx(6 / 0.7)
    assert math.isfinite(result["items_per_second"])


def test_statistics_reject_empty_samples_and_invalid_batch_size():
    from wemm_kaggle.benchmark import percentile, summarize_latencies

    with pytest.raises(ValueError, match="samples"):
        percentile([], 0.5)
    with pytest.raises(ValueError, match="percentile"):
        percentile([1.0], 1.1)
    with pytest.raises(ValueError, match="samples"):
        summarize_latencies([], 1)
    with pytest.raises(ValueError, match="batch"):
        summarize_latencies([1.0], 0)
