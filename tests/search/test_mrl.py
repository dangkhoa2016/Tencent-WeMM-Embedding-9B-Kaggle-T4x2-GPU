from __future__ import annotations

import math

import pytest

from wemm_kaggle.search.mrl import (
    DERIVED_DIMENSION,
    FULL_DIMENSION,
    l2_normalize,
    truncate_and_normalize,
)


def _unit(length: int) -> list[float]:
    return [1.0 / math.sqrt(length)] * length


def test_l2_normalize_unit():
    v = [3.0, 4.0]
    out = l2_normalize(v)
    assert len(out) == 2
    assert math.isclose(sum(x * x for x in out), 1.0, rel_tol=1e-9)
    assert math.isclose(out[0], 0.6, rel_tol=1e-9)
    assert math.isclose(out[1], 0.8, rel_tol=1e-9)


def test_l2_normalize_zero():
    with pytest.raises(ValueError):
        l2_normalize([0.0, 0.0, 0.0])


def test_l2_normalize_nan():
    with pytest.raises(ValueError):
        l2_normalize([1.0, float("nan")])


def test_truncate_and_normalize():
    v = _unit(FULL_DIMENSION)
    out = truncate_and_normalize(v, DERIVED_DIMENSION)
    assert len(out) == DERIVED_DIMENSION
    assert math.isclose(sum(x * x for x in out), 1.0, rel_tol=1e-9)


def test_truncate_is_pure_prefix_renorm():
    v = _unit(FULL_DIMENSION)
    out = truncate_and_normalize(v, DERIVED_DIMENSION)
    expected = l2_normalize(v[:DERIVED_DIMENSION])
    assert all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(out, expected))


@pytest.mark.parametrize(
    "dim",
    [0, 512, 4096, 4097, -1],
)
def test_bad_target_dimension(dim):
    with pytest.raises(ValueError):
        truncate_and_normalize(_unit(FULL_DIMENSION), dim)


def test_bad_input_length():
    with pytest.raises(ValueError):
        truncate_and_normalize(_unit(DERIVED_DIMENSION))


def test_nan_input():
    v = _unit(FULL_DIMENSION)
    v[0] = float("nan")
    with pytest.raises(ValueError):
        truncate_and_normalize(v)


def test_zero_truncated_vector():
    v = [0.0] * FULL_DIMENSION
    with pytest.raises(ValueError):
        truncate_and_normalize(v)


def test_constants():
    assert FULL_DIMENSION == 4096
    assert DERIVED_DIMENSION == 1024
