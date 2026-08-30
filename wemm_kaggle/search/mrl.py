from __future__ import annotations

import math

FULL_DIMENSION = 4096
DERIVED_DIMENSION = 1024
ALLOWED_TARGETS = frozenset({DERIVED_DIMENSION})


def l2_normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(float(x) * float(x) for x in values))
    if not math.isfinite(norm):
        raise ValueError("vector norm is not finite (NaN/Inf present)")
    if norm == 0.0:
        raise ValueError("cannot L2-normalize a zero-norm vector")
    return [float(x) / norm for x in values]


def truncate_and_normalize(
    values: list[float],
    dimension: int = DERIVED_DIMENSION,
) -> list[float]:
    if dimension not in ALLOWED_TARGETS:
        raise ValueError(f"allowed target dimension is {sorted(ALLOWED_TARGETS)}; got {dimension}")
    if len(values) != FULL_DIMENSION:
        raise ValueError(f"input must have length {FULL_DIMENSION}; got {len(values)}")
    for value in values:
        if not math.isfinite(float(value)):
            raise ValueError("input vector contains NaN/Inf values")
    truncated = values[:dimension]
    return l2_normalize(truncated)
