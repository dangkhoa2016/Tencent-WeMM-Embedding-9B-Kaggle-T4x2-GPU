from __future__ import annotations

import math

from wemm_runtime import (
    FROZEN_IMAGE_SHOWCASE,
    FROZEN_TEXT_SHOWCASE,
    QdrantConfig,
    RuntimeConfig,
    evaluate_four_paths,
)
from wemm_runtime.qdrant import QdrantSearchResult


def test_runtime_config_is_provider_neutral(tmp_path):
    cfg = RuntimeConfig(model_path=tmp_path / "model", gpu_ids=(0, 1))
    assert str(cfg.normalized_model_path()).endswith("model")
    assert cfg.max_memory()[0] == "14200MiB"
    assert cfg.max_memory()[1] == "14200MiB"


def test_qdrant_config_collection_contract():
    cfg = QdrantConfig()
    assert "4096" in cfg.collection(4096)
    assert "1024" in cfg.collection(1024)


def test_frozen_showcase_counts_and_uniqueness():
    assert len(FROZEN_TEXT_SHOWCASE) == 5
    assert len(FROZEN_IMAGE_SHOWCASE) == 4
    assert len({x.qid for x in FROZEN_TEXT_SHOWCASE}) == 5
    assert len({x.qid for x in FROZEN_IMAGE_SHOWCASE}) == 4
    assert all(x.authority_ranks == (1, 1, 1, 1) for x in FROZEN_TEXT_SHOWCASE)
    assert all(x.authority_ranks == (1, 1, 1, 1) for x in FROZEN_IMAGE_SHOWCASE)


def test_image_sha_contracts_are_hex_sha256():
    for item in FROZEN_IMAGE_SHOWCASE:
        assert len(item.normalized_png_sha256) == 64
        int(item.normalized_png_sha256, 16)


def test_four_path_evaluator_accepts_top1():
    expected = "Q1"

    def search(dimension, vector, using):
        assert dimension in {4096, 1024}
        assert using in {"en", "vi"}
        assert math.isfinite(float(vector[0]))
        return [
            QdrantSearchResult(
                rank=1,
                point_id="1",
                qid=expected,
                score=0.9,
                payload={"qid": expected},
            )
        ]

    outcome = evaluate_four_paths(
        expected_qid=expected,
        vectors={4096: [1.0], 1024: [1.0]},
        search=search,
    )
    assert outcome["all_top1"] is True
    assert outcome["strict_all_top3"] is True
    assert len(outcome["paths"]) == 4


def test_four_path_evaluator_fails_missing_expected():
    def search(dimension, vector, using):
        return [
            QdrantSearchResult(
                rank=1,
                point_id="2",
                qid="Q2",
                score=0.5,
                payload={"qid": "Q2"},
            )
        ]

    outcome = evaluate_four_paths(
        expected_qid="Q1",
        vectors={4096: [1.0], 1024: [1.0]},
        search=search,
    )
    assert outcome["all_top1"] is False
    assert outcome["strict_all_top3"] is False
