from __future__ import annotations

import pytest

from wemm_kaggle.search.point_id import qid_to_point_id


def test_basic_qid():
    assert qid_to_point_id("Q42") == 42
    assert qid_to_point_id("Q1") == 1
    assert qid_to_point_id("Q100000") == 100000


def test_large_qid():
    big = "Q" + str((1 << 64) - 1)
    assert qid_to_point_id(big) == (1 << 64) - 1


@pytest.mark.parametrize(
    "bad",
    ["", "42", "Q0", "q42", " Q42", "Q42 ", "Q4Q", "Q", "Q-1", "Q1e3", None, 42],
)
def test_invalid_qids(bad):
    with pytest.raises(ValueError):
        qid_to_point_id(bad)


def test_overflow():
    with pytest.raises(ValueError):
        qid_to_point_id("Q" + str((1 << 64)))
