from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from wemm_kaggle.search.qualification import (
    CanonicalAuditRow,
    FreshInputSet,
    LOCKED_DATASET_SHA256,
    SNAPSHOT_1024_PUBLISHED_BASENAME,
    SNAPSHOT_1024_RESTORE_BASENAME,
    SNAPSHOT_1024_SHA256,
    SNAPSHOT_1024_SIZE,
    SNAPSHOT_4096_PUBLISHED_BASENAME,
    SNAPSHOT_4096_RESTORE_BASENAME,
    SNAPSHOT_4096_SHA256,
    SNAPSHOT_4096_SIZE,
    EXPECTED_ELIGIBLE_ROWS,
    EXPECTED_RAW_ROWS,
    REJECT_REASON,
    deterministic_qid_sample,
    discover_fresh_inputs,
    expected_point_id_set,
    load_canonical_rows_from_parquet,
)
from wemm_kaggle.search.point_id import qid_to_point_id

_QMOD = "wemm_kaggle.search.qualification"


def _row(qid: str, source_row_index: int = 0, text_en: str = "en", text_vi: str = "vi") -> CanonicalAuditRow:
    return CanonicalAuditRow(
        qid=qid,
        point_id=qid_to_point_id(qid),
        text_en=text_en,
        text_vi=text_vi,
        source_row_index=source_row_index,
    )


# ---------------------------------------------------------------------------
# Locked constants
# ---------------------------------------------------------------------------

class TestLockedConstants:
    def test_dataset_sha256(self):
        assert LOCKED_DATASET_SHA256 == "1f69b4b64a0aecf41505aef3cf6fa5c5ce19e4a733d198765ee6f431beb711e9"

    def test_snapshot_1024_sha256(self):
        assert SNAPSHOT_1024_SHA256 == "6be5a750d1ec15463df59bca8e4fb676207af8700c97e609ee48e626158b5b90"

    def test_snapshot_4096_sha256(self):
        assert SNAPSHOT_4096_SHA256 == "e2490a1a0fbd25f7dac34e40bce25907b7cf24f0bfd4ca7a80d8d920734ae85e"

    def test_expected_row_counts(self):
        assert EXPECTED_RAW_ROWS == 100000
        assert EXPECTED_ELIGIBLE_ROWS == 99967

    def test_reject_reason(self):
        assert REJECT_REASON == "MISSING_ENGLISH_TEXT"

    def test_snapshot_sizes_are_positive_integers(self):
        assert isinstance(SNAPSHOT_1024_SIZE, int) and SNAPSHOT_1024_SIZE > 0
        assert isinstance(SNAPSHOT_4096_SIZE, int) and SNAPSHOT_4096_SIZE > 0

    def test_snapshot_basenames_suffixes(self):
        assert SNAPSHOT_1024_PUBLISHED_BASENAME.endswith(".s")
        assert SNAPSHOT_4096_PUBLISHED_BASENAME.endswith(".s")
        assert SNAPSHOT_1024_RESTORE_BASENAME.endswith(".snapshot")
        assert SNAPSHOT_4096_RESTORE_BASENAME.endswith(".snapshot")


# ---------------------------------------------------------------------------
# Canonical point ID identity
# ---------------------------------------------------------------------------

class TestCanonicalPointID:
    def test_point_id_is_qid_suffix(self):
        rows = [_row("Q42", 0), _row("Q1001", 1), _row("Q900000", 2)]
        assert expected_point_id_set(rows) == {42, 1001, 900000}

    def test_point_id_not_row_index(self):
        rows = [_row("Q42", 0), _row("Q1001", 1)]
        assert expected_point_id_set(rows) != {0, 1}

    def test_empty_rows(self):
        assert expected_point_id_set([]) == set()


# ---------------------------------------------------------------------------
# Deterministic sampling
# ---------------------------------------------------------------------------

class TestDeterministicSample:
    def test_determinism(self):
        rows = [_row(f"Q{i}", i) for i in range(1, 2001)]
        s1 = deterministic_qid_sample(rows, 500)
        s2 = deterministic_qid_sample(rows, 500)
        assert [r.qid for r in s1] == [r.qid for r in s2]

    def test_size_500(self):
        rows = [_row(f"Q{i}", i) for i in range(1, 2001)]
        sample = deterministic_qid_sample(rows, 500)
        assert len(sample) == 500

    def test_size_50(self):
        rows = [_row(f"Q{i}", i) for i in range(1, 2001)]
        sample = deterministic_qid_sample(rows, 50)
        assert len(sample) == 50

    def test_ordering_by_sha256_qid(self):
        rows = [_row(f"Q{i+1000}", i+1000) for i in range(100)]
        sample = deterministic_qid_sample(rows, 10)
        sha_keys = [
            hashlib.sha256(r.qid.encode("utf-8")).hexdigest()
            for r in sample
        ]
        assert sha_keys == sorted(sha_keys)

    def test_size_exceeds_available(self):
        rows = [_row("Q1"), _row("Q2")]
        assert len(deterministic_qid_sample(rows, 10)) == 2

    def test_zero_size_returns_empty(self):
        assert deterministic_qid_sample([_row("Q1")], 0) == []


# ---------------------------------------------------------------------------
# expected_point_id_set
# ---------------------------------------------------------------------------

class TestExpectedPointIdSet:
    def test_returns_set_of_point_ids(self):
        rows = [_row("Q42"), _row("Q100"), _row("Q9999")]
        result = expected_point_id_set(rows)
        assert result == {42, 100, 9999}
        assert isinstance(result, set)


# ---------------------------------------------------------------------------
# discover_fresh_inputs — fail-closed 0/1/>1 matching
# ---------------------------------------------------------------------------

def _create_fixture_tree(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    sub = root / "sub"
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "wikidata-en-vi-100k.parquet").write_bytes(b"pq-data")
    (root / SNAPSHOT_1024_PUBLISHED_BASENAME).write_bytes(b"snap-1024")
    (sub / SNAPSHOT_4096_PUBLISHED_BASENAME).write_bytes(b"snap-4096")
    (sub / "ingest-state.json").write_bytes(b'{"ok":true}')


class TestDiscoverFreshInputs:
    def test_zero_match_fails(self, tmp_path: Path):
        root = tmp_path / "empty"
        root.mkdir()
        with mock.patch(f"{_QMOD}.KAGGLE_INPUT_ROOT", root):
            with pytest.raises(ValueError, match="expected exactly 1.*parquet"):
                discover_fresh_inputs(root)

    def test_duplicate_parquet_fails(self, tmp_path: Path):
        root = tmp_path / "dup_pq"
        a = root / "a"
        b = root / "b"
        a.mkdir(parents=True)
        b.mkdir(parents=True)
        (a / "wikidata-en-vi-100k.parquet").write_bytes(b"pq1")
        (b / "wikidata-en-vi-100k.parquet").write_bytes(b"pq2")
        (root / SNAPSHOT_1024_PUBLISHED_BASENAME).write_bytes(b"s1")
        (root / SNAPSHOT_4096_PUBLISHED_BASENAME).write_bytes(b"s4")
        (root / "ingest-state.json").write_bytes(b"{}")
        with mock.patch(f"{_QMOD}.KAGGLE_INPUT_ROOT", root):
            with pytest.raises(ValueError, match="expected exactly 1.*parquet"):
                discover_fresh_inputs(root)

    def test_duplicate_snapshot_fails(self, tmp_path: Path):
        root = tmp_path / "dup_snap"
        root.mkdir()
        (root / "wikidata-en-vi-100k.parquet").write_bytes(b"pq")
        a = root / "a"
        b = root / "b"
        a.mkdir()
        b.mkdir()
        (a / SNAPSHOT_1024_PUBLISHED_BASENAME).write_bytes(b"s1a")
        (b / SNAPSHOT_1024_PUBLISHED_BASENAME).write_bytes(b"s1b")
        (root / SNAPSHOT_4096_PUBLISHED_BASENAME).write_bytes(b"s4")
        (root / "ingest-state.json").write_bytes(b"{}")
        with mock.patch(f"{_QMOD}.KAGGLE_INPUT_ROOT", root):
            with pytest.raises(ValueError, match="expected exactly 1.*1024"):
                discover_fresh_inputs(root)

    def test_exact_match_succeeds(self, tmp_path: Path):
        root = tmp_path / "exact"
        _create_fixture_tree(root)
        with mock.patch(f"{_QMOD}.KAGGLE_INPUT_ROOT", root):
            result = discover_fresh_inputs(root)
        assert isinstance(result, FreshInputSet)
        assert result.source_parquet.name == "wikidata-en-vi-100k.parquet"
        assert result.snapshot_1024_published.name == SNAPSHOT_1024_PUBLISHED_BASENAME
        assert result.snapshot_4096_published.name == SNAPSHOT_4096_PUBLISHED_BASENAME
        assert result.public_ingest_state.name == "ingest-state.json"

    def test_wrong_root_rejected(self, tmp_path: Path):
        real_input = tmp_path / "kaggle" / "input"
        real_input.mkdir(parents=True)
        outside = tmp_path / "outside"
        outside.mkdir()
        with mock.patch(f"{_QMOD}.KAGGLE_INPUT_ROOT", real_input):
            with pytest.raises(ValueError, match="must be under /kaggle/input"):
                discover_fresh_inputs(outside)

    def test_root_not_found(self, tmp_path: Path):
        nonexistent = tmp_path / "nope"
        with mock.patch(f"{_QMOD}.KAGGLE_INPUT_ROOT", nonexistent):
            with pytest.raises(FileNotFoundError, match="input root not found"):
                discover_fresh_inputs(nonexistent)


# ---------------------------------------------------------------------------
# load_canonical_rows_from_parquet — lightweight validation
# ---------------------------------------------------------------------------

class TestLoadCanonicalRows:
    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError, match="parquet not found"):
            load_canonical_rows_from_parquet(Path("/nonexistent/file.parquet"))

    def test_wrong_row_count_raises(self, tmp_path: Path):
        table = pa.table({
            "qid": ["Q1", "Q2"],
            "text_en": ["hello", "world"],
            "text_vi": ["xin", "chao"],
            "label_en": ["hello", "world"],
            "description_en": ["", ""],
            "label_vi": ["xin", "chao"],
            "description_vi": ["", ""],
        })
        pq_path = tmp_path / "small.parquet"
        pq.write_table(table, pq_path)
        with pytest.raises(ValueError, match="expected exactly 100000 raw rows"):
            load_canonical_rows_from_parquet(pq_path)

    def test_missing_qid_column_raises(self, tmp_path: Path):
        table = pa.table({
            "document_en": ["hello"],
            "document_vi": ["xin"],
            "label_en": ["hello"],
            "description_en": [""],
            "label_vi": ["xin"],
            "description_vi": [""],
        })
        pq_path = tmp_path / "no_qid.parquet"
        pq.write_table(table, pq_path)
        with pytest.raises(ValueError, match="missing QID column"):
            load_canonical_rows_from_parquet(pq_path)

    def test_missing_structured_field_raises(self, tmp_path: Path):
        table = pa.table({
            "qid": ["Q1"],
            "text_en": ["hello"],
            "text_vi": ["xin"],
        })
        pq_path = tmp_path / "no_struct.parquet"
        pq.write_table(table, pq_path)
        with pytest.raises(ValueError, match="MISSING_REQUIRED_COLUMN"):
            load_canonical_rows_from_parquet(pq_path)
