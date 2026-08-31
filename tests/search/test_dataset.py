from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from wemm_kaggle.search.dataset import (
    _build_emitted_text,
    build_contract,
    count_candidate,
    find_column,
    resolve_contract_paths,
)


def _write_parquet(rows: list[dict], path) -> None:
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def _canon(rows: list[dict]) -> list[dict]:
    """Fill the canonical structured fields required by the production path."""
    out = []
    for r in rows:
        item = {
            "qid": r["qid"],
            "document_en": r.get("document_en", r.get("text_en", "")),
            "document_vi": r.get("document_vi", r.get("text_vi", "")),
            "label_en": r.get("label_en", r.get("text_en", "")),
            "description_en": r.get("description_en", ""),
            "label_vi": r.get("label_vi", r.get("text_vi", "")),
            "description_vi": r.get("description_vi", ""),
        }
        out.append(item)
    return out


def test_find_column_and_count():
    cols = ["qid", "text_en", "english", "text_vi", "Country"]
    assert find_column(cols, ("text_en", "english_text", "en")) == "text_en"
    assert find_column(cols, ("text_vi", "vi_text")) == "text_vi"
    assert count_candidate(cols, ("text_en", "english")) == 2


def test_label_description_aliases_never_substitute(tmp_path):
    """D7.1: alias fields (english_label, description, ...) must not be used as
    canonical emission sources when the real canonical field is present."""
    rows = [
        {"qid": "Q1", "text_en": "hello", "text_vi": "xin chao",
         "label_en": "CANON", "description_en": "canon desc",
         "label_vi": "CANON VI", "description_vi": "canon vi desc",
         "english_label": "WRONG", "description": "WRONG DESC"},
    ]
    src = tmp_path / "d.parquet"
    _write_parquet(rows, src)
    out = tmp_path / "out"
    build_contract(src, output_dir=out, expected_raw_rows=1)
    text = json.loads(next(l for l in (out / "canonical-eligible.jsonl").read_text().splitlines() if l.strip()))
    assert text["text_en"] == "CANON. canon desc."
    assert "WRONG" not in text["text_en"]


def test_contract_basic(tmp_path):
    rows = _canon([
        {"qid": "Q1", "text_en": "hello", "text_vi": "xin chao"},
        {"qid": "Q2", "text_en": "world", "text_vi": "the gioi"},
    ])
    src = tmp_path / "d.parquet"
    _write_parquet(rows, src)
    out = tmp_path / "out"
    contract = build_contract(src, output_dir=out, expected_raw_rows=2)
    assert contract["raw_row_count"] == 2
    assert contract["eligible_bilingual_row_count"] == 2
    assert contract["rejected_row_count"] == 0
    assert contract["status"] == "PASS"
    manifested = [json.loads(l) for l in (out / "canonical-eligible.jsonl").read_text().splitlines() if l.strip()]
    assert manifested[0]["text_en"] == "hello."
    assert manifested[0]["text_vi"] == "xin chao."


def test_contract_rejects_missing_en(tmp_path):
    rows = _canon([
        {"qid": "Q1", "text_en": "hello", "text_vi": "xin chao"},
        {"qid": "Q2", "text_en": "", "text_vi": "chi viet"},
        {"qid": "Q3", "text_en": "three", "text_vi": "ba"},
    ])
    src = tmp_path / "d.parquet"
    _write_parquet(rows, src)
    out = tmp_path / "out"
    contract = build_contract(src, output_dir=out, expected_raw_rows=3)
    assert contract["eligible_bilingual_row_count"] == 2
    assert contract["rejected_row_count"] == 1
    assert contract["rejection_reasons"].get("MISSING_ENGLISH_TEXT") == 1
    assert contract["status"] == "PASS_WITH_KNOWN_SOURCE_EXCLUSIONS"
    manifested = [json.loads(l) for l in (out / "dataset-rejected-rows.jsonl").read_text().splitlines() if l.strip()]
    assert manifested[0]["has_en"] is False


def test_contract_rejects_missing_vi(tmp_path):
    rows = _canon([
        {"qid": "Q1", "text_en": "hello", "text_vi": ""},
    ])
    src = tmp_path / "d.parquet"
    _write_parquet(rows, src)
    with pytest.raises(ValueError):
        build_contract(src, output_dir=tmp_path / "out", expected_raw_rows=1)


def test_contract_bad_row_count(tmp_path):
    src = tmp_path / "d.parquet"
    _write_parquet([{"qid": "Q1", "text_en": "a", "text_vi": "b"}], src)
    with pytest.raises(ValueError):
        build_contract(src, output_dir=tmp_path / "out", expected_raw_rows=4)


def test_contract_duplicate_qid(tmp_path):
    rows = _canon([
        {"qid": "Q1", "text_en": "a", "text_vi": "b"},
        {"qid": "Q1", "text_en": "c", "text_vi": "d"},
    ])
    src = tmp_path / "d.parquet"
    _write_parquet(rows, src)
    with pytest.raises(ValueError):
        build_contract(src, output_dir=tmp_path / "out", expected_raw_rows=2)


def test_contract_sorts_by_point_id(tmp_path):
    rows = _canon([
        {"qid": "Q20", "text_en": "twenty", "text_vi": "hai muoi"},
        {"qid": "Q3", "text_en": "three", "text_vi": "ba"},
    ])
    src = tmp_path / "d.parquet"
    _write_parquet(rows, src)
    out = tmp_path / "out"
    contract = build_contract(src, output_dir=out, expected_raw_rows=2)
    eligible = [json.loads(l) for l in (out / "canonical-eligible.jsonl").read_text().splitlines() if l.strip()]
    assert eligible[0]["qid"] == "Q3"
    assert eligible[1]["qid"] == "Q20"


def test_contract_missing_structured_field_fails_closed(tmp_path):
    rows = [
        {"qid": "Q1", "text_en": "hello", "text_vi": "xin chao"},
    ]
    src = tmp_path / "d.parquet"
    _write_parquet(rows, src)
    with pytest.raises(ValueError, match="MISSING_REQUIRED_COLUMN"):
        build_contract(src, output_dir=tmp_path / "out", expected_raw_rows=1)


def test_resolve_contract_paths(tmp_path):
    canon = tmp_path / "canonical-eligible.jsonl"
    canon.write_text("")
    contract = {
        "canonical_eligible_path": "canonical-eligible.jsonl",
        "rejection_manifest": "dataset-rejected-rows.jsonl",
    }
    resolved = resolve_contract_paths(contract, tmp_path / "dataset-contract.json")
    assert resolved["canonical_eligible_path"] == str(canon)


def test_build_emitted_text_uses_candidate_a_contract():
    assert _build_emitted_text(
        {"label_en": "Co., Ltd.", "description_en": "company"},
        "label_en", "description_en",
    ) == "Co., Ltd.. company."
    assert _build_emitted_text(
        {"label_en": "hello", "description_en": ""}, "label_en", "description_en"
    ) == "hello."
