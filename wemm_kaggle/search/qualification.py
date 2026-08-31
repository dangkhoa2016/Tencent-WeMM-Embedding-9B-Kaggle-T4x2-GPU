"""Canonical corpus identity helpers for the Wikidata EN-VI 100K Qdrant corpus.

Owns the canonical-ID and deterministic-sampling rules.

QID contract:
    QID regex = ^Q[1-9][0-9]*$
    point_id  = numeric suffix of QID
    canonical sort = ascending numeric point_id
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .base_description import build_language_text
from .point_id import qid_to_point_id

_QID_RE = re.compile(r"^Q[1-9][0-9]*$")

EXPECTED_RAW_ROWS = 100000
EXPECTED_ELIGIBLE_ROWS = 99967
EXPECTED_REJECTED_ROWS = 33

LOCKED_DATASET_SHA256 = "1f69b4b64a0aecf41505aef3cf6fa5c5ce19e4a733d198765ee6f431beb711e9"
REJECT_REASON = "MISSING_ENGLISH_TEXT"


@dataclass(frozen=True)
class CanonicalAuditRow:
    qid: str
    point_id: int
    text_en: str
    text_vi: str
    source_row_index: int


@dataclass(frozen=True)
class FreshInputSet:
    source_parquet: Path
    public_ingest_state: Path
    snapshot_1024_published: Path
    snapshot_4096_published: Path


# --- Locked safety snapshot identities ---

SNAPSHOT_1024_PUBLISHED_BASENAME = "wikidata_en_vi_wemm9b_1024_v1-4378630946280344-2026-08-30-12-52-54.s"
SNAPSHOT_1024_RESTORE_BASENAME = "wikidata_en_vi_wemm9b_1024_v1-4378630946280344-2026-08-30-12-52-54.snapshot"
SNAPSHOT_1024_SIZE = 895926784
SNAPSHOT_1024_SHA256 = "6be5a750d1ec15463df59bca8e4fb676207af8700c97e609ee48e626158b5b90"

SNAPSHOT_4096_PUBLISHED_BASENAME = "wikidata_en_vi_wemm9b_4096_v1-4378630946280344-2026-08-30-12-53-06.s"
SNAPSHOT_4096_RESTORE_BASENAME = "wikidata_en_vi_wemm9b_4096_v1-4378630946280344-2026-08-30-12-53-06.snapshot"
SNAPSHOT_4096_SIZE = 3374581760
SNAPSHOT_4096_SHA256 = "e2490a1a0fbd25f7dac34e40bce25907b7cf24f0bfd4ca7a80d8d920734ae85e"


def load_canonical_rows_from_parquet(path: Path) -> list[CanonicalAuditRow]:
    """Load and validate the canonical eligible bilingual rows from a parquet file.

    Uses the existing dataset contract logic (pyarrow parquet → rejection → eligible).
    point_id is always derived from QID suffix, never from the parquet row index.
    """
    import pyarrow.parquet as pq

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"parquet not found: {path}")

    table = pq.read_table(str(path))
    columns = [c.lower().strip() for c in table.column_names]

    qid_aliases = ("qid", "QID", "wikidata_id", "wikidata_qid")
    en_aliases = ("document_en", "text_en", "en_text", "english_text", "english", "en")
    vi_aliases = ("document_vi", "text_vi", "vi_text", "vietnamese_text", "vietnamese", "vi")

    def find_col(aliases: tuple[str, ...]) -> str | None:
        for alias in aliases:
            norm = alias.strip().lower()
            if norm in columns:
                idx = columns.index(norm)
                return table.column_names[idx]
        return None

    qid_col = find_col(qid_aliases)
    en_col = find_col(en_aliases)
    vi_col = find_col(vi_aliases)

    if qid_col is None:
        raise ValueError(f"missing QID column among {table.column_names}")
    if en_col is None:
        raise ValueError(f"missing English column among {table.column_names}")
    if vi_col is None:
        raise ValueError(f"missing Vietnamese column among {table.column_names}")

    # Text emission is restricted to the exact canonical structured
    # fields. Alternate names never substitute for them.
    from .dataset import evaluate_required_structured_schema

    evaluate_required_structured_schema(table.column_names)
    label_en_col = "label_en"
    label_vi_col = "label_vi"
    desc_en_col = "description_en"
    desc_vi_col = "description_vi"

    raw_rows = len(table)
    if raw_rows != EXPECTED_RAW_ROWS:
        raise ValueError(f"expected exactly {EXPECTED_RAW_ROWS} raw rows; got {raw_rows}")

    qid_list = table.column(qid_col).to_pylist()
    en_list = table.column(en_col).to_pylist()
    vi_list = table.column(vi_col).to_pylist()
    label_en_list = table.column(label_en_col).to_pylist()
    label_vi_list = table.column(label_vi_col).to_pylist()
    desc_en_list = table.column(desc_en_col).to_pylist()
    desc_vi_list = table.column(desc_vi_col).to_pylist()

    eligible: list[CanonicalAuditRow] = []
    seen_qids: set[str] = set()

    for idx in range(raw_rows):
        qid_raw = qid_list[idx]
        en_raw = en_list[idx]
        vi_raw = vi_list[idx]

        qid = str(qid_raw).strip() if qid_raw is not None else ""
        text_en = str(en_raw).strip() if en_raw is not None else ""
        text_vi = str(vi_raw).strip() if vi_raw is not None else ""

        if not _QID_RE.match(qid):
            raise ValueError(f"invalid QID at row {idx}: {qid!r}")
        if qid in seen_qids:
            raise ValueError(f"duplicate QID at row {idx}: {qid}")
        seen_qids.add(qid)

        if not text_vi:
            raise ValueError(f"missing Vietnamese at row {idx} (qid {qid})")
        if not text_en:
            continue

        point_id = qid_to_point_id(qid)
        eligible.append(CanonicalAuditRow(
            qid=qid,
            point_id=point_id,
            text_en=build_language_text(label_en_list[idx], desc_en_list[idx]),
            text_vi=build_language_text(label_vi_list[idx], desc_vi_list[idx]),
            source_row_index=idx,
        ))

    eligible.sort(key=lambda r: r.point_id)
    if len(eligible) != EXPECTED_ELIGIBLE_ROWS:
        raise ValueError(
            f"expected {EXPECTED_ELIGIBLE_ROWS} eligible rows; got {len(eligible)}"
        )

    return eligible


def expected_point_id_set(rows: Iterable[CanonicalAuditRow]) -> set[int]:
    """Return the set of canonical point IDs derived from QID suffixes."""
    return {row.point_id for row in rows}


def deterministic_qid_sample(
    rows: Iterable[CanonicalAuditRow],
    size: int,
) -> list[CanonicalAuditRow]:
    """Deterministic sample using SHA256(QID) ascending ordering.

    Does not use random.sample or PRNG state.
    Sorts by (sha256(qid_hex), point_id) and takes the first `size` rows.
    """
    row_list = list(rows)
    if size <= 0:
        return []
    def _sort_key(row: CanonicalAuditRow) -> tuple[str, int]:
        return (hashlib.sha256(row.qid.encode("utf-8")).hexdigest(), row.point_id)
    row_list.sort(key=_sort_key)
    return row_list[:size]


KAGGLE_INPUT_ROOT = Path("/kaggle/input")


def discover_fresh_inputs(input_root: Path | None = None) -> FreshInputSet:
    """Discover exact inputs under /kaggle/input for the safety backup workflow.

    The live Kaggle input root is enforced for production use; tests may pass
    an explicit fixture root, which must resolve strictly under /kaggle/input.

    Requires exactly one match for each:
    - wikidata-en-vi-100k.parquet
    - 1024 published .s snapshot
    - 4096 published .s snapshot
    - ingest-state.json

    Zero or multiple matches fail closed.  No lexicographic selection is used.
    """
    if input_root is None:
        input_root = KAGGLE_INPUT_ROOT
    input_root = Path(input_root)
    if not input_root.is_dir():
        raise FileNotFoundError(f"input root not found: {input_root}")
    if str(input_root.resolve()) != str(KAGGLE_INPUT_ROOT.resolve()) and not str(
        input_root.resolve()
    ).startswith(str(KAGGLE_INPUT_ROOT.resolve()) + "/"):
        raise ValueError(f"input root must be under /kaggle/input: {input_root}")

    parquets = list(input_root.rglob("wikidata-en-vi-100k.parquet"))
    if len(parquets) != 1:
        raise ValueError(
            f"expected exactly 1 wikidata-en-vi-100k.parquet; found {len(parquets)}"
        )

    snaps_1024 = list(input_root.rglob(SNAPSHOT_1024_PUBLISHED_BASENAME))
    if len(snaps_1024) != 1:
        raise ValueError(
            f"expected exactly 1 {SNAPSHOT_1024_PUBLISHED_BASENAME}; found {len(snaps_1024)}"
        )

    snaps_4096 = list(input_root.rglob(SNAPSHOT_4096_PUBLISHED_BASENAME))
    if len(snaps_4096) != 1:
        raise ValueError(
            f"expected exactly 1 {SNAPSHOT_4096_PUBLISHED_BASENAME}; found {len(snaps_4096)}"
        )

    states = list(input_root.rglob("ingest-state.json"))
    if len(states) != 1:
        raise ValueError(
            f"expected exactly 1 ingest-state.json; found {len(states)}"
        )

    return FreshInputSet(
        source_parquet=parquets[0],
        public_ingest_state=states[0],
        snapshot_1024_published=snaps_1024[0],
        snapshot_4096_published=snaps_4096[0],
    )
