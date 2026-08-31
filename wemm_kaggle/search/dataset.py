from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base_description import build_language_text, TextBuildError
from .point_id import qid_to_point_id

_QID_RE = re.compile(r"^Q[1-9][0-9]*$")

QID_ALIASES = ("qid", "QID", "wikidata_id", "wikidata_qid")
EN_ALIASES = ("document_en", "text_en", "en_text", "english_text", "english", "en")
VI_ALIASES = ("document_vi", "text_vi", "vi_text", "vietnamese_text", "vietnamese", "vi")

# Text emission may use ONLY the four canonical structured fields.
# No alternate field name may substitute for any of them.
CANONICAL_STRUCTURED_FIELDS = ("label_en", "description_en", "label_vi", "description_vi")

REJECT_MISSING_ENGLISH = "MISSING_ENGLISH_TEXT"

_SUPPORTED_EXTS = (".parquet", ".jsonl", ".json", ".csv")


@dataclass(frozen=True)
class WikidataRow:
    qid: str
    point_id: int
    text_en: str
    text_vi: str
    source_row_index: int

    def payload(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "text_en": self.text_en,
            "text_vi": self.text_vi,
            "source_row_index": self.source_row_index,
        }


@dataclass(frozen=True)
class RejectedRow:
    source_row_index: int
    qid: str
    reason: str
    has_en: bool
    has_vi: bool


@dataclass(frozen=True)
class DatasetContract:
    path: str
    sha256: str
    file_size_bytes: int
    format: str
    raw_columns: list[str]
    qid_column: str
    en_column: str
    vi_column: str
    raw_row_count: int
    valid_row_count: int
    unique_qid_count: int
    eligible_bilingual_row_count: int = 0
    rejected_row_count: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    rejection_manifest: str = ""
    rejection_summary: str = ""
    canonical_eligible_path: str = ""
    canonical_eligible_sha256: str = ""
    dataset_contract_sha256: str = ""
    status: str = "PASS"

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "file_size_bytes": self.file_size_bytes,
            "format": self.format,
            "raw_columns": list(self.raw_columns),
            "qid_column": self.qid_column,
            "en_column": self.en_column,
            "vi_column": self.vi_column,
            "raw_row_count": self.raw_row_count,
            "valid_row_count": self.valid_row_count,
            "unique_qid_count": self.unique_qid_count,
            "unique_eligible_qid_count": self.eligible_bilingual_row_count,
            "eligible_bilingual_row_count": self.eligible_bilingual_row_count,
            "rejected_row_count": self.rejected_row_count,
            "rejection_reasons": dict(self.rejection_reasons),
            "rejection_manifest": self.rejection_manifest,
            "rejection_summary": self.rejection_summary,
            "canonical_eligible_path": self.canonical_eligible_path,
            "canonical_eligible_sha256": self.canonical_eligible_sha256,
            "dataset_contract_sha256": self.dataset_contract_sha256,
            "status": self.status,
        }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    try:
        _fsync_dir(path.parent)
    except OSError:
        pass


def _fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _normalise(columns: list[str]) -> list[str]:
    return [c.strip().lower() for c in columns]


def find_column(columns: list[str], aliases: tuple[str, ...]) -> str | None:
    norm = _normalise(columns)
    for alias in aliases:
        key = alias.strip().lower()
        if key in norm:
            idx = norm.index(key)
            return columns[idx]
    return None


def count_candidate(columns: list[str], aliases: tuple[str, ...]) -> int:
    norm_cols = set(_normalise(columns))
    norm_aliases = {alias.strip().lower() for alias in aliases}
    return sum(1 for alias in norm_aliases if alias in norm_cols)


def _read_table_rows(path: Path, fmt: str) -> tuple[list[str], list[dict[str, Any]]]:
    if fmt == "parquet":
        import pyarrow.parquet as pq

        table = pq.read_table(str(path))
        columns = list(table.column_names)
        rows: list[dict[str, Any]] = []
        for batch in table.to_batches():
            pd = batch.to_pydict()
            keys = list(pd.keys())
            n = len(pd[keys[0]])
            for i in range(n):
                rows.append({k: pd[k][i] for k in keys})
        return columns, rows
    if fmt == "jsonl":
        columns: list[str] = []
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                for key in obj:
                    if key not in columns:
                        columns.append(key)
                rows.append(obj)
        return columns, rows
    if fmt == "json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            if "rows" in data and isinstance(data["rows"], list):
                rows = data["rows"]
            elif "data" in data and isinstance(data["data"], list):
                rows = data["data"]
            else:
                rows = [data]
        else:
            rows = data
        columns = list({k for r in rows if isinstance(r, dict) for k in r})
        return columns, rows
    if fmt == "csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        columns = list(rows[0].keys()) if rows else []
        return columns, rows
    raise ValueError(f"unsupported format {fmt}")


def _text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return value.strip()


def validate_structured_field_overrides(
    label_en_column_override: str | None,
    description_en_column_override: str | None,
    label_vi_column_override: str | None,
    description_vi_column_override: str | None,
) -> None:
    """Bind each structured-field override to its exact field.

    Each override parameter is independent and parameter-specific. It accepts
    only its own exact canonical field name (or None); it may NOT be redirected
    to another structured field, an alias, or a differently-cased/whitespace
    variant of its own field. No hidden normalization is applied.
    """
    spec: list[tuple[str, str | None, str]] = [
        ("label_en_column_override", label_en_column_override, "label_en"),
        ("description_en_column_override", description_en_column_override, "description_en"),
        ("label_vi_column_override", label_vi_column_override, "label_vi"),
        ("description_vi_column_override", description_vi_column_override, "description_vi"),
    ]
    for name, value, canonical in spec:
        if value is not None and value != canonical:
            raise ValueError(
                f"NONCANONICAL_FIELD_OVERRIDE_FORBIDDEN: {name}={value!r} must be "
                f"exactly {canonical!r} or None"
            )


def evaluate_required_structured_schema(columns: list[str]) -> None:
    """D8.4: fail closed when any canonical structured column is absent.

    Exact source identity is enforced: the four canonical structured field names
    must be present verbatim. Alternate names, case variants, and surrounding
    whitespace never substitute. Raises ValueError with a stable
    MISSING_REQUIRED_COLUMN reason.
    """
    missing = [c for c in CANONICAL_STRUCTURED_FIELDS if c not in columns]
    if missing:
        raise ValueError(
            f"MISSING_REQUIRED_COLUMN: {','.join(missing)} not present among "
            f"{columns}"
        )


def _build_emitted_text(row: dict[str, Any], label_col: str, desc_col: str) -> str:
    """Build canonical bilingual output text from the exact structured fields.

    Routes through ``wemm-v030-canonical-bilingual-base-description-v1`` via
    ``build_language_text``. This is the ACTIVE production text-construction
    path consumed by ingest. Eligibility (document fields) is decided
    separately and never used to shape this output.
    """
    label = row.get(label_col)
    description = row.get(desc_col)
    return build_language_text(label, description)


def build_contract(
    path: Path,
    *,
    output_dir: Path,
    qid_column_override: str | None = None,
    en_column_override: str | None = None,
    vi_column_override: str | None = None,
    label_en_column_override: str | None = None,
    description_en_column_override: str | None = None,
    label_vi_column_override: str | None = None,
    description_vi_column_override: str | None = None,
    expected_raw_rows: int = 100000,
) -> dict[str, Any]:
    validate_structured_field_overrides(
        label_en_column_override=label_en_column_override,
        description_en_column_override=description_en_column_override,
        label_vi_column_override=label_vi_column_override,
        description_vi_column_override=description_vi_column_override,
    )
    fmt = "parquet" if path.suffix == ".parquet" else (
        "jsonl" if path.suffix == ".jsonl" else ("json" if path.suffix == ".json" else "csv")
    )
    columns, raw_rows = _read_table_rows(path, fmt)
    raw_row_count = len(raw_rows)
    if raw_row_count != expected_raw_rows:
        raise ValueError(f"expected exactly {expected_raw_rows} raw rows; got {raw_row_count}")

    qid_column = qid_column_override or find_column(columns, QID_ALIASES)
    en_column = en_column_override or find_column(columns, EN_ALIASES)
    vi_column = vi_column_override or find_column(columns, VI_ALIASES)

    label_en_column = label_en_column_override or "label_en"
    desc_en_column = description_en_column_override or "description_en"
    label_vi_column = label_vi_column_override or "label_vi"
    desc_vi_column = description_vi_column_override or "description_vi"

    if qid_column_override is None:
        if qid_column is None or count_candidate(columns, QID_ALIASES) != 1:
            raise ValueError(f"ambiguous/missing QID column among {columns}")
    if en_column_override is None:
        if en_column is None or count_candidate(columns, EN_ALIASES) != 1:
            raise ValueError(f"ambiguous/missing EN column among {columns}")
    if vi_column_override is None:
        if vi_column is None or count_candidate(columns, VI_ALIASES) != 1:
            raise ValueError(f"ambiguous/missing VI column among {columns}")

    # RC2 text construction requires the exact structured fields. If a canonical
    # label/description column is absent the contract cannot be satisfied and the
    # build fails closed (MISSING_REQUIRED_COLUMN). Alternate names never substitute.
    evaluate_required_structured_schema(columns)

    seen_qids: dict[str, int] = {}
    rejected: list[RejectedRow] = []
    eligible: list[WikidataRow] = []

    for idx, row in enumerate(raw_rows):
        qid = _text(row.get(qid_column, ""))
        text_en = _text(row.get(en_column, ""))
        text_vi = _text(row.get(vi_column, ""))
        has_en = bool(text_en)
        has_vi = bool(text_vi)

        if not _QID_RE.match(qid):
            raise ValueError(f"invalid QID at source row {idx}: {qid!r}")
        if qid in seen_qids:
            raise ValueError(f"duplicate QID at source row {idx}: {qid}")
        seen_qids[qid] = idx

        if not has_vi:
            raise ValueError(f"missing Vietnamese text at source row {idx} (qid {qid})")
        if not has_en:
            rejected.append(RejectedRow(idx, qid, REJECT_MISSING_ENGLISH, has_en=False, has_vi=True))
            continue

        en_text = _build_emitted_text(row, label_en_column, desc_en_column)
        vi_text = _build_emitted_text(row, label_vi_column, desc_vi_column)

        eligible.append(
            WikidataRow(
                qid=qid,
                point_id=qid_to_point_id(qid),
                text_en=en_text,
                text_vi=vi_text,
                source_row_index=idx,
            )
        )

    eligible.sort(key=lambda r: r.point_id)
    rejected.sort(key=lambda r: r.qid)

    reasons: dict[str, int] = {}
    for r in rejected:
        reasons[r.reason] = reasons.get(r.reason, 0) + 1

    output_dir.mkdir(parents=True, exist_ok=True)
    rejected_lines = "\n".join(
        json.dumps(
            {
                "source_row_index": r.source_row_index,
                "qid": r.qid,
                "reason": r.reason,
                "has_en": r.has_en,
                "has_vi": r.has_vi,
            },
            sort_keys=True,
        )
        for r in rejected
    )
    rejected_path = output_dir / "dataset-rejected-rows.jsonl"
    _atomic_write(rejected_path, rejected_lines + ("\n" if rejected_lines else ""))

    summary = {
        "raw_row_count": 100000,
        "eligible_bilingual_row_count": len(eligible),
        "rejected_row_count": len(rejected),
        "rejection_reasons": reasons,
        "unexpected_rejection_reasons": 0,
        "status": "PASS_WITH_KNOWN_SOURCE_EXCLUSIONS" if rejected else "PASS",
    }
    summary_path = output_dir / "dataset-rejection-summary.json"
    _atomic_write(summary_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")

    eligible_lines = "\n".join(
        json.dumps(
            {
                "point_id": r.point_id,
                "qid": r.qid,
                "source_row_index": r.source_row_index,
                "text_en": r.text_en,
                "text_vi": r.text_vi,
            },
            sort_keys=True,
        )
        for r in eligible
    )
    canonical_path = output_dir / "canonical-eligible.jsonl"
    _atomic_write(canonical_path, eligible_lines + "\n")

    contract = DatasetContract(
        path=str(path),
        sha256=_sha256_file(path),
        file_size_bytes=path.stat().st_size,
        format=fmt,
        raw_columns=list(columns),
        qid_column=qid_column,
        en_column=en_column,
        vi_column=vi_column,
        raw_row_count=raw_row_count,
        valid_row_count=len(eligible),
        unique_qid_count=len(seen_qids),
        eligible_bilingual_row_count=len(eligible),
        rejected_row_count=len(rejected),
        rejection_reasons=reasons,
        rejection_manifest=str(rejected_path.relative_to(output_dir)),
        rejection_summary=str(summary_path.relative_to(output_dir)),
        canonical_eligible_path=str(canonical_path.relative_to(output_dir)),
        canonical_eligible_sha256=_sha256_file(canonical_path),
        status="PASS_WITH_KNOWN_SOURCE_EXCLUSIONS" if rejected else "PASS",
    )
    return contract.to_dict()


def resolve_contract_paths(contract: dict[str, Any], contract_file: Path) -> dict[str, Any]:
    base = contract_file.resolve().parent
    for key in ("canonical_eligible_path", "rejection_manifest", "rejection_summary"):
        value = contract.get(key)
        if value is None:
            continue
        path = Path(value)
        if not path.is_absolute():
            contract[key] = str(base / path)
    return contract
