#!/usr/bin/env python3
"""Discover the attached Wikidata EN-VI 100K dataset and emit a fail-closed contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wemm_kaggle.search.dataset import build_contract, _sha256_file, _atomic_write


EXPECTED_RAW_ROWS = 100000
EXPECTED_ELIGIBLE = 99967
EXPECTED_REJECTED = 33


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument("--output", default="/kaggle/working/wemm-v030/evidence/dataset-contract.json")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--qid-column", default=None)
    parser.add_argument("--en-column", default=None)
    parser.add_argument("--vi-column", default=None)
    args = parser.parse_args()

    output = Path(args.output)
    evidence_dir = output.parent

    candidates: list[Path] = []
    if args.dataset_path:
        p = Path(args.dataset_path)
        if p.exists():
            candidates.append(p)
    else:
        root = Path(args.input_root)
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix in (".parquet", ".jsonl", ".json", ".csv"):
                name = p.name.lower()
                if ("wikidata" in name or "en-vi" in name or "en_vi" in name or "semantic" in name) and p.suffix == ".parquet":
                    candidates.append(p)

    if not candidates:
        raise SystemExit("error: no Wikidata dataset candidate found; fail-closed")
    if len(candidates) > 1:
        raise SystemExit(f"error: ambiguous dataset candidates: {candidates}; fail-closed")

    source = candidates[0]
    contract_dir = evidence_dir
    contract = build_contract(
        source,
        output_dir=contract_dir,
        qid_column_override=args.qid_column,
        en_column_override=args.en_column,
        vi_column_override=args.vi_column,
    )

    if contract["raw_row_count"] != EXPECTED_RAW_ROWS:
        raise SystemExit(f"error: raw rows {contract['raw_row_count']} != {EXPECTED_RAW_ROWS}")
    if contract["eligible_bilingual_row_count"] != EXPECTED_ELIGIBLE:
        raise SystemExit(
            f"error: eligible {contract['eligible_bilingual_row_count']} != {EXPECTED_ELIGIBLE}"
        )
    if contract["rejected_row_count"] != EXPECTED_REJECTED:
        raise SystemExit(
            f"error: rejected {contract['rejected_row_count']} != {EXPECTED_REJECTED}"
        )
    unexpected = {k: v for k, v in contract["rejection_reasons"].items() if k != "MISSING_ENGLISH_TEXT"}
    if unexpected:
        raise SystemExit(f"error: unexpected rejection reasons: {unexpected}")

    contract_path = (
        contract_dir / "dataset-contract.json"
        if not output.name.startswith("dataset-contract")
        else output
    )
    _atomic_write(contract_path, json.dumps(contract, indent=2, sort_keys=True) + "\n")

    discovery = {
        "selected_path": contract["path"],
        "sha256": contract["sha256"],
        "format": contract["format"],
        "raw_columns": contract["raw_columns"],
        "mapping": {
            "qid": contract["qid_column"],
            "en": contract["en_column"],
            "vi": contract["vi_column"],
        },
        "raw_row_count": contract["raw_row_count"],
        "eligible_bilingual_row_count": contract["eligible_bilingual_row_count"],
        "rejected_row_count": contract["rejected_row_count"],
        "unique_qid_count": contract["unique_qid_count"],
        "duplicate_qids": 0,
        "empty_en": contract["rejection_reasons"].get("MISSING_ENGLISH_TEXT", 0),
        "empty_vi": 0,
        "rejection_reasons": contract["rejection_reasons"],
        "status": "PASS_WITH_KNOWN_SOURCE_EXCLUSIONS",
    }
    _atomic_write(evidence_dir / "dataset-discovery.json", json.dumps(discovery, indent=2, sort_keys=True) + "\n")

    file_sha_path = evidence_dir / "dataset-file.sha256"
    file_sha_path.write_text(f"{contract['sha256']}  {contract['path']}\n")

    for rel, content in (
        ("dataset-rejected-rows.jsonl", None),
        ("dataset-rejection-summary.json", None),
        ("canonical-eligible.jsonl", None),
    ):
        p = contract_dir / rel
        if p.exists():
            sidecar = evidence_dir / f"{rel}.sha256"
            sidecar.write_text(f"{_sha256_file(p)}  {rel}\n")

    print(json.dumps(discovery, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
