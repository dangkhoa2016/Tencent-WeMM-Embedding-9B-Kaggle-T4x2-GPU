#!/usr/bin/env python3
"""Resumable dual-dimension bilingual Wikidata ingest into Qdrant.

Processes language texts in B4 groups, interleaving [en(r1), vi(r1), en(r2), vi(r2)].
4096 vectors come from the frozen embedding API; 1024 vectors are derived by
truncate + L2 re-normalization (never a second inference).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdrant_client import QdrantClient, models

from wemm_kaggle.search.embedding_client import EmbeddingClient, EmbeddingClientConfig
from wemm_kaggle.search.dataset import resolve_contract_paths
from wemm_kaggle.search.mrl import DERIVED_DIMENSION, FULL_DIMENSION, truncate_and_normalize
from wemm_kaggle.search.qdrant_store import (
    QdrantStore,
    COLLECTION_4096,
    COLLECTION_1024,
    QdrantSchemaError,
)


CHUNK_SIZE_QIDS = 64
DEFAULT_STATE_PATH = "/kaggle/working/wemm-v030/state/ingest-state.json"

STATUS_COMPLETE = "COMPLETE"
STATUS_CONTROLLED_STOP = "CONTROLLED_STOP"
STATUS_RUNNING = "RUNNING"

EXIT_OK = 0
EXIT_FAIL = 1

CHECKPOINT_SCHEMA_VERSION = 1
RC2_TARGET_ROWS = 99967

REQUIRED_RUNTIME_IDENTITY_ARGS = (
    "canonical-source-commit",
    "runtime-source-commit",
    "canonical-text-manifest-sha256",
    "model-manifest-sha256",
    "embedding-entrypoint-sha256",
)


def _fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fd = os.open(str(tmp), os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)
    os.replace(tmp, path)
    try:
        _fsync_dir(path.parent)
    except OSError:
        pass


def _percent(part: float, whole: float) -> str:
    return f"{100.0 * part / whole:.2f}%"


def _state_payload(
    contract: dict,
    canonical_total_rows: int,
    target_rows: int,
    completed: int,
    last_point_id,
    chunk_size: int,
    status: str,
    elapsed: float = 0.0,
    *,
    collection_4096: str = COLLECTION_4096,
    collection_1024: str = COLLECTION_1024,
    canonical_source_commit: str | None = None,
    runtime_source_commit: str | None = None,
    canonical_text_manifest_sha256: str | None = None,
    model_manifest_sha256: str | None = None,
    embedding_entrypoint_sha256: str | None = None,
) -> dict:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "canonical_source_commit": canonical_source_commit,
        "runtime_source_commit": runtime_source_commit,
        "dataset_sha256": contract["sha256"],
        "canonical_text_manifest_sha256": canonical_text_manifest_sha256,
        "model_manifest_sha256": model_manifest_sha256,
        "embedding_entrypoint_sha256": embedding_entrypoint_sha256,
        "dataset_path": contract["path"],
        "collection_4096": collection_4096,
        "collection_1024": collection_1024,
        "canonical_total_rows": canonical_total_rows,
        "target_rows": target_rows,
        "completed_rows": completed,
        "next_canonical_index": completed,
        "last_completed_point_id": last_point_id,
        "chunk_size_qids": chunk_size,
        "embedding_dimension": FULL_DIMENSION,
        "derived_dimension": DERIVED_DIMENSION,
        "raw_dataset_rows": 100000,
        "eligible_bilingual_rows": canonical_total_rows,
        "rejected_rows": 33,
        "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": status,
        "elapsed_seconds": round(elapsed, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-contract", required=True)
    parser.add_argument("--phase", required=True, choices=["5k", "100k"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--embedding-token", default=None)
    parser.add_argument("--state", default=DEFAULT_STATE_PATH)
    parser.add_argument("--qdrant-host", default="127.0.0.1")
    parser.add_argument("--qdrant-port", type=int, default=6333)
    parser.add_argument("--embedding-url", default="http://127.0.0.1:8090")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE_QIDS)
    parser.add_argument("--collection-4096", default=COLLECTION_4096)
    parser.add_argument("--collection-1024", default=COLLECTION_1024)
    for name in REQUIRED_RUNTIME_IDENTITY_ARGS:
        parser.add_argument(f"--{name}", default=None)
    args = parser.parse_args()

    identity_present = all(
        getattr(args, name.replace("-", "_")) for name in REQUIRED_RUNTIME_IDENTITY_ARGS
    )
    if args.phase == "100k" and not identity_present:
        raise SystemExit(
            "error: phase 100k requires "
            "--canonical-source-commit --runtime-source-commit "
            "--canonical-text-manifest-sha256 --model-manifest-sha256 "
            "--embedding-entrypoint-sha256"
        )

    contract_file = Path(args.dataset_contract)
    contract = resolve_contract_paths(json.loads(contract_file.read_text()), contract_file)

    if args.phase == "5k":
        target_rows = 5000
    else:
        target_rows = int(contract["eligible_bilingual_row_count"])

    rows = [json.loads(line) for line in open(contract["canonical_eligible_path"]) if line.strip()]
    canonical_total_rows = len(rows)
    if args.phase == "100k" and target_rows != canonical_total_rows:
        raise SystemExit(
            f"error: full target {target_rows} != canonical eligible {canonical_total_rows}"
        )
    if args.limit is not None:
        target_rows = min(target_rows, args.limit)

    token = args.embedding_token or os.environ.get("WEMM_API_TOKEN", "")
    if not token:
        raise SystemExit("error: WEMM_API_TOKEN required")

    embedding = EmbeddingClient(
        EmbeddingClientConfig(base_url=args.embedding_url, api_token=token)
    )
    client = QdrantClient(host=args.qdrant_host, port=args.qdrant_port, timeout=60.0)
    store = QdrantStore(
        client=client,
        collection_4096=args.collection_4096,
        collection_1024=args.collection_1024,
    )

    # Schema must exist and be exact; create only if absent.
    for dim in (FULL_DIMENSION, DERIVED_DIMENSION):
        try:
            store.verify_schema(dim, (0, 99967))
        except Exception:
            store.ensure_dual_collections((0, 99967))

    # Determine resume cursor from checkpoint.
    completed = 0
    last_point_id = None
    state_path = Path(args.state)
    payload_kwargs = dict(
        collection_4096=args.collection_4096,
        collection_1024=args.collection_1024,
        canonical_source_commit=args.canonical_source_commit,
        runtime_source_commit=args.runtime_source_commit,
        canonical_text_manifest_sha256=args.canonical_text_manifest_sha256,
        model_manifest_sha256=args.model_manifest_sha256,
        embedding_entrypoint_sha256=args.embedding_entrypoint_sha256,
    )
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise SystemExit(
                f"error: checkpoint schema_version "
                f"{state.get('schema_version')!r} != {CHECKPOINT_SCHEMA_VERSION}; "
                "REFUSE RESUME"
            )
        if state.get("dataset_sha256") != contract["sha256"]:
            raise SystemExit("error: dataset SHA mismatch; REFUSE RESUME")
        if state.get("collection_4096") != args.collection_4096:
            raise SystemExit("error: 4096 collection mismatch; REFUSE RESUME")
        if state.get("collection_1024") != args.collection_1024:
            raise SystemExit("error: 1024 collection mismatch; REFUSE RESUME")
        if state.get("chunk_size_qids") != args.chunk_size:
            raise SystemExit("error: chunk size mismatch; REFUSE RESUME")
        if identity_present:
            for arg_name in REQUIRED_RUNTIME_IDENTITY_ARGS:
                state_key = arg_name.replace("-", "_")
                expected = getattr(args, state_key)
                if state.get(state_key) != expected:
                    raise SystemExit(
                        f"error: {state_key} mismatch; REFUSE RESUME"
                    )
        completed = int(state.get("completed_rows", 0))
        last_point_id = state.get("last_completed_point_id")
        if not (0 <= completed <= RC2_TARGET_ROWS):
            raise SystemExit(
                f"error: completed_rows {completed} outside 0..{RC2_TARGET_ROWS}; "
                "REFUSE RESUME"
            )
        if completed > target_rows:
            raise SystemExit(f"error: completed {completed} > target {target_rows}; REFUSE RESUME")
        if state.get("next_canonical_index") != completed:
            raise SystemExit(f"error: next_canonical_index != completed; REFUSE RESUME")
        if completed >= target_rows:
            elapsed = 0.0
            _atomic_write(
                state_path,
                _state_payload(
                    contract, canonical_total_rows, target_rows, completed, last_point_id,
                    args.chunk_size, STATUS_COMPLETE, elapsed, **payload_kwargs,
                ),
            )
            summary = {
                "dataset_sha256": contract["sha256"],
                "dataset_path": contract["path"],
                "phase": args.phase,
                "canonical_total_rows": canonical_total_rows,
                "target_rows": target_rows,
                "completed_rows": completed,
                "last_completed_point_id": last_point_id,
                "chunk_size_qids": args.chunk_size,
                "embedding_dimension": FULL_DIMENSION,
                "derived_dimension": DERIVED_DIMENSION,
                "collection_4096": args.collection_4096,
                "collection_1024": args.collection_1024,
                "raw_dataset_rows": 100000,
                "eligible_bilingual_rows": canonical_total_rows,
                "rejected_rows": 33,
                "elapsed_seconds": round(elapsed, 3),
                "elapsed_status": "COMPLETED",
                "status": STATUS_COMPLETE,
                "numeric_exit_code": EXIT_OK,
            }
            print(json.dumps(summary, indent=2, sort_keys=True))
            return EXIT_OK
        # No chunk-boundary constraint on resume: each chunk is committed
        # atomically and the state advances only after full commits, so any
        # completed value is a valid resume cursor. This is required by the
        # exact-5000 acceptance stop (5000 is not a multiple of chunk_size),
        # after which the full 100K phase resumes from that non-aligned count.

    started = time.time()
    idx = completed

    def process_chunk(rows: list[dict]) -> tuple[int, int]:
        """Embed a chunk using B4 interleaved batches; return last point id."""
        points_4096 = []
        points_1024 = []
        buffer_text: list[str] = []
        buffer_meta: list[tuple[int, str, str]] = []  # (row_idx, lang, qid)
        emitted = {4096: 0, 1024: 0}

        def embed_buffer():
            if not buffer_text:
                return []
            vecs = embedding.embed_texts_4096(buffer_text)
            result = {}
            for (r_idx, lng, qid), vec in zip(buffer_meta, vecs):
                result[(r_idx, lng)] = vec
            buffer_text.clear()
            buffer_meta.clear()
            return result

        collected = {}
        for r_idx, row in enumerate(rows):
            # B4 grouping: [en(r0), vi(r0), en(r1), vi(r1)]
            buffer_text.append(row["text_en"])
            buffer_meta.append((r_idx, "en", row["qid"]))
            buffer_text.append(row["text_vi"])
            buffer_meta.append((r_idx, "vi", row["qid"]))
            if len(buffer_text) == 4:
                collected.update(embed_buffer())
        if buffer_text:
            collected.update(embed_buffer())

        for r_idx, row in enumerate(rows):
            full_en = collected[(r_idx, "en")]
            full_vi = collected[(r_idx, "vi")]
            mrl_en = truncate_and_normalize(full_en, DERIVED_DIMENSION)
            mrl_vi = truncate_and_normalize(full_vi, DERIVED_DIMENSION)
            payload = {
                "qid": row["qid"],
                "text_en": row["text_en"],
                "text_vi": row["text_vi"],
                "source_row_index": row["source_row_index"],
            }
            point_id = row["point_id"]
            points_4096.append(
                models.PointStruct(id=point_id, vector={"en": full_en, "vi": full_vi}, payload=payload)
            )
            points_1024.append(
                models.PointStruct(id=point_id, vector={"en": mrl_en, "vi": mrl_vi}, payload=payload)
            )

        store.upsert_points(FULL_DIMENSION, points_4096, wait=True)
        store.upsert_points(DERIVED_DIMENSION, points_1024, wait=True)
        return len(points_4096), points_4096[-1].id

    while idx < target_rows:
        remaining = target_rows - idx
        take = args.chunk_size if remaining >= args.chunk_size else remaining
        chunk_rows = rows[idx : idx + take]
        n, last_id = process_chunk(chunk_rows)
        idx += n
        completed = idx
        last_point_id = last_id
        status = STATUS_RUNNING
        _atomic_write(
            state_path,
            _state_payload(
                contract, canonical_total_rows, target_rows, completed, last_point_id,
                args.chunk_size, status, time.time() - started, **payload_kwargs,
            ),
        )
        print(
            json.dumps(
                {
                    "chunk_points": n,
                    "completed_rows": completed,
                    "target_rows": target_rows,
                    "progress": _percent(completed, target_rows),
                    "last_point_id": last_point_id,
                }
            )
        )

    final_status = STATUS_COMPLETE if completed >= target_rows else STATUS_CONTROLLED_STOP
    elapsed = time.time() - started
    _atomic_write(
        state_path,
        _state_payload(
            contract, canonical_total_rows, target_rows, completed, last_point_id,
            args.chunk_size, final_status, elapsed, **payload_kwargs,
        ),
    )
    summary = {
        "dataset_sha256": contract["sha256"],
        "dataset_path": contract["path"],
        "phase": args.phase,
        "canonical_total_rows": canonical_total_rows,
        "target_rows": target_rows,
        "completed_rows": completed,
        "last_completed_point_id": last_point_id,
        "chunk_size_qids": args.chunk_size,
        "embedding_dimension": FULL_DIMENSION,
        "derived_dimension": DERIVED_DIMENSION,
        "collection_4096": args.collection_4096,
        "collection_1024": args.collection_1024,
        "raw_dataset_rows": 100000,
        "eligible_bilingual_rows": canonical_total_rows,
        "rejected_rows": 33,
        "elapsed_seconds": round(elapsed, 3),
        "elapsed_status": "COMPLETED" if final_status == STATUS_COMPLETE else "CONTROLLED_STOP",
        "status": final_status,
        "numeric_exit_code": EXIT_OK,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
