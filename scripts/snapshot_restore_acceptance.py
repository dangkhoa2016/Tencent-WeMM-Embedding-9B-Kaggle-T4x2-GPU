#!/usr/bin/env python3
"""Qdrant snapshot creation and V4 future-bootstrap restore acceptance.

phase=create: record counts, create snapshots for both collections, verify
              snapshot files exist on disk.
phase=restore: bootstrap-restore each collection from a published .s snapshot
              using locked size/SHA constants from qualification.py.  Copies
              to a dedicated /kaggle/working restore directory, re-verifies
              the copy, then performs qdrant recover_snapshot with full
              checksum enforcement.  Any mismatch fails closed.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import shutil
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx
from qdrant_client import QdrantClient
from qdrant_client import models

from wemm_kaggle.search.config import QDRANT_HOST, QDRANT_PORT
from wemm_kaggle.search.qualification import (
    SNAPSHOT_1024_PUBLISHED_BASENAME,
    SNAPSHOT_1024_RESTORE_BASENAME,
    SNAPSHOT_1024_SHA256,
    SNAPSHOT_1024_SIZE,
    SNAPSHOT_4096_PUBLISHED_BASENAME,
    SNAPSHOT_4096_RESTORE_BASENAME,
    SNAPSHOT_4096_SHA256,
    SNAPSHOT_4096_SIZE,
)
from wemm_kaggle.search.qdrant_store import COLLECTION_4096, COLLECTION_1024


JSON_HEADERS = {"content-type": "application/json"}

EXPECTED_VECTOR_NAMES = {"en", "vi"}
EXPECTED_DISTANCE = "Cosine"
EXPECTED_POINTS_COUNT = 99967


def _counts(client: QdrantClient) -> dict:
    return {
        COLLECTION_4096: client.count(COLLECTION_4096, exact=True).count,
        COLLECTION_1024: client.count(COLLECTION_1024, exact=True).count,
    }


def _wait_count(client: QdrantClient, collection: str, expected: int, timeout_s: float = 600.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if client.count(collection, exact=True).count == expected:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def _probe_search(gateway: str, query: str, qlng: str, tlng: str) -> dict:
    try:
        resp = httpx.post(
            f"{gateway}/v1/search",
            headers=JSON_HEADERS,
            json={
                "query": query,
                "query_language": qlng,
                "target_language": tlng,
                "dimension": 4096,
                "limit": 5,
            },
            timeout=120.0,
        )
        if resp.status_code == 200:
            body = resp.json()
            return {"http": 200, "hits": len(body.get("hits", [])), "top_qids": [h["qid"] for h in body.get("hits", [])]}
        return {"http": resp.status_code, "hits": -1}
    except httpx.HTTPError as exc:
        return {"http": 0, "error": str(exc)}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_schema(client: QdrantClient, collection: str, expected_dim: int) -> dict:
    """Validate collection schema after restore. Fails closed on any mismatch.

    Handles the real qdrant-client 1.19.0 shape ``CollectionInfo.config.params.vectors``,
    which is a ``dict[str, VectorParams]`` for a named-vector collection (or a
    single ``VectorParams`` for a single-vector collection).  Requires the exact
    named-vector set {"en","vi"}, each of the expected dimension and Cosine
    distance, and an exact ``points_count`` of 99967.  Any non-dict vector
    configuration shape fails closed.
    """
    errors: list[str] = []
    details: dict = {}

    try:
        info = client.get_collection(collection)
    except Exception as exc:
        return {"valid": False, "errors": [f"failed to get collection info: {exc}"], "details": {}}

    vectors_cfg = None
    config = getattr(info, "config", None)
    if config is not None:
        params = getattr(config, "params", None)
        if params is not None:
            vectors_cfg = getattr(params, "vectors", None)

    if not isinstance(vectors_cfg, dict):
        errors.append(
            "expected named-vector mapping {'en','vi'}; "
            f"got {type(vectors_cfg).__name__}"
        )
    else:
        named_vectors: dict = vectors_cfg
        actual_names = set(named_vectors.keys())
        if actual_names != set(EXPECTED_VECTOR_NAMES):
            errors.append(f"vector names mismatch: expected {sorted(EXPECTED_VECTOR_NAMES)}, got {sorted(actual_names)}")
        details["vector_names"] = sorted(actual_names)
        for name, vec_param in named_vectors.items():
            actual_dim = getattr(vec_param, "size", None)
            actual_distance = str(getattr(vec_param, "distance", None))
            if actual_dim != expected_dim:
                errors.append(f"vector '{name}' dimension mismatch: expected {expected_dim}, got {actual_dim}")
            if actual_distance != EXPECTED_DISTANCE:
                errors.append(f"vector '{name}' distance mismatch: expected {EXPECTED_DISTANCE}, got {actual_distance}")
            details[f"vector_{name}_dim"] = actual_dim
            details[f"vector_{name}_distance"] = actual_distance

    points_count = getattr(info, "points_count", None)
    if not isinstance(points_count, int) or points_count != EXPECTED_POINTS_COUNT:
        errors.append(
            f"points_count mismatch: expected {EXPECTED_POINTS_COUNT}, got {points_count!r}"
        )
    details["points_count"] = points_count
    details["expected_points"] = EXPECTED_POINTS_COUNT

    return {"valid": not errors, "errors": errors, "details": details}


# ---------------------------------------------------------------------------
# V4: bootstrap and restore functions
# ---------------------------------------------------------------------------

def bootstrap_restore_snapshot(
    snapshot_published: Path,
    restore_dir: Path,
    collection: str,
    locked_size: int,
    locked_sha: str,
    published_basename: str,
    restore_basename: str,
) -> dict:
    """Validate a published .s snapshot against locked constants, copy to a
    dedicated restore directory, and re-verify the copy.

    /kaggle/input is treated as read-only; the copy lands under restore_dir.
    Every step fails closed: any mismatch raises or returns valid=False.
    """
    errors: list[str] = []

    # --- Source must be a regular file ---
    if not snapshot_published.is_file():
        errors.append(f"source snapshot does not exist: {snapshot_published}")
        return {
            "source_snapshot_path": str(snapshot_published),
            "source_basename_validated": False,
            "source_size_validated": False,
            "source_sha256_verified": False,
            "destination_path": None,
            "destination_basename": None,
            "destination_size_verified": False,
            "destination_sha256_verified": False,
            "valid": False,
            "errors": errors,
        }

    # --- Exact published basename match ---
    if snapshot_published.name != published_basename:
        errors.append(
            f"published basename mismatch: expected {published_basename!r}, got {snapshot_published.name!r}"
        )

    # --- Exact byte size ---
    source_size = snapshot_published.stat().st_size
    if source_size != locked_size:
        errors.append(f"source size mismatch: expected {locked_size}, got {source_size}")

    # --- SHA256 against locked hash ---
    if not errors:
        source_sha = _sha256_file(snapshot_published)
    else:
        source_sha = None
    if source_sha is not None and source_sha != locked_sha:
        errors.append(f"source SHA256 mismatch: expected {locked_sha}, got {source_sha}")

    if errors:
        return {
            "source_snapshot_path": str(snapshot_published),
            "source_basename_validated": snapshot_published.name == published_basename,
            "source_size_validated": source_size == locked_size,
            "source_sha256_verified": source_sha == locked_sha if source_sha else False,
            "destination_path": None,
            "destination_basename": None,
            "destination_size_verified": False,
            "destination_sha256_verified": False,
            "valid": False,
            "errors": errors,
        }

    # --- Copy to restore directory with the exact restore basename ---
    restore_dir.mkdir(parents=True, exist_ok=True)
    destination_path = restore_dir / restore_basename
    try:
        shutil.copy2(str(snapshot_published), str(destination_path))
    except OSError as exc:
        errors.append(f"copy failed: {exc}")
        return {
            "source_snapshot_path": str(snapshot_published),
            "source_basename_validated": True,
            "source_size_validated": True,
            "source_sha256_verified": True,
            "destination_path": str(destination_path),
            "destination_basename": restore_basename,
            "destination_size_verified": False,
            "destination_sha256_verified": False,
            "valid": False,
            "errors": errors,
        }

    # --- Destination basename must match ---
    if destination_path.name != restore_basename:
        errors.append(
            f"destination basename mismatch: expected {restore_basename!r}, got {destination_path.name!r}"
        )

    # --- Re-verify destination size ---
    dest_size = destination_path.stat().st_size
    if dest_size != locked_size:
        errors.append(f"destination size mismatch: expected {locked_size}, got {dest_size}")

    # --- Re-verify destination SHA256 ---
    dest_sha = _sha256_file(destination_path)
    if dest_sha != locked_sha:
        errors.append(f"destination SHA256 mismatch: expected {locked_sha}, got {dest_sha}")

    return {
        "source_snapshot_path": str(snapshot_published),
        "source_basename_validated": True,
        "source_size_validated": True,
        "source_sha256_verified": True,
        "destination_path": str(destination_path),
        "destination_basename": destination_path.name,
        "destination_size_verified": dest_size == locked_size,
        "destination_sha256_verified": dest_sha == locked_sha,
        "valid": not errors,
        "errors": errors,
    }


def perform_restore(
    client: QdrantClient,
    collection: str,
    destination_path: Path,
    locked_sha: str,
    expected_points: int,
) -> dict:
    """Perform the qdrant recover_snapshot call from an exact destination path.

    Gates on the installed API supporting checksum/priority/wait via
    inspect.signature.  Returns a dict with restoration outcome and count
    verification.
    """
    # --- Mandatory API parameter gate ---
    params = inspect.signature(client.recover_snapshot).parameters
    for name in ("checksum", "priority", "wait"):
        if name not in params:
            raise RuntimeError(f"qdrant-client recover_snapshot missing required parameter: {name}")

    if not destination_path.is_file():
        return {"restored": False, "from_path": str(destination_path), "count_ok": False, "error": "destination path does not exist"}

    location = f"file://{str(destination_path)}"
    client.recover_snapshot(
        collection_name=collection,
        location=location,
        checksum=locked_sha,
        priority=models.SnapshotPriority.SNAPSHOT,
        wait=True,
    )

    ok = _wait_count(client, collection, expected_points)
    return {"restored": ok, "from_path": str(destination_path), "count_ok": ok}


# ---------------------------------------------------------------------------
# Phase functions
# ---------------------------------------------------------------------------

def phase_create(
    qdrant_host: str,
    qdrant_port: int,
    snapshot_dir: Path,
    expected_points: int,
    gateway: str,
) -> dict:
    client = QdrantClient(host=qdrant_host, port=qdrant_port, timeout=120.0)
    pre_counts = _counts(client)
    created = {}
    for collection in (COLLECTION_4096, COLLECTION_1024):
        snap = client.create_snapshot(collection_name=collection)
        created[collection] = snap.name

    files = {}
    all_present = True
    for collection in (COLLECTION_4096, COLLECTION_1024):
        col_dir = snapshot_dir / collection
        matching = list(col_dir.glob("*.snapshot")) if col_dir.exists() else []
        files[collection] = [p.name for p in matching]
        if not matching:
            all_present = False

    return {
        "phase": "create",
        "pre_counts": pre_counts,
        "expected_points": expected_points,
        "snapshot_created": created,
        "snapshot_dir": str(snapshot_dir),
        "snapshot_files": files,
        "snapshot_files_present": all_present,
        "counts_match": all(pre_counts[c] == expected_points for c in (COLLECTION_4096, COLLECTION_1024)),
        "probe_before": _probe_search(gateway, "semantic search over wikidata entities", "en", "vi"),
    }


def phase_restore(
    qdrant_host: str,
    qdrant_port: int,
    restore_dir: Path,
    expected_points: int,
    gateway: str,
    snapshot_1024: Path,
    snapshot_4096: Path,
    probe_gateway: bool = False,
) -> dict:
    client = QdrantClient(host=qdrant_host, port=qdrant_port, timeout=120.0)
    absent_before = {}
    bootstrap_results = {}
    restore_results = {}
    schema_checks = {}

    collections_config = [
        (
            COLLECTION_1024,
            snapshot_1024,
            SNAPSHOT_1024_SIZE,
            SNAPSHOT_1024_SHA256,
            SNAPSHOT_1024_PUBLISHED_BASENAME,
            SNAPSHOT_1024_RESTORE_BASENAME,
            1024,
        ),
        (
            COLLECTION_4096,
            snapshot_4096,
            SNAPSHOT_4096_SIZE,
            SNAPSHOT_4096_SHA256,
            SNAPSHOT_4096_PUBLISHED_BASENAME,
            SNAPSHOT_4096_RESTORE_BASENAME,
            4096,
        ),
    ]

    for (
        collection,
        published_path,
        locked_size,
        locked_sha,
        pub_basename,
        rest_basename,
        expected_dim,
    ) in collections_config:
        absent_before[collection] = not client.collection_exists(collection)

        # --- V4: bootstrap from locked constants ---
        bootstrap = bootstrap_restore_snapshot(
            snapshot_published=published_path,
            restore_dir=restore_dir,
            collection=collection,
            locked_size=locked_size,
            locked_sha=locked_sha,
            published_basename=pub_basename,
            restore_basename=rest_basename,
        )
        bootstrap_results[collection] = bootstrap

        if not bootstrap["valid"]:
            restore_results[collection] = {
                "restored": False,
                "reason": "bootstrap validation failed",
                "errors": bootstrap.get("errors", []),
            }
            schema_checks[collection] = {"valid": False, "errors": ["skipped: bootstrap failed"], "details": {}}
            continue

        dest_path = Path(bootstrap["destination_path"])

        # --- Target collection must not already exist ---
        if client.collection_exists(collection):
            restore_results[collection] = {
                "restored": False,
                "reason": "target collection already exists; refusing",
            }
            schema_checks[collection] = {"valid": False, "errors": ["skipped: collection exists"], "details": {}}
            continue

        # --- V4: restore from verified destination path only ---
        restore_result = perform_restore(
            client=client,
            collection=collection,
            destination_path=dest_path,
            locked_sha=locked_sha,
            expected_points=expected_points,
        )
        restore_results[collection] = restore_result

        if restore_result["restored"]:
            schema_checks[collection] = _validate_schema(client, collection, expected_dim)
        else:
            schema_checks[collection] = {"valid": False, "errors": ["skipped: restore did not succeed"], "details": {}}

    post_counts = _counts(client)
    count_ok = all(post_counts[c] == expected_points for c in (COLLECTION_4096, COLLECTION_1024))
    schema_ok = all(schema_checks[c]["valid"] for c in (COLLECTION_4096, COLLECTION_1024))

    probe = _probe_search(gateway, "semantic search over wikidata entities", "en", "vi") if probe_gateway else {"http": None, "hits": None, "enabled": False}
    probe_independent = not probe_gateway or (probe["http"] == 200 and probe["hits"] > 0)

    verdict = "PASS" if count_ok and schema_ok and probe_independent else "FAIL"

    return {
        "phase": "restore",
        "absent_before": absent_before,
        "bootstrap_results": bootstrap_results,
        "restored": restore_results,
        "schema_checks": schema_checks,
        "post_counts": post_counts,
        "expected_points": expected_points,
        "counts_preserved": count_ok,
        "schema_valid": schema_ok,
        "probe_gateway_enabled": probe_gateway,
        "probe_after": probe,
        "verdict": verdict,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["create", "restore"])
    parser.add_argument("--gateway-url", default="http://127.0.0.1:8091")
    parser.add_argument("--qdrant-host", default=QDRANT_HOST)
    parser.add_argument("--qdrant-port", type=int, default=QDRANT_PORT)
    parser.add_argument("--snapshot-dir", default="/kaggle/working/wemm-qdrant/snapshots")
    parser.add_argument("--expected-points", type=int, required=True)
    parser.add_argument("--out", required=True)

    parser.add_argument(
        "--snapshot-1024",
        type=Path,
        default=None,
        help="Published .s path for the 1024-collection snapshot (read-only input)",
    )
    parser.add_argument(
        "--snapshot-4096",
        type=Path,
        default=None,
        help="Published .s path for the 4096-collection snapshot (read-only input)",
    )
    parser.add_argument(
        "--restore-dir",
        type=Path,
        default=Path("/kaggle/working/wemm-qdrant/restore"),
        help="Dedicated write directory for verified copies (default: /kaggle/working/wemm-qdrant/restore)",
    )
    parser.add_argument(
        "--probe-gateway",
        action="store_true",
        default=False,
        help="Run the optional live gateway search probe (independent of structural verdict)",
    )
    args = parser.parse_args()

    if args.phase == "restore":
        missing = []
        if args.snapshot_1024 is None:
            missing.append("--snapshot-1024")
        if args.snapshot_4096 is None:
            missing.append("--snapshot-4096")
        if missing:
            print(f"restore phase requires: {', '.join(missing)}", file=sys.stderr)
            return 1

    snapshot_dir = Path(args.snapshot_dir)
    if args.phase == "create":
        report = phase_create(
            args.qdrant_host, args.qdrant_port, snapshot_dir, args.expected_points, args.gateway_url
        )
    else:
        report = phase_restore(
            args.qdrant_host,
            args.qdrant_port,
            restore_dir=args.restore_dir,
            expected_points=args.expected_points,
            gateway=args.gateway_url,
            snapshot_1024=args.snapshot_1024,
            snapshot_4096=args.snapshot_4096,
            probe_gateway=args.probe_gateway,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))

    if args.phase == "restore" and report["verdict"] != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
