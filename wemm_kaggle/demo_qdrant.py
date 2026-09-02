from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from qdrant_client import QdrantClient, models

from .demo_config import (
    COLLECTIONS,
    DATASET,
    EXPECTED_POINTS,
    PROJECT_ROOT,
    QDRANT_ROOT,
    QDRANT_SEAL,
    SNAPSHOTS,
)
from .demo_io import run, write_json


def _fingerprint(root: Path) -> dict:
    root = Path(root)
    if not root.is_dir():
        return {"file_count": 0, "bytes": 0, "digest": None}
    rows = []
    total = 0
    for path in sorted(root.rglob("*"), key=lambda p: str(p.relative_to(root))):
        stat = path.lstat()
        rel = str(path.relative_to(root))
        if path.is_symlink():
            kind = "symlink"
            size = int(stat.st_size)
            target = os.readlink(path)
        elif path.is_dir():
            kind = "dir"
            size = 0
            target = None
        elif path.is_file():
            kind = "file"
            size = int(stat.st_size)
            total += size
            target = None
        else:
            kind = "other"
            size = int(stat.st_size)
            target = None
        rows.append(
            [
                rel,
                kind,
                size,
                int(stat.st_mtime_ns),
                int(stat.st_ctime_ns),
                int(stat.st_ino),
                target,
            ]
        )
    payload = json.dumps(rows, separators=(",", ":"), ensure_ascii=False)
    return {
        "file_count": sum(1 for row in rows if row[1] == "file"),
        "bytes": total,
        "digest": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def _contract() -> dict:
    return {
        "qdrant_version": "1.19.0",
        "points_per_collection": EXPECTED_POINTS,
        "collections": {str(k): v for k, v in COLLECTIONS.items()},
        "snapshot_sha256": {
            str(k): v["sha256"] for k, v in SNAPSHOTS.items()
        },
        "vector_names": ["en", "vi"],
        "distance": "Cosine",
    }


def _cache_status() -> tuple[bool, str]:
    if not QDRANT_ROOT.is_dir() or not QDRANT_SEAL.is_file():
        return False, "NO_COMPLETE_SEALED_CACHE"
    try:
        seal = json.loads(QDRANT_SEAL.read_text(encoding="utf-8"))
    except Exception:
        return False, "SEAL_READ_ERROR"
    if seal.get("contract") != _contract():
        return False, "CONTRACT_MISMATCH"
    current = _fingerprint(QDRANT_ROOT)
    if current != seal.get("fingerprint"):
        return False, "STORAGE_TOUCHED_OR_CHANGED"
    return True, "SEALED_STORAGE_UNCHANGED"


def _pid_path() -> Path:
    return QDRANT_ROOT / "run" / "qdrant.pid"


def stop_qdrant() -> None:
    pid_file = _pid_path()
    if not pid_file.is_file():
        return
    try:
        pid = int(pid_file.read_text().strip())
    except Exception:
        pid_file.unlink(missing_ok=True)
        return
    proc = Path(f"/proc/{pid}")
    if not proc.exists():
        pid_file.unlink(missing_ok=True)
        return
    cmdline = (
        (proc / "cmdline")
        .read_bytes()
        .replace(b"\0", b" ")
        .decode(errors="replace")
        .lower()
    )
    if "qdrant" not in cmdline:
        raise RuntimeError(
            f"Refuse to terminate PID {pid}: process is not provably Qdrant"
        )
    os.kill(pid, signal.SIGTERM)
    deadline = time.time() + 15
    while proc.exists() and time.time() < deadline:
        time.sleep(0.2)
    if proc.exists():
        os.kill(pid, signal.SIGKILL)
        time.sleep(0.5)
    if proc.exists():
        raise RuntimeError(f"Qdrant PID {pid} did not exit")
    pid_file.unlink(missing_ok=True)


def _start_qdrant() -> QdrantClient:
    env = dict(os.environ)
    env["WEMM_QDRANT_EVIDENCE_DIR"] = str(
        QDRANT_ROOT / "public-demo-startup-evidence"
    )
    run(
        ["bash", PROJECT_ROOT / "kaggle/setup-qdrant-download.sh"],
        env=env,
        capture=True,
    )
    run(
        ["bash", PROJECT_ROOT / "kaggle/run-qdrant.sh"],
        env=env,
        capture=True,
    )
    client = QdrantClient(host="127.0.0.1", port=6333, timeout=3600)
    info = client.info()
    version = str(getattr(info, "version", ""))
    if version != "1.19.0":
        raise RuntimeError(f"Unexpected Qdrant version: {version}")
    return client


def _inspect(client: QdrantClient) -> dict:
    names = sorted(c.name for c in client.get_collections().collections)
    expected = sorted(COLLECTIONS.values())
    if names != expected:
        raise RuntimeError(
            f"Qdrant collection set mismatch: observed={names} expected={expected}"
        )
    evidence = {}
    for dimension in (4096, 1024):
        collection = COLLECTIONS[dimension]
        info = client.get_collection(collection)
        count = int(client.count(collection, exact=True).count)
        vectors = info.config.params.vectors
        if not isinstance(vectors, dict) or set(vectors) != {"en", "vi"}:
            raise RuntimeError(f"Unexpected named vectors in {collection}: {vectors}")
        for name in ("en", "vi"):
            if int(vectors[name].size) != dimension:
                raise RuntimeError(
                    f"Vector size mismatch {collection}/{name}: {vectors[name].size}"
                )
            if "COSINE" not in str(vectors[name].distance).upper():
                raise RuntimeError(
                    f"Distance mismatch {collection}/{name}: {vectors[name].distance}"
                )
        status = getattr(info.status, "value", str(info.status)).lower()
        if "green" not in status or count != EXPECTED_POINTS:
            raise RuntimeError(
                f"Collection validation failed {collection}: status={status} count={count}"
            )
        evidence[str(dimension)] = {
            "collection": collection,
            "snapshot": SNAPSHOTS[dimension]["filename"],
            "snapshot_sha256": SNAPSHOTS[dimension]["sha256"],
            "points_count": count,
            "vector_names": ["en", "vi"],
            "vector_size": dimension,
            "distance": "Cosine",
            "status": "green",
        }
    return evidence


def prepare_qdrant() -> tuple[QdrantClient, dict, str, str]:
    stop_qdrant()
    reusable, reason = _cache_status()

    if reusable:
        QDRANT_SEAL.unlink(missing_ok=True)
    else:
        if QDRANT_ROOT.exists():
            shutil.rmtree(QDRANT_ROOT)
        QDRANT_SEAL.unlink(missing_ok=True)

    client = _start_qdrant()

    if reusable:
        try:
            return client, _inspect(client), "REUSED_VERIFIED_STORAGE", reason
        except Exception:
            client.close()
            stop_qdrant()
            shutil.rmtree(QDRANT_ROOT, ignore_errors=True)
            QDRANT_SEAL.unlink(missing_ok=True)
            client = _start_qdrant()
            reusable = False
            reason = "SEMANTIC_VALIDATION_FAILED_RESTORE_REQUIRED"

    existing = [c.name for c in client.get_collections().collections]
    if existing:
        raise RuntimeError(
            f"Fresh Qdrant must contain zero collections before restore: {existing}"
        )

    evidence = {}
    stage_dir = QDRANT_ROOT / "snapshots"
    stage_dir.mkdir(parents=True, exist_ok=True)
    for dimension in (4096, 1024):
        spec = SNAPSHOTS[dimension]
        source = DATASET / spec["filename"]
        staged = stage_dir / spec["filename"]
        incoming = staged.with_name(staged.name + ".incoming")
        if not source.is_file() or source.stat().st_size != spec["bytes"]:
            raise RuntimeError(f"Snapshot byte contract failed: {source}")
        incoming.unlink(missing_ok=True)
        staged.unlink(missing_ok=True)
        t0 = time.perf_counter()
        try:
            shutil.copyfile(source, incoming)
            if incoming.stat().st_size != spec["bytes"]:
                raise RuntimeError(f"Snapshot staging size mismatch: {incoming}")
            incoming.replace(staged)
            ok = client.recover_snapshot(
                collection_name=COLLECTIONS[dimension],
                location=staged.as_uri(),
                checksum=spec["sha256"],
                priority=models.SnapshotPriority.SNAPSHOT,
                wait=True,
            )
            if ok is not True and ok is not None:
                raise RuntimeError(f"Unexpected recover return value: {ok!r}")
            evidence[str(dimension)] = {
                "collection": COLLECTIONS[dimension],
                "snapshot": spec["filename"],
                "snapshot_sha256": spec["sha256"],
                "snapshot_source": "READ_ONLY_KAGGLE_INPUT",
                "snapshot_staging_policy": (
                    "TEMPORARY_COPY_INSIDE_QDRANT_SNAPSHOT_DIR"
                ),
                "snapshot_staging_deleted_after_recovery": True,
                "persistent_snapshot_copy_after_recovery": False,
                "qdrant_checksum_verified": True,
                "elapsed_seconds": round(time.perf_counter() - t0, 3),
            }
        finally:
            incoming.unlink(missing_ok=True)
            staged.unlink(missing_ok=True)

    validated = _inspect(client)
    for dimension in ("4096", "1024"):
        evidence[dimension].update(validated[dimension])

    leftovers = [
        p.name
        for p in stage_dir.iterdir()
        if p.is_file()
        and (
            p.name in {
                SNAPSHOTS[4096]["filename"],
                SNAPSHOTS[1024]["filename"],
            }
            or p.name.endswith(".incoming")
        )
    ]
    if leftovers:
        raise RuntimeError(f"Temporary snapshots survived recovery: {leftovers}")
    return (
        client,
        evidence,
        "RESTORED_WITH_TEMPORARY_SNAPSHOT_STAGING",
        reason,
    )


def write_cache_seal(data_mode: str) -> dict:
    fingerprint = _fingerprint(QDRANT_ROOT)
    if not fingerprint["digest"]:
        raise RuntimeError("Cannot seal missing Qdrant storage")
    seal = {
        "status": "CLEAN",
        "data_mode": str(data_mode),
        "contract": _contract(),
        "fingerprint": fingerprint,
    }
    write_json(QDRANT_SEAL, seal)
    return seal
