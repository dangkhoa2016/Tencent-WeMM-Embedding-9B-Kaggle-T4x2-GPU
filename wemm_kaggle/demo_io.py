from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys


def banner(step: str, title: str) -> None:
    print("\n" + "=" * 92, flush=True)
    print(f"[{step}] {title}", flush=True)
    print("=" * 92, flush=True)


def kv(label: str, value) -> None:
    print(f"  {label:<36} : {value}", flush=True)


def run(cmd, *, cwd=None, env=None, capture=False):
    result = subprocess.run(
        list(map(str, cmd)),
        cwd=str(cwd) if cwd else None,
        env=env,
        check=False,
        text=True,
        capture_output=capture,
    )
    if result.returncode:
        if capture and result.stdout:
            print(result.stdout, flush=True)
        if capture and result.stderr:
            print(result.stderr, file=sys.stderr, flush=True)
        raise subprocess.CalledProcessError(
            result.returncode,
            list(map(str, cmd)),
            output=result.stdout,
            stderr=result.stderr,
        )
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def truncate_vector(vector: list[float], dimension: int) -> list[float]:
    values = [float(x) for x in vector[: int(dimension)]]
    norm = math.sqrt(sum(x * x for x in values))
    if not math.isfinite(norm) or norm <= 0:
        raise RuntimeError("Invalid MRL vector norm")
    return [x / norm for x in values]


def gpu_snapshot() -> list[dict]:
    output = run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        capture=True,
    ).stdout
    rows = []
    for line in output.splitlines():
        index, name, total, used, free = [x.strip() for x in line.split(",")]
        rows.append(
            {
                "index": int(index),
                "name": name,
                "memory_total_mib": int(total),
                "memory_used_mib": int(used),
                "memory_free_mib": int(free),
            }
        )
    return rows


def gpu_processes() -> list[str]:
    output = run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture=True,
    ).stdout.strip()
    return [line.strip() for line in output.splitlines() if line.strip()]
