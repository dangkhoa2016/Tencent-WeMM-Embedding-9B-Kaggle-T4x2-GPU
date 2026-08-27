from __future__ import annotations

import hashlib
import json
import os
import resource
import subprocess
import zipfile
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, raw = line.split(":", 1)
            parts = raw.strip().split()
            if parts:
                values[key] = int(parts[0]) * 1024
    except (OSError, ValueError):
        pass
    return values


def _rss_bytes() -> int:
    try:
        pages = int(Path("/proc/self/statm").read_text().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak_kib) * 1024


def process_snapshot() -> dict[str, int]:
    mem = _meminfo()
    return {
        "pid": os.getpid(),
        "rss_bytes": _rss_bytes(),
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "host_mem_total_bytes": mem.get("MemTotal", 0),
        "host_mem_available_bytes": mem.get("MemAvailable", 0),
        "swap_total_bytes": mem.get("SwapTotal", 0),
        "swap_free_bytes": mem.get("SwapFree", 0),
    }


def nvidia_smi_snapshot() -> dict[str, Any]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc), "gpus": []}
    if proc.returncode != 0:
        return {"ok": False, "error": proc.stderr.strip(), "gpus": []}
    gpus = []
    for line in proc.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 7:
            continue
        gpus.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_total_mib": int(parts[2]),
                "memory_used_mib": int(parts[3]),
                "memory_free_mib": int(parts[4]),
                "utilization_percent": int(parts[5]),
                "temperature_c": int(parts[6]),
            }
        )
    return {"ok": True, "gpus": gpus}


def torch_cuda_memory_snapshot(torch_module) -> list[dict[str, int]]:
    rows = []
    for index in range(torch_module.cuda.device_count()):
        rows.append(
            {
                "index": index,
                "allocated_bytes": int(torch_module.cuda.memory_allocated(index)),
                "reserved_bytes": int(torch_module.cuda.memory_reserved(index)),
                "max_allocated_bytes": int(torch_module.cuda.max_memory_allocated(index)),
                "max_reserved_bytes": int(torch_module.cuda.max_memory_reserved(index)),
            }
        )
    return rows



def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_evidence_zip(run_dir: Path, zip_path: Path) -> dict[str, str]:
    run_dir = Path(run_dir).resolve()
    zip_path = Path(zip_path).resolve()
    if not run_dir.is_dir():
        raise ValueError(f"Evidence run directory does not exist: {run_dir}")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(path for path in run_dir.rglob("*") if path.is_file()):
            archive.write(item, arcname=str(Path(run_dir.name) / item.relative_to(run_dir)))
    digest = sha256_file(zip_path)
    sha256_path = Path(str(zip_path) + ".sha256")
    sha256_path.write_text(f"{digest}  {zip_path.name}\n", encoding="utf-8")
    return {
        "zip_path": str(zip_path),
        "sha256": digest,
        "sha256_path": str(sha256_path),
    }
