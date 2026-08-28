#!/usr/bin/env python3
"""Fail-closed setup guards for Kaggle system Torch reuse."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import sys
import sysconfig
from pathlib import Path
from typing import Any

FORBIDDEN_LOCAL_EXACT = {"torch", "triton"}
FORBIDDEN_LOCAL_PREFIXES = ("nvidia-", "cuda-")


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def capture_torch_identity() -> dict[str, Any]:
    import torch

    return {
        "torch_version": str(torch.__version__),
        "torch_file": str(Path(torch.__file__).resolve()),
        "torch_cuda": None if torch.version.cuda is None else str(torch.version.cuda),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
        "python_executable": str(Path(sys.executable).resolve()),
        "sys_prefix": str(Path(sys.prefix).resolve()),
        "sys_base_prefix": str(Path(sys.base_prefix).resolve()),
    }


def verify_identity_reuse(expected: dict[str, Any], actual: dict[str, Any], venv_dir: Path) -> dict[str, Any]:
    keys = ("torch_version", "torch_file", "torch_cuda", "cuda_available", "device_count")
    mismatches = {k: {"expected": expected.get(k), "actual": actual.get(k)} for k in keys if expected.get(k) != actual.get(k)}
    torch_file = Path(str(actual.get("torch_file", ""))).resolve()
    venv_dir = venv_dir.resolve()
    try:
        torch_file.relative_to(venv_dir)
        torch_inside_venv = True
    except ValueError:
        torch_inside_venv = False
    return {
        "system_torch_reused": not mismatches and not torch_inside_venv,
        "torch_reinstall_detected": bool(torch_inside_venv or mismatches),
        "torch_inside_venv": torch_inside_venv,
        "mismatches": mismatches,
        "expected": expected,
        "actual": actual,
    }


def local_distributions() -> list[dict[str, str]]:
    purelib = Path(sysconfig.get_path("purelib")).resolve()
    found: list[dict[str, str]] = []
    for dist in importlib.metadata.distributions(path=[str(purelib)]):
        raw = dist.metadata.get("Name") or ""
        if not raw:
            continue
        found.append({"name": _canonical(raw), "version": dist.version, "purelib": str(purelib)})
    return sorted(found, key=lambda item: item["name"])


def scan_forbidden_local() -> dict[str, Any]:
    dists = local_distributions()
    forbidden = []
    for item in dists:
        name = item["name"]
        if name in FORBIDDEN_LOCAL_EXACT or name.startswith(FORBIDDEN_LOCAL_PREFIXES):
            forbidden.append(item)
    return {
        "torch_reinstall_detected": bool(forbidden),
        "forbidden_local_distributions": forbidden,
        "local_distribution_count": len(dists),
    }



def check_distribution_requirements(distribution: str) -> dict[str, Any]:
    from packaging.requirements import Requirement

    failures = []
    checked = []
    for raw in importlib.metadata.requires(distribution) or []:
        req = Requirement(raw)
        if req.marker is not None and not req.marker.evaluate():
            continue
        name = _canonical(req.name)
        try:
            version = importlib.metadata.version(req.name)
        except importlib.metadata.PackageNotFoundError:
            failures.append({"name": name, "requirement": raw, "error": "missing"})
            continue
        ok = not req.specifier or req.specifier.contains(version, prereleases=True)
        checked.append({"name": name, "requirement": raw, "version": version, "satisfied": ok})
        if not ok:
            failures.append({"name": name, "requirement": raw, "version": version, "error": "version_mismatch"})
    return {
        "distribution": distribution,
        "requirements_satisfied": not failures,
        "failures": failures,
        "checked": checked,
    }


def _write(payload: dict[str, Any], output: str | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


def cmd_capture(args: argparse.Namespace) -> int:
    payload = capture_torch_identity()
    errors = []
    if args.require_cuda and not payload["cuda_available"]:
        errors.append("CUDA is not available")
    if payload["device_count"] < args.min_gpus:
        errors.append(f"expected at least {args.min_gpus} CUDA devices, got {payload['device_count']}")
    payload["status"] = "PASS" if not errors else "FAIL"
    payload["errors"] = errors
    _write(payload, args.output)
    return 0 if not errors else 2


def cmd_verify(args: argparse.Namespace) -> int:
    expected = json.loads(Path(args.expected).read_text(encoding="utf-8"))
    actual = json.loads(Path(args.actual).read_text(encoding="utf-8"))
    payload = verify_identity_reuse(expected, actual, Path(args.venv_dir))
    payload["status"] = "PASS" if payload["system_torch_reused"] else "FAIL"
    _write(payload, args.output)
    return 0 if payload["system_torch_reused"] else 3


def cmd_scan(args: argparse.Namespace) -> int:
    payload = scan_forbidden_local()
    payload["status"] = "FAIL" if payload["torch_reinstall_detected"] else "PASS"
    _write(payload, args.output)
    return 4 if payload["torch_reinstall_detected"] else 0


def cmd_check_requires(args: argparse.Namespace) -> int:
    payload = check_distribution_requirements(args.distribution)
    payload["status"] = "PASS" if payload["requirements_satisfied"] else "FAIL"
    _write(payload, args.output)
    return 0 if payload["requirements_satisfied"] else 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("capture")
    p.add_argument("--output")
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument("--min-gpus", type=int, default=0)
    p.set_defaults(func=cmd_capture)

    p = sub.add_parser("verify-reuse")
    p.add_argument("--expected", required=True)
    p.add_argument("--actual", required=True)
    p.add_argument("--venv-dir", required=True)
    p.add_argument("--output")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("scan-local")
    p.add_argument("--output")
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("check-requires")
    p.add_argument("--distribution", required=True)
    p.add_argument("--output")
    p.set_defaults(func=cmd_check_requires)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
