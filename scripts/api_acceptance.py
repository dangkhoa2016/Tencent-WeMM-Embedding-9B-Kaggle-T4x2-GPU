#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx
from PIL import Image, ImageDraw

ACCEPTANCE_RESULT_OK = "WEMM_T4X2_REST_API_ACCEPTANCE_PASS"
ACCEPTANCE_RESULT_FAIL = "WEMM_T4X2_REST_API_ACCEPTANCE_FAIL"

TEXT_FIXTURES = (
    "Hanoi is the capital of Vietnam.",
    "Hà Nội là thủ đô nước Cộng hòa xã hội chủ nghĩa Việt Nam.",
    "A diagram shows a red rectangle, a blue circle, and a green triangle on a white canvas.",
    "Một hình chữ nhật màu đỏ, một hình tròn màu xanh và một hình tam giác màu xanh lá cây.",
    "The quick brown fox jumps over the lazy dog near the busy highway.",
    "Whiskers the cat slept fourteen hours, dreaming of an endless corridor of tuna tins.",
)

DOCUMENTED_SERVICE_ERRORS = {200, 503, 504}


def text_fixture(index: int) -> str:
    return TEXT_FIXTURES[index % len(TEXT_FIXTURES)]


def png_bytes(index: int = 0, size: int = 256) -> bytes:
    color = ((index * 67) % 256, (index * 131) % 256, (index * 197) % 256)
    image = Image.new("RGB", (size, size), color)
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, size - 8, size // 2), fill=(255, 0, 0))
    draw.ellipse((size // 4, size // 4, size // 2, size // 2), fill=(0, 0, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def gif_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), "blue").save(buffer, format="GIF")
    return buffer.getvalue()


def matrix_config():
    return [
        (workload, dimension, batch)
        for workload in ("text", "image", "image_text")
        for dimension in (4096, 1024)
        for batch in (1, 2, 4)
    ]


def validate_embedding_values(embedding, dimension):
    length_ok = len(embedding) == dimension
    all_finite = False
    norm = 0.0
    if length_ok:
        all_finite = all(math.isfinite(float(value)) for value in embedding)
        if all_finite:
            norm = math.sqrt(sum(float(value) * float(value) for value in embedding))
    ok = length_ok and all_finite and 0.98 <= norm <= 1.02
    return {
        "length": len(embedding),
        "dimension": dimension,
        "all_finite": all_finite,
        "norm": norm,
        "ok": ok,
    }


def client_evidence_template():
    return {
        "schema_version": 1,
        "status": "RUNNING",
        "result": None,
        "base_url": None,
        "health": {},
        "ready_transition": {},
        "auth_negative": {},
        "matrix": [],
        "negative_validation": {},
        "concurrency": {},
        "backpressure": {},
        "verdict": {},
    }


def safe_error(body):
    if isinstance(body, dict):
        return {"error": body.get("error"), "message": str(body.get("message", ""))[:200]}
    return {"error": None, "message": str(body)[:200]}


def send_batch_request(client, auth_headers, workload, dimension, batch):
    started = time.perf_counter()
    if workload == "text":
        response = client.post(
            "/v1/embeddings/text",
            headers=auth_headers,
            json={"inputs": [text_fixture(i) for i in range(batch)], "dimension": dimension},
        )
    elif workload == "image":
        files = [("images", (f"m{i}.png", png_bytes(i), "image/png")) for i in range(batch)]
        response = client.post(
            "/v1/embeddings/image",
            headers=auth_headers,
            files=files,
            data={"dimension": str(dimension)},
        )
    elif workload == "image_text":
        files = [("images", (f"m{i}.png", png_bytes(i), "image/png")) for i in range(batch)]
        response = client.post(
            "/v1/embeddings/image-text",
            headers=auth_headers,
            files=files,
            data={
                "texts": [text_fixture(i) for i in range(batch)],
                "dimension": str(dimension),
            },
        )
    else:
        raise ValueError(f"unknown workload {workload}")
    elapsed = time.perf_counter() - started
    try:
        body = response.json()
    except Exception:
        body = {}
    return response.status_code, body, elapsed


def validate_response(body, workload, dimension, batch):
    count_ok = body.get("count") == batch
    request_id = body.get("request_id") or ""
    timing_ms = body.get("timing_ms") or {}
    timing_present = (
        {"queue", "inference", "total"} == set(timing_ms)
        and all(value is not None for value in timing_ms.values())
    )
    reports = [validate_embedding_values(item.get("embedding") or [], dimension) for item in body.get("data") or []]
    per_item_ok = all(report["ok"] for report in reports)
    ok = count_ok and bool(request_id) and timing_present and per_item_ok
    summary = {
        "count": body.get("count"),
        "count_ok": count_ok,
        "lengths_ok": all(report["length"] == dimension for report in reports),
        "all_finite": all(report["all_finite"] for report in reports),
        "norm_min": round(min((report["norm"] for report in reports), default=0.0), 4),
        "norm_max": round(max((report["norm"] for report in reports), default=0.0), 4),
        "request_id_present": bool(request_id),
        "timing_present": timing_present,
        "per_item_ok": per_item_ok,
        "request_id": request_id,
        "timing_ms": timing_ms,
        "model": body.get("model"),
        "workload": body.get("workload"),
    }
    return {"ok": ok, "summary": summary}


def run_matrix(client, auth_headers):
    records = []
    for workload, dimension, batch in matrix_config():
        status, body, elapsed = send_batch_request(client, auth_headers, workload, dimension, batch)
        record = {
            "workload": workload,
            "dimension": dimension,
            "batch": batch,
            "http_status": status,
            "elapsed_s": round(elapsed, 3),
        }
        if status != 200:
            record["status"] = "FAIL"
            record.update(safe_error(body))
            records.append(record)
            continue
        validation = validate_response(body, workload, dimension, batch)
        record["status"] = "PASS" if validation["ok"] else "FAIL"
        record.update(validation["summary"])
        records.append(record)
    return records


def run_auth_negative(client, token):
    results = {}
    response = client.post("/v1/embeddings/text", json={"inputs": ["no token"]})
    results["missing_auth"] = {
        "status": response.status_code,
        "ok": response.status_code == 401,
        "www_authenticate": response.headers.get("WWW-Authenticate"),
    }
    response = client.post(
        "/v1/embeddings/text",
        headers={"Authorization": f"Bearer {'x' * 48}"},
        json={"inputs": ["wrong token"]},
    )
    results["wrong_auth"] = {"status": response.status_code, "ok": response.status_code == 401}
    return results


def run_negative_validation(client, auth_headers):
    results = {}
    response = client.post(
        "/v1/embeddings/text",
        headers=auth_headers,
        json={"inputs": [text_fixture(i) for i in range(5)]},
    )
    results["b5_text"] = {"status": response.status_code, "ok": response.status_code == 422}
    response = client.post(
        "/v1/embeddings/text",
        headers=auth_headers,
        json={"inputs": ["hello"], "dimension": 512},
    )
    results["invalid_dimension"] = {"status": response.status_code, "ok": response.status_code == 422}
    response = client.post(
        "/v1/embeddings/image",
        headers=auth_headers,
        files=[("images", ("bad.png", b"\x00\x01\x02notanimage", "image/png"))],
        data={"dimension": "4096"},
    )
    results["malformed_image"] = {"status": response.status_code, "ok": response.status_code == 422}
    response = client.post(
        "/v1/embeddings/image",
        headers=auth_headers,
        files=[("images", ("image.gif", gif_bytes(), "image/gif"))],
        data={"dimension": "4096"},
    )
    results["disallowed_format"] = {"status": response.status_code, "ok": response.status_code == 415}
    response = client.post(
        "/v1/embeddings/image",
        headers=auth_headers,
        files=[("images", ("big.png", b"\xff" * (8 * 1024 * 1024 + 1), "image/png"))],
        data={"dimension": "4096"},
    )
    results["oversize_image"] = {"status": response.status_code, "ok": response.status_code == 413}
    return results


def run_concurrent(client, auth_headers, count=4, batch=1, workload="text", dimension=4096):
    with ThreadPoolExecutor(max_workers=count) as pool:
        futures = [
            pool.submit(send_batch_request, client, auth_headers, workload, dimension, batch)
            for _ in range(count)
        ]
        results = [future.result() for future in futures]
    statuses = sorted(item[0] for item in results)
    return {
        "requests": count,
        "statuses": statuses,
        "all_documented": all(status in DOCUMENTED_SERVICE_ERRORS for status in statuses),
        "accepted_200": statuses.count(200),
    }


def run_backpressure(client, auth_headers, count=12, batch=4):
    with ThreadPoolExecutor(max_workers=count) as pool:
        futures = [
            pool.submit(send_batch_request, client, auth_headers, "text", 4096, batch)
            for _ in range(count)
        ]
        results = [future.result() for future in futures]
    statuses = sorted(item[0] for item in results)
    queue_full = sum(
        1
        for (status, body, _) in results
        if status == 503 and isinstance(body, dict) and body.get("error") == "QUEUE_FULL"
    )
    accepted = statuses.count(200)
    return {
        "requests": count,
        "batch": batch,
        "statuses": statuses,
        "accepted_200": accepted,
        "queue_full_503": queue_full,
    }


def wait_until(predicate, timeout_s, poll_s=0.25):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll_s)
    return predicate()


def compute_verdict(evidence):
    checks = [
        ("health", evidence["health"].get("observed") is True),
        (
            "ready_transition",
            evidence["ready_transition"].get("not_ready_observed") is True
            and evidence["ready_transition"].get("ready_observed") is True,
        ),
        (
            "auth_negative",
            evidence["auth_negative"].get("missing_auth", {}).get("ok")
            and evidence["auth_negative"].get("wrong_auth", {}).get("ok"),
        ),
        (
            "matrix_18",
            len(evidence["matrix"]) == 18 and all(row.get("status") == "PASS" for row in evidence["matrix"]),
        ),
        (
            "negative_validation",
            all(row.get("ok") for row in evidence["negative_validation"].values()),
        ),
        (
            "concurrency",
            evidence["concurrency"].get("all_documented") is True
            and evidence["concurrency"].get("accepted_200", 0) >= 1,
        ),
        (
            "backpressure",
            evidence["backpressure"].get("queue_full_503", 0) >= 1
            and evidence["backpressure"].get("accepted_200", 0) >= 1
            and evidence["backpressure"].get("recovered") is True,
        ),
    ]
    passed = all(passed for _, passed in checks)
    return passed, checks


def run_acceptance(base_url, token, ready_timeout=180.0):
    evidence = client_evidence_template()
    evidence["base_url"] = base_url
    auth_headers = {"Authorization": f"Bearer {token}"}
    try:
        client = httpx.Client(base_url=base_url, timeout=60.0, follow_redirects=False)

        def healthy():
            try:
                response = client.get("/healthz")
                return response.status_code == 200
            except httpx.HTTPError:
                return False

        health_ok = wait_until(healthy, ready_timeout, poll_s=0.1)
        health_body = {}
        if health_ok:
            health_body = client.get("/healthz").json()
        evidence["health"] = {
            "observed": health_ok,
            "timeout_s": ready_timeout,
            "state": health_body.get("state") if health_ok else None,
        }
        if not health_ok:
            raise RuntimeError("server did not become healthy in time")

        started = time.monotonic()
        early_response = client.post(
            "/v1/embeddings/text",
            headers=auth_headers,
            json={"inputs": ["request before READY"]},
        )
        evidence["negative_validation"]["before_ready"] = {
            "status": early_response.status_code,
            "ok": early_response.status_code == 503,
        }

        not_ready_observed = False
        seen_states = []
        while time.monotonic() - started < ready_timeout:
            response = client.get("/readyz")
            seen_states.append(response.status_code)
            if response.status_code == 503:
                not_ready_observed = True
            elif response.status_code == 200:
                break
            time.sleep(0.25)
        ready_observed = response.status_code == 200 if seen_states else False
        evidence["ready_transition"] = {
            "not_ready_observed": not_ready_observed,
            "ready_observed": ready_observed,
            "ready_wait_s": round(time.monotonic() - started, 2),
            "seen_states": seen_states[:8],
        }
        if not (not_ready_observed and ready_observed):
            raise RuntimeError("did not observe /readyz transition 503 -> 200")

        evidence["auth_negative"] = run_auth_negative(client, token)
        evidence["matrix"] = run_matrix(client, auth_headers)
        evidence["negative_validation"].update(run_negative_validation(client, auth_headers))
        evidence["concurrency"] = run_concurrent(client, auth_headers, count=4)

        backpressure = run_backpressure(client, auth_headers, count=12)
        recovered = wait_until(
            lambda: client.get("/readyz").status_code == 200,
            ready_timeout,
        )
        backpressure["recovered"] = recovered
        evidence["backpressure"] = backpressure

        passed, checks = compute_verdict(evidence)
        evidence["verdict"] = {
            "passed": passed,
            "checks": {name: ok for name, ok in checks},
        }
        evidence["status"] = "PASS" if passed else "FAIL"
        evidence["result"] = ACCEPTANCE_RESULT_OK if passed else ACCEPTANCE_RESULT_FAIL
    except Exception as exc:
        evidence["status"] = "FAIL"
        evidence["result"] = ACCEPTANCE_RESULT_FAIL
        evidence["verdict"] = {"passed": False, "error": type(exc).__name__, "message": str(exc)[:500]}
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description="Tencent WeMM-Embedding-9B REST API client acceptance suite")
    parser.add_argument("--base-url", default="http://127.0.0.1:8090")
    parser.add_argument("--token-env", default="WEMM_API_TOKEN")
    parser.add_argument("--output", default="craft/client-acceptance.json")
    parser.add_argument("--ready-timeout", type=float, default=180.0)
    args = parser.parse_args(argv)

    token = os.environ.get(args.token_env, "")
    if not token:
        print(f"require {args.token_env} with a bearer token", file=sys.stderr)
        return 1

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    evidence = run_acceptance(args.base_url, token, args.ready_timeout)
    tmp = output.with_suffix(f"{output.suffix}.tmp")
    tmp.write_text(json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(output)
    print(json.dumps(evidence["verdict"], indent=2))
    print(evidence["result"])
    return 0 if evidence["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())