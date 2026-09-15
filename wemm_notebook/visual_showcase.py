"""Step 7B visual-robustness retrieval over a temporary four-image gallery."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import time
from pathlib import Path
from urllib.parse import quote

from IPython.display import display


VISUAL_THRESHOLD = 0.90
VISUAL_MAX_EDGE = 768
VISUAL_MAX_BYTES = 20 * 1024 * 1024
VISUAL_SPECS = [
    {"qid": "Q19217", "label_vi": "Lâm Trịnh Nguyệt Nga", "label_en": "Carrie Lam", "p18": "Carrie Lam 2019-04-09 (1).jpg", "curation_min_score": 0.980786},
    {"qid": "Q10489198", "label_vi": "Trường Đại học Sư phạm Thành phố Hồ Chí Minh", "label_en": "Ho Chi Minh University of Education", "p18": "Ho Chi Minh City Pedagogical University August 30, 2018.jpg", "curation_min_score": 0.969598},
    {"qid": "Q168751", "label_vi": "Đại học Duke", "label_en": "Duke University", "p18": "Duke Chapel 4 16 05.jpg", "curation_min_score": 0.963304},
    {"qid": "Q51756", "label_vi": "China Airlines", "label_en": "China Airlines", "p18": "B-18902 A350-900 China Airlines LHR 4.11.20.jpg", "curation_min_score": 0.950137},
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_normalized(session, spec: dict, image_dir: Path):
    from PIL import Image, ImageOps
    url = (
        "https://commons.wikimedia.org/wiki/Special:Redirect/file/"
        + quote(spec["p18"], safe="")
        + f"?width={VISUAL_MAX_EDGE}"
    )
    response = session.get(url, timeout=90)
    response.raise_for_status()
    raw = response.content
    if len(raw) > VISUAL_MAX_BYTES:
        raise RuntimeError(f"{spec['qid']}: image exceeds byte bound")

    with Image.open(io.BytesIO(raw)) as source:
        source.load()
        source = ImageOps.exif_transpose(source)
        rgb0 = source.convert("RGB")
        fresh = Image.new("RGB", rgb0.size)
        fresh.paste(rgb0)

    source_size = list(map(int, fresh.size))
    fresh.thumbnail((VISUAL_MAX_EDGE, VISUAL_MAX_EDGE), Image.Resampling.LANCZOS)
    path = image_dir / f"{spec['qid']}.png"
    fresh.save(path, format="PNG", compress_level=0)
    return path, {
        "download_url": url,
        "resolved_url": response.url,
        "downloaded_bytes": len(raw),
        "source_size": source_size,
        "normalized_size": list(map(int, fresh.size)),
        "normalized_png_sha256": _sha256(path),
    }


def _make_transforms(qid: str, source_path: Path, transform_root: Path):
    from PIL import Image, ImageEnhance
    out_dir = transform_root / qid
    out_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as source:
        base = source.convert("RGB")

    width, height = base.size
    paths = {}

    p = out_dir / "resize_80pct.png"
    base.resize(
        (max(64, int(width * 0.8)), max(64, int(height * 0.8))),
        Image.Resampling.LANCZOS,
    ).save(p, "PNG", compress_level=0)
    paths["resize_80pct"] = p

    p = out_dir / "jpeg_q90.jpg"
    base.save(p, "JPEG", quality=90, optimize=False, subsampling=0)
    paths["jpeg_q90"] = p

    dx = max(1, int(width * 0.02))
    dy = max(1, int(height * 0.02))
    cropped = base.crop((dx, dy, width - dx, height - dy)).resize(
        (width, height), Image.Resampling.LANCZOS
    )
    p = out_dir / "center_crop_96pct.png"
    cropped.save(p, "PNG", compress_level=0)
    paths["center_crop_96pct"] = p

    p = out_dir / "brightness_103pct.png"
    ImageEnhance.Brightness(base).enhance(1.03).save(p, "PNG", compress_level=0)
    paths["brightness_103pct"] = p
    return paths


def _display_preview(path: Path):
    from PIL import Image

    with Image.open(path) as shown:
        thumb = shown.copy()
        thumb.thumbnail((520, 360), Image.Resampling.LANCZOS)
        display(thumb)


def run_visual_showcase(demo) -> dict:
    """Execute the accepted 4 entities × 4 transforms × 2 dimensions matrix."""

    import requests
    from qdrant_client import QdrantClient, models
    from wemm_kaggle.demo_config import EVID, RUN_ROOT
    from wemm_kaggle.demo_io import truncate_vector
    from wemm_kaggle.demo_qdrant import write_cache_seal

    if demo.closed:
        raise RuntimeError("Demo session is already closed")
    if demo.image_results is None:
        raise RuntimeError("Step 7A must complete before Step 7B")

    phase_t0 = time.perf_counter()
    root = RUN_ROOT / "visual-robustness"
    image_dir = root / "images"
    transform_root = root / "transforms"
    qdrant_dir = root / "qdrant-local"

    if root.exists():
        shutil.rmtree(root)
    image_dir.mkdir(parents=True)
    transform_root.mkdir(parents=True)

    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Tencent-WeMM-Embedding-9B/1.0 visual-robustness public demo "
            "(https://github.com/dangkhoa2016/Tencent-WeMM-Embedding-9B-Kaggle-T4x2-GPU)"
        )
    })

    local_client = None
    visual_results = None
    try:
        assets = []
        for spec in VISUAL_SPECS:
            path, asset = _download_normalized(session, spec, image_dir)
            assets.append({**spec, "image_path": str(path), "asset": asset})

        local_client = QdrantClient(path=str(qdrant_dir))
        local_client.create_collection(
            collection_name="visual_originals",
            vectors_config={
                "image_4096": models.VectorParams(size=4096, distance=models.Distance.COSINE),
                "image_1024": models.VectorParams(size=1024, distance=models.Distance.COSINE),
            },
        )

        for rec in assets:
            vector_4096 = demo.worker.embed_image(rec["image_path"], 4096)
            vector_1024 = truncate_vector(vector_4096, 1024)
            local_client.upsert(
                collection_name="visual_originals",
                points=[models.PointStruct(
                    id=int(rec["qid"][1:]),
                    vector={"image_4096": vector_4096, "image_1024": vector_1024},
                    payload={
                        "qid": rec["qid"],
                        "label_vi": rec["label_vi"],
                        "label_en": rec["label_en"],
                        "p18": rec["p18"],
                        "original_sha256": rec["asset"]["normalized_png_sha256"],
                    },
                )],
                wait=True,
            )

        if int(local_client.count("visual_originals", exact=True).count) != 4:
            raise RuntimeError("Visual gallery point-count contract failed")

        rows, total_paths, passed_paths = [], 0, 0
        for index, rec in enumerate(assets, 1):
            qid = rec["qid"]
            transform_paths = _make_transforms(qid, Path(rec["image_path"]), transform_root)
            path_results = []

            print("\n" + "-" * 92)
            print(
                f"[HÌNH ẢNH ĐỘ TIN CẬY CAO / HIGH-CONFIDENCE VISUAL {index}/4] "
                f"{rec['label_vi']} / {rec['label_en']} — {qid}",
                flush=True,
            )
            print(f"  P18: {rec['p18']}", flush=True)
            print("  SHA-256 ảnh gốc / Original SHA-256: " + rec["asset"]["normalized_png_sha256"], flush=True)
            print("  Kích thước chuẩn hóa / Normalized size: " + str(rec["asset"]["normalized_size"]), flush=True)
            print(f"  Min cosine ở curation run / Curation reference min cosine: {rec['curation_min_score']:.6f}", flush=True)
            print("  Ảnh gốc / Original preview:", flush=True)
            _display_preview(Path(rec["image_path"]))
            representative = transform_paths["center_crop_96pct"]
            print("  Ảnh biến đổi / Transformed preview — center_crop_96pct:", flush=True)
            _display_preview(representative)

            for transform_name, transform_path in transform_paths.items():
                query_4096 = demo.worker.embed_image(transform_path, 4096)
                query_1024 = truncate_vector(query_4096, 1024)
                transform_sha = _sha256(transform_path)
                for dimension, query_vector in ((4096, query_4096), (1024, query_1024)):
                    response = local_client.query_points(
                        collection_name="visual_originals",
                        query=query_vector,
                        using=f"image_{dimension}",
                        limit=4,
                        with_payload=True,
                    )
                    points = list(response.points)
                    expected_rank = next(
                        (rank for rank, point in enumerate(points, 1)
                         if str((point.payload or {}).get("qid", "")) == qid),
                        None,
                    )
                    expected_score = next(
                        (float(point.score) for point in points
                         if str((point.payload or {}).get("qid", "")) == qid),
                        None,
                    )
                    path_pass = (
                        expected_rank == 1
                        and expected_score is not None
                        and expected_score >= VISUAL_THRESHOLD
                    )
                    total_paths += 1
                    passed_paths += int(path_pass)

                    top3 = []
                    for rank, point in enumerate(points[:3], 1):
                        payload = dict(point.payload or {})
                        top3.append({
                            "rank": rank,
                            "qid": str(payload.get("qid", "")),
                            "score": float(point.score),
                            "label_vi": str(payload.get("label_vi", "")),
                            "label_en": str(payload.get("label_en", "")),
                        })

                    path_results.append({
                        "transform": transform_name,
                        "dimension": dimension,
                        "transform_sha256": transform_sha,
                        "expected_rank": expected_rank,
                        "raw_cosine": expected_score,
                        "pass": path_pass,
                        "top3": top3,
                    })

                    print(
                        f"  {transform_name:<20} | {dimension:4d}d | "
                        f"rank=#{expected_rank} | raw cosine={expected_score:.6f} | "
                        f"{'PASS' if path_pass else 'FAIL'}",
                        flush=True,
                    )
                    for hit in top3:
                        print(
                            f"      #{hit['rank']} {hit['qid']:<11} "
                            f"score={hit['score']:.6f} | "
                            f"VI={hit['label_vi']} | EN={hit['label_en']}",
                            flush=True,
                        )

            scores = [p["raw_cosine"] for p in path_results]
            strict_pass = len(path_results) == 8 and all(p["pass"] for p in path_results)
            row = {
                "qid": qid,
                "label_vi": rec["label_vi"],
                "label_en": rec["label_en"],
                "p18": rec["p18"],
                "asset": rec["asset"],
                "curation_reference_min_cosine": rec["curation_min_score"],
                "runtime_paths": path_results,
                "runtime_min_raw_cosine": min(scores),
                "strict_pass": strict_pass,
            }
            rows.append(row)
            print(f"  Min raw cosine runtime / Runtime minimum raw cosine: {row['runtime_min_raw_cosine']:.6f}", flush=True)
            print("  Kết quả / Verdict: " + ("8/8 PATHS TOP-1 >= 0.90 — PASS" if strict_pass else "FAIL"), flush=True)

        runtime_min = min(row["runtime_min_raw_cosine"] for row in rows)
        all_strict = (
            total_paths == 32
            and passed_paths == 32
            and all(row["strict_pass"] for row in rows)
        )
        visual_results = {
            "verdict": "PASS" if all_strict else "FAIL",
            "threshold": VISUAL_THRESHOLD,
            "raw_cosine_rescaled": False,
            "threshold_relaxed": False,
            "production_collections_mutated": False,
            "example_count": len(rows),
            "total_paths": total_paths,
            "passed_paths": passed_paths,
            "runtime_min_raw_cosine": runtime_min,
            "examples": rows,
        }
        (EVID / "visual-robustness.json").write_text(
            json.dumps(visual_results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("\n" + "=" * 92)
        print("TỔNG KẾT TRUY XUẤT HÌNH ẢNH ĐỘ TIN CẬY CAO / HIGH-CONFIDENCE VISUAL SCORECARD")
        print("=" * 92)
        print(f"Ví dụ / Examples                         : {len(rows)}/4")
        print(f"Đường truy xuất TOP-1 / TOP-1 paths      : {passed_paths}/{total_paths}")
        print(f"Raw cosine thấp nhất / Minimum raw cosine : {runtime_min:.6f}")
        print(f"Ngưỡng strict / Strict threshold          : {VISUAL_THRESHOLD:.2f}")
        print("Rescale raw cosine                         : NO / KHÔNG")
        print("Nới ngưỡng / Threshold relaxation          : NO / KHÔNG")
        print("Thay đổi production collections            : NO / KHÔNG")

        if not all_strict:
            raise RuntimeError(
                f"Visual robustness failed: passed={passed_paths}/{total_paths}, "
                f"min_raw_cosine={runtime_min:.6f}"
            )

        print("HIGH_CONFIDENCE_VISUAL_RETRIEVAL=PASS")
        print("VISUAL_RETRIEVAL_EXAMPLES=4")
        print("VISUAL_RETRIEVAL_PATHS_TOP1=32/32")
        print(f"VISUAL_RETRIEVAL_MIN_RAW_COSINE={runtime_min:.6f}")
        print("VISUAL_RETRIEVAL_THRESHOLD=0.90")
        print("VISUAL_RETRIEVAL_RAW_COSINE_RESCALED=NO")
        print("VISUAL_RETRIEVAL_THRESHOLD_RELAXED=NO")
        print("VISUAL_RETRIEVAL_PRODUCTION_COLLECTIONS_MUTATED=NO")
    except BaseException:
        try:
            if local_client is not None:
                local_client.close()
        finally:
            if qdrant_dir.exists():
                shutil.rmtree(qdrant_dir, ignore_errors=True)
        if not demo.closed:
            demo.abort()
            try:
                write_cache_seal(demo.data_mode)
            except Exception:
                pass
        raise
    finally:
        if local_client is not None:
            try:
                local_client.close()
            except Exception:
                pass
        if qdrant_dir.exists():
            shutil.rmtree(qdrant_dir, ignore_errors=True)

    if visual_results is None:
        raise RuntimeError("Visual robustness result was not produced")

    demo.phase_times["7b_visual_robustness"] = time.perf_counter() - phase_t0
    print(f"Thời gian Step 7B / Step 7B time              : {demo.phase_times['7b_visual_robustness']:.3f} s")
    print("VISUAL_RETRIEVAL_TEMP_QDRANT=DELETED")
    return visual_results
