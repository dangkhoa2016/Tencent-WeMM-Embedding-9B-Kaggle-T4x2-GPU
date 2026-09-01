from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from .qdrant import QdrantSearchResult


@dataclass(frozen=True)
class FrozenTextExample:
    qid: str
    name: str
    category: str
    text_en: str
    text_vi: str
    authority_ranks: tuple[int, int, int, int] = (1, 1, 1, 1)


@dataclass(frozen=True)
class FrozenImageExample:
    qid: str
    name: str
    category: str
    p18_filename: str
    normalized_png_sha256: str
    authority_ranks: tuple[int, int, int, int] = (1, 1, 1, 1)


FROZEN_TEXT_SHOWCASE = (
    FrozenTextExample(
        qid="Q105076326",
        name="Saigon",
        category="place / Vietnam / urban history",
        text_en=(
            "Saigon. historical city at the center of present-day Ho Chi Minh City "
            "until its merger with Cholon in 1956."
        ),
        text_vi=(
            "Sài Gòn. thành phố thời Pháp thuộc được thành lập năm 1877, mở rộng "
            "dưới thời Việt Nam Cộng hòa và đổi tên thành TP. Hồ Chí Minh 1976."
        ),
    ),
    FrozenTextExample(
        qid="Q80398",
        name="Pericles",
        category="person / ancient history",
        text_en="Pericles. Athenian statesman, orator and general (c. 495 – 429 BC).",
        text_vi=(
            "Perikles. Là một nhà chính trị, nhà hùng biện, tướng lĩnh tài ba và "
            "có nhiều ảnh hưởng của Athena trong Thời đại Hoàng kim của thị quốc này."
        ),
    ),
    FrozenTextExample(
        qid="Q336",
        name="science",
        category="abstract scientific concept",
        text_en=(
            "science. systematic endeavor that builds and organizes knowledge, and "
            "the set of knowledge produced by this system."
        ),
        text_vi=(
            "khoa học. hệ thống kiến thức về những định luật, cấu trúc và cách vận "
            "hành của thế giới tự nhiên, được đúc kết thông qua việc quan sát, mô tả, "
            "đo đạc, thực nghiệm, phát triển lý thuyết."
        ),
    ),
    FrozenTextExample(
        qid="Q826858",
        name="International Federation of the Phonographic Industry",
        category="organization / music industry",
        text_en=(
            "International Federation of the Phonographic Industry. organization "
            "that represents the interests of the recording industry."
        ),
        text_vi=(
            "Liên đoàn Công nghiệp ghi âm quốc tế. tổ chức đại diện cho lợi ích của "
            "ngành công nghiệp ghi âm trên toàn thế giới."
        ),
    ),
    FrozenTextExample(
        qid="Q482539",
        name="Fila, Inc.",
        category="company / consumer brand",
        text_en="Fila, Inc.. South Korean clothing and consumer goods manufacturer.",
        text_vi=(
            "Fila. Công ty đồ thể thao của Ý, có trụ sở chính được đặt tại Hàn Quốc."
        ),
    ),
)


FROZEN_IMAGE_SHOWCASE = (
    FrozenImageExample(
        qid="Q43304",
        name="Marcello Lippi",
        category="person",
        p18_filename=(
            "Marcello Lippi by Martina De Siervo - International Journalism Festival 2010.jpg"
        ),
        normalized_png_sha256=(
            "e510c9117a5896409f4be454abb9e8c32002d49752f47b6d57702a90151876ad"
        ),
    ),
    FrozenImageExample(
        qid="Q546",
        name="Trieste",
        category="place",
        p18_filename="Trieste (28766391880).jpg",
        normalized_png_sha256=(
            "e250eba1477fe0f056590c079c72ad21077aa16860f95830e9611c1db1f49705"
        ),
    ),
    FrozenImageExample(
        qid="Q1416632",
        name="Finance University under the Government of the Russian Federation",
        category="institution/building",
        p18_filename="Finuniver mainbuilding.jpg",
        normalized_png_sha256=(
            "15be44219ae32460102158faaf3dc0222fc3e9118f68982a66e7067fbc5f71aa"
        ),
    ),
    FrozenImageExample(
        qid="Q1130757",
        name="Northern Rock",
        category="organization / real-world scene",
        p18_filename="Northern Rock Queue.jpg",
        normalized_png_sha256=(
            "eefcc9673e9a0d52368bb5efe0e21eaa5723a1c9c5bd5f85be2a71209bc6d929"
        ),
    ),
)


def _rank(rows: Sequence[QdrantSearchResult], expected_qid: str) -> int | None:
    for row in rows:
        if row.qid == expected_qid:
            return row.rank
    return None


def evaluate_four_paths(
    *,
    expected_qid: str,
    vectors: Mapping[int, Sequence[float]],
    search: Callable[[int, Sequence[float], str], Sequence[QdrantSearchResult]],
    paths: Sequence[tuple[int, str]] = (
        (4096, "vi"),
        (1024, "vi"),
        (4096, "en"),
        (1024, "en"),
    ),
) -> dict:
    results = []
    for dimension, vector_name in paths:
        rows = list(search(int(dimension), vectors[int(dimension)], vector_name))
        rank = _rank(rows, expected_qid)
        top3 = []
        for row in rows[:3]:
            payload = dict(row.payload or {})

            def first_value(keys):
                return next(
                    (str(payload[key]).strip() for key in keys if payload.get(key)),
                    "",
                )

            label_en = first_value(
                ("label_en", "name_en", "title_en", "text_en")
            )
            label_vi = first_value(
                ("label_vi", "name_vi", "title_vi", "text_vi")
            )
            generic = first_value(("label", "name", "title"))

            if not label_en and generic:
                label_en = generic
            if not label_vi and generic:
                label_vi = generic

            top3.append(
                {
                    "rank": int(row.rank),
                    "qid": str(row.qid),
                    "score": float(row.score),
                    "label_en": label_en[:220],
                    "label_vi": label_vi[:220],
                }
            )
        results.append(
            {
                "dimension": int(dimension),
                "vector_name": str(vector_name),
                "rank": rank,
                "top_score": float(rows[0].score),
                "expected_score": next(
                    (float(row.score) for row in rows if row.qid == expected_qid),
                    None,
                ),
                "top3": top3,
            }
        )
    ranks = [item["rank"] for item in results]
    return {
        "qid": expected_qid,
        "paths": results,
        "all_top1": all(rank == 1 for rank in ranks),
        "strict_all_top3": all(
            rank is not None and int(rank) <= 3 for rank in ranks
        ),
    }
