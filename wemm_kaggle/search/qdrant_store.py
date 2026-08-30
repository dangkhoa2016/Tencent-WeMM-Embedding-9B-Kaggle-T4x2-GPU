from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient, models

from .mrl import DERIVED_DIMENSION, FULL_DIMENSION

COLLECTION_4096 = "wikidata_en_vi_wemm9b_4096_v030_rc2_d819dc7_v1"
COLLECTION_1024 = "wikidata_en_vi_wemm9b_1024_v030_rc2_d819dc7_v1"
VECTOR_NAMES = ("en", "vi")

MAX_POINTS = 99967


class QdrantSchemaError(Exception):
    pass


def collection_name_for(dimension: int) -> str:
    if dimension == FULL_DIMENSION:
        return COLLECTION_4096
    if dimension == DERIVED_DIMENSION:
        return COLLECTION_1024
    raise QdrantSchemaError(f"unsupported dimension {dimension}")


class QdrantStore:
    def __init__(
        self,
        client: QdrantClient | None = None,
        host: str = "127.0.0.1",
        port: int = 6333,
        timeout: float = 60.0,
        collection_4096: str = COLLECTION_4096,
        collection_1024: str = COLLECTION_1024,
    ) -> None:
        if client is not None:
            self._client = client
        else:
            self._client = QdrantClient(host=host, port=port, timeout=timeout)
        self._collection_4096 = collection_4096
        self._collection_1024 = collection_1024

    def collection_for(self, dimension: int) -> str:
        if dimension == FULL_DIMENSION:
            return self._collection_4096
        if dimension == DERIVED_DIMENSION:
            return self._collection_1024
        raise QdrantSchemaError(f"unsupported dimension {dimension}")

    def create_collection(self, dimension: int) -> None:
        from qdrant_client.http import models as m

        collection = self.collection_for(dimension)
        if self._client.collection_exists(collection):
            return
        self._client.create_collection(
            collection_name=collection,
            vectors_config={
                "en": m.VectorParams(size=dimension, distance=m.Distance.COSINE),
                "vi": m.VectorParams(size=dimension, distance=m.Distance.COSINE),
            },
            shard_number=1,
            replication_factor=1,
            write_consistency_factor=1,
        )

    def _read_vectors(self, info: Any, collection: str) -> dict[str, Any]:
        vectors = getattr(info, "vectors", None)
        if not isinstance(vectors, dict):
            params = getattr(info, "config", None)
            if params is not None:
                params = getattr(params, "params", None)
            if params is not None:
                vectors = getattr(params, "vectors", None)
        if not isinstance(vectors, dict):
            raise QdrantSchemaError(f"collection {collection} missing named vectors en/vi")
        return vectors

    def _read_params(self, info: Any) -> Any:
        params = getattr(info, "config", None)
        if params is not None:
            return getattr(params, "params", None)
        return getattr(info, "params", None)

    def verify_schema(self, dimension: int, allowed_range: tuple[int, int]) -> dict[str, Any]:
        collection = self.collection_for(dimension)
        info = self._client.get_collection(collection_name=collection)
        vectors = self._read_vectors(info, collection)
        names = set(vectors.keys())

        expected = {"en": dimension, "vi": dimension}
        if names != set(VECTOR_NAMES):
            raise QdrantSchemaError(
                f"collection {collection} named vectors must be exactly {set(VECTOR_NAMES)}; got {names}"
            )

        params = self._read_params(info) or {}
        shard = getattr(params, "shard_number", None)
        repl = getattr(params, "replication_factor", None)
        wcf = getattr(params, "write_consistency_factor", None)

        distances: dict[str, str] = {}
        for name in VECTOR_NAMES:
            vc = vectors.get(name)
            if vc is None:
                raise QdrantSchemaError(
                    f"collection {collection} missing named vector {name!r}"
                )
            size = getattr(vc, "size", None)
            distance = getattr(vc, "distance", None)
            if size != expected.get(name):
                raise QdrantSchemaError(
                    f"collection {collection} vector {name} size {size} != {expected.get(name)}"
                )
            if "cosine" not in str(distance).lower():
                raise QdrantSchemaError(
                    f"collection {collection} vector {name} distance {distance} is not Cosine"
                )
            distances[name] = str(distance)

        point_count = getattr(info, "points_count", 0)
        lo, hi = allowed_range
        if not (lo <= point_count <= hi):
            raise QdrantSchemaError(
                f"collection {collection} point count {point_count} outside allowed {allowed_range}"
            )
        if point_count > MAX_POINTS:
            raise QdrantSchemaError(f"collection {collection} point count exceeds {MAX_POINTS}")

        return {
            "collection": collection,
            "dimension": dimension,
            "vector_names": sorted(names),
            "distances": distances,
            "shard_number": shard,
            "replication_factor": repl,
            "write_consistency_factor": wcf,
            "point_count": point_count,
            "allowed_point_range": [lo, hi],
        }

    def query(
        self,
        *,
        dimension: int,
        query: list[float],
        using: str,
        limit: int,
        with_payload: bool = True,
    ) -> Any:
        collection = self.collection_for(dimension)
        if using not in VECTOR_NAMES:
            raise QdrantSchemaError(
                f"unsupported vector name {using!r}; expected one of {VECTOR_NAMES}"
            )
        return self._client.query_points(
            collection_name=collection,
            query=list(query),
            using=str(using),
            limit=int(limit),
            with_payload=with_payload,
        )

    def ensure_dual_collections(self, allowed_range: tuple[int, int]) -> dict[int, Any]:
        report: dict[int, Any] = {}
        for dimension in (FULL_DIMENSION, DERIVED_DIMENSION):
            self.create_collection(dimension)
            report[dimension] = self.verify_schema(dimension, allowed_range)
        return report

    def point_count(self, dimension: int) -> int:
        collection = self.collection_for(dimension)
        info = self._client.get_collection(collection_name=collection)
        return int(getattr(info, "points_count", 0))

    def upsert_points(
        self,
        dimension: int,
        points: list[models.PointStruct],
        wait: bool = True,
    ) -> None:
        collection = self.collection_for(dimension)
        self._client.upsert(collection_name=collection, points=points, wait=wait)
