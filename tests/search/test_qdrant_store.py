from __future__ import annotations

import pytest

from wemm_kaggle.search.qdrant_store import (
    COLLECTION_4096,
    COLLECTION_1024,
    QdrantStore,
    QdrantSchemaError,
    collection_name_for,
    VECTOR_NAMES,
)
from wemm_kaggle.search.mrl import FULL_DIMENSION, DERIVED_DIMENSION


class FakeVector:
    def __init__(self, size, distance="Cosine"):
        self.size = size
        self.distance = distance


class FakeParams:
    def __init__(self, sizes=(FULL_DIMENSION,), dtype=None):
        self.vectors = {}
        self.shard_number = 1
        self.replication_factor = 1
        self.write_consistency_factor = 1


class FakeConfig:
    def __init__(self, sizes, names=("en", "vi"), distance="Cosine"):
        self.params = FakeParams()
        self.params.vectors = {name: FakeVector(sizes, distance) for name in names}


class FakeInfo:
    def __init__(self, sizes, count, names=("en", "vi"), distance="Cosine"):
        self.config = FakeConfig(sizes, names=names, distance=distance)
        self.vectors = None  # Qdrant 1.19 returns None; must read via config.params.vectors
        self.points_count = count


class FakeClient:
    def __init__(self, info=None, fs_calls=None):
        self._info = info
        self._fs_calls = fs_calls if fs_calls is not None else {}
        self.collections = set()

    def get_collection(self, collection_name):
        if self._info is not None:
            return self._info
        raise KeyError(collection_name)

    def collection_exists(self, collection_name):
        return collection_name in self.collections

    def create_collection(self, **kw):
        self.collections.add(kw["collection_name"])

    def upsert(self, **kw):
        pass


def test_collection_name_for():
    assert collection_name_for(FULL_DIMENSION) == COLLECTION_4096
    assert collection_name_for(DERIVED_DIMENSION) == COLLECTION_1024
    with pytest.raises(QdrantSchemaError):
        collection_name_for(2048)


def test_default_collection_names_match_public_authority():
    assert COLLECTION_4096 == "wikidata_en_vi_wemm9b_4096_v030_rc2_d819dc7_v1"
    assert COLLECTION_1024 == "wikidata_en_vi_wemm9b_1024_v030_rc2_d819dc7_v1"


def test_collection_for_uses_configured_names():
    store = QdrantStore(
        client=FakeClient(),
        collection_4096="custom_4096",
        collection_1024="custom_1024",
    )
    assert store.collection_for(FULL_DIMENSION) == "custom_4096"
    assert store.collection_for(DERIVED_DIMENSION) == "custom_1024"
    assert store.collection_for(FULL_DIMENSION) != COLLECTION_4096
    assert store.collection_for(DERIVED_DIMENSION) != COLLECTION_1024
    with pytest.raises(QdrantSchemaError):
        store.collection_for(2048)


def test_constants():
    assert VECTOR_NAMES == ("en", "vi")


def test_verify_schema_reads_config_params_vectors():
    # Simulates Qdrant 1.19 where info.vectors is None and named vectors live in config.params.vectors
    info = FakeInfo(FULL_DIMENSION, 5000)
    store = QdrantStore(client=FakeClient(info=info))
    report = store.verify_schema(FULL_DIMENSION, (0, 99967))
    assert report["dimension"] == FULL_DIMENSION
    assert set(report["vector_names"]) == {"en", "vi"}
    assert report["point_count"] == 5000
    assert report["distances"]["en"] == "Cosine"
    assert report["distances"]["vi"] == "Cosine"


def test_verify_schema_rejects_non_cosine_distance():
    info = FakeInfo(FULL_DIMENSION, 5000, distance="Dot")
    store = QdrantStore(client=FakeClient(info=info))
    with pytest.raises(QdrantSchemaError):
        store.verify_schema(FULL_DIMENSION, (0, 99967))


def test_verify_schema_rejects_missing_named_vector():
    info = FakeInfo(FULL_DIMENSION, 5000, names=("en",))
    store = QdrantStore(client=FakeClient(info=info))
    with pytest.raises(QdrantSchemaError):
        store.verify_schema(FULL_DIMENSION, (0, 99967))


def test_verify_schema_rejects_unexpected_named_vector():
    info = FakeInfo(FULL_DIMENSION, 5000, names=("en", "vi", "extra"))
    store = QdrantStore(client=FakeClient(info=info))
    with pytest.raises(QdrantSchemaError):
        store.verify_schema(FULL_DIMENSION, (0, 99967))


def test_verify_schema_wrong_size():
    info = FakeInfo(DERIVED_DIMENSION, 5000)
    store = QdrantStore(client=FakeClient(info=info))
    with pytest.raises(QdrantSchemaError):
        store.verify_schema(FULL_DIMENSION, (0, 99967))


def test_verify_schema_point_range():
    info = FakeInfo(FULL_DIMENSION, 100000)
    store = QdrantStore(client=FakeClient(info=info))
    with pytest.raises(QdrantSchemaError):
        store.verify_schema(FULL_DIMENSION, (0, 99967))


def test_create_collection_when_absent():
    fc = FakeClient()
    store = QdrantStore(client=fc)
    store.create_collection(FULL_DIMENSION)
    assert COLLECTION_4096 in fc.collections


def test_create_collection_skips_when_present():
    fc = FakeClient()
    fc.collections.add(COLLECTION_4096)
    store = QdrantStore(client=fc)
    store.create_collection(FULL_DIMENSION)
    assert COLLECTION_4096 in fc.collections
