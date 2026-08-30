from __future__ import annotations

import json

import pytest

from wemm_kaggle.search.config import SearchSettings, LOOPBACK_HOSTS
from wemm_kaggle.search.point_id import qid_to_point_id


def test_defaults_loopback():
    s = SearchSettings.from_mapping({})
    assert s.search_host == "127.0.0.1"
    assert s.search_port == 8091
    assert s.embedding_url == "http://127.0.0.1:8090"


def test_env_parsing():
    s = SearchSettings.from_mapping(
        {
            "WEMM_SEARCH_HOST": "localhost",
            "WEMM_SEARCH_PORT": "9000",
            "WEMM_API_TOKEN": "secret",
            "WEMM_EMBEDDING_URL": "http://127.0.0.1:8090",
        }
    )
    assert s.search_port == 9000
    assert s.embedding_token == "secret"


def test_loopback_hosts():
    assert LOOPBACK_HOSTS == {"127.0.0.1", "localhost", "::1"}


@pytest.mark.parametrize("host", ["0.0.0.0", "10.0.0.5", "example.com", ""])
def test_reject_non_loopback(host):
    with pytest.raises(ValueError):
        SearchSettings(search_host=host)


def test_reject_bad_port():
    with pytest.raises(ValueError):
        SearchSettings.from_mapping({"WEMM_SEARCH_PORT": "not-a-number"})


def test_collection_names_defaults_match_public_authority():
    s = SearchSettings()
    assert s.collections_4096 == "wikidata_en_vi_wemm9b_4096_v030_rc2_d819dc7_v1"
    assert s.collections_1024 == "wikidata_en_vi_wemm9b_1024_v030_rc2_d819dc7_v1"


def test_collection_names_env_override():
    s = SearchSettings.from_mapping(
        {
            "WEMM_COLLECTION_4096": "custom_4096",
            "WEMM_COLLECTION_1024": "custom_1024",
        }
    )
    assert s.collections_4096 == "custom_4096"
    assert s.collections_1024 == "custom_1024"
