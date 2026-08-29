import pytest

from wemm_kaggle.api.config import ApiSettings


def make(mapping):
    return ApiSettings.from_mapping(mapping)


def test_defaults_from_minimal_mapping():
    settings = make({"WEMM_API_TOKEN": "x" * 48})
    assert settings.host == "127.0.0.1"
    assert settings.port == 8090
    assert settings.max_batch == 4
    assert settings.queue_max_items == 16
    assert settings.request_timeout_s == 30.0
    assert settings.max_text_chars == 8192
    assert settings.max_image_bytes == 8 * 1024 * 1024
    assert settings.max_image_edge == 4096
    assert settings.max_image_pixels == 16_777_216
    assert settings.per_gpu_mib == 14200


def test_missing_token_is_a_startup_failure():
    with pytest.raises(ValueError):
        make({})


def test_short_token_is_a_startup_failure():
    with pytest.raises(ValueError):
        make({"WEMM_API_TOKEN": "x" * 31})


def test_zero_0_0_0_0_bind_is_rejected():
    with pytest.raises(ValueError):
        make({"WEMM_API_TOKEN": "x" * 48, "WEMM_API_HOST": "0.0.0.0"})


def test_public_ip_bind_is_rejected():
    with pytest.raises(ValueError):
        make({"WEMM_API_TOKEN": "x" * 48, "WEMM_API_HOST": "8.8.8.8"})


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_loopback_hosts_accepted(host):
    settings = make({"WEMM_API_TOKEN": "x" * 48, "WEMM_API_HOST": host})
    assert settings.host == host


def test_custom_limits_are_parsed():
    settings = make(
        {
            "WEMM_API_TOKEN": "x" * 48,
            "WEMM_API_PORT": "8888",
            "WEMM_MAX_BATCH": "2",
            "WEMM_MAX_TEXT_CHARS": "4096",
            "WEMM_QUEUE_MAX_ITEMS": "64",
            "WEMM_REQUEST_TIMEOUT_S": "60.5",
            "WEMM_MAX_IMAGE_BYTES": str(4 * 1024 * 1024),
            "WEMM_API_EVIDENCE_DIR": "/tmp/evidences",
        }
    )
    assert settings.port == 8888
    assert settings.max_batch == 2
    assert settings.max_text_chars == 4096
    assert settings.queue_max_items == 64
    assert settings.request_timeout_s == 60.5
    assert settings.max_image_bytes == 4 * 1024 * 1024
    assert settings.evidence_dir is not None
    assert str(settings.evidence_dir) == "/tmp/evidences"

@pytest.mark.parametrize(("name","value"),[("WEMM_MAX_BATCH","0"),("WEMM_MAX_BATCH","5"),("WEMM_MAX_TEXT_CHARS","0"),("WEMM_MAX_TEXT_CHARS","8193")])
def test_request_protocol_caps_fail_closed(name,value):
    with pytest.raises(ValueError):
        make({"WEMM_API_TOKEN":"x"*48,name:value})
