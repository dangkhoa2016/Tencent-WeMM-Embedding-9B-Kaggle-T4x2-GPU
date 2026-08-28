import pytest


def test_build_max_memory_reserves_two_gpu_limits():
    from wemm_kaggle.gpu import build_max_memory

    assert build_max_memory(14200) == {0: "14200MiB", 1: "14200MiB", "cpu": "2GiB"}


def test_validate_device_map_requires_both_gpus():
    from wemm_kaggle.gpu import validate_device_map

    with pytest.raises(RuntimeError, match="both GPU 0 and GPU 1"):
        validate_device_map({"model": 0, "lm_head": 0})


def test_validate_device_map_rejects_cpu_or_disk_offload():
    from wemm_kaggle.gpu import validate_device_map

    with pytest.raises(RuntimeError, match="offload"):
        validate_device_map({"model.embed": 0, "model.layers.0": 1, "lm_head": "cpu"})
    with pytest.raises(RuntimeError, match="offload"):
        validate_device_map({"model.embed": 0, "model.layers.0": 1, "lm_head": "disk"})


def test_validate_device_map_accepts_int_and_cuda_string_targets():
    from wemm_kaggle.gpu import validate_device_map

    report = validate_device_map({"vision": 0, "language": "cuda:1", "lm_head": 1})
    assert report.gpu_ids == (0, 1)
    assert report.offload_targets == ()
    assert report.module_counts[0] == 1
    assert report.module_counts[1] == 2
