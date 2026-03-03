import pytest

from h2t.config import _parse_scalar, apply_overrides, deep_merge, validate_config


def test_parse_scalar_basic() -> None:
    assert _parse_scalar("true") is True
    assert _parse_scalar("3") == 3


def test_deep_merge_and_overrides() -> None:
    base = {"a": {"b": 1, "c": 2}, "flag": False}
    override = {"a": {"c": 7}}
    merged = deep_merge(base, override)
    assert merged["a"]["b"] == 1
    assert merged["a"]["c"] == 7

    updated = apply_overrides(merged, ["a.b=9", "flag=true"])
    assert updated["a"]["b"] == 9
    assert updated["flag"] is True


def test_validate_config_accepts_expected_shape() -> None:
    config = {
        "paths": {"data_dir": "data", "artifacts_dir": "artifacts", "results_dir": "results"},
        "dataset": {"name": "synthetic"},
        "training": {"epochs": 1, "batch_size": 16},
        "export": {"representative_samples": 8},
        "bench": {
            "host": {"warmup_runs": 1, "num_runs": 2, "threads": 1},
            "android": {"repeat": 1, "warmup_runs": 1, "num_runs": 2, "threads": 1},
        },
    }
    validate_config(config)


def test_validate_config_rejects_invalid_values() -> None:
    bad = {
        "paths": {"data_dir": "data", "artifacts_dir": "artifacts", "results_dir": "results"},
        "dataset": {"name": "synthetic"},
        "training": {"epochs": 0, "batch_size": 16},
        "export": {"representative_samples": 8},
        "bench": {
            "host": {"warmup_runs": 1, "num_runs": 2, "threads": 1},
            "android": {"repeat": 1, "warmup_runs": 1, "num_runs": 2, "threads": 1},
        },
    }
    with pytest.raises(ValueError):
        validate_config(bad)
