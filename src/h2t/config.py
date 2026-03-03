from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a dictionary: {path}")
    return data


def validate_config(config: dict[str, Any]) -> None:
    required_top = ("paths", "dataset", "training", "export", "bench")
    for key in required_top:
        if key not in config or not isinstance(config[key], dict):
            raise ValueError(f"Missing or invalid config section: {key}")

    paths = config["paths"]
    for key in ("data_dir", "artifacts_dir", "results_dir"):
        value = paths.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"paths.{key} must be a non-empty string")

    training = config["training"]
    _require_positive_int(training, "epochs")
    _require_positive_int(training, "batch_size")

    export = config["export"]
    _require_positive_int(export, "representative_samples")

    host_bench = config["bench"].get("host", {})
    _require_non_negative_int(host_bench, "warmup_runs")
    _require_positive_int(host_bench, "num_runs")
    _require_positive_int(host_bench, "threads")

    android_bench = config["bench"].get("android", {})
    _require_positive_int(android_bench, "repeat")
    _require_non_negative_int(android_bench, "warmup_runs")
    _require_positive_int(android_bench, "num_runs")
    _require_positive_int(android_bench, "threads")


def _require_positive_int(section: dict[str, Any], key: str) -> None:
    value = section.get(key)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")


def _require_non_negative_int(section: dict[str, Any], key: str) -> None:
    value = section.get(key)
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer")


def apply_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    if not overrides:
        return config
    merged = deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override format: {item}")
        key, raw_value = item.split("=", 1)
        parts = key.split(".")
        cursor = merged
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = _parse_scalar(raw_value)
    return merged


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    for caster in (int, float):
        try:
            return caster(value)
        except ValueError:
            pass
    return value
