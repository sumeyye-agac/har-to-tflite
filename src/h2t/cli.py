from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from h2t.config import apply_overrides, deep_merge, load_config
from h2t.constants import DEFAULT_CONFIG_PATH
from h2t.data.registry import load_dataset
from h2t.export.tflite_export import export_tflite_variants
from h2t.logging_utils import setup_logging
from h2t.training.train import train_pipeline
from h2t.utils.jsonio import write_json
from h2t.utils.paths import ensure_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="h2t", description="HAR to TFLite pipeline")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to YAML config")
    parser.add_argument("--set", action="append", default=[], help="Override key=value (dot paths)")

    subparsers = parser.add_subparsers(dest="command")
    for name in ("data", "train", "export", "bench-host", "bench-android", "report", "run-all"):
        subparsers.add_parser(name)
    return parser


def load_runtime_config(config_path: str, overrides: list[str]) -> dict:
    default_cfg = load_config(DEFAULT_CONFIG_PATH)
    user_cfg = load_config(config_path) if Path(config_path).exists() else {}
    merged = deep_merge(default_cfg, user_cfg)
    return apply_overrides(merged, overrides)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0

    config = load_runtime_config(args.config, args.set)
    logger = setup_logging(Path(config["paths"]["results_dir"]) / "run.log")
    command = args.command
    if command == "data":
        return _run_data(config, logger)
    if command == "train":
        return _run_train(config, logger)
    if command == "export":
        return _run_export(config, logger)
    if command in {"bench-host", "bench-android", "report", "run-all"}:
        logger.info("Command %s is scaffolded and will be implemented in subsequent milestones.", command)
        return 0
    return 0


def _run_data(config: dict[str, Any], logger) -> int:
    artifacts_dir = ensure_dir(config["paths"]["artifacts_dir"])
    dataset = _load_data(config, logger)
    summary = {
        "source": dataset["source"],
        "x_train_shape": list(dataset["x_train"].shape),
        "x_test_shape": list(dataset["x_test"].shape),
        "num_classes": dataset["num_classes"],
        "input_shape": list(dataset["input_shape"]),
    }
    write_json(artifacts_dir / "data_summary.json", summary)
    logger.info("Saved data summary to %s", artifacts_dir / "data_summary.json")
    return 0


def _run_train(config: dict[str, Any], logger) -> int:
    dataset = _load_data(config, logger)
    _write_data_summary(config, dataset, logger)
    train_pipeline(config, dataset, logger)
    return 0


def _load_data(config: dict[str, Any], logger) -> dict[str, Any]:
    return load_dataset(config, logger)


def _run_export(config: dict[str, Any], logger) -> int:
    dataset = _load_data(config, logger)
    _write_data_summary(config, dataset, logger)
    train_result = train_pipeline(config, dataset, logger)
    export_tflite_variants(config, dataset, train_result, logger)
    return 0


def _write_data_summary(config: dict[str, Any], dataset: dict[str, Any], logger) -> None:
    artifacts_dir = ensure_dir(config["paths"]["artifacts_dir"])
    summary = {
        "source": dataset["source"],
        "x_train_shape": list(dataset["x_train"].shape),
        "x_test_shape": list(dataset["x_test"].shape),
        "num_classes": dataset["num_classes"],
        "input_shape": list(dataset["input_shape"]),
    }
    write_json(artifacts_dir / "data_summary.json", summary)
    logger.info("Saved data summary to %s", artifacts_dir / "data_summary.json")
