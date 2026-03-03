from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from h2t.config import apply_overrides, deep_merge, load_config
from h2t.bench.android import benchmark_android
from h2t.bench.host import benchmark_host
from h2t.constants import DEFAULT_CONFIG_PATH
from h2t.data.registry import load_dataset
from h2t.export.tflite_export import export_tflite_variants
from h2t.logging_utils import setup_logging
from h2t.reporting.report import write_summary
from h2t.training.train import train_pipeline
from h2t.utils.jsonio import write_json
from h2t.utils.paths import ensure_dir


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to YAML config")
    common.add_argument("--set", action="append", default=[], help="Override key=value (dot paths)")
    parser = argparse.ArgumentParser(prog="h2t", description="HAR to TFLite pipeline", parents=[common])

    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("data", parents=[common])
    subparsers.add_parser("train", parents=[common])
    subparsers.add_parser("export", parents=[common])
    subparsers.add_parser("bench-host", parents=[common])
    android = subparsers.add_parser("bench-android", parents=[common])
    android.add_argument("--serial", default=None, help="ADB device serial")
    android.add_argument("--benchmark-bin", default=None, help="Path to benchmark_model binary on host")
    android.add_argument("--threads", type=int, default=None, help="CPU threads")
    android.add_argument("--use-nnapi", action="store_true", default=None, help="Enable NNAPI")
    android.add_argument("--repeat", type=int, default=None, help="Repeat count")
    android.add_argument("--cooldown-s", type=float, default=None, help="Cooldown in seconds")
    android.add_argument("--warmup-runs", type=int, default=None, help="Warmup runs")
    android.add_argument("--num-runs", type=int, default=None, help="Benchmark runs per repeat")
    subparsers.add_parser("report", parents=[common])
    subparsers.add_parser("run-all", parents=[common])
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
    if command == "bench-host":
        return _run_bench_host(config, logger)
    if command == "bench-android":
        overrides = {
            "serial": args.serial,
            "benchmark_bin": args.benchmark_bin,
            "threads": args.threads,
            "use_nnapi": args.use_nnapi,
            "repeat": args.repeat,
            "cooldown_s": args.cooldown_s,
            "warmup_runs": args.warmup_runs,
            "num_runs": args.num_runs,
        }
        return _run_bench_android(config, logger, overrides)
    if command == "report":
        return _run_report(config, logger)
    if command == "run-all":
        return _run_all(config, logger)
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


def _run_bench_host(config: dict[str, Any], logger) -> int:
    dataset = _load_data(config, logger)
    _write_data_summary(config, dataset, logger)
    train_result = train_pipeline(config, dataset, logger)
    manifest = export_tflite_variants(config, dataset, train_result, logger)
    benchmark_host(config, dataset, manifest, logger)
    return 0


def _run_bench_android(config: dict[str, Any], logger, overrides: dict[str, Any]) -> int:
    dataset = _load_data(config, logger)
    _write_data_summary(config, dataset, logger)
    train_result = train_pipeline(config, dataset, logger)
    manifest = export_tflite_variants(config, dataset, train_result, logger)
    benchmark_android(config, manifest, logger, cli_overrides=overrides)
    return 0


def _run_report(config: dict[str, Any], logger) -> int:
    summary_path, leaderboard_path = write_summary(config)
    logger.info("Saved summary report to %s", summary_path)
    logger.info("Updated leaderboard at %s", leaderboard_path)
    return 0


def _run_all(config: dict[str, Any], logger) -> int:
    dataset = _load_data(config, logger)
    _write_data_summary(config, dataset, logger)

    train_result = train_pipeline(config, dataset, logger)
    manifest = export_tflite_variants(config, dataset, train_result, logger)

    host_enabled = bool(config.get("bench", {}).get("host", {}).get("enabled", True))
    if host_enabled:
        benchmark_host(config, dataset, manifest, logger)

    android_enabled = bool(config.get("bench", {}).get("android", {}).get("enabled", True))
    if android_enabled:
        benchmark_android(config, manifest, logger)

    summary_path, leaderboard_path = write_summary(config)
    logger.info("Saved summary report to %s", summary_path)
    logger.info("Updated leaderboard at %s", leaderboard_path)
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
