from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path
from typing import Any

from h2t.bench.android import benchmark_android
from h2t.bench.host import benchmark_host
from h2t.config import apply_overrides, deep_merge, load_config, validate_config
from h2t.constants import DEFAULT_CONFIG_PATH
from h2t.data.registry import load_dataset
from h2t.data.synthetic import generate_synthetic_har
from h2t.export.tflite_export import export_tflite_variants
from h2t.logging_utils import setup_logging
from h2t.reporting.report import write_summary
from h2t.training.train import train_pipeline
from h2t.utils.jsonio import write_json
from h2t.utils.paths import ensure_dir
from h2t.utils.reproducibility import write_env_snapshot


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to YAML config")
    common.add_argument("--set", action="append", default=[], help="Override key=value (dot paths)")

    parser = argparse.ArgumentParser(prog="h2t", description="HAR to TFLite pipeline", parents=[common])
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("data", parents=[common])
    subparsers.add_parser("train", parents=[common])
    subparsers.add_parser("export", parents=[common])
    subparsers.add_parser("report", parents=[common])
    subparsers.add_parser("run-all", parents=[common])

    bench = subparsers.add_parser("bench", parents=[common], help="Benchmarks")
    bench_subparsers = bench.add_subparsers(dest="bench_command")

    host = bench_subparsers.add_parser("host", parents=[common], help="Host TFLite benchmark")
    host.add_argument("--threads", type=int, default=None, help="CPU threads")
    host.add_argument("--warmup-runs", type=int, default=None, help="Warmup runs")
    host.add_argument("--num-runs", type=int, default=None, help="Run count")

    android = bench_subparsers.add_parser("android", parents=[common], help="Android benchmark via ADB")
    android.add_argument("--serial", default=None, help="ADB device serial")
    android.add_argument("--benchmark-bin", default=None, help="Path to benchmark_model binary on host")
    android.add_argument("--threads", type=int, default=None, help="CPU threads")
    android.add_argument("--use-nnapi", action="store_true", default=None, help="Enable NNAPI")
    android.add_argument("--repeat", type=int, default=None, help="Repeat count")
    android.add_argument("--cooldown-s", type=float, default=None, help="Cooldown in seconds")
    android.add_argument("--warmup-runs", type=int, default=None, help="Warmup runs")
    android.add_argument("--num-runs", type=int, default=None, help="Benchmark runs per repeat")

    return parser


def load_runtime_config(config_path: str, overrides: list[str]) -> dict[str, Any]:
    default_cfg = load_config(DEFAULT_CONFIG_PATH)
    user_cfg = load_config(config_path) if Path(config_path).exists() else {}
    merged = deep_merge(default_cfg, user_cfg)
    validated = apply_overrides(merged, overrides)
    validate_config(validated)
    return validated


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0

    config = load_runtime_config(args.config, args.set)
    logger = setup_logging(Path(config["paths"]["results_dir"]) / "run.log")
    _write_env(config)

    command = args.command
    if command == "data":
        return _run_data(config, logger)
    if command == "train":
        return _run_train(config, logger)
    if command == "export":
        return _run_export(config, logger)
    if command == "report":
        return _run_report(config, logger)
    if command == "run-all":
        return _run_all(config, logger)

    if command == "bench":
        return _run_bench_command(command, args, config, logger)

    logger.error("Unsupported command: %s", command)
    return 1


def _run_bench_command(command: str, args: argparse.Namespace, config: dict[str, Any], logger) -> int:
    if args.bench_command == "host":
        overrides = {"threads": args.threads, "warmup_runs": args.warmup_runs, "num_runs": args.num_runs}
        return _run_bench_host(config, logger, overrides)
    if args.bench_command == "android":
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
    logger.error("Usage: h2t bench {host|android} ...")
    return 2


def _run_data(config: dict[str, Any], logger) -> int:
    dataset = _load_data_with_failsafe(config, logger)
    _write_data_summary(config, dataset, logger)
    return 0


def _run_train(config: dict[str, Any], logger) -> int:
    dataset = _load_data_with_failsafe(config, logger)
    _write_data_summary(config, dataset, logger)
    train_pipeline(config, dataset, logger)
    return 0


def _run_export(config: dict[str, Any], logger) -> int:
    dataset = _load_data_with_failsafe(config, logger)
    _write_data_summary(config, dataset, logger)
    train_result = train_pipeline(config, dataset, logger)
    export_tflite_variants(config, dataset, train_result, logger)
    return 0


def _run_bench_host(config: dict[str, Any], logger, overrides: dict[str, Any] | None = None) -> int:
    local_cfg = _merge_stage_overrides(config, ["bench", "host"], overrides or {})
    dataset = _load_data_with_failsafe(local_cfg, logger)
    _write_data_summary(local_cfg, dataset, logger)
    train_result = train_pipeline(local_cfg, dataset, logger)
    manifest = export_tflite_variants(local_cfg, dataset, train_result, logger)
    benchmark_host(local_cfg, dataset, manifest, logger)
    return 0


def _run_bench_android(config: dict[str, Any], logger, overrides: dict[str, Any]) -> int:
    dataset = _load_data_with_failsafe(config, logger)
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
    dataset = _load_data_with_failsafe(config, logger)
    _write_data_summary(config, dataset, logger)

    train_result = _safe_train(config, dataset, logger)
    manifest = _safe_export(config, dataset, train_result, logger)

    if bool(config.get("bench", {}).get("host", {}).get("enabled", True)):
        try:
            benchmark_host(config, dataset, manifest, logger)
        except Exception as exc:
            logger.exception("Host benchmark failed: %s", exc)
            _write_stub_bench(config, "bench_host.csv", "host_benchmark_failed")

    if bool(config.get("bench", {}).get("android", {}).get("enabled", True)):
        try:
            benchmark_android(config, manifest, logger)
        except Exception as exc:
            logger.exception("Android benchmark failed: %s", exc)
            _write_stub_bench(config, "bench_android.csv", "android_benchmark_failed")

    summary_path, leaderboard_path = write_summary(config)
    logger.info("Saved summary report to %s", summary_path)
    logger.info("Updated leaderboard at %s", leaderboard_path)
    return 0


def _safe_train(config: dict[str, Any], dataset: dict[str, Any], logger) -> dict[str, Any]:
    try:
        return train_pipeline(config, dataset, logger)
    except Exception as exc:
        logger.exception("Training failed, writing stub metrics: %s", exc)
        artifacts_dir = ensure_dir(config["paths"]["artifacts_dir"])
        payload = {
            "status": "failed",
            "backend": "none",
            "student_model_path": "",
            "teacher_model_path": "",
            "eval_loss": 0.0,
            "eval_accuracy": 0.0,
            "reason": str(exc),
        }
        write_json(artifacts_dir / "train_metrics.json", payload)
        return payload


def _safe_export(config: dict[str, Any], dataset: dict[str, Any], train_result: dict[str, Any], logger) -> dict[str, Any]:
    try:
        return export_tflite_variants(config, dataset, train_result, logger)
    except Exception as exc:
        logger.exception("Export failed, writing stub manifest: %s", exc)
        artifacts_dir = ensure_dir(config["paths"]["artifacts_dir"])
        payload = {
            "status": "failed",
            "backend": train_result.get("backend", "unknown"),
            "model_source_path": train_result.get("student_model_path", ""),
            "variants": {
                name: {
                    "status": "failed",
                    "path": "",
                    "size_bytes": 0,
                    "reason": str(exc),
                }
                for name in ("fp32", "fp16", "int8", "drq")
            },
        }
        write_json(artifacts_dir / "export_manifest.json", payload)
        return payload


def _load_data_with_failsafe(config: dict[str, Any], logger) -> dict[str, Any]:
    try:
        return load_dataset(config, logger)
    except Exception as exc:
        logger.exception("Dataset load failed, switching to synthetic fallback: %s", exc)
        fallback_cfg = deepcopy(config)
        fallback_cfg.setdefault("dataset", {})["name"] = "synthetic"
        fallback_cfg["dataset"]["synthetic_fallback"] = True
        synth = generate_synthetic_har(seed=int(config.get("seed", 1337)), train_samples=512, test_samples=128)
        return {
            "x_train": synth.x_train,
            "y_train": synth.y_train,
            "x_test": synth.x_test,
            "y_test": synth.y_test,
            "input_shape": tuple(synth.x_train.shape[1:]),
            "num_classes": int(synth.y_train.max(initial=0) + 1),
            "source": "synthetic_fallback_on_error",
        }


def _merge_stage_overrides(config: dict[str, Any], path_parts: list[str], overrides: dict[str, Any]) -> dict[str, Any]:
    local_cfg = deepcopy(config)
    cursor = local_cfg
    for part in path_parts:
        cursor = cursor.setdefault(part, {})
    for key, value in overrides.items():
        if value is not None:
            cursor[key] = value
    return local_cfg


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


def _write_stub_bench(config: dict[str, Any], filename: str, reason: str) -> None:
    results_dir = ensure_dir(config["paths"]["results_dir"])
    path = results_dir / filename
    fieldnames = ["variant", "status", "mean_ms", "p50_ms", "p90_ms", "runs", "threads", "use_nnapi", "reason"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "variant": "none",
                "status": "stub",
                "mean_ms": 0.0,
                "p50_ms": 0.0,
                "p90_ms": 0.0,
                "runs": 0,
                "threads": 0,
                "use_nnapi": False,
                "reason": reason,
            }
        )


def _write_env(config: dict[str, Any]) -> None:
    results_dir = ensure_dir(config["paths"]["results_dir"])
    write_env_snapshot(results_dir / "env.txt")
