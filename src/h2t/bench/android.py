from __future__ import annotations

import csv
import logging
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from h2t.bench.parse_benchmark_output import parse_android_benchmark_output
from h2t.utils.paths import ensure_dir
from h2t.utils.subprocess_utils import run_command


def benchmark_android(
    config: dict[str, Any],
    manifest: dict[str, Any],
    logger: logging.Logger,
    cli_overrides: dict[str, Any] | None = None,
) -> Path:
    results_dir = ensure_dir(config["paths"]["results_dir"])
    bench_path = results_dir / "bench_android.csv"
    raw_dir = ensure_dir(results_dir / "android_raw")
    overrides = cli_overrides or {}
    android_cfg = dict(config.get("bench", {}).get("android", {}))
    android_cfg.update({k: v for k, v in overrides.items() if v is not None})

    serial = str(android_cfg.get("serial", "")).strip()
    benchmark_bin = str(android_cfg.get("benchmark_bin", "")).strip()
    threads = int(android_cfg.get("threads", 1))
    use_nnapi = bool(android_cfg.get("use_nnapi", False))
    repeat = int(android_cfg.get("repeat", 3))
    cooldown_s = float(android_cfg.get("cooldown_s", 1.0))
    warmup_runs = int(android_cfg.get("warmup_runs", 5))
    num_runs = int(android_cfg.get("num_runs", 30))

    rows: list[dict[str, Any]] = []
    adb = shutil.which("adb")
    if not adb:
        rows.append(_stub_row("none", "adb_missing"))
        _write_rows(bench_path, rows)
        logger.warning("ADB is not installed; wrote stub android benchmark csv.")
        return bench_path

    adb_base = [adb]
    if serial:
        adb_base.extend(["-s", serial])

    if not _device_connected(adb_base):
        rows.append(_stub_row("none", "no_android_device"))
        _write_rows(bench_path, rows)
        logger.warning("No Android device detected; wrote stub android benchmark csv.")
        return bench_path

    device_bin = "/data/local/tmp/benchmark_model"
    local_bin = _resolve_benchmark_bin(benchmark_bin)
    if not local_bin:
        rows.append(_stub_row("none", "benchmark_binary_missing"))
        _write_rows(bench_path, rows)
        logger.warning("benchmark_model binary not found; wrote stub android benchmark csv.")
        return bench_path

    push_bin = run_command(adb_base + ["push", local_bin, device_bin], timeout=60)
    run_command(adb_base + ["shell", "chmod", "+x", device_bin], timeout=30)
    if push_bin.code != 0:
        rows.append(_stub_row("none", f"benchmark_binary_push_failed: {push_bin.stderr.strip()}"))
        _write_rows(bench_path, rows)
        logger.warning("Failed to push benchmark binary; wrote stub csv.")
        return bench_path

    variants = manifest.get("variants", {})
    runnable = [(name, Path(info["path"])) for name, info in variants.items() if info.get("status") == "ok" and info.get("path")]
    if not runnable:
        rows.append(_stub_row("none", "no_tflite_models_available"))
        _write_rows(bench_path, rows)
        return bench_path

    for variant, local_model in runnable:
        device_model = f"/data/local/tmp/{local_model.name}"
        push_model = run_command(adb_base + ["push", str(local_model), device_model], timeout=60)
        if push_model.code != 0:
            rows.append(_stub_row(variant, f"model_push_failed: {push_model.stderr.strip()}"))
            continue

        repeat_values: list[float] = []
        last_output = ""
        for rep in range(max(1, repeat)):
            cmd = [
                device_bin,
                f"--graph={device_model}",
                f"--num_threads={threads}",
                f"--warmup_runs={warmup_runs}",
                f"--num_runs={num_runs}",
                f"--use_nnapi={'true' if use_nnapi else 'false'}",
            ]
            run = run_command(adb_base + ["shell"] + cmd, timeout=120)
            last_output = (run.stdout or "") + ("\n" + run.stderr if run.stderr else "")
            parsed = parse_android_benchmark_output(last_output)
            raw_path = raw_dir / f"{variant}_repeat_{rep + 1}.log"
            raw_path.write_text(last_output, encoding="utf-8")

            if run.code == 0 and parsed.get("ok"):
                repeat_values.append(float(parsed["avg_ms"]))
            if rep < repeat - 1:
                time.sleep(max(0.0, cooldown_s))

        if repeat_values:
            rows.append(
                {
                    "variant": variant,
                    "status": "ok",
                    "mean_ms": round(float(np.mean(repeat_values)), 4),
                    "p50_ms": round(float(np.percentile(repeat_values, 50)), 4),
                    "p90_ms": round(float(np.percentile(repeat_values, 90)), 4),
                    "runs": len(repeat_values),
                    "threads": threads,
                    "use_nnapi": use_nnapi,
                    "reason": "",
                }
            )
        else:
            reason = parse_android_benchmark_output(last_output).get("reason", "parse_failed")
            rows.append(_stub_row(variant, reason, threads=threads, use_nnapi=use_nnapi))

    _write_rows(bench_path, rows)
    logger.info("Saved android benchmark csv to %s", bench_path)
    return bench_path


def _device_connected(adb_base: list[str]) -> bool:
    result = run_command(adb_base + ["devices"], timeout=30)
    if result.code != 0:
        return False
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return any("\tdevice" in line for line in lines[1:])


def _resolve_benchmark_bin(configured: str) -> str:
    if configured:
        p = Path(configured)
        if p.exists():
            return str(p)
    from_path = shutil.which("benchmark_model")
    if from_path:
        return from_path
    return ""


def _stub_row(variant: str, reason: str, threads: int = 0, use_nnapi: bool = False) -> dict[str, Any]:
    return {
        "variant": variant,
        "status": "stub",
        "mean_ms": 0.0,
        "p50_ms": 0.0,
        "p90_ms": 0.0,
        "runs": 0,
        "threads": threads,
        "use_nnapi": use_nnapi,
        "reason": reason,
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["variant", "status", "mean_ms", "p50_ms", "p90_ms", "runs", "threads", "use_nnapi", "reason"],
        )
        writer.writeheader()
        writer.writerows(rows)
