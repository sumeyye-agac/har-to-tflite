from __future__ import annotations

import csv
import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from h2t.utils.paths import ensure_dir


def benchmark_host(config: dict[str, Any], dataset: dict[str, Any], manifest: dict[str, Any], logger: logging.Logger) -> Path:
    results_dir = ensure_dir(config["paths"]["results_dir"])
    bench_path = results_dir / "bench_host.csv"
    host_cfg = config.get("bench", {}).get("host", {})
    warmup_runs = int(host_cfg.get("warmup_runs", 5))
    num_runs = int(host_cfg.get("num_runs", 20))
    threads = int(host_cfg.get("threads", 1))

    rows: list[dict[str, Any]] = []
    variants = manifest.get("variants", {})
    runnable = [(name, Path(info["path"])) for name, info in variants.items() if info.get("status") == "ok" and info.get("path")]
    if not runnable:
        rows.append(_stub_row("none", "no_tflite_models_available"))
        _write_rows(bench_path, rows)
        logger.warning("No runnable TFLite models found; wrote stub host benchmark csv.")
        return bench_path

    try:
        import tensorflow as tf
    except Exception as exc:
        rows.append(_stub_row("none", f"tensorflow_unavailable: {exc}"))
        _write_rows(bench_path, rows)
        logger.warning("TensorFlow unavailable for host benchmark; wrote stub csv.")
        return bench_path

    x_test = dataset["x_test"]
    for variant, model_path in runnable:
        try:
            interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=threads)
            interpreter.allocate_tensors()
            input_detail = interpreter.get_input_details()[0]
            input_idx = input_detail["index"]
            input_dtype = input_detail["dtype"]
            shape = tuple(input_detail["shape"])
            timings_ms: list[float] = []

            for _ in range(max(0, warmup_runs)):
                sample = _prepare_sample(x_test, shape, input_dtype, input_detail)
                interpreter.set_tensor(input_idx, sample)
                interpreter.invoke()

            for run_idx in range(max(1, num_runs)):
                sample = _prepare_sample(x_test, shape, input_dtype, input_detail, index=run_idx % x_test.shape[0])
                start = perf_counter()
                interpreter.set_tensor(input_idx, sample)
                interpreter.invoke()
                elapsed_ms = (perf_counter() - start) * 1000.0
                timings_ms.append(elapsed_ms)

            rows.append(
                {
                    "variant": variant,
                    "status": "ok",
                    "mean_ms": round(float(np.mean(timings_ms)), 4),
                    "p50_ms": round(float(np.percentile(timings_ms, 50)), 4),
                    "p90_ms": round(float(np.percentile(timings_ms, 90)), 4),
                    "num_runs": len(timings_ms),
                    "threads": threads,
                    "reason": "",
                }
            )
        except Exception as exc:  # pragma: no cover
            rows.append(_stub_row(variant, f"benchmark_failed: {exc}"))
            logger.warning("Host benchmark failed for %s: %s", variant, exc)

    _write_rows(bench_path, rows)
    logger.info("Saved host benchmark csv to %s", bench_path)
    return bench_path


def _prepare_sample(
    x_test: np.ndarray,
    expected_shape: tuple[int, ...],
    expected_dtype,
    input_detail: dict[str, Any],
    index: int = 0,
) -> np.ndarray:
    sample = x_test[index : index + 1]
    if tuple(sample.shape) != tuple(expected_shape):
        sample = np.resize(sample, expected_shape)
    if expected_dtype == np.int8:
        scale, zero = input_detail.get("quantization", (0.0, 0))
        if scale and scale > 0:
            sample = np.clip(np.round(sample / scale + zero), -128, 127).astype(np.int8)
        else:
            sample = sample.astype(np.int8)
    else:
        sample = sample.astype(expected_dtype)
    return sample


def _stub_row(variant: str, reason: str) -> dict[str, Any]:
    return {
        "variant": variant,
        "status": "stub",
        "mean_ms": 0.0,
        "p50_ms": 0.0,
        "p90_ms": 0.0,
        "num_runs": 0,
        "threads": 0,
        "reason": reason,
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["variant", "status", "mean_ms", "p50_ms", "p90_ms", "num_runs", "threads", "reason"],
        )
        writer.writeheader()
        writer.writerows(rows)
