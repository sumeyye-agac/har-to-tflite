# har-to-tflite

[![CI](https://github.com/sumeyye-agac/har-to-tflite/actions/workflows/ci.yml/badge.svg)](https://github.com/sumeyye-agac/har-to-tflite/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)

Train a tiny HAR classifier and export reproducible TensorFlow Lite variants (`fp32`, `fp16`, `int8`, `drq`) with host and Android benchmarks.

## Project status

This repository is under active development. APIs, configs, and benchmark outputs may change until the first stable release.

## Quickstart

```bash
pip install -e .
h2t run-all --config configs/smoke.yaml
cat results/summary.md
```

`configs/smoke.yaml` is CPU-only and offline-safe (synthetic data), and completes quickly.

## Why this repo

- End-to-end CLI: data -> train -> export -> bench -> report
- Robust fallbacks: pipeline finishes even without internet, TensorFlow, ADB, device, or benchmark binary
- Reproducibility output: `results/env.txt`, deterministic seeds, structured manifests/CSVs

## Quantization modes

- `fp32`: baseline float model, highest fidelity, largest size
- `fp16`: reduced size with float16 weights
- `int8` (full integer): representative-dataset calibration for best edge latency/size
- `drq` (dynamic-range quantization): automatic fallback when full-int8 conversion fails

Export details are written to `artifacts/export_manifest.json`.

## CLI

```bash
h2t data --config configs/default.yaml
h2t train --config configs/default.yaml
h2t export --config configs/default.yaml
h2t bench host --config configs/default.yaml --threads 4
h2t bench android --config configs/default.yaml --serial <device> --benchmark-bin /path/to/benchmark_model
h2t report --config configs/default.yaml
h2t run-all --config configs/default.yaml
```

Extra docs:

- `docs/ARCHITECTURE.md`
- `CHANGELOG.md`

## Example output tree

```text
artifacts/
  data_summary.json
  train_metrics.json
  export_manifest.json
  models/
    student.keras | student_stub.npz
  tflite/
    model_fp32.tflite
    model_fp16.tflite
    model_int8.tflite
    model_drq.tflite
results/
  run.log
  env.txt
  config_effective.yaml
  git_rev.txt
  bench_host.csv
  bench_android.csv
  summary.md
  leaderboard.csv
  android_raw/
    *.log
```

## Android prerequisites

Android benchmarking is optional. `run-all` still succeeds without Android.

Required for real Android runs:

- `adb` in `PATH`
- Connected device (`adb devices` shows `device`)
- TFLite `benchmark_model` binary (host path via `--benchmark-bin` or `bench.android.benchmark_bin`)

If missing, `results/bench_android.csv` is written as a stub with the reason.

See:

- `scripts/android/README_android_bench.md`
- `scripts/android/build_or_get_benchmark_model.md`

## Common bottlenecks & auto-fallbacks

- Dataset download blocked/offline: auto-fallback to synthetic HAR-like dataset
- TensorFlow missing: deterministic centroid-model training + stub export/bench manifests
- Full int8 conversion fails: DRQ fallback is exported and reason is recorded
- ADB/device/binary missing: Android benchmark writes stub CSV and pipeline continues
- Android output parse variance: multiple regex patterns + raw log capture to `results/android_raw/`

## Configuration

`configs/default.yaml` is the full run profile. `configs/smoke.yaml` is fast and offline.

Override any key at runtime:

```bash
h2t run-all --config configs/default.yaml --set training.epochs=1 --set bench.host.threads=2
```

## Development

```bash
python -m pip install -e .
PYTHONPATH=. pytest -q
```

## Roadmap

- Optional distillation flow from teacher to tiny student
- Better per-op profiling summaries for TFLite backends
- Device matrix leaderboard export for multi-phone comparisons

## License

MIT. See `LICENSE`.
