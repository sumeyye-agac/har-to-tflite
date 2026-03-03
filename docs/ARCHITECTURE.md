# Architecture Overview

The pipeline is intentionally linear and fault-tolerant.

## Flow

1. `data`: load UCI HAR, fallback to synthetic if unavailable.
2. `train`: train tiny model (or deterministic stub backend if TensorFlow missing).
3. `export`: write TFLite variants and an `export_manifest.json`.
4. `bench host`: benchmark available TFLite models on CPU.
5. `bench android`: run ADB benchmark if environment is ready; otherwise emit stub.
6. `report`: aggregate all artifacts into `results/summary.md` and `results/leaderboard.csv`.

## Main artifacts

- `artifacts/data_summary.json`
- `artifacts/train_metrics.json`
- `artifacts/export_manifest.json`
- `results/bench_host.csv`
- `results/bench_android.csv`
- `results/summary.md`
- `results/leaderboard.csv`

## Failure model

- Data unavailable: synthetic generation fallback
- TensorFlow unavailable: deterministic stub training and export skip manifest
- Android not available: stub CSV with reason

The top-level `run-all` command should complete successfully in all of the above scenarios.
