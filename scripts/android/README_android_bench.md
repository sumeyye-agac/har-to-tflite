# Android Benchmarking Guide

This project can benchmark exported `.tflite` models on Android with TensorFlow Lite's `benchmark_model` binary.

## Prerequisites

1. Android SDK platform-tools (`adb`) available in `PATH`
2. USB debugging enabled on target device
3. Device authorized and visible in `adb devices`
4. `benchmark_model` binary available locally

## Basic command

```bash
h2t bench android \
  --config configs/default.yaml \
  --serial <device_serial> \
  --benchmark-bin /absolute/path/to/benchmark_model \
  --threads 4 \
  --repeat 5 \
  --cooldown-s 1.0 \
  --warmup-runs 5 \
  --num-runs 50
```

## Outputs

- `results/bench_android.csv`: parsed benchmark metrics or stub reason
- `results/android_raw/*.log`: raw benchmark stdout/stderr per repeat

## Variance control options

- `--repeat`: repeat benchmark command multiple times
- `--cooldown-s`: pause between repeats to reduce thermal drift
- `--threads`: pin benchmark thread count
- `--use-nnapi`: best-effort NNAPI path

## Troubleshooting

### `adb` not found

Install Android platform-tools and ensure `adb` is in `PATH`.

### No device detected

- Check cable/debugging permissions
- Run `adb kill-server && adb start-server`
- Reconnect device and re-authorize debugging prompt

### `benchmark_binary_missing`

Set `--benchmark-bin` to a valid local binary path or configure `bench.android.benchmark_bin` in YAML.

### Parse failures

If parsing fails, inspect `results/android_raw/*.log`. The pipeline keeps raw logs for manual inspection.

### Permission denied on device

The tool pushes binary to `/data/local/tmp/benchmark_model` and runs `chmod +x`. If shell restrictions persist, use a debuggable build/userdebug device.
