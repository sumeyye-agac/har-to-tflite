# Obtain `benchmark_model` Binary

`benchmark_model` is part of TensorFlow Lite tools and is not bundled in this repository.

## Option 1: Build from TensorFlow source (recommended)

1. Clone TensorFlow source matching your target ABI/toolchain.
2. Build `benchmark_model` with Bazel for your Android ABI.
3. Copy resulting binary to a local path, for example:
   - `/tmp/benchmark_model_arm64`

Then run:

```bash
h2t bench android --benchmark-bin /tmp/benchmark_model_arm64 --config configs/default.yaml
```

## Option 2: Use prebuilt/internal binary

If your org keeps vetted prebuilts, point `--benchmark-bin` to that local file.

## Verify binary before running

```bash
file /path/to/benchmark_model
chmod +x /path/to/benchmark_model
```

## Notes

- Keep binaries out of git history.
- Match ABI to device architecture (for example arm64-v8a).
- This repository writes stub Android benchmark output instead of failing when binary is absent.
