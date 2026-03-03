# har-to-tflite

Convert Human Activity Recognition (HAR) workflows into robust TensorFlow Lite artifacts with host and Android benchmarking, plus auto-fallbacks for offline/limited environments.

## Quickstart

```bash
pip install -e .
h2t run-all --config configs/smoke.yaml
cat results/summary.md
```

## Status

Repository scaffold is ready. Full pipeline features are implemented incrementally in subsequent commits.
