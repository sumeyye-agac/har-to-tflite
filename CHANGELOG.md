# Changelog

All notable changes to this repository are tracked here.

## Unreleased

### Planned
- Improve model distillation path (teacher -> student)
- Extend Android benchmark parser coverage
- Add richer per-run benchmark metadata

## 2026-03-04

### Added
- End-to-end `h2t` CLI with `data`, `train`, `export`, `bench`, `report`, `run-all`
- UCI HAR dataset downloader with synthetic fallback
- Tiny and teacher CNN model builders
- Deterministic training pipeline with artifact outputs
- TFLite export variants (`fp32`, `fp16`, `int8`, `drq`) with fallback manifest
- Host benchmark CSV output
- Android benchmark integration with graceful fallback behavior
- Summary and leaderboard reporting
- Offline-capable CI tests

### Changed
- Improved README with quickstart and troubleshooting
- Added reproducibility environment snapshot output (`results/env.txt`)

### Removed
- Repository license file and license metadata references
