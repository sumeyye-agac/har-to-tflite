from __future__ import annotations

from pathlib import Path

from h2t.cli import main


def test_smoke_pipeline_runs_offline(tmp_path: Path) -> None:
    cfg_path = tmp_path / "smoke_test.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "seed: 1",
                "paths:",
                f"  data_dir: {tmp_path / 'data'}",
                f"  artifacts_dir: {tmp_path / 'artifacts'}",
                f"  results_dir: {tmp_path / 'results'}",
                "dataset:",
                "  name: synthetic",
                "  synthetic_fallback: true",
                "  synthetic_samples:",
                "    train: 128",
                "    test: 32",
                "training:",
                "  epochs: 1",
                "  batch_size: 16",
                "bench:",
                "  host:",
                "    enabled: true",
                "    warmup_runs: 1",
                "    num_runs: 2",
                "  android:",
                "    enabled: true",
                "    repeat: 1",
                "    cooldown_s: 0.0",
                "    warmup_runs: 1",
                "    num_runs: 1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    rc = main(["run-all", "--config", str(cfg_path)])
    assert rc == 0
    assert (tmp_path / "results" / "summary.md").exists()
    assert (tmp_path / "results" / "leaderboard.csv").exists()
