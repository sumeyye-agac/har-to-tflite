from __future__ import annotations

import json
from pathlib import Path

from h2t.bench.android import benchmark_android


def test_license_file_exists() -> None:
    assert Path("LICENSE").exists()


def test_android_device_info_written_when_adb_missing(monkeypatch, tmp_path: Path) -> None:
    import h2t.bench.android as android_mod

    monkeypatch.setattr(android_mod.shutil, "which", lambda _: None)

    cfg = {
        "paths": {
            "results_dir": str(tmp_path / "results"),
            "artifacts_dir": str(tmp_path / "artifacts"),
            "data_dir": str(tmp_path / "data"),
        },
        "bench": {"android": {"threads": 1, "repeat": 1, "cooldown_s": 0.0, "warmup_runs": 1, "num_runs": 1}},
    }

    bench_path = benchmark_android(cfg, manifest={"variants": {}}, logger=_dummy_logger())
    info_path = tmp_path / "results" / "android_device_info.json"

    assert bench_path.exists()
    assert info_path.exists()

    payload = json.loads(info_path.read_text(encoding="utf-8"))
    assert payload["status"] == "stub"
    assert payload["reason"] == "adb_missing"

    csv_text = bench_path.read_text(encoding="utf-8")
    assert "device_info_file" in csv_text
    assert "android_device_info.json" in csv_text


def _dummy_logger():
    import logging

    return logging.getLogger("test-android-stub")
