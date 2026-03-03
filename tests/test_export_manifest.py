from __future__ import annotations

import logging
from pathlib import Path

from h2t.export.tflite_export import export_tflite_variants


def test_export_manifest_stub_when_no_tf_model(tmp_path: Path) -> None:
    cfg = {
        "paths": {"artifacts_dir": tmp_path / "artifacts", "results_dir": tmp_path / "results"},
        "export": {"representative_samples": 8},
    }
    dataset = {"x_train": [], "source": "synthetic"}
    train_result = {"backend": "numpy_stub", "student_model_path": str(tmp_path / "missing.npz")}
    logger = logging.getLogger("test")

    manifest = export_tflite_variants(cfg, dataset, train_result, logger)
    assert manifest["backend"] == "numpy_stub"
    assert manifest["variants"]["fp32"]["status"] == "skipped"
    assert (tmp_path / "artifacts" / "export_manifest.json").exists()
