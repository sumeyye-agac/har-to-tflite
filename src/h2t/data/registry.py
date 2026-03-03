from __future__ import annotations

import logging
from typing import Any

import numpy as np

from h2t.data.synthetic import generate_synthetic_har
from h2t.data.uci_har import UCIHARDownloadError, load_uci_har


def load_dataset(config: dict[str, Any], logger: logging.Logger) -> dict[str, Any]:
    data_cfg = config.get("dataset", {})
    paths_cfg = config.get("paths", {})
    name = str(data_cfg.get("name", "uci_har")).lower()
    force_download = bool(data_cfg.get("force_download", False))
    data_dir = paths_cfg.get("data_dir", "data")
    seed = int(config.get("seed", 1337))

    if name == "uci_har":
        try:
            dataset = load_uci_har(data_dir=data_dir, force_download=force_download, logger=logger)
            logger.info("Loaded UCI HAR dataset: train=%s test=%s", dataset.x_train.shape, dataset.x_test.shape)
            return _as_payload(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test, dataset.source)
        except UCIHARDownloadError as exc:
            if not data_cfg.get("synthetic_fallback", True):
                raise
            logger.warning("UCI HAR unavailable (%s). Falling back to synthetic dataset.", exc)

    synth_cfg = data_cfg.get("synthetic_samples", {})
    synthetic = generate_synthetic_har(
        seed=seed,
        train_samples=int(synth_cfg.get("train", 1024)),
        test_samples=int(synth_cfg.get("test", 256)),
    )
    logger.info("Generated synthetic HAR dataset: train=%s test=%s", synthetic.x_train.shape, synthetic.x_test.shape)
    return _as_payload(synthetic.x_train, synthetic.y_train, synthetic.x_test, synthetic.y_test, synthetic.source)


def _as_payload(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, source: str) -> dict[str, Any]:
    classes = int(max(y_train.max(initial=0), y_test.max(initial=0)) + 1)
    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "input_shape": tuple(x_train.shape[1:]),
        "num_classes": classes,
        "source": source,
    }
