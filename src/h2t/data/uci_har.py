from __future__ import annotations

import logging
import shutil
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from h2t.constants import UCI_HAR_ZIP_URL


class UCIHARDownloadError(RuntimeError):
    """Raised when downloading or extracting UCI HAR fails."""


@dataclass
class UCIHARData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    source: str


def load_uci_har(data_dir: str | Path, force_download: bool = False, logger: logging.Logger | None = None) -> UCIHARData:
    root = Path(data_dir)
    raw_dir = root / "raw"
    cache_zip = raw_dir / "uci_har.zip"
    extracted = raw_dir / "UCI HAR Dataset"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if force_download and extracted.exists():
        shutil.rmtree(extracted, ignore_errors=True)

    if not extracted.exists():
        _download_and_extract(cache_zip, extracted.parent, logger=logger)

    try:
        x_train = _load_matrix(extracted / "train" / "X_train.txt")
        y_train = _load_vector(extracted / "train" / "y_train.txt")
        x_test = _load_matrix(extracted / "test" / "X_test.txt")
        y_test = _load_vector(extracted / "test" / "y_test.txt")
    except OSError as exc:
        raise UCIHARDownloadError(f"UCI HAR cache exists but dataset files are invalid: {exc}") from exc

    # UCI labels are 1..6; normalize to 0..5.
    y_train = y_train - 1
    y_test = y_test - 1

    # Convert flattened 561 features to (timesteps=187, channels=3) synthetic 1D shape.
    x_train = _reshape_features(x_train)
    x_test = _reshape_features(x_test)
    return UCIHARData(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, source="uci_har")


def _download_and_extract(zip_path: Path, extract_root: Path, logger: logging.Logger | None = None) -> None:
    if logger:
        logger.info("Downloading UCI HAR dataset from %s", UCI_HAR_ZIP_URL)
    try:
        with urllib.request.urlopen(UCI_HAR_ZIP_URL, timeout=30) as response:
            payload = response.read()
        zip_path.write_bytes(payload)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_root)
    except (urllib.error.URLError, TimeoutError, zipfile.BadZipFile, OSError) as exc:
        raise UCIHARDownloadError(str(exc)) from exc


def _load_matrix(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=np.float32)


def _load_vector(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=np.int32)


def _reshape_features(matrix: np.ndarray) -> np.ndarray:
    n_samples, n_features = matrix.shape
    channels = 3
    timesteps = n_features // channels
    trimmed = matrix[:, : timesteps * channels]
    return trimmed.reshape(n_samples, timesteps, channels).astype(np.float32)
