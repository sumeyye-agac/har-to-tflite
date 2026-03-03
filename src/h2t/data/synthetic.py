from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SyntheticHARData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    source: str


def generate_synthetic_har(
    seed: int,
    train_samples: int = 1024,
    test_samples: int = 256,
    timesteps: int = 128,
    channels: int = 9,
    num_classes: int = 6,
) -> SyntheticHARData:
    rng = np.random.default_rng(seed)
    centroids = rng.normal(0, 1, size=(num_classes, timesteps, channels)).astype(np.float32)

    y_train = rng.integers(0, num_classes, size=(train_samples,), endpoint=False)
    y_test = rng.integers(0, num_classes, size=(test_samples,), endpoint=False)

    x_train = centroids[y_train] + rng.normal(0, 0.35, size=(train_samples, timesteps, channels)).astype(np.float32)
    x_test = centroids[y_test] + rng.normal(0, 0.35, size=(test_samples, timesteps, channels)).astype(np.float32)
    return SyntheticHARData(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, source="synthetic")
