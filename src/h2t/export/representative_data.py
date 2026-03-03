from __future__ import annotations

import numpy as np


def representative_dataset(x_data: np.ndarray, num_samples: int):
    sample_count = min(int(num_samples), x_data.shape[0])
    for idx in range(sample_count):
        sample = x_data[idx : idx + 1].astype(np.float32)
        yield [sample]
