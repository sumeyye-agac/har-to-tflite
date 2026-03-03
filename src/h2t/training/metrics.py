from __future__ import annotations

from typing import Any

import numpy as np


def classification_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    return float((y_true == y_pred).mean())


def history_tail(history: Any) -> dict[str, float]:
    if history is None:
        return {}
    if not hasattr(history, "history"):
        return {}
    result: dict[str, float] = {}
    for key, values in history.history.items():
        if values:
            result[key] = float(values[-1])
    return result
