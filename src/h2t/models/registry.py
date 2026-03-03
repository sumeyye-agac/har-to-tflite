from __future__ import annotations

from typing import Any, Callable

from h2t.models.teacher_cnn import build_teacher_cnn
from h2t.models.tiny_cnn import build_tiny_cnn


def get_builder(model_name: str) -> Callable[..., Any]:
    name = model_name.lower()
    if name == "tiny_cnn":
        return build_tiny_cnn
    if name == "teacher_cnn":
        return build_teacher_cnn
    raise ValueError(f"Unknown model name: {model_name}")


def build_model(model_name: str, input_shape: tuple[int, ...], num_classes: int, learning_rate: float) -> Any:
    builder = get_builder(model_name)
    return builder(input_shape=input_shape, num_classes=num_classes, learning_rate=learning_rate)
