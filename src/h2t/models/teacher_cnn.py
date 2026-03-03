from __future__ import annotations

from typing import Any


def build_teacher_cnn(input_shape: tuple[int, ...], num_classes: int, learning_rate: float = 1e-3) -> Any:
    try:
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover - exercised in TF-less environment
        raise RuntimeError("TensorFlow is not installed") from exc

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv1D(32, kernel_size=5, padding="same", activation="relu"),
            tf.keras.layers.Conv1D(48, kernel_size=3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(64, kernel_size=3, padding="same", activation="relu"),
            tf.keras.layers.Conv1D(64, kernel_size=3, padding="same", activation="relu"),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
