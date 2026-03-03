from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from h2t.models.registry import build_model
from h2t.training.metrics import classification_accuracy, history_tail
from h2t.utils.jsonio import write_json
from h2t.utils.paths import ensure_dir
from h2t.utils.reproducibility import set_seed, set_tensorflow_determinism


def train_pipeline(config: dict[str, Any], dataset: dict[str, Any], logger: logging.Logger) -> dict[str, Any]:
    seed = int(config.get("seed", 1337))
    set_seed(seed)
    tf_ready = set_tensorflow_determinism(seed)

    paths = config["paths"]
    artifacts_dir = ensure_dir(paths["artifacts_dir"])
    models_dir = ensure_dir(artifacts_dir / "models")

    started = perf_counter()
    if tf_ready:
        result = _train_tensorflow(config, dataset, models_dir, logger)
    else:
        logger.warning("TensorFlow unavailable; using deterministic centroid-based stub model.")
        result = _train_stub(dataset, models_dir)
    result["duration_s"] = round(perf_counter() - started, 4)

    metrics_path = artifacts_dir / "train_metrics.json"
    write_json(metrics_path, result)
    logger.info("Saved training metrics to %s", metrics_path)
    return result


def _train_tensorflow(config: dict[str, Any], dataset: dict[str, Any], models_dir: Path, logger: logging.Logger) -> dict[str, Any]:
    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    model_name = model_cfg.get("name", "tiny_cnn")
    learning_rate = float(model_cfg.get("learning_rate", 1e-3))

    model = build_model(
        model_name=model_name,
        input_shape=tuple(dataset["input_shape"]),
        num_classes=int(dataset["num_classes"]),
        learning_rate=learning_rate,
    )
    history = model.fit(
        dataset["x_train"],
        dataset["y_train"],
        validation_split=float(train_cfg.get("validation_split", 0.1)),
        epochs=int(train_cfg.get("epochs", 3)),
        batch_size=int(train_cfg.get("batch_size", 64)),
        verbose=0,
        shuffle=False,
    )
    eval_loss, eval_acc = model.evaluate(dataset["x_test"], dataset["y_test"], verbose=0)
    student_path = models_dir / "student.keras"
    model.save(student_path)

    teacher_path = None
    if bool(train_cfg.get("train_teacher", False)):
        try:
            teacher = build_model(
                model_name=model_cfg.get("teacher_name", "teacher_cnn"),
                input_shape=tuple(dataset["input_shape"]),
                num_classes=int(dataset["num_classes"]),
                learning_rate=learning_rate,
            )
            teacher.fit(
                dataset["x_train"],
                dataset["y_train"],
                validation_split=float(train_cfg.get("validation_split", 0.1)),
                epochs=max(1, int(train_cfg.get("epochs", 3))),
                batch_size=int(train_cfg.get("batch_size", 64)),
                verbose=0,
                shuffle=False,
            )
            teacher_path = models_dir / "teacher.keras"
            teacher.save(teacher_path)
        except Exception as exc:
            logger.warning("Teacher training failed; continuing without teacher artifact: %s", exc)

    result = {
        "status": "ok",
        "backend": "tensorflow",
        "model_name": model_name,
        "train_source": dataset["source"],
        "student_model_path": str(student_path),
        "teacher_model_path": str(teacher_path) if teacher_path else "",
        "eval_loss": float(eval_loss),
        "eval_accuracy": float(eval_acc),
        "history_tail": history_tail(history),
    }
    return result


def _train_stub(dataset: dict[str, Any], models_dir: Path) -> dict[str, Any]:
    x_train = dataset["x_train"]
    y_train = dataset["y_train"]
    x_test = dataset["x_test"]
    y_test = dataset["y_test"]
    num_classes = int(dataset["num_classes"])
    centroids = []
    for label in range(num_classes):
        class_data = x_train[y_train == label]
        if class_data.size == 0:
            centroids.append(np.zeros(dataset["input_shape"], dtype=np.float32))
        else:
            centroids.append(class_data.mean(axis=0))
    centroid_arr = np.stack(centroids).astype(np.float32)

    flat_test = x_test.reshape(x_test.shape[0], -1)
    flat_centroids = centroid_arr.reshape(num_classes, -1)
    distances = ((flat_test[:, None, :] - flat_centroids[None, :, :]) ** 2).sum(axis=2)
    preds = distances.argmin(axis=1).astype(np.int32)
    acc = classification_accuracy(y_test, preds)

    stub_path = models_dir / "student_stub.npz"
    np.savez_compressed(stub_path, centroids=centroid_arr, input_shape=np.asarray(dataset["input_shape"]))
    return {
        "status": "ok",
        "backend": "numpy_stub",
        "model_name": "centroid_classifier",
        "train_source": dataset["source"],
        "student_model_path": str(stub_path),
        "teacher_model_path": "",
        "eval_loss": 0.0,
        "eval_accuracy": acc,
        "history_tail": {"accuracy": acc},
    }
