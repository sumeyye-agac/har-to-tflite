from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from h2t.export.representative_data import representative_dataset
from h2t.utils.jsonio import write_json
from h2t.utils.paths import ensure_dir


def export_tflite_variants(
    config: dict[str, Any],
    dataset: dict[str, Any],
    train_result: dict[str, Any],
    logger: logging.Logger,
) -> dict[str, Any]:
    artifacts_dir = ensure_dir(config["paths"]["artifacts_dir"])
    export_dir = ensure_dir(artifacts_dir / "tflite")
    manifest_path = artifacts_dir / "export_manifest.json"

    manifest: dict[str, Any] = {
        "status": "ok",
        "backend": train_result.get("backend", "unknown"),
        "model_source_path": train_result.get("student_model_path", ""),
        "variants": {},
    }
    model_path = Path(train_result.get("student_model_path", ""))
    if train_result.get("backend") != "tensorflow" or not model_path.exists():
        logger.warning("TensorFlow model not available. Writing stub export manifest.")
        for variant in ("fp32", "fp16", "int8", "drq"):
            manifest["variants"][variant] = {
                "status": "skipped",
                "path": "",
                "size_bytes": 0,
                "reason": "tensorflow_model_unavailable",
            }
        write_json(manifest_path, manifest)
        return manifest

    try:
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover
        logger.warning("TensorFlow import failed during export: %s", exc)
        for variant in ("fp32", "fp16", "int8", "drq"):
            manifest["variants"][variant] = {
                "status": "skipped",
                "path": "",
                "size_bytes": 0,
                "reason": "tensorflow_import_failed",
            }
        write_json(manifest_path, manifest)
        return manifest

    model = tf.keras.models.load_model(model_path)
    rep_samples = int(config.get("export", {}).get("representative_samples", 128))
    x_rep = dataset["x_train"]

    fp32_path = export_dir / "model_fp32.tflite"
    _convert_and_write(lambda: _converter_from_model(tf, model), fp32_path, "fp32", manifest, logger)

    fp16_path = export_dir / "model_fp16.tflite"
    _convert_and_write(
        lambda: _converter_fp16(tf, model),
        fp16_path,
        "fp16",
        manifest,
        logger,
    )

    int8_path = export_dir / "model_int8.tflite"
    int8_error = _convert_and_write(
        lambda: _converter_int8(tf, model, x_rep, rep_samples),
        int8_path,
        "int8",
        manifest,
        logger,
    )

    drq_path = export_dir / "model_drq.tflite"
    drq_error = _convert_and_write(lambda: _converter_drq(tf, model), drq_path, "drq", manifest, logger)

    if int8_error is not None:
        manifest["variants"]["int8"]["fallback"] = "drq"
        manifest["variants"]["int8"]["fallback_reason"] = int8_error
        if drq_error is not None:
            manifest["status"] = "partial"
    write_json(manifest_path, manifest)
    logger.info("Saved export manifest to %s", manifest_path)
    return manifest


def _converter_from_model(tf, model):
    return tf.lite.TFLiteConverter.from_keras_model(model)


def _converter_fp16(tf, model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    return converter


def _converter_int8(tf, model, x_rep, rep_samples):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(x_rep, rep_samples)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter


def _converter_drq(tf, model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    return converter


def _convert_and_write(factory, output_path: Path, variant: str, manifest: dict[str, Any], logger: logging.Logger) -> str | None:
    try:
        converter = factory()
        tflite_model = converter.convert()
        output_path.write_bytes(tflite_model)
        manifest["variants"][variant] = {
            "status": "ok",
            "path": str(output_path),
            "size_bytes": output_path.stat().st_size,
            "reason": "",
        }
        logger.info("Exported %s model to %s", variant, output_path)
        return None
    except Exception as exc:  # pragma: no cover - exercised when conversion fails
        manifest["variants"][variant] = {
            "status": "failed",
            "path": str(output_path),
            "size_bytes": 0,
            "reason": str(exc),
        }
        logger.warning("Export failed for %s: %s", variant, exc)
        return str(exc)
