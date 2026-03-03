from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from h2t.reporting.leaderboard import update_leaderboard
from h2t.utils.jsonio import read_json


def write_summary(config: dict[str, Any]) -> tuple[Path, Path]:
    artifacts_dir = Path(config["paths"]["artifacts_dir"])
    results_dir = Path(config["paths"]["results_dir"])
    summary_path = results_dir / "summary.md"
    leaderboard_path = results_dir / "leaderboard.csv"

    data_summary = _read_json_if_exists(artifacts_dir / "data_summary.json")
    train_metrics = _read_json_if_exists(artifacts_dir / "train_metrics.json")
    export_manifest = _read_json_if_exists(artifacts_dir / "export_manifest.json")
    host_rows = _read_csv_if_exists(results_dir / "bench_host.csv")
    android_rows = _read_csv_if_exists(results_dir / "bench_android.csv")

    lines = [
        "# har-to-tflite summary",
        "",
        "## Dataset",
        f"- source: {data_summary.get('source', 'unknown')}",
        f"- train shape: {data_summary.get('x_train_shape', [])}",
        f"- test shape: {data_summary.get('x_test_shape', [])}",
        "",
        "## Training",
        f"- backend: {train_metrics.get('backend', 'unknown')}",
        f"- eval_accuracy: {train_metrics.get('eval_accuracy', 0.0)}",
        f"- model: {train_metrics.get('student_model_path', '')}",
        "",
        "## Export",
    ]
    variants = export_manifest.get("variants", {})
    if variants:
        for name, info in variants.items():
            lines.append(
                f"- {name}: status={info.get('status', 'missing')} size={info.get('size_bytes', 0)} reason={info.get('reason', '')}"
            )
    else:
        lines.append("- no export manifest found")

    lines.extend(["", "## Host Benchmark"])
    if host_rows:
        for row in host_rows:
            lines.append(
                f"- {row.get('variant')}: status={row.get('status')} mean_ms={row.get('mean_ms')} reason={row.get('reason','')}"
            )
    else:
        lines.append("- no host benchmark output found")

    lines.extend(["", "## Android Benchmark"])
    if android_rows:
        for row in android_rows:
            lines.append(
                f"- {row.get('variant')}: status={row.get('status')} mean_ms={row.get('mean_ms')} reason={row.get('reason','')}"
            )
    else:
        lines.append("- no android benchmark output found")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    update_leaderboard(leaderboard_path, train_metrics, export_manifest, host_rows, android_rows)
    return summary_path, leaderboard_path


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return read_json(path)


def _read_csv_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
