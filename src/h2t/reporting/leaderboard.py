from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def update_leaderboard(
    path: Path,
    train_metrics: dict[str, Any],
    export_manifest: dict[str, Any],
    host_rows: list[dict[str, Any]],
    android_rows: list[dict[str, Any]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    best_host = _best_latency(host_rows)
    best_android = _best_latency(android_rows)
    int8_status = export_manifest.get("variants", {}).get("int8", {}).get("status", "missing")

    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "backend": train_metrics.get("backend", "unknown"),
        "eval_accuracy": train_metrics.get("eval_accuracy", 0.0),
        "int8_status": int8_status,
        "best_host_ms": best_host if best_host is not None else "",
        "best_android_ms": best_android if best_android is not None else "",
    }

    file_exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["timestamp_utc", "backend", "eval_accuracy", "int8_status", "best_host_ms", "best_android_ms"],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    return path


def _best_latency(rows: list[dict[str, Any]]) -> float | None:
    values = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        try:
            values.append(float(row.get("mean_ms", 0.0)))
        except (TypeError, ValueError):
            continue
    return min(values) if values else None
