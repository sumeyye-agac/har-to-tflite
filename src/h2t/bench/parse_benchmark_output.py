from __future__ import annotations

import re
from typing import Any


_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"Inference \\(avg\\):\\s*([0-9]*\\.?[0-9]+)\\s*us", re.IGNORECASE), "us"),
    (re.compile(r"Average inference timings in us:\\s*([0-9]*\\.?[0-9]+)", re.IGNORECASE), "us"),
    (re.compile(r"avg\\s*=\\s*([0-9]*\\.?[0-9]+)\\s*(ms|us)", re.IGNORECASE), "dynamic"),
    (re.compile(r"mean\\s*[:=]\\s*([0-9]*\\.?[0-9]+)\\s*(ms|us)", re.IGNORECASE), "dynamic"),
]


def parse_android_benchmark_output(text: str) -> dict[str, Any]:
    for pattern, unit_hint in _PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        value = float(match.group(1))
        unit = unit_hint
        if unit_hint == "dynamic":
            unit = match.group(2).lower()
        if unit == "us":
            value /= 1000.0
        return {"ok": True, "avg_ms": value}
    return {"ok": False, "avg_ms": None, "reason": "no_supported_pattern"}
