from h2t.bench.parse_benchmark_output import parse_android_benchmark_output


def test_parse_inference_avg_us_pattern() -> None:
    text = "Inference (avg): 1234.5 us"
    parsed = parse_android_benchmark_output(text)
    assert parsed["ok"] is True
    assert abs(parsed["avg_ms"] - 1.2345) < 1e-6


def test_parse_average_timings_pattern() -> None:
    text = "Average inference timings in us: 950"
    parsed = parse_android_benchmark_output(text)
    assert parsed["ok"] is True
    assert abs(parsed["avg_ms"] - 0.95) < 1e-6


def test_parse_dynamic_ms_pattern() -> None:
    text = "avg = 3.2 ms"
    parsed = parse_android_benchmark_output(text)
    assert parsed["ok"] is True
    assert abs(parsed["avg_ms"] - 3.2) < 1e-6


def test_parse_fails_for_unknown_text() -> None:
    parsed = parse_android_benchmark_output("no timing fields")
    assert parsed["ok"] is False
    assert parsed["avg_ms"] is None
    assert parsed["reason"] == "no_supported_pattern"
