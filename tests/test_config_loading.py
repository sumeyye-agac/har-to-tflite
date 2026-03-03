from h2t.config import _parse_scalar, apply_overrides, deep_merge


def test_parse_scalar_basic() -> None:
    assert _parse_scalar("true") is True
    assert _parse_scalar("3") == 3


def test_deep_merge_and_overrides() -> None:
    base = {"a": {"b": 1, "c": 2}, "flag": False}
    override = {"a": {"c": 7}}
    merged = deep_merge(base, override)
    assert merged["a"]["b"] == 1
    assert merged["a"]["c"] == 7

    updated = apply_overrides(merged, ["a.b=9", "flag=true"])
    assert updated["a"]["b"] == 9
    assert updated["flag"] is True
