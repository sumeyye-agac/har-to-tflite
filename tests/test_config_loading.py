from h2t.config import _parse_scalar


def test_parse_scalar_basic() -> None:
    assert _parse_scalar("true") is True
    assert _parse_scalar("3") == 3
