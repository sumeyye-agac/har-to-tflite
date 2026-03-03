from __future__ import annotations

import argparse
from pathlib import Path

from h2t.config import apply_overrides, deep_merge, load_config
from h2t.constants import DEFAULT_CONFIG_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="h2t", description="HAR to TFLite pipeline")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to YAML config")
    parser.add_argument("--set", action="append", default=[], help="Override key=value (dot paths)")

    subparsers = parser.add_subparsers(dest="command")
    for name in ("data", "train", "export", "bench-host", "bench-android", "report", "run-all"):
        subparsers.add_parser(name)
    return parser


def load_runtime_config(config_path: str, overrides: list[str]) -> dict:
    default_cfg = load_config(DEFAULT_CONFIG_PATH)
    user_cfg = load_config(config_path) if Path(config_path).exists() else {}
    merged = deep_merge(default_cfg, user_cfg)
    return apply_overrides(merged, overrides)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0

    config = load_runtime_config(args.config, args.set)
    print(f"Configured command={args.command} seed={config.get('seed')}")
    print("Pipeline stages will be implemented by subsequent commits.")
    return 0
