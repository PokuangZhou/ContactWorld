from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def load_yaml_config(path: str | None) -> dict:
    if path is None:
        return {}

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config at {config_path} must be a YAML mapping, got {type(data)}")

    return data


def build_config_aware_parser(builder):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file.")
    pre_args, _ = pre_parser.parse_known_args()

    parser = builder(pre_parser)
    config = load_yaml_config(pre_args.config)
    if config:
        valid_keys = {action.dest for action in parser._actions}
        unknown_keys = sorted(set(config) - valid_keys)
        if unknown_keys:
            raise ValueError(f"Unknown config keys in {pre_args.config}: {unknown_keys}")
        parser.set_defaults(**config)

    return parser
