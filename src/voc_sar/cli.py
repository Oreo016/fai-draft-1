"""Command-line entry point for reproducible SAR experiments."""

import argparse
from pathlib import Path
from typing import Any

import yaml

from voc_sar.experiments.runner import _validate_config, run_experiments


def load_config(path: str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Config must contain a YAML mapping at its root")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run receiver-aware value-of-communication SAR experiments"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate configuration without running episodes",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    _validate_config(config)
    if args.validate_only:
        roles = ", ".join(config["environment"].get("roles", ()))
        print(f"Configuration valid: {args.config}")
        print(f"Roles: {roles}")
        return
    summary = run_experiments(config)
    print(f"Completed {summary['episode_count']} episodes.")
    print(f"Results: {summary['run_directory']}")
    print(f"Graphs: {summary['visualizations_directory']}\\index.html")
    print(f"Runtime: {summary['runtime_seconds']:.2f} seconds")


if __name__ == "__main__":
    main()
