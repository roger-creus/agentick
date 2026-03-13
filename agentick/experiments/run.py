"""CLI entry point for running experiments from YAML configs.

Usage:
    uv run python -m agentick.experiments.run --config path/to/config.yaml
    uv run python -m agentick.experiments.run --config config.yaml --resume results/prev_run/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agentick.experiments.config import load_config
from agentick.experiments.runner import ExperimentRunner


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run an Agentick experiment from a YAML config.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML config file.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to previous run directory to resume from.",
    )
    parser.add_argument(
        "--n-parallel",
        type=int,
        default=1,
        help="Number of tasks to run in parallel (default: 1).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory (all results go here).",
    )
    parser.add_argument(
        "--difficulties",
        nargs="+",
        default=None,
        help="Override difficulties to run (e.g. easy expert).",
    )

    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)

    if args.difficulties:
        config.difficulties = args.difficulties

    runner = ExperimentRunner(config)
    runner.run(resume_from=args.resume, n_parallel=args.n_parallel, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
