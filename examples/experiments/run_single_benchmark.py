"""Run a single benchmark experiment from config.

This script loads a YAML configuration file and runs a benchmark experiment,
saving results, videos, and interaction traces. Automatically routes PPO
training configs to TrainingBenchmarkRunner and evaluation configs to
ExperimentRunner.

Usage:
    uv run python examples/experiments/run_single_benchmark.py examples/experiments/configs/random_agent.yaml
    uv run python examples/experiments/run_single_benchmark.py examples/experiments/configs/oracle_agent.yaml

For LLM agents (requires API keys):
    uv run python examples/experiments/run_single_benchmark.py examples/experiments/configs/openai_text.yaml
    uv run python examples/experiments/run_single_benchmark.py examples/experiments/configs/anthropic_text.yaml

For PPO training:
    uv run python examples/experiments/run_single_benchmark.py examples/experiments/configs/ppo_pixels_dense.yaml
"""

import argparse
import sys
from pathlib import Path

from agentick.experiments.config import load_config


def main():
    """Run a single benchmark experiment."""
    parser = argparse.ArgumentParser(
        description="Run a single benchmark experiment from config"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to experiment config YAML file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory (default: results/<experiment_name>)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Override tasks (e.g. GoToGoal-v0 MazeNavigation-v0)",
    )
    parser.add_argument(
        "--difficulties",
        type=str,
        nargs="+",
        default=None,
        help="Override difficulties (e.g. easy medium)",
    )
    args = parser.parse_args()

    # Check config exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        print("\nAvailable configs in examples/experiments/configs/:")
        configs_dir = Path("examples/experiments/configs")
        if configs_dir.exists():
            for cfg in sorted(configs_dir.glob("*.yaml")):
                print(f"  - {cfg.name}")
        sys.exit(1)

    # Load config
    config = load_config(config_path)

    # Apply CLI overrides
    if args.tasks:
        config.tasks = args.tasks
    if args.difficulties:
        config.difficulties = args.difficulties

    print("=" * 80)
    print("BENCHMARK EXPERIMENT")
    print("=" * 80)
    print(f"Config: {config_path}")
    print(f"Name: {config.name}")
    print()

    # Route based on config type
    is_ppo = (
        config.training is not None
        and config.agent.type == "ppo"
    )

    if is_ppo:
        from agentick.experiments.training_runner import TrainingBenchmarkRunner

        runner = TrainingBenchmarkRunner(config)
        runner.run(output_dir=args.output_dir)
    else:
        from agentick.experiments.runner import ExperimentRunner

        runner = ExperimentRunner(config)
        results = runner.run(output_dir=args.output_dir)

        # Print summary
        print("\n" + "=" * 80)
        print("EXPERIMENT COMPLETE")
        print("=" * 80)
        print(f"Results saved to: {results.output_dir}/")


if __name__ == "__main__":
    main()
