"""
Run a predefined experiment configuration.

This example demonstrates:
- Loading experiment configs from YAML files
- Running experiments programmatically
- Saving results

Requirements:
    uv sync --extra all

Usage:
    uv run python examples/experiments/run_predefined.py
    uv run python examples/experiments/run_predefined.py --config examples/experiments/configs/oracle_agent.yaml
"""

import argparse
from pathlib import Path

from agentick.experiments.config import load_config
from agentick.experiments.runner import ExperimentRunner


def main():
    """Run predefined experiment."""
    parser = argparse.ArgumentParser(description="Run predefined experiment config")
    parser.add_argument(
        "--config",
        type=str,
        default="examples/experiments/configs/random_agent.yaml",
        help="Path to experiment config file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results",
    )
    args = parser.parse_args()

    print("Running Predefined Experiment")
    print("=" * 80)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        print("\nAvailable configs:")
        print("  - examples/experiments/configs/random_agent.yaml")
        print("  - examples/experiments/configs/oracle_agent.yaml")
        print("  - examples/experiments/configs/openai_text.yaml")
        return

    print(f"\nConfig: {config_path}")
    print()

    # Load and run
    config = load_config(config_path)
    runner = ExperimentRunner(config)
    results = runner.run(output_dir=args.output_dir)

    # Print summary
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {results.output_dir}")


if __name__ == "__main__":
    main()
