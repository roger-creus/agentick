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
"""

import argparse
from pathlib import Path

from agentick.leaderboard.experiment import ExperimentRunner


def main():
    """Run predefined experiment."""
    parser = argparse.ArgumentParser(description="Run predefined experiment config")
    parser.add_argument(
        "--config",
        type=str,
        default="examples/experiments/configs/quick/sanity_check.yaml",
        help="Path to experiment config file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results",
    )
    args = parser.parse_args()

    print("Running Predefined Experiment")
    print("=" * 80)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        print("\nAvailable configs:")
        print("  - examples/experiments/configs/quick/sanity_check.yaml (fast, 2 min)")
        print("  - examples/experiments/full_benchmark/configs/random_agent.yaml")
        print("  - examples/experiments/full_benchmark/configs/oracle_agent.yaml")
        return

    print(f"\nConfig: {config_path}")
    print(f"Output: {args.output_dir}")
    print()

    # Create runner
    runner = ExperimentRunner(
        config_path=str(config_path),
        output_dir=args.output_dir,
    )

    # Run experiment
    print("Starting experiment...")
    results = runner.run()

    # Print summary
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"Total episodes: {len(results)}")
    print(f"Results saved to: {args.output_dir}")
    print()

    # Compute basic statistics
    if results:
        avg_reward = sum(r.get('total_reward', 0) for r in results) / len(results)
        avg_steps = sum(r.get('steps', 0) for r in results) / len(results)
        success_rate = sum(1 for r in results if r.get('success', False)) / len(results)

        print(f"Average reward: {avg_reward:.2f}")
        print(f"Average steps: {avg_steps:.1f}")
        print(f"Success rate: {success_rate:.1%}")

    print("\n💡 Tip: Use examples/experiments/generate_plots.py to visualize results")


if __name__ == "__main__":
    main()
