"""
Compare two or more experiments.

This example demonstrates:
- Loading experiment results
- Computing comparison metrics
- Displaying side-by-side comparisons

Requirements:
    uv sync --extra all

Usage:
    uv run python examples/experiments/compare_experiments.py exp1_dir exp2_dir
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_experiment_results(results_dir: str) -> dict:
    """Load experiment results from directory."""
    results_path = Path(results_dir)

    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    # Look for results file
    results_file = results_path / "results.json"
    if not results_file.exists():
        # Try to find any JSON file
        json_files = list(results_path.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No results files found in: {results_dir}")
        results_file = json_files[0]

    with open(results_file) as f:
        results = json.load(f)

    return results


def compute_metrics(results: list[dict]) -> dict:
    """Compute summary metrics from results."""
    if not results:
        return {}

    rewards = [r.get('total_reward', 0) for r in results]
    steps = [r.get('steps', 0) for r in results]
    successes = [r.get('success', False) for r in results]

    return {
        'num_episodes': len(results),
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_steps': np.mean(steps),
        'std_steps': np.std(steps),
        'success_rate': np.mean(successes),
    }


def main():
    """Compare experiments."""
    parser = argparse.ArgumentParser(description="Compare experiment results")
    parser.add_argument(
        "experiments",
        nargs="+",
        help="Paths to experiment result directories",
    )
    args = parser.parse_args()

    print("Experiment Comparison")
    print("=" * 80)

    if len(args.experiments) < 2:
        print("❌ Need at least 2 experiments to compare")
        print("Usage: python compare_experiments.py exp1_dir exp2_dir [exp3_dir ...]")
        return

    # Load all experiments
    experiments = []

    for exp_dir in args.experiments:
        try:
            results = load_experiment_results(exp_dir)
            exp_name = Path(exp_dir).name
            experiments.append((exp_name, results))
            print(f"✓ Loaded: {exp_name} ({len(results)} episodes)")
        except Exception as e:
            print(f"❌ Failed to load {exp_dir}: {e}")
            continue

    if len(experiments) < 2:
        print("\n❌ Not enough valid experiments to compare")
        return

    print()

    # Compute metrics for each
    comparison = []

    for name, results in experiments:
        metrics = compute_metrics(results)
        comparison.append((name, metrics))

    # Print comparison table
    print("=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)

    # Header
    print(f"{'Experiment':<25} {'Episodes':>10} {'Reward':>12} {'Steps':>10} {'Success':>10}")
    print("-" * 80)

    # Rows
    for name, metrics in comparison:
        if not metrics:
            continue

        print(f"{name:<25} "
              f"{metrics['num_episodes']:>10} "
              f"{metrics['mean_reward']:>11.2f}± "
              f"{metrics['std_reward']:<4.2f} "
              f"{metrics['mean_steps']:>8.1f}± "
              f"{metrics['std_steps']:<4.1f} "
              f"{metrics['success_rate']:>9.1%}")

    print("=" * 80)

    # Find best performer
    best_idx = max(range(len(comparison)), key=lambda i: comparison[i][1].get('mean_reward', -float('inf')))
    best_name, best_metrics = comparison[best_idx]

    print(f"\n🏆 Best reward: {best_name} ({best_metrics['mean_reward']:.2f})")

    # Most efficient (best success rate)
    best_success_idx = max(range(len(comparison)), key=lambda i: comparison[i][1].get('success_rate', 0))
    success_name, success_metrics = comparison[best_success_idx]

    print(f"✓ Best success rate: {success_name} ({success_metrics['success_rate']:.1%})")

    # Statistical comparison
    if len(comparison) == 2:
        name1, metrics1 = comparison[0]
        name2, metrics2 = comparison[1]

        reward_diff = metrics2['mean_reward'] - metrics1['mean_reward']
        success_diff = metrics2['success_rate'] - metrics1['success_rate']

        print(f"\n{name2} vs {name1}:")
        print(f"  Reward difference: {reward_diff:+.2f}")
        print(f"  Success rate difference: {success_diff:+.1%}")

    print("\n💡 Tip: Use examples/experiments/generate_plots.py to visualize differences")


if __name__ == "__main__":
    main()
