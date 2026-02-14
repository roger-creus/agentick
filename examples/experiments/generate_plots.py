"""
Generate plots from experiment results.

This example demonstrates:
- Loading experiment results
- Creating visualizations
- Saving plots to files

Requirements:
    uv sync --extra all

Usage:
    uv run python examples/experiments/generate_plots.py results_dir
"""

import argparse
import json
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib not available. Install with: uv sync --extra all")


def load_experiment_results(results_dir: str) -> list[dict]:
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


def plot_reward_distribution(results: list[dict], output_path: str):
    """Plot reward distribution histogram."""
    rewards = [r.get('total_reward', 0) for r in results]

    plt.figure(figsize=(10, 6))
    plt.hist(rewards, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('Total Reward')
    plt.ylabel('Frequency')
    plt.title('Reward Distribution')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"✓ Saved: {output_path}")


def plot_success_by_task(results: list[dict], output_path: str):
    """Plot success rate by task."""
    # Group by task
    task_results = {}
    for result in results:
        task = result.get('task', 'unknown')
        if task not in task_results:
            task_results[task] = []
        task_results[task].append(result.get('success', False))

    # Compute success rates
    tasks = list(task_results.keys())
    success_rates = [np.mean(task_results[task]) for task in tasks]

    # Sort by success rate
    sorted_indices = np.argsort(success_rates)[::-1]
    tasks = [tasks[i] for i in sorted_indices]
    success_rates = [success_rates[i] for i in sorted_indices]

    # Plot
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(tasks)), success_rates, edgecolor='black', alpha=0.7)
    plt.xticks(range(len(tasks)), tasks, rotation=45, ha='right')
    plt.ylabel('Success Rate')
    plt.title('Success Rate by Task')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"✓ Saved: {output_path}")


def plot_steps_vs_reward(results: list[dict], output_path: str):
    """Plot steps vs reward scatter."""
    steps = [r.get('steps', 0) for r in results]
    rewards = [r.get('total_reward', 0) for r in results]
    successes = [r.get('success', False) for r in results]

    plt.figure(figsize=(10, 6))

    # Color by success
    colors = ['green' if s else 'red' for s in successes]
    plt.scatter(steps, rewards, c=colors, alpha=0.6, edgecolors='black')

    plt.xlabel('Steps')
    plt.ylabel('Total Reward')
    plt.title('Steps vs Reward (Green=Success, Red=Failure)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"✓ Saved: {output_path}")


def plot_learning_curve(results: list[dict], output_path: str):
    """Plot learning curve (if results are sequential)."""
    # Assume results are in order
    rewards = [r.get('total_reward', 0) for r in results]

    # Compute rolling average
    window = min(10, len(rewards) // 10 + 1)
    rolling_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')

    plt.figure(figsize=(12, 6))
    plt.plot(rewards, alpha=0.3, label='Raw')
    plt.plot(range(window - 1, len(rewards)), rolling_avg, linewidth=2, label=f'Rolling Avg ({window})')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"✓ Saved: {output_path}")


def main():
    """Generate plots from experiment results."""
    if not MATPLOTLIB_AVAILABLE:
        return

    parser = argparse.ArgumentParser(description="Generate plots from experiment results")
    parser.add_argument(
        "results_dir",
        type=str,
        help="Path to experiment results directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plots (defaults to results_dir/plots)",
    )
    args = parser.parse_args()

    print("Generating Experiment Plots")
    print("=" * 80)

    # Load results
    try:
        results = load_experiment_results(args.results_dir)
        print(f"✓ Loaded {len(results)} results from {args.results_dir}")
    except Exception as e:
        print(f"❌ Failed to load results: {e}")
        return

    # Setup output directory
    if args.output_dir is None:
        output_dir = Path(args.results_dir) / "plots"
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving plots to: {output_dir}")
    print()

    # Generate plots
    plot_reward_distribution(results, str(output_dir / "reward_distribution.png"))
    plot_success_by_task(results, str(output_dir / "success_by_task.png"))
    plot_steps_vs_reward(results, str(output_dir / "steps_vs_reward.png"))
    plot_learning_curve(results, str(output_dir / "learning_curve.png"))

    print("\n" + "=" * 80)
    print("All plots generated!")
    print(f"View plots in: {output_dir}")


if __name__ == "__main__":
    main()
