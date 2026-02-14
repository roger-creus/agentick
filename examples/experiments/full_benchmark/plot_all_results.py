"""Generate all plots from benchmark results.

This script loads all experiment results from results/full_benchmark/
and generates comprehensive visualizations.

Requirements:
    uv sync --extra all

Usage:
    uv run python examples/experiments/full_benchmark/plot_all_results.py
    uv run python examples/experiments/full_benchmark/plot_all_results.py --results-dir results/full_benchmark
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


def load_all_results(results_dir: Path) -> dict[str, list[dict]]:
    """Load all experiment results from subdirectories."""
    all_results = {}

    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        print("   Run some benchmarks first!")
        return {}

    # Find all experiment subdirectories
    for exp_dir in results_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        # Look for results JSON file
        json_files = list(exp_dir.glob("*_results.json"))
        if not json_files:
            continue

        results_file = json_files[0]
        with open(results_file) as f:
            results = json.load(f)

        exp_name = exp_dir.name
        all_results[exp_name] = results
        print(f"✓ Loaded {exp_name}: {len(results)} episodes")

    return all_results


def plot_comparison_bar_chart(all_results: dict, output_path: Path):
    """Plot bar chart comparing all agents."""
    if not MATPLOTLIB_AVAILABLE:
        return

    # Compute average reward for each agent
    agent_rewards = {}
    for exp_name, results in all_results.items():
        if results:
            avg_reward = np.mean([r.get("total_reward", 0) for r in results])
            agent_rewards[exp_name] = avg_reward

    if not agent_rewards:
        print("⚠️  No results to plot")
        return

    # Sort by reward
    agents = sorted(agent_rewards.keys(), key=lambda x: agent_rewards[x], reverse=True)
    rewards = [agent_rewards[a] for a in agents]

    # Plot
    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(agents)))
    plt.bar(range(len(agents)), rewards, color=colors, edgecolor='black', alpha=0.8)
    plt.xticks(range(len(agents)), agents, rotation=45, ha='right')
    plt.ylabel('Average Reward')
    plt.title('Agent Performance Comparison')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"✓ Saved: {output_path}")


def plot_per_task_comparison(all_results: dict, output_path: Path):
    """Plot per-task performance for all agents."""
    if not MATPLOTLIB_AVAILABLE:
        return

    # Organize by task
    task_data = {}
    for exp_name, results in all_results.items():
        for r in results:
            task = r.get("task", "unknown")
            if task not in task_data:
                task_data[task] = {}
            if exp_name not in task_data[task]:
                task_data[task][exp_name] = []
            task_data[task][exp_name].append(r.get("total_reward", 0))

    if not task_data:
        return

    # Create subplot for each task
    tasks = sorted(task_data.keys())
    n_tasks = len(tasks)
    fig, axes = plt.subplots(1, min(n_tasks, 3), figsize=(15, 5))
    if n_tasks == 1:
        axes = [axes]

    for idx, task in enumerate(tasks[:3]):  # Show first 3 tasks
        ax = axes[idx] if idx < len(axes) else None
        if ax is None:
            continue

        agents = sorted(task_data[task].keys())
        avg_rewards = [np.mean(task_data[task][a]) for a in agents]

        ax.bar(range(len(agents)), avg_rewards, edgecolor='black', alpha=0.7)
        ax.set_xticks(range(len(agents)))
        ax.set_xticklabels(agents, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Avg Reward')
        ax.set_title(task)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"✓ Saved: {output_path}")


def plot_success_rates(all_results: dict, output_path: Path):
    """Plot success rates for all agents."""
    if not MATPLOTLIB_AVAILABLE:
        return

    # Compute success rate for each agent
    agent_success = {}
    for exp_name, results in all_results.items():
        if results:
            success_rate = np.mean([r.get("success", False) for r in results])
            agent_success[exp_name] = success_rate

    if not agent_success:
        return

    # Sort by success rate
    agents = sorted(agent_success.keys(), key=lambda x: agent_success[x], reverse=True)
    rates = [agent_success[a] for a in agents]

    # Plot
    plt.figure(figsize=(12, 6))
    colors = ['green' if r > 0.5 else 'orange' if r > 0.2 else 'red' for r in rates]
    plt.bar(range(len(agents)), rates, color=colors, edgecolor='black', alpha=0.7)
    plt.xticks(range(len(agents)), agents, rotation=45, ha='right')
    plt.ylabel('Success Rate')
    plt.title('Success Rate Comparison')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"✓ Saved: {output_path}")


def main():
    """Generate all plots."""
    parser = argparse.ArgumentParser(description="Generate all benchmark plots")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/full_benchmark",
        help="Directory containing experiment results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/full_benchmark/figures",
        help="Directory to save plots",
    )
    args = parser.parse_args()

    if not MATPLOTLIB_AVAILABLE:
        print("❌ matplotlib not available")
        print("   Install with: uv sync --extra all")
        return

    print("=" * 80)
    print("GENERATING ALL BENCHMARK PLOTS")
    print("=" * 80)
    print()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all results
    print("Loading results...")
    all_results = load_all_results(results_dir)

    if not all_results:
        print("\n❌ No results found in", results_dir)
        print("   Run some benchmarks first:")
        print("   uv run python examples/experiments/full_benchmark/run_single_benchmark.py ...")
        return

    print(f"\nFound {len(all_results)} experiments")
    print()

    # Generate plots
    print("Generating plots...")
    plot_comparison_bar_chart(all_results, output_dir / "comparison_bar.png")
    plot_per_task_comparison(all_results, output_dir / "per_task_comparison.png")
    plot_success_rates(all_results, output_dir / "success_rates.png")

    print()
    print("=" * 80)
    print("PLOTS COMPLETE")
    print("=" * 80)
    print(f"Saved to: {output_dir}/")
    print()
    print("Next step: Generate HTML report")
    print("  uv run python examples/experiments/full_benchmark/generate_report.py")


if __name__ == "__main__":
    main()
