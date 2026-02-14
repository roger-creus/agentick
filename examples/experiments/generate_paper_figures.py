"""
Generate publication-ready figures from experiment results.

This example demonstrates:
- Creating high-quality publication figures
- Using LaTeX-style formatting
- Exporting to PDF and PNG

Requirements:
    uv sync --extra all

Usage:
    uv run python examples/experiments/generate_paper_figures.py results_dir
"""

import argparse
import json
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True

    # Configure matplotlib for publication quality
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'text.usetex': False,  # Set to True if LaTeX is installed
        'figure.figsize': (8, 6),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'lines.linewidth': 2,
        'lines.markersize': 8,
    })

except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib not available. Install with: uv sync --extra all")


def load_experiment_results(results_dir: str) -> list[dict]:
    """Load experiment results from directory."""
    results_path = Path(results_dir)

    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    results_file = results_path / "results.json"
    if not results_file.exists():
        json_files = list(results_path.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No results files found in: {results_dir}")
        results_file = json_files[0]

    with open(results_file) as f:
        results = json.load(f)

    return results


def create_performance_comparison(results: list[dict], output_path: str):
    """Create publication-quality performance comparison figure."""
    # Group by task
    task_results = {}
    for result in results:
        task = result.get('task', 'unknown')
        if task not in task_results:
            task_results[task] = []
        task_results[task].append(result)

    # Compute metrics
    tasks = []
    rewards = []
    reward_stds = []
    success_rates = []

    for task, task_res in sorted(task_results.items()):
        task_rewards = [r.get('total_reward', 0) for r in task_res]
        task_successes = [r.get('success', False) for r in task_res]

        tasks.append(task.replace('-v0', ''))
        rewards.append(np.mean(task_rewards))
        reward_stds.append(np.std(task_rewards))
        success_rates.append(np.mean(task_successes))

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot 1: Rewards
    x = np.arange(len(tasks))
    ax1.bar(x, rewards, yerr=reward_stds, capsize=5, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Performance Across Tasks')
    ax1.set_xticks(x)
    ax1.set_xticklabels(tasks, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Success rates
    colors = ['green' if sr > 0.5 else 'red' for sr in success_rates]
    ax2.bar(x, success_rates, alpha=0.7, edgecolor='black', color=colors)
    ax2.set_ylabel('Success Rate')
    ax2.set_xlabel('Task')
    ax2.set_xticks(x)
    ax2.set_xticklabels(tasks, rotation=45, ha='right')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), format='png', bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {output_path}")


def create_capability_analysis(results: list[dict], output_path: str):
    """Create capability analysis figure."""
    # Define capability categories
    categories = {
        'Navigation': ['GoToGoal', 'MazeNavigation', 'MultiRoomEscape'],
        'Reasoning': ['ProgramSynthesis', 'RuleInduction', 'GraphColoring'],
        'Memory': ['SequenceMemory', 'DelayedGratification', 'BacktrackPuzzle'],
        'Coordination': ['CooperativeTransport', 'CompetitiveTag', 'Herding'],
    }

    # Compute performance by category
    category_scores = {}

    for category, task_list in categories.items():
        category_results = []

        for result in results:
            task = result.get('task', '')
            if any(t in task for t in task_list):
                category_results.append(result)

        if category_results:
            rewards = [r.get('total_reward', 0) for r in category_results]
            successes = [r.get('success', False) for r in category_results]

            category_scores[category] = {
                'reward': np.mean(rewards),
                'success': np.mean(successes),
            }

    if not category_scores:
        print("⚠️  Not enough task coverage for capability analysis")
        return

    # Create radar chart
    categories_list = list(category_scores.keys())
    success_rates = [category_scores[c]['success'] for c in categories_list]

    angles = np.linspace(0, 2 * np.pi, len(categories_list), endpoint=False).tolist()
    success_rates += success_rates[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    ax.plot(angles, success_rates, 'o-', linewidth=2)
    ax.fill(angles, success_rates, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories_list)
    ax.set_ylim(0, 1)
    ax.set_title('Capability Profile', pad=20, fontsize=18)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), format='png', bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {output_path}")


def main():
    """Generate publication-ready figures."""
    if not MATPLOTLIB_AVAILABLE:
        return

    parser = argparse.ArgumentParser(description="Generate publication-ready figures")
    parser.add_argument(
        "results_dir",
        type=str,
        help="Path to experiment results directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save figures (defaults to results_dir/figures)",
    )
    args = parser.parse_args()

    print("Generating Publication Figures")
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
        output_dir = Path(args.results_dir) / "figures"
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving figures to: {output_dir}")
    print()

    # Generate figures
    create_performance_comparison(results, str(output_dir / "performance_comparison.pdf"))
    create_capability_analysis(results, str(output_dir / "capability_analysis.pdf"))

    print("\n" + "=" * 80)
    print("Publication figures generated!")
    print(f"View figures in: {output_dir}")
    print("\n💡 Figures are saved as both PDF (vector) and PNG (raster)")


if __name__ == "__main__":
    main()
