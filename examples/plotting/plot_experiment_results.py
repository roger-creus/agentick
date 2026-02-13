"""
Plot experiment results using ExperimentPlotter.

This example demonstrates:
- Loading experiment results from multiple runs
- Creating learning curves
- Generating comparison plots
- Saving figures for papers/reports

Requirements:
    uv sync --extra plotting

Usage:
    # After running an experiment:
    uv run python examples/plotting/plot_experiment_results.py --results-dir results/my_experiment/
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from agentick.visualization.experiment_plots import ExperimentPlotter


def plot_single_experiment(results_dir: Path, output_dir: Path):
    """Plot results from a single experiment."""
    print(f"Loading results from: {results_dir}")

    plotter = ExperimentPlotter(results_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Learning curves (reward over time)
    print("Generating learning curves...")
    fig = plotter.plot_learning_curves(
        metrics=["reward", "success_rate"],
        smoothing=10,
        title="Training Progress",
    )
    fig.savefig(output_dir / "learning_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'learning_curves.png'}")

    # 2. Final performance comparison
    print("Generating final performance comparison...")
    fig = plotter.plot_final_performance(
        metric="success_rate",
        title="Final Success Rate by Task",
    )
    fig.savefig(output_dir / "final_performance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'final_performance.png'}")

    # 3. Task difficulty scaling
    print("Generating difficulty scaling plot...")
    fig = plotter.plot_difficulty_scaling(
        tasks=["GoToGoal-v0", "MazeNavigation-v0"],
        metric="reward",
    )
    fig.savefig(output_dir / "difficulty_scaling.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'difficulty_scaling.png'}")

    # 4. Episode length distribution
    print("Generating episode length distribution...")
    fig = plotter.plot_episode_lengths(
        title="Episode Length Distribution",
    )
    fig.savefig(output_dir / "episode_lengths.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'episode_lengths.png'}")

    print(f"\n✓ All plots saved to: {output_dir}")
    print("\nGenerated files:")
    for file in sorted(output_dir.glob("*.png")):
        print(f"  - {file.name}")


def main():
    """Plot experiment results."""
    parser = argparse.ArgumentParser(description="Plot experiment results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/latest"),
        help="Directory containing experiment results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Directory to save generated plots",
    )

    args = parser.parse_args()

    # Check if results directory exists
    if not args.results_dir.exists():
        print(f"Error: Results directory not found: {args.results_dir}")
        print("\nRun an experiment first:")
        print("  uv run python examples/experiments/run_predefined.py")
        print("\nOr specify a different results directory:")
        print("  uv run python examples/plotting/plot_experiment_results.py --results-dir results/my_exp/")
        return

    # Plot results
    plot_single_experiment(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
