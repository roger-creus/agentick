"""Generate plots from PPO training benchmark results.

Usage:
    # Plot a single run:
    uv run python examples/experiments/full_benchmark/plot_ppo_results.py \
        --results-dir results/ppo_benchmarks/ppo-pixels-dense-300k_TIMESTAMP

    # Compare dense vs sparse:
    uv run python examples/experiments/full_benchmark/plot_ppo_results.py \
        --results-dir results/ppo_benchmarks/ppo-pixels-dense-300k_TIMESTAMP \
        --compare results/ppo_benchmarks/ppo-pixels-sparse-300k_TIMESTAMP
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot PPO training benchmark results")
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Path to training benchmark results directory",
    )
    parser.add_argument(
        "--compare",
        type=str,
        default=None,
        help="Path to a second results directory for comparison",
    )
    args = parser.parse_args()

    from agentick.visualization.training_plots import TrainingBenchmarkPlotter

    # Plot primary results
    print(f"Plotting results from: {args.results_dir}")
    plotter = TrainingBenchmarkPlotter(args.results_dir)
    plotter.plot_all()

    # Comparison
    if args.compare:
        print(f"\nPlotting comparison with: {args.compare}")
        plotter2 = TrainingBenchmarkPlotter(args.compare)
        plotter2.plot_all()

        # Generate simple comparison summary
        _print_comparison(args.results_dir, args.compare)


def _print_comparison(dir1: str, dir2: str) -> None:
    """Print a simple textual comparison of two runs."""
    s1 = _load_summary(dir1)
    s2 = _load_summary(dir2)

    if not s1 or not s2:
        return

    name1 = s1.get("config_name", Path(dir1).name)
    name2 = s2.get("config_name", Path(dir2).name)

    print(f"\n{'='*60}")
    print(f"Comparison: {name1} vs {name2}")
    print(f"{'='*60}")

    r1 = s1.get("results", {})
    r2 = s2.get("results", {})

    # Compute overlap
    common_keys = set(r1.keys()) & set(r2.keys())
    if not common_keys:
        print("No overlapping (task, difficulty) pairs found.")
        return

    sr1 = [r1[k].get("success_rate", 0.0) for k in common_keys]
    sr2 = [r2[k].get("success_rate", 0.0) for k in common_keys]

    import numpy as np

    print(f"  {name1}: mean success rate = {np.mean(sr1):.2%}")
    print(f"  {name2}: mean success rate = {np.mean(sr2):.2%}")

    wins1 = sum(1 for a, b in zip(sr1, sr2) if a > b)
    wins2 = sum(1 for a, b in zip(sr1, sr2) if b > a)
    ties = sum(1 for a, b in zip(sr1, sr2) if a == b)

    print(f"  {name1} wins: {wins1}, {name2} wins: {wins2}, ties: {ties}")
    print(f"  (across {len(common_keys)} task-difficulty pairs)")


def _load_summary(result_dir: str) -> dict | None:
    path = Path(result_dir) / "training_summary.json"
    if not path.exists():
        print(f"  Warning: {path} not found")
        return None
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    main()
