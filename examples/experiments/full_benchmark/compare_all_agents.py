#!/usr/bin/env python
"""Generate cross-agent comparison plots from multiple benchmark result directories.

Usage:
    uv run python examples/experiments/full_benchmark/compare_all_agents.py \
        results/llm_benchmarks/run1/ \
        results/api_benchmarks/run2/ \
        results/ppo_benchmarks/run3/

    # Or auto-detect all result dirs under a parent:
    uv run python examples/experiments/full_benchmark/compare_all_agents.py \
        --scan results/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def find_result_dirs(parent: Path) -> list[Path]:
    """Find all result directories that contain a summary.json."""
    dirs = []
    for p in sorted(parent.rglob("summary.json")):
        dirs.append(p.parent)
    return dirs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare results from multiple Agentick benchmark runs."
    )
    parser.add_argument(
        "result_dirs",
        nargs="*",
        type=str,
        help="Paths to individual result directories.",
    )
    parser.add_argument(
        "--scan",
        type=str,
        default=None,
        help="Scan this directory recursively for result dirs.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/comparison",
        help="Output directory for comparison plots.",
    )

    args = parser.parse_args()

    result_dirs: list[Path] = []

    if args.scan:
        scan_path = Path(args.scan)
        if not scan_path.exists():
            print(f"Error: scan path does not exist: {scan_path}")
            sys.exit(1)
        result_dirs = find_result_dirs(scan_path)
        print(f"Found {len(result_dirs)} result directories under {scan_path}")

    for d in args.result_dirs:
        p = Path(d)
        if p.exists():
            result_dirs.append(p)
        else:
            print(f"Warning: skipping non-existent path: {d}")

    if len(result_dirs) < 2:
        print("Error: need at least 2 result directories to compare.")
        print("Usage: compare_all_agents.py dir1/ dir2/ [dir3/ ...]")
        sys.exit(1)

    print(f"Comparing {len(result_dirs)} agent runs:")
    for d in result_dirs:
        print(f"  - {d}")
    print()

    from agentick.visualization.comparison_plots import AgentComparisonPlotter

    plotter = AgentComparisonPlotter(result_dirs, output_dir=Path(args.output))
    plotter.plot_all()

    print(f"\nComparison complete. Figures saved to: {plotter.figures_dir}")


if __name__ == "__main__":
    main()
