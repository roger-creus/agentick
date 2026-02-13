"""
Create bar chart comparisons of agent performance.

Requirements:
    uv sync --extra all

Usage:
    uv run python examples/plotting/bar_comparison.py results1.json results2.json
"""

import argparse
import json

try:
    import matplotlib.pyplot as plt
    import numpy as np

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib not available. Install with: uv sync --extra all")


def main():
    """Create bar comparison plot."""
    if not MATPLOTLIB_AVAILABLE:
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", help="Result JSON files")
    parser.add_argument("--output", default="bar_comparison.png")
    args = parser.parse_args()

    # Load results
    agents = []
    for path in args.results:
        with open(path) as f:
            data = json.load(f)
            name = data.get("submission", path)
            results = data.get("results", [])
            rewards = [r["mean_reward"] for r in results]
            agents.append((name, np.mean(rewards) if rewards else 0))

    # Plot
    names, scores = zip(*agents)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(names)), scores)
    plt.xticks(range(len(names)), names, rotation=45, ha="right")
    plt.ylabel("Average Reward")
    plt.title("Agent Performance Comparison")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(args.output)
    print(f"✓ Saved: {args.output}")


if __name__ == "__main__":
    main()
