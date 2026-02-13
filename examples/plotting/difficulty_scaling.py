"""
Plot performance vs difficulty scaling.

Requirements:
    uv sync --extra all

Usage:
    uv run python examples/plotting/difficulty_scaling.py results.json
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
    """Plot difficulty scaling."""
    if not MATPLOTLIB_AVAILABLE:
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("results", help="Results JSON file")
    parser.add_argument("--output", default="difficulty_scaling.png")
    args = parser.parse_args()

    # Load results
    with open(args.results) as f:
        data = json.load(f)
        results = data.get("results", [])

    # Group by difficulty if available
    difficulty_scores = {"easy": [], "medium": [], "hard": [], "expert": []}

    for r in results:
        task = r["task"]
        # Try to infer difficulty from task name or metadata
        for diff in difficulty_scores:
            if diff in task.lower():
                difficulty_scores[diff].append(r["success_rate"])
                break

    # Plot
    difficulties = []
    scores = []

    for diff in ["easy", "medium", "hard", "expert"]:
        if difficulty_scores[diff]:
            difficulties.append(diff)
            scores.append(np.mean(difficulty_scores[diff]))

    if not difficulties:
        print("⚠️  No difficulty data found")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(difficulties, scores, marker="o", linewidth=2, markersize=10)
    plt.xlabel("Difficulty")
    plt.ylabel("Success Rate")
    plt.title("Performance vs Difficulty")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output)
    print(f"✓ Saved: {args.output}")


if __name__ == "__main__":
    main()
