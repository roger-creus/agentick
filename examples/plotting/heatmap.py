"""
Create task × agent performance heatmap.

Requirements:
    uv sync --extra all

Usage:
    uv run python examples/plotting/heatmap.py results1.json results2.json
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
    """Create heatmap."""
    if not MATPLOTLIB_AVAILABLE:
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", help="Result JSON files")
    parser.add_argument("--output", default="heatmap.png")
    args = parser.parse_args()

    # Load all results
    agents = []
    all_tasks = set()

    for path in args.results:
        with open(path) as f:
            data = json.load(f)
            name = data.get("submission", path)
            results = data.get("results", [])

            task_scores = {r["task"]: r["success_rate"] for r in results}
            agents.append((name, task_scores))
            all_tasks.update(task_scores.keys())

    tasks = sorted(all_tasks)

    # Build matrix
    matrix = []
    for name, task_scores in agents:
        row = [task_scores.get(task, 0) for task in tasks]
        matrix.append(row)

    matrix = np.array(matrix)

    # Plot
    fig, ax = plt.subplots(figsize=(max(12, len(tasks) * 0.5), len(agents) * 0.8))

    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([t.replace("-v0", "") for t in tasks], rotation=45, ha="right")

    ax.set_yticks(range(len(agents)))
    ax.set_yticklabels([name for name, _ in agents])

    plt.colorbar(im, label="Success Rate")
    plt.title("Agent × Task Performance Heatmap")
    plt.tight_layout()
    plt.savefig(args.output)
    print(f"✓ Saved: {args.output}")


if __name__ == "__main__":
    main()
