"""
Generate LaTeX tables from results.

Requirements:
    uv sync

Usage:
    uv run python examples/plotting/latex_tables.py results.json
"""

import argparse
import json


def main():
    """Generate LaTeX table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("results", help="Results JSON file")
    parser.add_argument("--output", default="table.tex")
    args = parser.parse_args()

    # Load results
    with open(args.results) as f:
        data = json.load(f)
        results = data.get("results", [])

    # Generate LaTeX
    latex = r"""\begin{table}[h]
\centering
\begin{tabular}{lcc}
\toprule
Task & Reward & Success \\
\midrule
"""

    for r in results[:10]:  # Top 10 tasks
        task = r["task"].replace("_", r"\_")
        reward = r["mean_reward"]
        success = r["success_rate"] * 100

        latex += f"{task} & {reward:.2f} & {success:.1f}\\% \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\caption{Agent performance on Agentick tasks.}
\label{tab:results}
\end{table}
"""

    # Save
    with open(args.output, "w") as f:
        f.write(latex)

    print(f"✓ Saved LaTeX table: {args.output}")
    print("\nInclude in your paper with:")
    print(f"  \\input{{{args.output}}}")


if __name__ == "__main__":
    main()
