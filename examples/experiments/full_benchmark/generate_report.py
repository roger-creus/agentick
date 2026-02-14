"""Generate HTML report from benchmark results.

This script creates a comprehensive HTML report with:
- Summary statistics table
- Embedded plots
- Links to videos
- Sample interaction traces

Requirements:
    uv sync --extra all

Usage:
    uv run python examples/experiments/full_benchmark/generate_report.py
"""

import argparse
import json
from datetime import datetime
from pathlib import Path


def load_all_results(results_dir: Path) -> dict[str, list[dict]]:
    """Load all experiment results."""
    all_results = {}

    if not results_dir.exists():
        return {}

    for exp_dir in results_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        json_files = list(exp_dir.glob("*_results.json"))
        if not json_files:
            continue

        with open(json_files[0]) as f:
            results = json.load(f)

        all_results[exp_dir.name] = results

    return all_results


def generate_html_report(all_results: dict, figures_dir: Path, output_path: Path):
    """Generate HTML report."""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AgentICK Full Benchmark Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        h1 {{ margin: 0; font-size: 2.5em; }}
        .timestamp {{ opacity: 0.9; margin-top: 10px; }}
        .summary {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
        }}
        .plot {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .plot img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 0 auto;
        }}
        .metric {{
            display: inline-block;
            background: #e3f2fd;
            padding: 10px 20px;
            border-radius: 5px;
            margin: 10px 10px 10px 0;
        }}
        .metric-label {{
            font-size: 0.9em;
            color: #666;
        }}
        .metric-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #1976d2;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 AgentICK Full Benchmark Report</h1>
        <div class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
    </div>

    <div class="summary">
        <h2>Summary Statistics</h2>
"""

    # Add metrics
    total_experiments = len(all_results)
    total_episodes = sum(len(results) for results in all_results.values())

    html += f"""
        <div class="metric">
            <div class="metric-label">Total Experiments</div>
            <div class="metric-value">{total_experiments}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Total Episodes</div>
            <div class="metric-value">{total_episodes}</div>
        </div>
"""

    # Add table
    html += """
        <h3>Per-Experiment Results</h3>
        <table>
            <tr>
                <th>Experiment</th>
                <th>Episodes</th>
                <th>Avg Reward</th>
                <th>Success Rate</th>
                <th>Avg Steps</th>
            </tr>
"""

    for exp_name, results in sorted(all_results.items()):
        if results:
            avg_reward = sum(r.get("total_reward", 0) for r in results) / len(results)
            success_rate = sum(1 for r in results if r.get("success", False)) / len(results)
            avg_steps = sum(r.get("steps", 0) for r in results) / len(results)

            html += f"""
            <tr>
                <td><strong>{exp_name}</strong></td>
                <td>{len(results)}</td>
                <td>{avg_reward:.2f}</td>
                <td>{success_rate:.1%}</td>
                <td>{avg_steps:.1f}</td>
            </tr>
"""

    html += """
        </table>
    </div>
"""

    # Add plots
    html += """
    <div class="summary">
        <h2>Visualizations</h2>
"""

    plot_files = [
        ("comparison_bar.png", "Agent Performance Comparison"),
        ("per_task_comparison.png", "Per-Task Performance"),
        ("success_rates.png", "Success Rate Comparison"),
    ]

    for plot_file, title in plot_files:
        plot_path = figures_dir / plot_file
        if plot_path.exists():
            html += f"""
        <div class="plot">
            <h3>{title}</h3>
            <img src="{plot_path.relative_to(output_path.parent)}" alt="{title}">
        </div>
"""

    html += """
    </div>

    <div class="summary">
        <h2>Videos and Traces</h2>
        <p>Videos and interaction traces are saved in each experiment's directory:</p>
        <ul>
"""

    for exp_name in sorted(all_results.keys()):
        html += f"""
            <li><strong>{exp_name}/</strong>
                <ul>
                    <li>Videos: <code>results/full_benchmark/{exp_name}/videos/</code></li>
                    <li>Traces: <code>results/full_benchmark/{exp_name}/traces/</code></li>
                </ul>
            </li>
"""

    html += """
        </ul>
    </div>

    <div class="summary">
        <h2>Reproduction</h2>
        <p>To reproduce these results:</p>
        <pre><code>bash examples/experiments/full_benchmark/run_all_benchmarks.sh</code></pre>
    </div>

</body>
</html>
"""

    with open(output_path, "w") as f:
        f.write(html)


def main():
    """Generate HTML report."""
    parser = argparse.ArgumentParser(description="Generate HTML benchmark report")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/full_benchmark",
        help="Directory containing experiment results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/full_benchmark/report.html",
        help="Output HTML file path",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("GENERATING HTML REPORT")
    print("=" * 80)
    print()

    results_dir = Path(args.results_dir)
    output_path = Path(args.output)
    figures_dir = results_dir / "figures"

    # Load results
    print("Loading results...")
    all_results = load_all_results(results_dir)

    if not all_results:
        print("\n❌ No results found in", results_dir)
        return

    print(f"Found {len(all_results)} experiments")
    print()

    # Generate report
    print("Generating HTML report...")
    generate_html_report(all_results, figures_dir, output_path)

    print()
    print("=" * 80)
    print("REPORT COMPLETE")
    print("=" * 80)
    print(f"Report saved to: {output_path}")
    print()
    print("Open in browser:")
    print(f"  file://{output_path.absolute()}")


if __name__ == "__main__":
    main()
