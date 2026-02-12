"""LaTeX table generation for publication."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def generate_main_results_table(
    results_dict: dict[str, dict[str, Any]],
    output_path: str | None = None,
    formats: list[str] | None = None,
) -> str:
    """
    Generate main results table with booktabs formatting.

    Args:
        results_dict: Agent -> (capability -> {mean, ci_lower, ci_upper})
        output_path: Output path (without extension)
        formats: Output formats (latex, markdown, csv)

    Returns:
        LaTeX table string
    """
    if formats is None:
        formats = ["latex", "markdown", "csv"]

    agents = list(results_dict.keys())
    capabilities = sorted(set().union(*[set(r.keys()) for r in results_dict.values()]))

    # Build LaTeX table
    latex = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\begin{tabular}{l" + "c" * len(capabilities) + "c}",
    ]
    latex.append("\\toprule")

    # Header
    header = "Agent & " + " & ".join(capabilities) + " & Aggregate \\\\"
    latex.append(header)
    latex.append("\\midrule")

    # Find best per column
    best_per_cap = {}
    for cap in capabilities:
        values = [
            results_dict[agent][cap]["mean"] for agent in agents if cap in results_dict[agent]
        ]
        if values:
            best_per_cap[cap] = max(values)

    # Rows
    for agent in agents:
        row_parts = [agent]

        for cap in capabilities:
            if cap in results_dict[agent]:
                data = results_dict[agent][cap]
                mean = data["mean"]
                ci_lower = data.get("ci_lower", mean)
                ci_upper = data.get("ci_upper", mean)

                # Format with ± CI
                ci_width = (ci_upper - ci_lower) / 2
                value_str = f"{mean:.2f}±{ci_width:.2f}"

                # Bold if best
                if abs(mean - best_per_cap.get(cap, -1)) < 1e-6:
                    value_str = f"\\textbf{{{value_str}}}"

                row_parts.append(value_str)
            else:
                row_parts.append("—")

        # Aggregate (mean across capabilities)
        cap_means = [
            results_dict[agent][cap]["mean"] for cap in capabilities if cap in results_dict[agent]
        ]
        if cap_means:
            aggregate = np.mean(cap_means)
            row_parts.append(f"{aggregate:.2f}")
        else:
            row_parts.append("—")

        latex.append(" & ".join(row_parts) + " \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append(
        "\\caption{Main results. Values are mean ± 95\\% CI. Bold indicates best per column.}"
    )
    latex.append("\\label{tab:main_results}")
    latex.append("\\end{table}")

    latex_str = "\n".join(latex)

    # Save if output_path specified
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if "latex" in formats:
            with open(output_path.with_suffix(".tex"), "w") as f:
                f.write(latex_str)

        if "markdown" in formats:
            md = _latex_to_markdown(latex_str)
            with open(output_path.with_suffix(".md"), "w") as f:
                f.write(md)

        if "csv" in formats:
            csv = _latex_to_csv(latex_str)
            with open(output_path.with_suffix(".csv"), "w") as f:
                f.write(csv)

    return latex_str


def generate_per_task_table(
    results_dict: dict[str, dict[str, Any]],
    output_path: str | None = None,
) -> str:
    """Generate per-task detailed breakdown table."""
    # Similar to main_results_table but with tasks as columns
    return generate_main_results_table(results_dict, output_path)


def generate_ablation_table(
    baseline: dict[str, Any],
    ablations: dict[str, dict[str, Any]],
    output_path: str | None = None,
) -> str:
    """Generate ablation study table."""
    # Build table showing baseline vs each ablation
    latex = ["\\begin{table}[ht]", "\\centering", "\\begin{tabular}{lcc}"]
    latex.append("\\toprule")
    latex.append("Ablation & Score & $\\Delta$ \\\\")
    latex.append("\\midrule")

    # Baseline
    baseline_score = baseline.get("aggregate_score", 0)
    latex.append(f"Baseline & {baseline_score:.3f} & — \\\\")

    # Ablations
    for name, data in ablations.items():
        score = data.get("aggregate_score", 0)
        delta = score - baseline_score
        sign = "+" if delta >= 0 else ""
        latex.append(f"{name} & {score:.3f} & {sign}{delta:.3f} \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\caption{Ablation study results.}")
    latex.append("\\label{tab:ablation}")
    latex.append("\\end{table}")

    latex_str = "\n".join(latex)

    if output_path:
        with open(output_path, "w") as f:
            f.write(latex_str)

    return latex_str


def generate_comparison_table(
    comparison_result: Any,
    output_path: str | None = None,
) -> str:
    """Generate win/tie/loss comparison table."""
    latex = ["\\begin{table}[ht]", "\\centering", "\\begin{tabular}{lccc}"]
    latex.append("\\toprule")
    latex.append("Agent A vs Agent B & Wins & Ties & Losses \\\\")
    latex.append("\\midrule")

    wins_a = comparison_result.wins_a
    wins_b = comparison_result.wins_b
    ties = comparison_result.ties

    latex.append(f"A vs B & {wins_a} & {ties} & {wins_b} \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\caption{Pairwise comparison (significant differences only).}")
    latex.append("\\label{tab:comparison}")
    latex.append("\\end{table}")

    latex_str = "\n".join(latex)

    if output_path:
        with open(output_path, "w") as f:
            f.write(latex_str)

    return latex_str


def _latex_to_markdown(latex_str: str) -> str:
    """Convert LaTeX table to Markdown (simple)."""
    # Very basic conversion - just for quick viewing
    lines = latex_str.split("\n")
    md_lines = []

    for line in lines:
        if "\\\\" in line:
            md_line = line.replace("\\\\", "").replace("&", "|")
            md_lines.append("|" + md_line + "|")
        elif "\\midrule" in line or "\\toprule" in line:
            # Add separator
            md_lines.append("|---|" * 5)

    return "\n".join(md_lines)


def _latex_to_csv(latex_str: str) -> str:
    """Convert LaTeX table to CSV."""
    lines = latex_str.split("\n")
    csv_lines = []

    for line in lines:
        if "\\\\" in line and "&" in line:
            csv_line = line.replace("\\\\", "").replace("&", ",").strip()
            # Remove LaTeX commands
            csv_line = csv_line.replace("\\textbf{", "").replace("}", "")
            csv_lines.append(csv_line)

    return "\n".join(csv_lines)
