"""Auto-report generation for papers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def generate_report(
    results_dict: dict[str, Any],
    output_dir: str | Path,
    format: str = "latex",
) -> None:
    """
    Generate complete results section for paper.

    Args:
        results_dict: Results dictionary
        output_dir: Output directory
        format: Output format (latex or markdown)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if format == "latex":
        report_path = output_dir / "results.tex"
        content = _generate_latex_report(results_dict)
    else:
        report_path = output_dir / "results.md"
        content = _generate_markdown_report(results_dict)

    with open(report_path, "w") as f:
        f.write(content)


def generate_supplementary(
    results_dict: dict[str, Any],
    output_dir: str | Path,
) -> None:
    """
    Generate supplementary materials.

    Args:
        results_dict: Results dictionary
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-task detailed results
    # Per-seed results
    # Full statistical test outputs
    # Episode trajectory examples
    pass


def _generate_latex_report(results_dict: dict[str, Any]) -> str:
    """Generate LaTeX results section."""
    latex = [
        "\\section{Results}",
        "",
        "Table~\\ref{tab:main_results} shows the main results.",
        "Figure~\\ref{fig:capability_radar} shows the capability profile.",
        "",
        "\\input{tables/main_results.tex}",
        "",
        "\\begin{figure}[ht]",
        "\\centering",
        "\\includegraphics[width=0.8\\linewidth]{figures/capability_radar.pdf}",
        "\\caption{Capability radar chart.}",
        "\\label{fig:capability_radar}",
        "\\end{figure}",
    ]

    return "\n".join(latex)


def _generate_markdown_report(results_dict: dict[str, Any]) -> str:
    """Generate Markdown results section."""
    md = [
        "# Results",
        "",
        "## Main Results",
        "",
        "The main results are shown in the table below.",
        "",
        "## Capability Analysis",
        "",
        "The capability radar chart shows...",
    ]

    return "\n".join(md)
