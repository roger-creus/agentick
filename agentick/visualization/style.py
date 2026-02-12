"""Publication-ready plot styling and color schemes."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt

# Colorblind-safe palette (Wong 2011)
COLORBLIND_PALETTE = [
    "#E69F00",  # Orange
    "#56B4E9",  # Sky Blue
    "#009E73",  # Bluish Green
    "#F0E442",  # Yellow
    "#0072B2",  # Blue
    "#D55E00",  # Vermillion
    "#CC79A7",  # Reddish Purple
    "#000000",  # Black
]

# Agent name → color mapping (consistent across all figures)
AGENT_COLORS = {
    "random": COLORBLIND_PALETTE[7],  # Black
    "greedy": COLORBLIND_PALETTE[3],  # Yellow
    "oracle": COLORBLIND_PALETTE[4],  # Blue
    "ppo": COLORBLIND_PALETTE[0],  # Orange
    "dqn": COLORBLIND_PALETTE[1],  # Sky Blue
    "sac": COLORBLIND_PALETTE[2],  # Bluish Green
    "gpt-4o": COLORBLIND_PALETTE[5],  # Vermillion
    "claude": COLORBLIND_PALETTE[6],  # Reddish Purple
    "gpt-4": COLORBLIND_PALETTE[5],
    "human": COLORBLIND_PALETTE[4],
}

# Marker styles
AGENT_MARKERS = {
    "random": "x",
    "greedy": "+",
    "oracle": "*",
    "ppo": "o",
    "dqn": "s",
    "sac": "^",
    "gpt-4o": "D",
    "claude": "v",
    "gpt-4": "D",
    "human": "p",
}

# Line styles
AGENT_LINESTYLES = {
    "random": "--",
    "greedy": ":",
    "oracle": "-.",
    "ppo": "-",
    "dqn": "-",
    "sac": "-",
    "gpt-4o": "-",
    "claude": "-",
    "gpt-4": "-",
    "human": "-",
}


# Publication style presets
PAPER_STYLES = {
    "paper_single_column": {
        "figure.figsize": (3.25, 2.5),
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "lines.linewidth": 1.0,
        "lines.markersize": 3,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    },
    "paper_double_column": {
        "figure.figsize": (6.75, 3.5),
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "axes.linewidth": 1.0,
        "grid.linewidth": 0.6,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
    },
    "poster": {
        "figure.figsize": (10, 7),
        "font.size": 18,
        "axes.labelsize": 20,
        "axes.titlesize": 22,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "lines.linewidth": 3.0,
        "lines.markersize": 10,
        "axes.linewidth": 2.0,
        "grid.linewidth": 1.5,
        "xtick.major.width": 2.0,
        "ytick.major.width": 2.0,
    },
    "presentation": {
        "figure.figsize": (8, 6),
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "axes.linewidth": 1.5,
        "grid.linewidth": 1.0,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
    },
}


def set_style(style_name: str = "paper_double_column") -> None:
    """
    Apply publication-ready plot style.

    Args:
        style_name: One of paper_single_column, paper_double_column, poster, presentation
    """
    if style_name not in PAPER_STYLES:
        raise ValueError(f"Unknown style: {style_name}. Choose from {list(PAPER_STYLES.keys())}")

    style = PAPER_STYLES[style_name]

    # Base style
    plt.style.use("seaborn-v0_8-paper")

    # Apply custom settings
    plt.rcParams.update(style)

    # Additional settings for publication quality
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "text.usetex": False,  # Don't use LaTeX by default (compatibility)
            "pdf.fonttype": 42,  # TrueType fonts (required by some journals)
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
        }
    )


def get_agent_color(agent_name: str) -> str:
    """
    Get consistent color for agent.

    Args:
        agent_name: Agent name

    Returns:
        Hex color string
    """
    # Normalize agent name
    agent_name = agent_name.lower().replace("_", "-")

    return AGENT_COLORS.get(agent_name, COLORBLIND_PALETTE[0])


def get_agent_marker(agent_name: str) -> str:
    """Get consistent marker for agent."""
    agent_name = agent_name.lower().replace("_", "-")
    return AGENT_MARKERS.get(agent_name, "o")


def get_agent_linestyle(agent_name: str) -> str:
    """Get consistent linestyle for agent."""
    agent_name = agent_name.lower().replace("_", "-")
    return AGENT_LINESTYLES.get(agent_name, "-")


def get_palette(n_colors: int | None = None) -> list[str]:
    """
    Get colorblind-safe palette.

    Args:
        n_colors: Number of colors (default: all 8)

    Returns:
        List of hex color strings
    """
    if n_colors is None:
        return COLORBLIND_PALETTE
    else:
        return COLORBLIND_PALETTE[:n_colors]


def save_figure(
    fig: Any,
    output_path: str,
    formats: list[str] | None = None,
    dpi: int = 300,
) -> None:
    """
    Save figure in multiple formats.

    Args:
        fig: Matplotlib figure
        output_path: Output path (without extension)
        formats: List of formats (default: ["pdf", "png"])
        dpi: DPI for raster formats
    """
    if formats is None:
        formats = ["pdf", "png"]

    from pathlib import Path

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        save_path = output_path.with_suffix(f".{fmt}")

        if fmt == "pdf":
            fig.savefig(save_path, format="pdf", bbox_inches="tight")
        elif fmt == "svg":
            fig.savefig(save_path, format="svg", bbox_inches="tight")
        elif fmt == "png":
            fig.savefig(save_path, format="png", dpi=dpi, bbox_inches="tight")
        else:
            raise ValueError(f"Unknown format: {fmt}")

    plt.close(fig)
