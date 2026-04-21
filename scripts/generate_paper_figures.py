"""
Generate all Agentick paper figures with unified visual style.

Usage:
    python scripts/generate_paper_figures.py [--outdir DIR]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# AGENTICK STYLE
# ═══════════════════════════════════════════════════════════════════════════════

AGENTICK_BLUE = "#3B82F6"
AGENTICK_NAVY = "#1E3A5F"
AGENTICK_BG = "#F0F4FF"
AGENTICK_GRID = "#B8D4FF"

CAT_COLORS = {
    "navigation": "#3B82F6", "planning": "#8B5CF6", "reasoning": "#06B6D4",
    "memory": "#F59E0B", "generalization": "#EC4899", "multi_agent": "#10B981",
}
CAT_LABELS = {
    "navigation": "Navigation", "planning": "Planning", "reasoning": "Reasoning",
    "memory": "Memory", "generalization": "Generalization", "multi_agent": "Multi-Agent",
}
CAT_ORDER = ["navigation", "planning", "reasoning", "memory", "generalization", "multi_agent"]

AGENT_COLORS = {
    "GPT-5 mini": "#3B82F6", "PPO Dense (2M)": "#10B981", "Qwen3.5-4B": "#8B5CF6",
    "PPO Dense (500k)": "#34D399", "Gemini 2.5 Flash Lite": "#F59E0B",
    "Qwen3.5-2B": "#A78BFA", "Qwen3.5-0.8B": "#C4B5FD", "Qwen3-4B": "#DDD6FE",
    "PPO Sparse (500k)": "#6EE7B7", "Random Agent": "#94A3B8",
    "Claude Haiku 4.5": "#F87171", "Gemini 3.1 Flash Lite": "#FBBF24",
}
FRONTIER_COLORS = {
    "GPT-5 mini": "#3B82F6", "Gemini 3.1 Flash Lite": "#F59E0B",
    "Claude Haiku 4.5": "#EC4899",
}


def apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10, "font.weight": "medium",
        "axes.labelsize": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "axes.linewidth": 1.8, "axes.edgecolor": AGENTICK_NAVY,
        "lines.linewidth": 2.0, "lines.markersize": 6,
        "axes.grid": True, "grid.alpha": 0.35, "grid.color": AGENTICK_GRID,
        "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.width": 1.5, "ytick.major.width": 1.5,
        "xtick.major.size": 5, "ytick.major.size": 5,
        "xtick.color": AGENTICK_NAVY, "ytick.color": AGENTICK_NAVY,
        "axes.labelcolor": AGENTICK_NAVY,
        "figure.facecolor": "white", "axes.facecolor": AGENTICK_BG,
        "figure.dpi": 300, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.15,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "legend.framealpha": 0.95, "legend.edgecolor": AGENTICK_GRID,
        "legend.fancybox": True,
    })


def save_fig(fig, outdir, name):
    path = outdir / f"{name}.pdf"
    fig.savefig(path, format="pdf", facecolor=fig.get_facecolor())
    print(f"  -> {path}")
    plt.close(fig)


def load_leaderboard(path):
    with open(path) as f:
        return json.load(f)["entries"]


def best_per_agent(entries):
    best = {}
    for e in entries:
        n = e["agent_name"]
        s = e.get("scores", {}).get("agentick_score", 0) or 0
        if n not in best or s > (best[n].get("scores", {}).get("agentick_score", 0) or 0):
            best[n] = e
    return best


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1: Overall ONS Bar (taller to match radar)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_overall_ons(entries, outdir):
    best = best_per_agent(entries)
    agents = [(n, (e.get("scores", {}).get("agentick_score", 0) or 0) * 100)
              for n, e in best.items()
              if e.get("scores", {}).get("per_category") and len(e["scores"]["per_category"]) >= 6
              and n not in ("Oracle Agent", "Random Agent")]
    agents.sort(key=lambda x: x[1], reverse=True)

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    y = np.arange(len(agents))
    names, scores = zip(*agents)
    colors = [AGENT_COLORS.get(n, "#94A3B8") for n in names]
    bars = ax.barh(y, scores, height=0.65, color=colors,
                   edgecolor=AGENTICK_NAVY, linewidth=1.2, zorder=3)
    for bar, s in zip(bars, scores):
        ax.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2,
                f"{s:.1f}%", va="center", ha="left", fontsize=9,
                fontweight="bold", color=AGENTICK_NAVY)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontweight="bold")
    ax.set_xlabel("Overall ONS (%)", fontweight="bold")
    ax.set_xlim(0, max(scores) * 1.18)
    ax.invert_yaxis()
    ax.set_title("Agentick Leaderboard — Overall ONS", fontweight="bold", pad=12)
    save_fig(fig, outdir, "overall_ons_bar")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2: Radar (4 agents, single-row legend)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_radar(entries, outdir):
    best = best_per_agent(entries)
    show = ["GPT-5 mini", "PPO Dense (2M)", "Qwen3.5-4B", "Gemini 2.5 Flash Lite"]
    labels = [CAT_LABELS[c] for c in CAT_ORDER]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5.5, 5.0), subplot_kw={"polar": True})
    ax.set_facecolor(AGENTICK_BG)
    for name in show:
        if name not in best:
            continue
        cats = best[name].get("scores", {}).get("per_category", {})
        if not cats:
            continue
        vals = [cats.get(c, 0) * 100 for c in CAT_ORDER] + [cats.get(CAT_ORDER[0], 0) * 100]
        color = AGENT_COLORS.get(name, "#94A3B8")
        ax.plot(angles, vals, linewidth=2.5, label=name, color=color, zorder=3)
        ax.fill(angles, vals, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontweight="bold", fontsize=10, color=AGENTICK_NAVY)
    ax.set_ylim(0, 55)
    ax.set_yticks([0, 15, 30, 45])
    ax.set_yticklabels(["0%", "15%", "30%", "45%"], fontsize=8, color="#6B7280")
    for spine in ["polar"]:
        ax.spines[spine].set_color(AGENTICK_GRID)
        ax.spines[spine].set_linewidth(1.2)
    ax.yaxis.grid(True, color=AGENTICK_GRID, linewidth=0.8, alpha=0.5)
    ax.xaxis.grid(True, color=AGENTICK_GRID, linewidth=0.8, alpha=0.5)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.15),
              framealpha=0.95, edgecolor=AGENTICK_GRID, fontsize=8.5, ncol=4)
    ax.set_title("Capability Profiles — Per-Category ONS", fontweight="bold",
                 pad=20, fontsize=13, color=AGENTICK_NAVY)
    save_fig(fig, outdir, "category_ons_radar")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3: Category Breakdown (vertical, legend outside right)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_category_breakdown(entries, outdir):
    best = best_per_agent(entries)
    show = ["GPT-5 mini", "PPO Dense (2M)", "Qwen3.5-4B",
            "Gemini 2.5 Flash Lite", "Qwen3.5-2B"]
    n_cats = len(CAT_ORDER)
    bar_w = 0.14

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, name in enumerate(show):
        if name not in best:
            continue
        cats = best[name].get("scores", {}).get("per_category", {})
        if not cats:
            continue
        vals = [cats.get(c, 0) * 100 for c in CAT_ORDER]
        ax.bar(np.arange(n_cats) + i * bar_w, vals, width=bar_w, label=name,
               color=AGENT_COLORS.get(name, "#94A3B8"),
               edgecolor=AGENTICK_NAVY, linewidth=0.8, zorder=3)

    ax.set_xticks(np.arange(n_cats) + (len(show) - 1) * bar_w / 2)
    ax.set_xticklabels([CAT_LABELS[c] for c in CAT_ORDER], fontweight="bold", fontsize=9)
    ax.set_ylabel("ONS (%)", fontweight="bold")
    ax.set_ylim(0, 55)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_title("Per-Category ONS Breakdown", fontweight="bold", pad=12)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
              framealpha=0.95, edgecolor=AGENTICK_GRID, fontsize=8.5)
    save_fig(fig, outdir, "category_breakdown")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4: Frontier Hard (3-col, LARGER, wider bars)
# ═══════════════════════════════════════════════════════════════════════════════

NAV_TASKS = ["CuriosityMaze-v0", "DynamicObstacles-v0", "GoToGoal-v0",
             "InstructionFollowing-v0", "MazeNavigation-v0", "RecursiveRooms-v0",
             "ShortestPath-v0", "TimingChallenge-v0"]
PLAN_TASKS = ["BacktrackPuzzle-v0", "KeyDoorPuzzle-v0", "PackingPuzzle-v0",
              "PreciseNavigation-v0", "RecipeAssembly-v0", "ResourceManagement-v0",
              "SokobanPush-v0", "TileSorting-v0", "ToolUse-v0"]
REASON_TASKS = ["DeceptiveReward-v0", "GraphColoring-v0", "LightsOut-v0",
                "ProgramSynthesis-v0", "RuleInduction-v0", "SwitchCircuit-v0",
                "SymbolMatching-v0", "TaskInterference-v0"]


def _hard_sr(entry, tasks):
    pt = entry.get("scores", {}).get("per_task", {})
    return [pt.get(t, {}).get("hard", {}).get("success_rate", 0) * 100 for t in tasks]


def _short(t):
    s = t.replace("-v0", "")
    # Abbreviate long names for readability
    abbr = {"DynamicObstacles": "DynObst", "InstructionFollowing": "InstrFollow",
            "MazeNavigation": "MazeNav", "RecursiveRooms": "RecRooms",
            "TimingChallenge": "Timing", "CuriosityMaze": "Curiosity",
            "BacktrackPuzzle": "Backtrack", "PreciseNavigation": "PrecNav",
            "RecipeAssembly": "Recipe", "ResourceManagement": "Resource",
            "DeceptiveReward": "Deceptive", "GraphColoring": "GraphCol",
            "ProgramSynthesis": "ProgSynth", "RuleInduction": "RuleInd",
            "SwitchCircuit": "SwitchCir", "SymbolMatching": "SymMatch",
            "TaskInterference": "TaskInterf"}
    return abbr.get(s, s)


def fig_frontier_hard(entries, outdir):
    frontier = {}
    for e in entries:
        n = e["agent_name"]
        if n == "GPT-5 mini" and "Reasoner" in str(e.get("harness", "")):
            frontier[n] = e
        elif n in ("Claude Haiku 4.5", "Gemini 3.1 Flash Lite"):
            frontier[n] = e

    models = ["GPT-5 mini", "Gemini 3.1 Flash Lite", "Claude Haiku 4.5"]
    colors = [FRONTIER_COLORS[m] for m in models]
    panels = [("Navigation", NAV_TASKS), ("Planning", PLAN_TASKS), ("Reasoning", REASON_TASKS)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0), sharey=True)
    fig.subplots_adjust(wspace=0.06)

    for ax, (title, tasks) in zip(axes, panels):
        n_t = len(tasks)
        x = np.arange(n_t)
        w = 0.27
        for i, m in enumerate(models):
            if m not in frontier:
                continue
            vals = _hard_sr(frontier[m], tasks)
            vis = [max(v, 1.5) for v in vals]
            ax.bar(x + i * w - w, vis, w * 0.88,
                   color=colors[i], edgecolor=AGENTICK_NAVY,
                   linewidth=0.8, zorder=3, alpha=0.88)
        ax.set_xticks(x)
        ax.set_xticklabels([_short(t) for t in tasks], rotation=45, ha="right",
                           fontsize=7.5, fontweight="medium")
        ax.set_ylim(0, 108)
        ax.set_title(title, fontweight="bold", fontsize=12, color=AGENTICK_NAVY, pad=8)

    axes[0].set_ylabel("Success Rate (%)", fontsize=10, fontweight="bold")
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=c, edgecolor=AGENTICK_NAVY, alpha=0.88, label=n)
                        for n, c in zip(models, colors)],
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03),
               framealpha=0.95, edgecolor=AGENTICK_GRID, fontsize=10)
    save_fig(fig, outdir, "frontier_hard_pertask")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5: Qwen Harness (hatching on BOTH, larger)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_qwen_harness(entries, outdir):
    models = ["Qwen3-4B", "Qwen3.5-0.8B", "Qwen3.5-2B", "Qwen3.5-4B"]

    def _ons(name, obs, h):
        for e in entries:
            if e["agent_name"] == name and e.get("observation_mode") == obs and h in str(e.get("harness", "")):
                return (e.get("scores", {}).get("agentick_score", 0) or 0) * 100
        return 0

    a_zs = [_ons(m, "ascii", "zero_shot") for m in models]
    a_re = [_ons(m, "ascii", "reasoner") for m in models]
    l_zs = [_ons(m, "language", "zero_shot") for m in models]
    l_re = [_ons(m, "language", "reasoner") for m in models]
    a_boost = [r - z for r, z in zip(a_re, a_zs)]
    l_boost = [r - z for r, z in zip(l_re, l_zs)]

    x = np.arange(len(models))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    # ASCII: solid base + hatched reasoner boost
    ax.bar(x - w / 2, a_zs, w, color=AGENTICK_BLUE, alpha=0.90,
           edgecolor=AGENTICK_NAVY, linewidth=1.0, zorder=3)
    b_ascii = ax.bar(x - w / 2, a_boost, w, bottom=a_zs,
           color=AGENTICK_BLUE, alpha=0.35, edgecolor=AGENTICK_NAVY,
           linewidth=1.0, zorder=3, hatch="///")
    # Force white hatch lines so they're visible on blue
    for patch in b_ascii:
        patch.set_edgecolor("white")
        patch.set_linewidth(0.5)

    # Language: solid base + hatched reasoner boost
    ax.bar(x + w / 2, l_zs, w, color="#F59E0B", alpha=0.90,
           edgecolor=AGENTICK_NAVY, linewidth=1.0, zorder=3)
    b_lang = ax.bar(x + w / 2, l_boost, w, bottom=l_zs,
           color="#F59E0B", alpha=0.35, edgecolor="#D97706",
           linewidth=1.0, zorder=3, hatch="///")
    for patch in b_lang:
        patch.set_edgecolor("white")
        patch.set_linewidth(0.5)

    for i in range(len(models)):
        ax.text(x[i] - w / 2, a_re[i] + 0.3, f"{a_re[i]:.1f}%",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=AGENTICK_BLUE)
        ax.text(x[i] + w / 2, l_re[i] + 0.3, f"{l_re[i]:.1f}%",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#C07820")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight="bold", fontsize=10)
    ax.set_ylabel("Agentick Score (ONS %)", fontweight="bold")
    ax.set_ylim(0, 27)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_title("Observation Mode & Reasoning Harness — Qwen Models",
                 fontweight="bold", pad=12)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=AGENTICK_BLUE, alpha=0.90, edgecolor=AGENTICK_NAVY, label="ASCII"),
        Patch(facecolor="#F59E0B", alpha=0.90, edgecolor=AGENTICK_NAVY, label="Language"),
        Patch(facecolor="#888888", alpha=0.50, edgecolor="#666", label="+Reasoner", hatch="//"),
    ], loc="upper left", framealpha=0.95, edgecolor=AGENTICK_GRID)
    save_fig(fig, outdir, "qwen_harness_comparison")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 6: SFT vs. baselines (grouped bar)
# FIG 7: SFT data-scaling curve
# ═══════════════════════════════════════════════════════════════════════════════

SFT_SIZES = ["120k", "250k", "500k"]


def _find_ons(entries, agent_name: str, obs_mode: str, harness_substring: str) -> float:
    """Return ONS (%) for the best matching entry, or nan if not found."""
    best = None
    for e in entries:
        if e.get("agent_name") != agent_name:
            continue
        if e.get("observation_mode") != obs_mode:
            continue
        if harness_substring not in str(e.get("harness", "")):
            continue
        s = (e.get("scores", {}).get("agentick_score", 0) or 0) * 100
        if best is None or s > best:
            best = s
    return float("nan") if best is None else best


def fig_sft_vs_baselines(entries, outdir):
    """Grouped bar: baseline + 3 SFT sizes, under Markov and Reasoner harnesses."""
    import math

    # Collect numbers
    baseline_markov = _find_ons(entries, "Qwen3.5-4B", "ascii", "zero_shot")
    baseline_reasoner = _find_ons(entries, "Qwen3.5-4B", "ascii", "reasoner")
    sft_markov = [
        _find_ons(entries, f"Qwen3.5-4B SFT-{s}", "ascii", "zero_shot") for s in SFT_SIZES
    ]
    sft_reasoner = [
        _find_ons(entries, f"Qwen3.5-4B SFT-{s}", "ascii", "reasoner") for s in SFT_SIZES
    ]

    all_markov = [baseline_markov] + sft_markov
    all_reasoner = [baseline_reasoner] + sft_reasoner

    # Skip if all are nan (no SFT data yet)
    if all(math.isnan(v) for v in sft_markov + sft_reasoner):
        print("    [sft_vs_baselines] No SFT entries found in entries.json - skipping.")
        return

    groups = ["Baseline", "SFT-120k", "SFT-250k", "SFT-500k"]
    x = np.arange(len(groups))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    # Markov bars (solid blue)
    ax.bar(x - w/2,
           [0 if math.isnan(v) else v for v in all_markov],
           w, color=AGENTICK_BLUE, alpha=0.90,
           edgecolor=AGENTICK_NAVY, linewidth=1.0, zorder=3,
           label="Markov harness")
    # Reasoner bars (solid orange)
    ax.bar(x + w/2,
           [0 if math.isnan(v) else v for v in all_reasoner],
           w, color="#F59E0B", alpha=0.90,
           edgecolor=AGENTICK_NAVY, linewidth=1.0, zorder=3,
           label="Reasoner harness")

    # Value labels
    for i, v in enumerate(all_markov):
        if not math.isnan(v):
            ax.text(x[i] - w/2, v + 0.4, f"{v:.1f}%", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold", color=AGENTICK_BLUE)
    for i, v in enumerate(all_reasoner):
        if not math.isnan(v):
            ax.text(x[i] + w/2, v + 0.4, f"{v:.1f}%", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold", color="#C07820")

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontweight="bold")
    ax.set_ylabel("Agentick Score (ONS %)", fontweight="bold")
    ymax = max(
        [0 if math.isnan(v) else v for v in all_markov + all_reasoner], default=1
    )
    ax.set_ylim(0, ymax * 1.22 + 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_title("Qwen3.5-4B ASCII: SFT vs. Baselines", fontweight="bold", pad=12)
    ax.legend(loc="upper left", framealpha=0.95, edgecolor=AGENTICK_GRID)
    save_fig(fig, outdir, "sft_vs_baselines")


def fig_sft_scaling_curve(entries, outdir):
    """Data-scaling line: ONS vs dataset size (log x), one line per harness."""
    import math

    sizes_num = [120_000, 250_000, 500_000]
    sft_markov = [
        _find_ons(entries, f"Qwen3.5-4B SFT-{s}", "ascii", "zero_shot") for s in SFT_SIZES
    ]
    sft_reasoner = [
        _find_ons(entries, f"Qwen3.5-4B SFT-{s}", "ascii", "reasoner") for s in SFT_SIZES
    ]

    if all(math.isnan(v) for v in sft_markov + sft_reasoner):
        print("    [sft_scaling_curve] No SFT entries found in entries.json - skipping.")
        return

    baseline_markov = _find_ons(entries, "Qwen3.5-4B", "ascii", "zero_shot")
    baseline_reasoner = _find_ons(entries, "Qwen3.5-4B", "ascii", "reasoner")

    fig, ax = plt.subplots(figsize=(6.0, 3.6))

    ax.plot(sizes_num, sft_markov, marker="o", linewidth=2.2,
            color=AGENTICK_BLUE, label="SFT - Markov eval", zorder=3)
    ax.plot(sizes_num, sft_reasoner, marker="s", linewidth=2.2,
            color="#F59E0B", label="SFT - Reasoner eval", zorder=3)

    if not math.isnan(baseline_markov):
        ax.axhline(baseline_markov, ls="--", c=AGENTICK_BLUE, alpha=0.6, lw=1.4,
                   label=f"Baseline Markov ({baseline_markov:.1f}%)")
    if not math.isnan(baseline_reasoner):
        ax.axhline(baseline_reasoner, ls="--", c="#F59E0B", alpha=0.6, lw=1.4,
                   label=f"Baseline Reasoner ({baseline_reasoner:.1f}%)")

    ax.set_xscale("log")
    ax.set_xlabel("SFT dataset size (rows, log scale)", fontweight="bold")
    ax.set_ylabel("Agentick Score (ONS %)", fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_title("SFT Data-Scaling - Qwen3.5-4B ASCII", fontweight="bold", pad=12)
    ax.set_xticks(sizes_num)
    ax.set_xticklabels(["120k", "250k", "500k"])
    ax.legend(loc="best", framealpha=0.95, edgecolor=AGENTICK_GRID, fontsize=8.5)
    save_fig(fig, outdir, "sft_scaling_curve")


# ═══════════════════════════════════════════════════════════════════════════════

def main():
    outdir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("agentick_paper/figures")
    outdir.mkdir(parents=True, exist_ok=True)
    entries = load_leaderboard(Path(__file__).resolve().parent.parent / "leaderboard_data" / "entries.json")
    print(f"Loaded {len(entries)} entries → {outdir}\n")
    apply_style()
    for name, fn in [("Overall ONS", fig_overall_ons), ("Radar", fig_radar),
                     ("Category Breakdown", fig_category_breakdown),
                     ("Frontier Hard", fig_frontier_hard), ("Qwen Harness", fig_qwen_harness),
                     ("SFT vs Baselines", fig_sft_vs_baselines),
                     ("SFT Scaling Curve", fig_sft_scaling_curve)]:
        print(f"  {name}")
        fn(entries, outdir)
    print("\nDone.")

if __name__ == "__main__":
    main()
