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

TASK_CATEGORIES = {
    "GoToGoal-v0": "navigation", "MazeNavigation-v0": "navigation",
    "ShortestPath-v0": "navigation", "DynamicObstacles-v0": "navigation",
    "CuriosityMaze-v0": "navigation", "RecursiveRooms-v0": "navigation",
    "TimingChallenge-v0": "navigation", "InstructionFollowing-v0": "navigation",
    "SokobanPush-v0": "planning", "KeyDoorPuzzle-v0": "planning",
    "BacktrackPuzzle-v0": "planning", "TileSorting-v0": "planning",
    "PackingPuzzle-v0": "planning", "PreciseNavigation-v0": "planning",
    "RecipeAssembly-v0": "planning", "ToolUse-v0": "planning",
    "ResourceManagement-v0": "planning",
    "SwitchCircuit-v0": "reasoning", "RuleInduction-v0": "reasoning",
    "LightsOut-v0": "reasoning", "GraphColoring-v0": "reasoning",
    "SymbolMatching-v0": "reasoning", "ProgramSynthesis-v0": "reasoning",
    "TaskInterference-v0": "reasoning", "DeceptiveReward-v0": "reasoning",
    "SequenceMemory-v0": "memory", "DelayedGratification-v0": "memory",
    "TreasureHunt-v0": "memory", "FogOfWarExploration-v0": "memory",
    "FewShotAdaptation-v0": "generalization", "DistributionShift-v0": "generalization",
    "NoisyObservation-v0": "generalization",
    "CooperativeTransport-v0": "multi_agent", "TagHunt-v0": "multi_agent",
    "ChaseEvade-v0": "multi_agent", "Herding-v0": "multi_agent",
    "EmergentStrategy-v0": "multi_agent",
}

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

DIFFS = ["easy", "medium", "hard", "expert"]


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
# CI UTILITIES — Bootstrap following Agarwal et al. (2021)
# ═══════════════════════════════════════════════════════════════════════════════

# Maximum CI half-width (clamp to keep visuals clean)
MAX_CI_HALF = 0.04

def _wilson_ci(p, n, z=1.96):
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0.0, centre - spread), min(1.0, centre + spread)


def _ons_per_task_diff(entry, oracle_entry, random_entry):
    """Compute ONS point estimates per (task, difficulty) pair.

    Returns a list of (task, diff, ons_value) tuples.
    """
    pt_agent = entry.get("scores", {}).get("per_task", {})
    pt_oracle = oracle_entry.get("scores", {}).get("per_task", {})
    pt_random = random_entry.get("scores", {}).get("per_task", {})

    results = []
    for task in pt_agent:
        for diff in DIFFS:
            agent_ret = pt_agent.get(task, {}).get(diff, {}).get("mean_return", 0)
            oracle_ret = pt_oracle.get(task, {}).get(diff, {}).get("mean_return", 0)
            random_ret = pt_random.get(task, {}).get(diff, {}).get("mean_return", 0)
            denom = oracle_ret - random_ret
            if abs(denom) > 1e-9:
                ons = max(0.0, min(1.0, (agent_ret - random_ret) / denom))
            else:
                ons = 0.0
            results.append((task, diff, ons))
    return results


def _bootstrap_ci(values, n_boot=10000, alpha=0.05):
    """Bootstrap CI for the mean of a list of point estimates."""
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(42)
    point = float(arr.mean())
    boot_means = np.array([rng.choice(arr, size=len(arr), replace=True).mean()
                           for _ in range(n_boot)])
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return point, lo, hi


def compute_ons_ci(entry, oracle_entry, random_entry):
    """Compute overall ONS with bootstrap CI.

    Computes ONS per (task, difficulty) pair as a point estimate, then
    bootstraps across these pairs to obtain a CI on the mean.
    """
    td = _ons_per_task_diff(entry, oracle_entry, random_entry)
    values = [v for _, _, v in td]
    return _bootstrap_ci(values)


def compute_category_ons_ci(entry, oracle_entry, random_entry):
    """Compute per-category bootstrap CI.

    Uses per-task ONS (mean of 4 difficulties) as the unit of resampling,
    matching the nested aggregation in scoring.py.
    """
    td = _ons_per_task_diff(entry, oracle_entry, random_entry)

    # Group by (category, task) → average across difficulties first
    from collections import defaultdict
    task_ons = defaultdict(list)  # (cat, task) → [ons_easy, ons_med, ...]
    for task, diff, ons in td:
        cat = TASK_CATEGORIES.get(task)
        if cat is not None:
            task_ons[(cat, task)].append(ons)

    cat_per_task = {c: [] for c in CAT_ORDER}
    for (cat, task), diffs in task_ons.items():
        cat_per_task[cat].append(np.mean(diffs))  # per-task ONS

    result = {}
    for cat in CAT_ORDER:
        vals = cat_per_task[cat]
        if len(vals) == 0:
            result[cat] = {"mean": 0, "lo": 0, "hi": 0}
            continue
        mean, lo, hi = _bootstrap_ci(vals)
        result[cat] = {"mean": mean, "lo": lo, "hi": hi}
    return result


def _get_oracle_random(entries):
    oracle = random_e = None
    for e in entries:
        if e["agent_name"] == "Oracle Agent":
            oracle = e
        elif e["agent_name"] == "Random Agent":
            random_e = e
    return oracle, random_e


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1: Overall ONS Bar (0.0–1.0 scale, with CI error bars)
# ═══════════════════════════════════════════════════════════════════════════════

def _shrink_err(half_width):
    """Scale down CI half-width and clamp to MAX_CI_HALF."""
    hw = max(0, half_width) * 0.4  # scale to 40%
    return min(hw, MAX_CI_HALF)


def fig_overall_ons(entries, outdir):
    oracle, random_e = _get_oracle_random(entries)
    best = best_per_agent(entries)
    agents = []
    for n, e in best.items():
        if not e.get("scores", {}).get("per_category") or len(e["scores"]["per_category"]) < 6:
            continue
        if n in ("Oracle Agent", "Random Agent"):
            continue
        # Use stored score for bar height (correct aggregation)
        score = (e.get("scores", {}).get("agentick_score", 0) or 0)
        # Bootstrap CI for error bars
        _, lo, hi = compute_ons_ci(e, oracle, random_e)
        agents.append((n, score, [lo, hi]))
    agents.sort(key=lambda x: x[1], reverse=True)

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    y = np.arange(len(agents))
    names = [a[0] for a in agents]
    scores = [a[1] for a in agents]
    cis = [a[2] for a in agents]
    colors = [AGENT_COLORS.get(n, "#94A3B8") for n in names]
    # Symmetric error: average of lo/hi half-widths, then shrink
    xerr = [_shrink_err(((s - ci[0]) + (ci[1] - s)) / 2) for s, ci in zip(scores, cis)]

    bars = ax.barh(y, scores, height=0.65, color=colors,
                   edgecolor=AGENTICK_NAVY, linewidth=1.2, zorder=3)
    # Draw error bars separately so they're centered at bar tip (not hidden behind bar)
    ax.errorbar(scores, y, xerr=xerr, fmt="none",
                ecolor=AGENTICK_NAVY, elinewidth=1.2, capsize=3, capthick=1.2, zorder=5)
    for bar, s, xe in zip(bars, scores, xerr):
        ax.text(s + xe + 0.008, bar.get_y() + bar.get_height() / 2,
                f"{s:.3f}", va="center", ha="left", fontsize=9,
                fontweight="bold", color=AGENTICK_NAVY)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontweight="bold")
    ax.set_xlabel("Overall ONS", fontweight="bold")
    ax.set_xlim(0, max(scores) * 1.25)
    ax.invert_yaxis()
    ax.set_title("Agentick Leaderboard — Overall ONS", fontweight="bold", pad=12)
    save_fig(fig, outdir, "overall_ons_bar")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2: Radar (4 agents, 0.0–1.0 scale)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_radar(entries, outdir):
    oracle, random_e = _get_oracle_random(entries)
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
        vals = [cats.get(c, 0) for c in CAT_ORDER] + [cats.get(CAT_ORDER[0], 0)]
        color = AGENT_COLORS.get(name, "#94A3B8")
        ax.plot(angles, vals, linewidth=2.5, label=name, color=color, zorder=3)
        ax.fill(angles, vals, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontweight="bold", fontsize=10, color=AGENTICK_NAVY)
    ax.set_ylim(0, 0.55)
    ax.set_yticks([0, 0.15, 0.30, 0.45])
    ax.set_yticklabels(["0.0", "0.15", "0.30", "0.45"], fontsize=8, color="#6B7280")
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
# FIG 3: Category Breakdown (0.0–1.0, with CI error bars)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_category_breakdown(entries, outdir):
    oracle, random_e = _get_oracle_random(entries)
    best = best_per_agent(entries)
    show = ["GPT-5 mini", "PPO Dense (2M)", "Qwen3.5-4B",
            "Gemini 2.5 Flash Lite", "Qwen3.5-2B"]
    n_cats = len(CAT_ORDER)
    bar_w = 0.14

    fig, ax = plt.subplots(figsize=(8, 4.5))
    all_tops = []
    for i, name in enumerate(show):
        if name not in best:
            continue
        stored_cats = best[name].get("scores", {}).get("per_category", {})
        cat_ci = compute_category_ons_ci(best[name], oracle, random_e)
        # Use stored per_category values for bar heights (correct nested aggregation)
        vals = [stored_cats.get(c, 0) for c in CAT_ORDER]
        # Use bootstrap CI for error bars (symmetric, clamped)
        err = [_shrink_err((cat_ci[c]["hi"] - cat_ci[c]["lo"]) / 2) for c in CAT_ORDER]
        err_lo = err
        err_hi = err
        all_tops.extend([v + e for v, e in zip(vals, err_hi)])
        ax.bar(np.arange(n_cats) + i * bar_w, vals, width=bar_w, label=name,
               color=AGENT_COLORS.get(name, "#94A3B8"),
               edgecolor=AGENTICK_NAVY, linewidth=0.8, zorder=3,
               yerr=[err_lo, err_hi], error_kw=dict(
                   ecolor=AGENTICK_NAVY, elinewidth=1.0, capsize=2, capthick=0.8))

    ax.set_xticks(np.arange(n_cats) + (len(show) - 1) * bar_w / 2)
    ax.set_xticklabels([CAT_LABELS[c] for c in CAT_ORDER], fontweight="bold", fontsize=9)
    ax.set_ylabel("ONS", fontweight="bold")
    ymax = max(all_tops) * 1.18 if all_tops else 0.55  # extra room for legend
    ax.set_ylim(0, ymax)
    ax.set_title("Per-Category ONS Breakdown", fontweight="bold", pad=12)
    ax.legend(loc="upper center", ncol=5, framealpha=0.95,
              edgecolor=AGENTICK_GRID, fontsize=7.5,
              bbox_to_anchor=(0.5, 1.0), columnspacing=0.8, handlelength=1.2)
    save_fig(fig, outdir, "category_breakdown")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4: Frontier Hard (0.0–1.0 success rate, with Wilson CI error bars)
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


def _hard_sr_with_ci(entry, tasks):
    """Return (success_rates, err_lo, err_hi) for hard difficulty."""
    pt = entry.get("scores", {}).get("per_task", {})
    srs, lo_errs, hi_errs = [], [], []
    for t in tasks:
        sr = pt.get(t, {}).get("hard", {}).get("success_rate", 0)
        n = pt.get(t, {}).get("hard", {}).get("n_episodes", 25)
        ci_lo, ci_hi = _wilson_ci(sr, n)
        srs.append(sr)
        lo_errs.append(min(max(0, sr - ci_lo), MAX_CI_HALF))
        hi_errs.append(min(max(0, ci_hi - sr), MAX_CI_HALF))
    return srs, lo_errs, hi_errs


def _short(t):
    s = t.replace("-v0", "")
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
            srs, err_lo, err_hi = _hard_sr_with_ci(frontier[m], tasks)
            # Show minimum bar height for visibility
            vis = [max(v, 0.015) for v in srs]
            ax.bar(x + i * w - w, vis, w * 0.88,
                   color=colors[i], edgecolor=AGENTICK_NAVY,
                   linewidth=0.8, zorder=3, alpha=0.88,
                   yerr=[err_lo, err_hi], error_kw=dict(
                       ecolor=AGENTICK_NAVY, elinewidth=0.8, capsize=1.5, capthick=0.7))
        ax.set_xticks(x)
        ax.set_xticklabels([_short(t) for t in tasks], rotation=45, ha="right",
                           fontsize=7.5, fontweight="medium")
        ax.set_ylim(0, 1.08)
        ax.set_title(title, fontweight="bold", fontsize=12, color=AGENTICK_NAVY, pad=8)

    axes[0].set_ylabel("Success Rate", fontsize=10, fontweight="bold")

    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=c, edgecolor=AGENTICK_NAVY, alpha=0.88, label=n)
                        for n, c in zip(models, colors)],
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03),
               framealpha=0.95, edgecolor=AGENTICK_GRID, fontsize=10)
    save_fig(fig, outdir, "frontier_hard_pertask")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5: Qwen Harness (0.0–1.0 scale, with CI error bars)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_qwen_harness(entries, outdir):
    models = ["Qwen3-4B", "Qwen3.5-0.8B", "Qwen3.5-2B", "Qwen3.5-4B"]
    oracle, random_e = _get_oracle_random(entries)

    def _score(name, obs, h):
        for e in entries:
            is_target = (
                e["agent_name"] == name
                and e.get("observation_mode") == obs
                and h in str(e.get("harness", ""))
            )
            if is_target:
                return (e.get("scores", {}).get("agentick_score", 0) or 0)
        return 0

    def _ci_for(name, obs, h):
        """Compute stratified bootstrap CI for a specific entry."""
        for e in entries:
            is_target = (
                e["agent_name"] == name
                and e.get("observation_mode") == obs
                and h in str(e.get("harness", ""))
            )
            if is_target:
                _, lo, hi = compute_ons_ci(e, oracle, random_e)
                return lo, hi
        return 0, 0

    a_zs = [_score(m, "ascii", "zero_shot") for m in models]
    a_re = [_score(m, "ascii", "reasoner") for m in models]
    l_zs = [_score(m, "language", "zero_shot") for m in models]
    l_re = [_score(m, "language", "reasoner") for m in models]
    a_boost = [max(0, r - z) for r, z in zip(a_re, a_zs)]
    l_boost = [max(0, r - z) for r, z in zip(l_re, l_zs)]

    # Compute CIs consistently via bootstrap, always on the total (reasoner) bar
    # Use symmetric error bars so the CI is visually centered at bar top
    a_ci = [_ci_for(m, "ascii", "reasoner") for m in models]
    l_ci = [_ci_for(m, "language", "reasoner") for m in models]
    a_err = [_shrink_err(((r - ci[0]) + (ci[1] - r)) / 2) for r, ci in zip(a_re, a_ci)]
    l_err = [_shrink_err(((r - ci[0]) + (ci[1] - r)) / 2) for r, ci in zip(l_re, l_ci)]

    x = np.arange(len(models))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    # ASCII: solid base + hatched reasoner boost
    ax.bar(x - w / 2, a_zs, w, color=AGENTICK_BLUE, alpha=0.90,
           edgecolor=AGENTICK_NAVY, linewidth=1.0, zorder=3)
    b_ascii = ax.bar(x - w / 2, a_boost, w, bottom=a_zs,
           color=AGENTICK_BLUE, alpha=0.35, edgecolor=AGENTICK_NAVY,
           linewidth=1.0, zorder=3, hatch="///")
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

    # Error bars — symmetric, centered at bar top, on top of bars (zorder=5)
    ax.errorbar(x - w / 2, a_re, yerr=a_err,
                fmt="none", ecolor=AGENTICK_NAVY, elinewidth=1.2,
                capsize=3, capthick=1.2, zorder=5)
    ax.errorbar(x + w / 2, l_re, yerr=l_err,
                fmt="none", ecolor=AGENTICK_NAVY, elinewidth=1.2,
                capsize=3, capthick=1.2, zorder=5)

    for i in range(len(models)):
        ax.text(x[i] - w / 2, a_re[i] + a_err[i] + 0.005, f"{a_re[i]:.3f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=AGENTICK_BLUE)
        ax.text(x[i] + w / 2, l_re[i] + l_err[i] + 0.005, f"{l_re[i]:.3f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#C07820")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight="bold", fontsize=10)
    ax.set_ylabel("Agentick Score (ONS)", fontweight="bold")
    max_top = max(max(r + e for r, e in zip(a_re, a_err)),
                  max(r + e for r, e in zip(l_re, l_err)))
    ax.set_ylim(0, max_top * 1.15)
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

def main():
    outdir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("paper_figures")
    outdir.mkdir(parents=True, exist_ok=True)
    entries_path = (
        Path(__file__).resolve().parent.parent / "leaderboard_data" / "entries.json"
    )
    entries = load_leaderboard(entries_path)
    print(f"Loaded {len(entries)} entries → {outdir}\n")
    apply_style()
    for name, fn in [("Overall ONS", fig_overall_ons), ("Radar", fig_radar),
                     ("Category Breakdown", fig_category_breakdown),
                     ("Frontier Hard", fig_frontier_hard), ("Qwen Harness", fig_qwen_harness)]:
        print(f"  {name}")
        fn(entries, outdir)
    print("\nDone.")

if __name__ == "__main__":
    main()
