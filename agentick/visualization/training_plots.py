"""Training benchmark visualization and plotting.

Generates diagnostic plots from PPO training benchmark results:
- Success rate heatmap (tasks x difficulties)
- Learning curves by category
- Difficulty scaling
- Capability radar chart
- Per-task final scores
- Reward distribution
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from agentick.experiments.training_runner import (
    ALL_CATEGORIES,
    get_task_category,
)
from agentick.visualization.style import (
    COLORBLIND_PALETTE,
    get_palette,
    save_figure,
    set_style,
)

# Category display colors (one per category, cycling palette)
CATEGORY_COLORS: dict[str, str] = {}
for i, cat in enumerate(ALL_CATEGORIES):
    CATEGORY_COLORS[cat] = COLORBLIND_PALETTE[i % len(COLORBLIND_PALETTE)]


class TrainingBenchmarkPlotter:
    """Generate diagnostic plots from training benchmark results.

    Usage:
        plotter = TrainingBenchmarkPlotter("results/ppo_benchmarks/run_dir/")
        plotter.plot_all()
    """

    def __init__(self, result_dir: str | Path):
        self.result_dir = Path(result_dir)
        if not self.result_dir.exists():
            raise ValueError(f"Result directory not found: {self.result_dir}")

        self.figures_dir = self.result_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)

        self.data = self._load_data()

    def _load_data(self) -> dict[str, Any]:
        """Load training_summary.json from results directory."""
        summary_path = self.result_dir / "training_summary.json"
        if not summary_path.exists():
            raise ValueError(f"training_summary.json not found in {self.result_dir}")
        with open(summary_path) as f:
            return json.load(f)

    @property
    def tasks(self) -> list[str]:
        return self.data.get("tasks", [])

    @property
    def difficulties(self) -> list[str]:
        return self.data.get("difficulties", [])

    @property
    def results(self) -> dict[str, Any]:
        return self.data.get("results", {})

    def plot_all(self) -> None:
        """Generate all plots and save to figures/ directory."""
        set_style("paper_double_column")

        print(f"Generating training benchmark plots for {self.result_dir.name}...")

        self.plot_success_rate_heatmap()
        self.plot_learning_curves_by_category()
        self.plot_difficulty_scaling()
        self.plot_capability_radar()
        self.plot_per_task_final_scores()
        self.plot_reward_distribution()

        print(f"All plots saved to {self.figures_dir}")

    def plot_success_rate_heatmap(self) -> None:
        """Heatmap: tasks (grouped by category) x difficulties, color=success rate."""
        tasks = self.tasks
        difficulties = self.difficulties
        results = self.results

        if not tasks or not difficulties:
            return

        # Group tasks by category and sort
        cat_tasks: dict[str, list[str]] = {c: [] for c in ALL_CATEGORIES}
        for t in tasks:
            cat = get_task_category(t)
            if cat in cat_tasks:
                cat_tasks[cat].append(t)
            else:
                cat_tasks.setdefault("unknown", []).append(t)

        ordered_tasks = []
        category_breaks = []
        category_labels = []
        for cat in ALL_CATEGORIES:
            if cat_tasks[cat]:
                category_breaks.append(len(ordered_tasks))
                category_labels.append(cat)
                ordered_tasks.extend(sorted(cat_tasks[cat]))

        n_tasks = len(ordered_tasks)
        n_diffs = len(difficulties)

        # Build matrix
        matrix = np.full((n_tasks, n_diffs), np.nan)
        for i, task in enumerate(ordered_tasks):
            for j, diff in enumerate(difficulties):
                key = f"{task}_{diff}"
                if key in results:
                    matrix[i, j] = results[key].get("success_rate", 0.0)

        # Plot
        fig_height = max(6, n_tasks * 0.28)
        fig, ax = plt.subplots(figsize=(5, fig_height))

        im = ax.imshow(
            matrix,
            aspect="auto",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )

        ax.set_xticks(range(n_diffs))
        ax.set_xticklabels(difficulties)
        ax.set_yticks(range(n_tasks))
        ax.set_yticklabels(
            [t.replace("-v0", "") for t in ordered_tasks],
            fontsize=7,
        )

        # Annotate cells
        for i in range(n_tasks):
            for j in range(n_diffs):
                val = matrix[i, j]
                if not np.isnan(val):
                    color = "white" if val < 0.4 or val > 0.8 else "black"
                    ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=6, color=color)

        # Draw category separators
        for brk in category_breaks[1:]:
            ax.axhline(y=brk - 0.5, color="black", linewidth=1.0)

        # Category labels on the right
        for idx, (brk, label) in enumerate(zip(category_breaks, category_labels)):
            next_brk = category_breaks[idx + 1] if idx + 1 < len(category_breaks) else n_tasks
            mid = (brk + next_brk - 1) / 2
            ax.text(
                n_diffs + 0.3,
                mid,
                label.replace("_", " "),
                va="center",
                fontsize=7,
                fontstyle="italic",
            )

        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.12)
        cbar.set_label("Success Rate")

        ax.set_title("PPO Success Rate: Tasks x Difficulties", fontsize=10)
        ax.set_xlabel("Difficulty")

        plt.tight_layout()
        save_figure(fig, str(self.figures_dir / "success_rate_heatmap"), formats=["png", "pdf"])

    def plot_learning_curves_by_category(self) -> None:
        """10 subplots, one per category, showing learning curves at medium difficulty."""
        results = self.results
        difficulties = self.difficulties
        target_diff = "medium" if "medium" in difficulties else difficulties[0]

        active_cats = []
        for cat in ALL_CATEGORIES:
            tasks_in_cat = [t for t in self.tasks if get_task_category(t) == cat]
            has_curves = any(
                results.get(f"{t}_{target_diff}", {}).get("training_curve")
                for t in tasks_in_cat
            )
            if has_curves:
                active_cats.append(cat)

        if not active_cats:
            return

        n_cats = len(active_cats)
        n_cols = min(3, n_cats)
        n_rows = (n_cats + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)

        palette = get_palette()

        for idx, cat in enumerate(active_cats):
            ax = axes[idx // n_cols][idx % n_cols]
            tasks_in_cat = sorted(
                t for t in self.tasks if get_task_category(t) == cat
            )

            for ti, task in enumerate(tasks_in_cat):
                key = f"{task}_{target_diff}"
                curve = results.get(key, {}).get("training_curve", [])
                if not curve:
                    continue
                ts = [p["timestep"] for p in curve]
                means = [p["mean_reward"] for p in curve]
                color = palette[ti % len(palette)]
                ax.plot(ts, means, label=task.replace("-v0", ""), color=color, linewidth=1.2)

            ax.set_title(cat.replace("_", " ").title(), fontsize=9)
            ax.set_xlabel("Timesteps")
            ax.set_ylabel("Mean Eval Reward")
            ax.legend(fontsize=6, loc="best")
            ax.grid(True, alpha=0.3)

        # Hide unused axes
        for idx in range(n_cats, n_rows * n_cols):
            axes[idx // n_cols][idx % n_cols].set_visible(False)

        fig.suptitle(f"Learning Curves ({target_diff} difficulty)", fontsize=11, y=1.02)
        plt.tight_layout()
        save_figure(
            fig, str(self.figures_dir / "learning_curves_by_category"), formats=["png", "pdf"]
        )

    def plot_difficulty_scaling(self) -> None:
        """Line plot: x=difficulty, y=mean success rate across all tasks."""
        difficulties = self.difficulties
        results = self.results

        if len(difficulties) < 2:
            return

        means = []
        stds = []
        for diff in difficulties:
            rates = []
            for task in self.tasks:
                key = f"{task}_{diff}"
                if key in results:
                    rates.append(results[key].get("success_rate", 0.0))
            means.append(float(np.mean(rates)) if rates else 0.0)
            stds.append(float(np.std(rates)) if rates else 0.0)

        fig, ax = plt.subplots(figsize=(5, 3.5))

        ax.errorbar(
            difficulties,
            means,
            yerr=stds,
            marker="o",
            linewidth=2,
            markersize=8,
            capsize=4,
            color=COLORBLIND_PALETTE[4],
        )
        ax.set_xlabel("Difficulty")
        ax.set_ylabel("Mean Success Rate")
        ax.set_title("Difficulty Scaling")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_figure(fig, str(self.figures_dir / "difficulty_scaling"), formats=["png", "pdf"])

    def plot_capability_radar(self) -> None:
        """Radar chart with 10 spokes (categories), value = mean success rate."""
        results = self.results

        cat_scores: dict[str, list[float]] = {c: [] for c in ALL_CATEGORIES}
        for key, res in results.items():
            cat = res.get("category", "unknown")
            if cat in cat_scores:
                cat_scores[cat].append(res.get("success_rate", 0.0))

        active_cats = [c for c in ALL_CATEGORIES if cat_scores[c]]
        if len(active_cats) < 3:
            return

        labels = [c.replace("_", " ").title() for c in active_cats]
        values = [float(np.mean(cat_scores[c])) for c in active_cats]

        num_vars = len(active_cats)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        values_plot = values + values[:1]
        angles_plot = angles + angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection="polar"))

        ax.plot(angles_plot, values_plot, "o-", linewidth=2, color=COLORBLIND_PALETTE[4])
        ax.fill(angles_plot, values_plot, alpha=0.2, color=COLORBLIND_PALETTE[4])
        ax.set_xticks(angles)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title("Capability Radar (PPO)", pad=20, fontsize=11)
        ax.grid(True)

        plt.tight_layout()
        save_figure(fig, str(self.figures_dir / "capability_radar"), formats=["png", "pdf"])

    def plot_per_task_final_scores(self) -> None:
        """Horizontal bar chart of final mean return per task, sorted, color-coded by category."""
        results = self.results
        difficulties = self.difficulties
        # Use medium difficulty or first available
        target_diff = "medium" if "medium" in difficulties else difficulties[0]

        task_scores = []
        for task in self.tasks:
            key = f"{task}_{target_diff}"
            if key in results:
                task_scores.append({
                    "task": task.replace("-v0", ""),
                    "mean_return": results[key].get("mean_return", 0.0),
                    "category": get_task_category(task),
                })

        if not task_scores:
            return

        # Sort by score
        task_scores.sort(key=lambda x: x["mean_return"])

        fig_height = max(4, len(task_scores) * 0.3)
        fig, ax = plt.subplots(figsize=(6, fig_height))

        names = [t["task"] for t in task_scores]
        values = [t["mean_return"] for t in task_scores]
        colors = [CATEGORY_COLORS.get(t["category"], "#999999") for t in task_scores]

        ax.barh(range(len(names)), values, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("Mean Return")
        ax.set_title(f"Per-Task Final Scores ({target_diff})", fontsize=10)
        ax.grid(axis="x", alpha=0.3)

        # Legend for categories
        from matplotlib.patches import Patch

        legend_patches = [
            Patch(facecolor=CATEGORY_COLORS[c], label=c.replace("_", " "))
            for c in ALL_CATEGORIES
            if any(t["category"] == c for t in task_scores)
        ]
        ax.legend(handles=legend_patches, fontsize=6, loc="lower right")

        plt.tight_layout()
        save_figure(
            fig, str(self.figures_dir / "per_task_final_scores"), formats=["png", "pdf"]
        )

    def plot_reward_distribution(self) -> None:
        """Box plot of eval returns across all tasks (at medium difficulty)."""
        results = self.results
        difficulties = self.difficulties
        target_diff = "medium" if "medium" in difficulties else difficulties[0]

        data = []
        labels = []
        for task in sorted(self.tasks):
            key = f"{task}_{target_diff}"
            if key in results:
                returns = results[key].get("eval_returns", [])
                if returns:
                    data.append(returns)
                    labels.append(task.replace("-v0", ""))

        if not data:
            return

        fig_height = max(4, len(data) * 0.3)
        fig, ax = plt.subplots(figsize=(6, fig_height))

        bp = ax.boxplot(
            data,
            vert=False,
            labels=labels,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.5),
        )

        # Color boxes by category
        for i, task in enumerate(sorted(self.tasks)):
            key = f"{task}_{target_diff}"
            if key in results and results[key].get("eval_returns"):
                cat = get_task_category(task)
                color = CATEGORY_COLORS.get(cat, "#cccccc")
                bp["boxes"][i].set_facecolor(color)
                bp["boxes"][i].set_alpha(0.6)

        ax.set_xlabel("Eval Return")
        ax.set_title(f"Return Distribution ({target_diff})", fontsize=10)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        save_figure(
            fig, str(self.figures_dir / "reward_distribution"), formats=["png", "pdf"]
        )
