"""Cross-agent comparison plots for benchmark results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from agentick.experiments.training_runner import ALL_CATEGORIES, TASK_CATEGORIES
from agentick.visualization.style import (
    get_palette,
    save_figure,
    set_style,
)


class AgentComparisonPlotter:
    """Generate cross-agent comparison plots from multiple result directories.

    Usage:
        plotter = AgentComparisonPlotter([
            Path("results/random_20240101/"),
            Path("results/claude_20240101/"),
            Path("results/gpt4o_20240101/"),
        ])
        plotter.plot_all()
    """

    def __init__(self, result_dirs: list[Path], output_dir: Path | None = None):
        self.result_dirs = [Path(d) for d in result_dirs]
        self.output_dir = output_dir or Path("results/comparison")
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        self.agents: dict[str, dict[str, Any]] = {}
        for d in self.result_dirs:
            data = self._load_result(d)
            if data is not None:
                self.agents[data["name"]] = data

    def _load_result(self, result_dir: Path) -> dict[str, Any] | None:
        """Load summary and per-task results from a run directory."""
        summary_path = result_dir / "summary.json"
        config_path = result_dir / "config.yaml"
        if not summary_path.exists():
            return None

        with open(summary_path) as f:
            summary = json.load(f)

        # Try to get agent name from config
        name = result_dir.name
        if config_path.exists():
            import yaml

            with open(config_path) as f:
                config = yaml.safe_load(f)
            name = config.get("name", name)

        # Load per-task results
        per_task: dict[str, Any] = {}
        per_task_dir = result_dir / "per_task"
        if per_task_dir.exists():
            for task_dir in per_task_dir.iterdir():
                if task_dir.is_dir():
                    metrics_path = task_dir / "metrics.json"
                    if metrics_path.exists():
                        with open(metrics_path) as f:
                            per_task[task_dir.name] = json.load(f)

        return {
            "name": name,
            "dir": result_dir,
            "summary": summary,
            "per_task": per_task,
        }

    def plot_all(self) -> None:
        """Generate all comparison plots."""
        if len(self.agents) < 2:
            print("Need at least 2 agent results to compare.")
            return

        set_style("paper_double_column")
        print(f"Generating comparison plots for {len(self.agents)} agents...")

        self.plot_success_rate_comparison()
        self.plot_radar_overlay()
        self.plot_difficulty_scaling_comparison()
        self.plot_agent_heatmap()
        self.plot_cost_efficiency()

        print(f"Comparison plots saved to {self.figures_dir}")

    def plot_success_rate_comparison(self) -> None:
        """Grouped bar chart: tasks x agents, showing success rate."""
        # Collect all tasks across agents
        all_tasks = set()
        for data in self.agents.values():
            all_tasks.update(data["per_task"].keys())
        tasks = sorted(all_tasks)

        if not tasks:
            return

        agent_names = list(self.agents.keys())
        n_agents = len(agent_names)
        n_tasks = len(tasks)

        bar_width = 0.8 / n_agents
        x = np.arange(n_tasks)

        fig_height = max(6, n_tasks * 0.35)
        fig, ax = plt.subplots(figsize=(8, fig_height))

        palette = get_palette(n_agents)
        for i, agent_name in enumerate(agent_names):
            data = self.agents[agent_name]
            rates = []
            for task in tasks:
                task_data = data["per_task"].get(task, {})
                agg = task_data.get("aggregate_metrics", {})
                rates.append(agg.get("success_rate", 0.0))

            positions = x + i * bar_width - (n_agents - 1) * bar_width / 2
            ax.barh(
                positions,
                rates,
                height=bar_width * 0.9,
                label=agent_name,
                color=palette[i % len(palette)],
                edgecolor="white",
                linewidth=0.3,
            )

        ax.set_yticks(x)
        ax.set_yticklabels([t.replace("-v0", "") for t in tasks], fontsize=7)
        ax.set_xlabel("Success Rate")
        ax.set_xlim(0, 1.05)
        ax.set_title("Success Rate Comparison")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        save_figure(fig, str(self.figures_dir / "success_rate_comparison"))

    def plot_radar_overlay(self) -> None:
        """Overlaid radar charts per agent (10 category spokes)."""
        # Compute category-level success rates per agent
        agent_cat_scores: dict[str, dict[str, float]] = {}
        for agent_name, data in self.agents.items():
            cat_scores: dict[str, list[float]] = {c: [] for c in ALL_CATEGORIES}
            for task, task_data in data["per_task"].items():
                cat = TASK_CATEGORIES.get(task, "unknown")
                if cat in cat_scores:
                    agg = task_data.get("aggregate_metrics", {})
                    cat_scores[cat].append(agg.get("success_rate", 0.0))
            agent_cat_scores[agent_name] = {
                c: float(np.mean(scores)) if scores else 0.0 for c, scores in cat_scores.items()
            }

        active_cats = [
            c
            for c in ALL_CATEGORIES
            if any(agent_cat_scores[a].get(c, 0) > 0 for a in agent_cat_scores)
        ]
        if len(active_cats) < 3:
            return

        labels = [c.replace("_", " ").title() for c in active_cats]
        num_vars = len(active_cats)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles_plot = angles + angles[:1]

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(projection="polar"))
        palette = get_palette(len(self.agents))

        for i, (agent_name, cat_scores) in enumerate(agent_cat_scores.items()):
            values = [cat_scores.get(c, 0.0) for c in active_cats]
            values_plot = values + values[:1]
            color = palette[i % len(palette)]
            ax.plot(
                angles_plot,
                values_plot,
                "o-",
                linewidth=1.5,
                label=agent_name,
                color=color,
            )
            ax.fill(angles_plot, values_plot, alpha=0.1, color=color)

        ax.set_xticks(angles)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title("Capability Radar Overlay", pad=20, fontsize=11)
        ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)

        plt.tight_layout()
        save_figure(fig, str(self.figures_dir / "radar_overlay"))

    def plot_difficulty_scaling_comparison(self) -> None:
        """Line plot per agent: x=difficulty, y=mean success rate."""
        difficulties = ["easy", "medium", "hard", "expert"]

        fig, ax = plt.subplots(figsize=(6, 4))
        palette = get_palette(len(self.agents))

        for i, (agent_name, data) in enumerate(self.agents.items()):
            means = []
            for diff in difficulties:
                rates = []
                for task_data in data["per_task"].values():
                    diff_data = task_data.get("per_difficulty", {}).get(diff, {})
                    metrics = diff_data.get("metrics", {})
                    if "success_rate" in metrics:
                        rates.append(metrics["success_rate"])
                means.append(float(np.mean(rates)) if rates else 0.0)

            color = palette[i % len(palette)]
            ax.plot(
                difficulties,
                means,
                marker="o",
                linewidth=2,
                markersize=6,
                label=agent_name,
                color=color,
            )

        ax.set_xlabel("Difficulty")
        ax.set_ylabel("Mean Success Rate")
        ax.set_title("Difficulty Scaling Comparison")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_figure(fig, str(self.figures_dir / "difficulty_scaling_comparison"))

    def plot_agent_heatmap(self) -> None:
        """Heatmap: agents x tasks, color=success rate."""
        agent_names = list(self.agents.keys())
        all_tasks = set()
        for data in self.agents.values():
            all_tasks.update(data["per_task"].keys())
        tasks = sorted(all_tasks)

        if not tasks or not agent_names:
            return

        matrix = np.full((len(agent_names), len(tasks)), np.nan)
        for i, agent_name in enumerate(agent_names):
            data = self.agents[agent_name]
            for j, task in enumerate(tasks):
                task_data = data["per_task"].get(task, {})
                agg = task_data.get("aggregate_metrics", {})
                if "success_rate" in agg:
                    matrix[i, j] = agg["success_rate"]

        fig_width = max(8, len(tasks) * 0.4)
        fig_height = max(3, len(agent_names) * 0.5 + 1)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        im = ax.imshow(
            matrix,
            aspect="auto",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )

        ax.set_xticks(range(len(tasks)))
        ax.set_xticklabels(
            [t.replace("-v0", "") for t in tasks], rotation=45, ha="right", fontsize=6
        )
        ax.set_yticks(range(len(agent_names)))
        ax.set_yticklabels(agent_names, fontsize=8)

        # Annotate cells
        for i in range(len(agent_names)):
            for j in range(len(tasks)):
                val = matrix[i, j]
                if not np.isnan(val):
                    color = "white" if val < 0.4 or val > 0.8 else "black"
                    ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=5, color=color)

        cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
        cbar.set_label("Success Rate")
        ax.set_title("Agent x Task Success Rate Heatmap", fontsize=10)

        plt.tight_layout()
        save_figure(fig, str(self.figures_dir / "agent_heatmap"))

    def plot_cost_efficiency(self) -> None:
        """Scatter: total tokens vs success rate (API agents only)."""
        points = []
        for agent_name, data in self.agents.items():
            summary = data["summary"]
            agent_stats = summary.get("agent_stats", {})
            total_tokens = agent_stats.get("total_tokens", 0)
            if total_tokens == 0:
                continue
            success_rate = summary.get("success_rate", 0.0)
            points.append(
                {
                    "name": agent_name,
                    "tokens": total_tokens,
                    "success_rate": success_rate,
                }
            )

        if len(points) < 2:
            return

        fig, ax = plt.subplots(figsize=(6, 4))
        palette = get_palette(len(points))

        for i, pt in enumerate(points):
            color = palette[i % len(palette)]
            ax.scatter(
                pt["tokens"],
                pt["success_rate"],
                s=100,
                color=color,
                edgecolors="black",
                linewidths=0.5,
                zorder=3,
            )
            ax.annotate(
                pt["name"],
                (pt["tokens"], pt["success_rate"]),
                fontsize=7,
                xytext=(5, 5),
                textcoords="offset points",
            )

        ax.set_xlabel("Total Tokens")
        ax.set_ylabel("Success Rate")
        ax.set_title("Cost Efficiency: Tokens vs Success Rate")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_figure(fig, str(self.figures_dir / "cost_efficiency"))
