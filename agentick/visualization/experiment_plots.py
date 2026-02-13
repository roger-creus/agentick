"""
Experiment result visualization and plotting.

Provides ExperimentPlotter class for generating all plots from experiment results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


class ExperimentPlotter:
    """
    Generate plots from experiment result directories.

    Usage:
        plotter = ExperimentPlotter("results/my_experiment_20260212/")
        plotter.plot_all()  # Generate all plots
    """

    def __init__(self, result_dir: str | Path):
        """
        Initialize plotter with experiment result directory.

        Args:
            result_dir: Path to experiment results directory
        """
        self.result_dir = Path(result_dir)
        if not self.result_dir.exists():
            raise ValueError(f"Result directory not found: {self.result_dir}")

        self.figures_dir = self.result_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)

        # Load results
        self.results = self._load_results()

    def _load_results(self) -> dict[str, Any]:
        """Load experiment results from directory."""
        results_file = self.result_dir / "results.json"
        if not results_file.exists():
            raise ValueError(f"Results file not found: {results_file}")

        with open(results_file) as f:
            return json.load(f)

    def plot_all(self) -> None:
        """Generate all available plots and save to figures/."""
        print(f"Generating plots for {self.result_dir.name}...")

        self.plot_per_task_scores()
        self.plot_capability_radar()
        self.plot_score_distribution()
        self.plot_episode_length_distribution()
        self.plot_success_rate()

        # Only plot difficulty scaling if multiple difficulties
        if len(self.results.get("difficulties", [])) > 1:
            self.plot_difficulty_scaling()

        print(f"All plots saved to {self.figures_dir}")

    def plot_per_task_scores(self) -> None:
        """Generate bar chart of scores per task."""
        tasks = self.results.get("tasks", [])
        scores = self.results.get("task_scores", {})

        if not tasks or not scores:
            print("Warning: No task scores to plot")
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        task_names = list(scores.keys())
        mean_scores = [scores[t]["mean"] for t in task_names]

        ax.bar(range(len(task_names)), mean_scores, color="steelblue")
        ax.set_xticks(range(len(task_names)))
        ax.set_xticklabels(task_names, rotation=45, ha="right")
        ax.set_ylabel("Normalized Score")
        ax.set_title("Performance per Task")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(self.figures_dir / "per_task_scores.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_capability_radar(self) -> None:
        """Generate capability radar chart."""
        capability_scores = self.results.get("capability_scores", {})

        if not capability_scores:
            print("Warning: No capability scores to plot")
            return

        capabilities = list(capability_scores.keys())
        scores = [capability_scores[c]["mean"] for c in capabilities]

        # Create radar chart
        num_vars = len(capabilities)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        scores_plot = scores + scores[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))

        ax.plot(angles, scores_plot, "o-", linewidth=2, color="steelblue")
        ax.fill(angles, scores_plot, alpha=0.25, color="steelblue")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(capabilities)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Normalized Score", labelpad=30)
        ax.set_title("Capability Radar Chart", pad=20, fontsize=14, fontweight="bold")
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(self.figures_dir / "capability_radar.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_score_distribution(self) -> None:
        """Generate violin/box plot of score distribution per task."""
        tasks = self.results.get("tasks", [])
        scores = self.results.get("task_scores", {})

        if not tasks or not scores:
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        # Get all episode scores per task
        task_names = list(scores.keys())
        score_data = [scores[t].get("episode_scores", [scores[t]["mean"]]) for t in task_names]

        ax.violinplot(score_data, showmeans=True, showmedians=True)
        ax.set_xticks(range(1, len(task_names) + 1))
        ax.set_xticklabels(task_names, rotation=45, ha="right")
        ax.set_ylabel("Score")
        ax.set_title("Score Distribution per Task")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.figures_dir / "score_distribution.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_episode_length_distribution(self) -> None:
        """Generate histogram of episode lengths."""
        tasks = self.results.get("tasks", [])
        scores = self.results.get("task_scores", {})

        if not tasks or not scores:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        all_lengths = []
        for task in scores:
            lengths = scores[task].get("episode_lengths", [])
            all_lengths.extend(lengths)

        if all_lengths:
            ax.hist(all_lengths, bins=30, color="steelblue", alpha=0.7, edgecolor="black")
            ax.set_xlabel("Episode Length (steps)")
            ax.set_ylabel("Frequency")
            ax.set_title("Episode Length Distribution")
            ax.grid(axis="y", alpha=0.3)

            plt.tight_layout()
            plt.savefig(
                self.figures_dir / "episode_length_distribution.png",
                dpi=300,
                bbox_inches="tight",
            )
        plt.close()

    def plot_success_rate(self) -> None:
        """Generate bar chart of success rate per task."""
        tasks = self.results.get("tasks", [])
        scores = self.results.get("task_scores", {})

        if not tasks or not scores:
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        task_names = list(scores.keys())
        success_rates = [scores[t].get("success_rate", 0) for t in task_names]

        ax.bar(range(len(task_names)), success_rates, color="green", alpha=0.7)
        ax.set_xticks(range(len(task_names)))
        ax.set_xticklabels(task_names, rotation=45, ha="right")
        ax.set_ylabel("Success Rate")
        ax.set_title("Success Rate per Task")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.figures_dir / "success_rate.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_difficulty_scaling(self) -> None:
        """Generate plot showing performance vs difficulty level."""
        difficulties = self.results.get("difficulties", [])
        if len(difficulties) <= 1:
            return

        # Group scores by difficulty
        difficulty_scores = {}
        for task, task_data in self.results.get("task_scores", {}).items():
            diff = task_data.get("difficulty", "unknown")
            if diff not in difficulty_scores:
                difficulty_scores[diff] = []
            difficulty_scores[diff].append(task_data["mean"])

        fig, ax = plt.subplots(figsize=(10, 6))

        diff_order = ["easy", "medium", "hard", "expert"]
        ordered_diffs = [d for d in diff_order if d in difficulty_scores]

        means = [np.mean(difficulty_scores[d]) for d in ordered_diffs]
        stds = [np.std(difficulty_scores[d]) for d in ordered_diffs]

        ax.errorbar(ordered_diffs, means, yerr=stds, marker="o", linewidth=2, markersize=8)
        ax.set_xlabel("Difficulty Level")
        ax.set_ylabel("Mean Normalized Score")
        ax.set_title("Performance vs Difficulty")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(self.figures_dir / "difficulty_scaling.png", dpi=300, bbox_inches="tight")
        plt.close()


def compare_experiments(
    result_dirs: list[str | Path],
    output_dir: str | Path = "results/comparison",
) -> None:
    """
    Compare multiple experiments and generate comparison plots.

    Args:
        result_dirs: List of experiment result directories
        output_dir: Output directory for comparison plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Comparing {len(result_dirs)} experiments...")

    # Load all results
    all_results = {}
    for result_dir in result_dirs:
        result_dir = Path(result_dir)
        results_file = result_dir / "results.json"
        if results_file.exists():
            with open(results_file) as f:
                all_results[result_dir.name] = json.load(f)

    # Generate comparison bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    exp_names = list(all_results.keys())
    overall_scores = [all_results[exp].get("overall_score", 0) for exp in exp_names]

    ax.bar(range(len(exp_names)), overall_scores, color="steelblue")
    ax.set_xticks(range(len(exp_names)))
    ax.set_xticklabels(exp_names, rotation=45, ha="right")
    ax.set_ylabel("Overall Score")
    ax.set_title("Experiment Comparison")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "experiment_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Comparison plots saved to {output_dir}")
