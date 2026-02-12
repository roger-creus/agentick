"""Core plotting functions for publication-quality figures."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from agentick.visualization.style import (
    get_agent_color,
    get_agent_linestyle,
    get_agent_marker,
    save_figure,
    set_style,
)


def plot_bar_comparison(
    results_dict: dict[str, dict[str, Any]],
    metric: str = "mean_return",
    output_path: str | None = None,
    style: str = "paper_double_column",
    **kwargs: Any,
) -> Any:
    """
    Grouped bar chart comparing agents across tasks.

    Args:
        results_dict: Agent name -> (task -> {metric: value, ci_lower, ci_upper})
        metric: Metric to plot
        output_path: Output path (without extension)
        style: Plot style
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib figure
    """
    set_style(style)

    # Get tasks and agents
    agents = list(results_dict.keys())
    tasks = sorted(set().union(*[set(r.keys()) for r in results_dict.values()]))

    # Build data matrix
    n_tasks = len(tasks)
    n_agents = len(agents)

    means = np.zeros((n_tasks, n_agents))
    errors = np.zeros((n_tasks, n_agents))

    for i, task in enumerate(tasks):
        for j, agent in enumerate(agents):
            if task in results_dict[agent]:
                data = results_dict[agent][task]
                means[i, j] = data.get(metric, 0)
                # Error bar is half the CI width
                ci_lower = data.get("ci_lower", means[i, j])
                ci_upper = data.get("ci_upper", means[i, j])
                errors[i, j] = (ci_upper - ci_lower) / 2

    # Create figure
    fig, ax = plt.subplots()

    x = np.arange(n_tasks)
    width = 0.8 / n_agents

    for j, agent in enumerate(agents):
        offset = (j - n_agents / 2 + 0.5) * width
        color = get_agent_color(agent)

        ax.bar(
            x + offset,
            means[:, j],
            width,
            yerr=errors[:, j],
            label=agent,
            color=color,
            capsize=3,
        )

    ax.set_xlabel("Task")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=45, ha="right")
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path)

    return fig


def plot_capability_radar(
    results_dict: dict[str, dict[str, float]],
    output_path: str | None = None,
    style: str = "paper_double_column",
    **kwargs: Any,
) -> Any:
    """
    Radar/spider chart showing capability profile (SIGNATURE FIGURE).

    Args:
        results_dict: Agent name -> (capability -> score)
        output_path: Output path
        style: Plot style
        **kwargs: Additional arguments

    Returns:
        Matplotlib figure
    """
    set_style(style)

    agents = list(results_dict.keys())
    capabilities = sorted(set().union(*[set(r.keys()) for r in results_dict.values()]))

    n_capabilities = len(capabilities)

    # Angles for each capability
    angles = np.linspace(0, 2 * np.pi, n_capabilities, endpoint=False).tolist()
    angles += angles[:1]  # Close the plot

    fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))

    for agent in agents:
        values = [results_dict[agent].get(cap, 0) for cap in capabilities]
        values += values[:1]  # Close the plot

        color = get_agent_color(agent)
        linestyle = get_agent_linestyle(agent)

        ax.plot(angles, values, linestyle, linewidth=2, label=agent, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(capabilities)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)

    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path)

    return fig


def plot_learning_curves(
    curves_dict: dict[str, np.ndarray],
    output_path: str | None = None,
    style: str = "paper_double_column",
    xlabel: str = "Episodes",
    ylabel: str = "Return",
    **kwargs: Any,
) -> Any:
    """
    Learning curves with CI bands.

    Args:
        curves_dict: Agent name -> learning curve array or dict with mean/ci
        output_path: Output path
        style: Plot style
        xlabel: X-axis label
        ylabel: Y-axis label
        **kwargs: Additional arguments

    Returns:
        Matplotlib figure
    """
    set_style(style)

    fig, ax = plt.subplots()

    for agent, data in curves_dict.items():
        color = get_agent_color(agent)
        linestyle = get_agent_linestyle(agent)

        if isinstance(data, dict):
            mean_curve = data["mean_curve"]
            ci_lower = data.get("ci_lower")
            ci_upper = data.get("ci_upper")

            x = np.arange(len(mean_curve))

            ax.plot(x, mean_curve, linestyle, linewidth=2, label=agent, color=color)

            if ci_lower is not None and ci_upper is not None:
                ax.fill_between(x, ci_lower, ci_upper, alpha=0.2, color=color)
        else:
            # Simple array
            x = np.arange(len(data))
            ax.plot(x, data, linestyle, linewidth=2, label=agent, color=color)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path)

    return fig


def plot_normalized_scores(
    results_dict: dict[str, dict[str, float]],
    output_path: str | None = None,
    style: str = "paper_double_column",
    **kwargs: Any,
) -> Any:
    """
    Normalized scores [0,1] bar chart across all tasks.

    Args:
        results_dict: Agent name -> (task -> normalized_score)
        output_path: Output path
        style: Plot style
        **kwargs: Additional arguments

    Returns:
        Matplotlib figure
    """
    return plot_bar_comparison(
        results_dict, metric="normalized_score", output_path=output_path, style=style, **kwargs
    )


def plot_difficulty_scaling(
    results_dict: dict[str, dict[str, float]],
    output_path: str | None = None,
    style: str = "paper_double_column",
    **kwargs: Any,
) -> Any:
    """
    Performance vs difficulty level.

    Args:
        results_dict: Agent name -> (difficulty -> score)
        output_path: Output path
        style: Plot style
        **kwargs: Additional arguments

    Returns:
        Matplotlib figure
    """
    set_style(style)

    fig, ax = plt.subplots()

    difficulties = ["easy", "medium", "hard", "expert"]
    difficulty_indices = {d: i for i, d in enumerate(difficulties)}

    for agent, difficulty_scores in results_dict.items():
        color = get_agent_color(agent)
        marker = get_agent_marker(agent)
        linestyle = get_agent_linestyle(agent)

        x = []
        y = []
        for diff, score in difficulty_scores.items():
            if diff in difficulty_indices:
                x.append(difficulty_indices[diff])
                y.append(score)

        if x:
            ax.plot(
                x, y, linestyle, marker=marker, linewidth=2, markersize=8, label=agent, color=color
            )

    ax.set_xlabel("Difficulty")
    ax.set_ylabel("Score")
    ax.set_xticks(list(range(len(difficulties))))
    ax.set_xticklabels(difficulties)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path)

    return fig


def plot_aggregate_table(
    results_dict: dict[str, dict[str, float]],
    output_path: str | None = None,
    style: str = "paper_double_column",
    **kwargs: Any,
) -> Any:
    """
    Combined bar chart of aggregate scores per agent.

    Args:
        results_dict: Agent name -> aggregate score
        output_path: Output path
        style: Plot style
        **kwargs: Additional arguments

    Returns:
        Matplotlib figure
    """
    set_style(style)

    fig, ax = plt.subplots()

    agents = list(results_dict.keys())
    scores = [results_dict[agent] for agent in agents]

    colors = [get_agent_color(agent) for agent in agents]

    x = np.arange(len(agents))
    ax.bar(x, scores, color=colors)

    ax.set_xlabel("Agent")
    ax.set_ylabel("Aggregate Score")
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path)

    return fig


def plot_capability_heatmap(
    results_dict: dict[str, dict[str, float]],
    output_path: str | None = None,
    style: str = "paper_double_column",
    **kwargs: Any,
) -> Any:
    """
    Tasks × Agents heatmap showing normalized scores.

    Args:
        results_dict: Agent name -> (task -> normalized score)
        output_path: Output path
        style: Plot style
        **kwargs: Additional arguments

    Returns:
        Matplotlib figure
    """
    set_style(style)

    import seaborn as sns

    # Get all tasks and agents
    agents = list(results_dict.keys())
    all_tasks = sorted(set().union(*[set(r.keys()) for r in results_dict.values()]))

    # Build matrix
    matrix = np.zeros((len(all_tasks), len(agents)))
    for i, task in enumerate(all_tasks):
        for j, agent in enumerate(agents):
            if task in results_dict[agent]:
                matrix[i, j] = results_dict[agent][task]
            else:
                matrix[i, j] = np.nan

    # Create heatmap
    fig, ax = plt.subplots(figsize=(len(agents) * 0.8 + 2, len(all_tasks) * 0.3 + 2))

    sns.heatmap(
        matrix,
        xticklabels=agents,
        yticklabels=all_tasks,
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Normalized Score"},
        ax=ax,
    )

    ax.set_xlabel("Agent")
    ax.set_ylabel("Task")

    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path)

    return fig


def plot_sample_efficiency(
    curves_dict: dict[str, dict[str, Any]],
    threshold: float = 0.8,
    output_path: str | None = None,
    style: str = "paper_double_column",
    **kwargs: Any,
) -> Any:
    """
    Steps to reach threshold performance.

    Args:
        curves_dict: Agent name -> convergence data with 'convergence_episode'
        threshold: Performance threshold
        output_path: Output path
        style: Plot style
        **kwargs: Additional arguments

    Returns:
        Matplotlib figure
    """
    set_style(style)

    fig, ax = plt.subplots()

    agents = []
    episodes = []
    colors_list = []

    for agent, data in curves_dict.items():
        conv_episode = data.get("convergence_episode")
        if conv_episode is not None:
            agents.append(agent)
            episodes.append(conv_episode)
            colors_list.append(get_agent_color(agent))

    if not agents:
        # No convergence data
        return fig

    x = np.arange(len(agents))
    ax.bar(x, episodes, color=colors_list)

    ax.set_xlabel("Agent")
    ax.set_ylabel(f"Episodes to {threshold:.0%} Performance")
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path)

    return fig


def plot_return_distribution(
    results_dict: dict[str, list[float]],
    task: str,
    output_path: str | None = None,
    style: str = "paper_double_column",
    plot_type: str = "violin",
    **kwargs: Any,
) -> Any:
    """
    Violin or box plot of per-episode returns.

    Args:
        results_dict: Agent name -> list of episode returns
        task: Task name (for title)
        output_path: Output path
        style: Plot style
        plot_type: 'violin' or 'box'
        **kwargs: Additional arguments

    Returns:
        Matplotlib figure
    """
    set_style(style)

    fig, ax = plt.subplots()

    agents = list(results_dict.keys())
    data = [results_dict[agent] for agent in agents]

    if plot_type == "violin":
        parts = ax.violinplot(data, positions=range(len(agents)), showmeans=True, showmedians=True)
        for i, agent in enumerate(agents):
            color = get_agent_color(agent)
            parts["bodies"][i].set_facecolor(color)
            parts["bodies"][i].set_alpha(0.7)
    else:  # box plot
        bp = ax.boxplot(data, labels=agents, patch_artist=True)
        for i, (agent, patch) in enumerate(zip(agents, bp["boxes"])):
            color = get_agent_color(agent)
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    ax.set_xlabel("Agent")
    ax.set_ylabel("Return")
    ax.set_title(f"Return Distribution: {task}")
    ax.set_xticks(range(len(agents)))
    ax.set_xticklabels(agents, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path)

    return fig


def plot_episode_length_distribution(
    results_dict: dict[str, list[int]],
    task: str,
    output_path: str | None = None,
    style: str = "paper_double_column",
    **kwargs: Any,
) -> Any:
    """
    Distribution of episode lengths across agents.

    Args:
        results_dict: Agent name -> list of episode lengths
        task: Task name
        output_path: Output path
        style: Plot style
        **kwargs: Additional arguments

    Returns:
        Matplotlib figure
    """
    set_style(style)

    fig, ax = plt.subplots()

    agents = list(results_dict.keys())
    data = [results_dict[agent] for agent in agents]

    bp = ax.boxplot(data, labels=agents, patch_artist=True)
    for i, (agent, patch) in enumerate(zip(agents, bp["boxes"])):
        color = get_agent_color(agent)
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel("Agent")
    ax.set_ylabel("Episode Length (steps)")
    ax.set_title(f"Episode Length Distribution: {task}")
    ax.set_xticks(range(len(agents)))
    ax.set_xticklabels(agents, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path)

    return fig


def plot_worldmodel_results(
    results_dict: dict[str, dict[str, float]],
    output_path: str | None = None,
    style: str = "paper_double_column",
    **kwargs: Any,
) -> Any:
    """
    Grouped bar chart of world model evaluation metrics.

    Args:
        results_dict: Agent name -> {metric_name: value}
        output_path: Output path
        style: Plot style
        **kwargs: Additional arguments

    Returns:
        Matplotlib figure
    """
    set_style(style)

    fig, ax = plt.subplots()

    agents = list(results_dict.keys())
    metrics = ["prediction_acc", "planning_success", "change_detection_f1", "counterfactual_acc"]

    n_agents = len(agents)
    n_metrics = len(metrics)

    x = np.arange(n_metrics)
    width = 0.8 / n_agents

    for i, agent in enumerate(agents):
        offset = (i - n_agents / 2 + 0.5) * width
        values = [results_dict[agent].get(m, 0) for m in metrics]
        color = get_agent_color(agent)

        ax.bar(x + offset, values, width, label=agent, color=color)

    ax.set_xlabel("World Model Metric")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Prediction", "Planning", "Change Detection", "Counterfactual"], rotation=45, ha="right"
    )
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path)

    return fig


def plot_critical_difference(
    results_dict: dict[str, float],
    critical_difference: float,
    output_path: str | None = None,
    style: str = "paper_double_column",
    **kwargs: Any,
) -> Any:
    """
    Critical difference diagram (Nemenyi post-hoc test visualization).

    Args:
        results_dict: Agent name -> average rank
        critical_difference: CD threshold from Nemenyi test
        output_path: Output path
        style: Plot style
        **kwargs: Additional arguments

    Returns:
        Matplotlib figure
    """
    set_style(style)

    fig, ax = plt.subplots(figsize=(10, 4))

    # Sort agents by rank
    sorted_agents = sorted(results_dict.items(), key=lambda x: x[1])
    agents = [a for a, _ in sorted_agents]
    ranks = [r for _, r in sorted_agents]

    # Plot horizontal lines for ranks
    y_positions = range(len(agents))
    ax.barh(y_positions, ranks, color=[get_agent_color(a) for a in agents], alpha=0.7)

    # Draw CD bars
    ax.axvline(
        x=critical_difference, color="red", linestyle="--", label=f"CD={critical_difference:.2f}"
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(agents)
    ax.set_xlabel("Average Rank")
    ax.set_title("Critical Difference Diagram")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path)

    return fig


def plot_capability_correlation(
    per_task_scores: dict[str, float],
    task_capability_map: dict[str, str],
    output_path: str | None = None,
    style: str = "paper_double_column",
    **kwargs: Any,
) -> Any:
    """
    Scatter matrix showing correlation between capability scores.

    Args:
        per_task_scores: Task name -> score
        task_capability_map: Task name -> capability
        output_path: Output path
        style: Plot style
        **kwargs: Additional arguments

    Returns:
        Matplotlib figure
    """
    set_style(style)

    # Group tasks by capability
    capability_scores: dict[str, list[float]] = {}
    for task, score in per_task_scores.items():
        cap = task_capability_map.get(task)
        if cap:
            if cap not in capability_scores:
                capability_scores[cap] = []
            capability_scores[cap].append(score)

    # Compute mean per capability
    cap_means = {cap: np.mean(scores) for cap, scores in capability_scores.items()}

    # Create scatter matrix would require pandas, just do simple correlation plot
    fig, ax = plt.subplots()

    capabilities = list(cap_means.keys())
    values = list(cap_means.values())

    colors = [get_agent_color(cap) for cap in capabilities]
    ax.scatter(range(len(capabilities)), values, c=colors, s=100, alpha=0.7)

    ax.set_xlabel("Capability")
    ax.set_ylabel("Mean Score")
    ax.set_xticks(range(len(capabilities)))
    ax.set_xticklabels(capabilities, rotation=45, ha="right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path)

    return fig
