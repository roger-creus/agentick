"""Agent comparison utilities."""

from __future__ import annotations

from typing import Any

from agentick.leaderboard.result import EvaluationResult


def compare_entries(
    entry_a: EvaluationResult,
    entry_b: EvaluationResult,
) -> dict[str, Any]:
    """
    Pairwise comparison of two entries.

    Args:
        entry_a: First entry
        entry_b: Second entry

    Returns:
        Comparison results
    """
    comparison = {
        "agent_a": entry_a.submission.agent_name,
        "agent_b": entry_b.submission.agent_name,
        "score_a": entry_a.agentick_score,
        "score_b": entry_b.agentick_score,
        "delta": entry_a.agentick_score - entry_b.agentick_score,
        "per_task": {},
    }

    # Per-task comparison
    for task_name in entry_a.per_task.keys():
        if task_name in entry_b.per_task:
            score_a = entry_a.per_task[task_name].get("normalized_score", 0.0)
            score_b = entry_b.per_task[task_name].get("normalized_score", 0.0)

            comparison["per_task"][task_name] = {
                "score_a": score_a,
                "score_b": score_b,
                "delta": score_a - score_b,
                "winner": "a" if score_a > score_b else "b" if score_b > score_a else "tie",
            }

    return comparison


def head_to_head_matrix(entries: list[EvaluationResult]) -> dict[str, dict[str, str]]:
    """
    Compute NxN head-to-head win/tie/loss matrix.

    Args:
        entries: List of entries

    Returns:
        Matrix of comparisons
    """
    matrix = {}

    for entry_a in entries:
        name_a = entry_a.submission.agent_name
        matrix[name_a] = {}

        for entry_b in entries:
            name_b = entry_b.submission.agent_name

            if name_a == name_b:
                matrix[name_a][name_b] = "-"
                continue

            # Compare
            comp = compare_entries(entry_a, entry_b)
            if comp["delta"] > 0:
                matrix[name_a][name_b] = "W"
            elif comp["delta"] < 0:
                matrix[name_a][name_b] = "L"
            else:
                matrix[name_a][name_b] = "T"

    return matrix


def pareto_frontier(
    entries: list[EvaluationResult],
    x_metric: str = "estimated_cost_usd",
    y_metric: str = "agentick_score",
) -> list[str]:
    """
    Compute Pareto frontier (e.g., cost vs score).

    Args:
        entries: List of entries
        x_metric: Metric for x-axis (to minimize)
        y_metric: Metric for y-axis (to maximize)

    Returns:
        List of agent names on Pareto frontier
    """
    # Extract points
    points = []
    for entry in entries:
        x = getattr(entry, x_metric, None) or 0.0
        y = entry.agentick_score

        points.append((entry.submission.agent_name, x, y))

    # Find Pareto frontier
    frontier = []

    for name, x, y in points:
        is_dominated = False

        for other_name, other_x, other_y in points:
            if other_name == name:
                continue

            # Check if other dominates this (lower cost, higher score)
            if other_x <= x and other_y >= y and (other_x < x or other_y > y):
                is_dominated = True
                break

        if not is_dominated:
            frontier.append(name)

    return frontier
