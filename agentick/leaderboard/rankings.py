"""Ranking computation and leaderboard generation."""

from __future__ import annotations

from typing import Any

from agentick.leaderboard.result import EvaluationResult


def compute_rankings(
    entries: list[EvaluationResult],
    sort_by: str = "agentick_score",
) -> list[dict[str, Any]]:
    """
    Compute rankings from evaluation results.

    Args:
        entries: List of evaluation results
        sort_by: Metric to sort by

    Returns:
        List of ranked entries with rank annotations
    """
    if not entries:
        return []

    # Extract scores
    ranked_entries = []

    for entry in entries:
        score = entry.agentick_score
        ci_lower, ci_upper = entry.agentick_score_ci

        ranked_entries.append(
            {
                "rank": 0,  # Will be set below
                "agent_name": entry.submission.agent_name,
                "author": entry.submission.author,
                "agent_type": entry.submission.agent_type,
                "observation_mode": entry.submission.observation_mode,
                "score": score,
                "score_ci_lower": ci_lower,
                "score_ci_upper": ci_upper,
                "per_capability": entry.per_capability,
                "total_api_calls": entry.total_api_calls,
                "estimated_cost_usd": entry.estimated_cost_usd,
                "open_weights": entry.submission.open_weights,
            }
        )

    # Sort by score (descending)
    ranked_entries.sort(key=lambda x: x["score"], reverse=True)

    # Assign ranks
    for i, entry in enumerate(ranked_entries):
        entry["rank"] = i + 1

    return ranked_entries


def compute_per_capability_rankings(
    entries: list[EvaluationResult],
) -> dict[str, list[dict[str, Any]]]:
    """
    Compute per-capability rankings.

    Args:
        entries: List of evaluation results

    Returns:
        Dictionary mapping capability to ranked entries
    """
    # Extract all capabilities
    all_capabilities = set()
    for entry in entries:
        all_capabilities.update(entry.per_capability.keys())

    # Compute rankings per capability
    rankings = {}

    for capability in all_capabilities:
        cap_entries = []

        for entry in entries:
            if capability in entry.per_capability:
                cap_data = entry.per_capability[capability]
                score = cap_data.get("mean_normalized_score", 0.0)

                cap_entries.append(
                    {
                        "rank": 0,
                        "agent_name": entry.submission.agent_name,
                        "score": score,
                    }
                )

        # Sort and rank
        cap_entries.sort(key=lambda x: x["score"], reverse=True)
        for i, e in enumerate(cap_entries):
            e["rank"] = i + 1

        rankings[capability] = cap_entries

    return rankings


def is_significantly_better(
    score_a: float,
    ci_a: tuple[float, float],
    score_b: float,
    ci_b: tuple[float, float],
) -> bool:
    """
    Check if score A is significantly better than score B.

    Args:
        score_a: Score A
        ci_a: Confidence interval for A
        score_b: Score B
        ci_b: Confidence interval for B

    Returns:
        True if A is significantly better than B
    """
    # Non-overlapping CIs
    if ci_a[0] > ci_b[1]:
        return True

    return False
