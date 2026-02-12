"""Agent comparison utilities with full statistical rigor."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from agentick.analysis.statistics import (
    cliff_delta,
    cohens_d,
    holm_bonferroni,
    mann_whitney_u,
    permutation_test,
    welch_t_test,
)


class ComparisonResult:
    """Container for agent comparison results."""

    def __init__(self, **kwargs: Any):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self) -> str:
        return f"ComparisonResult(n_tasks={self.n_tasks}, winner={self.winner})"


def compare_agents(
    results_a: dict[str, Any], results_b: dict[str, Any], alpha: float = 0.05
) -> ComparisonResult:
    """
    Full statistical comparison of two agents.

    Args:
        results_a: Results dict for agent A (task -> list of returns)
        results_b: Results dict for agent B (task -> list of returns)
        alpha: Significance threshold

    Returns:
        ComparisonResult with comprehensive comparison
    """
    # Get common tasks
    tasks = sorted(set(results_a.keys()) & set(results_b.keys()))

    per_task_comparisons = {}
    p_values = []

    for task in tasks:
        returns_a = np.array(results_a[task])
        returns_b = np.array(results_b[task])

        # Paired comparison (assuming same seeds)
        t_test = welch_t_test(returns_a, returns_b)
        mw_test = mann_whitney_u(returns_a, returns_b)
        perm_test = permutation_test(returns_a, returns_b)
        effect_cohens = cohens_d(returns_a, returns_b)
        effect_cliff = cliff_delta(returns_a, returns_b)

        per_task_comparisons[task] = {
            "mean_a": float(np.mean(returns_a)),
            "mean_b": float(np.mean(returns_b)),
            "diff": float(np.mean(returns_a) - np.mean(returns_b)),
            "t_test": {
                "t": t_test.t_statistic,
                "p": t_test.p_value,
                "significant": t_test.significant,
            },
            "mann_whitney": {
                "u": mw_test.u_statistic,
                "p": mw_test.p_value,
                "significant": mw_test.significant,
            },
            "permutation": {
                "p": perm_test.p_value,
                "significant": perm_test.significant,
            },
            "cohens_d": {
                "d": effect_cohens.d,
                "interpretation": effect_cohens.interpretation,
            },
            "cliff_delta": {
                "delta": effect_cliff.delta,
                "interpretation": effect_cliff.interpretation,
            },
        }

        p_values.append(perm_test.p_value)

    # Multiple comparison correction
    correction = holm_bonferroni(p_values)

    # Aggregate comparison
    all_returns_a = np.concatenate([results_a[task] for task in tasks])
    all_returns_b = np.concatenate([results_b[task] for task in tasks])

    aggregate_t = welch_t_test(all_returns_a, all_returns_b)
    aggregate_effect = cohens_d(all_returns_a, all_returns_b)

    # Win/tie/loss counts
    wins_a = 0
    wins_b = 0
    ties = 0

    for i, task in enumerate(tasks):
        if correction.significant[i]:
            if per_task_comparisons[task]["diff"] > 0:
                wins_a += 1
            else:
                wins_b += 1
        else:
            ties += 1

    # Determine winner
    if wins_a > wins_b:
        winner = "agent_a"
    elif wins_b > wins_a:
        winner = "agent_b"
    else:
        winner = "tie"

    return ComparisonResult(
        n_tasks=len(tasks),
        tasks=tasks,
        per_task=per_task_comparisons,
        p_values=p_values,
        corrected_p_values=correction.adjusted_p_values,
        aggregate_t_test=aggregate_t,
        aggregate_effect=aggregate_effect,
        wins_a=wins_a,
        wins_b=wins_b,
        ties=ties,
        winner=winner,
    )


def compare_multiple(
    results_dict: dict[str, dict[str, Any]], alpha: float = 0.05
) -> dict[str, Any]:
    """
    Multi-agent comparison using Friedman + Nemenyi.

    Args:
        results_dict: Agent name -> (task -> list of returns)
        alpha: Significance threshold

    Returns:
        Dict with Friedman test, Nemenyi post-hoc, and rankings
    """
    agent_names = list(results_dict.keys())
    tasks = sorted(set().union(*[set(r.keys()) for r in results_dict.values()]))

    # Build matrix: tasks x agents
    n_tasks = len(tasks)
    n_agents = len(agent_names)

    means_matrix = np.zeros((n_tasks, n_agents))

    for i, task in enumerate(tasks):
        for j, agent in enumerate(agent_names):
            if task in results_dict[agent]:
                means_matrix[i, j] = np.mean(results_dict[agent][task])
            else:
                means_matrix[i, j] = np.nan

    # Friedman test
    friedman_result = stats.friedmanchisquare(*means_matrix.T)

    # Rankings (lower rank = better performance)
    # For each task, rank agents
    rankings = np.zeros((n_tasks, n_agents))
    for i in range(n_tasks):
        task_scores = means_matrix[i, :]
        # Higher score = lower rank (1 = best)
        rankings[i, :] = stats.rankdata(-task_scores, method="average")

    # Average rank per agent
    avg_ranks = np.mean(rankings, axis=0)

    # Nemenyi critical difference
    # CD = q_alpha * sqrt(k * (k + 1) / (6 * N))
    # where k = number of agents, N = number of tasks
    # q_alpha values from Nemenyi table (approximate)
    q_alpha_values = {0.05: 2.343, 0.01: 2.728}  # for k=3, adjust as needed
    q_alpha = q_alpha_values.get(alpha, 2.343)

    cd = q_alpha * np.sqrt(n_agents * (n_agents + 1) / (6 * n_tasks))

    # Pairwise comparisons
    pairwise = {}
    for i, agent_a in enumerate(agent_names):
        for j, agent_b in enumerate(agent_names):
            if i < j:
                rank_diff = abs(avg_ranks[i] - avg_ranks[j])
                significant = rank_diff > cd
                pairwise[f"{agent_a}_vs_{agent_b}"] = {
                    "rank_diff": float(rank_diff),
                    "significant": significant,
                    "critical_difference": float(cd),
                }

    return {
        "friedman": {
            "statistic": float(friedman_result.statistic),
            "p_value": float(friedman_result.pvalue),
            "significant": friedman_result.pvalue < alpha,
        },
        "rankings": {agent: float(rank) for agent, rank in zip(agent_names, avg_ranks)},
        "critical_difference": float(cd),
        "pairwise": pairwise,
        "n_tasks": n_tasks,
        "n_agents": n_agents,
    }


def ablation_analysis(
    baseline_results: dict[str, Any], ablation_results_dict: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """
    Structured ablation study analysis.

    Args:
        baseline_results: Baseline results (task -> returns)
        ablation_results_dict: Ablation name -> (task -> returns)

    Returns:
        Dict with per-ablation comparisons and ranking
    """
    ablations = {}

    for ablation_name, ablation_results in ablation_results_dict.items():
        comparison = compare_agents(baseline_results, ablation_results)

        # Aggregate effect
        tasks = comparison.tasks
        mean_diff = np.mean([comparison.per_task[task]["diff"] for task in tasks])

        ablations[ablation_name] = {
            "comparison": comparison,
            "mean_diff": float(mean_diff),
            "relative_performance": float(
                mean_diff / np.mean([comparison.per_task[task]["mean_a"] for task in tasks])
            )
            if tasks
            else 0.0,
        }

    # Rank ablations by impact (most negative = biggest drop)
    ranked = sorted(ablations.items(), key=lambda x: x[1]["mean_diff"])

    return {
        "ablations": ablations,
        "ranked_by_impact": [
            {
                "name": name,
                "mean_diff": data["mean_diff"],
                "relative_performance": data["relative_performance"],
            }
            for name, data in ranked
        ],
    }
