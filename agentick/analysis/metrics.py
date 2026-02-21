"""Extended metrics for agent evaluation with statistical rigor."""

from __future__ import annotations

from typing import Any

import numpy as np

from agentick.analysis.statistics import bootstrap_ci


def normalized_score(
    returns: np.ndarray | list[float],
    optimal: float,
    random_baseline: float,
    ci: float = 0.95,
) -> dict[str, Any]:
    """
    Compute normalized score [0, 1] with bootstrap CI.

    Score = (mean_return - random) / (optimal - random)

    Args:
        returns: Agent returns
        optimal: Optimal score
        random_baseline: Random agent baseline
        ci: Confidence level

    Returns:
        Dict with score, ci_lower, ci_upper
    """
    returns = np.asarray(returns)

    # Compute normalized scores for each episode
    normalized = (returns - random_baseline) / (optimal - random_baseline)
    normalized = np.clip(normalized, 0, 1)

    # Bootstrap CI
    result = bootstrap_ci(normalized, statistic_fn=np.mean, ci=ci)

    return {
        "score": result.point_estimate,
        "ci_lower": result.ci_lower,
        "ci_upper": result.ci_upper,
        "ci_level": ci,
    }


def agentick_score(
    per_task_scores: dict[str, float], weights: dict[str, float] | None = None, ci: float = 0.95
) -> dict[str, Any]:
    """
    Compute aggregate Agentick score across tasks with CI.

    Args:
        per_task_scores: Task name -> normalized score
        weights: Optional task weights (default: uniform)
        ci: Confidence level

    Returns:
        Dict with aggregate_score, ci_lower, ci_upper
    """
    tasks = list(per_task_scores.keys())
    scores = np.array([per_task_scores[task] for task in tasks])

    if weights is None:
        # Uniform weights
        weights_array = np.ones(len(tasks)) / len(tasks)
    else:
        weights_array = np.array([weights.get(task, 1.0) for task in tasks])
        weights_array = weights_array / np.sum(weights_array)

    # Weighted mean
    aggregate = np.sum(scores * weights_array)

    # Bootstrap CI (resample tasks)
    def weighted_mean_fn(indices):
        return np.sum(scores[indices] * weights_array[indices]) / np.sum(weights_array[indices])

    result = bootstrap_ci(np.arange(len(tasks)), statistic_fn=weighted_mean_fn, ci=ci)

    return {
        "aggregate_score": float(aggregate),
        "ci_lower": result.ci_lower,
        "ci_upper": result.ci_upper,
        "ci_level": ci,
    }


def capability_profile(
    per_task_scores: dict[str, float],
    task_capability_map: dict[str, str],
    ci: float = 0.95,
) -> dict[str, dict[str, Any]]:
    """
    Compute per-capability scores with CIs.

    Args:
        per_task_scores: Task name -> normalized score
        task_capability_map: Task name -> capability name
        ci: Confidence level

    Returns:
        Dict of capability -> {score, ci_lower, ci_upper}
    """
    # Group tasks by capability
    capability_tasks: dict[str, list[str]] = {}
    for task, capability in task_capability_map.items():
        if task in per_task_scores:
            if capability not in capability_tasks:
                capability_tasks[capability] = []
            capability_tasks[capability].append(task)

    # Compute per-capability scores
    profile = {}
    for capability, tasks in capability_tasks.items():
        scores = np.array([per_task_scores[task] for task in tasks])

        result = bootstrap_ci(scores, statistic_fn=np.mean, ci=ci)

        profile[capability] = {
            "score": result.point_estimate,
            "ci_lower": result.ci_lower,
            "ci_upper": result.ci_upper,
            "n_tasks": len(tasks),
        }

    return profile


def sample_efficiency_curve(returns_over_time: np.ndarray) -> dict[str, Any]:
    """
    Compute sample efficiency (AUC of learning curve).

    Args:
        returns_over_time: Array of returns over time (episodes or steps)

    Returns:
        Dict with auc, normalized_auc
    """
    returns_over_time = np.asarray(returns_over_time)

    # Compute AUC using trapezoidal rule
    auc = np.trapz(returns_over_time)

    # Normalize by max possible AUC (if always at max return)
    max_return = np.max(returns_over_time)
    max_auc = max_return * len(returns_over_time)

    normalized_auc = auc / max_auc if max_auc > 0 else 0.0

    return {
        "auc": float(auc),
        "normalized_auc": float(normalized_auc),
        "max_return": float(max_return),
        "n_points": len(returns_over_time),
    }


def action_efficiency(agent_steps: int, optimal_steps: int) -> dict[str, Any]:
    """
    Action efficiency (ARC-AGI-3 style scoring).

    Efficiency = optimal_steps / agent_steps

    Args:
        agent_steps: Number of steps agent took
        optimal_steps: Optimal number of steps

    Returns:
        Dict with efficiency, excess_steps
    """
    efficiency = optimal_steps / agent_steps if agent_steps > 0 else 0.0
    excess_steps = agent_steps - optimal_steps

    return {
        "efficiency": float(efficiency),
        "excess_steps": int(excess_steps),
        "agent_steps": int(agent_steps),
        "optimal_steps": int(optimal_steps),
    }


def exploration_efficiency(states_visited: int, total_states: int, n_steps: int) -> dict[str, Any]:
    """
    Exploration efficiency: coverage per step.

    Args:
        states_visited: Number of unique states visited
        total_states: Total number of states in environment
        n_steps: Number of steps taken

    Returns:
        Dict with coverage, coverage_rate, exploration_efficiency
    """
    coverage = states_visited / total_states if total_states > 0 else 0.0
    coverage_rate = states_visited / n_steps if n_steps > 0 else 0.0

    # Exploration efficiency: how quickly agent explores
    exploration_efficiency = coverage_rate

    return {
        "coverage": float(coverage),
        "coverage_rate": float(coverage_rate),
        "exploration_efficiency": float(exploration_efficiency),
        "states_visited": int(states_visited),
        "total_states": int(total_states),
    }


def consistency_score(per_seed_returns: list[float]) -> dict[str, Any]:
    """
    Consistency score: stability across seeds.

    Lower coefficient of variation = higher consistency.

    Args:
        per_seed_returns: Returns for each seed

    Returns:
        Dict with consistency, mean, std, cv
    """
    per_seed_returns = np.asarray(per_seed_returns)

    mean = np.mean(per_seed_returns)
    std = np.std(per_seed_returns)
    cv = std / mean if mean != 0 else float("inf")

    # Consistency score: inverse of CV, capped at 1
    consistency = min(1.0 / (1.0 + cv), 1.0)

    return {
        "consistency": float(consistency),
        "mean": float(mean),
        "std": float(std),
        "cv": float(cv),
    }


def difficulty_scaling(per_difficulty_scores: dict[str, float]) -> dict[str, Any]:
    """
    Difficulty scaling: how gracefully performance degrades.

    Args:
        per_difficulty_scores: Difficulty level -> score

    Returns:
        Dict with scaling_slope, scaling_r2
    """
    difficulties = ["easy", "medium", "hard", "expert"]
    difficulty_indices = {d: i for i, d in enumerate(difficulties)}

    # Get scores in order
    x = []
    y = []
    for diff, score in per_difficulty_scores.items():
        if diff in difficulty_indices:
            x.append(difficulty_indices[diff])
            y.append(score)

    x = np.array(x)
    y = np.array(y)

    if len(x) < 2:
        return {"scaling_slope": None, "scaling_r2": None}

    # Linear fit
    coeffs = np.polyfit(x, y, 1)
    slope = coeffs[0]

    # R^2
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "scaling_slope": float(slope),
        "scaling_r2": float(r2),
    }


def transfer_score(
    train_task_scores: dict[str, float], test_task_scores: dict[str, float]
) -> dict[str, Any]:
    """
    Transfer score: generalization from train to test tasks.

    Args:
        train_task_scores: Training task scores
        test_task_scores: Test task scores

    Returns:
        Dict with transfer_ratio, train_mean, test_mean
    """
    train_mean = np.mean(list(train_task_scores.values()))
    test_mean = np.mean(list(test_task_scores.values()))

    transfer_ratio = test_mean / train_mean if train_mean > 0 else 0.0

    return {
        "transfer_ratio": float(transfer_ratio),
        "train_mean": float(train_mean),
        "test_mean": float(test_mean),
        "generalization_gap": float(train_mean - test_mean),
    }
