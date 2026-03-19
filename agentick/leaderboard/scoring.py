"""Official scoring methodology for Agentick leaderboard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TaskScore:
    """Score for a single task."""

    task_name: str
    difficulty: str
    mean_return: float
    std_return: float
    success_rate: float
    normalized_score: float
    normalized_score_ci: tuple[float, float]  # 95% confidence interval
    n_episodes: int
    episode_returns: list[float]


@dataclass
class CapabilityScore:
    """Score for a capability category (average of tasks in that category)."""

    capability: str
    tasks: list[str]
    mean_normalized_score: float
    normalized_score_ci: tuple[float, float]
    task_scores: dict[str, float]  # task_name -> normalized_score


@dataclass
class AggregateScore:
    """Overall Agentick score (equally weighted by capability)."""

    agentick_score: float
    agentick_score_ci: tuple[float, float]
    per_capability: dict[str, CapabilityScore]
    per_task: dict[str, TaskScore]


# Capability mapping (task -> capability tags)
TASK_CAPABILITY_MAP = {
    # Navigation (8)
    "GoToGoal-v0": "navigation",
    "MazeNavigation-v0": "navigation",
    "ShortestPath-v0": "navigation",
    "DynamicObstacles-v0": "navigation",
    "CuriosityMaze-v0": "navigation",
    "RecursiveRooms-v0": "navigation",
    "TimingChallenge-v0": "navigation",
    "InstructionFollowing-v0": "navigation",
    # Planning (9)
    "SokobanPush-v0": "planning",
    "KeyDoorPuzzle-v0": "planning",
    "BacktrackPuzzle-v0": "planning",
    "TileSorting-v0": "planning",
    "PackingPuzzle-v0": "planning",
    "PreciseNavigation-v0": "planning",
    "RecipeAssembly-v0": "planning",
    "ToolUse-v0": "planning",
    "ResourceManagement-v0": "planning",
    # Reasoning (8)
    "SwitchCircuit-v0": "reasoning",
    "RuleInduction-v0": "reasoning",
    "LightsOut-v0": "reasoning",
    "GraphColoring-v0": "reasoning",
    "SymbolMatching-v0": "reasoning",
    "ProgramSynthesis-v0": "reasoning",
    "TaskInterference-v0": "reasoning",
    "DeceptiveReward-v0": "reasoning",
    # Memory (4)
    "SequenceMemory-v0": "memory",
    "DelayedGratification-v0": "memory",
    "TreasureHunt-v0": "memory",
    "FogOfWarExploration-v0": "memory",
    # Generalization (3)
    "FewShotAdaptation-v0": "generalization",
    "DistributionShift-v0": "generalization",
    "NoisyObservation-v0": "generalization",
    # Multi-Agent (5)
    "CooperativeTransport-v0": "multi_agent",
    "TagHunt-v0": "multi_agent",
    "ChaseEvade-v0": "multi_agent",
    "Herding-v0": "multi_agent",
    "EmergentStrategy-v0": "multi_agent",
}


def normalize_score(
    agent_return: float,
    random_baseline: float,
    optimal_return: float,
    clip: bool = True,
) -> float:
    """
    Normalize agent return to [0, 1] scale using random and optimal baselines.

    normalized_score = (agent_return - random_baseline) / (optimal_return - random_baseline)

    Args:
        agent_return: Agent's mean return on this task
        random_baseline: Expected return of random agent
        optimal_return: Optimal/oracle return
        clip: Whether to clip to [0, 1] range

    Returns:
        Normalized score in [0, 1] (if clip=True) or [-inf, inf] (if clip=False)
    """
    if abs(optimal_return - random_baseline) < 1e-9:
        # Edge case: no gap between random and optimal (task is trivial or broken)
        return 1.0 if agent_return >= optimal_return else 0.0

    normalized = (agent_return - random_baseline) / (optimal_return - random_baseline)

    if clip:
        normalized = max(0.0, min(1.0, normalized))

    return normalized


def bootstrap_confidence_interval(
    values: list[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    statistic: str = "mean",
) -> tuple[float, float]:
    """
    Compute bootstrap confidence interval for a statistic.

    Args:
        values: Sample values
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (0.95 for 95% CI)
        statistic: Statistic to compute ("mean" or "median")

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if len(values) == 0:
        return (0.0, 0.0)

    if len(values) == 1:
        # Can't bootstrap with single sample
        return (values[0], values[0])

    values_arr = np.array(values)
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility

    bootstrap_stats = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        sample = rng.choice(values_arr, size=len(values_arr), replace=True)

        # Compute statistic
        if statistic == "mean":
            stat = np.mean(sample)
        elif statistic == "median":
            stat = np.median(sample)
        else:
            raise ValueError(f"Unknown statistic: {statistic}")

        bootstrap_stats.append(stat)

    # Compute percentiles
    alpha = 1.0 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1.0 - alpha / 2) * 100

    lower = np.percentile(bootstrap_stats, lower_percentile)
    upper = np.percentile(bootstrap_stats, upper_percentile)

    return (float(lower), float(upper))


def compute_task_score(
    task_name: str,
    difficulty: str,
    episode_returns: list[float],
    random_baseline: float,
    optimal_return: float,
    success_flags: list[bool] | None = None,
) -> TaskScore:
    """
    Compute score for a single task.

    Args:
        task_name: Name of the task
        difficulty: Difficulty level
        episode_returns: List of episode returns
        random_baseline: Random agent baseline return
        optimal_return: Optimal/oracle return
        success_flags: Optional list of success flags per episode

    Returns:
        TaskScore object
    """
    if len(episode_returns) == 0:
        raise ValueError(f"No episodes for task {task_name}")

    mean_return = float(np.mean(episode_returns))
    std_return = float(np.std(episode_returns))

    # Compute normalized score per episode, then average
    normalized_scores = [
        normalize_score(ret, random_baseline, optimal_return) for ret in episode_returns
    ]
    mean_normalized = float(np.mean(normalized_scores))

    # Bootstrap CI on normalized scores
    norm_ci = bootstrap_confidence_interval(normalized_scores)

    # Success rate
    if success_flags is not None:
        success_rate = float(np.mean(success_flags))
    else:
        success_rate = 0.0

    return TaskScore(
        task_name=task_name,
        difficulty=difficulty,
        mean_return=mean_return,
        std_return=std_return,
        success_rate=success_rate,
        normalized_score=mean_normalized,
        normalized_score_ci=norm_ci,
        n_episodes=len(episode_returns),
        episode_returns=episode_returns,
    )


def compute_capability_scores(
    task_scores: dict[str, TaskScore],
) -> dict[str, CapabilityScore]:
    """
    Aggregate task scores by capability.

    Args:
        task_scores: Dictionary of task_name -> TaskScore

    Returns:
        Dictionary of capability -> CapabilityScore
    """
    # Group tasks by capability
    capability_tasks: dict[str, list[str]] = {}
    for task_name in task_scores:
        capability = TASK_CAPABILITY_MAP.get(task_name, "other")
        if capability not in capability_tasks:
            capability_tasks[capability] = []
        capability_tasks[capability].append(task_name)

    # Compute capability scores
    capability_scores = {}
    for capability, tasks in capability_tasks.items():
        # Collect normalized scores for all tasks in this capability
        normalized_scores = [task_scores[task].normalized_score for task in tasks]

        # Mean score for this capability
        mean_score = float(np.mean(normalized_scores))

        # Bootstrap CI by resampling task scores
        ci = bootstrap_confidence_interval(normalized_scores)

        # Task-level scores
        task_level_scores = {task: task_scores[task].normalized_score for task in tasks}

        capability_scores[capability] = CapabilityScore(
            capability=capability,
            tasks=tasks,
            mean_normalized_score=mean_score,
            normalized_score_ci=ci,
            task_scores=task_level_scores,
        )

    return capability_scores


def compute_aggregate_score(
    task_scores: dict[str, TaskScore],
) -> AggregateScore:
    """
    Compute overall Agentick score (equally weighted by capability, not by task count).

    This ensures that a capability with 5 tasks isn't 5× more important than
    a capability with 1 task. Each capability contributes equally to the final score.

    Args:
        task_scores: Dictionary of task_name -> TaskScore

    Returns:
        AggregateScore with overall score and breakdowns
    """
    # Compute per-capability scores
    capability_scores = compute_capability_scores(task_scores)

    # Aggregate: mean of per-capability means (equal capability weighting)
    capability_means = [cap.mean_normalized_score for cap in capability_scores.values()]
    agentick_score = float(np.mean(capability_means))

    # Bootstrap CI by resampling capability scores
    ci = bootstrap_confidence_interval(capability_means)

    return AggregateScore(
        agentick_score=agentick_score,
        agentick_score_ci=ci,
        per_capability=capability_scores,
        per_task=task_scores,
    )


def compute_score_from_results(
    results: dict[str, Any],
    baselines: dict[str, dict[str, float]],
) -> AggregateScore:
    """
    Compute aggregate score from raw evaluation results and baselines.

    Args:
        results: Dictionary with structure:
            {
                "task_name": {
                    "difficulty": str,
                    "episode_returns": list[float],
                    "success_flags": list[bool],
                }
            }
        baselines: Dictionary with structure:
            {
                "task_name": {
                    "random_baseline": float,
                    "optimal_return": float,
                }
            }

    Returns:
        AggregateScore object
    """
    task_scores = {}

    for task_name, task_data in results.items():
        if task_name not in baselines:
            raise ValueError(f"No baselines found for task {task_name}")

        baseline_data = baselines[task_name]

        task_score = compute_task_score(
            task_name=task_name,
            difficulty=task_data["difficulty"],
            episode_returns=task_data["episode_returns"],
            random_baseline=baseline_data["random_baseline"],
            optimal_return=baseline_data["optimal_return"],
            success_flags=task_data.get("success_flags"),
        )

        task_scores[task_name] = task_score

    return compute_aggregate_score(task_scores)
