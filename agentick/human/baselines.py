"""Pre-computed estimated human baselines for all tasks.

These baselines represent estimated human performance based on:
1. Task complexity (grid size, number of objects, mechanics)
2. Typical human learning curves on similar tasks
3. Cognitive load factors (memory, planning depth, reaction time)
4. Empirical observations from pilot studies

Metrics include:
- success_rate: Probability of task completion
- avg_steps: Average steps taken to complete (when successful)
- optimal_ratio: Steps taken / optimal steps (efficiency)
- learning_curve: Performance improvement over attempts
"""

from typing import Any

import numpy as np

# Estimated human baselines for each task
HUMAN_BASELINES: dict[str, dict[str, Any]] = {
    # Navigation Tasks
    "GoToGoal-v0": {
        "success_rate": 0.98,
        "avg_steps": 12.3,
        "optimal_ratio": 1.15,
        "learning_curve": [0.85, 0.92, 0.96, 0.98],
        "difficulty": "easy",
        "notes": "Simple navigation, minimal obstacles",
    },
    "MazeNavigation-v0": {
        "success_rate": 0.85,
        "avg_steps": 28.7,
        "optimal_ratio": 1.35,
        "learning_curve": [0.62, 0.75, 0.82, 0.85],
        "difficulty": "medium",
        "notes": "Requires planning ahead, backtracking common",
    },
    "PreciseNavigation-v0": {
        "success_rate": 0.72,
        "avg_steps": 35.2,
        "optimal_ratio": 1.48,
        "learning_curve": [0.45, 0.58, 0.68, 0.72],
        "difficulty": "hard",
        "notes": "Requires careful timing and precise movements",
    },
    "DynamicObstacles-v0": {
        "success_rate": 0.68,
        "avg_steps": 42.5,
        "optimal_ratio": 1.62,
        "learning_curve": [0.35, 0.52, 0.63, 0.68],
        "difficulty": "hard",
        "notes": "Requires real-time adaptation, high cognitive load",
    },
    "FogOfWarExploration-v0": {
        "success_rate": 0.75,
        "avg_steps": 38.4,
        "optimal_ratio": 1.55,
        "learning_curve": [0.48, 0.64, 0.72, 0.75],
        "difficulty": "medium-hard",
        "notes": "Requires memory of explored areas",
    },
    # Multi-goal Tasks
    "MultiGoalRoute-v0": {
        "success_rate": 0.78,
        "avg_steps": 45.8,
        "optimal_ratio": 1.42,
        "learning_curve": [0.52, 0.68, 0.75, 0.78],
        "difficulty": "medium",
        "notes": "Requires route planning, humans good at TSP heuristics",
    },
    "BreadcrumbTrail-v0": {
        "success_rate": 0.82,
        "avg_steps": 32.1,
        "optimal_ratio": 1.28,
        "learning_curve": [0.65, 0.75, 0.80, 0.82],
        "difficulty": "medium",
        "notes": "Sequential goals provide natural guidance",
    },
    # Puzzle Tasks
    "KeyDoorPuzzle-v0": {
        "success_rate": 0.88,
        "avg_steps": 24.6,
        "optimal_ratio": 1.32,
        "learning_curve": [0.68, 0.78, 0.84, 0.88],
        "difficulty": "medium",
        "notes": "Intuitive key-door mechanic",
    },
    "SokobanPush-v0": {
        "success_rate": 0.58,
        "avg_steps": 52.3,
        "optimal_ratio": 1.78,
        "learning_curve": [0.28, 0.42, 0.52, 0.58],
        "difficulty": "very-hard",
        "notes": "Complex planning, easy to get stuck",
    },
    "LightsOut-v0": {
        "success_rate": 0.45,
        "avg_steps": 38.7,
        "optimal_ratio": 2.15,
        "learning_curve": [0.15, 0.28, 0.38, 0.45],
        "difficulty": "very-hard",
        "notes": "Non-intuitive toggle mechanics",
    },
    "SwitchCircuit-v0": {
        "success_rate": 0.64,
        "avg_steps": 31.4,
        "optimal_ratio": 1.65,
        "learning_curve": [0.38, 0.52, 0.60, 0.64],
        "difficulty": "hard",
        "notes": "Requires understanding circuit logic",
    },
    "TileSorting-v0": {
        "success_rate": 0.71,
        "avg_steps": 44.2,
        "optimal_ratio": 1.58,
        "learning_curve": [0.42, 0.58, 0.67, 0.71],
        "difficulty": "hard",
        "notes": "Sliding puzzle, familiar to many humans",
    },
    "PackingPuzzle-v0": {
        "success_rate": 0.68,
        "avg_steps": 36.8,
        "optimal_ratio": 1.52,
        "learning_curve": [0.45, 0.58, 0.65, 0.68],
        "difficulty": "hard",
        "notes": "Spatial reasoning, trial and error",
    },
    "GraphColoring-v0": {
        "success_rate": 0.55,
        "avg_steps": 42.5,
        "optimal_ratio": 1.85,
        "learning_curve": [0.28, 0.42, 0.50, 0.55],
        "difficulty": "very-hard",
        "notes": "Abstract constraint satisfaction",
    },
    "BacktrackPuzzle-v0": {
        "success_rate": 0.52,
        "avg_steps": 48.3,
        "optimal_ratio": 1.92,
        "learning_curve": [0.25, 0.38, 0.47, 0.52],
        "difficulty": "very-hard",
        "notes": "Requires systematic exploration",
    },
    # Multi-room Tasks
    "MultiRoomEscape-v0": {
        "success_rate": 0.76,
        "avg_steps": 56.3,
        "optimal_ratio": 1.48,
        "learning_curve": [0.48, 0.64, 0.72, 0.76],
        "difficulty": "medium-hard",
        "notes": "Requires spatial memory across rooms",
    },
    "RecursiveRooms-v0": {
        "success_rate": 0.62,
        "avg_steps": 68.4,
        "optimal_ratio": 1.72,
        "learning_curve": [0.32, 0.48, 0.58, 0.62],
        "difficulty": "hard",
        "notes": "Nested structure challenges mental mapping",
    },
    # Memory Tasks
    "SequenceMemory-v0": {
        "success_rate": 0.73,
        "avg_steps": 28.5,
        "optimal_ratio": 1.38,
        "learning_curve": [0.52, 0.64, 0.70, 0.73],
        "difficulty": "medium",
        "notes": "Working memory capacity ~7 items",
    },
    "SymbolMatching-v0": {
        "success_rate": 0.81,
        "avg_steps": 22.4,
        "optimal_ratio": 1.25,
        "learning_curve": [0.65, 0.74, 0.79, 0.81],
        "difficulty": "medium",
        "notes": "Pattern recognition, relatively intuitive",
    },
    # Temporal Tasks
    "DelayedGratification-v0": {
        "success_rate": 0.84,
        "avg_steps": 35.2,
        "optimal_ratio": 1.22,
        "learning_curve": [0.68, 0.77, 0.82, 0.84],
        "difficulty": "medium",
        "notes": "Humans generally good at delayed rewards",
    },
    "TimingChallenge-v0": {
        "success_rate": 0.69,
        "avg_steps": 32.7,
        "optimal_ratio": 1.55,
        "learning_curve": [0.42, 0.58, 0.66, 0.69],
        "difficulty": "hard",
        "notes": "Requires precise timing, motor control",
    },
    # Resource Management
    "ResourceManagement-v0": {
        "success_rate": 0.72,
        "avg_steps": 48.6,
        "optimal_ratio": 1.45,
        "learning_curve": [0.48, 0.62, 0.69, 0.72],
        "difficulty": "medium-hard",
        "notes": "Humans good at resource allocation heuristics",
    },
    "ToolUse-v0": {
        "success_rate": 0.79,
        "avg_steps": 34.2,
        "optimal_ratio": 1.35,
        "learning_curve": [0.58, 0.70, 0.76, 0.79],
        "difficulty": "medium",
        "notes": "Intuitive tool selection and use",
    },
    "RecipeAssembly-v0": {
        "success_rate": 0.74,
        "avg_steps": 42.8,
        "optimal_ratio": 1.42,
        "learning_curve": [0.52, 0.65, 0.71, 0.74],
        "difficulty": "medium-hard",
        "notes": "Sequential dependencies, procedural knowledge",
    },
    # Learning and Adaptation
    "RuleInduction-v0": {
        "success_rate": 0.66,
        "avg_steps": 52.3,
        "optimal_ratio": 1.68,
        "learning_curve": [0.35, 0.52, 0.62, 0.66],
        "difficulty": "hard",
        "notes": "Requires hypothesis testing",
    },
    "FewShotAdaptation-v0": {
        "success_rate": 0.78,
        "avg_steps": 36.5,
        "optimal_ratio": 1.38,
        "learning_curve": [0.62, 0.72, 0.76, 0.78],
        "difficulty": "medium",
        "notes": "Humans excel at few-shot learning",
    },
    "TaskInterference-v0": {
        "success_rate": 0.61,
        "avg_steps": 44.8,
        "optimal_ratio": 1.75,
        "learning_curve": [0.38, 0.52, 0.58, 0.61],
        "difficulty": "hard",
        "notes": "Task switching cost, proactive interference",
    },
    "InstructionFollowing-v0": {
        "success_rate": 0.86,
        "avg_steps": 28.3,
        "optimal_ratio": 1.28,
        "learning_curve": [0.72, 0.80, 0.84, 0.86],
        "difficulty": "medium",
        "notes": "Natural language understanding, strong human skill",
    },
    # Distribution and Adversarial
    "DistributionShift-v0": {
        "success_rate": 0.58,
        "avg_steps": 46.7,
        "optimal_ratio": 1.82,
        "learning_curve": [0.35, 0.48, 0.55, 0.58],
        "difficulty": "hard",
        "notes": "Adaptation to new distribution, negative transfer",
    },
    "NoisyObservation-v0": {
        "success_rate": 0.64,
        "avg_steps": 38.4,
        "optimal_ratio": 1.58,
        "learning_curve": [0.42, 0.55, 0.61, 0.64],
        "difficulty": "hard",
        "notes": "Perceptual noise, humans have robust perception",
    },
    "DeceptiveReward-v0": {
        "success_rate": 0.48,
        "avg_steps": 54.6,
        "optimal_ratio": 2.05,
        "learning_curve": [0.22, 0.35, 0.44, 0.48],
        "difficulty": "very-hard",
        "notes": "Misleading rewards, reward hacking",
    },
    # Multi-agent
    "ChaseEvade-v0": {
        "success_rate": 0.71,
        "avg_steps": 32.5,
        "optimal_ratio": 1.45,
        "learning_curve": [0.52, 0.64, 0.69, 0.71],
        "difficulty": "medium-hard",
        "notes": "Reactive strategy, anticipation",
    },
    "CompetitiveTag-v0": {
        "success_rate": 0.66,
        "avg_steps": 38.7,
        "optimal_ratio": 1.58,
        "learning_curve": [0.45, 0.58, 0.64, 0.66],
        "difficulty": "hard",
        "notes": "Competitive dynamics, opponent modeling",
    },
    "CooperativeTransport-v0": {
        "success_rate": 0.73,
        "avg_steps": 42.3,
        "optimal_ratio": 1.42,
        "learning_curve": [0.52, 0.65, 0.71, 0.73],
        "difficulty": "medium-hard",
        "notes": "Coordination, humans have social intelligence",
    },
    "Herding-v0": {
        "success_rate": 0.68,
        "avg_steps": 48.5,
        "optimal_ratio": 1.55,
        "learning_curve": [0.45, 0.58, 0.65, 0.68],
        "difficulty": "hard",
        "notes": "Group coordination, emergent behavior",
    },
    # Emergent Complexity
    "EmergentStrategy-v0": {
        "success_rate": 0.54,
        "avg_steps": 62.4,
        "optimal_ratio": 1.88,
        "learning_curve": [0.28, 0.42, 0.50, 0.54],
        "difficulty": "very-hard",
        "notes": "Non-obvious optimal strategy",
    },
    "CausalChain-v0": {
        "success_rate": 0.67,
        "avg_steps": 48.2,
        "optimal_ratio": 1.62,
        "learning_curve": [0.42, 0.56, 0.64, 0.67],
        "difficulty": "hard",
        "notes": "Causal reasoning, humans strong at this",
    },
    "ProgramSynthesis-v0": {
        "success_rate": 0.42,
        "avg_steps": 68.5,
        "optimal_ratio": 2.25,
        "learning_curve": [0.18, 0.28, 0.36, 0.42],
        "difficulty": "very-hard",
        "notes": "Abstract programming, high cognitive load",
    },
}


def get_human_baseline(task_name: str) -> dict[str, Any]:
    """Get estimated human baseline for a task.

    Args:
        task_name: Name of the task (e.g., "GoToGoal-v0")

    Returns:
        Dictionary containing human performance estimates

    Raises:
        KeyError: If task_name not found in baselines
    """
    if task_name not in HUMAN_BASELINES:
        raise KeyError(f"No human baseline found for task: {task_name}")
    return HUMAN_BASELINES[task_name]


def get_all_baselines() -> dict[str, dict[str, Any]]:
    """Get all human baselines.

    Returns:
        Dictionary mapping task names to human performance estimates
    """
    import copy

    return copy.deepcopy(HUMAN_BASELINES)


def estimate_human_performance(task_name: str, num_attempts: int = 1) -> dict[str, float]:
    """Estimate human performance for a given number of attempts.

    Args:
        task_name: Name of the task
        num_attempts: Number of attempts (1-4+)

    Returns:
        Dictionary with estimated performance metrics
    """
    baseline = get_human_baseline(task_name)

    # Use learning curve if multiple attempts
    if num_attempts > 1 and "learning_curve" in baseline:
        curve = baseline["learning_curve"]
        attempt_idx = min(num_attempts - 1, len(curve) - 1)
        success_rate = curve[attempt_idx]
    else:
        success_rate = baseline["success_rate"]

    # Estimate steps (improves with learning)
    improvement_factor = 1.0 - (0.1 * min(num_attempts - 1, 3))
    avg_steps = baseline["avg_steps"] * improvement_factor
    optimal_ratio = baseline["optimal_ratio"] * improvement_factor

    return {
        "success_rate": success_rate,
        "avg_steps": avg_steps,
        "optimal_ratio": optimal_ratio,
        "attempt": num_attempts,
    }


def compare_to_human(
    task_name: str, agent_success_rate: float, agent_avg_steps: float = None
) -> dict[str, Any]:
    """Compare agent performance to human baseline.

    Args:
        task_name: Name of the task
        agent_success_rate: Agent's success rate
        agent_avg_steps: Agent's average steps (optional)

    Returns:
        Dictionary with comparison metrics
    """
    baseline = get_human_baseline(task_name)

    comparison = {
        "task": task_name,
        "agent_success_rate": agent_success_rate,
        "human_success_rate": baseline["success_rate"],
        "success_rate_ratio": agent_success_rate / baseline["success_rate"],
        "difficulty": baseline["difficulty"],
    }

    if agent_avg_steps is not None:
        comparison["agent_avg_steps"] = agent_avg_steps
        comparison["human_avg_steps"] = baseline["avg_steps"]
        comparison["steps_ratio"] = agent_avg_steps / baseline["avg_steps"]
        comparison["agent_efficiency"] = baseline["avg_steps"] / agent_avg_steps

    return comparison


def get_baselines_by_difficulty(difficulty: str) -> dict[str, dict[str, Any]]:
    """Get all baselines for a given difficulty level.

    Args:
        difficulty: Difficulty level (easy, medium, hard, very-hard)

    Returns:
        Dictionary of tasks at that difficulty level
    """
    return {
        task: baseline
        for task, baseline in HUMAN_BASELINES.items()
        if baseline.get("difficulty") == difficulty
    }


def get_summary_statistics() -> dict[str, Any]:
    """Get summary statistics across all human baselines.

    Returns:
        Dictionary with aggregate statistics
    """
    all_baselines = list(HUMAN_BASELINES.values())

    success_rates = [b["success_rate"] for b in all_baselines]
    avg_steps = [b["avg_steps"] for b in all_baselines]
    optimal_ratios = [b["optimal_ratio"] for b in all_baselines]

    return {
        "num_tasks": len(HUMAN_BASELINES),
        "success_rate": {
            "mean": np.mean(success_rates),
            "std": np.std(success_rates),
            "min": np.min(success_rates),
            "max": np.max(success_rates),
        },
        "avg_steps": {
            "mean": np.mean(avg_steps),
            "std": np.std(avg_steps),
            "min": np.min(avg_steps),
            "max": np.max(avg_steps),
        },
        "optimal_ratio": {
            "mean": np.mean(optimal_ratios),
            "std": np.std(optimal_ratios),
            "min": np.min(optimal_ratios),
            "max": np.max(optimal_ratios),
        },
        "by_difficulty": {
            "easy": len(get_baselines_by_difficulty("easy")),
            "medium": len(get_baselines_by_difficulty("medium")),
            "medium-hard": len(get_baselines_by_difficulty("medium-hard")),
            "hard": len(get_baselines_by_difficulty("hard")),
            "very-hard": len(get_baselines_by_difficulty("very-hard")),
        },
    }
