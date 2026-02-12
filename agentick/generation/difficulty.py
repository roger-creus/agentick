"""Difficulty estimation and calibration for procedural generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from agentick.core.grid import Grid
from agentick.generation.validation import compute_solution_stats


@dataclass
class DifficultyMetrics:
    """Metrics used to estimate difficulty."""

    solution_length: int  # Steps in optimal solution
    branching_factor: float  # Average choices per step
    num_subgoals: int  # Number of intermediate goals (keys, switches, etc.)
    grid_size: int  # Total grid area
    wall_density: float  # Proportion of walls
    turns_in_path: int  # Number of direction changes
    estimated_difficulty: float  # 0-1 normalized difficulty score


class DifficultyEstimator:
    """Estimate and calibrate difficulty of generated levels."""

    def __init__(self):
        """Initialize difficulty estimator."""
        # Weights for different difficulty factors
        self.weights = {
            "solution_length": 0.3,
            "branching_factor": 0.2,
            "num_subgoals": 0.25,
            "wall_density": 0.1,
            "turns_in_path": 0.15,
        }

    def estimate(
        self,
        grid: Grid,
        start_pos: tuple[int, int],
        goal_positions: list[tuple[int, int]],
        config: dict[str, Any] | None = None,
    ) -> DifficultyMetrics:
        """
        Estimate difficulty of a level.

        Args:
            grid: Grid to analyze
            start_pos: Starting position
            goal_positions: Goal positions
            config: Additional configuration

        Returns:
            DifficultyMetrics with estimated difficulty
        """
        config = config or {}

        # Get solution stats
        stats = compute_solution_stats(grid, start_pos, goal_positions, config)

        if not stats.get("solvable"):
            # Unsolvable levels are infinitely difficult
            return DifficultyMetrics(
                solution_length=0,
                branching_factor=0,
                num_subgoals=0,
                grid_size=grid.width * grid.height,
                wall_density=0,
                turns_in_path=0,
                estimated_difficulty=float("inf"),
            )

        # Extract metrics
        solution_length = stats["optimal_length"]
        branching_factor = stats["branching_factor"]
        turns_in_path = stats["turns_in_path"]

        # Count subgoals
        num_subgoals = 0
        if config.get("keys"):
            num_subgoals += len(config["keys"])
        if config.get("switches"):
            num_subgoals += len(config["switches"])
        if config.get("boxes"):
            num_subgoals += len(config["boxes"])

        # Grid metrics
        grid_size = grid.width * grid.height
        num_walls = np.sum(grid.terrain == 1)  # CellType.WALL = 1
        wall_density = num_walls / grid_size

        # Normalize metrics to 0-1 range
        # These are heuristic ranges
        norm_length = min(1.0, solution_length / 100.0)
        norm_branching = min(1.0, branching_factor / 4.0)
        norm_subgoals = min(1.0, num_subgoals / 10.0)
        norm_walls = wall_density  # Already 0-1
        norm_turns = min(1.0, turns_in_path / 50.0)

        # Compute weighted difficulty score
        difficulty = (
            self.weights["solution_length"] * norm_length
            + self.weights["branching_factor"] * norm_branching
            + self.weights["num_subgoals"] * norm_subgoals
            + self.weights["wall_density"] * norm_walls
            + self.weights["turns_in_path"] * norm_turns
        )

        return DifficultyMetrics(
            solution_length=solution_length,
            branching_factor=branching_factor,
            num_subgoals=num_subgoals,
            grid_size=grid_size,
            wall_density=wall_density,
            turns_in_path=turns_in_path,
            estimated_difficulty=difficulty,
        )

    def calibrate(
        self,
        generator_fn,
        target_difficulty: float,
        tolerance: float = 0.1,
        max_attempts: int = 50,
    ) -> tuple[Any, DifficultyMetrics] | None:
        """
        Generate a level calibrated to target difficulty.

        Args:
            generator_fn: Function that generates (grid, start, goals, config)
            target_difficulty: Target difficulty (0-1)
            tolerance: Acceptable difficulty range
            max_attempts: Maximum generation attempts

        Returns:
            Tuple of (generated_data, metrics) or None if failed
        """
        for attempt in range(max_attempts):
            # Generate level
            grid, start_pos, goal_positions, config = generator_fn()

            # Estimate difficulty
            metrics = self.estimate(grid, start_pos, goal_positions, config)

            # Check if within tolerance
            if abs(metrics.estimated_difficulty - target_difficulty) <= tolerance:
                return ((grid, start_pos, goal_positions, config), metrics)

        # Failed to calibrate within max attempts
        return None


def estimate_difficulty(
    grid: Grid,
    start_pos: tuple[int, int],
    goal_positions: list[tuple[int, int]],
    config: dict[str, Any] | None = None,
) -> float:
    """
    Quick difficulty estimation.

    Args:
        grid: Grid to analyze
        start_pos: Starting position
        goal_positions: Goal positions
        config: Additional configuration

    Returns:
        Estimated difficulty score (0-1, or inf if unsolvable)
    """
    estimator = DifficultyEstimator()
    metrics = estimator.estimate(grid, start_pos, goal_positions, config)
    return metrics.estimated_difficulty


def calibrate_difficulty(
    generator_fn,
    target_difficulty: float,
    tolerance: float = 0.1,
    max_attempts: int = 50,
) -> tuple[Any, float] | None:
    """
    Generate level calibrated to target difficulty.

    Args:
        generator_fn: Generator function
        target_difficulty: Target difficulty
        tolerance: Tolerance
        max_attempts: Max attempts

    Returns:
        Tuple of (generated_data, actual_difficulty) or None
    """
    estimator = DifficultyEstimator()
    result = estimator.calibrate(generator_fn, target_difficulty, tolerance, max_attempts)

    if result:
        data, metrics = result
        return data, metrics.estimated_difficulty
    else:
        return None
