"""Base class for task specifications."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agentick.core.actions import ActionType
from agentick.core.grid import Grid
from agentick.tasks.configs import DifficultyConfig


class TaskSpec(ABC):
    """
    Abstract base class for task specifications.

    Each task must implement:
    - generate(): Procedurally generate a task instance
    - compute_reward(): Calculate reward for a transition
    - check_success(): Determine if task is complete

    And provide:
    - name, description, capability_tags
    - optimal_return, random_baseline
    - difficulty_levels
    """

    # Task metadata (to be set by subclasses)
    name: str = "BaseTask"
    description: str = "Base task specification"
    capability_tags: list[str] = []

    # Difficulty configurations
    difficulty_configs: dict[str, DifficultyConfig] = {}

    def __init__(self, difficulty: str = "medium", **kwargs: Any):
        """
        Initialize task.

        Args:
            difficulty: Difficulty level name
            **kwargs: Additional task parameters
        """
        self.difficulty = difficulty
        self.kwargs = kwargs

        if difficulty not in self.difficulty_configs:
            raise ValueError(
                f"Unknown difficulty '{difficulty}'. "
                f"Available: {list(self.difficulty_configs.keys())}"
            )

        self.difficulty_config = self.difficulty_configs[difficulty]

    @abstractmethod
    def generate(self, seed: int) -> tuple[Grid, dict[str, Any]]:
        """
        Generate a task instance.

        Args:
            seed: Random seed for generation

        Returns:
            (grid, config_dict) where config_dict contains:
                - agent_start: Agent starting position
                - max_steps: Maximum steps
                - Any task-specific configuration
        """
        pass

    @abstractmethod
    def compute_dense_reward(
        self,
        old_state: dict[str, Any],
        action: ActionType,
        new_state: dict[str, Any],
        info: dict[str, Any],
    ) -> float:
        """
        Compute shaped dense reward.

        Args:
            old_state: State before action
            action: Action taken
            new_state: State after action
            info: Info dict

        Returns:
            Dense reward value
        """
        pass

    def compute_sparse_reward(
        self,
        old_state: dict[str, Any],
        action: ActionType,
        new_state: dict[str, Any],
        info: dict[str, Any],
    ) -> float:
        """
        Compute sparse reward (default: 1.0 on success, 0.0 otherwise).

        Can be overridden for task-specific sparse rewards.
        """
        if self.check_success(new_state):
            return 1.0
        return 0.0

    @abstractmethod
    def check_success(self, state: dict[str, Any]) -> bool:
        """
        Check if task is successfully completed.

        Args:
            state: Current state dict

        Returns:
            True if task is complete
        """
        pass

    def check_done(self, state: dict[str, Any]) -> bool:
        """
        Check if the episode should end (may or may not be a success).

        Override this in tasks where the episode can end without success
        (e.g. stepping on a decoy that traps the agent). The default
        implementation delegates to check_success, meaning episodes only
        end on success or time-limit truncation.

        Args:
            state: Current state dict

        Returns:
            True if the episode should terminate (regardless of success)
        """
        return bool(self.check_success(state))

    def get_optimal_return(self, difficulty: str | None = None) -> float:
        """
        Get theoretical optimal return for this task.

        Args:
            difficulty: Difficulty level (uses self.difficulty if None)

        Returns:
            Optimal return value
        """
        # Default implementation, should be overridden
        return 1.0

    def get_random_baseline(self, difficulty: str | None = None) -> float:
        """
        Get expected return for a random agent.

        Args:
            difficulty: Difficulty level (uses self.difficulty if None)

        Returns:
            Expected random agent return
        """
        # Default implementation, should be overridden
        return 0.0

    def get_max_steps(self) -> int:
        """Get maximum steps for current difficulty."""
        return self.difficulty_config.max_steps

    def validate_instance(self, grid: Grid, config: dict[str, Any]) -> bool:
        """
        Validate that generated instance is solvable.

        Default implementation checks basic reachability.
        Override for task-specific validation.

        Args:
            grid: Generated grid
            config: Generation config

        Returns:
            True if instance is valid/solvable
        """
        # Check that agent can reach at least one goal
        if "agent_start" not in config or "goal_positions" not in config:
            return True  # Can't validate without this info

        agent_pos = config["agent_start"]
        goals = config["goal_positions"]

        if not goals:
            return True  # No goals to reach

        # Check if at least one goal is reachable
        reachable = grid.flood_fill(agent_pos)
        for goal in goals:
            if goal in reachable:
                return True

        return False
