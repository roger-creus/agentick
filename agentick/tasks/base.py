"""Base class for task specifications."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

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

    # Public observation config controls. ``task_config`` is also returned in
    # Gymnasium ``info`` and embedded in state_dict observations, so subclasses
    # must keep hidden solution state out of this public view.
    public_config_exclude: set[str] = set()
    public_config_include_private: set[str] = set()

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

    def get_public_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Return the task config safe to expose through observations/info.

        Full internal config remains available on ``env.task_config`` for task
        mechanics and oracle baselines. Public observations omit underscored
        runtime internals by default, plus any task-specific hidden fields.
        """
        public: dict[str, Any] = {}
        for key, value in config.items():
            if key in self.public_config_exclude:
                continue
            if key.startswith("_") and key not in self.public_config_include_private:
                continue
            if isinstance(value, np.random.Generator):
                continue
            public[key] = self._to_public_value(value)
        return public

    @classmethod
    def _to_public_value(cls, value: Any) -> Any:
        """Convert common task config values into public, comparable values."""
        if isinstance(value, np.random.Generator):
            return None
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {
                cls._public_key(k): cls._to_public_value(v)
                for k, v in value.items()
                if not isinstance(v, np.random.Generator)
            }
        if isinstance(value, set):
            return [cls._to_public_value(v) for v in sorted(value, key=str)]
        if isinstance(value, tuple):
            return tuple(cls._to_public_value(v) for v in value)
        if isinstance(value, list):
            return [cls._to_public_value(v) for v in value]
        return value

    @staticmethod
    def _public_key(key: Any) -> str:
        """Make dict keys stable for public config serialization."""
        if isinstance(key, tuple):
            return ",".join(str(part) for part in key)
        return str(key)
