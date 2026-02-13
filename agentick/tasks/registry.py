"""Task registry for dynamic task creation and discovery."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from agentick.core.env import AgentickEnv
from agentick.tasks.base import TaskSpec

logger = logging.getLogger(__name__)

# Global task registry
_TASK_REGISTRY: dict[str, type[TaskSpec]] = {}


def register_task(
    name: str,
    tags: list[str] | None = None,
) -> Callable[[type[TaskSpec]], type[TaskSpec]]:
    """
    Decorator to register a task class.

    Args:
        name: Task name (e.g., "GoToGoal-v0")
        tags: Capability tags

    Returns:
        Decorator function

    Example:
        @register_task("GoToGoal-v0", tags=["basic_navigation"])
        class GoToGoalTask(TaskSpec):
            ...
    """

    def decorator(task_class: type[TaskSpec]) -> type[TaskSpec]:
        # Set name and tags on class
        task_class.name = name
        if tags:
            task_class.capability_tags = tags

        # Register
        _TASK_REGISTRY[name] = task_class

        return task_class

    return decorator


def list_tasks(
    capability: str | None = None,
    difficulty: str | None = None,
) -> list[str]:
    """
    List all registered tasks, optionally filtered.

    Args:
        capability: Filter by capability tag
        difficulty: Filter by difficulty level

    Returns:
        List of task names
    """
    tasks = []

    for name, task_class in _TASK_REGISTRY.items():
        # Filter by capability
        if capability and capability not in task_class.capability_tags:
            continue

        # Filter by difficulty
        if difficulty and difficulty not in task_class.difficulty_configs:
            continue

        tasks.append(name)

    return sorted(tasks)


def get_task_class(name: str) -> type[TaskSpec]:
    """
    Get task class by name.

    Args:
        name: Task name

    Returns:
        Task class

    Raises:
        ValueError: If task not found
    """
    if name not in _TASK_REGISTRY:
        available = ", ".join(list_tasks())
        raise ValueError(f"Task '{name}' not found. Available tasks: {available}")

    return _TASK_REGISTRY[name]


def make(
    task_name: str,
    difficulty: str = "medium",
    render_mode: str = "ascii",
    reward_mode: str = "sparse",
    seed: int | None = None,
    fast_mode: bool = False,
    **kwargs: Any,
) -> AgentickEnv:
    """
    Create an environment for the specified task.

    Args:
        task_name: Name of the task (e.g., "GoToGoal-v0")
        difficulty: Difficulty level (easy, medium, hard, expert)
        render_mode: Rendering mode
        reward_mode: Reward mode (sparse, dense)
        seed: Random seed for task generation
        fast_mode: Enable fast mode for state_dict rendering (skip expensive conversions)
        **kwargs: Additional task parameters

    Returns:
        AgentickEnv instance

    Example:
        env = make("GoToGoal-v0", difficulty="hard", render_mode="rgb_array")
    """
    # Get task class
    task_class = get_task_class(task_name)

    # Create task instance
    task = task_class(difficulty=difficulty, **kwargs)

    # Generate task instance
    if seed is None:
        import time

        seed = int(time.time() * 1000) % (2**31)

    grid, config = task.generate(seed)

    # Validate instance
    if not task.validate_instance(grid, config):
        raise RuntimeError(
            f"Generated instance for '{task_name}' failed validation. Instance may not be solvable."
        )

    # Create environment
    env = TaskEnv(
        task=task,
        grid=grid,
        config=config,
        render_mode=render_mode,
        reward_mode=reward_mode,
        fast_mode=fast_mode,
        **kwargs,
    )

    return env


def make_suite(
    suite_name: str = "full",
    difficulty: str = "medium",
    **kwargs: Any,
) -> list[AgentickEnv]:
    """
    Create a suite of environments for benchmarking.

    Args:
        suite_name: Suite name (full, quick, navigation, memory, etc.)
        difficulty: Difficulty level for all tasks
        **kwargs: Additional arguments for environment creation

    Returns:
        List of environments

    Example:
        envs = make_suite("navigation", difficulty="hard")
    """
    if suite_name == "full":
        task_names = list_tasks()
    elif suite_name == "quick":
        # Quick suite: 5 representative tasks
        task_names = [
            "GoToGoal-v0",
            "MazeNavigation-v0",
            "KeyDoorPuzzle-v0",
            "SokobanPush-v0",
            "PreciseNavigation-v0",
        ]
    elif suite_name == "navigation":
        task_names = list_tasks(capability="navigation")
    elif suite_name == "memory":
        task_names = list_tasks(capability="memory")
    elif suite_name == "reasoning":
        task_names = list_tasks(capability="reasoning")
    elif suite_name == "control":
        task_names = list_tasks(capability="control")
    else:
        raise ValueError(f"Unknown suite: {suite_name}")

    # Filter to only tasks that exist
    task_names = [name for name in task_names if name in _TASK_REGISTRY]

    # Create environments
    envs = []
    for name in task_names:
        try:
            env = make(name, difficulty=difficulty, **kwargs)
            envs.append(env)
        except Exception as e:
            logger.warning(f"Failed to create environment '{name}': {e}")

    return envs


class EnvSpec:
    """Simple environment spec for gymnasium compatibility."""

    def __init__(self, id: str, max_episode_steps: int):
        self.id = id
        self.max_episode_steps = max_episode_steps
        self.nondeterministic = False
        self.reward_threshold = None
        self.order_enforce = True

    def make(self, **kwargs):
        """Create environment instance (for gymnasium compatibility)."""
        # Extract task name from id
        from agentick import make as agentick_make

        return agentick_make(self.id, **kwargs)


class TaskEnv(AgentickEnv):
    """Environment wrapper that integrates TaskSpec."""

    def __init__(
        self,
        task: TaskSpec,
        grid: Any,
        config: dict[str, Any],
        **kwargs: Any,
    ):
        """
        Initialize task environment.

        Args:
            task: TaskSpec instance
            grid: Generated grid
            config: Task configuration
            **kwargs: Arguments for AgentickEnv
        """
        self.task = task
        self.task_config = config

        # Get max_steps from task
        max_steps = config.get("max_steps", task.get_max_steps())

        # Initialize base environment
        super().__init__(grid=grid, max_steps=max_steps, **kwargs)

        # Set spec for gymnasium compatibility
        self.spec = EnvSpec(id=task.name, max_episode_steps=max_steps)

        # Set agent start position
        if "agent_start" in config:
            self.agent.position = config["agent_start"]

    def _reset_state(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        """Reset using task generation."""
        if seed is None:
            # Use internal RNG for deterministic resets
            seed = self.np_random.integers(0, 2**31)

        # Regenerate task
        grid, config = self.task.generate(seed)

        if not self.task.validate_instance(grid, config):
            raise RuntimeError("Generated instance failed validation")

        # Update environment state
        self.grid = grid
        self.task_config = config
        self.max_steps = config.get("max_steps", self.task.get_max_steps())

        if "agent_start" in config:
            self.agent.position = config["agent_start"]

    def _compute_reward(
        self,
        old_state: dict[str, Any],
        action: Any,
        new_state: dict[str, Any],
    ) -> float:
        """Compute reward using task's reward function."""
        if self.reward_mode == "sparse":
            return self.task.compute_sparse_reward(old_state, action, new_state, self._get_info())
        elif self.reward_mode == "dense":
            return self.task.compute_dense_reward(old_state, action, new_state, self._get_info())
        else:
            return super()._compute_reward(old_state, action, new_state)

    def _check_success(self) -> bool:
        """Check success using task's success condition."""
        state = self._get_state_for_reward()
        state.update(
            {
                "grid": self.grid,
                "agent": self.agent,
                "config": self.task_config,
            }
        )
        return self.task.check_success(state)

