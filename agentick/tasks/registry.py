"""Task registry for dynamic task creation and discovery."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

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
        render_mode: Rendering mode. ``"rgb_array"`` for isometric sprites, ``"ascii"`` for text grid, etc.
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
        self._task_name = task.name

        # Get max_steps from task
        max_steps = config.get("max_steps", task.get_max_steps())

        # Initialize base environment
        super().__init__(grid=grid, max_steps=max_steps, **kwargs)

        # Set spec for gymnasium compatibility
        self.spec = EnvSpec(id=task.name, max_episode_steps=max_steps)

        # Set agent start position
        if "agent_start" in config:
            self.agent.position = config["agent_start"]

    def step(self, action):
        """Step with optional post-action task hook (e.g. moving obstacles)."""
        obs, reward, terminated, truncated, info = super().step(action)
        # Fix info["success"]: base class sets terminated=True for ANY check_done(),
        # but tasks may terminate without success (e.g. decoy-taken in DelayedGratification).
        # _last_success was set in _check_success() to reflect true goal achievement.
        if terminated and hasattr(self, "_last_success"):
            info["success"] = self._last_success
        # For survival tasks (no explicit done): check success on truncation too
        if truncated and not terminated:
            state = self._get_state_for_reward()
            state.update({"grid": self.grid, "agent": self.agent, "config": self.task_config})
            info["success"] = bool(self.task.check_success(state))
        # Allow task to update world after agent acts (NPCs, obstacles, etc.)
        if hasattr(self.task, "on_env_step"):
            self.task.on_env_step(self.agent, self.grid, self.task_config, self.step_count)
        return obs, reward, terminated, truncated, info

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

        # Allow task to initialize dynamic state
        if hasattr(self.task, "on_env_reset"):
            self.task.on_env_reset(self.agent, self.grid, self.task_config)

    def _try_move_to(self, new_pos) -> None:
        """Attempt to move agent to new_pos with task-specific checks."""
        if hasattr(self.task, "can_agent_enter"):
            # Tasks with can_agent_enter handle their own entry logic
            # (e.g. Sokoban pushing, tile sliding, walkable switches).
            # Check bounds first; walkability is checked unless the task
            # explicitly overrides terrain (e.g. ToolUse hammer breaks walls).
            if not self.grid.in_bounds(new_pos):
                return
            if not self.task.can_agent_enter(new_pos, self.agent, self.grid):
                return
            if not getattr(self.task, "overrides_walkable", False):
                if not self.grid.is_walkable(new_pos):
                    return
        else:
            # Standard terrain check
            if not self.grid.is_walkable(new_pos):
                return
            # Object blocking check (DOOR/LEVER/SWITCH are solid)
            if self.grid.is_object_blocking(new_pos):
                return

        self.agent.position = new_pos

        # Post-move hook (e.g., auto-pickup key)
        if hasattr(self.task, "on_agent_moved"):
            self.task.on_agent_moved(new_pos, self.agent, self.grid)

    def _move_agent(self, action_type) -> None:
        """Move agent, delegating task-specific entry rules to the task."""
        from agentick.core.actions import get_move_delta
        from agentick.core.types import ActionType as AT
        from agentick.core.types import Direction

        _ACTION_DIR = {
            AT.MOVE_UP: Direction.NORTH,
            AT.MOVE_DOWN: Direction.SOUTH,
            AT.MOVE_LEFT: Direction.WEST,
            AT.MOVE_RIGHT: Direction.EAST,
        }

        # Allow tasks to remap actions (e.g. DistributionShift swaps directions)
        remap = self.task_config.get("_action_remap")
        if remap and action_type in remap:
            action_type = remap[action_type]

        delta = get_move_delta(action_type)
        if delta is None:
            return

        # Update orientation to face movement direction
        if action_type in _ACTION_DIR:
            self.agent.orientation = _ACTION_DIR[action_type]

        dx, dy = delta
        new_pos = (self.agent.position[0] + dx, self.agent.position[1] + dy)
        self._try_move_to(new_pos)

    def _move_forward(self) -> None:
        """Move agent forward, with task-specific entry checks."""
        dx, dy = self.agent.orientation.to_delta()
        new_pos = (self.agent.position[0] + dx, self.agent.position[1] + dy)
        self._try_move_to(new_pos)

    def _execute_action(self, action_type) -> None:
        """Execute action with task-specific INTERACT hook.

        INTERACT targets the cell the agent is facing (orientation + 1 step).
        Tasks with ``interact_self = True`` (e.g. LightsOut) target the
        agent's own position instead.
        """
        from agentick.core.types import ActionType as AT

        if action_type == AT.INTERACT and hasattr(self.task, "on_agent_interact"):
            if getattr(self.task, "interact_self", False):
                target_pos = self.agent.position
            else:
                dx, dy = self.agent.orientation.to_delta()
                target_pos = (self.agent.position[0] + dx, self.agent.position[1] + dy)
            self.task.on_agent_interact(target_pos, self.agent, self.grid)
        else:
            super()._execute_action(action_type)

    def _get_state_for_reward(self) -> dict[str, Any]:
        """Get state snapshot for reward computation (includes full task state)."""
        state = super()._get_state_for_reward()
        state.update(
            {
                "grid": self.grid,
                "agent": self.agent,
                "config": self.task_config,
            }
        )
        return state

    def _get_info(self) -> dict[str, Any]:
        """Get info dict with task config for renderer access.

        Gymnasium's check_step_determinism compares info dicts, so
        we strip non-comparable objects (RNG instances) from the copy.
        """
        info = super()._get_info()
        info["task_config"] = {
            k: v
            for k, v in self.task_config.items()
            if not isinstance(v, np.random.Generator)
        }
        return info

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
        """Check if episode should terminate (via check_done) and record true success."""
        state = self._get_state_for_reward()
        state.update(
            {
                "grid": self.grid,
                "agent": self.agent,
                "config": self.task_config,
            }
        )
        done = self.task.check_done(state)
        # Store the true success flag separately so we can fix info["success"] in step()
        self._last_success = bool(self.task.check_success(state)) if done else False
        return done
