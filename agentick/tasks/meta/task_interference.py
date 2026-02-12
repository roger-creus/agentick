"""Task Interference - Multiple interleaved tasks."""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("TaskInterference-v0", tags=["multi_task", "interference", "meta_learning"])
class TaskInterferenceTask(TaskSpec):
    """Test multi-task attention by handling interleaved tasks on a shared grid.

    The agent must simultaneously pursue multiple objectives (goals)
    placed on the same grid. The tasks share spatial resources and may
    interfere with each other: progress toward one goal can conflict
    with progress toward another. The number of concurrent tasks scales
    with difficulty, demanding increasing attentional capacity and the
    ability to interleave sub-plans without losing track of any
    objective. This measures an agent's resistance to interference when
    managing competing task demands.

    Difficulty Levels:
        - easy: 10x10 grid with 2 interleaved tasks, 200 max steps.
        - medium: 13x13 grid with 2 interleaved tasks and more spatial
          separation, 300 max steps.
        - hard: 15x15 grid with 3 interleaved tasks, 400 max steps.
        - expert: 18x18 grid with 3 interleaved tasks requiring
          sustained multi-task coordination, 600 max steps.

    Capabilities Tested:
        - multi_task: The agent must manage and complete multiple
          distinct objectives within a single episode.
        - interference_resistance: The agent must avoid letting progress
          on one task degrade performance on another.
        - attention: The agent must allocate attention across multiple
          concurrent goals and switch contexts efficiently.

    Example:
        >>> env = agentick.make("TaskInterference-v0", difficulty="medium")
        >>> obs, info = env.reset(seed=42)
        >>> # Complete all interleaved goals without interference
    """

    name = "TaskInterference-v0"
    description = "Handle two interleaved tasks simultaneously"
    capability_tags = ["multi_task", "interference_resistance", "attention"]

    difficulty_configs = {
        "easy": DifficultyConfig(name="easy", grid_size=10, max_steps=200, params={"n_tasks": 2}),
        "medium": DifficultyConfig(
            name="medium", grid_size=13, max_steps=300, params={"n_tasks": 2}
        ),
        "hard": DifficultyConfig(name="hard", grid_size=15, max_steps=400, params={"n_tasks": 3}),
        "expert": DifficultyConfig(
            name="expert", grid_size=18, max_steps=600, params={"n_tasks": 3}
        ),
    }

    def generate(self, seed):
        """Generate a task interference instance.

        Creates a walled grid with multiple goal positions placed
        randomly, representing concurrent interleaved tasks. The number
        of goals (tasks) scales with difficulty. Goals are placed at
        random interior positions using the seeded RNG.

        Args:
            seed: Random seed for reproducible procedural generation.

        Returns:
            tuple: (grid, metadata) where grid is the initial Grid state
                with walls and multiple goals, and metadata contains
                agent_start, goal_positions, max_steps, and n_tasks.
        """
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        n_tasks = self.difficulty_config.params.get("n_tasks", 2)

        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        agent_pos = (1, 1)

        # Place multiple goals for different tasks
        goals = []
        for i in range(n_tasks):
            goal_x = rng.integers(2, size - 2)
            goal_y = rng.integers(2, size - 2)
            goals.append((goal_x, goal_y))
            grid.objects[goal_y, goal_x] = ObjectType.GOAL

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": goals,
            "max_steps": self.get_max_steps(),
            "n_tasks": n_tasks,
        }

    def compute_dense_reward(self, old_state, action, new_state, info):
        """Compute dense reward for a state transition.

        Uses a constant step penalty to encourage the agent to complete
        all interleaved tasks efficiently without interference.

        Args:
            old_state: State dict before the action.
            action: Action taken by the agent.
            new_state: State dict after the action.
            info: Additional info dict from the environment step.

        Returns:
            Constant penalty of -0.01 per step.
        """
        return -0.01

    def check_success(self, state):
        """Check if the task objective is complete.

        The task succeeds when all goal cells have been visited and
        cleared from the grid, indicating that every interleaved task
        has been completed.

        Args:
            state: Current state dict containing 'grid' and 'agent' keys.

        Returns:
            True if no goal cells remain on the grid (all tasks
            completed), False otherwise.
        """
        if "grid" not in state or "agent" not in state:
            return False

        grid = state["grid"]

        # Count remaining goals
        remaining_goals = 0
        for y in range(grid.height):
            for x in range(grid.width):
                if grid.objects[y, x] == ObjectType.GOAL:
                    remaining_goals += 1

        # Success if no goals remain (all completed)
        return remaining_goals == 0

    def get_optimal_return(self, difficulty=None):
        """Get the optimal (maximum possible) return for this task.

        Args:
            difficulty: Difficulty level string, or None to use the
                current instance difficulty.

        Returns:
            Optimal return of 1.0 (sparse success reward for completing
            all interleaved tasks).
        """
        return 1.0

    def get_random_baseline(self, difficulty=None):
        """Get expected return for a random agent baseline.

        A random agent is unlikely to visit all goal positions and
        complete every interleaved task, yielding near-zero expected
        return.

        Args:
            difficulty: Difficulty level string, or None to use the
                current instance difficulty.

        Returns:
            Expected random agent return of 0.0.
        """
        return 0.0
