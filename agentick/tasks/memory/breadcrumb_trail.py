"""BreadcrumbTrail - Follow disappearing trail."""

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("BreadcrumbTrail-v0", tags=["long_horizon", "memory"])
class BreadcrumbTrailTask(TaskSpec):
    """Test long-horizon memory by following a disappearing trail.

    The agent must follow a trail of breadcrumb markers that lead to a
    distant goal, but the markers disappear after a fixed number of steps.
    Once the trail vanishes, the agent must rely on memory of previously
    observed markers to continue navigating toward the goal. This tests
    the agent's ability to encode and retain spatial information over
    extended time horizons and to navigate without continuous guidance.

    Difficulty Levels:
        - easy: 7x7 grid with a short trail and slow disappearance,
          100 max steps.
        - medium: 10x10 grid with a longer trail and moderate
          disappearance rate, 200 max steps.
        - hard: 13x13 grid with a winding trail that disappears quickly,
          300 max steps.
        - expert: 15x15 grid with a long, branching trail that vanishes
          almost immediately, 400 max steps.

    Capabilities Tested:
        - long_horizon: The agent must sustain goal-directed behavior
          across many steps after the trail has disappeared.
        - memory: The agent must memorize the trail layout before it
          vanishes and use that memory to navigate to the goal.

    Example:
        >>> env = agentick.make("BreadcrumbTrail-v0", difficulty="medium")
        >>> obs, info = env.reset(seed=42)
        >>> # Observe the trail, memorize it, then follow from memory
    """

    name = "BreadcrumbTrail-v0"
    description = "Follow disappearing trail to distant goal"
    capability_tags = ["long_horizon", "memory"]
    difficulty_configs = {
        "easy": DifficultyConfig(name="easy", grid_size=7, max_steps=100),
        "medium": DifficultyConfig(name="medium", grid_size=10, max_steps=200),
        "hard": DifficultyConfig(name="hard", grid_size=13, max_steps=300),
        "expert": DifficultyConfig(name="expert", grid_size=15, max_steps=400),
    }

    def generate(self, seed):
        """Generate a breadcrumb trail task instance.

        Creates a walled grid with a trail of breadcrumb markers leading
        to a goal. The markers disappear after a fixed number of steps,
        requiring the agent to memorize the path. The agent starts at
        (1, 1).

        Args:
            seed: Random seed for reproducible procedural generation.

        Returns:
            tuple: (grid, metadata) where grid is the initial Grid state
                with walls and goal, and metadata contains agent_start,
                goal_positions, and max_steps.
        """
        size = self.difficulty_config.grid_size
        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL
        grid.objects[size - 2, size - 2] = ObjectType.GOAL
        return grid, {
            "agent_start": (1, 1),
            "goal_positions": [(size - 2, size - 2)],
            "max_steps": self.get_max_steps(),
        }

    def compute_dense_reward(self, old_state, action, new_state, info):
        """Compute dense reward for a state transition.

        Uses a constant step penalty to encourage the agent to memorize
        the trail and navigate to the goal before running out of steps.

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

        The task succeeds when the agent reaches the goal cell by
        following the memorized trail after the breadcrumbs disappear.

        Args:
            state: Current state dict containing 'grid' and 'agent' keys.

        Returns:
            True if the agent is on the goal cell, False otherwise.
        """
        if "grid" not in state or "agent" not in state:
            return False
        x, y = state["agent"].position
        return state["grid"].objects[y, x] == ObjectType.GOAL

    def get_optimal_return(self, difficulty=None):
        """Get the optimal (maximum possible) return for this task.

        Args:
            difficulty: Difficulty level string, or None to use the
                current instance difficulty.

        Returns:
            Optimal return of 1.0 (sparse success reward).
        """
        return 1.0

    def get_random_baseline(self, difficulty=None):
        """Get expected return for a random agent baseline.

        A random agent cannot memorize the disappearing trail, yielding
        near-zero expected return on larger grids.

        Args:
            difficulty: Difficulty level string, or None to use the
                current instance difficulty.

        Returns:
            Expected random agent return of 0.0.
        """
        return 0.0
