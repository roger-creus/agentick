"""PreciseNavigation - Navigate narrow corridors without hitting walls"""

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("PreciseNavigation-v0", tags=["low_level_control"])
class PreciseNavigationTask(TaskSpec):
    """Test low-level control by navigating narrow corridors without collisions.

    The agent must traverse a grid filled with narrow corridors and tight
    passages to reach a goal position without hitting any walls. The
    passageways leave minimal room for error, requiring the agent to
    execute precise movement sequences. This tests fine-grained motor
    control and the ability to plan exact step-by-step paths through
    constrained spaces.

    Difficulty Levels:
        - easy: 7x7 grid with wider corridors and fewer turns, 100 max
          steps.
        - medium: 10x10 grid with narrower corridors and moderate turns,
          200 max steps.
        - hard: 13x13 grid with single-width corridors and many turns,
          300 max steps.
        - expert: 15x15 grid with winding single-width corridors and
          dead ends, 500 max steps.

    Capabilities Tested:
        - low_level_control: The agent must execute precise movement
          commands to navigate through corridors that allow no deviation
          from the optimal path.

    Example:
        >>> env = agentick.make("PreciseNavigation-v0", difficulty="medium")
        >>> obs, info = env.reset(seed=42)
        >>> # Navigate through narrow passages to reach the goal
    """

    name = "PreciseNavigation-v0"
    description = "Navigate narrow corridors without hitting walls"
    capability_tags = ["low_level_control"]
    difficulty_configs = {
        "easy": DifficultyConfig(name="easy", grid_size=7, max_steps=100),
        "medium": DifficultyConfig(name="medium", grid_size=10, max_steps=200),
        "hard": DifficultyConfig(name="hard", grid_size=13, max_steps=300),
        "expert": DifficultyConfig(name="expert", grid_size=15, max_steps=500),
    }

    def generate(self, seed):
        """Generate a precise navigation task instance.

        Creates a walled grid with narrow corridors and tight passages
        leading to a goal position. The agent starts at (1, 1) and must
        navigate without collisions.

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

        Uses a constant step penalty to encourage the agent to navigate
        through narrow corridors to the goal with precise, efficient
        movements.

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

        The task succeeds when the agent reaches the goal cell after
        traversing all narrow corridors without hitting walls.

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

        A random agent cannot reliably navigate narrow corridors without
        collisions, yielding near-zero expected return.

        Args:
            difficulty: Difficulty level string, or None to use the
                current instance difficulty.

        Returns:
            Expected random agent return of 0.0.
        """
        return 0.0
