"""LightsOut - Classic lights-out puzzle on grid"""

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("LightsOut-v0", tags=["combinatorial_logic"])
class LightsOutTask(TaskSpec):
    """Test combinatorial logic through the classic lights-out puzzle.

    The agent faces a grid of cells that are either lit or unlit. Toggling
    a cell also toggles its orthogonal neighbors, creating cascading state
    changes. The objective is to turn all lights off by selecting the
    correct sequence of toggles. This requires the agent to reason about
    indirect effects and plan toggle sequences that account for
    interdependencies between cells.

    Difficulty Levels:
        - easy: 7x7 grid with fewer lit cells, 100 max steps.
        - medium: 10x10 grid with moderate initial lit pattern, 200 max
          steps.
        - hard: 13x13 grid with dense initial lit pattern, 300 max steps.
        - expert: 15x15 grid with near-full initial lit pattern, 500 max
          steps.

    Capabilities Tested:
        - combinatorial_logic: The agent must determine which combination
          of cell toggles will clear the board, reasoning about how each
          toggle propagates to neighboring cells.

    Example:
        >>> env = agentick.make("LightsOut-v0", difficulty="medium")
        >>> obs, info = env.reset(seed=42)
        >>> # Toggle cells strategically to turn all lights off
    """

    name = "LightsOut-v0"
    description = "Classic lights-out puzzle on grid"
    capability_tags = ["combinatorial_logic"]
    difficulty_configs = {
        "easy": DifficultyConfig(name="easy", grid_size=7, max_steps=100),
        "medium": DifficultyConfig(name="medium", grid_size=10, max_steps=200),
        "hard": DifficultyConfig(name="hard", grid_size=13, max_steps=300),
        "expert": DifficultyConfig(name="expert", grid_size=15, max_steps=500),
    }

    def generate(self, seed):
        """Generate a lights-out puzzle instance.

        Creates a walled grid with lit/unlit cells and a goal position.
        The agent starts at (1, 1) and must toggle cells to turn all
        lights off before reaching the goal.

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

        Uses a constant step penalty to encourage the agent to determine
        the correct toggle sequence and reach the goal quickly.

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

        The task succeeds when the agent reaches the goal cell, indicating
        that all lights have been turned off via correct toggle sequences.

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

        A random agent is unlikely to find the correct toggle combination
        to clear all lights, yielding near-zero expected return.

        Args:
            difficulty: Difficulty level string, or None to use the
                current instance difficulty.

        Returns:
            Expected random agent return of 0.0.
        """
        return 0.0
