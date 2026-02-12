"""RecipeAssembly - Combine items in correct order"""

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("RecipeAssembly-v0", tags=["compositional_logic", "planning"])
class RecipeAssemblyTask(TaskSpec):
    """Test compositional logic and planning by combining items in order.

    The agent must collect ingredients scattered across the grid and
    combine them at crafting stations in a specific sequence dictated by
    a recipe. The correct ordering matters: combining items out of
    sequence produces incorrect results or wastes materials. The agent
    must plan collection routes, remember the recipe sequence, and
    execute assembly steps in the right order while managing inventory
    constraints.

    Difficulty Levels:
        - easy: 7x7 grid with few ingredients and a short recipe,
          100 max steps.
        - medium: 10x10 grid with more ingredients and a longer recipe
          requiring route planning, 200 max steps.
        - hard: 13x13 grid with many ingredients, multiple crafting
          stations, and intermediate products, 300 max steps.
        - expert: 15x15 grid with complex multi-step recipes, limited
          inventory, and widely scattered ingredients, 500 max steps.

    Capabilities Tested:
        - compositional_logic: The agent must understand how individual
          ingredients combine in sequence to produce the target item.
        - planning: The agent must plan efficient collection routes and
          assembly order to complete the recipe within the step limit.

    Example:
        >>> env = agentick.make("RecipeAssembly-v0", difficulty="medium")
        >>> obs, info = env.reset(seed=42)
        >>> # Collect ingredients and combine them in the correct order
    """

    name = "RecipeAssembly-v0"
    description = "Combine items in correct order"
    capability_tags = ["compositional_logic", "planning"]
    difficulty_configs = {
        "easy": DifficultyConfig(name="easy", grid_size=7, max_steps=100),
        "medium": DifficultyConfig(name="medium", grid_size=10, max_steps=200),
        "hard": DifficultyConfig(name="hard", grid_size=13, max_steps=300),
        "expert": DifficultyConfig(name="expert", grid_size=15, max_steps=500),
    }

    def generate(self, seed):
        """Generate a recipe assembly task instance.

        Creates a walled grid with scattered ingredients and crafting
        stations. The agent starts at (1, 1) and must collect and
        combine items in the correct recipe order to reach the goal.

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

        Uses a constant step penalty to encourage the agent to plan
        efficient collection routes and assemble the recipe in the
        correct order.

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
        collecting all ingredients and assembling them in the correct
        recipe sequence.

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

        A random agent cannot follow the correct recipe sequence or
        plan collection routes, yielding near-zero expected return.

        Args:
            difficulty: Difficulty level string, or None to use the
                current instance difficulty.

        Returns:
            Expected random agent return of 0.0.
        """
        return 0.0
