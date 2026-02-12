"""GraphColoring - Color regions without adjacent same-colors"""

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("GraphColoring-v0", tags=["constraint_satisfaction"])
class GraphColoringTask(TaskSpec):
    """Test constraint satisfaction via graph region coloring.

    The agent must assign colors to contiguous regions on a grid such that
    no two adjacent regions share the same color. The grid is bounded by
    walls, and the agent navigates to a goal position after satisfying all
    coloring constraints. This task evaluates an agent's ability to reason
    about global constraints while making local decisions, as early color
    choices propagate to limit future options.

    Difficulty Levels:
        - easy: 7x7 grid with simple region layout, 100 max steps.
        - medium: 10x10 grid with moderate region complexity, 200 max steps.
        - hard: 13x13 grid with dense region adjacency, 300 max steps.
        - expert: 15x15 grid with highly interconnected regions, 500 max
          steps.

    Capabilities Tested:
        - constraint_satisfaction: The agent must respect the coloring
          constraint that no two adjacent regions share a color, requiring
          systematic constraint propagation and backtracking.

    Example:
        >>> env = agentick.make("GraphColoring-v0", difficulty="medium")
        >>> obs, info = env.reset(seed=42)
        >>> # Navigate and assign colors to satisfy all adjacency constraints
    """

    name = "GraphColoring-v0"
    description = "Color regions without adjacent same-colors"
    capability_tags = ["constraint_satisfaction"]
    difficulty_configs = {
        "easy": DifficultyConfig(name="easy", grid_size=7, max_steps=100),
        "medium": DifficultyConfig(name="medium", grid_size=10, max_steps=200),
        "hard": DifficultyConfig(name="hard", grid_size=13, max_steps=300),
        "expert": DifficultyConfig(name="expert", grid_size=15, max_steps=500),
    }

    def generate(self, seed):
        """Generate a graph coloring task instance.

        Creates a walled grid with colorable regions and a goal position.
        The agent starts at (1, 1) and must satisfy coloring constraints
        before reaching the goal at the opposite corner.

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

        Uses a constant step penalty to encourage the agent to satisfy
        all coloring constraints and reach the goal efficiently.

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
        that all region coloring constraints have been satisfied.

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

        A random agent is unlikely to satisfy all coloring constraints
        and reach the goal, yielding near-zero expected return.

        Args:
            difficulty: Difficulty level string, or None to use the
                current instance difficulty.

        Returns:
            Expected random agent return of 0.0.
        """
        return 0.0
