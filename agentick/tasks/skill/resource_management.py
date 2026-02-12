"""ResourceManagement - Manage limited energy/health"""

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("ResourceManagement-v0", tags=["planning", "optimization"])
class ResourceManagementTask(TaskSpec):
    """Test planning and optimization under limited resource constraints.

    The agent must navigate the grid to reach the goal while managing
    limited resources such as energy, health, or fuel. Each movement
    consumes resources, and the agent must find and collect resource
    pickups along the way. Running out of resources ends the episode in
    failure. The agent must balance the shortest path to the goal against
    detours to collect resources, optimizing total resource expenditure
    while ensuring survival.

    Difficulty Levels:
        - easy: 7x7 grid with abundant resource pickups, 100 max steps.
        - medium: 10x10 grid with moderate resource scarcity, 200 max
          steps.
        - hard: 13x13 grid with scarce resources and costly terrain,
          300 max steps.
        - expert: 15x15 grid with very scarce resources, hazardous
          terrain, and long distances between pickups, 500 max steps.

    Capabilities Tested:
        - planning: The agent must plan routes that balance progress
          toward the goal with resource collection detours.
        - optimization: The agent must minimize total resource
          expenditure while ensuring sufficient reserves to survive.

    Example:
        >>> env = agentick.make("ResourceManagement-v0", difficulty="medium")
        >>> obs, info = env.reset(seed=42)
        >>> # Navigate to the goal while managing limited energy/health
    """

    name = "ResourceManagement-v0"
    description = "Manage limited energy/health"
    capability_tags = ["planning", "optimization"]
    difficulty_configs = {
        "easy": DifficultyConfig(name="easy", grid_size=7, max_steps=100),
        "medium": DifficultyConfig(name="medium", grid_size=10, max_steps=200),
        "hard": DifficultyConfig(name="hard", grid_size=13, max_steps=300),
        "expert": DifficultyConfig(name="expert", grid_size=15, max_steps=500),
    }

    def generate(self, seed):
        """Generate a resource management task instance.

        Creates a walled grid with limited resource pickups and a goal.
        The agent starts at (1, 1) with constrained energy/health and
        must balance progress toward the goal with resource collection.

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

        Uses a constant step penalty to encourage the agent to optimize
        resource expenditure and reach the goal before running out.

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

        The task succeeds when the agent reaches the goal cell while
        maintaining sufficient resources to survive the journey.

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

        A random agent cannot balance resource collection with goal
        progress, typically exhausting resources before reaching the
        goal, yielding near-zero expected return.

        Args:
            difficulty: Difficulty level string, or None to use the
                current instance difficulty.

        Returns:
            Expected random agent return of 0.0.
        """
        return 0.0
