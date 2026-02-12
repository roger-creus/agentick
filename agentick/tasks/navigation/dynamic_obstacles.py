"""DynamicObstacles task - Navigate while obstacles move."""

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("DynamicObstacles-v0", tags=["reactive_planning", "navigation"])
class DynamicObstaclesTask(TaskSpec):
    """Test reactive planning by navigating through a grid with moving obstacles.

    The agent must reach a goal while avoiding obstacles that move in
    patterns across the grid. Unlike static navigation tasks, the agent
    must continuously re-plan its path as obstacles shift positions each
    step. The number of moving obstacles scales with difficulty, requiring
    increasingly sophisticated real-time avoidance and path adaptation.
    This measures an agent's ability to plan reactively in a dynamic
    environment where the obstacle configuration changes over time.

    Difficulty Levels:
        - easy: 7x7 grid with 2 moving obstacles, 50 max steps.
        - medium: 10x10 grid with 3 moving obstacles, 100 max steps.
        - hard: 13x13 grid with 4 moving obstacles, 150 max steps.
        - expert: 15x15 grid with 5 moving obstacles requiring precise
          timing and avoidance, 200 max steps.

    Capabilities Tested:
        - reactive_planning: The agent must continuously adapt its plan
          in response to obstacle movements.
        - navigation: The agent must find and follow a safe path to the
          goal while obstacles are in motion.

    Example:
        >>> env = agentick.make("DynamicObstacles-v0", difficulty="medium")
        >>> obs, info = env.reset(seed=42)
        >>> # Navigate to the goal while avoiding 3 moving obstacles
    """

    name = "DynamicObstacles-v0"
    description = "Navigate while obstacles move"
    capability_tags = ["reactive_planning", "navigation"]

    difficulty_configs = {
        "easy": DifficultyConfig(name="easy", grid_size=7, max_steps=50, params={"n_obstacles": 2}),
        "medium": DifficultyConfig(
            name="medium", grid_size=10, max_steps=100, params={"n_obstacles": 3}
        ),
        "hard": DifficultyConfig(
            name="hard", grid_size=13, max_steps=150, params={"n_obstacles": 4}
        ),
        "expert": DifficultyConfig(
            name="expert", grid_size=15, max_steps=200, params={"n_obstacles": 5}
        ),
    }

    def generate(self, seed):
        """Generate a dynamic obstacles task instance.

        Creates a walled grid with a goal at the far corner. Moving
        obstacles are configured through the difficulty parameters but
        initialized dynamically during episode execution. The grid
        itself provides the static layout.

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

        agent_pos = (1, 1)
        goal_pos = (size - 2, size - 2)
        grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.GOAL

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [goal_pos],
            "max_steps": self.get_max_steps(),
        }

    def compute_dense_reward(self, old_state, action, new_state, info):
        """Compute dense reward for a state transition.

        Uses a constant step penalty to encourage the agent to navigate
        to the goal efficiently while avoiding moving obstacles.

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
        successfully avoiding all moving obstacles.

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

        A random agent is likely to collide with moving obstacles or
        fail to reach the goal, yielding near-zero expected return.

        Args:
            difficulty: Difficulty level string, or None to use the
                current instance difficulty.

        Returns:
            Expected random agent return of 0.0.
        """
        return 0.0
