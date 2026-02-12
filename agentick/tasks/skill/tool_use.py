"""ToolUse - Craft/find tools to overcome obstacles"""

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("ToolUse-v0", tags=["skill_discovery", "tool_use"])
class ToolUseTask(TaskSpec):
    """Test skill discovery and tool use by crafting or finding tools.

    The agent encounters obstacles on the grid that cannot be overcome
    with basic movement alone. Instead, the agent must find or craft
    tools (such as keys, bridges, or hammers) to bypass specific
    obstacle types. Each tool has a specific function, and the agent
    must discover which tool solves which obstacle through
    experimentation. This evaluates the ability to recognize when a
    tool is needed, locate or create it, and apply it correctly.

    Difficulty Levels:
        - easy: 7x7 grid with one tool type and obvious obstacles,
          100 max steps.
        - medium: 10x10 grid with multiple tool types and varied
          obstacles, 200 max steps.
        - hard: 13x13 grid with craftable tools requiring ingredient
          collection, 300 max steps.
        - expert: 15x15 grid with complex tool chains where one tool
          is needed to obtain another, 500 max steps.

    Capabilities Tested:
        - skill_discovery: The agent must discover through exploration
          which tools exist and what functions they serve.
        - tool_use: The agent must acquire the correct tool and apply
          it to the appropriate obstacle to make progress.

    Example:
        >>> env = agentick.make("ToolUse-v0", difficulty="medium")
        >>> obs, info = env.reset(seed=42)
        >>> # Find or craft the right tools to overcome each obstacle
    """

    name = "ToolUse-v0"
    description = "Craft/find tools to overcome obstacles"
    capability_tags = ["skill_discovery", "tool_use"]
    difficulty_configs = {
        "easy": DifficultyConfig(name="easy", grid_size=7, max_steps=100),
        "medium": DifficultyConfig(name="medium", grid_size=10, max_steps=200),
        "hard": DifficultyConfig(name="hard", grid_size=13, max_steps=300),
        "expert": DifficultyConfig(name="expert", grid_size=15, max_steps=500),
    }

    def generate(self, seed):
        """Generate a tool use task instance.

        Creates a walled grid with obstacles that require specific tools
        to overcome. The agent starts at (1, 1) and must find or craft
        the correct tools to bypass obstacles and reach the goal.

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

        Uses a constant step penalty to encourage the agent to discover
        and apply the correct tools to obstacles efficiently.

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
        using the correct tools to overcome all blocking obstacles.

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

        A random agent cannot discover which tools solve which obstacles
        or acquire them in the right order, yielding near-zero expected
        return.

        Args:
            difficulty: Difficulty level string, or None to use the
                current instance difficulty.

        Returns:
            Expected random agent return of 0.0.
        """
        return 0.0
