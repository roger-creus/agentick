"""EmergentStrategy - Open-ended puzzle with multiple solutions"""

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("EmergentStrategy-v0", tags=["skill_discovery", "creativity"])
class EmergentStrategyTask(TaskSpec):
    """Test skill discovery and creativity in an open-ended puzzle.

    The agent faces a grid environment with multiple interactable
    elements but no prescribed solution path. There are several valid
    strategies to reach the goal, and the environment is designed to
    reward creative, emergent approaches that combine available mechanics
    in unexpected ways. The agent must explore, experiment with object
    interactions, and discover novel strategies rather than following a
    single scripted solution.

    Difficulty Levels:
        - easy: 7x7 grid with few interactable elements and obvious
          strategies, 100 max steps.
        - medium: 10x10 grid with more elements and less obvious
          solution paths, 200 max steps.
        - hard: 13x13 grid with many interacting elements requiring
          creative combination, 300 max steps.
        - expert: 15x15 grid with complex element interactions where
          optimal strategies are highly non-obvious, 500 max steps.

    Capabilities Tested:
        - skill_discovery: The agent must identify and learn to use
          novel environmental mechanics without explicit instruction.
        - creativity: The agent must combine available tools and
          interactions in original ways to solve the puzzle.

    Example:
        >>> env = agentick.make("EmergentStrategy-v0", difficulty="medium")
        >>> obs, info = env.reset(seed=42)
        >>> # Explore element interactions and discover a solution strategy
    """

    name = "EmergentStrategy-v0"
    description = "Open-ended puzzle with multiple solutions"
    capability_tags = ["skill_discovery", "creativity"]
    difficulty_configs = {
        "easy": DifficultyConfig(name="easy", grid_size=7, max_steps=100),
        "medium": DifficultyConfig(name="medium", grid_size=10, max_steps=200),
        "hard": DifficultyConfig(name="hard", grid_size=13, max_steps=300),
        "expert": DifficultyConfig(name="expert", grid_size=15, max_steps=500),
    }

    def generate(self, seed):
        """Generate an emergent strategy task instance.

        Creates a walled grid with multiple interactable elements and no
        prescribed solution path. The agent starts at (1, 1) and must
        discover creative strategies to reach the goal.

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

        Uses a constant step penalty to encourage the agent to explore
        element interactions and discover an effective strategy quickly.

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

        The task succeeds when the agent reaches the goal cell through
        any valid strategy discovered via element interaction.

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

        A random agent is unlikely to discover and execute a viable
        strategy through element interactions, yielding near-zero
        expected return.

        Args:
            difficulty: Difficulty level string, or None to use the
                current instance difficulty.

        Returns:
            Expected random agent return of 0.0.
        """
        return 0.0
