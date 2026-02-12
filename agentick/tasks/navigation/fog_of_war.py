"""FogOfWarExploration task - Explore grid with limited visibility."""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("FogOfWarExploration-v0", tags=["exploration", "memory", "navigation"])
class FogOfWarExplorationTask(TaskSpec):
    """Test exploration and memory under limited visibility fog-of-war conditions.

    The agent must navigate a partially observable grid where only cells
    within a limited visibility radius are visible at any time. The rest
    of the grid is hidden under fog of war, requiring the agent to
    explore systematically, remember previously visited areas, and build
    an internal map to locate the goal. The visibility radius and grid
    size scale with difficulty, demanding increasingly effective
    exploration strategies and spatial memory. Random interior walls add
    navigational complexity.

    Difficulty Levels:
        - easy: 7x7 grid with visibility radius 2, 100 max steps.
        - medium: 10x10 grid with visibility radius 2, 200 max steps.
        - hard: 13x13 grid with visibility radius 1 (very limited),
          300 max steps.
        - expert: 15x15 grid with visibility radius 1 requiring
          extensive systematic exploration, 400 max steps.

    Capabilities Tested:
        - exploration: The agent must systematically uncover the grid
          to locate the hidden goal under fog-of-war conditions.
        - memory: The agent must remember previously explored regions
          to avoid revisiting them and to build a mental map.
        - navigation: The agent must navigate efficiently through
          partially revealed terrain with random wall obstacles.

    Example:
        >>> env = agentick.make("FogOfWarExploration-v0", difficulty="medium")
        >>> obs, info = env.reset(seed=42)
        >>> # Explore the foggy grid to find and reach the hidden goal
    """

    name = "FogOfWarExploration-v0"
    description = "Explore grid with limited visibility, find goal"
    capability_tags = ["exploration", "memory", "navigation"]

    difficulty_configs = {
        "easy": DifficultyConfig(name="easy", grid_size=7, max_steps=100, params={"visibility": 2}),
        "medium": DifficultyConfig(
            name="medium", grid_size=10, max_steps=200, params={"visibility": 2}
        ),
        "hard": DifficultyConfig(
            name="hard", grid_size=13, max_steps=300, params={"visibility": 1}
        ),
        "expert": DifficultyConfig(
            name="expert", grid_size=15, max_steps=400, params={"visibility": 1}
        ),
    }

    def generate(self, seed):
        """Generate a fog-of-war exploration task instance.

        Creates a walled grid with randomly placed interior walls and a
        goal at a reachable position. The generator tries up to 10
        attempts to produce a solvable instance where the goal is
        reachable from the agent start via flood fill. Falls back to a
        simple open layout if all attempts fail.

        Args:
            seed: Random seed for reproducible procedural generation.

        Returns:
            tuple: (grid, metadata) where grid is the initial Grid state
                with walls and goal, and metadata contains agent_start,
                goal_positions, and max_steps.
        """
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size

        # Try multiple times to generate a valid instance
        max_attempts = 10
        for attempt in range(max_attempts):
            grid = Grid(size, size)
            grid.terrain[0, :] = CellType.WALL
            grid.terrain[-1, :] = CellType.WALL
            grid.terrain[:, 0] = CellType.WALL
            grid.terrain[:, -1] = CellType.WALL

            # Add some walls
            for _ in range(size):
                x, y = rng.integers(1, size - 1), rng.integers(1, size - 1)
                grid.terrain[y, x] = CellType.WALL

            agent_pos = (1, 1)

            # Find reachable positions from agent
            reachable = grid.flood_fill(agent_pos)
            if len(reachable) < 2:
                continue  # Try again

            # Place goal in a reachable position
            reachable_list = list(reachable)
            reachable_list.remove(agent_pos)
            goal_pos = reachable_list[rng.choice(len(reachable_list))]
            goal_x, goal_y = goal_pos
            grid.objects[goal_y, goal_x] = ObjectType.GOAL

            return grid, {
                "agent_start": agent_pos,
                "goal_positions": [(goal_x, goal_y)],
                "max_steps": self.get_max_steps(),
            }

        # Fallback: simple solvable instance
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

        Uses a constant step penalty to encourage efficient exploration
        and goal-finding under fog-of-war visibility constraints.

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
        exploring the fog-covered grid.

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

        A random agent explores inefficiently under fog-of-war
        conditions, yielding near-zero expected return.

        Args:
            difficulty: Difficulty level string, or None to use the
                current instance difficulty.

        Returns:
            Expected random agent return of 0.0.
        """
        return 0.0
