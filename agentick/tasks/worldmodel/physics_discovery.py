"""Physics Discovery - Discover hidden physics rules."""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("PhysicsDiscovery-v0", tags=["world_model", "exploration", "physics"])
class PhysicsDiscoveryTask(TaskSpec):
    """Test world-model building by discovering hidden physics rules through exploration.

    The agent must navigate a grid where certain tiles have hidden
    physics properties (gravity, slippery ice, conveyors) that are not
    revealed in advance. The agent must discover these properties
    through trial-and-error interaction, build an accurate world model
    of how different tile types affect movement, and then exploit that
    knowledge to reach the goal. The physics type and grid complexity
    scale with difficulty, with the expert level combining multiple
    physics types simultaneously.

    Difficulty Levels:
        - easy: 7x7 grid with gravity physics, 150 max steps.
        - medium: 10x10 grid with slippery (ice) surfaces, 250 max
          steps.
        - hard: 13x13 grid with conveyor belts, 350 max steps.
        - expert: 15x15 grid with mixed physics types requiring
          discovery of multiple interacting rules, 500 max steps.

    Capabilities Tested:
        - world_model: The agent must construct a model of hidden
          physics rules from observed movement outcomes on special tiles.
        - exploration: The agent must deliberately explore to discover
          which tiles have special physics properties.
        - adaptation: The agent must adapt its navigation strategy once
          it understands how the hidden physics affect movement.

    Example:
        >>> env = agentick.make("PhysicsDiscovery-v0", difficulty="medium")
        >>> obs, info = env.reset(seed=42)
        >>> # Discover slippery ice tiles and navigate to the goal
    """

    name = "PhysicsDiscovery-v0"
    description = "Discover hidden physics and reach goal"
    capability_tags = ["world_model", "exploration", "adaptation"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy", grid_size=7, max_steps=150, params={"physics_type": "gravity"}
        ),
        "medium": DifficultyConfig(
            name="medium", grid_size=10, max_steps=250, params={"physics_type": "slippery"}
        ),
        "hard": DifficultyConfig(
            name="hard", grid_size=13, max_steps=350, params={"physics_type": "conveyor"}
        ),
        "expert": DifficultyConfig(
            name="expert", grid_size=15, max_steps=500, params={"physics_type": "mixed"}
        ),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.physics_type = None

    def generate(self, seed):
        """Generate a physics discovery task instance.

        Creates a walled grid with a goal at the far corner and special
        physics tiles based on the configured physics type. For slippery
        physics, ice tiles are randomly placed on empty cells. The
        physics type is stored for use during reward computation and
        success checking.

        Args:
            seed: Random seed for reproducible procedural generation.

        Returns:
            tuple: (grid, metadata) where grid is the initial Grid state
                with walls, goal, and physics tiles, and metadata
                contains agent_start, goal_positions, max_steps, and
                physics_type.
        """
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        self.physics_type = self.difficulty_config.params.get("physics_type", "gravity")

        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        agent_pos = (1, 1)
        goal_pos = (size - 2, size - 2)
        grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.GOAL

        # Add physics tiles
        if self.physics_type == "slippery":
            for _ in range(size):
                x, y = rng.integers(1, size - 1, 2)
                if grid.terrain[y, x] == CellType.EMPTY:
                    grid.terrain[y, x] = CellType.ICE

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [goal_pos],
            "max_steps": self.get_max_steps(),
            "physics_type": self.physics_type,
        }

    def compute_dense_reward(self, old_state, action, new_state, info):
        """Compute dense reward for a state transition.

        Uses a constant step penalty to encourage the agent to discover
        hidden physics rules and navigate to the goal efficiently.

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
        discovering and accounting for hidden physics rules.

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

        A random agent cannot discover or leverage hidden physics rules,
        yielding near-zero expected return.

        Args:
            difficulty: Difficulty level string, or None to use the
                current instance difficulty.

        Returns:
            Expected random agent return of 0.0.
        """
        return 0.0
