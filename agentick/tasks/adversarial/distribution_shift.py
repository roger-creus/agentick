"""Distribution Shift - OOD generalization test."""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("DistributionShift-v0", tags=["generalization", "ood", "robustness"])
class DistributionShiftTask(TaskSpec):
    """Test generalization under out-of-distribution environment conditions.

    The agent is evaluated on grid instances that differ from the training
    distribution along a controlled axis. Shift types include changes in
    grid size, wall density, layout topology, or a mix of all three. The
    generator applies the specified shift to produce environments that
    challenge learned heuristics. This task directly measures an agent's
    ability to generalize beyond its training distribution and maintain
    performance under novel conditions.

    Difficulty Levels:
        - easy: 7x7 grid with size-based shift only, 150 max steps.
        - medium: 12x12 grid with density-based shift (extra interior
          walls), 250 max steps.
        - hard: 18x18 grid with layout-based shift (altered topology),
          350 max steps.
        - expert: 25x25 grid with mixed shift combining all shift types,
          500 max steps.

    Capabilities Tested:
        - generalization: The agent must perform well on environments
          that differ from those seen during training.
        - robustness: The agent must maintain navigation competence when
          wall configurations and grid properties change unexpectedly.
        - ood: The agent must handle out-of-distribution instances
          without catastrophic performance degradation.

    Example:
        >>> env = agentick.make("DistributionShift-v0", difficulty="hard")
        >>> obs, info = env.reset(seed=42)
        >>> # Navigate to goal despite unfamiliar layout topology
    """

    name = "DistributionShift-v0"
    description = "Test generalization to out-of-distribution instances"
    capability_tags = ["generalization", "robustness", "ood"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy", grid_size=7, max_steps=150, params={"shift_type": "size"}
        ),
        "medium": DifficultyConfig(
            name="medium", grid_size=12, max_steps=250, params={"shift_type": "density"}
        ),
        "hard": DifficultyConfig(
            name="hard", grid_size=18, max_steps=350, params={"shift_type": "layout"}
        ),
        "expert": DifficultyConfig(
            name="expert", grid_size=25, max_steps=500, params={"shift_type": "mixed"}
        ),
    }

    def generate(self, seed):
        """Generate an out-of-distribution task instance.

        Creates a walled grid with a distribution shift applied based on
        the configured shift type. For density shifts, additional walls
        are placed randomly. The generator retries up to 10 times to
        ensure the goal is reachable from the agent start, falling back
        to a simple open layout if all attempts fail.

        Args:
            seed: Random seed for reproducible procedural generation.

        Returns:
            tuple: (grid, metadata) where grid is the initial Grid state
                with walls and goal, and metadata contains agent_start,
                goal_positions, max_steps, and shift_type.
        """
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        shift_type = self.difficulty_config.params.get("shift_type", "size")

        max_attempts = 10
        for attempt in range(max_attempts):
            grid = Grid(size, size)
            grid.terrain[0, :] = CellType.WALL
            grid.terrain[-1, :] = CellType.WALL
            grid.terrain[:, 0] = CellType.WALL
            grid.terrain[:, -1] = CellType.WALL

            # Apply distribution shift
            if shift_type == "density":
                # Higher wall density
                for _ in range(size):
                    x, y = rng.integers(1, size - 1, 2)
                    grid.terrain[y, x] = CellType.WALL

            agent_pos = (1, 1)
            goal_pos = (size - 2, size - 2)

            # Verify solvable
            reachable = grid.flood_fill(agent_pos)
            if goal_pos in reachable:
                grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.GOAL
                return grid, {
                    "agent_start": agent_pos,
                    "goal_positions": [goal_pos],
                    "max_steps": self.get_max_steps(),
                    "shift_type": shift_type,
                }

        # Fallback
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
            "shift_type": shift_type,
        }

    def compute_dense_reward(self, old_state, action, new_state, info):
        """Compute dense reward for a state transition.

        Uses a constant step penalty to encourage efficient navigation
        despite the distribution shift in the environment.

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

        The task succeeds when the agent reaches the goal cell despite
        the out-of-distribution environment conditions.

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

        A random agent has very low probability of reaching the goal
        in an out-of-distribution environment, yielding near-zero
        expected return.

        Args:
            difficulty: Difficulty level string, or None to use the
                current instance difficulty.

        Returns:
            Expected random agent return of 0.0.
        """
        return 0.0
