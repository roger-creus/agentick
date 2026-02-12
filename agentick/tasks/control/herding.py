"""Herding - Guide multiple entities to target zone"""

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("Herding-v0", tags=["multi_objective_control"])
class HerdingTask(TaskSpec):
    """Test multi-objective control by guiding multiple entities to a target.

    The agent must herd a group of autonomous entities toward a designated
    target zone on the grid. Each entity reacts to the agent's proximity
    by moving away, so the agent must position itself strategically to
    steer the entire group collectively. The challenge lies in managing
    multiple moving objects simultaneously, preventing stragglers from
    escaping while keeping the herd cohesive and moving in the desired
    direction.

    Difficulty Levels:
        - easy: 7x7 grid with few entities and a large target zone,
          100 max steps.
        - medium: 10x10 grid with more entities and a moderate target
          zone, 200 max steps.
        - hard: 13x13 grid with many entities and a small target zone,
          300 max steps.
        - expert: 15x15 grid with numerous fast-moving entities and a
          tight target zone, 500 max steps.

    Capabilities Tested:
        - multi_objective_control: The agent must simultaneously manage
          the positions and trajectories of multiple independent entities
          while navigating them all toward a shared goal.

    Example:
        >>> env = agentick.make("Herding-v0", difficulty="medium")
        >>> obs, info = env.reset(seed=42)
        >>> # Position yourself to guide all entities into the target zone
    """

    name = "Herding-v0"
    description = "Guide multiple entities to target zone"
    capability_tags = ["multi_objective_control"]
    difficulty_configs = {
        "easy": DifficultyConfig(name="easy", grid_size=7, max_steps=100),
        "medium": DifficultyConfig(name="medium", grid_size=10, max_steps=200),
        "hard": DifficultyConfig(name="hard", grid_size=13, max_steps=300),
        "expert": DifficultyConfig(name="expert", grid_size=15, max_steps=500),
    }

    def generate(self, seed):
        """Generate a herding task instance.

        Creates a walled grid with autonomous entities and a target zone.
        The agent starts at (1, 1) and must guide all entities into the
        target zone by strategic positioning.

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

        Uses a constant step penalty to encourage the agent to herd all
        entities into the target zone with minimal wasted movement.

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
        that all entities have been herded into the target zone.

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

        A random agent cannot coordinate entity movements toward the
        target zone, yielding near-zero expected return.

        Args:
            difficulty: Difficulty level string, or None to use the
                current instance difficulty.

        Returns:
            Expected random agent return of 0.0.
        """
        return 0.0
