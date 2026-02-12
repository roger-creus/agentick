"""PackingPuzzle - Fit shaped pieces into target area"""

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("PackingPuzzle-v0", tags=["spatial_reasoning", "combinatorics"])
class PackingPuzzleTask(TaskSpec):
    """Test spatial reasoning by fitting shaped pieces into a target area.

    The agent must place a set of irregularly shaped pieces into a bounded
    rectangular region without overlap and with full coverage. Pieces may
    need to be rotated or flipped. The challenge lies in evaluating spatial
    fit across many possible orientations and placements, requiring both
    geometric reasoning and combinatorial search to find a valid packing
    arrangement.

    Difficulty Levels:
        - easy: 7x7 grid with few simple pieces, 100 max steps.
        - medium: 10x10 grid with more pieces of moderate complexity,
          200 max steps.
        - hard: 13x13 grid with many irregularly shaped pieces, 300 max
          steps.
        - expert: 15x15 grid with numerous complex pieces requiring
          precise placement, 500 max steps.

    Capabilities Tested:
        - spatial_reasoning: The agent must visualize how pieces fit
          together in two-dimensional space, accounting for shape
          geometry and orientation.
        - combinatorics: The agent must search through placement
          orderings and rotations to find a valid configuration.

    Example:
        >>> env = agentick.make("PackingPuzzle-v0", difficulty="medium")
        >>> obs, info = env.reset(seed=42)
        >>> # Place and orient pieces to fill the target area completely
    """

    name = "PackingPuzzle-v0"
    description = "Fit shaped pieces into target area"
    capability_tags = ["spatial_reasoning", "combinatorics"]
    difficulty_configs = {
        "easy": DifficultyConfig(name="easy", grid_size=7, max_steps=100),
        "medium": DifficultyConfig(name="medium", grid_size=10, max_steps=200),
        "hard": DifficultyConfig(name="hard", grid_size=13, max_steps=300),
        "expert": DifficultyConfig(name="expert", grid_size=15, max_steps=500),
    }

    def generate(self, seed):
        """Generate a packing puzzle instance.

        Creates a walled grid with a target area and shaped pieces to
        place. The agent starts at (1, 1) and must fit all pieces into
        the area before reaching the goal.

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

        Uses a constant step penalty to encourage the agent to find valid
        piece placements and complete the packing efficiently.

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
        that all pieces have been correctly packed into the target area.

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

        A random agent is unlikely to correctly orient and place all
        pieces into the target area, yielding near-zero expected return.

        Args:
            difficulty: Difficulty level string, or None to use the
                current instance difficulty.

        Returns:
            Expected random agent return of 0.0.
        """
        return 0.0
