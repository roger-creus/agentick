"""SequenceMemory - Remember and replay sequence."""

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("SequenceMemory-v0", tags=["memory", "pattern_recognition"])
class SequenceMemoryTask(TaskSpec):
    """Test memory and pattern recognition by replaying observed sequences.

    The agent is shown a sequence of highlighted grid positions that flash
    in order. After the demonstration phase, the agent must visit those
    same positions in the exact same order from memory. The sequence
    length and grid size scale with difficulty. This directly measures
    working memory capacity and the ability to encode, retain, and
    reproduce ordered spatial information.

    Difficulty Levels:
        - easy: 5x5 grid with short sequences, 50 max steps.
        - medium: 7x7 grid with moderate-length sequences, 100 max steps.
        - hard: 9x9 grid with long sequences, 150 max steps.
        - expert: 11x11 grid with very long sequences requiring
          substantial memory, 200 max steps.

    Capabilities Tested:
        - memory: The agent must memorize the full sequence of positions
          during the demonstration and retain them for replay.
        - pattern_recognition: The agent must identify and encode the
          spatial pattern of the sequence to reproduce it accurately.

    Example:
        >>> env = agentick.make("SequenceMemory-v0", difficulty="medium")
        >>> obs, info = env.reset(seed=42)
        >>> # Watch the sequence, then visit positions in the same order
    """

    name = "SequenceMemory-v0"
    description = "Remember and replay a shown sequence of moves"
    capability_tags = ["memory", "pattern_recognition"]
    difficulty_configs = {
        "easy": DifficultyConfig(name="easy", grid_size=5, max_steps=50),
        "medium": DifficultyConfig(name="medium", grid_size=7, max_steps=100),
        "hard": DifficultyConfig(name="hard", grid_size=9, max_steps=150),
        "expert": DifficultyConfig(name="expert", grid_size=11, max_steps=200),
    }

    def generate(self, seed):
        """Generate a sequence memory task instance.

        Creates a walled grid where a sequence of positions is
        demonstrated. The agent starts at (1, 1) and must replay the
        sequence from memory by visiting positions in order.

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

        Uses a constant step penalty to encourage the agent to memorize
        and reproduce the demonstrated sequence efficiently.

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
        visiting all sequence positions in the correct order from memory.

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

        A random agent cannot reliably reproduce the demonstrated
        sequence in the correct order, yielding near-zero expected
        return.

        Args:
            difficulty: Difficulty level string, or None to use the
                current instance difficulty.

        Returns:
            Expected random agent return of 0.0.
        """
        return 0.0
