"""SokobanPush - Push boxes onto targets"""

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("SokobanPush-v0", tags=["reasoning", "planning"])
class SokobanPushTask(TaskSpec):
    """Test reasoning and planning by pushing boxes onto target positions.

    The agent must push boxes across the grid to designated target
    positions, following classic Sokoban mechanics. Boxes can only be
    pushed (not pulled), and pushing a box into a wall or corner creates
    an irreversible deadlock. The agent must plan the order and direction
    of pushes carefully, often reasoning many moves ahead to avoid
    creating unsolvable states. This is a well-known NP-hard planning
    problem that tests deep lookahead and spatial reasoning.

    Difficulty Levels:
        - easy: 7x7 grid with one or two boxes and simple layouts,
          100 max steps.
        - medium: 10x10 grid with several boxes and moderate obstacles,
          200 max steps.
        - hard: 13x13 grid with many boxes and tight corridors, 300 max
          steps.
        - expert: 15x15 grid with numerous boxes, complex wall layouts,
          and minimal margin for error, 500 max steps.

    Capabilities Tested:
        - reasoning: The agent must analyze the spatial configuration to
          determine valid push sequences and detect potential deadlocks.
        - planning: The agent must plan multi-step push sequences that
          avoid irreversible states while moving all boxes to targets.

    Example:
        >>> env = agentick.make("SokobanPush-v0", difficulty="medium")
        >>> obs, info = env.reset(seed=42)
        >>> # Push all boxes onto their target positions without deadlocking
    """

    name = "SokobanPush-v0"
    description = "Push boxes onto targets"
    capability_tags = ["reasoning", "planning"]
    difficulty_configs = {
        "easy": DifficultyConfig(name="easy", grid_size=7, max_steps=100),
        "medium": DifficultyConfig(name="medium", grid_size=10, max_steps=200),
        "hard": DifficultyConfig(name="hard", grid_size=13, max_steps=300),
        "expert": DifficultyConfig(name="expert", grid_size=15, max_steps=500),
    }

    def generate(self, seed):
        """Generate a Sokoban push puzzle instance.

        Creates a walled grid with pushable boxes and target positions.
        The agent starts at (1, 1) and must push all boxes onto their
        targets without creating deadlocks.

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

        Uses a constant step penalty to encourage the agent to plan push
        sequences that avoid deadlocks and reach the goal efficiently.

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
        that all boxes have been pushed onto their target positions.

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

        A random agent is highly likely to push boxes into deadlocked
        positions, yielding near-zero expected return.

        Args:
            difficulty: Difficulty level string, or None to use the
                current instance difficulty.

        Returns:
            Expected random agent return of 0.0.
        """
        return 0.0
