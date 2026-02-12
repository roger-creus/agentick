"""MultiRoomEscape - Solve puzzles across multiple rooms"""

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("MultiRoomEscape-v0", tags=["skill_composition", "long_horizon"])
class MultiRoomEscapeTask(TaskSpec):
    """Test skill composition across a sequence of connected room puzzles.

    The agent must escape through a series of interconnected rooms, each
    containing a distinct puzzle that must be solved to unlock the door
    to the next room. Puzzles may involve key collection, switch
    activation, pattern matching, or other mechanics. The agent must
    compose different skills across rooms and may need to carry items
    or information from earlier rooms to solve later ones. This tests
    long-horizon planning and the ability to chain diverse subskills.

    Difficulty Levels:
        - easy: 7x7 grid with few rooms and simple puzzles, 100 max
          steps.
        - medium: 10x10 grid with more rooms and moderate puzzle
          complexity, 200 max steps.
        - hard: 13x13 grid with many rooms and puzzles requiring
          inter-room item transport, 300 max steps.
        - expert: 15x15 grid with numerous rooms, complex puzzles, and
          dependencies between distant rooms, 500 max steps.

    Capabilities Tested:
        - skill_composition: The agent must combine different problem-
          solving skills across sequential puzzle rooms.
        - long_horizon: The agent must maintain goal pursuit across
          many rooms and dozens of intermediate subgoals.

    Example:
        >>> env = agentick.make("MultiRoomEscape-v0", difficulty="medium")
        >>> obs, info = env.reset(seed=42)
        >>> # Solve each room's puzzle to progress through all rooms
    """

    name = "MultiRoomEscape-v0"
    description = "Solve puzzles across multiple rooms"
    capability_tags = ["skill_composition", "long_horizon"]
    difficulty_configs = {
        "easy": DifficultyConfig(name="easy", grid_size=7, max_steps=100),
        "medium": DifficultyConfig(name="medium", grid_size=10, max_steps=200),
        "hard": DifficultyConfig(name="hard", grid_size=13, max_steps=300),
        "expert": DifficultyConfig(name="expert", grid_size=15, max_steps=500),
    }

    def generate(self, seed):
        """Generate a multi-room escape task instance.

        Creates a walled grid divided into interconnected rooms, each
        with a distinct puzzle. The agent starts at (1, 1) and must solve
        each room's puzzle sequentially to escape to the goal.

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

        Uses a constant step penalty to encourage the agent to solve
        room puzzles and progress through all rooms efficiently.

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
        solving all room puzzles and escaping through the final room.

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

        A random agent cannot solve the sequence of room puzzles
        requiring diverse skills, yielding near-zero expected return.

        Args:
            difficulty: Difficulty level string, or None to use the
                current instance difficulty.

        Returns:
            Expected random agent return of 0.0.
        """
        return 0.0
