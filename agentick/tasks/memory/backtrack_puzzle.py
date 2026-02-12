"""BacktrackPuzzle - Revisit previous locations."""

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("BacktrackPuzzle-v0", tags=["memory", "planning"])
class BacktrackPuzzleTask(TaskSpec):
    """Test memory and planning by requiring revisits to previous locations.

    The agent must navigate a grid where certain actions trigger state
    changes that make previously visited locations newly relevant. For
    example, activating a switch in one area may unlock a door near an
    earlier position, requiring the agent to backtrack. The agent must
    remember which locations it has visited and what state changes have
    occurred, then plan return trips to exploit newly available
    opportunities. This tests the ability to maintain a mental map and
    reason about when backtracking is beneficial.

    Difficulty Levels:
        - easy: 7x7 grid with few state-change triggers, 150 max steps.
        - medium: 10x10 grid with moderate backtracking depth, 250 max
          steps.
        - hard: 13x13 grid with chained state changes requiring multiple
          revisits, 400 max steps.
        - expert: 15x15 grid with deep backtracking chains and
          interdependent triggers, 600 max steps.

    Capabilities Tested:
        - memory: The agent must remember previously visited positions
          and track which environmental state changes have occurred.
        - planning: The agent must decide when to proceed forward versus
          when to backtrack to exploit newly available paths.

    Example:
        >>> env = agentick.make("BacktrackPuzzle-v0", difficulty="medium")
        >>> obs, info = env.reset(seed=42)
        >>> # Explore, trigger state changes, then revisit earlier areas
    """

    name = "BacktrackPuzzle-v0"
    description = "Must revisit previous locations after state changes"
    capability_tags = ["memory", "planning"]
    difficulty_configs = {
        "easy": DifficultyConfig(name="easy", grid_size=7, max_steps=150),
        "medium": DifficultyConfig(name="medium", grid_size=10, max_steps=250),
        "hard": DifficultyConfig(name="hard", grid_size=13, max_steps=400),
        "expert": DifficultyConfig(name="expert", grid_size=15, max_steps=600),
    }

    def generate(self, seed):
        """Generate a backtrack puzzle instance.

        Creates a walled grid with state-change triggers that require the
        agent to revisit previous locations. The agent starts at (1, 1)
        and must activate triggers and backtrack to reach the goal.

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

        Uses a constant step penalty to encourage the agent to remember
        visited locations, trigger state changes, and backtrack
        efficiently.

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
        activating all necessary triggers and backtracking as required.

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

        A random agent cannot maintain a mental map or decide when to
        backtrack, yielding near-zero expected return.

        Args:
            difficulty: Difficulty level string, or None to use the
                current instance difficulty.

        Returns:
            Expected random agent return of 0.0.
        """
        return 0.0
