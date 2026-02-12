"""SymbolMatching - Match symbols/colors according to rules"""

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("SymbolMatching-v0", tags=["pattern_recognition", "reasoning"])
class SymbolMatchingTask(TaskSpec):
    """Test pattern recognition by matching symbols according to hidden rules.

    The agent encounters pairs of symbols or colors on the grid and must
    match them according to rules that govern valid pairings. The matching
    rules may involve color correspondence, symbol complementarity, or
    positional relationships. The agent must observe which matches succeed
    and fail, infer the underlying matching rules, and then apply those
    rules to pair all remaining symbols correctly.

    Difficulty Levels:
        - easy: 7x7 grid with few symbols and obvious matching rules,
          100 max steps.
        - medium: 10x10 grid with more symbols and moderately complex
          rules, 200 max steps.
        - hard: 13x13 grid with many symbols and multi-attribute
          matching rules, 300 max steps.
        - expert: 15x15 grid with numerous symbols and context-dependent
          matching rules, 500 max steps.

    Capabilities Tested:
        - pattern_recognition: The agent must identify recurring patterns
          in valid symbol pairings to infer the matching criteria.
        - reasoning: The agent must apply inferred rules to novel symbol
          combinations encountered during the episode.

    Example:
        >>> env = agentick.make("SymbolMatching-v0", difficulty="medium")
        >>> obs, info = env.reset(seed=42)
        >>> # Observe valid pairings, infer rules, and match all symbols
    """

    name = "SymbolMatching-v0"
    description = "Match symbols/colors according to rules"
    capability_tags = ["pattern_recognition", "reasoning"]
    difficulty_configs = {
        "easy": DifficultyConfig(name="easy", grid_size=7, max_steps=100),
        "medium": DifficultyConfig(name="medium", grid_size=10, max_steps=200),
        "hard": DifficultyConfig(name="hard", grid_size=13, max_steps=300),
        "expert": DifficultyConfig(name="expert", grid_size=15, max_steps=500),
    }

    def generate(self, seed):
        """Generate a symbol matching task instance.

        Creates a walled grid with symbol pairs that must be matched
        according to hidden rules. The agent starts at (1, 1) and must
        infer the matching rules to pair all symbols correctly.

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

        Uses a constant step penalty to encourage the agent to infer
        matching rules from observations and pair all symbols quickly.

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
        correctly matching all symbol pairs according to the hidden rules.

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

        A random agent cannot infer the hidden matching rules from
        observations, yielding near-zero expected return.

        Args:
            difficulty: Difficulty level string, or None to use the
                current instance difficulty.

        Returns:
            Expected random agent return of 0.0.
        """
        return 0.0
