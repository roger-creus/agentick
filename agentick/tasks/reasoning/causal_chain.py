"""CausalChain - Trigger chain of events in correct order"""

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("CausalChain-v0", tags=["causal_reasoning"])
class CausalChainTask(TaskSpec):
    """Test causal reasoning by triggering a chain of events in correct order.

    The agent must activate a series of triggers on the grid in a specific
    causal sequence. Each trigger causes an environmental change that
    enables the next trigger in the chain. Activating triggers out of
    order has no effect or may block progress. The agent must observe
    cause-and-effect relationships, infer the correct activation order,
    and execute the full chain to reach the goal state.

    Difficulty Levels:
        - easy: 7x7 grid with a short, linear causal chain, 100 max
          steps.
        - medium: 10x10 grid with a longer chain and some spatial
          separation between triggers, 200 max steps.
        - hard: 13x13 grid with branching causal dependencies, 300 max
          steps.
        - expert: 15x15 grid with complex, non-obvious causal chains
          and red-herring triggers, 500 max steps.

    Capabilities Tested:
        - causal_reasoning: The agent must identify cause-and-effect
          relationships between triggers and determine the correct
          activation sequence through observation and inference.

    Example:
        >>> env = agentick.make("CausalChain-v0", difficulty="medium")
        >>> obs, info = env.reset(seed=42)
        >>> # Discover and activate triggers in the correct causal order
    """

    name = "CausalChain-v0"
    description = "Trigger chain of events in correct order"
    capability_tags = ["causal_reasoning"]
    difficulty_configs = {
        "easy": DifficultyConfig(name="easy", grid_size=7, max_steps=100),
        "medium": DifficultyConfig(name="medium", grid_size=10, max_steps=200),
        "hard": DifficultyConfig(name="hard", grid_size=13, max_steps=300),
        "expert": DifficultyConfig(name="expert", grid_size=15, max_steps=500),
    }

    def generate(self, seed):
        """Generate a causal chain task instance.

        Creates a walled grid with a series of causal triggers that must
        be activated in the correct sequence. The agent starts at (1, 1)
        and must discover and execute the causal chain to reach the goal.

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

        Uses a constant step penalty to encourage the agent to identify
        the correct causal activation order and reach the goal quickly.

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
        activating all triggers in the correct causal sequence.

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

        A random agent cannot determine the correct causal activation
        order, yielding near-zero expected return.

        Args:
            difficulty: Difficulty level string, or None to use the
                current instance difficulty.

        Returns:
            Expected random agent return of 0.0.
        """
        return 0.0
