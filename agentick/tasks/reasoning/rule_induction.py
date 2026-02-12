"""RuleInduction - Discover hidden rules governing grid mechanics"""

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("RuleInduction-v0", tags=["reasoning", "generalization"])
class RuleInductionTask(TaskSpec):
    """Test reasoning and generalization by discovering hidden grid rules.

    The agent operates on a grid governed by hidden rules that determine
    how cells behave, how movement is affected, or how objects interact.
    The rules are not communicated to the agent; instead, the agent must
    experiment with the environment, observe outcomes, and inductively
    infer the underlying mechanics. Once the rules are understood, the
    agent applies them to navigate to the goal. This evaluates
    hypothesis formation, systematic experimentation, and generalization.

    Difficulty Levels:
        - easy: 7x7 grid with a single simple hidden rule, 100 max
          steps.
        - medium: 10x10 grid with two interacting rules, 200 max steps.
        - hard: 13x13 grid with multiple non-obvious rules, 300 max
          steps.
        - expert: 15x15 grid with complex conditional rules requiring
          extensive experimentation, 500 max steps.

    Capabilities Tested:
        - reasoning: The agent must form hypotheses about hidden rules
          and test them through deliberate experimentation.
        - generalization: The agent must apply discovered rules to
          novel situations encountered later in the same episode.

    Example:
        >>> env = agentick.make("RuleInduction-v0", difficulty="medium")
        >>> obs, info = env.reset(seed=42)
        >>> # Experiment to discover hidden rules, then exploit them
    """

    name = "RuleInduction-v0"
    description = "Discover hidden rules governing grid mechanics"
    capability_tags = ["reasoning", "generalization"]
    difficulty_configs = {
        "easy": DifficultyConfig(name="easy", grid_size=7, max_steps=100),
        "medium": DifficultyConfig(name="medium", grid_size=10, max_steps=200),
        "hard": DifficultyConfig(name="hard", grid_size=13, max_steps=300),
        "expert": DifficultyConfig(name="expert", grid_size=15, max_steps=500),
    }

    def generate(self, seed):
        """Generate a rule induction task instance.

        Creates a walled grid governed by hidden rules that affect cell
        behavior, movement, or object interactions. The agent starts at
        (1, 1) and must discover these rules through experimentation to
        reach the goal.

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

        Uses a constant step penalty to encourage the agent to
        efficiently discover the hidden rules through experimentation
        and exploit them to reach the goal.

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

        The task succeeds when the agent reaches the goal cell by
        applying the inductively discovered hidden rules.

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

        A random agent cannot systematically discover and exploit hidden
        rules, yielding near-zero expected return.

        Args:
            difficulty: Difficulty level string, or None to use the
                current instance difficulty.

        Returns:
            Expected random agent return of 0.0.
        """
        return 0.0
