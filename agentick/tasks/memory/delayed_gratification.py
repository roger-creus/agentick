"""DelayedGratification - Choose delayed larger rewards."""

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("DelayedGratification-v0", tags=["credit_assignment", "long_horizon"])
class DelayedGratificationTask(TaskSpec):
    """Test credit assignment by choosing delayed larger rewards over immediate ones.

    The agent navigates a grid containing both nearby small rewards and
    distant large rewards. The immediate rewards are tempting and easy to
    reach, but the optimal strategy requires bypassing them in favor of a
    longer path to a significantly larger payoff. This evaluates the
    agent's ability to perform temporal credit assignment and resist
    greedy short-term policies in favor of long-horizon optimization.

    Difficulty Levels:
        - easy: 7x7 grid with clear reward differentiation, 100 max
          steps.
        - medium: 10x10 grid with more distractors along the path,
          150 max steps.
        - hard: 13x13 grid with many tempting intermediate rewards,
          250 max steps.
        - expert: 15x15 grid with elaborate decoy reward paths and a
          well-hidden optimal reward, 350 max steps.

    Capabilities Tested:
        - credit_assignment: The agent must correctly attribute value to
          actions whose rewards are delayed many steps into the future.
        - long_horizon: The agent must commit to a longer path despite
          the availability of immediately accessible smaller rewards.

    Example:
        >>> env = agentick.make("DelayedGratification-v0", difficulty="medium")
        >>> obs, info = env.reset(seed=42)
        >>> # Bypass nearby small rewards to reach the distant large reward
    """

    name = "DelayedGratification-v0"
    description = "Choose paths with delayed larger rewards vs immediate small"
    capability_tags = ["credit_assignment", "long_horizon"]
    difficulty_configs = {
        "easy": DifficultyConfig(name="easy", grid_size=7, max_steps=100),
        "medium": DifficultyConfig(name="medium", grid_size=10, max_steps=150),
        "hard": DifficultyConfig(name="hard", grid_size=13, max_steps=250),
        "expert": DifficultyConfig(name="expert", grid_size=15, max_steps=350),
    }

    def generate(self, seed):
        """Generate a delayed gratification task instance.

        Creates a walled grid with nearby small rewards and a distant
        large reward at the goal. The agent starts at (1, 1) and must
        bypass immediate temptations to reach the optimal reward.

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

        Uses a constant step penalty to encourage the agent to resist
        immediate small rewards and reach the distant optimal reward
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

        The task succeeds when the agent reaches the goal cell containing
        the large delayed reward, bypassing smaller immediate rewards.

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
            Optimal return of 1.0 (sparse success reward for reaching
            the delayed large reward).
        """
        return 1.0

    def get_random_baseline(self, difficulty=None):
        """Get expected return for a random agent baseline.

        A random agent is likely to collect nearby small rewards rather
        than navigate to the distant optimal reward, yielding near-zero
        expected return.

        Args:
            difficulty: Difficulty level string, or None to use the
                current instance difficulty.

        Returns:
            Expected random agent return of 0.0.
        """
        return 0.0
