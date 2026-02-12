"""CompetitiveTag - Tag/evade game against scripted opponent"""

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("CompetitiveTag-v0", tags=["multi_agent", "competition"])
class CompetitiveTagTask(TaskSpec):
    """Test multi-agent competition through a tag and evade game.

    The agent plays a tag game against a scripted opponent on a walled
    grid arena. When the agent is "it," it must chase and tag the
    opponent; when the opponent is "it," the agent must evade. Roles
    may switch upon tagging. The scripted opponent uses increasingly
    sophisticated strategies at higher difficulties. Success requires
    reading the opponent's behavior, predicting its moves, and
    adapting strategy based on the current role.

    Difficulty Levels:
        - easy: 7x7 grid with a slow, predictable opponent, 100 max
          steps.
        - medium: 10x10 grid with a moderately skilled opponent,
          200 max steps.
        - hard: 13x13 grid with a fast, strategic opponent, 300 max
          steps.
        - expert: 15x15 grid with a highly adaptive opponent using
          advanced tactics, 500 max steps.

    Capabilities Tested:
        - multi_agent: The agent must reason about another agent's
          behavior and adapt its own strategy accordingly.
        - competition: The agent must outperform the scripted opponent
          in a zero-sum pursuit-evasion game.

    Example:
        >>> env = agentick.make("CompetitiveTag-v0", difficulty="medium")
        >>> obs, info = env.reset(seed=42)
        >>> # Chase the opponent when "it" or evade when being chased
    """

    name = "CompetitiveTag-v0"
    description = "Tag/evade game against scripted opponent"
    capability_tags = ["multi_agent", "competition"]
    difficulty_configs = {
        "easy": DifficultyConfig(name="easy", grid_size=7, max_steps=100),
        "medium": DifficultyConfig(name="medium", grid_size=10, max_steps=200),
        "hard": DifficultyConfig(name="hard", grid_size=13, max_steps=300),
        "expert": DifficultyConfig(name="expert", grid_size=15, max_steps=500),
    }

    def generate(self, seed):
        """Generate a competitive tag task instance.

        Creates a walled grid arena for a tag game against a scripted
        opponent. The agent starts at (1, 1) and must tag the opponent
        or evade being tagged to reach the goal.

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
        outmaneuver the scripted opponent and reach the goal quickly.

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

        The task succeeds when the agent reaches the goal cell,
        indicating it has won the tag game against the scripted opponent.

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

        A random agent cannot effectively compete against the scripted
        opponent in a pursuit-evasion game, yielding near-zero expected
        return.

        Args:
            difficulty: Difficulty level string, or None to use the
                current instance difficulty.

        Returns:
            Expected random agent return of 0.0.
        """
        return 0.0
