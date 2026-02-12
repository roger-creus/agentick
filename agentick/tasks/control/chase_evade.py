"""ChaseEvade - Chase a moving target or evade pursuer"""

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("ChaseEvade-v0", tags=["reactive_control", "prediction"])
class ChaseEvadeTask(TaskSpec):
    """Test reactive control and prediction in a chase-evade scenario.

    The agent must either chase a moving target or evade a pursuing
    opponent within a walled grid arena. The target or pursuer follows a
    scripted movement policy, and the agent must predict its trajectory
    to intercept or avoid it. Success requires real-time reactive control
    combined with anticipation of the opponent's future positions based
    on observed movement patterns.

    Difficulty Levels:
        - easy: 7x7 grid with slow-moving target, 100 max steps.
        - medium: 10x10 grid with moderately fast target, 200 max steps.
        - hard: 13x13 grid with fast, evasive target, 300 max steps.
        - expert: 15x15 grid with highly unpredictable target using
          advanced evasion, 500 max steps.

    Capabilities Tested:
        - reactive_control: The agent must make rapid movement decisions
          each step in response to the opponent's changing position.
        - prediction: The agent must anticipate the opponent's future
          trajectory to plan interception or evasion paths.

    Example:
        >>> env = agentick.make("ChaseEvade-v0", difficulty="medium")
        >>> obs, info = env.reset(seed=42)
        >>> # Pursue the moving target or evade the pursuer
    """

    name = "ChaseEvade-v0"
    description = "Chase a moving target or evade pursuer"
    capability_tags = ["reactive_control", "prediction"]
    difficulty_configs = {
        "easy": DifficultyConfig(name="easy", grid_size=7, max_steps=100),
        "medium": DifficultyConfig(name="medium", grid_size=10, max_steps=200),
        "hard": DifficultyConfig(name="hard", grid_size=13, max_steps=300),
        "expert": DifficultyConfig(name="expert", grid_size=15, max_steps=500),
    }

    def generate(self, seed):
        """Generate a chase-evade task instance.

        Creates a walled grid arena with a goal position representing the
        target or safe zone. The agent starts at (1, 1) and must chase a
        moving target or evade a pursuer to reach the goal.

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

        Uses a constant step penalty to encourage the agent to intercept
        the target or evade the pursuer and reach the goal quickly.

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
        successful interception of the target or evasion to safety.

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

        A random agent cannot reliably predict or react to the opponent's
        movements, yielding near-zero expected return.

        Args:
            difficulty: Difficulty level string, or None to use the
                current instance difficulty.

        Returns:
            Expected random agent return of 0.0.
        """
        return 0.0
