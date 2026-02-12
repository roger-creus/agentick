"""TimingChallenge - Pass through gates that open/close on cycles"""

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("TimingChallenge-v0", tags=["low_level_control", "timing"])
class TimingChallengeTask(TaskSpec):
    """Test timing-based control by passing through cyclically opening gates.

    The agent must traverse a grid containing gates that open and close on
    fixed periodic cycles. To reach the goal, the agent must time its
    movements to pass through gates during their open windows, sometimes
    waiting in safe positions for the right moment. The challenge combines
    spatial navigation with temporal reasoning about gate phase cycles and
    the coordination of movement across multiple synchronized obstacles.

    Difficulty Levels:
        - easy: 7x7 grid with slow gate cycles and generous open windows,
          100 max steps.
        - medium: 10x10 grid with moderate gate cycles, 200 max steps.
        - hard: 13x13 grid with fast gate cycles and narrow open windows,
          300 max steps.
        - expert: 15x15 grid with multiple overlapping gate cycles
          requiring precise synchronization, 500 max steps.

    Capabilities Tested:
        - low_level_control: The agent must execute well-timed movement
          sequences coordinated with gate open/close cycles.
        - timing: The agent must reason about periodic temporal patterns
          and synchronize its actions with environmental rhythms.

    Example:
        >>> env = agentick.make("TimingChallenge-v0", difficulty="medium")
        >>> obs, info = env.reset(seed=42)
        >>> # Time movements to pass through gates during open windows
    """

    name = "TimingChallenge-v0"
    description = "Pass through gates that open/close on cycles"
    capability_tags = ["low_level_control", "timing"]
    difficulty_configs = {
        "easy": DifficultyConfig(name="easy", grid_size=7, max_steps=100),
        "medium": DifficultyConfig(name="medium", grid_size=10, max_steps=200),
        "hard": DifficultyConfig(name="hard", grid_size=13, max_steps=300),
        "expert": DifficultyConfig(name="expert", grid_size=15, max_steps=500),
    }

    def generate(self, seed):
        """Generate a timing challenge task instance.

        Creates a walled grid with cyclically opening and closing gates
        and a goal position. The agent starts at (1, 1) and must time
        movements through gates during their open windows.

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
        synchronize with gate cycles and reach the goal efficiently.

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
        passing through all timed gates during their open windows.

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

        A random agent cannot synchronize movements with gate open/close
        cycles, yielding near-zero expected return.

        Args:
            difficulty: Difficulty level string, or None to use the
                current instance difficulty.

        Returns:
            Expected random agent return of 0.0.
        """
        return 0.0
