"""SwitchCircuit - Activate switches in correct combination"""

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("SwitchCircuit-v0", tags=["compositional_logic"])
class SwitchCircuitTask(TaskSpec):
    """Test compositional logic by activating switches in correct combinations.

    The agent must navigate a grid containing interconnected switches that
    control gates, doors, or other mechanisms. Each switch affects one or
    more circuit elements, and only the correct combination of switch
    states will open the path to the goal. The agent must reason about
    how individual switches compose into circuit-level behavior, testing
    combinations systematically to find the solution. Switch interactions
    may include AND, OR, and XOR logic.

    Difficulty Levels:
        - easy: 7x7 grid with few switches and simple logic, 100 max
          steps.
        - medium: 10x10 grid with more switches and two-level logic,
          200 max steps.
        - hard: 13x13 grid with many switches and nested logic gates,
          300 max steps.
        - expert: 15x15 grid with a complex circuit of interdependent
          switches and multi-level logic, 500 max steps.

    Capabilities Tested:
        - compositional_logic: The agent must understand how individual
          switch states compose through logical operations to determine
          the overall circuit output and find the correct activation
          pattern.

    Example:
        >>> env = agentick.make("SwitchCircuit-v0", difficulty="medium")
        >>> obs, info = env.reset(seed=42)
        >>> # Activate the correct combination of switches to open the path
    """

    name = "SwitchCircuit-v0"
    description = "Activate switches in correct combination"
    capability_tags = ["compositional_logic"]
    difficulty_configs = {
        "easy": DifficultyConfig(name="easy", grid_size=7, max_steps=100),
        "medium": DifficultyConfig(name="medium", grid_size=10, max_steps=200),
        "hard": DifficultyConfig(name="hard", grid_size=13, max_steps=300),
        "expert": DifficultyConfig(name="expert", grid_size=15, max_steps=500),
    }

    def generate(self, seed):
        """Generate a switch circuit task instance.

        Creates a walled grid with interconnected switches that control
        gates and doors through logical combinations. The agent starts
        at (1, 1) and must find the correct switch combination to open
        the path to the goal.

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

        Uses a constant step penalty to encourage the agent to reason
        about switch logic and find the correct activation pattern
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
        setting all switches to the correct combination that opens the
        path.

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

        A random agent cannot systematically determine the correct
        switch combination, yielding near-zero expected return.

        Args:
            difficulty: Difficulty level string, or None to use the
                current instance difficulty.

        Returns:
            Expected random agent return of 0.0.
        """
        return 0.0
