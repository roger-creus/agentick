"""CooperativeTransport - Two agents carry object together"""

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("CooperativeTransport-v0", tags=["multi_agent", "cooperation"])
class CooperativeTransportTask(TaskSpec):
    """Test multi-agent cooperation by jointly transporting an object.

    Two agents must work together to carry a large object across the grid
    to a target location. The object is too heavy for either agent to move
    alone, so both must be adjacent to it and move in coordinated
    directions simultaneously. The agents must synchronize their
    movements, navigate around obstacles together, and maintain their
    positions relative to the shared payload throughout transport.

    Difficulty Levels:
        - easy: 7x7 grid with a clear path and no obstacles, 100 max
          steps.
        - medium: 10x10 grid with some obstacles requiring coordinated
          detours, 200 max steps.
        - hard: 13x13 grid with narrow passages requiring precise
          coordination, 300 max steps.
        - expert: 15x15 grid with complex obstacle layouts demanding
          tight synchronization, 500 max steps.

    Capabilities Tested:
        - multi_agent: The agent must coordinate actions with a partner
          agent to achieve a shared objective.
        - cooperation: Both agents must synchronize their movement
          timing and direction to transport the object successfully.

    Example:
        >>> env = agentick.make("CooperativeTransport-v0", difficulty="medium")
        >>> obs, info = env.reset(seed=42)
        >>> # Coordinate with partner to carry the object to the target
    """

    name = "CooperativeTransport-v0"
    description = "Two agents carry object together"
    capability_tags = ["multi_agent", "cooperation"]
    difficulty_configs = {
        "easy": DifficultyConfig(name="easy", grid_size=7, max_steps=100),
        "medium": DifficultyConfig(name="medium", grid_size=10, max_steps=200),
        "hard": DifficultyConfig(name="hard", grid_size=13, max_steps=300),
        "expert": DifficultyConfig(name="expert", grid_size=15, max_steps=500),
    }

    def generate(self, seed):
        """Generate a cooperative transport task instance.

        Creates a walled grid with a heavy object and a target location.
        Two agents must coordinate to carry the object together. The
        primary agent starts at (1, 1).

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

        Uses a constant step penalty to encourage both agents to
        coordinate their movements and transport the object to the
        target efficiently.

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
        indicating the object has been transported to the target
        location through coordinated movement.

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

        A random agent cannot synchronize with the partner agent to
        transport the shared object, yielding near-zero expected return.

        Args:
            difficulty: Difficulty level string, or None to use the
                current instance difficulty.

        Returns:
            Expected random agent return of 0.0.
        """
        return 0.0
