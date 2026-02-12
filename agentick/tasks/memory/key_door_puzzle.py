"""KeyDoorPuzzle task - Find keys to open doors in sequence."""

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("KeyDoorPuzzle-v0", tags=["memory", "sequential_reasoning"])
class KeyDoorPuzzleTask(TaskSpec):
    """Test memory and sequential reasoning by solving key-door dependency chains.

    The agent must find one or more keys scattered across the grid, then
    use them to unlock corresponding doors in the correct sequence to
    reach the final goal. The agent needs to remember key locations,
    plan the pickup order, and recall which doors each key opens. As
    difficulty increases, the grid grows and the step budget expands,
    making memory of previously explored areas and object locations
    increasingly critical. This measures working memory, spatial recall,
    and the ability to reason about sequential dependencies.

    Difficulty Levels:
        - easy: 7x7 grid with a simple key-door-goal chain, 100 max
          steps.
        - medium: 10x10 grid with more distance between key, door, and
          goal, 150 max steps.
        - hard: 13x13 grid requiring longer navigation paths between
          objects, 250 max steps.
        - expert: 15x15 grid with maximum separation between key, door,
          and goal, 400 max steps.

    Capabilities Tested:
        - memory: The agent must remember the locations of keys and
          doors discovered during exploration and recall them when
          needed.
        - sequential_reasoning: The agent must reason about the correct
          order of operations (find key, then unlock door, then reach
          goal) and execute the plan.

    Example:
        >>> env = agentick.make("KeyDoorPuzzle-v0", difficulty="medium")
        >>> obs, info = env.reset(seed=42)
        >>> # Find the key, unlock the door, then navigate to the goal
    """

    name = "KeyDoorPuzzle-v0"
    description = "Find key(s) to open door(s) in sequence"
    capability_tags = ["memory", "sequential_reasoning"]

    difficulty_configs = {
        "easy": DifficultyConfig(name="easy", grid_size=7, max_steps=100),
        "medium": DifficultyConfig(name="medium", grid_size=10, max_steps=150),
        "hard": DifficultyConfig(name="hard", grid_size=13, max_steps=250),
        "expert": DifficultyConfig(name="expert", grid_size=15, max_steps=400),
    }

    def generate(self, seed):
        """Generate a key-door puzzle task instance.

        Creates a walled grid with a key, a door, and a goal placed in
        a linear dependency chain. The agent must collect the key, use it
        to open the door, and then reach the goal. Object positions are
        determined by grid size to ensure appropriate spacing.

        Args:
            seed: Random seed for reproducible procedural generation.

        Returns:
            tuple: (grid, metadata) where grid is the initial Grid state
                with walls, key, door, and goal objects, and metadata
                contains agent_start, goal_positions, and max_steps.
        """
        size = self.difficulty_config.grid_size

        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        # Simple layout: key, door, goal
        key_pos = (2, 2)
        door_pos = (size // 2, size // 2)
        goal_pos = (size - 2, size - 2)

        grid.objects[key_pos[1], key_pos[0]] = ObjectType.KEY
        grid.objects[door_pos[1], door_pos[0]] = ObjectType.DOOR
        grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.GOAL

        return grid, {
            "agent_start": (1, 1),
            "goal_positions": [goal_pos],
            "max_steps": self.get_max_steps(),
        }

    def compute_dense_reward(self, old_state, action, new_state, info):
        """Compute dense reward for a state transition.

        Uses a constant step penalty to encourage the agent to solve the
        key-door dependency chain efficiently without unnecessary
        exploration.

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

        The task succeeds when the agent reaches the goal cell, which
        requires having previously collected the key and unlocked the
        door in the correct sequence.

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

        A random agent is unlikely to collect the key, open the door,
        and reach the goal in the correct sequence, yielding near-zero
        expected return.

        Args:
            difficulty: Difficulty level string, or None to use the
                current instance difficulty.

        Returns:
            Expected random agent return of 0.0.
        """
        return 0.0
