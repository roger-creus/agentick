"""Environment Shift - Dynamics change mid-episode."""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("EnvironmentShift-v0", tags=["world_model", "adaptation", "change_detection"])
class EnvironmentShiftTask(TaskSpec):
    """Test world-model adaptation by detecting and responding to mid-episode dynamics changes.

    The agent navigates a grid where the environment dynamics change
    abruptly at a configurable step during the episode. Before the
    change, standard movement rules apply; after the change, the reward
    structure, movement costs, or other mechanics shift. The agent must
    detect that the dynamics have changed, update its internal world
    model, and adapt its strategy accordingly. The timing of the change
    and the grid complexity scale with difficulty. This measures an
    agent's ability to maintain and update an internal world model in
    response to non-stationary environments.

    Difficulty Levels:
        - easy: 7x7 grid with dynamics change at step 50, 200 max
          steps.
        - medium: 10x10 grid with dynamics change at step 100, 300 max
          steps.
        - hard: 13x13 grid with dynamics change at step 150, 400 max
          steps.
        - expert: 15x15 grid with dynamics change at step 200 requiring
          rapid re-adaptation, 600 max steps.

    Capabilities Tested:
        - world_model: The agent must maintain an internal model of
          environment dynamics and detect when it becomes invalid.
        - adaptation: The agent must quickly adapt its strategy after
          detecting the mid-episode dynamics shift.
        - change_detection: The agent must recognize that the rules have
          changed based on observed discrepancies from predictions.

    Example:
        >>> env = agentick.make("EnvironmentShift-v0", difficulty="medium")
        >>> obs, info = env.reset(seed=42)
        >>> # Navigate to goal, adapting when dynamics change at step 100
    """

    name = "EnvironmentShift-v0"
    description = "Detect and adapt to mid-episode dynamics changes"
    capability_tags = ["world_model", "adaptation", "change_detection"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy", grid_size=7, max_steps=200, params={"change_step": 50}
        ),
        "medium": DifficultyConfig(
            name="medium", grid_size=10, max_steps=300, params={"change_step": 100}
        ),
        "hard": DifficultyConfig(
            name="hard", grid_size=13, max_steps=400, params={"change_step": 150}
        ),
        "expert": DifficultyConfig(
            name="expert", grid_size=15, max_steps=600, params={"change_step": 200}
        ),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dynamics_changed = False
        self.change_step = 0

    def generate(self, seed):
        """Generate an environment shift task instance.

        Creates a walled grid with randomly placed interior walls and a
        goal at the far corner. The dynamics change step is configured
        from difficulty parameters. The generator retries up to 10 times
        to ensure solvability via flood fill, falling back to a simple
        open layout if needed. The dynamics_changed flag is reset for
        each new instance.

        Args:
            seed: Random seed for reproducible procedural generation.

        Returns:
            tuple: (grid, metadata) where grid is the initial Grid state
                with walls and goal, and metadata contains agent_start,
                goal_positions, max_steps, and change_step.
        """
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        self.change_step = self.difficulty_config.params.get("change_step", 50)

        max_attempts = 10
        for attempt in range(max_attempts):
            grid = Grid(size, size)
            grid.terrain[0, :] = CellType.WALL
            grid.terrain[-1, :] = CellType.WALL
            grid.terrain[:, 0] = CellType.WALL
            grid.terrain[:, -1] = CellType.WALL

            # Add some walls
            for _ in range(size // 2):
                x, y = rng.integers(1, size - 1, 2)
                grid.terrain[y, x] = CellType.WALL

            agent_pos = (1, 1)
            goal_pos = (size - 2, size - 2)

            # Verify solvable
            reachable = grid.flood_fill(agent_pos)
            if goal_pos in reachable:
                grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.GOAL
                self.dynamics_changed = False

                return grid, {
                    "agent_start": agent_pos,
                    "goal_positions": [goal_pos],
                    "max_steps": self.get_max_steps(),
                    "change_step": self.change_step,
                }

        # Fallback: simple solvable layout
        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL
        agent_pos = (1, 1)
        goal_pos = (size - 2, size - 2)
        grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.GOAL
        self.dynamics_changed = False

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [goal_pos],
            "max_steps": self.get_max_steps(),
            "change_step": self.change_step,
        }

    def change_dynamics(self, rng):
        """Change environment dynamics (called by world model evaluator)."""
        self.dynamics_changed = True
        # Example: could modify rewards, movement costs, etc.

    def compute_dense_reward(self, old_state, action, new_state, info):
        """Compute dense reward for a state transition.

        Uses a step penalty that changes after the dynamics shift. Before
        the shift, the penalty is -0.01 per step. After the dynamics
        change, the penalty increases to -0.02 per step, reflecting the
        altered environment mechanics that the agent must adapt to.

        Args:
            old_state: State dict before the action.
            action: Action taken by the agent.
            new_state: State dict after the action.
            info: Additional info dict from the environment step.

        Returns:
            Step penalty of -0.01 before dynamics change, or -0.02 after.
        """
        base_reward = -0.01

        # After dynamics change, reward structure changes
        if self.dynamics_changed:
            base_reward = -0.02  # Higher time penalty after change

        return base_reward

    def check_success(self, state):
        """Check if the task objective is complete.

        The task succeeds when the agent reaches the goal cell, which
        may require adapting to changed dynamics mid-episode.

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

        A random agent cannot detect or adapt to mid-episode dynamics
        changes, yielding near-zero expected return.

        Args:
            difficulty: Difficulty level string, or None to use the
                current instance difficulty.

        Returns:
            Expected random agent return of 0.0.
        """
        return 0.0
