"""Noisy Observation - Standard tasks with observation noise."""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("NoisyObservation-v0", tags=["robustness", "navigation", "noise"])
class NoisyObservationTask(TaskSpec):
    """Test robustness to noisy and unreliable observation signals.

    The agent must navigate a standard grid-world to reach a goal, but
    its observations are corrupted by configurable noise. The noise level
    controls how much the agent's perceived state deviates from ground
    truth. At higher difficulties, observations become increasingly
    unreliable, requiring the agent to filter noise, maintain belief
    states, or adopt robust navigation strategies. This measures the
    agent's ability to act effectively under perceptual uncertainty.

    Difficulty Levels:
        - easy: 7x7 grid with 10% noise level, 150 max steps.
        - medium: 10x10 grid with 20% noise level, 250 max steps.
        - hard: 13x13 grid with 30% noise level, 350 max steps.
        - expert: 15x15 grid with 40% noise level making observations
          highly unreliable, 500 max steps.

    Capabilities Tested:
        - robustness: The agent must perform reliably despite corrupted
          observations that obscure the true environment state.
        - navigation: The agent must still find and reach the goal under
          noisy perception conditions.
        - noise_handling: The agent must filter or compensate for
          observation noise to maintain effective decision-making.

    Example:
        >>> env = agentick.make("NoisyObservation-v0", difficulty="medium")
        >>> obs, info = env.reset(seed=42)
        >>> # Navigate to goal despite noisy, unreliable observations
    """

    name = "NoisyObservation-v0"
    description = "Navigate to goal with noisy observations"
    capability_tags = ["robustness", "navigation", "noise_handling"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy", grid_size=7, max_steps=150, params={"noise_level": 0.1}
        ),
        "medium": DifficultyConfig(
            name="medium", grid_size=10, max_steps=250, params={"noise_level": 0.2}
        ),
        "hard": DifficultyConfig(
            name="hard", grid_size=13, max_steps=350, params={"noise_level": 0.3}
        ),
        "expert": DifficultyConfig(
            name="expert", grid_size=15, max_steps=500, params={"noise_level": 0.4}
        ),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_level = 0.1

    def generate(self, seed):
        """Generate a noisy observation task instance.

        Creates a walled grid with a goal at the far corner. The noise
        level is configured from the difficulty parameters and will be
        applied to the agent's observations during evaluation. The grid
        itself is a standard open layout to isolate the effect of
        observation noise on agent performance.

        Args:
            seed: Random seed for reproducible procedural generation.

        Returns:
            tuple: (grid, metadata) where grid is the initial Grid state
                with walls and goal, and metadata contains agent_start,
                goal_positions, max_steps, and noise_level.
        """
        _ = np.random.default_rng(seed)  # For future randomization
        size = self.difficulty_config.grid_size
        self.noise_level = self.difficulty_config.params.get("noise_level", 0.1)

        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        agent_pos = (1, 1)
        goal_pos = (size - 2, size - 2)
        grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.GOAL

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [goal_pos],
            "max_steps": self.get_max_steps(),
            "noise_level": self.noise_level,
        }

    def compute_dense_reward(self, old_state, action, new_state, info):
        """Compute dense reward for a state transition.

        Uses a constant step penalty to encourage efficient navigation
        despite noisy observations that may mislead the agent.

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

        The task succeeds when the agent reaches the goal cell despite
        having received noisy observations throughout the episode.

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

        A random agent cannot compensate for observation noise,
        yielding near-zero expected return.

        Args:
            difficulty: Difficulty level string, or None to use the
                current instance difficulty.

        Returns:
            Expected random agent return of 0.0.
        """
        return 0.0
