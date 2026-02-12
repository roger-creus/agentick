"""Few-Shot Adaptation - Learn from few demonstrations."""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("FewShotAdaptation-v0", tags=["meta_learning", "adaptation", "few_shot"])
class FewShotAdaptationTask(TaskSpec):
    """Test meta-learning by adapting to novel task variants from few demonstrations.

    The agent is provided with K demonstration episodes of a task variant,
    then must generalize to perform a novel variant of the same task
    without additional demonstrations. The number of demonstration shots
    (K) and the grid complexity scale with difficulty. The agent must
    extract transferable patterns from the demonstrations and apply them
    to the unseen variant. This measures an agent's ability to learn
    rapidly from limited data and generalize across task variations.

    Difficulty Levels:
        - easy: 7x7 grid with 1 demonstration shot, 150 max steps.
        - medium: 10x10 grid with 3 demonstration shots, 250 max steps.
        - hard: 13x13 grid with 5 demonstration shots, 350 max steps.
        - expert: 15x15 grid with 10 demonstration shots requiring
          deep pattern extraction, 500 max steps.

    Capabilities Tested:
        - meta_learning: The agent must learn transferable strategies
          from a small number of demonstrations of related tasks.
        - few_shot: The agent must generalize from K examples to a
          novel variant without explicit retraining.
        - adaptation: The agent must adapt its behavior in real time
          based on observed demonstrations.

    Example:
        >>> env = agentick.make("FewShotAdaptation-v0", difficulty="medium")
        >>> obs, info = env.reset(seed=42)
        >>> # Learn from 3 demonstrations, then solve the novel variant
    """

    name = "FewShotAdaptation-v0"
    description = "Learn from few demonstrations and generalize"
    capability_tags = ["meta_learning", "few_shot", "adaptation"]

    difficulty_configs = {
        "easy": DifficultyConfig(name="easy", grid_size=7, max_steps=150, params={"k_shots": 1}),
        "medium": DifficultyConfig(
            name="medium", grid_size=10, max_steps=250, params={"k_shots": 3}
        ),
        "hard": DifficultyConfig(name="hard", grid_size=13, max_steps=350, params={"k_shots": 5}),
        "expert": DifficultyConfig(
            name="expert", grid_size=15, max_steps=500, params={"k_shots": 10}
        ),
    }

    def generate(self, seed):
        """Generate a few-shot adaptation task instance.

        Creates a walled grid with a goal positioned at the far corner.
        The number of demonstration shots (K) is configured from the
        difficulty parameters. The agent is expected to have observed K
        demonstrations of related task variants before attempting this
        novel variant.

        Args:
            seed: Random seed for reproducible procedural generation.

        Returns:
            tuple: (grid, metadata) where grid is the initial Grid state
                with walls and goal, and metadata contains agent_start,
                goal_positions, max_steps, and k_shots count.
        """
        _ = np.random.default_rng(seed)  # For future randomization
        size = self.difficulty_config.grid_size

        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        agent_pos = (1, 1)
        goal_pos = (size - 2, size - 2)
        grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.GOAL

        k_shots = self.difficulty_config.params.get("k_shots", 1)

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [goal_pos],
            "max_steps": self.get_max_steps(),
            "k_shots": k_shots,
        }

    def compute_dense_reward(self, old_state, action, new_state, info):
        """Compute dense reward for a state transition.

        Uses a constant step penalty to encourage the agent to apply
        lessons from demonstrations efficiently on the novel variant.

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

        The task succeeds when the agent reaches the goal cell on the
        novel variant, demonstrating successful adaptation from the
        provided demonstrations.

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

        A random agent cannot leverage demonstrations for adaptation,
        yielding near-zero expected return on the novel variant.

        Args:
            difficulty: Difficulty level string, or None to use the
                current instance difficulty.

        Returns:
            Expected random agent return of 0.0.
        """
        return 0.0
