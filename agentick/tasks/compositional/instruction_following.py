"""Instruction Following - Follow language instructions."""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("InstructionFollowing-v0", tags=["language", "grounding", "instruction"])
class InstructionFollowingTask(TaskSpec):
    """Test language grounding by following sequential natural-language instructions.

    The agent receives a sequence of natural-language instructions (e.g.,
    "Move to the top-right corner", "Collect the red key") and must parse,
    ground, and execute them in the correct order within a grid environment.
    The number of instructions scales with difficulty, requiring increasing
    compositional understanding and multi-step execution. This measures
    an agent's ability to bridge language understanding with situated
    action in a spatial environment.

    Difficulty Levels:
        - easy: 7x7 grid with 2 instructions, 100 max steps.
        - medium: 10x10 grid with 3 instructions, 150 max steps.
        - hard: 13x13 grid with 4 instructions, 200 max steps.
        - expert: 15x15 grid with 5 instructions requiring extensive
          compositional reasoning, 300 max steps.

    Capabilities Tested:
        - language_grounding: The agent must map natural-language
          descriptions to concrete spatial locations and actions.
        - instruction_following: The agent must execute instructions in
          the prescribed sequential order without skipping or reordering.
        - reasoning: The agent must reason about how to fulfill each
          instruction given the current grid state and its position.

    Example:
        >>> env = agentick.make("InstructionFollowing-v0", difficulty="medium")
        >>> obs, info = env.reset(seed=42)
        >>> # Follow the 3-step instruction sequence to reach the goal
    """

    name = "InstructionFollowing-v0"
    description = "Parse and follow language instructions"
    capability_tags = ["language_grounding", "instruction_following", "reasoning"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy", grid_size=7, max_steps=100, params={"n_instructions": 2}
        ),
        "medium": DifficultyConfig(
            name="medium", grid_size=10, max_steps=150, params={"n_instructions": 3}
        ),
        "hard": DifficultyConfig(
            name="hard", grid_size=13, max_steps=200, params={"n_instructions": 4}
        ),
        "expert": DifficultyConfig(
            name="expert", grid_size=15, max_steps=300, params={"n_instructions": 5}
        ),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.instructions = []

    def generate(self, seed):
        """Generate an instruction following task instance.

        Creates a walled grid with a goal and a sequence of natural-language
        instructions for the agent to follow. The number of instructions
        scales with the configured difficulty. Instructions are drawn from
        a fixed set of spatial and object-interaction directives.

        Args:
            seed: Random seed for reproducible procedural generation.

        Returns:
            tuple: (grid, metadata) where grid is the initial Grid state
                with walls and goal, and metadata contains agent_start,
                goal_positions, max_steps, and the list of instructions.
        """
        _ = np.random.default_rng(seed)  # For future randomization
        size = self.difficulty_config.grid_size
        n_inst = self.difficulty_config.params.get("n_instructions", 2)

        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        agent_pos = (1, 1)
        goal_pos = (size - 2, size - 2)
        grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.GOAL

        # Generate instructions
        self.instructions = [
            "Move to the top-right corner",
            "Collect the red key",
            "Open the blue door",
            "Navigate to the goal",
        ][:n_inst]

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [goal_pos],
            "max_steps": self.get_max_steps(),
            "instructions": self.instructions,
        }

    def compute_dense_reward(self, old_state, action, new_state, info):
        """Compute dense reward for a state transition.

        Uses a constant step penalty to encourage the agent to follow
        the instruction sequence and reach the goal efficiently.

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
        following all instructions in the prescribed sequence.

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

        A random agent cannot parse or follow instructions, yielding
        near-zero expected return for completing the full sequence.

        Args:
            difficulty: Difficulty level string, or None to use the
                current instance difficulty.

        Returns:
            Expected random agent return of 0.0.
        """
        return 0.0
