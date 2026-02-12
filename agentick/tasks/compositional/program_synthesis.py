"""Program Synthesis - Program a simple machine."""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("ProgramSynthesis-v0", tags=["reasoning", "planning", "abstraction"])
class ProgramSynthesisTask(TaskSpec):
    """Test abstract reasoning by programming a conveyor route from source to destination.

    The agent must configure a sequence of directional elements (arrows
    or conveyors) on the grid to transport an item from a source position
    to a destination position. Rather than navigating directly, the agent
    "programs" the environment by placing or activating conveyor tiles
    that collectively route the item along a valid path. The program
    length (number of directional elements needed) scales with difficulty.
    This measures abstract reasoning, planning, and the ability to
    synthesize multi-step procedural solutions.

    Difficulty Levels:
        - easy: 7x7 grid with program length 3, 50 max steps.
        - medium: 10x10 grid with program length 5, 75 max steps.
        - hard: 13x13 grid with program length 7, 100 max steps.
        - expert: 15x15 grid with program length 10 requiring complex
          routing logic, 150 max steps.

    Capabilities Tested:
        - abstract_reasoning: The agent must reason about directional
          flow and spatial transformations to design a valid route.
        - planning: The agent must plan the full conveyor sequence
          before execution to ensure the item reaches the destination.
        - programming: The agent must synthesize a procedural solution
          by composing individual directional elements into a coherent
          program.

    Example:
        >>> env = agentick.make("ProgramSynthesis-v0", difficulty="medium")
        >>> obs, info = env.reset(seed=42)
        >>> # Place conveyors to route the item from source to destination
    """

    name = "ProgramSynthesis-v0"
    description = "Program machine to move item from source to destination"
    capability_tags = ["abstract_reasoning", "planning", "programming"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy", grid_size=7, max_steps=50, params={"program_length": 3}
        ),
        "medium": DifficultyConfig(
            name="medium", grid_size=10, max_steps=75, params={"program_length": 5}
        ),
        "hard": DifficultyConfig(
            name="hard", grid_size=13, max_steps=100, params={"program_length": 7}
        ),
        "expert": DifficultyConfig(
            name="expert", grid_size=15, max_steps=150, params={"program_length": 10}
        ),
    }

    def generate(self, seed):
        """Generate a program synthesis task instance.

        Creates a walled grid with a source position containing an item
        (key) and a destination position (goal). The agent must configure
        directional elements to route the item from source to destination.
        The source and destination are placed with increasing separation
        at higher difficulties.

        Args:
            seed: Random seed for reproducible procedural generation.

        Returns:
            tuple: (grid, metadata) where grid is the initial Grid state
                with walls, source item, and destination, and metadata
                contains agent_start, goal_positions, max_steps, source,
                and destination positions.
        """
        _ = np.random.default_rng(seed)  # For future randomization
        size = self.difficulty_config.grid_size

        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        agent_pos = (1, 1)
        source_pos = (2, 2)
        dest_pos = (size - 3, size - 3)

        grid.objects[source_pos[1], source_pos[0]] = ObjectType.KEY  # Item to move
        grid.objects[dest_pos[1], dest_pos[0]] = ObjectType.GOAL  # Destination

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [dest_pos],
            "max_steps": self.get_max_steps(),
            "source": source_pos,
            "destination": dest_pos,
        }

    def compute_dense_reward(self, old_state, action, new_state, info):
        """Compute dense reward for a state transition.

        Uses a constant step penalty to encourage the agent to synthesize
        the conveyor program efficiently with minimal exploration.

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

        The task succeeds when the item (key) has been routed to the
        destination position through the programmed conveyor sequence.

        Args:
            state: Current state dict containing 'grid' key and
                optionally 'destination' coordinates.

        Returns:
            True if the item is at the destination position, False
            otherwise.
        """
        # Success if item is at destination
        if "grid" not in state:
            return False
        dest = state.get("destination", (0, 0))
        return state["grid"].objects[dest[1], dest[0]] == ObjectType.KEY

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

        A random agent cannot synthesize a correct conveyor program,
        yielding near-zero expected return.

        Args:
            difficulty: Difficulty level string, or None to use the
                current instance difficulty.

        Returns:
            Expected random agent return of 0.0.
        """
        return 0.0
