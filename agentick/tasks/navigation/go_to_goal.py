"""GoToGoal task - Navigate to a visible goal."""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.generation.validation import find_optimal_path, verify_solvable
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("GoToGoal-v0", tags=["basic_navigation", "navigation"])
class GoToGoalTask(TaskSpec):
    """
    Navigate to a visible goal in an open grid.

    Difficulty scaling:
    - easy: 5x5 grid, no obstacles
    - medium: 10x10 grid, sparse walls
    - hard: 15x15 grid, moderate walls
    - expert: 20x20 grid, dense walls
    """

    name = "GoToGoal-v0"
    description = "Navigate to a visible goal in open grid"
    capability_tags = ["basic_navigation", "navigation"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=5,
            max_steps=20,
            params={"wall_density": 0.0},
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=10,
            max_steps=50,
            params={"wall_density": 0.1},
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=15,
            max_steps=100,
            params={"wall_density": 0.2},
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=20,
            max_steps=200,
            params={"wall_density": 0.25},
        ),
    }

    def generate(self, seed):
        """Generate a go-to-goal task instance.

        Creates a walled grid with optional random interior walls based
        on wall density. The agent is placed at a random valid position,
        and the goal is placed at the farthest reachable position to
        maximize path length. The generator verifies solvability and
        computes the optimal path. Retries up to 10 times, falling back
        to a simple open layout if needed.

        Args:
            seed: Random seed for reproducible procedural generation.

        Returns:
            tuple: (grid, config) where grid is the initial Grid state
                with walls and goal, and config contains agent_start,
                goal_positions, max_steps, and optionally the optimal
                solution length and path.
        """
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        wall_density = self.difficulty_config.params.get("wall_density", 0.0)

        # Try multiple times to generate a valid instance
        max_attempts = 10
        for attempt in range(max_attempts):
            grid = Grid(size, size)

            # Add border walls
            grid.terrain[0, :] = CellType.WALL
            grid.terrain[-1, :] = CellType.WALL
            grid.terrain[:, 0] = CellType.WALL
            grid.terrain[:, -1] = CellType.WALL

            # Add random interior walls
            if wall_density > 0:
                for y in range(1, size - 1):
                    for x in range(1, size - 1):
                        if rng.random() < wall_density:
                            grid.terrain[y, x] = CellType.WALL

            # Find valid positions
            valid_positions = []
            for y in range(1, size - 1):
                for x in range(1, size - 1):
                    if grid.terrain[y, x] == CellType.EMPTY:
                        valid_positions.append((x, y))

            if len(valid_positions) < 2:
                continue  # Try again

            # Place agent
            agent_idx = rng.choice(len(valid_positions))
            agent_pos = valid_positions[agent_idx]

            # Find reachable positions from agent
            reachable = grid.flood_fill(agent_pos)
            reachable_positions = [
                pos for pos in valid_positions if pos in reachable and pos != agent_pos
            ]

            if not reachable_positions:
                continue  # Try again

            # Find goal position (maximize distance from agent, must be reachable)
            goal_pos = reachable_positions[0]
            max_dist = 0
            for pos in reachable_positions:
                dist = abs(pos[0] - agent_pos[0]) + abs(pos[1] - agent_pos[1])
                if dist > max_dist:
                    max_dist = dist
                    goal_pos = pos

            # Place goal
            grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.GOAL

            # Verify solvability and compute optimal path
            if verify_solvable(grid, agent_pos, [goal_pos]):
                optimal_path, optimal_length = find_optimal_path(grid, agent_pos, [goal_pos])

                config = {
                    "agent_start": agent_pos,
                    "goal_positions": [goal_pos],
                    "max_steps": self.get_max_steps(),
                    "_optimal_solution_length": optimal_length,
                    "_optimal_path": optimal_path,
                }
                return grid, config

        # Fallback: create a simple solvable instance
        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        agent_pos = (1, 1)
        goal_pos = (size - 2, size - 2)
        grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.GOAL

        config = {
            "agent_start": agent_pos,
            "goal_positions": [goal_pos],
            "max_steps": self.get_max_steps(),
        }

        return grid, config

    def compute_dense_reward(self, old_state, action, new_state, info):
        """Distance-based shaping reward."""
        # Small step penalty
        reward = -0.01

        # Reward for getting closer to goal
        if "config" in new_state:
            config = new_state["config"]
            if "goal_positions" in config and config["goal_positions"]:
                goal_pos = config["goal_positions"][0]
                agent_pos = new_state["agent_position"]

                old_dist = abs(old_state["agent_position"][0] - goal_pos[0]) + abs(
                    old_state["agent_position"][1] - goal_pos[1]
                )
                new_dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])

                # Reward for getting closer, penalty for getting farther
                reward += 0.1 * (old_dist - new_dist)

        return reward

    def check_success(self, state):
        """Check if agent reached the goal."""
        if "grid" not in state or "agent" not in state:
            return False

        grid = state["grid"]
        agent = state["agent"]
        x, y = agent.position

        return grid.objects[y, x] == ObjectType.GOAL

    def get_optimal_return(self, difficulty=None):
        """Optimal return is 1.0 (sparse reward on success)."""
        return 1.0

    def get_random_baseline(self, difficulty=None):
        """Random agent has very low chance of reaching goal."""
        diff = difficulty or self.difficulty
        size = self.difficulty_configs[diff].grid_size
        # Random walk in NxN grid has low success rate
        return 1.0 / (size * size)
