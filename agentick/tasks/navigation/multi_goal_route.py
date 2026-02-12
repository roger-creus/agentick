"""MultiGoalRoute task - Visit multiple goals in optimal order (TSP-like)."""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("MultiGoalRoute-v0", tags=["planning", "optimization", "navigation"])
class MultiGoalRouteTask(TaskSpec):
    """
    Visit N goals in optimal order (traveling salesman problem variant).

    Agent must visit all goals. Reward is higher for shorter paths.
    """

    name = "MultiGoalRoute-v0"
    description = "Visit N goals in optimal order (TSP-like)"
    capability_tags = ["planning", "optimization", "navigation"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=7,
            max_steps=100,
            params={"n_goals": 2},
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=10,
            max_steps=200,
            params={"n_goals": 3},
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=13,
            max_steps=300,
            params={"n_goals": 4},
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=15,
            max_steps=500,
            params={"n_goals": 5},
        ),
    }

    def generate(self, seed):
        """Generate a multi-goal route task instance.

        Creates a walled grid with random interior obstacles and N goal
        positions placed at random valid locations. The agent is placed
        at a random valid position. The number of goals scales with
        difficulty. If insufficient valid positions exist, the interior
        is cleared as a fallback.

        Args:
            seed: Random seed for reproducible procedural generation.

        Returns:
            tuple: (grid, config) where grid is the initial Grid state
                with walls and multiple goals, and config contains
                agent_start, goal_positions, goals_visited list, and
                max_steps.
        """
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        n_goals = self.difficulty_config.params.get("n_goals", 2)

        grid = Grid(size, size)

        # Add border walls
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        # Add some random obstacles
        for _ in range(size // 2):
            x, y = rng.integers(1, size - 1), rng.integers(1, size - 1)
            if grid.terrain[y, x] == CellType.EMPTY:
                grid.terrain[y, x] = CellType.WALL

        # Find valid positions
        valid_positions = []
        for y in range(1, size - 1):
            for x in range(1, size - 1):
                if grid.terrain[y, x] == CellType.EMPTY:
                    valid_positions.append((x, y))

        if len(valid_positions) < n_goals + 1:
            # Fallback: clear more space
            grid.terrain[1 : size - 1, 1 : size - 1] = CellType.EMPTY
            valid_positions = [(x, y) for x in range(1, size - 1) for y in range(1, size - 1)]

        # Place agent
        agent_idx = rng.choice(len(valid_positions))
        agent_pos = valid_positions[agent_idx]
        valid_positions.pop(agent_idx)

        # Place goals spread out
        goal_positions = []
        for _ in range(n_goals):
            if not valid_positions:
                break
            goal_idx = rng.choice(len(valid_positions))
            goal_pos = valid_positions[goal_idx]
            goal_positions.append(goal_pos)
            valid_positions.pop(goal_idx)
            grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.GOAL

        config = {
            "agent_start": agent_pos,
            "goal_positions": goal_positions,
            "goals_visited": [],
            "max_steps": self.get_max_steps(),
        }

        return grid, config

    def compute_dense_reward(self, old_state, action, new_state, info):
        """Reward for visiting goals and moving toward nearest unvisited goal."""
        reward = -0.01  # Step penalty

        # Check if visited a new goal
        if "config" in new_state:
            config = new_state["config"]
            agent_pos = new_state["agent_position"]

            # Track visited goals in state
            if "goals_visited" not in old_state:
                old_state["goals_visited"] = []
            if "goals_visited" not in new_state:
                new_state["goals_visited"] = old_state["goals_visited"].copy()

            # Check if at a goal position
            if agent_pos in config["goal_positions"]:
                if agent_pos not in new_state["goals_visited"]:
                    new_state["goals_visited"].append(agent_pos)
                    reward += 1.0  # Big reward for visiting new goal

        return reward

    def check_success(self, state):
        """Check if all goals have been visited."""
        if "config" not in state:
            return False

        config = state["config"]
        goals_visited = state.get("goals_visited", [])

        # All goals must be visited
        return len(goals_visited) == len(config["goal_positions"])

    def get_optimal_return(self, difficulty=None):
        """Optimal is visiting all goals with minimum steps."""
        diff = difficulty or self.difficulty
        n_goals = self.difficulty_configs[diff].params.get("n_goals", 2)
        return float(n_goals)  # 1.0 per goal in dense reward

    def get_random_baseline(self, difficulty=None):
        """Random agent unlikely to visit all goals."""
        return 0.0
