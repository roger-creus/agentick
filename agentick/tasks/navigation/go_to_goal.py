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
            params={
                "wall_density": 0.0,
                "n_guards": 0,
                "n_hazards": 0,
                "n_decoys": 0,
            },
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=10,
            max_steps=50,
            params={
                "wall_density": 0.12,
                "n_guards": 1,
                "n_hazards": 0,
                "n_decoys": 0,
            },
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=15,
            max_steps=100,
            params={
                "wall_density": 0.20,
                "n_guards": 2,
                "n_hazards": 4,
                "n_decoys": 2,
            },
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=20,
            max_steps=200,
            params={
                "wall_density": 0.25,
                "n_guards": 3,
                "n_hazards": 8,
                "n_decoys": 4,
            },
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

                # Place hazard terrain cells (agent loses if stepped on)
                n_hazards = self.difficulty_config.params.get("n_hazards", 0)
                hazard_candidates = [
                    p
                    for p in reachable_positions
                    if p != goal_pos and p not in (optimal_path or [])
                ]
                rng.shuffle(hazard_candidates)
                for hx, hy in hazard_candidates[:n_hazards]:
                    grid.terrain[hy, hx] = CellType.HAZARD

                # Place decoy goals (look like goal but aren't)
                n_decoys = self.difficulty_config.params.get("n_decoys", 0)
                decoy_candidates = [
                    p
                    for p in reachable_positions
                    if p != goal_pos
                    and grid.terrain[p[1], p[0]] == CellType.EMPTY
                    and p not in (optimal_path or [])
                ]
                rng.shuffle(decoy_candidates)
                for dx, dy in decoy_candidates[:n_decoys]:
                    grid.objects[dy, dx] = ObjectType.TARGET

                # Place guard NPCs on reachable empty cells
                n_guards = self.difficulty_config.params.get("n_guards", 0)
                guard_candidates = [
                    p
                    for p in reachable_positions
                    if p != goal_pos
                    and grid.terrain[p[1], p[0]] == CellType.EMPTY
                    and abs(p[0] - agent_pos[0]) + abs(p[1] - agent_pos[1]) > 2
                ]
                rng.shuffle(guard_candidates)
                guard_positions = guard_candidates[:n_guards]
                for gx, gy in guard_positions:
                    grid.objects[gy, gx] = ObjectType.NPC

                config = {
                    "agent_start": agent_pos,
                    "goal_positions": [goal_pos],
                    "max_steps": self.get_max_steps(),
                    "_optimal_solution_length": optimal_length,
                    "_optimal_path": optimal_path,
                    "_guard_positions": guard_positions,
                    "_guard_dirs": [int(rng.integers(0, 4)) for _ in guard_positions],
                    "_guard_seed": int(rng.integers(0, 2**31)),
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
            "_guard_positions": [],
            "_guard_dirs": [],
            "_guard_seed": 0,
        }

        return grid, config

    _DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    def on_env_reset(self, agent, grid, config):
        config["_guard_collision"] = False
        config["_hazard_hit"] = False
        config["_guard_rng"] = np.random.default_rng(config.get("_guard_seed", 0))
        self._config = config

    def on_agent_moved(self, pos, agent, grid):
        x, y = pos
        config = getattr(self, "_config", {})
        if grid.terrain[y, x] == CellType.HAZARD:
            config["_hazard_hit"] = True
        if grid.objects[y, x] == ObjectType.NPC:
            config["_guard_collision"] = True

    def on_env_step(self, agent, grid, config, step_count):
        guards = config.get("_guard_positions", [])
        dirs = config.get("_guard_dirs", [])
        rng = config.get("_guard_rng")
        ax, ay = agent.position
        if not guards or rng is None:
            return
        for gx, gy in guards:
            if grid.objects[gy, gx] == ObjectType.NPC:
                grid.objects[gy, gx] = ObjectType.NONE
        new_g, new_d = [], []
        for i, (gx, gy) in enumerate(guards):
            d = dirs[i]
            dx, dy = self._DIRS[d]
            nx, ny = gx + dx, gy + dy
            if (
                0 < nx < grid.width - 1
                and 0 < ny < grid.height - 1
                and grid.terrain[ny, nx] == CellType.EMPTY
                and grid.objects[ny, nx] != ObjectType.GOAL
            ):
                new_g.append((nx, ny))
            else:
                d = int(rng.integers(0, 4))
                new_g.append((gx, gy))
            new_d.append(d)
            if (new_g[-1][0], new_g[-1][1]) == (ax, ay):
                config["_guard_collision"] = True
        config["_guard_positions"] = new_g
        config["_guard_dirs"] = new_d
        for gx, gy in new_g:
            if grid.terrain[gy, gx] == CellType.EMPTY:
                grid.objects[gy, gx] = ObjectType.NPC

    def compute_dense_reward(self, old_state, action, new_state, info):
        """Distance-based shaping reward with success bonus."""
        config = new_state.get("config", {})
        if config.get("_guard_collision", False) or config.get("_hazard_hit", False):
            return -1.0
        reward = -0.01
        if "config" in new_state:
            if "goal_positions" in config and config["goal_positions"]:
                goal_pos = config["goal_positions"][0]
                if "agent" in new_state:
                    ax, ay = new_state["agent"].position
                    ox, oy = old_state.get("agent_position", (ax, ay))
                    old_dist = abs(ox - goal_pos[0]) + abs(oy - goal_pos[1])
                    new_dist = abs(ax - goal_pos[0]) + abs(ay - goal_pos[1])
                    reward += 0.1 * (old_dist - new_dist)
        if self.check_success(new_state):
            reward += 1.0
        return reward

    def check_done(self, state):
        config = state.get("config", {})
        if config.get("_guard_collision", False) or config.get("_hazard_hit", False):
            return True
        return self.check_success(state)

    def check_success(self, state):
        config = state.get("config", {})
        if config.get("_guard_collision", False) or config.get("_hazard_hit", False):
            return False
        if "grid" not in state or "agent" not in state:
            return False
        x, y = state["agent"].position
        return bool(state["grid"].objects[y, x] == ObjectType.GOAL)

    def get_optimal_return(self, difficulty=None):
        """Optimal return is 1.0 (sparse reward on success)."""
        return 1.0

    def get_random_baseline(self, difficulty=None):
        """Random agent has very low chance of reaching goal."""
        diff = difficulty or self.difficulty
        size = self.difficulty_configs[diff].grid_size
        # Random walk in NxN grid has low success rate
        return 1.0 / (size * size)
