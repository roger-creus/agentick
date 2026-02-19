"""MazeNavigation task - Solve procedurally generated mazes."""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.generation.maze import MazeConfig, MazeGenerator
from agentick.generation.validation import find_optimal_path, verify_solvable
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("MazeNavigation-v0", tags=["planning", "spatial_reasoning", "navigation"])
class MazeNavigationTask(TaskSpec):
    """
    Navigate through a procedurally generated maze to reach the goal.

    Uses recursive backtracking to generate perfect mazes.
    """

    name = "MazeNavigation-v0"
    description = "Solve procedurally generated mazes"
    capability_tags = ["planning", "spatial_reasoning", "navigation"]

    difficulty_configs = {
        # loop_freq: how many shortcut loops added | n_guards: patrolling enemies
        # algorithm: easy=sidewinder (simpler structure), hard=recursive_backtracker (denser)
        "easy": DifficultyConfig(
            name="easy",
            grid_size=7,
            max_steps=60,
            params={"loop_freq": 0.30, "n_guards": 0, "n_hazards": 0, "algorithm": "binary_tree"},
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=11,
            max_steps=120,
            params={"loop_freq": 0.15, "n_guards": 1, "n_hazards": 2, "algorithm": "recursive_backtracker"},
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=15,
            max_steps=250,
            params={"loop_freq": 0.08, "n_guards": 2, "n_hazards": 4, "algorithm": "recursive_backtracker"},
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=21,
            max_steps=500,
            params={"loop_freq": 0.03, "n_guards": 3, "n_hazards": 6, "algorithm": "recursive_backtracker"},
        ),
    }

    def _generate_maze(
        self, grid: Grid, rng: np.random.Generator,
        algorithm: str = "recursive_backtracker", loop_freq: float = 0.1
    ):
        """Generate maze using generation engine."""
        maze_config = MazeConfig(
            algorithm=algorithm,
            loop_frequency=loop_freq,
            dead_end_density=0.7,  # Keep most dead ends
        )

        maze_gen = MazeGenerator(rng=rng)
        terrain = maze_gen.generate(grid.width, grid.height, maze_config)
        grid.terrain[:, :] = terrain

    def generate(self, seed):
        """Generate a maze navigation task instance.

        Creates a procedurally generated maze using recursive
        backtracking. The agent is placed near the top-left and the
        goal near the bottom-right to maximize path length. The
        generator ensures odd grid dimensions for maze compatibility,
        verifies solvability with up to 10 retries, and computes the
        optimal path. Falls back to a simple open grid if maze
        generation repeatedly fails.

        Args:
            seed: Random seed for reproducible procedural generation.

        Returns:
            tuple: (grid, config) where grid is the initial Grid state
                with maze walls and goal, and config contains
                agent_start, goal_positions, max_steps, and the
                optimal solution length and path.
        """
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        p         = self.difficulty_config.params or {}
        algorithm = p.get("algorithm", "recursive_backtracker")
        loop_freq = p.get("loop_freq", 0.1)
        n_guards  = p.get("n_guards", 0)

        # Ensure odd size for maze generation
        if size % 2 == 0:
            size += 1

        grid = Grid(size, size)

        # Generate maze (pass algorithm from difficulty params)
        self._generate_maze(grid, rng, algorithm=algorithm, loop_freq=loop_freq)

        # Find valid positions (empty cells)
        valid_positions = []
        for y in range(size):
            for x in range(size):
                if grid.terrain[y, x] == CellType.EMPTY:
                    valid_positions.append((x, y))

        if len(valid_positions) < 2:
            raise ValueError("Generated maze has fewer than 2 empty cells")

        # Place agent at start (first valid position near top-left)
        agent_pos = None
        for y in range(1, size // 2):
            for x in range(1, size // 2):
                if grid.terrain[y, x] == CellType.EMPTY:
                    agent_pos = (x, y)
                    break
            if agent_pos:
                break

        if not agent_pos:
            agent_pos = valid_positions[0]

        # Place goal at opposite end (near bottom-right)
        goal_pos = None
        for y in range(size - 2, size // 2, -1):
            for x in range(size - 2, size // 2, -1):
                if grid.terrain[y, x] == CellType.EMPTY and (x, y) != agent_pos:
                    goal_pos = (x, y)
                    break
            if goal_pos:
                break

        if not goal_pos:
            # Find farthest valid position from agent
            max_dist = 0
            for pos in valid_positions:
                if pos != agent_pos:
                    dist = abs(pos[0] - agent_pos[0]) + abs(pos[1] - agent_pos[1])
                    if dist > max_dist:
                        max_dist = dist
                        goal_pos = pos
        grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.GOAL

        # Verify solvability and get optimal path
        max_retries = 10
        retry_count = 0
        while retry_count < max_retries:
            is_solvable = verify_solvable(grid, agent_pos, [goal_pos])
            if is_solvable:
                break
            # Regenerate maze if not solvable
            self._generate_maze(grid, rng, algorithm=algorithm, loop_freq=loop_freq)
            retry_count += 1

        if retry_count >= max_retries:
            # Fallback: create simple empty grid
            grid.terrain[:, :] = CellType.EMPTY
            grid.terrain[0, :] = CellType.WALL
            grid.terrain[-1, :] = CellType.WALL
            grid.terrain[:, 0] = CellType.WALL
            grid.terrain[:, -1] = CellType.WALL

        # Compute optimal solution length
        optimal_path, optimal_length = find_optimal_path(grid, agent_pos, [goal_pos])

        # Place guard NPCs in dead-end corridors (far from agent, not on goal or path)
        path_set = set(optimal_path) if optimal_path else set()
        guard_candidates = [
            pos for pos in valid_positions
            if pos != agent_pos and pos != goal_pos and pos not in path_set
        ]
        rng.shuffle(guard_candidates)
        guard_positions = guard_candidates[:n_guards]
        for gx, gy in guard_positions:
            grid.objects[gy, gx] = ObjectType.NPC

        # Place hazards at dead ends (cells with only one open neighbor)
        n_hazards = p.get("n_hazards", 0)
        if n_hazards > 0:
            dead_ends = []
            for pos in valid_positions:
                if pos == agent_pos or pos == goal_pos or pos in path_set:
                    continue
                if pos in set(guard_positions):
                    continue
                px, py = pos
                neighbors = sum(
                    1 for ddx, ddy in self._DIRS
                    if 0 <= px+ddx < size and 0 <= py+ddy < size
                    and grid.terrain[py+ddy, px+ddx] == CellType.EMPTY
                )
                if neighbors == 1:
                    dead_ends.append(pos)
            rng.shuffle(dead_ends)
            for hx, hy in dead_ends[:n_hazards]:
                grid.terrain[hy, hx] = CellType.HAZARD

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

    _DIRS = [(0,-1),(0,1),(-1,0),(1,0)]

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
        dirs   = config.get("_guard_dirs", [])
        rng    = config.get("_guard_rng")
        ax, ay = agent.position
        if not guards or rng is None: return
        for gx, gy in guards:
            if grid.objects[gy, gx] == ObjectType.NPC: grid.objects[gy, gx] = ObjectType.NONE
        new_g, new_d = [], []
        for i, (gx, gy) in enumerate(guards):
            d = dirs[i]; dx, dy = self._DIRS[d]; nx, ny = gx+dx, gy+dy
            if (0 < nx < grid.width-1 and 0 < ny < grid.height-1
                    and grid.terrain[ny, nx] == CellType.EMPTY
                    and grid.objects[ny, nx] != ObjectType.GOAL):
                new_g.append((nx, ny))
            else:
                d = int(rng.integers(0, 4)); new_g.append((gx, gy))
            new_d.append(d)
            if (new_g[-1][0], new_g[-1][1]) == (ax, ay): config["_guard_collision"] = True
        config["_guard_positions"] = new_g; config["_guard_dirs"] = new_d
        for gx, gy in new_g:
            if grid.terrain[gy, gx] == CellType.EMPTY: grid.objects[gy, gx] = ObjectType.NPC

    def check_done(self, state):
        config = state.get("config", {})
        if config.get("_guard_collision", False) or config.get("_hazard_hit", False):
            return True
        return self.check_success(state)

    def compute_dense_reward(self, old_state, action, new_state, info):
        """Distance-based reward with exploration bonus."""
        reward = -0.01  # Step penalty

        if "config" in new_state:
            config = new_state["config"]
            if "goal_positions" in config and config["goal_positions"]:
                goal_pos = config["goal_positions"][0]
                agent_pos = new_state["agent_position"]

                old_dist = abs(old_state["agent_position"][0] - goal_pos[0]) + abs(
                    old_state["agent_position"][1] - goal_pos[1]
                )
                new_dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])

                reward += 0.05 * (old_dist - new_dist)

        return reward

    def check_success(self, state):
        """Check if agent reached the goal."""
        config = state.get("config", {})
        if config.get("_guard_collision", False) or config.get("_hazard_hit", False):
            return False
        if "grid" not in state or "agent" not in state:
            return False
        x, y = state["agent"].position
        return bool(state["grid"].objects[y, x] == ObjectType.GOAL)

    def get_optimal_return(self, difficulty=None):
        """Optimal return is 1.0 for sparse reward."""
        return 1.0

    def get_random_baseline(self, difficulty=None):
        """Random agent is unlikely to solve maze."""
        return 0.0
