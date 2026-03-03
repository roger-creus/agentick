"""DynamicObstacles - Navigate while moving NPC obstacles patrol the grid.

MECHANICS:
  - NPC objects move probabilistically each step (configurable move probability)
  - Collision with any NPC ends the episode in failure
  - At expert: 15% of NPC moves are pursuing (toward agent) instead of random

DIFFICULTY AXES:
  - easy:   2 NPCs, small map, slow (50% move chance)
  - medium: 3 NPCs, medium map, normal speed (75% move chance), 3 walls
  - hard:   5 NPCs, large map, fast (100% move chance), 6 walls
  - expert: 7 NPCs, largest map, 90% move chance + 15% pursuit, 9 walls
"""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("DynamicObstacles-v0", tags=["reactive_planning", "navigation"])
class DynamicObstaclesTask(TaskSpec):
    """Navigate to goal while NPC obstacles patrol the grid each timestep."""

    name = "DynamicObstacles-v0"
    description = "Navigate while obstacles move"
    capability_tags = ["reactive_planning", "navigation"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=7,
            max_steps=60,
            params={"n_obs": 2, "move_prob": 0.50, "pursuing": False, "n_walls": 0},
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=10,
            max_steps=100,
            params={"n_obs": 3, "move_prob": 0.75, "pursuing": False, "n_walls": 3},
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=13,
            max_steps=150,
            params={"n_obs": 5, "move_prob": 1.00, "pursuing": False, "n_walls": 6},
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=15,
            max_steps=200,
            params={
                "n_obs": 7,
                "move_prob": 0.90,
                "pursuing": True,
                "pursue_prob": 0.15,
                "n_walls": 9,
            },
        ),
    }

    _DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    # Map _DIRS index → metadata direction (0=up, 1=right, 2=down, 3=left)
    _DIR_TO_META = {0: 0, 1: 2, 2: 3, 3: 1}

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        p = self.difficulty_config.params
        n_obs = p.get("n_obs", 2)
        move_prob = p.get("move_prob", 0.5)
        pursuing = p.get("pursuing", False)
        pursue_prob = p.get("pursue_prob", 1.0)

        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        # Random agent and goal positions (opposite corners/areas)
        agent_pos = (int(rng.integers(1, 3)), int(rng.integers(1, 3)))
        goal_pos = (int(rng.integers(size - 3, size - 1)), int(rng.integers(size - 3, size - 1)))

        grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.GOAL

        # Add interior walls to create chokepoints
        n_walls = p.get("n_walls", 0)
        if n_walls > 0:
            wall_candidates = [
                (x, y)
                for x in range(1, size - 1)
                for y in range(1, size - 1)
                if (x, y) != agent_pos and (x, y) != goal_pos
            ]
            rng.shuffle(wall_candidates)
            placed = 0
            for wx, wy in wall_candidates:
                if placed >= n_walls:
                    break
                grid.terrain[wy, wx] = CellType.WALL
                if goal_pos in grid.flood_fill(agent_pos):
                    placed += 1
                else:
                    grid.terrain[wy, wx] = CellType.EMPTY

        # Place obstacles away from agent start (safe zone radius 2)
        candidates = [
            (x, y)
            for x in range(1, size - 1)
            for y in range(1, size - 1)
            if (x, y) != agent_pos
            and (x, y) != goal_pos
            and abs(x - agent_pos[0]) + abs(y - agent_pos[1]) > 2
        ]
        rng.shuffle(candidates)
        obstacle_positions = candidates[: min(n_obs, len(candidates))]

        obs_dirs = [int(rng.integers(0, 4)) for _ in obstacle_positions]

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [goal_pos],
            "max_steps": self.get_max_steps(),
            "_obstacle_positions": obstacle_positions,
            "_obstacle_dirs": obs_dirs,
            "_obs_seed": int(rng.integers(0, 2**31)),
            "_move_prob": move_prob,
            "_pursuing": pursuing,
            "_pursue_prob": pursue_prob,
        }

    def on_env_reset(self, agent, grid, config):
        config["_live_obstacles"] = list(config["_obstacle_positions"])
        config["_live_dirs"] = list(config["_obstacle_dirs"])
        config["_obs_rng"] = np.random.default_rng(config["_obs_seed"])
        config["_collision"] = False
        self._draw_obstacles(
            grid, config["_live_obstacles"], config["_live_dirs"], draw=True,
        )

    def on_env_step(self, agent, grid, config, step_count):
        obstacles = config["_live_obstacles"]
        dirs = config["_live_dirs"]
        rng = config["_obs_rng"]
        move_prob = config.get("_move_prob", 1.0)
        pursuing = config.get("_pursuing", False)
        pursue_prob = config.get("_pursue_prob", 1.0)
        ax, ay = agent.position

        self._draw_obstacles(grid, obstacles, dirs, draw=False)

        new_obs, new_dirs = [], []
        for i, (ox, oy) in enumerate(obstacles):
            if rng.random() > move_prob:
                new_obs.append((ox, oy))
                new_dirs.append(dirs[i])
                continue

            d = dirs[i]
            if pursuing and rng.random() < pursue_prob:
                # Move toward agent (stochastic pursuit)
                best, best_d = (ox, oy), abs(ox - ax) + abs(oy - ay)
                for di, (ddx, ddy) in enumerate(self._DIRS):
                    nx, ny = ox + ddx, oy + ddy
                    if (
                        0 < nx < grid.width - 1
                        and 0 < ny < grid.height - 1
                        and grid.terrain[ny, nx] == CellType.EMPTY
                        and grid.objects[ny, nx] != ObjectType.GOAL
                    ):
                        dist = abs(nx - ax) + abs(ny - ay)
                        if dist < best_d:
                            best_d, best = dist, (nx, ny)
                            d = di
                new_obs.append(best)
            else:
                # Bounce/random walk
                dx, dy = self._DIRS[d]
                nx, ny = ox + dx, oy + dy
                if (
                    nx <= 0
                    or nx >= grid.width - 1
                    or ny <= 0
                    or ny >= grid.height - 1
                    or grid.terrain[ny, nx] != CellType.EMPTY
                    or grid.objects[ny, nx] == ObjectType.GOAL
                ):
                    # Try a random new direction
                    for _ in range(4):
                        d = int(rng.integers(0, 4))
                        dx, dy = self._DIRS[d]
                        nx, ny = ox + dx, oy + dy
                        if (
                            0 < nx < grid.width - 1
                            and 0 < ny < grid.height - 1
                            and grid.terrain[ny, nx] == CellType.EMPTY
                            and grid.objects[ny, nx] != ObjectType.GOAL
                        ):
                            break
                    else:
                        nx, ny = ox, oy
                new_obs.append((nx, ny))
            new_dirs.append(d)

        config["_live_obstacles"] = new_obs
        config["_live_dirs"] = new_dirs
        self._draw_obstacles(grid, new_obs, new_dirs, draw=True)

        # Collision check: use GRID OBJECT (robust, no X,Y flip bug)
        cur_ax, cur_ay = agent.position
        # Check if any obstacle is now on the agent's cell
        if grid.objects[cur_ay, cur_ax] == ObjectType.NPC:
            config["_collision"] = True

    def _draw_obstacles(self, grid, obstacles, dirs, draw: bool):
        for idx, (ox, oy) in enumerate(obstacles):
            if 0 <= ox < grid.width and 0 <= oy < grid.height:
                if draw:
                    if grid.objects[oy, ox] != ObjectType.GOAL:
                        grid.objects[oy, ox] = ObjectType.NPC
                        meta = self._DIR_TO_META.get(dirs[idx], 2) if idx < len(dirs) else 2
                        grid.metadata[oy, ox] = meta
                else:
                    if grid.objects[oy, ox] == ObjectType.NPC:
                        grid.objects[oy, ox] = ObjectType.NONE
                        grid.metadata[oy, ox] = 0

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        if new_state.get("config", {}).get("_collision", False):
            return reward - 1.0
        if self.check_success(new_state):
            reward += 1.0
        config = new_state.get("config", {})
        goal = config.get("goal_positions", [None])[0]
        if goal and "agent" in new_state:
            ax, ay = new_state["agent"].position
            ox, oy = old_state.get("agent_position", (ax, ay))
            reward += 0.05 * (
                (abs(ox - goal[0]) + abs(oy - goal[1])) - (abs(ax - goal[0]) + abs(ay - goal[1]))
            )
        return reward

    def compute_sparse_reward(self, old_state, action, new_state, info):
        if new_state.get("config", {}).get("_collision", False):
            return -1.0
        if "grid" in new_state and "agent" in new_state:
            x, y = new_state["agent"].position
            if new_state["grid"].objects[y, x] == ObjectType.GOAL:
                return 1.0
        return 0.0

    def check_done(self, state):
        if state.get("config", {}).get("_collision", False):
            return True
        return self.check_success(state)

    def check_success(self, state):
        if state.get("config", {}).get("_collision", False):
            return False
        if "grid" not in state or "agent" not in state:
            return False
        x, y = state["agent"].position
        # Use grid object check (robust)
        return bool(state["grid"].objects[y, x] == ObjectType.GOAL)

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0
