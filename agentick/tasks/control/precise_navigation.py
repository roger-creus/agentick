"""PreciseNavigation - Navigate narrow corridors where missteps are fatal.

MECHANICS:
  - Narrow 1-cell-wide corridors carved in a grid
  - HAZARD cells border the corridors (stepping off path = death)
  - Multiple waypoints (GEM) to collect along the path
  - After all gems collected, reach the GOAL
  - At hard+: moving waypoints and time pressure
  - Tests motor control precision, not path planning
"""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("PreciseNavigation-v0", tags=["motor_control", "planning"])
class PreciseNavigationTask(TaskSpec):
    """Navigate narrow hazard-bordered corridors, collecting gems to reach goal."""

    name = "PreciseNavigation-v0"
    description = "Navigate narrow corridors with fatal hazard borders"
    capability_tags = ["motor_control", "planning"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=9,
            max_steps=100,
            params={
                "n_waypoints": 2,
                "n_moving_waypoints": 0,
            },
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=12,
            max_steps=180,
            params={
                "n_waypoints": 3,
                "n_moving_waypoints": 0,
            },
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=15,
            max_steps=300,
            params={
                "n_waypoints": 4,
                "n_moving_waypoints": 1,
            },
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=18,
            max_steps=500,
            params={
                "n_waypoints": 5,
                "n_moving_waypoints": 2,
            },
        ),
    }

    _DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        n_wp = self.difficulty_config.params.get("n_waypoints", 2)
        n_mwp = self.difficulty_config.params.get("n_moving_waypoints", 0)

        grid = Grid(size, size)

        # Fill entire grid with HAZARD (fatal missteps)
        for y in range(size):
            for x in range(size):
                grid.terrain[y, x] = CellType.HAZARD
        # Border is walls
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        # Carve narrow corridors using randomized DFS maze
        # (cells at odd coordinates form a maze grid)
        visited = set()
        stack = [(1, 1)]
        visited.add((1, 1))
        grid.terrain[1, 1] = CellType.EMPTY

        while stack:
            cx, cy = stack[-1]
            dirs = [(2, 0), (-2, 0), (0, 2), (0, -2)]
            rng.shuffle(dirs)
            moved = False
            for dx, dy in dirs:
                nx, ny = cx + dx, cy + dy
                if 1 <= nx < size - 1 and 1 <= ny < size - 1 and (nx, ny) not in visited:
                    # Carve path cell and connecting cell
                    grid.terrain[cy + dy // 2, cx + dx // 2] = CellType.EMPTY
                    grid.terrain[ny, nx] = CellType.EMPTY
                    visited.add((nx, ny))
                    stack.append((nx, ny))
                    moved = True
                    break
            if not moved:
                stack.pop()

        agent_pos = (1, 1)

        # Collect all empty cells for placing objects
        empties = [
            (x, y)
            for y in range(1, size - 1)
            for x in range(1, size - 1)
            if grid.terrain[y, x] == CellType.EMPTY and (x, y) != agent_pos
        ]
        rng.shuffle(empties)

        # Place waypoints (GEM objects) and goal
        waypoints = empties[:n_wp]
        goal_pos = empties[n_wp] if len(empties) > n_wp else empties[-1]

        for wx, wy in waypoints:
            grid.objects[wy, wx] = ObjectType.GEM
        grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.GOAL

        # Tag which waypoints are "moving" (last n_mwp)
        n_mw = min(n_mwp, len(waypoints))
        moving_wp_indices = list(range(len(waypoints) - n_mw, len(waypoints)))

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [goal_pos],
            "waypoints": waypoints,
            "moving_wp_indices": moving_wp_indices,
            "_rng_seed": int(rng.integers(0, 2**31)),
            "max_steps": self.get_max_steps(),
        }

    # ── Hooks ────────────────────────────────────────────────────────────────

    def on_env_reset(self, agent, grid, config):
        config["_waypoints_remaining"] = list(config.get("waypoints", []))
        config["_dead"] = False
        config["_wp_rng"] = np.random.default_rng(config.get("_rng_seed", 0))
        self._config = config
        self._last_waypoints_rem = len(config.get("waypoints", []))

    def can_agent_enter(self, pos, agent, grid) -> bool:
        """Allow entry to hazard (will kill) but block walls."""
        x, y = pos
        if grid.terrain[y, x] == CellType.WALL:
            return False
        return True

    def on_agent_moved(self, pos, agent, grid):
        config = getattr(self, "_config", {})
        x, y = pos

        # Fatal misstep: stepping on hazard kills agent
        if grid.terrain[y, x] == CellType.HAZARD:
            config["_dead"] = True
            return

        # Collect waypoint gem
        remaining = config.get("_waypoints_remaining", [])
        if (x, y) in remaining:
            remaining.remove((x, y))
            grid.objects[y, x] = ObjectType.NONE
            config["_waypoints_remaining"] = remaining

    def on_env_step(self, agent, grid, config, step_count):
        """Move mobile waypoints periodically."""
        remaining = config.get("_waypoints_remaining", [])
        moving_indices = config.get("moving_wp_indices", [])
        all_wp = config.get("waypoints", [])
        rng = config.get("_wp_rng")
        if rng is None or not moving_indices:
            return

        if step_count % 4 == 0:
            for idx in moving_indices:
                if idx >= len(all_wp):
                    continue
                wp = all_wp[idx]
                if wp not in remaining:
                    continue
                wx, wy = wp
                moves = [(wx + dx, wy + dy) for dx, dy in self._DIRS]
                valid = [
                    (nx, ny)
                    for nx, ny in moves
                    if (
                        1 <= nx < grid.width - 1
                        and 1 <= ny < grid.height - 1
                        and grid.terrain[ny, nx] == CellType.EMPTY
                        and grid.objects[ny, nx] == ObjectType.NONE
                        and (nx, ny) not in remaining
                    )
                ]
                if valid:
                    new_pos = valid[int(rng.integers(len(valid)))]
                    if grid.objects[wy, wx] == ObjectType.GEM:
                        grid.objects[wy, wx] = ObjectType.NONE
                    nx, ny = new_pos
                    grid.objects[ny, nx] = ObjectType.GEM
                    ri = remaining.index(wp)
                    remaining[ri] = new_pos
                    all_wp[idx] = new_pos

            config["_waypoints_remaining"] = remaining

    # ── Reward & success ─────────────────────────────────────────────────────

    def compute_dense_reward(self, old_state, action, new_state, info):
        config = new_state.get("config", {})
        if config.get("_dead", False):
            return -1.0
        reward = -0.01
        new_rem = len(config.get("_waypoints_remaining", []))
        if new_rem < self._last_waypoints_rem:
            reward += 0.3 * (self._last_waypoints_rem - new_rem)
        self._last_waypoints_rem = new_rem
        # Shaping toward next waypoint or goal
        target = None
        remaining = config.get("_waypoints_remaining", [])
        if remaining:
            target = remaining[0]
        else:
            target = config.get("goal_positions", [None])[0]
        if target and "agent" in new_state:
            ax, ay = new_state["agent"].position
            ox, oy = old_state.get("agent_position", (ax, ay))
            old_d = abs(ox - target[0]) + abs(oy - target[1])
            new_d = abs(ax - target[0]) + abs(ay - target[1])
            reward += 0.05 * (old_d - new_d)
        if self.check_success(new_state):
            reward += 1.0
        return reward

    def compute_sparse_reward(self, old_state, action, new_state, info):
        config = new_state.get("config", {})
        if config.get("_dead", False):
            return -1.0
        if self.check_success(new_state):
            return 1.0
        return 0.0

    def check_success(self, state):
        """Agent at goal AND all waypoints collected AND not dead."""
        config = state.get("config", {})
        if config.get("_dead", False):
            return False
        remaining = config.get("_waypoints_remaining", None)
        if remaining is None:
            remaining = config.get("waypoints", [])
        if len(remaining) > 0:
            return False
        if "grid" not in state or "agent" not in state:
            return False
        x, y = state["agent"].position
        return bool(state["grid"].objects[y, x] == ObjectType.GOAL)

    def check_done(self, state):
        config = state.get("config", {})
        if config.get("_dead", False):
            return True
        return self.check_success(state)

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0
