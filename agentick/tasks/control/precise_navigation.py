"""PreciseNavigation - Navigate to exact randomized targets in a narrow maze.

MECHANICS:
  - A narrow maze with tight corridors (width=1 cells)
  - Multiple waypoints that must all be visited
  - Agent must visit EVERY waypoint (TARGET) to reach the GOAL
  - Tests fine motor control and spatial precision
"""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("PreciseNavigation-v0", tags=["motor_control", "planning"])
class PreciseNavigationTask(TaskSpec):
    """Navigate narrow corridors, visiting all waypoints before the goal."""

    name = "PreciseNavigation-v0"
    description = "Visit all waypoints in a narrow maze to reach goal"
    capability_tags = ["motor_control", "planning"]

    difficulty_configs = {
        "easy":   DifficultyConfig(
            name="easy", grid_size=9, max_steps=100,
            params={
                "n_waypoints": 2, "tight_windows": False,
                "n_hazards": 0,
                "time_windows": 0, "n_moving_waypoints": 0,
            },
        ),
        "medium": DifficultyConfig(
            name="medium", grid_size=12, max_steps=180,
            params={
                "n_waypoints": 3, "tight_windows": False,
                "n_hazards": 2,
                "time_windows": 0, "n_moving_waypoints": 0,
            },
        ),
        "hard":   DifficultyConfig(
            name="hard", grid_size=15, max_steps=300,
            params={
                "n_waypoints": 4, "tight_windows": True,
                "n_hazards": 4,
                "time_windows": 30, "n_moving_waypoints": 1,
            },
        ),
        "expert": DifficultyConfig(
            name="expert", grid_size=18, max_steps=500,
            params={
                "n_waypoints": 5, "tight_windows": True,
                "n_hazards": 7,
                "time_windows": 20, "n_moving_waypoints": 2,
            },
        ),
    }

    _DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size              = self.difficulty_config.grid_size
        n_wp              = self.difficulty_config.params.get(
            "n_waypoints", 2
        )
        tight_windows     = self.difficulty_config.params.get(
            "tight_windows", False
        )
        n_hazards         = self.difficulty_config.params.get(
            "n_hazards", 0
        )
        time_windows      = self.difficulty_config.params.get(
            "time_windows", 0
        )
        n_moving_wp       = self.difficulty_config.params.get(
            "n_moving_waypoints", 0
        )

        grid = Grid(size, size)

        for y in range(size):
            for x in range(size):
                grid.terrain[y, x] = CellType.WALL

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
                if (1 <= nx < size - 1
                        and 1 <= ny < size - 1
                        and (nx, ny) not in visited):
                    grid.terrain[
                        cy + dy // 2, cx + dx // 2
                    ] = CellType.EMPTY
                    grid.terrain[ny, nx] = CellType.EMPTY
                    visited.add((nx, ny))
                    stack.append((nx, ny))
                    moved = True
                    break
            if not moved:
                stack.pop()

        agent_pos = (1, 1)

        empties = [
            (x, y) for y in range(1, size - 1)
            for x in range(1, size - 1)
            if grid.terrain[y, x] == CellType.EMPTY
            and (x, y) != agent_pos
        ]
        rng.shuffle(empties)

        waypoints = empties[:n_wp]
        goal_pos = (
            empties[n_wp] if len(empties) > n_wp else empties[-1]
        )
        used = {agent_pos, goal_pos} | set(waypoints)

        for wx, wy in waypoints:
            grid.objects[wy, wx] = ObjectType.TARGET
        grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.GOAL

        hazard_positions = []
        hazard_candidates = [
            p for p in empties[n_wp + 1:] if p not in used
        ]
        if tight_windows:
            rng.shuffle(hazard_candidates)
        for p in hazard_candidates:
            if len(hazard_positions) >= n_hazards:
                break
            hx, hy = p
            grid.terrain[hy, hx] = CellType.HAZARD
            reachable = grid.flood_fill(agent_pos)
            critical = [goal_pos] + list(waypoints)
            if all(q in reachable for q in critical):
                hazard_positions.append(p)
            else:
                grid.terrain[hy, hx] = CellType.EMPTY

        # Tag which waypoints are "moving" (last n_moving_wp)
        n_mw = min(n_moving_wp, len(waypoints))
        moving_wp_indices = list(range(len(waypoints) - n_mw, len(waypoints)))

        return grid, {
            "agent_start":       agent_pos,
            "goal_positions":    [goal_pos],
            "waypoints":         waypoints,
            "tight_windows":     tight_windows,
            "hazard_positions":  hazard_positions,
            "time_windows":      time_windows,
            "moving_wp_indices": moving_wp_indices,
            "_rng_seed":         int(rng.integers(0, 2**31)),
            "max_steps":         self.get_max_steps(),
        }

    # ── Hooks ────────────────────────────────────────────────────────────────

    def on_env_reset(self, agent, grid, config):
        config["_waypoints_remaining"] = list(
            config.get("waypoints", [])
        )
        config["_wp_rng"] = np.random.default_rng(
            config.get("_rng_seed", 0)
        )
        config["_wp_spawn_step"] = {}
        for i, wp in enumerate(config.get("waypoints", [])):
            config["_wp_spawn_step"][i] = 0
        self._config = config
        self._last_waypoints_rem = len(
            config.get("waypoints", [])
        )

    def on_agent_moved(self, pos, agent, grid):
        config = getattr(self, "_config", {})
        remaining = config.get("_waypoints_remaining", [])
        ax, ay = pos
        if (ax, ay) in remaining:
            remaining.remove((ax, ay))
            grid.objects[ay, ax] = ObjectType.NONE
            config["_waypoints_remaining"] = remaining

    def on_env_step(self, agent, grid, config, step_count):
        remaining = config.get("_waypoints_remaining", [])
        time_win = config.get("time_windows", 0)
        moving_indices = config.get("moving_wp_indices", [])
        all_wp = config.get("waypoints", [])
        rng = config.get("_wp_rng")
        if rng is None:
            return

        # Time windows: remove waypoints that have been alive too long
        if time_win > 0:
            spawn_steps = config.get("_wp_spawn_step", {})
            expired = []
            for wp in list(remaining):
                try:
                    idx = all_wp.index(wp)
                except ValueError:
                    continue
                age = step_count - spawn_steps.get(idx, 0)
                if age > time_win:
                    expired.append(wp)
            for wp in expired:
                wx, wy = wp
                if grid.objects[wy, wx] == ObjectType.TARGET:
                    grid.objects[wy, wx] = ObjectType.NONE
                remaining.remove(wp)
                try:
                    idx = all_wp.index(wp)
                except ValueError:
                    continue
                empties = [
                    (x, y)
                    for x in range(1, grid.width - 1)
                    for y in range(1, grid.height - 1)
                    if (grid.terrain[y, x] == CellType.EMPTY
                        and grid.objects[y, x] == ObjectType.NONE
                        and (x, y) not in remaining
                        and (x, y) != agent.position)
                ]
                if empties:
                    new_pos = empties[
                        int(rng.integers(len(empties)))
                    ]
                    nx, ny = new_pos
                    grid.objects[ny, nx] = ObjectType.TARGET
                    remaining.append(new_pos)
                    all_wp[idx] = new_pos
                    spawn_steps[idx] = step_count

        # Moving waypoints: shift position every 4 steps
        if moving_indices and step_count % 4 == 0:
            for idx in moving_indices:
                if idx >= len(all_wp):
                    continue
                wp = all_wp[idx]
                if wp not in remaining:
                    continue
                wx, wy = wp
                moves = [
                    (wx + dx, wy + dy)
                    for dx, dy in self._DIRS
                ]
                valid = [
                    (x, y) for x, y in moves
                    if (1 <= x < grid.width - 1
                        and 1 <= y < grid.height - 1
                        and grid.terrain[y, x] == CellType.EMPTY
                        and grid.objects[y, x] == ObjectType.NONE
                        and (x, y) not in remaining)
                ]
                if valid:
                    new_pos = valid[
                        int(rng.integers(len(valid)))
                    ]
                    if grid.objects[wy, wx] == ObjectType.TARGET:
                        grid.objects[wy, wx] = ObjectType.NONE
                    nx, ny = new_pos
                    grid.objects[ny, nx] = ObjectType.TARGET
                    ri = remaining.index(wp)
                    remaining[ri] = new_pos
                    all_wp[idx] = new_pos

        config["_waypoints_remaining"] = remaining

    # ── Reward & success ─────────────────────────────────────────────────────

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        config = new_state.get("config", {})
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
        if target and "agent_position" in new_state:
            ax, ay = new_state["agent_position"]
            ox, oy = old_state.get("agent_position", (ax, ay))
            old_d = abs(ox - target[0]) + abs(oy - target[1])
            new_d = abs(ax - target[0]) + abs(ay - target[1])
            reward += 0.05 * (old_d - new_d)
        if self.check_success(new_state):
            reward += 1.0
        return reward

    def check_success(self, state):
        """Agent at goal AND all waypoints collected."""
        config = state.get("config", {})
        remaining = config.get("_waypoints_remaining", None)
        if remaining is None:
            remaining = config.get("waypoints", [])
        if len(remaining) > 0:
            return False
        if "grid" not in state or "agent" not in state:
            return False
        x, y = state["agent"].position
        return bool(state["grid"].objects[y, x] == ObjectType.GOAL)

    def get_optimal_return(self, difficulty=None): return 1.0
    def get_random_baseline(self, difficulty=None): return 0.0
