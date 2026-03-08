"""NoisyObservation - Navigate to the true goal among heavy visual noise.

MECHANICS:
  - One TRUE goal (GOAL) placed at a random position
  - N decoy targets (TARGET) look similar to goal — touching them wastes time
  - Ghost objects: random SCROLL/ORB objects appear and disappear each step
  - Moving decoys drift around the grid (hard+)
  - Guards patrol and collide with agent (medium+)
  - Terrain noise: random ICE/WATER patches flicker in metadata layer
  - Tests robustness to distracting/misleading observations
"""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task

_NO_DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]


@register_task("NoisyObservation-v0", tags=["robustness", "navigation", "noise"])
class NoisyObservationTask(TaskSpec):
    """Navigate to the true goal through heavy visual noise."""

    name = "NoisyObservation-v0"
    description = "Find true goal among heavy noise and ghost objects"
    capability_tags = ["robustness", "navigation", "noise"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=7,
            max_steps=100,
            params={
                "n_decoys": 3,
                "n_obstacles": 3,
                "n_guards": 0,
                "moving_decoys": False,
                "n_moving_decoys": 0,
                "n_ghosts": 2,
            },
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=9,
            max_steps=150,
            params={
                "n_decoys": 6,
                "n_obstacles": 5,
                "n_guards": 1,
                "moving_decoys": False,
                "n_moving_decoys": 0,
                "n_ghosts": 4,
            },
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=11,
            max_steps=220,
            params={
                "n_decoys": 10,
                "n_obstacles": 8,
                "n_guards": 2,
                "moving_decoys": True,
                "n_moving_decoys": 4,
                "n_ghosts": 6,
            },
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=13,
            max_steps=320,
            params={
                "n_decoys": 14,
                "n_obstacles": 12,
                "n_guards": 3,
                "moving_decoys": True,
                "n_moving_decoys": 6,
                "n_ghosts": 10,
            },
        ),
    }

    _DIRS = _NO_DIRS
    _GHOST_TYPES = [ObjectType.SCROLL, ObjectType.ORB, ObjectType.LEVER, ObjectType.COIN]

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        n_decoys = self.difficulty_config.params.get("n_decoys", 3)
        n_obs = self.difficulty_config.params.get("n_obstacles", 3)
        n_guards = self.difficulty_config.params.get("n_guards", 0)
        moving_decoys = self.difficulty_config.params.get("moving_decoys", False)
        n_moving_decoys = self.difficulty_config.params.get("n_moving_decoys", 0)
        n_ghosts = self.difficulty_config.params.get("n_ghosts", 2)

        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        corners = [(1, 1), (size - 2, 1), (1, size - 2), (size - 2, size - 2)]
        agent_pos = tuple(corners[int(rng.integers(0, len(corners)))])
        interior = [
            (x, y) for x in range(1, size - 1) for y in range(1, size - 1) if (x, y) != agent_pos
        ]
        rng.shuffle(interior)

        non_corner = [(x, y) for (x, y) in interior if not (x == size - 2 and y == size - 2)]

        # For hard+ difficulties, enforce minimum goal distance from agent
        min_dist = size // 2 if self.difficulty in ("hard", "expert") else 0
        goal_pos = non_corner[0]
        if min_dist > 0:
            far = [
                p for p in non_corner
                if abs(p[0] - agent_pos[0]) + abs(p[1] - agent_pos[1]) >= min_dist
            ]
            goal_pos = far[0] if far else non_corner[0]
        used = {goal_pos, agent_pos}

        decoys = []
        for p in non_corner[1:]:
            if p not in used and len(decoys) < n_decoys:
                decoys.append(p)
                used.add(p)

        walls = []
        candidates = [p for p in interior if p not in used]
        for p in candidates:
            if len(walls) >= n_obs:
                break
            wx, wy = p
            grid.terrain[wy, wx] = CellType.WALL
            reachable = grid.flood_fill(agent_pos)
            if goal_pos not in reachable:
                grid.terrain[wy, wx] = CellType.EMPTY
            else:
                walls.append(p)
                used.add(p)

        for dx2, dy2 in decoys:
            grid.objects[dy2, dx2] = ObjectType.TARGET
        grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.GOAL

        # Guards
        reachable = grid.flood_fill(agent_pos)
        guard_candidates = [
            p
            for p in reachable
            if p not in used
            and p != goal_pos
            and abs(p[0] - agent_pos[0]) + abs(p[1] - agent_pos[1]) > 2
        ]
        rng.shuffle(guard_candidates)
        guard_positions = guard_candidates[:n_guards]
        for gx, gy in guard_positions:
            grid.objects[gy, gx] = ObjectType.NPC
            used.add((gx, gy))

        # Moving decoys subset
        md_positions = []
        md_dirs = []
        if moving_decoys and n_moving_decoys > 0 and decoys:
            n_md = min(n_moving_decoys, len(decoys))
            md_positions = list(decoys[:n_md])
            md_dirs = [int(rng.integers(0, 4)) for _ in md_positions]

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [goal_pos],
            "decoy_positions": decoys,
            "moving_decoys": moving_decoys,
            "n_moving_decoys": n_moving_decoys,
            "n_ghosts": n_ghosts,
            "_md_positions": md_positions,
            "_md_dirs": md_dirs,
            "_md_seed": int(rng.integers(0, 2**31)),
            "_guard_positions": guard_positions,
            "_guard_dirs": [int(rng.integers(0, 4)) for _ in guard_positions],
            "_guard_seed": int(rng.integers(0, 2**31)),
            "_ghost_seed": int(rng.integers(0, 2**31)),
            "max_steps": self.get_max_steps(),
        }

    def on_env_reset(self, agent, grid, config):
        config["_guard_collision"] = False
        config["_guard_rng"] = np.random.default_rng(config.get("_guard_seed", 0))
        config["_md_rng"] = np.random.default_rng(config.get("_md_seed", 0))
        config["_ghost_rng"] = np.random.default_rng(config.get("_ghost_seed", 0))
        config["_ghost_cells"] = []  # currently placed ghost objects
        self._config = config

    def on_agent_moved(self, pos, agent, grid):
        x, y = pos
        config = getattr(self, "_config", {})
        if grid.objects[y, x] == ObjectType.NPC:
            config["_guard_collision"] = True

    def on_env_step(self, agent, grid, config, step_count):
        ax, ay = agent.position

        # Move guards
        guards = config.get("_guard_positions", [])
        dirs = config.get("_guard_dirs", [])
        rng = config.get("_guard_rng")
        if guards and rng is not None:
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
                    and grid.objects[ny, nx] not in (ObjectType.GOAL, ObjectType.TARGET)
                ):
                    new_g.append((nx, ny))
                else:
                    d = int(rng.integers(0, 4))
                    new_g.append((gx, gy))
                new_d.append(d)
                if new_g[-1] == (ax, ay):
                    config["_guard_collision"] = True
            config["_guard_positions"] = new_g
            config["_guard_dirs"] = new_d
            for gx, gy in new_g:
                if grid.terrain[gy, gx] == CellType.EMPTY:
                    grid.objects[gy, gx] = ObjectType.NPC

        # Move decoys
        md_pos = config.get("_md_positions", [])
        md_dirs = config.get("_md_dirs", [])
        md_rng = config.get("_md_rng")
        if md_pos and md_rng is not None:
            for mx, my in md_pos:
                if grid.objects[my, mx] == ObjectType.TARGET:
                    grid.objects[my, mx] = ObjectType.NONE
            new_m, new_md = [], []
            for i, (mx, my) in enumerate(md_pos):
                d = md_dirs[i]
                ddx, ddy = self._DIRS[d]
                nx, ny = mx + ddx, my + ddy
                if (
                    0 < nx < grid.width - 1
                    and 0 < ny < grid.height - 1
                    and grid.terrain[ny, nx] == CellType.EMPTY
                    and grid.objects[ny, nx] == ObjectType.NONE
                ):
                    new_m.append((nx, ny))
                else:
                    d = int(md_rng.integers(0, 4))
                    new_m.append((mx, my))
                new_md.append(d)
            config["_md_positions"] = new_m
            config["_md_dirs"] = new_md
            for mx, my in new_m:
                if grid.objects[my, mx] == ObjectType.NONE:
                    grid.objects[my, mx] = ObjectType.TARGET

        # Ghost objects: remove old, place new random ones
        ghost_rng = config.get("_ghost_rng")
        n_ghosts = config.get("n_ghosts", 0)
        old_ghosts = config.get("_ghost_cells", [])
        if ghost_rng is not None and n_ghosts > 0:
            # Remove previous ghosts
            for gx, gy in old_ghosts:
                obj = grid.objects[gy, gx]
                if obj in self._GHOST_TYPES:
                    grid.objects[gy, gx] = ObjectType.NONE

            # Place new ghosts on random empty cells
            new_ghosts = []
            empties = [
                (x, y)
                for y in range(1, grid.height - 1)
                for x in range(1, grid.width - 1)
                if (
                    grid.terrain[y, x] == CellType.EMPTY
                    and grid.objects[y, x] == ObjectType.NONE
                    and (x, y) != (ax, ay)
                )
            ]
            if empties:
                n_place = min(n_ghosts, len(empties))
                idxs = ghost_rng.choice(len(empties), size=n_place, replace=False)
                for idx in idxs:
                    gx, gy = empties[idx]
                    ghost_type = self._GHOST_TYPES[
                        int(ghost_rng.integers(0, len(self._GHOST_TYPES)))
                    ]
                    grid.objects[gy, gx] = ghost_type
                    new_ghosts.append((gx, gy))
            config["_ghost_cells"] = new_ghosts

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        config = new_state.get("config", {})
        goal = config.get("goal_positions", [None])[0]
        if goal and "agent" in new_state:
            ax, ay = new_state["agent"].position
            ox, oy = old_state.get("agent_position", new_state["agent"].position)
            reward += 0.05 * (
                abs(ox - goal[0]) + abs(oy - goal[1]) - abs(ax - goal[0]) - abs(ay - goal[1])
            )
        if self.check_success(new_state):
            reward += 1.0
        return reward

    def check_success(self, state):
        config = state.get("config", {})
        if config.get("_guard_collision", False):
            return False
        if "grid" not in state or "agent" not in state:
            return False
        x, y = state["agent"].position
        return bool(state["grid"].objects[y, x] == ObjectType.GOAL)

    def check_done(self, state):
        config = state.get("config", {})
        if config.get("_guard_collision", False):
            return True
        return self.check_success(state)

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0
