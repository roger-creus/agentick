"""MultiRoomEscape - Navigate through a sequence of interconnected rooms.

MECHANICS:
  - Grid is divided into rooms by internal walls
  - Each room has a single doorway to the next room
  - Agent must navigate room 1 → room 2 → ... → goal in final room
  - Doorway positions are randomized per seed
  - Tests spatial planning across connected subspaces
"""

import numpy as np
from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task

_MRE_DIRS = [(0,-1),(0,1),(-1,0),(1,0)]


@register_task("MultiRoomEscape-v0", tags=["skill_composition", "long_horizon"])
class MultiRoomEscapeTask(TaskSpec):
    """Navigate through multiple rooms to reach the goal."""

    name = "MultiRoomEscape-v0"
    description = "Navigate through connected rooms to escape"
    capability_tags = ["skill_composition", "long_horizon"]

    difficulty_configs = {
        "easy":   DifficultyConfig(name="easy",   grid_size=11, max_steps=100, params={"n_rooms": 2, "n_guards": 0, "dark": False}),
        "medium": DifficultyConfig(name="medium",  grid_size=15, max_steps=200, params={"n_rooms": 3, "n_guards": 1, "dark": False}),
        "hard":   DifficultyConfig(name="hard",    grid_size=19, max_steps=350, params={"n_rooms": 4, "n_guards": 2, "dark": True}),
        "expert": DifficultyConfig(name="expert",  grid_size=23, max_steps=550, params={"n_rooms": 5, "n_guards": 3, "dark": True}),
    }

    _DIRS = _MRE_DIRS

    def generate(self, seed):
        rng      = np.random.default_rng(seed)
        size     = self.difficulty_config.grid_size
        n_rooms  = self.difficulty_config.params.get("n_rooms", 2)
        n_guards = self.difficulty_config.params.get("n_guards", 0)
        dark     = self.difficulty_config.params.get("dark", False)

        grid = Grid(size, size)
        grid.terrain[0, :]  = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0]  = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        room_width = (size - 2) // n_rooms
        divider_cols = [1 + room_width * i for i in range(1, n_rooms)]

        for col in divider_cols:
            if col < size - 1:
                gap_row = int(rng.integers(1, size - 1))
                for row in range(1, size - 1):
                    if row != gap_row:
                        grid.terrain[row, col] = CellType.WALL

        agent_pos = (1, 1)
        last_room_start = divider_cols[-1] + 1 if divider_cols else 1
        goal_candidates = [
            (x, y) for x in range(last_room_start, size - 1)
                    for y in range(1, size - 1)
            if grid.terrain[y, x] == CellType.EMPTY and (x, y) != agent_pos
        ]
        if goal_candidates:
            goal_pos = max(goal_candidates,
                           key=lambda p: abs(p[0]-agent_pos[0])+abs(p[1]-agent_pos[1]))
        else:
            goal_pos = (size - 2, size - 2)

        reachable = grid.flood_fill(agent_pos)
        if goal_pos not in reachable:
            grid = Grid(size, size)
            grid.terrain[0, :] = CellType.WALL; grid.terrain[-1, :] = CellType.WALL
            grid.terrain[:, 0] = CellType.WALL; grid.terrain[:, -1] = CellType.WALL
            goal_pos = (size - 2, size - 2)
            reachable = grid.flood_fill(agent_pos)

        goal_x, goal_y = goal_pos
        grid.objects[goal_y, goal_x] = ObjectType.GOAL

        # Guards in intermediate rooms (not first or last room)
        used = {agent_pos, goal_pos}
        guard_positions = []
        if n_guards > 0:
            mid_room_cells = [p for p in reachable if p not in used
                              and p[0] > 1 and p[0] < last_room_start
                              and abs(p[0]-agent_pos[0])+abs(p[1]-agent_pos[1]) > 2]
            if not mid_room_cells:
                mid_room_cells = [p for p in reachable if p not in used
                                  and abs(p[0]-agent_pos[0])+abs(p[1]-agent_pos[1]) > 2]
            rng.shuffle(mid_room_cells)
            guard_positions = mid_room_cells[:n_guards]
            for gx, gy in guard_positions:
                grid.objects[gy, gx] = ObjectType.NPC
                used.add((gx, gy))

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [(goal_x, goal_y)],
            "dark": dark,
            "_guard_positions": guard_positions,
            "_guard_dirs": [int(rng.integers(0, 4)) for _ in guard_positions],
            "_guard_seed": int(rng.integers(0, 2**31)),
            "max_steps": self.get_max_steps(),
        }

    def on_env_reset(self, agent, grid, config):
        config["_guard_collision"] = False
        config["_guard_rng"] = np.random.default_rng(config.get("_guard_seed", 0))
        self._config = config  # cache for on_agent_moved

    def on_agent_moved(self, pos, agent, grid):
        x, y = pos
        config = getattr(self, "_config", {})
        if grid.objects[y, x] == ObjectType.NPC:
            config["_guard_collision"] = True

    def on_env_step(self, agent, grid, config, step_count):
        guards = config.get("_guard_positions", [])
        dirs   = config.get("_guard_dirs", [])
        rng    = config.get("_guard_rng")
        ax, ay = agent.position
        if not guards or rng is None:
            return
        for gx, gy in guards:
            if grid.objects[gy, gx] == ObjectType.NPC:
                grid.objects[gy, gx] = ObjectType.NONE
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
            if new_g[-1] == (ax, ay):
                config["_guard_collision"] = True
        config["_guard_positions"] = new_g
        config["_guard_dirs"] = new_d
        for gx, gy in new_g:
            if grid.terrain[gy, gx] == CellType.EMPTY:
                grid.objects[gy, gx] = ObjectType.NPC

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        if self.check_success(new_state):
            reward += 1.0
        elif "agent" in new_state and "config" in new_state:
            goal = new_state["config"].get("goal_positions", [None])[0]
            if goal:
                ax, ay = new_state["agent"].position
                ox, oy = old_state.get("agent_position", (ax, ay))
                reward += 0.05 * (abs(ox-goal[0]) + abs(oy-goal[1])
                                  - abs(ax-goal[0]) - abs(ay-goal[1]))
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

    def get_optimal_return(self, difficulty=None): return 1.0
    def get_random_baseline(self, difficulty=None): return 0.0
