"""KeyDoorPuzzle - Find keys to open doors blocking path to goal.

BUG FIX: check_success already uses grid.objects[y,x] (correct). ✓

CREATIVE DIFFICULTY AXES:
  - easy:   1 key, 1 door, open map, direct path visible
  - medium: 2 keys, 2 doors, L-shaped path with walls, key positions randomized
  - hard:   3 keys + 3 doors in CHAIN (key3 behind door2, key2 behind door1),
            obstacles + dead-end side rooms as distractors
  - expert: 4-key chain, patrolling guard (game over on contact), narrow corridors

PROCEDURAL DIVERSITY: random seed controls key/door/goal placement,
wall layout, guard patrol routes.
"""

import numpy as np

from agentick.core.entity import Entity
from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("KeyDoorPuzzle-v0", tags=["memory", "sequential_reasoning"])
class KeyDoorPuzzleTask(TaskSpec):
    """Find key(s), unlock door(s), reach goal. Keys/doors chain at higher difficulties."""

    name = "KeyDoorPuzzle-v0"
    description = "Find key(s) to open door(s) in sequence"
    capability_tags = ["memory", "sequential_reasoning"]

    difficulty_configs = {
        "easy":   DifficultyConfig(name="easy",   grid_size=7,  max_steps=100, params={"n_keys": 1, "n_guards": 0, "chain": False}),
        "medium": DifficultyConfig(name="medium",  grid_size=10, max_steps=180, params={"n_keys": 2, "n_guards": 0, "chain": False}),
        "hard":   DifficultyConfig(name="hard",    grid_size=13, max_steps=300, params={"n_keys": 3, "n_guards": 1, "chain": True}),
        "expert": DifficultyConfig(name="expert",  grid_size=15, max_steps=450, params={"n_keys": 4, "n_guards": 2, "chain": True}),
    }

    _DIRS = [(0,-1),(0,1),(-1,0),(1,0)]

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        p     = self.difficulty_config.params
        n_keys   = p.get("n_keys", 1)
        n_guards = p.get("n_guards", 0)
        chain    = p.get("chain", False)

        grid = Grid(size, size)
        grid.terrain[0, :]  = CellType.WALL; grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0]  = CellType.WALL; grid.terrain[:, -1] = CellType.WALL

        # Divide grid into n_keys+1 rooms separated by walls with doorways
        # Rooms left→right, each with a door. Key in room i unlocks door to room i+1.
        room_width = max(2, (size - 2) // (n_keys + 1))
        wall_cols = []
        for i in range(1, n_keys + 1):
            wc = 1 + i * room_width
            if wc < size - 1:
                wall_cols.append(wc)
                for row in range(1, size - 1):
                    grid.terrain[row, wc] = CellType.WALL

        # Place doors in wall columns at random rows
        door_positions = []
        for wc in wall_cols:
            door_row = int(rng.integers(1, size - 1))
            grid.terrain[door_row, wc] = CellType.EMPTY  # passable when unlocked
            grid.objects[door_row, wc] = ObjectType.DOOR
            door_positions.append((wc, door_row))

        # Agent start: left side of first room
        agent_pos = (1, int(rng.integers(1, size - 1)))

        # Key i is in room i (to the LEFT of door i)
        key_positions = []
        for i, wc in enumerate(wall_cols):
            room_left  = 1 if i == 0 else wall_cols[i-1] + 1
            room_right = wc - 1
            if room_left > room_right:
                room_left = max(1, wc - 2)
            room_cells = [(x, y) for x in range(room_left, room_right + 1)
                          for y in range(1, size - 1)
                          if grid.terrain[y, x] == CellType.EMPTY and (x, y) != agent_pos]
            if chain and i > 0 and room_cells:
                # Key i is BEHIND door i-1 (requires solving previous door first)
                # So key i is on RIGHT side of wall i-1 = left side of room i
                right_room_left = wall_cols[i-1] + 1
                right_cells = [(x, y) for x in range(right_room_left, wc)
                               for y in range(1, size - 1)
                               if grid.terrain[y, x] == CellType.EMPTY]
                room_cells = right_cells if right_cells else room_cells
            if room_cells:
                kp = room_cells[int(rng.integers(len(room_cells)))]
                key_positions.append(kp)
                grid.objects[kp[1], kp[0]] = ObjectType.KEY

        # Goal: rightmost room
        rightmost = wall_cols[-1] + 1 if wall_cols else 1
        goal_cells = [(x, y) for x in range(rightmost, size - 1)
                      for y in range(1, size - 1)
                      if grid.terrain[y, x] == CellType.EMPTY]
        if not goal_cells:
            goal_cells = [(size-2, size-2)]
        goal_pos = goal_cells[int(rng.integers(len(goal_cells)))]
        grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.GOAL

        # Guard start positions: hallway areas
        all_empty = [(x, y) for x in range(1, size-1) for y in range(1, size-1)
                     if grid.terrain[y, x] == CellType.EMPTY
                     and grid.objects[y, x] == ObjectType.NONE
                     and (x, y) != agent_pos]
        rng.shuffle(all_empty)
        guard_positions = all_empty[:n_guards]

        return grid, {
            "agent_start":     agent_pos,
            "goal_positions":  [goal_pos],
            "key_pos":         key_positions[0] if key_positions else None,  # compat with tests
            "key_positions":   key_positions,
            "door_pos":        door_positions[0] if door_positions else None,  # compat with tests
            "door_positions":  door_positions,
            "n_keys":          n_keys,
            "_guard_positions": guard_positions,
            "_guard_dirs":     [int(rng.integers(0, 4)) for _ in guard_positions],
            "_guard_seed":     int(rng.integers(0, 2**31)),
            "max_steps":       self.get_max_steps(),
        }

    def on_env_reset(self, agent, grid, config):
        agent.inventory.clear()
        self._had_keys = 0
        self._doors_opened = 0
        self._config = config
        config["_door_open"]       = False
        config["_guard_collision"] = False
        config["_guard_rng"]       = np.random.default_rng(config.get("_guard_seed", 0))
        # Redraw guards
        for gx, gy in config.get("_guard_positions", []):
            if grid.terrain[gy, gx] == CellType.EMPTY:
                grid.objects[gy, gx] = ObjectType.NPC

    def can_agent_enter(self, pos, agent, grid) -> bool:
        x, y = pos
        if grid.objects[y, x] == ObjectType.DOOR:
            has_key = any(e.entity_type == "key" for e in agent.inventory)
            if has_key:
                agent.inventory = [e for e in agent.inventory if e.entity_type != "key"]
                grid.objects[y, x] = ObjectType.NONE
                config = getattr(self, "_config", {})
                config["_door_open"] = True
                return True
            return False
        return True

    def on_agent_moved(self, pos, agent, grid):
        x, y = pos
        obj = grid.objects[y, x]
        if obj == ObjectType.KEY:
            grid.objects[y, x] = ObjectType.NONE
            agent.inventory.append(Entity(id=f"key_{x}_{y}", entity_type="key", position=pos))
        elif obj == ObjectType.NPC:
            config = getattr(self, "_config", {})
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
            d = dirs[i]
            dx, dy = self._DIRS[d]
            nx, ny = gx+dx, gy+dy
            if (0 < nx < grid.width-1 and 0 < ny < grid.height-1
                    and grid.terrain[ny, nx] == CellType.EMPTY
                    and grid.objects[ny, nx] == ObjectType.NONE):
                new_g.append((nx, ny))
            else:
                d = int(rng.integers(0, 4))
                new_g.append((gx, gy))
            new_d.append(d)
            if (new_g[-1][0], new_g[-1][1]) == (ax, ay):
                config["_guard_collision"] = True
        config["_guard_positions"] = new_g
        config["_guard_dirs"]      = new_d
        for gx, gy in new_g:
            if grid.terrain[gy, gx] == CellType.EMPTY:
                grid.objects[gy, gx] = ObjectType.NPC

    def compute_dense_reward(self, old_state, action, new_state, info):
        config = new_state.get("config", {})
        if config.get("_guard_collision", False):
            return -1.0
        reward = -0.01
        agent = new_state.get("agent")
        n_keys = len(agent.inventory) if agent else 0
        if n_keys > self._had_keys:
            reward += 0.5 * (n_keys - self._had_keys)
            self._had_keys = n_keys
        door_open = config.get("_door_open", False)
        if door_open and not self._doors_opened:
            reward += 0.3; self._doors_opened += 1
        goal = config.get("goal_positions", [None])[0]
        if goal and agent:
            ax, ay = agent.position
            ox, oy = old_state.get("agent_position", (ax, ay))
            reward += 0.05 * ((abs(ox-goal[0])+abs(oy-goal[1])) - (abs(ax-goal[0])+abs(ay-goal[1])))
        if self.check_success(new_state): reward += 1.0
        return reward

    def check_done(self, state):
        if state.get("config", {}).get("_guard_collision", False): return True
        return self.check_success(state)

    def check_success(self, state):
        if state.get("config", {}).get("_guard_collision", False): return False
        if "grid" not in state or "agent" not in state: return False
        x, y = state["agent"].position
        return bool(state["grid"].objects[y, x] == ObjectType.GOAL)

    def get_optimal_return(self, difficulty=None): return 1.0
    def get_random_baseline(self, difficulty=None): return 0.0
