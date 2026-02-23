"""KeyDoorPuzzle - Find color-coded keys to open matching doors.

CREATIVE DIFFICULTY AXES:
  - easy:   1 key, 1 door, linear rooms, color-coded (gold)
  - medium: 2 keys, 2 doors, linear rooms, color-coded (gold, red)
  - hard:   3 keys + 3 doors in CHAIN, hub-and-spoke layout,
            color-coded (gold, red, blue), 1 guard
  - expert: 4-key chain, hub-and-spoke, color-coded, 2 guards

Color model: grid.metadata[y,x] stores color index on KEY and DOOR cells:
  0 = gold, 1 = red, 2 = blue (extensible)

Hub-and-spoke layout (hard/expert): Central hub connects to branch rooms via
colored doors. Key for door N is found behind door N-1, forcing back-and-forth
traversal through the hub.

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
    """Find color-coded key(s), unlock matching door(s), reach goal."""

    name = "KeyDoorPuzzle-v0"
    description = "Find color-coded key(s) to open matching door(s) in sequence"
    capability_tags = ["memory", "sequential_reasoning"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=7,
            max_steps=100,
            params={"n_keys": 1, "n_guards": 0, "chain": False},
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=10,
            max_steps=180,
            params={"n_keys": 2, "n_guards": 0, "chain": False},
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=13,
            max_steps=300,
            params={"n_keys": 3, "n_guards": 1, "chain": True},
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=15,
            max_steps=450,
            params={"n_keys": 4, "n_guards": 2, "chain": True},
        ),
    }

    _DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    # Color names for reference: 0=gold, 1=red, 2=blue, 3=gold (wraps)
    _COLORS = ["gold", "red", "blue"]

    def _generate_linear(self, rng, grid, size, n_keys, agent_pos):
        """Generate linear room layout (easy/medium)."""
        room_width = max(2, (size - 2) // (n_keys + 1))
        wall_cols = []
        for i in range(1, n_keys + 1):
            wc = 1 + i * room_width
            if wc < size - 1:
                wall_cols.append(wc)
                for row in range(1, size - 1):
                    grid.terrain[row, wc] = CellType.WALL

        # Place doors in wall columns at random rows, with color metadata
        door_positions = []
        for idx, wc in enumerate(wall_cols):
            door_row = int(rng.integers(1, size - 1))
            grid.terrain[door_row, wc] = CellType.EMPTY
            grid.objects[door_row, wc] = ObjectType.DOOR
            color = idx % len(self._COLORS)
            grid.metadata[door_row, wc] = color
            door_positions.append((wc, door_row))

        # Key i is in room i (to the LEFT of door i) with matching color
        key_positions = []
        for i, wc in enumerate(wall_cols):
            room_left = 1 if i == 0 else wall_cols[i - 1] + 1
            room_right = wc - 1
            if room_left > room_right:
                room_left = max(1, wc - 2)
            room_cells = [
                (x, y)
                for x in range(room_left, room_right + 1)
                for y in range(1, size - 1)
                if grid.terrain[y, x] == CellType.EMPTY and (x, y) != agent_pos
            ]
            if room_cells:
                kp = room_cells[int(rng.integers(len(room_cells)))]
                key_positions.append(kp)
                grid.objects[kp[1], kp[0]] = ObjectType.KEY
                color = i % len(self._COLORS)
                grid.metadata[kp[1], kp[0]] = color

        # Goal: rightmost room
        rightmost = wall_cols[-1] + 1 if wall_cols else 1
        goal_cells = [
            (x, y)
            for x in range(rightmost, size - 1)
            for y in range(1, size - 1)
            if grid.terrain[y, x] == CellType.EMPTY
        ]
        if not goal_cells:
            goal_cells = [(size - 2, size - 2)]
        goal_pos = goal_cells[int(rng.integers(len(goal_cells)))]
        grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.GOAL

        return door_positions, key_positions, goal_pos

    def _generate_hub_spoke(self, rng, grid, size, n_keys, agent_pos):
        """Generate hub-and-spoke layout (hard/expert).

        Central hub in the middle, branch rooms connected by colored doors.
        Key for door N is behind door N-1, forcing backtracking through the hub.
        """
        # Hub area: center of the grid
        hub_left = size // 4
        hub_right = size - size // 4 - 1
        hub_top = size // 4
        hub_bot = size - size // 4 - 1

        # Ensure hub is at least 3x3
        if hub_right - hub_left < 3:
            hub_left = max(1, size // 2 - 2)
            hub_right = min(size - 2, size // 2 + 2)
        if hub_bot - hub_top < 3:
            hub_top = max(1, size // 2 - 2)
            hub_bot = min(size - 2, size // 2 + 2)

        # Build walls around hub
        for x in range(hub_left, hub_right + 1):
            grid.terrain[hub_top, x] = CellType.WALL
            grid.terrain[hub_bot, x] = CellType.WALL
        for y in range(hub_top, hub_bot + 1):
            grid.terrain[y, hub_left] = CellType.WALL
            grid.terrain[y, hub_right] = CellType.WALL

        # Clear hub interior
        for y in range(hub_top + 1, hub_bot):
            for x in range(hub_left + 1, hub_right):
                grid.terrain[y, x] = CellType.EMPTY

        # Branch directions: north, east, south, west
        branch_sides = [
            ("north", hub_left + (hub_right - hub_left) // 2, hub_top, 0, -1),
            ("east", hub_right, hub_top + (hub_bot - hub_top) // 2, 1, 0),
            ("south", hub_left + (hub_right - hub_left) // 2, hub_bot, 0, 1),
            ("west", hub_left, hub_top + (hub_bot - hub_top) // 2, -1, 0),
        ]

        rng.shuffle(branch_sides)
        # We have 4 branches max. Use min(n_keys, 4) for door branches.
        # Goal goes in the last branch if we have room, otherwise in the
        # last key's branch room.
        n_door_branches = min(n_keys, len(branch_sides))
        branches = branch_sides[:n_door_branches]
        # Determine if we have a separate goal branch
        has_goal_branch = n_door_branches < len(branch_sides)
        if has_goal_branch:
            goal_branch = branch_sides[n_door_branches]

        door_positions = []
        key_positions = []
        # Track carved room cells per branch for key placement
        branch_room_cells: list[list[tuple[int, int]]] = []

        # Place doors on hub walls, carve branch rooms
        for idx, (side, wx, wy, dx, dy) in enumerate(branches):
            # Carve door in hub wall
            grid.terrain[wy, wx] = CellType.EMPTY
            grid.objects[wy, wx] = ObjectType.DOOR
            color = idx % len(self._COLORS)
            grid.metadata[wy, wx] = color
            door_positions.append((wx, wy))

            # Carve a small branch room beyond the door
            room_cells = []
            for dist in range(1, 4):
                rx, ry = wx + dx * dist, wy + dy * dist
                if 1 <= rx < size - 1 and 1 <= ry < size - 1:
                    grid.terrain[ry, rx] = CellType.EMPTY
                    grid.objects[ry, rx] = ObjectType.NONE
                    room_cells.append((rx, ry))
                # Also carve width
                for wd in [-1, 1]:
                    if dx == 0:  # vertical branch
                        sx, sy = rx + wd, ry
                    else:  # horizontal branch
                        sx, sy = rx, ry + wd
                    if 1 <= sx < size - 1 and 1 <= sy < size - 1:
                        grid.terrain[sy, sx] = CellType.EMPTY
                        grid.objects[sy, sx] = ObjectType.NONE
                        room_cells.append((sx, sy))
            branch_room_cells.append(room_cells)

            # Place key for this door
            # Key 0 goes in the hub (accessible without any door)
            # Key i (i>0) goes in branch i-1 (behind door i-1), creating chain
            if idx == 0:
                # First key: place in hub
                hub_cells = [
                    (x, y)
                    for x in range(hub_left + 1, hub_right)
                    for y in range(hub_top + 1, hub_bot)
                    if grid.terrain[y, x] == CellType.EMPTY
                    and grid.objects[y, x] == ObjectType.NONE
                    and (x, y) != agent_pos
                ]
                if hub_cells:
                    kp = hub_cells[int(rng.integers(len(hub_cells)))]
                    key_positions.append(kp)
                    grid.objects[kp[1], kp[0]] = ObjectType.KEY
                    grid.metadata[kp[1], kp[0]] = color
            else:
                # Key i: place in branch i-1 (behind door i-1)
                prev_room = [
                    c for c in branch_room_cells[idx - 1]
                    if grid.objects[c[1], c[0]] == ObjectType.NONE
                ]
                if prev_room:
                    kp = prev_room[int(rng.integers(len(prev_room)))]
                elif room_cells:
                    kp = room_cells[int(rng.integers(len(room_cells)))]
                else:
                    kp = (wx + dx, wy + dy)
                key_positions.append(kp)
                grid.objects[kp[1], kp[0]] = ObjectType.KEY
                grid.metadata[kp[1], kp[0]] = color

        # Place goal
        if has_goal_branch:
            # Separate goal branch
            _, gwx, gwy, gdx, gdy = goal_branch
            grid.terrain[gwy, gwx] = CellType.EMPTY
            goal_room = []
            for dist in range(1, 4):
                rx, ry = gwx + gdx * dist, gwy + gdy * dist
                if 1 <= rx < size - 1 and 1 <= ry < size - 1:
                    grid.terrain[ry, rx] = CellType.EMPTY
                    goal_room.append((rx, ry))
                for wd in [-1, 1]:
                    if gdx == 0:
                        sx, sy = rx + wd, ry
                    else:
                        sx, sy = rx, ry + wd
                    if 1 <= sx < size - 1 and 1 <= sy < size - 1:
                        grid.terrain[sy, sx] = CellType.EMPTY
                        goal_room.append((sx, sy))
            if goal_room:
                goal_pos = goal_room[int(rng.integers(len(goal_room)))]
            else:
                goal_pos = (gwx + gdx, gwy + gdy)
                if 1 <= goal_pos[0] < size - 1 and 1 <= goal_pos[1] < size - 1:
                    grid.terrain[goal_pos[1], goal_pos[0]] = CellType.EMPTY
        else:
            # No separate goal branch — place goal in last key's branch room
            last_room = [
                c for c in branch_room_cells[-1]
                if grid.objects[c[1], c[0]] == ObjectType.NONE
            ]
            if last_room:
                goal_pos = last_room[int(rng.integers(len(last_room)))]
            else:
                # Fallback: anywhere empty in grid
                all_empty = [
                    (x, y)
                    for x in range(1, size - 1)
                    for y in range(1, size - 1)
                    if grid.terrain[y, x] == CellType.EMPTY
                    and grid.objects[y, x] == ObjectType.NONE
                    and (x, y) != agent_pos
                ]
                goal_pos = all_empty[int(rng.integers(len(all_empty)))] if all_empty else (size - 2, size - 2)
        grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.GOAL

        # Ensure agent start is in hub
        hub_cells = [
            (x, y)
            for x in range(hub_left + 1, hub_right)
            for y in range(hub_top + 1, hub_bot)
            if grid.terrain[y, x] == CellType.EMPTY
            and grid.objects[y, x] == ObjectType.NONE
        ]
        if hub_cells and agent_pos not in hub_cells:
            agent_pos = hub_cells[int(rng.integers(len(hub_cells)))]

        return door_positions, key_positions, goal_pos, agent_pos

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        p = self.difficulty_config.params
        n_keys = p.get("n_keys", 1)
        n_guards = p.get("n_guards", 0)
        chain = p.get("chain", False)

        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        # Agent start
        agent_pos = (1, int(rng.integers(1, size - 1)))

        if chain:
            # Hub-and-spoke layout for hard/expert
            door_positions, key_positions, goal_pos, agent_pos = (
                self._generate_hub_spoke(rng, grid, size, n_keys, agent_pos)
            )
        else:
            # Linear rooms for easy/medium
            door_positions, key_positions, goal_pos = self._generate_linear(
                rng, grid, size, n_keys, agent_pos
            )

        # Guard start positions: empty areas
        all_empty = [
            (x, y)
            for x in range(1, size - 1)
            for y in range(1, size - 1)
            if grid.terrain[y, x] == CellType.EMPTY
            and grid.objects[y, x] == ObjectType.NONE
            and (x, y) != agent_pos
        ]
        rng.shuffle(all_empty)
        guard_positions = all_empty[:n_guards]

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [goal_pos],
            "key_pos": key_positions[0] if key_positions else None,
            "key_positions": key_positions,
            "door_pos": door_positions[0] if door_positions else None,
            "door_positions": door_positions,
            "n_keys": n_keys,
            "_guard_positions": guard_positions,
            "_guard_dirs": [int(rng.integers(0, 4)) for _ in guard_positions],
            "_guard_seed": int(rng.integers(0, 2**31)),
            "max_steps": self.get_max_steps(),
        }

    def on_env_reset(self, agent, grid, config):
        agent.inventory.clear()
        self._had_keys = 0
        self._doors_opened = 0
        self._config = config
        config["_door_open"] = False
        config["_guard_collision"] = False
        config["_guard_rng"] = np.random.default_rng(config.get("_guard_seed", 0))
        # Redraw guards with default direction
        for gx, gy in config.get("_guard_positions", []):
            if grid.terrain[gy, gx] == CellType.EMPTY:
                grid.objects[gy, gx] = ObjectType.NPC
                grid.metadata[gy, gx] = 2  # default facing down

    def can_agent_enter(self, pos, agent, grid) -> bool:
        x, y = pos
        if grid.objects[y, x] == ObjectType.DOOR:
            door_meta = int(grid.metadata[y, x])
            # Open doors (meta >= 10) are passable
            if door_meta >= 10:
                return True
            door_color = door_meta
            matching = next(
                (
                    e
                    for e in agent.inventory
                    if e.entity_type == "key"
                    and e.properties.get("color") == door_color
                ),
                None,
            )
            if matching:
                agent.inventory.remove(matching)
                # Keep door object, mark as open via meta + 10
                grid.metadata[y, x] = door_color + 10
                config = getattr(self, "_config", {})
                config["_door_open"] = True
                return True
            return False
        return True

    def on_agent_moved(self, pos, agent, grid):
        x, y = pos
        obj = grid.objects[y, x]
        if obj == ObjectType.KEY:
            color = int(grid.metadata[y, x])
            grid.objects[y, x] = ObjectType.NONE
            grid.metadata[y, x] = 0
            agent.inventory.append(
                Entity(
                    id=f"key_{x}_{y}",
                    entity_type="key",
                    position=pos,
                    properties={"color": color},
                )
            )
        elif obj == ObjectType.NPC:
            config = getattr(self, "_config", {})
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
                grid.metadata[gy, gx] = 0
        new_g, new_d = [], []
        for i, (gx, gy) in enumerate(guards):
            d = dirs[i]
            dx, dy = self._DIRS[d]
            nx, ny = gx + dx, gy + dy
            if (
                0 < nx < grid.width - 1
                and 0 < ny < grid.height - 1
                and grid.terrain[ny, nx] == CellType.EMPTY
                and grid.objects[ny, nx] == ObjectType.NONE
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
        for i, (gx, gy) in enumerate(new_g):
            if grid.terrain[gy, gx] == CellType.EMPTY:
                grid.objects[gy, gx] = ObjectType.NPC
                # Direction metadata from old -> new position
                old_gx, old_gy = guards[i] if i < len(guards) else (gx, gy)
                ddx, ddy = gx - old_gx, gy - old_gy
                if ddx > 0:
                    grid.metadata[gy, gx] = 1  # right
                elif ddx < 0:
                    grid.metadata[gy, gx] = 3  # left
                elif ddy < 0:
                    grid.metadata[gy, gx] = 0  # up
                elif ddy > 0:
                    grid.metadata[gy, gx] = 2  # down
                else:
                    grid.metadata[gy, gx] = 2  # default down

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
            reward += 0.3
            self._doors_opened += 1
        goal = config.get("goal_positions", [None])[0]
        if goal and agent:
            ax, ay = agent.position
            ox, oy = old_state.get("agent_position", (ax, ay))
            reward += 0.05 * (
                (abs(ox - goal[0]) + abs(oy - goal[1]))
                - (abs(ax - goal[0]) + abs(ay - goal[1]))
            )
        if self.check_success(new_state):
            reward += 1.0
        return reward

    def check_done(self, state):
        if state.get("config", {}).get("_guard_collision", False):
            return True
        return self.check_success(state)

    def check_success(self, state):
        if state.get("config", {}).get("_guard_collision", False):
            return False
        if "grid" not in state or "agent" not in state:
            return False
        x, y = state["agent"].position
        return bool(state["grid"].objects[y, x] == ObjectType.GOAL)

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0
