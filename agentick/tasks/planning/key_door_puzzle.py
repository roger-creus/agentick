"""KeyDoorPuzzle - Nonlinear key chain where goal is behind ALL doors.

CREATIVE DIFFICULTY AXES:
  - easy:   1 key, 1 door, goal behind the single door.
  - medium: 2 keys, 2 doors in series, key_B behind door_A forcing backtracking.
  - hard:   3 keys, 3 doors in series, nonlinear key chain.
  - expert: 4 keys, 4 doors in series, complex nonlinear dependencies.

Color model: grid.metadata[y,x] stores color index on KEY and DOOR cells:
  0 = gold, 1 = red, 2 = blue, 3 = green (extensible)

Layout: Rooms connected by 1-cell-wide corridors. Each corridor has exactly
one door. Because corridors are 1 cell wide, each door blocks all traffic.
The rooms form a linear chain: hub -> room_0 -> room_1 -> ... -> goal_room,
with each transition gated by a door. Side branch rooms connect to main rooms
and hold keys for later doors, creating a nonlinear dependency chain.

PROCEDURAL DIVERSITY: random seed controls corridor direction, room sizes,
branch placement, key/door positions.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from agentick.core.entity import Entity
from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("KeyDoorPuzzle-v0", tags=["memory", "sequential_reasoning"])
class KeyDoorPuzzleTask(TaskSpec):
    """Collect color-coded keys in nonlinear order, unlock all doors, reach goal."""

    name = "KeyDoorPuzzle-v0"
    description = "Find color-coded key(s) to open matching door(s) with nonlinear key chain"
    capability_tags = ["memory", "sequential_reasoning"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=9,
            max_steps=100,
            params={"n_keys": 1},
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=11,
            max_steps=200,
            params={"n_keys": 2},
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=15,
            max_steps=500,
            params={"n_keys": 3},
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=19,
            max_steps=800,
            params={"n_keys": 4},
        ),
    }

    _DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    # Color names for reference: 0=gold, 1=red, 2=blue, 3=green
    _COLORS = ["gold", "red", "blue", "green"]

    def generate(self, seed):
        """Generate with retry logic for validation failures."""
        for attempt in range(100):
            result = self._try_generate(seed + attempt)
            if result is not None:
                return result
        # Last resort: return final attempt (still validated)
        result = self._try_generate(seed + 100)
        if result is not None:
            return result
        raise RuntimeError(
            f"KeyDoorPuzzle: failed to generate solvable instance after 101 attempts "
            f"(seed={seed}, difficulty={self.difficulty})"
        )

    def _carve_rect(self, grid, x1, y1, x2, y2):
        """Carve a rectangular room. Returns list of carved cells."""
        size = grid.width
        cells = []
        for y in range(max(1, y1), min(size - 1, y2 + 1)):
            for x in range(max(1, x1), min(size - 1, x2 + 1)):
                grid.terrain[y, x] = CellType.EMPTY
                cells.append((x, y))
        return cells

    def _try_generate(self, seed, skip_validation=False):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        p = self.difficulty_config.params
        n_keys = p.get("n_keys", 1)

        grid = Grid(size, size)
        # Fill everything with walls
        for y in range(size):
            for x in range(size):
                grid.terrain[y, x] = CellType.WALL

        # ---- Compute room layout along a horizontal or vertical axis ----
        # We lay out rooms along one axis. Each room is a rectangle.
        # Between consecutive rooms is a 1-cell wall with a door in it.
        #
        # Total rooms: 1 hub + n_keys rooms (last one is goal room)
        # Plus side branches for key placement.

        n_rooms = 1 + n_keys  # hub + one room per door (last = goal room)

        # Choose axis: 0=horizontal (rooms left to right), 1=vertical (top to bottom)
        axis = int(rng.integers(0, 2))

        # Available interior space along axis
        interior = size - 2  # cells between outer walls

        # Hub needs at least 2 cells, other rooms need at least 1 cell,
        # plus 1 door cell between each pair of rooms.
        # Total minimum: 2 + (n_rooms-1)*1 + (n_rooms-1)*1 = 2*n_rooms
        min_needed = 2 * n_rooms
        if interior < min_needed:
            return None

        # Distribute space: hub gets 2, others get 1, then distribute remaining
        room_sizes = [2] + [1] * (n_rooms - 1)
        remaining = interior - min_needed
        for _ in range(remaining):
            idx = int(rng.integers(n_rooms))
            room_sizes[idx] += 1

        # Cap room sizes to keep things reasonable
        max_room = max(4, interior // n_rooms)
        room_sizes = [min(s, max_room) for s in room_sizes]

        # Compute positions along axis
        # Room i occupies [room_starts[i], room_ends[i]] along the axis
        room_starts = []
        room_ends = []
        pos = 1  # start after outer wall
        for i in range(n_rooms):
            room_starts.append(pos)
            room_ends.append(pos + room_sizes[i] - 1)
            pos += room_sizes[i]
            if i < n_rooms - 1:
                pos += 1  # wall between rooms

        # Perpendicular extent: full interior width, but we'll use a fraction
        perp_interior = size - 2
        # Room height (perpendicular): at least 3, at most perp_interior
        room_perp = max(3, min(perp_interior, 3 + (size - 9)))
        perp_start = 1 + (perp_interior - room_perp) // 2
        perp_end = perp_start + room_perp - 1

        # ---- Carve rooms ----
        all_room_cells: list[list[tuple[int, int]]] = []
        for i in range(n_rooms):
            if axis == 0:  # horizontal
                cells = self._carve_rect(
                    grid, room_starts[i], perp_start, room_ends[i], perp_end
                )
            else:  # vertical
                cells = self._carve_rect(
                    grid, perp_start, room_starts[i], perp_end, room_ends[i]
                )
            all_room_cells.append(cells)

        # ---- Place doors between rooms ----
        door_positions = []
        # The wall between room i and room i+1 is at position room_ends[i]+1
        # along the axis. We carve a single cell opening with a door.
        for i in range(n_rooms - 1):
            wall_pos = room_ends[i] + 1  # along axis

            # Door position: random perpendicular position within room extent
            # (must be within the perpendicular range of both adjacent rooms)
            door_perp = int(rng.integers(perp_start, perp_end + 1))

            if axis == 0:  # horizontal: wall is a vertical column
                dx, dy = wall_pos, door_perp
            else:  # vertical: wall is a horizontal row
                dx, dy = door_perp, wall_pos

            if 1 <= dx <= size - 2 and 1 <= dy <= size - 2:
                grid.terrain[dy, dx] = CellType.EMPTY
                grid.objects[dy, dx] = ObjectType.DOOR
                color = i % len(self._COLORS)
                grid.metadata[dy, dx] = color
                door_positions.append((dx, dy))

        if len(door_positions) != n_keys:
            return None

        # ---- Carve side branches off main rooms (for key placement) ----
        # Each non-goal room can have a branch room extending perpendicular.
        branch_rooms: list[list[tuple[int, int]]] = []
        for i in range(n_rooms - 1):  # no branch off goal room
            room_cells = all_room_cells[i]
            if not room_cells:
                branch_rooms.append([])
                continue

            # Choose a side (above or below for horizontal axis;
            # left or right for vertical axis)
            if axis == 0:
                # Branch above or below the room
                side = int(rng.integers(0, 2))
                if side == 0:  # above
                    branch_y_start = perp_start - 3
                    branch_y_end = perp_start - 1
                else:  # below
                    branch_y_start = perp_end + 1
                    branch_y_end = perp_end + 3
                # Branch x: somewhere within the room's x range
                mid_x = (room_starts[i] + room_ends[i]) // 2
                branch_x_start = mid_x
                branch_x_end = mid_x + 1

                # First carve connector from room
                conn_y = perp_start - 1 if side == 0 else perp_end + 1
                if 1 <= conn_y <= size - 2 and 1 <= mid_x <= size - 2:
                    grid.terrain[conn_y, mid_x] = CellType.EMPTY

                cells = self._carve_rect(
                    grid, branch_x_start, branch_y_start,
                    branch_x_end, branch_y_end
                )
            else:
                # Branch left or right of the room
                side = int(rng.integers(0, 2))
                if side == 0:  # left
                    branch_x_start = perp_start - 3
                    branch_x_end = perp_start - 1
                else:  # right
                    branch_x_start = perp_end + 1
                    branch_x_end = perp_end + 3
                mid_y = (room_starts[i] + room_ends[i]) // 2
                branch_y_start = mid_y
                branch_y_end = mid_y + 1

                conn_x = perp_start - 1 if side == 0 else perp_end + 1
                if 1 <= conn_x <= size - 2 and 1 <= mid_y <= size - 2:
                    grid.terrain[mid_y, conn_x] = CellType.EMPTY

                cells = self._carve_rect(
                    grid, branch_x_start, branch_y_start,
                    branch_x_end, branch_y_end
                )

            branch_rooms.append(cells)

        # ---- Place agent in hub (room 0) ----
        hub_cells = [
            c for c in all_room_cells[0]
            if grid.objects[c[1], c[0]] == ObjectType.NONE
        ]
        if not hub_cells:
            return None
        agent_pos = hub_cells[int(rng.integers(len(hub_cells)))]
        used_cells: set[tuple[int, int]] = {agent_pos}

        # ---- Place keys ----
        key_positions = []

        if n_keys == 1:
            # Key in hub (room 0), door_0 -> goal room (room 1)
            kp_candidates = [
                c for c in hub_cells
                if c not in used_cells and grid.objects[c[1], c[0]] == ObjectType.NONE
            ]
            if not kp_candidates:
                return None
            kp = kp_candidates[int(rng.integers(len(kp_candidates)))]
            grid.objects[kp[1], kp[0]] = ObjectType.KEY
            grid.metadata[kp[1], kp[0]] = 0
            key_positions.append(kp)
            used_cells.add(kp)

        elif n_keys == 2:
            # 2 keys: key 0 in hub, key 1 in room 1 (behind door 0)
            # Key 0: in hub
            kp_candidates = [
                c for c in hub_cells
                if c not in used_cells and grid.objects[c[1], c[0]] == ObjectType.NONE
            ]
            if not kp_candidates:
                return None
            kp = kp_candidates[int(rng.integers(len(kp_candidates)))]
            grid.objects[kp[1], kp[0]] = ObjectType.KEY
            grid.metadata[kp[1], kp[0]] = 0
            key_positions.append(kp)
            used_cells.add(kp)

            # Key 1: in room 1 or branch_1 (behind door 0)
            candidates = []
            for c in all_room_cells[1]:
                if (c not in used_cells
                        and grid.objects[c[1], c[0]] == ObjectType.NONE):
                    candidates.append(c)
            if 1 < len(branch_rooms):
                for c in branch_rooms[1]:
                    if (c not in used_cells
                            and grid.objects[c[1], c[0]] == ObjectType.NONE):
                        candidates.append(c)
            if not candidates:
                return None
            kp = candidates[int(rng.integers(len(candidates)))]
            grid.objects[kp[1], kp[0]] = ObjectType.KEY
            grid.metadata[kp[1], kp[0]] = 1
            key_positions.append(kp)
            used_cells.add(kp)

        else:
            # n_keys >= 3: nonlinear backtracking chain
            # Key for door 0: always in hub (room 0) — reachable without any door
            # Key for door 1: in room 1 (behind door 0) — forward
            # Key for door 2: in hub or branch_0 — BACKTRACK to room 0
            # Key for door 3: in room 1 or branch_1 — BACKTRACK to room 1
            # Pattern: key_i for i >= 2 goes in room (i % 2 == 0 ? 0 : 1)

            # Key 0: in hub
            kp_candidates = [
                c for c in hub_cells
                if c not in used_cells and grid.objects[c[1], c[0]] == ObjectType.NONE
            ]
            if not kp_candidates:
                return None
            kp = kp_candidates[int(rng.integers(len(kp_candidates)))]
            grid.objects[kp[1], kp[0]] = ObjectType.KEY
            grid.metadata[kp[1], kp[0]] = 0
            key_positions.append(kp)
            used_cells.add(kp)

            # Key 1: in room 1 (behind door 0)
            candidates = []
            for c in all_room_cells[1]:
                if (c not in used_cells
                        and grid.objects[c[1], c[0]] == ObjectType.NONE):
                    candidates.append(c)
            if 1 < len(branch_rooms):
                for c in branch_rooms[1]:
                    if (c not in used_cells
                            and grid.objects[c[1], c[0]] == ObjectType.NONE):
                        candidates.append(c)
            if not candidates:
                return None
            kp = candidates[int(rng.integers(len(candidates)))]
            grid.objects[kp[1], kp[0]] = ObjectType.KEY
            grid.metadata[kp[1], kp[0]] = 1
            key_positions.append(kp)
            used_cells.add(kp)

            # Keys 2..n_keys-1: backtracking pattern
            for door_i in range(2, n_keys):
                color = door_i % len(self._COLORS)
                # Even door index -> hub (room 0) or branch_0
                # Odd door index -> room 1 or branch_1
                target_room = 0 if door_i % 2 == 0 else 1

                candidates = []
                for c in all_room_cells[target_room]:
                    if (c not in used_cells
                            and grid.objects[c[1], c[0]] == ObjectType.NONE):
                        candidates.append(c)
                if target_room < len(branch_rooms):
                    for c in branch_rooms[target_room]:
                        if (c not in used_cells
                                and grid.objects[c[1], c[0]] == ObjectType.NONE):
                            candidates.append(c)

                if not candidates:
                    return None
                kp = candidates[int(rng.integers(len(candidates)))]
                grid.objects[kp[1], kp[0]] = ObjectType.KEY
                grid.metadata[kp[1], kp[0]] = color
                key_positions.append(kp)
                used_cells.add(kp)

        # ---- Place goal in last room ----
        goal_room = all_room_cells[-1]
        goal_free = [
            c for c in goal_room
            if c not in used_cells and grid.objects[c[1], c[0]] == ObjectType.NONE
        ]
        if not goal_free:
            return None
        goal_pos = goal_free[int(rng.integers(len(goal_free)))]
        grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.GOAL
        used_cells.add(goal_pos)

        config = {
            "agent_start": agent_pos,
            "goal_positions": [goal_pos],
            "key_pos": key_positions[0] if key_positions else None,
            "key_positions": key_positions,
            "door_pos": door_positions[0] if door_positions else None,
            "door_positions": door_positions,
            "n_keys": n_keys,
            "max_steps": self.get_max_steps(),
        }

        if skip_validation or self.validate_instance(grid, config):
            return grid, config
        return None

    def on_env_reset(self, agent, grid, config):
        agent.inventory.clear()
        self._doors_opened = 0
        self._config = config
        config["_door_open"] = False

    def can_agent_enter(self, pos, agent, grid) -> bool:
        x, y = pos
        if grid.objects[y, x] == ObjectType.DOOR:
            return int(grid.metadata[y, x]) >= 10
        return True

    def on_agent_interact(self, pos, agent, grid):
        """INTERACT on a closed door with matching key unlocks it."""
        if not grid.in_bounds(pos):
            return
        x, y = pos
        if grid.objects[y, x] != ObjectType.DOOR:
            return
        door_meta = int(grid.metadata[y, x])
        if door_meta >= 10:
            return
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
            grid.metadata[y, x] = door_color + 10
            config = getattr(self, "_config", {})
            config["_door_open"] = True

    def on_agent_moved(self, pos, agent, grid):
        x, y = pos
        obj = grid.objects[y, x]
        if obj == ObjectType.KEY:
            new_color = int(grid.metadata[y, x])
            # Enforce 1-key inventory limit: swap if already holding a key
            existing_key = next(
                (e for e in agent.inventory if e.entity_type == "key"), None
            )
            if existing_key is not None:
                # Drop existing key at this position
                old_color = existing_key.properties.get("color", 0)
                grid.objects[y, x] = ObjectType.KEY
                grid.metadata[y, x] = old_color
                agent.inventory.remove(existing_key)
            else:
                grid.objects[y, x] = ObjectType.NONE
                grid.metadata[y, x] = 0
            # Pick up new key
            agent.inventory.append(
                Entity(
                    id=f"key_{x}_{y}",
                    entity_type="key",
                    position=pos,
                    properties={"color": new_color},
                )
            )

    def compute_dense_reward(self, old_state, action, new_state, info):
        config = new_state.get("config", {})
        reward = -0.01
        # Reward for opening a door (each door opened once)
        door_open = config.get("_door_open", False)
        if door_open:
            # Count currently open doors on the grid
            grid = new_state.get("grid")
            if grid is not None:
                open_count = 0
                for dy in range(grid.height):
                    for dx in range(grid.width):
                        if (grid.objects[dy, dx] == ObjectType.DOOR
                                and int(grid.metadata[dy, dx]) >= 10):
                            open_count += 1
                if open_count > self._doors_opened:
                    reward += 0.5 * (open_count - self._doors_opened)
                    self._doors_opened = open_count
        agent = new_state.get("agent")
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
        return self.check_success(state)

    def check_success(self, state):
        if "grid" not in state or "agent" not in state:
            return False
        x, y = state["agent"].position
        return bool(state["grid"].objects[y, x] == ObjectType.GOAL)

    def validate_instance(self, grid, config):
        """Verify the key invariants with 1-key-at-a-time constraint:

        1. Goal is NOT reachable without opening ANY door
        2. Each individual door is necessary (goal unreachable if any single door
           remains closed while all others are open)
        3. Goal IS reachable when all doors are opened
        4. Solvable with 1-key limit: agent can carry only one key at a time,
           must pick up keys in a specific order to unlock all doors sequentially
        """
        agent_pos = config.get("agent_start")
        goal_pos = config.get("goal_positions", [None])[0]
        if agent_pos is None or goal_pos is None:
            return True

        door_set = set()
        for dp in config.get("door_positions", []):
            door_set.add(tuple(dp))

        # 1. Goal must NOT be reachable with all doors closed
        if self._bfs_reachable(grid, agent_pos, goal_pos, passable_doors=set()):
            return False

        # 2. Each door must be necessary
        for skip_door in door_set:
            passable = door_set - {skip_door}
            if self._bfs_reachable(grid, agent_pos, goal_pos,
                                   passable_doors=passable):
                return False

        # 3. Goal must be reachable when ALL doors are open
        if not self._bfs_reachable(grid, agent_pos, goal_pos,
                                   passable_doors=door_set):
            return False

        # 4. Solvability with 1-key-at-a-time constraint
        # Agent can hold at most 1 key. Must pick up key_i, walk to door_i,
        # open it, then go get key_{i+1}. We simulate this sequentially:
        # doors must be opened in order 0, 1, 2, ... because that's how
        # the linear chain works.
        door_color_map: dict[tuple[int, int], int] = {}
        for dp in config.get("door_positions", []):
            dp_t = tuple(dp)
            door_color_map[dp_t] = int(grid.metadata[dp_t[1], dp_t[0]]) % 10

        key_positions = config.get("key_positions", [])
        key_color_map: dict[tuple[int, int], int] = {}
        for kp in key_positions:
            kp_t = tuple(kp)
            key_color_map[kp_t] = int(grid.metadata[kp_t[1], kp_t[0]])

        # Build color->key_pos and color->door_pos maps
        color_to_keys: dict[int, list[tuple[int, int]]] = {}
        for kp_t, kc in key_color_map.items():
            color_to_keys.setdefault(kc, []).append(kp_t)
        color_to_door: dict[int, tuple[int, int]] = {}
        for dp_t, dc in door_color_map.items():
            color_to_door[dc] = dp_t

        # Simulate sequential door opening: open door 0, then 1, then 2, ...
        opened_doors: set[tuple[int, int]] = set()
        used_keys: set[tuple[int, int]] = set()
        door_positions_ordered = config.get("door_positions", [])

        for door_idx in range(len(door_positions_ordered)):
            dp_t = tuple(door_positions_ordered[door_idx])
            door_color = door_color_map[dp_t]

            # Find a reachable key of matching color
            available_keys = [
                kp for kp in color_to_keys.get(door_color, [])
                if kp not in used_keys
            ]
            reachable = self._bfs_reachable_set(grid, agent_pos, opened_doors)

            found_key = False
            for kp in available_keys:
                if kp in reachable:
                    # Can reach this key, now check if we can reach the door
                    # from the key position (with currently opened doors)
                    if self._bfs_reachable(
                        grid, kp, dp_t,
                        passable_doors=opened_doors | {dp_t}
                    ):
                        used_keys.add(kp)
                        opened_doors.add(dp_t)
                        found_key = True
                        break
            if not found_key:
                return False

        # After opening all doors, check if goal is reachable
        reachable = self._bfs_reachable_set(grid, agent_pos, opened_doors)
        return tuple(goal_pos) in reachable

    def _bfs_reachable(self, grid, start, target, passable_doors):
        """BFS from start to target. Doors in passable_doors are walkable."""
        start = tuple(start)
        target = tuple(target)
        visited = {start}
        queue = deque([start])
        while queue:
            cx, cy = queue.popleft()
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = cx + dx, cy + dy
                if (nx, ny) in visited:
                    continue
                if not (0 <= nx < grid.width and 0 <= ny < grid.height):
                    continue
                if grid.terrain[ny, nx] == CellType.WALL:
                    continue
                if (grid.objects[ny, nx] == ObjectType.DOOR
                        and (nx, ny) not in passable_doors):
                    continue
                if (nx, ny) == target:
                    return True
                visited.add((nx, ny))
                queue.append((nx, ny))
        return False

    def _bfs_reachable_set(self, grid, start, passable_doors):
        """BFS from start, returning all reachable positions."""
        start = tuple(start)
        visited = {start}
        queue = deque([start])
        while queue:
            cx, cy = queue.popleft()
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = cx + dx, cy + dy
                if (nx, ny) in visited:
                    continue
                if not (0 <= nx < grid.width and 0 <= ny < grid.height):
                    continue
                if grid.terrain[ny, nx] == CellType.WALL:
                    continue
                if (grid.objects[ny, nx] == ObjectType.DOOR
                        and (nx, ny) not in passable_doors):
                    continue
                visited.add((nx, ny))
                queue.append((nx, ny))
        return visited

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0
