"""SwitchCircuit - Room-based switch dependency puzzle with spatial chains.

MECHANICS:
  - Grid divided into rooms connected by single-cell doors (barriers)
  - N switches, one per room (Si in room Ri)
  - Each switch opens exactly ONE door and NEVER closes anything
  - INTERACT (action 5) on a switch toggles it ON/OFF
  - A barrier is OPEN if at least one switch that 'opens' it is ON
  - Puzzle complexity comes from SPATIAL LAYOUT: switches behind locked doors
    create a chain of dependencies requiring back-and-forth navigation

TOPOLOGY:
  - Easy (n=2): linear chain R0--D0--R1--D1--GoalRoom
  - Medium (n=3): chain rooms with GoalRoom below R0 via D_goal
  - Hard (n=4): hub-and-spoke layout
      S0 in hub → opens D0 (spoke0 door)
      S1 in spoke0 → opens D1 (spoke1 door)
      S2 in spoke1 → opens D2 (spoke2 door)
      S3 in spoke2 → opens D_goal (goal door)
  - Expert (n=5): 2x3 grid layout, all connections gated
      S0 in R0 → opens B0 (R0→R1)
      S1 in R1 → opens B1 (R1→R2)
      S2 in R2 → opens B2 (R0→R3)
      S3 in R3 → opens B3 (R3→R4)
      S4 in R4 → opens B4 (R4→R5 goal)

  - Success = agent on GOAL cell
"""

from __future__ import annotations

from collections import deque

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


def _bfs_reachable(terrain, start, height, width):
    """BFS flood-fill on terrain array. Returns set of reachable (x,y)."""
    visited = {start}
    q = deque([start])
    while q:
        cx, cy = q.popleft()
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited:
                t = int(terrain[ny, nx])
                if t not in (int(CellType.WALL), int(CellType.HAZARD)):
                    visited.add((nx, ny))
                    q.append((nx, ny))
    return visited


def _room_connected(terrain, room_bounds, objects=None):
    """Check if all walkable cells in a room are connected via BFS.

    When *objects* is provided, cells with non-walkable objects (SWITCH, LEVER,
    DOOR) are excluded from the free set but must still be reachable from an
    adjacent cell for the room to count as connected.
    """
    from agentick.core.types import NON_WALKABLE_OBJECTS

    x_s, x_e, y_s, y_e = room_bounds
    free = []
    for y in range(y_s, y_e + 1):
        for x in range(x_s, x_e + 1):
            if int(terrain[y, x]) != int(CellType.EMPTY):
                continue
            if objects is not None:
                obj = ObjectType(int(objects[y, x]))
                if obj in NON_WALKABLE_OBJECTS:
                    continue
            free.append((x, y))
    if len(free) <= 1:
        return True
    visited = {free[0]}
    q = deque([free[0]])
    while q:
        cx, cy = q.popleft()
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            nx, ny = cx + dx, cy + dy
            if (
                x_s <= nx <= x_e and y_s <= ny <= y_e
                and (nx, ny) not in visited
                and int(terrain[ny, nx]) == int(CellType.EMPTY)
            ):
                if objects is not None:
                    obj = ObjectType(int(objects[ny, nx]))
                    if obj in NON_WALKABLE_OBJECTS:
                        continue
                visited.add((nx, ny))
                q.append((nx, ny))
    return len(visited) == len(free)


@register_task("SwitchCircuit-v0", tags=["combinatorial_logic", "reasoning"])
class SwitchCircuitTask(TaskSpec):
    """Room-based switch puzzle with spatial dependency chains.

    Rooms connected by single-cell doors. Each switch opens exactly one door
    and never closes anything. Puzzle complexity comes from spatial layout:
    switches are placed behind locked doors, creating a chain of dependencies
    that requires back-and-forth navigation.
    """

    name = "SwitchCircuit-v0"
    description = "Plan switch activation order to open barriers and reach the goal"
    capability_tags = ["combinatorial_logic", "reasoning"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=11,
            max_steps=100,
            params={"n_switches": 2},
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=13,
            max_steps=250,
            params={"n_switches": 3},
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=15,
            max_steps=400,
            params={"n_switches": 4},
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=19,
            max_steps=600,
            params={"n_switches": 5},
        ),
    }

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        n = self.difficulty_config.params.get("n_switches", 2)

        grid = Grid(size, size)
        grid.terrain[:, :] = CellType.WALL

        if n >= 5:
            rooms, barriers, goal_room = self._layout_grid_2x3(grid, n, size, rng)
        elif n == 4:
            rooms, barriers, goal_room = self._layout_hub_spoke(grid, n, size, rng)
        elif n == 3:
            rooms, barriers, goal_room = self._layout_dual(grid, n, size, rng)
        else:
            rooms, barriers, goal_room = self._layout_easy(grid, n, size, rng)

        # Place switches: Si in Ri (one switch per chain room)
        used_positions: set[tuple[int, int]] = set()
        switch_positions = []
        chain_rooms = rooms[:n]

        for i in range(n):
            x_s, x_e, y_s, y_e = chain_rooms[i]
            free = [
                (x, y)
                for y in range(y_s, y_e + 1)
                for x in range(x_s, x_e + 1)
                if int(grid.terrain[y, x]) == int(CellType.EMPTY)
                and (x, y) not in used_positions
            ]
            rng.shuffle(free)
            sx, sy = free[0]
            switch_positions.append((sx, sy))
            used_positions.add((sx, sy))
            grid.objects[sy, sx] = ObjectType.SWITCH
            grid.metadata[sy, sx] = 0

        # Place agent in R0
        r0 = chain_rooms[0]
        free = [
            (x, y)
            for y in range(r0[2], r0[3] + 1)
            for x in range(r0[0], r0[1] + 1)
            if int(grid.terrain[y, x]) == int(CellType.EMPTY)
            and (x, y) not in used_positions
        ]
        rng.shuffle(free)
        agent_pos = free[0]
        used_positions.add(agent_pos)

        # Place goal in GoalRoom
        free = [
            (x, y)
            for y in range(goal_room[2], goal_room[3] + 1)
            for x in range(goal_room[0], goal_room[1] + 1)
            if int(grid.terrain[y, x]) == int(CellType.EMPTY)
            and (x, y) not in used_positions
        ]
        rng.shuffle(free)
        goal_pos = free[0]
        used_positions.add(goal_pos)
        grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.GOAL

        # Build switch effects
        n_barriers = len(barriers)
        switch_effects = self._build_dependency_graph(n, n_barriers, rng=rng)

        # Add scatter walls for visual interest (with connectivity check)
        self._add_scatter_walls(grid, chain_rooms, goal_room,
                                used_positions, barriers, rng)

        # Validate solvability
        valid = self._validate_solvable_bfs(
            n, barriers, switch_effects, grid, agent_pos, goal_pos,
            switch_positions, size,
        )
        if not valid:
            switch_effects = self._build_simple_chain(n, n_barriers)

        config = {
            "agent_start": agent_pos,
            "goal_positions": [goal_pos],
            "switch_positions": switch_positions,
            "switch_states": [False] * n,
            "barriers": barriers,
            "switch_effects": switch_effects,
            "max_steps": self.get_max_steps(),
        }

        return grid, config

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------

    def _layout_easy(self, grid, n, size, rng):
        """Easy: N+1 rooms in a single horizontal row, GoalRoom is last."""
        n_rooms = n + 1
        interior_width = size - 2
        n_walls = n_rooms - 1
        room_space = interior_width - n_walls
        base_w = room_space // n_rooms
        remainder = room_space % n_rooms

        rooms = []
        x = 1
        for i in range(n_rooms):
            w = base_w + (1 if i < remainder else 0)
            rooms.append((x, x + w - 1, 1, size - 2))
            x += w
            if i < n_rooms - 1:
                x += 1  # wall column

        # Carve rooms
        for x_s, x_e, y_s, y_e in rooms:
            for y in range(y_s, y_e + 1):
                for x in range(x_s, x_e + 1):
                    grid.terrain[y, x] = CellType.EMPTY

        # Place doors between rooms
        barriers = []
        for i in range(n_rooms - 1):
            wall_x = rooms[i][1] + 1
            y_s, y_e = rooms[i][2], rooms[i][3]
            door_y = int(rng.integers(y_s, y_e + 1))
            barriers.append({"cells": [[wall_x, door_y]], "open": False})

        goal_room = rooms[-1]
        return rooms, barriers, goal_room

    def _layout_dual(self, grid, n, size, rng):
        """Medium (n=3): N chain rooms on top, GoalRoom below R0."""
        n_chain = n

        # Vertical split: top ~70%, bottom ~30%
        interior_height = size - 2
        top_height = max(3, interior_height * 7 // 10)
        sep_row = 1 + top_height
        if sep_row > size - 4:
            sep_row = size - 4
        if sep_row < 4:
            sep_row = 4

        # Chain rooms: horizontal row in top section
        interior_width = size - 2
        n_walls = n_chain - 1
        room_space = interior_width - n_walls
        base_w = room_space // n_chain
        remainder = room_space % n_chain

        rooms = []
        x = 1
        for i in range(n_chain):
            w = base_w + (1 if i < remainder else 0)
            rooms.append((x, x + w - 1, 1, sep_row - 1))
            x += w
            if i < n_chain - 1:
                x += 1

        # Carve chain rooms
        for x_s, x_e, y_s, y_e in rooms:
            for y in range(y_s, y_e + 1):
                for x in range(x_s, x_e + 1):
                    grid.terrain[y, x] = CellType.EMPTY

        # Doors between chain rooms: D0..D(n_chain-2)
        barriers = []
        for i in range(n_chain - 1):
            wall_x = rooms[i][1] + 1
            y_s, y_e = rooms[i][2], rooms[i][3]
            door_y = int(rng.integers(y_s, y_e + 1))
            barriers.append({"cells": [[wall_x, door_y]], "open": False})

        # GoalRoom below R0
        r0_x_s, r0_x_e = rooms[0][0], rooms[0][1]
        goal_room = (r0_x_s, r0_x_e, sep_row + 1, size - 2)

        # Carve GoalRoom
        for y in range(goal_room[2], goal_room[3] + 1):
            for x in range(goal_room[0], goal_room[1] + 1):
                grid.terrain[y, x] = CellType.EMPTY

        # D_goal: door in separator wall connecting R0 to GoalRoom
        door_x = int(rng.integers(r0_x_s, r0_x_e + 1))
        barriers.append({"cells": [[door_x, sep_row]], "open": False})

        return rooms, barriers, goal_room

    def _layout_hub_spoke(self, grid, n, size, rng):
        """Hard: hub-and-spoke layout with 4 switches."""
        # Define regions
        third = size // 3

        # Hub room: center
        hub = (third, 2 * third - 1, third, 2 * third - 1)

        # Spoke 0: top (above hub)
        spoke0 = (third, 2 * third - 1, 1, third - 2)

        # Spoke 1: right (right of hub)
        spoke1 = (2 * third + 1, size - 2, third, 2 * third - 1)

        # Spoke 2: left (left of hub)
        spoke2 = (1, third - 2, third, 2 * third - 1)

        # Goal room: bottom (below hub)
        goal_room = (third, 2 * third - 1, 2 * third + 1, size - 2)

        # Carve all rooms
        rooms = [hub, spoke0, spoke1, spoke2]
        for x_s, x_e, y_s, y_e in rooms + [goal_room]:
            for y in range(y_s, y_e + 1):
                for x in range(x_s, x_e + 1):
                    grid.terrain[y, x] = CellType.EMPTY

        # Place doors (barriers)
        barriers = []

        # D0: hub to spoke0 (top wall of hub)
        d0_x = int(rng.integers(hub[0], hub[1] + 1))
        d0_y = third - 1  # wall row between spoke0 and hub
        barriers.append({"cells": [[d0_x, d0_y]], "open": False})

        # D1: hub to spoke1 (right wall of hub)
        d1_x = 2 * third  # wall column between hub and spoke1
        d1_y = int(rng.integers(hub[2], hub[3] + 1))
        barriers.append({"cells": [[d1_x, d1_y]], "open": False})

        # D2: hub to spoke2 (left wall of hub)
        d2_x = third - 1  # wall column between spoke2 and hub
        d2_y = int(rng.integers(hub[2], hub[3] + 1))
        barriers.append({"cells": [[d2_x, d2_y]], "open": False})

        # D_goal: hub to goal room (bottom wall of hub)
        dg_x = int(rng.integers(hub[0], hub[1] + 1))
        dg_y = 2 * third  # wall row between hub and goal room
        barriers.append({"cells": [[dg_x, dg_y]], "open": False})

        return rooms, barriers, goal_room

    def _layout_grid_2x3(self, grid, n, size, rng):
        """Expert: 2x3 grid layout with n switches and n doors.

        6 rooms in 2 rows x 3 columns. ALL connections are gated (no open
        passages). Only 5 of the 7 possible connections exist, chosen to
        create a sequential chain requiring back-and-forth navigation.

        Gated (barriers 0..4):
          B0: R0->R1, B1: R1->R2, B2: R0->R3, B3: R3->R4, B4: R4->R5

        No connection: R1->R4, R2->R5 (solid walls, no passage).

        Chain: S0(R0)->B0->R1, S1(R1)->B1->R2 (dead end), S2(R2)->B2
        -> backtrack to R0->R3, S3(R3)->B3->R4, S4(R4)->B4->R5(goal).
        """
        n_cols = 3
        n_rows = 2
        interior_w = size - 2
        interior_h = size - 2

        col_walls = n_cols - 1
        row_walls = n_rows - 1
        col_space = interior_w - col_walls
        row_space = interior_h - row_walls
        col_w = col_space // n_cols
        row_h = row_space // n_rows

        rooms_grid = []
        for r in range(n_rows):
            row_rooms = []
            for c in range(n_cols):
                x_s = 1 + c * (col_w + 1)
                x_e = x_s + col_w - 1
                y_s = 1 + r * (row_h + 1)
                y_e = y_s + row_h - 1
                row_rooms.append((x_s, x_e, y_s, y_e))
            rooms_grid.append(row_rooms)

        # Carve rooms
        for r in range(n_rows):
            for c in range(n_cols):
                x_s, x_e, y_s, y_e = rooms_grid[r][c]
                for y in range(y_s, y_e + 1):
                    for x in range(x_s, x_e + 1):
                        grid.terrain[y, x] = CellType.EMPTY

        barriers = []

        # B0: R0->R1 (top row, horizontal)
        left = rooms_grid[0][0]
        wall_x = left[1] + 1
        door_y = int(rng.integers(left[2], left[3] + 1))
        barriers.append({"cells": [[wall_x, door_y]], "open": False})

        # B1: R1->R2 (top row, horizontal)
        left = rooms_grid[0][1]
        wall_x = left[1] + 1
        door_y = int(rng.integers(left[2], left[3] + 1))
        barriers.append({"cells": [[wall_x, door_y]], "open": False})

        # B2: R0->R3 (vertical, left column)
        top = rooms_grid[0][0]
        wall_y = top[3] + 1
        door_x = int(rng.integers(top[0], top[1] + 1))
        barriers.append({"cells": [[door_x, wall_y]], "open": False})

        # B3: R3->R4 (bottom row, left-to-middle)
        left_bot = rooms_grid[1][0]
        wall_x = left_bot[1] + 1
        door_y = int(rng.integers(left_bot[2], left_bot[3] + 1))
        barriers.append({"cells": [[wall_x, door_y]], "open": False})

        # B4: R4->R5 (bottom row, middle-to-right)
        mid_bot = rooms_grid[1][1]
        wall_x = mid_bot[1] + 1
        door_y = int(rng.integers(mid_bot[2], mid_bot[3] + 1))
        barriers.append({"cells": [[wall_x, door_y]], "open": False})

        # No open passages — R1->R4 and R2->R5 remain solid walls

        # Flatten rooms: R0, R1, R2, R3, R4 (5 rooms for 5 switches)
        rooms = [
            rooms_grid[0][0], rooms_grid[0][1], rooms_grid[0][2],
            rooms_grid[1][0], rooms_grid[1][1],
        ]

        goal_room = rooms_grid[1][2]  # R5 = bottom-right

        return rooms, barriers, goal_room

    # ------------------------------------------------------------------
    # Dependency graph
    # ------------------------------------------------------------------

    def _build_dependency_graph(self, n_switches, n_barriers, rng=None):
        """Simple chain: switch i opens barrier i, no closes.

        All difficulties use the same simple dependency logic. Puzzle
        complexity comes from the spatial layout (switches behind locked
        doors), not from switches having dual open/close effects.
        """
        return self._build_simple_chain(n_switches, n_barriers)

    def _build_simple_chain(self, n_switches, n_barriers):
        """Simple chain: switch i opens barrier i, no closes."""
        effects = []
        for i in range(n_switches):
            b_idx = min(i, n_barriers - 1)
            effects.append({"opens": [b_idx], "closes": []})
        return effects

    # ------------------------------------------------------------------
    # Scatter walls
    # ------------------------------------------------------------------

    def _add_scatter_walls(self, grid, chain_rooms, goal_room,
                           used_positions, barriers, rng):
        """Add random internal walls for visual interest, preserving connectivity."""
        protected = set(used_positions)

        # Protect cells adjacent to doors
        for barrier in barriers:
            for cell in barrier.get("cells", []):
                dx, dy = cell[0], cell[1]
                for ddx, ddy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                    nx, ny = dx + ddx, dy + ddy
                    protected.add((nx, ny))

        # Protect cells adjacent to switches/agent/goal (solid objects)
        for px, py in used_positions:
            for ddx, ddy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                protected.add((px + ddx, py + ddy))

        all_rooms = list(chain_rooms) + [goal_room]
        for room in all_rooms:
            x_s, x_e, y_s, y_e = room
            room_area = (x_e - x_s + 1) * (y_e - y_s + 1)
            n_walls = max(0, room_area // 12)
            for _ in range(n_walls):
                wx = int(rng.integers(x_s, x_e + 1))
                wy = int(rng.integers(y_s, y_e + 1))
                if (wx, wy) in protected:
                    continue
                if int(grid.terrain[wy, wx]) != int(CellType.EMPTY):
                    continue
                grid.terrain[wy, wx] = CellType.WALL
                if not _room_connected(grid.terrain, room, grid.objects):
                    grid.terrain[wy, wx] = CellType.EMPTY

    # ------------------------------------------------------------------
    # Barrier state logic
    # ------------------------------------------------------------------

    def _compute_barrier_states(self, switch_states, switch_effects, n_barriers):
        """Compute which barriers are open given current switch states.

        A barrier is OPEN if at least one switch that 'opens' it is ON,
        AND no switch that 'closes' it is ON.
        """
        barrier_open = [False] * n_barriers
        barrier_has_closer_on = [False] * n_barriers

        for sw_idx, is_on in enumerate(switch_states):
            if not is_on or sw_idx >= len(switch_effects):
                continue
            effects = switch_effects[sw_idx]
            for b_idx in effects.get("opens", []):
                if b_idx < n_barriers:
                    barrier_open[b_idx] = True
            for b_idx in effects.get("closes", []):
                if b_idx < n_barriers:
                    barrier_has_closer_on[b_idx] = True

        return [
            barrier_open[i] and not barrier_has_closer_on[i]
            for i in range(n_barriers)
        ]

    def _apply_barrier_states(self, grid, barriers, barrier_states):
        """Apply computed barrier states to the grid terrain."""
        for b_idx, barrier in enumerate(barriers):
            cells = barrier.get("cells", [])
            is_open = barrier_states[b_idx] if b_idx < len(barrier_states) else False
            barrier["open"] = is_open
            for cell in cells:
                cx, cy = cell[0], cell[1]
                if is_open:
                    grid.terrain[cy, cx] = CellType.EMPTY
                else:
                    grid.terrain[cy, cx] = CellType.WALL
                    grid.metadata[cy, cx] = b_idx

    def _validate_solvable_bfs(self, n_switches, barriers, switch_effects,
                               grid, agent_pos, goal_pos,
                               switch_positions, size):
        """BFS over (position_index, switch_bitmask) to check solvability."""
        if n_switches == 0:
            reachable = _bfs_reachable(grid.terrain, agent_pos, size, size)
            return goal_pos in reachable

        n_barriers = len(barriers)
        barrier_cell_sets = [
            [(c[0], c[1]) for c in b.get("cells", [])] for b in barriers
        ]

        key_positions = [agent_pos] + list(switch_positions) + [goal_pos]

        def _get_terrain_for_mask(mask):
            states = [(mask >> i) & 1 for i in range(n_switches)]
            b_open = self._compute_barrier_states(states, switch_effects, n_barriers)
            t = grid.terrain.copy()
            for b_idx, cells in enumerate(barrier_cell_sets):
                is_open = b_open[b_idx] if b_idx < len(b_open) else False
                for cx, cy in cells:
                    t[cy, cx] = int(CellType.EMPTY) if is_open else int(CellType.WALL)
            return t

        start_state = (0, 0)
        visited = {start_state}
        q = deque([start_state])

        while q:
            pos_idx, mask = q.popleft()
            pos = key_positions[pos_idx]
            terrain = _get_terrain_for_mask(mask)
            reachable = _bfs_reachable(terrain, pos, size, size)

            if goal_pos in reachable:
                return True

            for sw_i in range(n_switches):
                sw_pos = switch_positions[sw_i]
                if sw_pos not in reachable:
                    continue
                new_mask = mask ^ (1 << sw_i)
                new_state = (sw_i + 1, new_mask)
                if new_state not in visited:
                    visited.add(new_state)
                    q.append(new_state)

        return False

    # ---------------------------------------------------------------
    # Runtime hooks
    # ---------------------------------------------------------------

    def on_env_reset(self, agent, grid, config):
        """Initialize runtime state: all switches OFF, all barriers closed."""
        n = len(config.get("switch_positions", []))
        config["switch_states"] = [False] * n

        barriers = config.get("barriers", [])
        for b_idx, barrier in enumerate(barriers):
            barrier["open"] = False
            for cell in barrier.get("cells", []):
                cx, cy = cell[0], cell[1]
                grid.terrain[cy, cx] = CellType.WALL
                grid.metadata[cy, cx] = b_idx

        goal_pos = config.get("goal_positions", [None])[0]
        if goal_pos:
            gx, gy = goal_pos
            grid.objects[gy, gx] = ObjectType.GOAL

        switch_positions = config.get("switch_positions", [])
        for _i, (sx, sy) in enumerate(switch_positions):
            grid.objects[sy, sx] = ObjectType.SWITCH
            grid.metadata[sy, sx] = 0

        self._config = config

    def on_agent_interact(self, pos, agent, grid):
        """INTERACT on a SWITCH toggles it ON/OFF."""
        config = getattr(self, "_config", {})
        switch_positions = config.get("switch_positions", [])
        switch_states = config.get("switch_states", [])
        barriers = config.get("barriers", [])
        switch_effects = config.get("switch_effects", [])
        ax, ay = pos

        sw_idx = None
        for i, sp in enumerate(switch_positions):
            if (sp[0], sp[1]) == (ax, ay):
                sw_idx = i
                break

        if sw_idx is None:
            return

        switch_states[sw_idx] = not switch_states[sw_idx]

        if switch_states[sw_idx]:
            grid.metadata[ay, ax] = 100
        else:
            grid.metadata[ay, ax] = 0

        n_barriers = len(barriers)
        barrier_open = self._compute_barrier_states(
            switch_states, switch_effects, n_barriers,
        )
        self._apply_barrier_states(grid, barriers, barrier_open)
        config["switch_states"] = switch_states

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01

        if self.check_success(new_state):
            reward += 1.0
            return reward

        config = new_state.get("config", {})
        if "agent" in new_state:
            ax, ay = new_state["agent"].position
            ox, oy = old_state.get("agent_position", (ax, ay))
            switch_states = config.get("switch_states", [])
            switch_positions = config.get("switch_positions", [])

            # Shape toward nearest OFF switch, or goal if all correct
            target = None
            best_dist = 9999
            for i, sp in enumerate(switch_positions):
                if i < len(switch_states) and not switch_states[i]:
                    d = abs(ax - sp[0]) + abs(ay - sp[1])
                    if d < best_dist:
                        best_dist = d
                        target = (sp[0], sp[1])

            if target is None:
                goal = config.get("goal_positions", [None])[0]
                if goal:
                    target = (goal[0], goal[1])

            if target is not None:
                tx, ty = target
                d_new = abs(ax - tx) + abs(ay - ty)
                d_old = abs(ox - tx) + abs(oy - ty)
                reward += 0.05 * (d_old - d_new)

        return reward

    def check_success(self, state):
        if "grid" not in state or "agent" not in state:
            return False
        x, y = state["agent"].position
        return bool(state["grid"].objects[y, x] == ObjectType.GOAL)

    def check_done(self, state):
        return self.check_success(state)

    def validate_instance(self, grid, config):
        """Verify the puzzle is solvable via BFS over toggle state space."""
        agent_pos = config.get("agent_start", (1, 1))
        switch_positions = config.get("switch_positions", [])
        goal_positions = config.get("goal_positions", [])
        barriers = config.get("barriers", [])
        switch_effects = config.get("switch_effects", [])
        n = len(switch_positions)

        if not switch_positions or not goal_positions:
            return True

        goal_pos = tuple(goal_positions[0])
        size = grid.height

        return self._validate_solvable_bfs(
            n, barriers, switch_effects, grid, agent_pos, goal_pos,
            switch_positions, size,
        )

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0
