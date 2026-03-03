"""SwitchCircuit - Room-based dual-toggle switch dependency puzzle.

MECHANICS:
  - Grid divided into rooms connected by single-cell doors
  - N switches, one per room (Si in room Ri)
  - Each switch opens the NEXT door; dual switches also close the PREVIOUS door
  - INTERACT (action 8) on a switch toggles it ON/OFF; effects reverse when OFF
  - A barrier is OPEN if at least one switch that 'opens' it is ON
    AND no switch that 'closes' it is ON

TOPOLOGY:
  - Easy: linear chain R0--D0--R1--D1--GoalRoom (no dual switches)
  - Medium+: chain rooms with GoalRoom hanging off Hub (R0) via D_goal:
      R0(S0)--D0--R1(S1)--...--R(N-1)(S(N-1))
       |
      D_goal
       |
      GoalRoom(GOAL)
    Dual switches open the next door but close the previous, forcing the
    agent to unwind toggles (ON->OFF) to get back to Hub and reach GoalRoom.

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


def _room_connected(terrain, room_bounds):
    """Check if all EMPTY cells in a room are connected via BFS."""
    x_s, x_e, y_s, y_e = room_bounds
    free = []
    for y in range(y_s, y_e + 1):
        for x in range(x_s, x_e + 1):
            if int(terrain[y, x]) == int(CellType.EMPTY):
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
                visited.add((nx, ny))
                q.append((nx, ny))
    return len(visited) == len(free)


@register_task("SwitchCircuit-v0", tags=["combinatorial_logic", "reasoning"])
class SwitchCircuitTask(TaskSpec):
    """Room-based switch puzzle with forced toggle cycles.

    Rooms connected by single-cell doors. Each switch opens the next door;
    dual switches (medium+) also close the previous door, forcing the agent
    to toggle switches ON->OFF->ON to navigate back and reach the goal.
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
        has_dual = n >= 3

        grid = Grid(size, size)
        grid.terrain[:, :] = CellType.WALL

        if has_dual:
            rooms, barriers, goal_room = self._layout_dual(grid, n, size, rng)
        else:
            rooms, barriers, goal_room = self._layout_easy(grid, n, size, rng)

        # Place switches: Si in Ri (one switch per chain room)
        used_positions: set[tuple[int, int]] = set()
        switch_positions = []
        chain_rooms = rooms if has_dual else rooms[:n]

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
        switch_effects = self._build_dependency_graph(n, n_barriers)

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
        """Medium+: N chain rooms on top, GoalRoom below R0."""
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

    # ------------------------------------------------------------------
    # Dependency graph
    # ------------------------------------------------------------------

    def _build_dependency_graph(self, n_switches, n_barriers):
        """Build switch effects: Si opens Di, dual switches close D(i-1).

        Easy (n<=2): simple chain, no closes.
        Medium+ (n>=3): S0 opens D0, Si (0<i<n-1) opens Di + closes D(i-1),
                        S(n-1) opens D_goal (last barrier).
        """
        if n_switches == 0 or n_barriers == 0:
            return []

        if n_switches <= 2:
            return self._build_simple_chain(n_switches, n_barriers)

        effects = []
        for i in range(n_switches):
            opens = [i]
            closes = []
            if 0 < i < n_switches - 1:
                closes = [i - 1]
            effects.append({"opens": opens, "closes": closes})
        return effects

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
                if not _room_connected(grid.terrain, room):
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
