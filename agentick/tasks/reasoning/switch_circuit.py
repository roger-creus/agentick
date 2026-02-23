"""SwitchCircuit - Complementary color-coded toggle switches with barrier zones.

MECHANICS:
  - N colored switches and N colored wall barriers divide the map into zones
  - Toggling switch i (color i) OPENS barrier i but CLOSES barrier (i+1)%N
  - Agent must toggle switches in order 0..N-1, navigating through each
    newly opened barrier before the complementary effect closes anything
  - All barriers start as WALL (closed); goal is in the last zone
  - Success = agent on GOAL cell after passing all barriers
"""

from __future__ import annotations

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("SwitchCircuit-v0", tags=["combinatorial_logic", "reasoning"])
class SwitchCircuitTask(TaskSpec):
    """Toggle color-coded switches in order to open barriers and reach the goal.

    N horizontal wall barriers divide the grid into N+1 zones. Each switch
    has a matching color index. Toggling switch i opens barrier i but closes
    barrier (i+1)%N, so the agent must proceed through each barrier
    immediately after opening it.
    """

    name = "SwitchCircuit-v0"
    description = "Toggle color-coded switches to open barriers and reach the goal"
    capability_tags = ["combinatorial_logic", "reasoning"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=7,
            max_steps=80,
            params={"n_switches": 2},
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=9,
            max_steps=150,
            params={"n_switches": 3},
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=11,
            max_steps=250,
            params={"n_switches": 4},
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=13,
            max_steps=400,
            params={"n_switches": 5},
        ),
    }

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        n = self.difficulty_config.params.get("n_switches", 2)

        grid = Grid(size, size)

        # Border walls
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        agent_pos = (1, 1)

        # Compute barrier rows so every zone has at least one interior row
        # for switch/goal placement.
        #
        # Interior rows available: 1 .. size-2 inclusive.
        # We need n barriers plus n+1 zones each containing >= 1 free row.
        # Minimum rows: 2*n + 1.  The difficulty configs guarantee size >= 2*n+3.
        #
        # Barriers may occupy rows 2 .. size-3 inclusive so that:
        #   - row 1 is always free for zone 0 (agent start zone)
        #   - row size-2 is always free for zone n (goal zone)
        # This guarantees each boundary zone has at least one free row.
        #
        # Divide the range [2 .. size-3] into n+1 equal-ish segments and place
        # barrier i at the boundary between segment i and i+1.
        barrier_min_row = 2
        barrier_max_row = size - 3  # inclusive; row size-2 reserved for goal zone

        # Number of barrier-eligible rows
        row_range = barrier_max_row - barrier_min_row + 1  # >= 1
        # spacing between barrier positions (rows per segment, including barrier)
        spacing = max(2, row_range // n)
        barrier_rows = []
        for i in range(n):
            row = barrier_min_row + i * spacing
            row = min(row, barrier_max_row)
            barrier_rows.append(row)

        # Enforce separation: each barrier must be >= 2 rows from the previous
        # so there is at least one free row between consecutive barriers.
        fixed: list[int] = []
        for row in barrier_rows:
            if fixed:
                row = max(row, fixed[-1] + 2)
            row = min(row, barrier_max_row)
            fixed.append(row)
        barrier_rows = fixed[:n]

        # Record barrier cells per color index (color = barrier index 0..n-1)
        # Leave one gap column in each barrier row so passage is possible
        # when the barrier is opened. The gap is at column = size//2 (center).
        gap_col = size // 2
        barrier_cells: dict[int, list[tuple[int, int]]] = {}
        for color_idx, row in enumerate(barrier_rows):
            cells = []
            for x in range(1, size - 1):
                if x != gap_col:
                    grid.terrain[row, x] = CellType.WALL
                    grid.metadata[row, x] = color_idx
                    cells.append((x, row))
            barrier_cells[color_idx] = cells

        # Place switches — one per zone BEFORE its corresponding barrier.
        # Zone i is between barrier[i-1] (or top) and barrier[i].
        # Switch[i] goes in zone i.
        zone_bounds: list[tuple[int, int]] = []
        prev_top = 1  # inclusive first row of zone
        for i in range(n):
            zone_top = prev_top
            zone_bot = barrier_rows[i] - 1  # inclusive last row before barrier
            zone_bounds.append((zone_top, zone_bot))
            prev_top = barrier_rows[i] + 1

        switch_positions: list[tuple[int, int]] = []
        used_positions = {agent_pos}
        for i in range(n):
            zone_top, zone_bot = zone_bounds[i]
            # Gather free cells in this zone
            free = [
                (x, y)
                for y in range(zone_top, zone_bot + 1)
                for x in range(1, size - 1)
                if (x, y) not in used_positions
                and grid.terrain[y, x] == CellType.EMPTY
                and grid.objects[y, x] == ObjectType.NONE
            ]
            if not free:
                # Fallback: any non-wall, non-agent interior cell
                free = [
                    (x, y)
                    for y in range(1, size - 1)
                    for x in range(1, size - 1)
                    if (x, y) not in used_positions
                    and grid.terrain[y, x] == CellType.EMPTY
                ]
            rng.shuffle(free)
            sx, sy = free[0]
            switch_positions.append((sx, sy))
            used_positions.add((sx, sy))
            grid.objects[sy, sx] = ObjectType.SWITCH
            grid.metadata[sy, sx] = i  # color index = switch index

        # Goal in the last zone (between last barrier and bottom wall)
        last_zone_top = barrier_rows[-1] + 1
        goal_candidates = [
            (x, y)
            for y in range(last_zone_top, size - 1)
            for x in range(1, size - 1)
            if (x, y) not in used_positions
            and grid.terrain[y, x] == CellType.EMPTY
        ]
        if not goal_candidates:
            # Fallback to bottom-right area
            goal_candidates = [
                (x, y)
                for y in range(1, size - 1)
                for x in range(1, size - 1)
                if (x, y) not in used_positions and grid.terrain[y, x] == CellType.EMPTY
            ]
        rng.shuffle(goal_candidates)
        goal_pos = goal_candidates[0]
        grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.GOAL

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [goal_pos],
            "switch_positions": switch_positions,
            "barrier_rows": barrier_rows,
            "barrier_cells": {k: [list(c) for c in v] for k, v in barrier_cells.items()},
            "gap_col": gap_col,
            "max_steps": self.get_max_steps(),
        }

    def on_env_reset(self, agent, grid, config):
        """Initialise runtime state: all switches OFF, barriers closed."""
        config["_switches_on"] = set()

        # Re-close all barriers (they may have been opened in a prior episode)
        barrier_cells = config.get("barrier_cells", {})
        for color_idx_str, cells in barrier_cells.items():
            color_idx = int(color_idx_str)
            for cell in cells:
                cx, cy = cell[0], cell[1]
                grid.terrain[cy, cx] = CellType.WALL
                grid.metadata[cy, cx] = color_idx

        # Place GOAL object
        goal_pos = config.get("goal_positions", [None])[0]
        if goal_pos:
            gx, gy = goal_pos
            grid.objects[gy, gx] = ObjectType.GOAL

        # Restore switch objects (in case a previous episode removed them)
        switch_positions = config.get("switch_positions", [])
        for i, (sx, sy) in enumerate(switch_positions):
            grid.objects[sy, sx] = ObjectType.SWITCH
            grid.metadata[sy, sx] = i

        self._config = config
        self._last_n_switches = 0

    def on_agent_moved(self, pos, agent, grid):
        """Detect if agent stepped on a SWITCH and toggle it."""
        config = getattr(self, "_config", {})
        switch_positions = config.get("switch_positions", [])
        active = config.get("_switches_on", set())
        barrier_cells = config.get("barrier_cells", {})
        n = len(switch_positions)
        ax, ay = pos

        # Find which switch (if any) the agent is standing on
        if (ax, ay) not in switch_positions:
            return

        switch_color = switch_positions.index((ax, ay))

        if switch_color in active:
            # Toggle OFF: close barrier[switch_color], open barrier[(switch_color+1)%n]
            active.discard(switch_color)
            # Restore the switch visual
            grid.objects[ay, ax] = ObjectType.SWITCH

            # Close barrier of this switch's color
            self._close_barrier(grid, switch_color, barrier_cells)

            # Open the complementary barrier: (switch_color + 1) % n
            comp = (switch_color + 1) % n
            self._open_barrier(grid, comp, barrier_cells)
        else:
            # Toggle ON: open barrier[switch_color], close barrier[(switch_color+1)%n]
            active.add(switch_color)
            # Visual: switch stays as SWITCH object (or remove it to show activated)
            grid.objects[ay, ax] = ObjectType.NONE

            # Open barrier of this switch's color
            self._open_barrier(grid, switch_color, barrier_cells)

            # Close the complementary barrier: (switch_color + 1) % n
            comp = (switch_color + 1) % n
            self._close_barrier(grid, comp, barrier_cells)
            # If complementary switch was ON, it is now blocked again but remains ON
            # (the barrier state is physical; the switch logical state is independent)

        config["_switches_on"] = active

    def _open_barrier(self, grid, color_idx, barrier_cells):
        """Remove all wall cells belonging to barrier color_idx."""
        key = color_idx
        cells = barrier_cells.get(key, barrier_cells.get(str(key), []))
        for cell in cells:
            cx, cy = cell[0], cell[1]
            grid.terrain[cy, cx] = CellType.EMPTY

    def _close_barrier(self, grid, color_idx, barrier_cells):
        """Restore all wall cells belonging to barrier color_idx."""
        key = color_idx
        cells = barrier_cells.get(key, barrier_cells.get(str(key), []))
        for cell in cells:
            cx, cy = cell[0], cell[1]
            grid.terrain[cy, cx] = CellType.WALL
            grid.metadata[cy, cx] = color_idx

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        config = new_state.get("config", {})
        new_n = len(config.get("_switches_on", set()))

        if new_n > self._last_n_switches:
            reward += 0.2 * (new_n - self._last_n_switches)
        elif new_n < self._last_n_switches:
            reward -= 0.1 * (self._last_n_switches - new_n)
        self._last_n_switches = new_n

        # Approach shaping: guide toward the next switch to activate or the goal
        if "agent" in new_state:
            ax, ay = new_state["agent"].position
            ox, oy = old_state.get("agent_position", (ax, ay))
            active = config.get("_switches_on", set())
            switch_positions = config.get("switch_positions", [])

            # Next switch = lowest color index not yet ON
            next_switch = None
            for i, spos in enumerate(switch_positions):
                if i not in active:
                    next_switch = spos
                    break

            if next_switch is not None:
                sx, sy = next_switch
                d_new = abs(ax - sx) + abs(ay - sy)
                d_old = abs(ox - sx) + abs(oy - sy)
                reward += 0.05 * (d_old - d_new)
            else:
                goal = config.get("goal_positions", [None])[0]
                if goal:
                    d_new = abs(ax - goal[0]) + abs(ay - goal[1])
                    d_old = abs(ox - goal[0]) + abs(oy - goal[1])
                    reward += 0.05 * (d_old - d_new)

        if self.check_success(new_state):
            reward += 1.0
        return reward

    def check_success(self, state):
        if "grid" not in state or "agent" not in state:
            return False
        x, y = state["agent"].position
        return bool(state["grid"].objects[y, x] == ObjectType.GOAL)

    def check_done(self, state):
        return self.check_success(state)

    def validate_instance(self, grid, config):
        """Verify the canonical solution path is feasible.

        Simulates toggling switches in order 0..N-1 on a copy of the grid,
        checking reachability at each step.
        """
        agent_pos = config.get("agent_start", (1, 1))
        switch_positions = config.get("switch_positions", [])
        goal_positions = config.get("goal_positions", [])
        barrier_cells = config.get("barrier_cells", {})
        n = len(switch_positions)

        if not switch_positions or not goal_positions:
            return True

        # Simulate the canonical order: toggle switch 0, then 1, ..., then n-1
        # Use a copied terrain to track open/closed barriers
        import copy

        sim_terrain = grid.terrain.copy()
        active = set()

        def open_barrier(color_idx):
            cells = barrier_cells.get(color_idx, barrier_cells.get(str(color_idx), []))
            for cell in cells:
                sim_terrain[cell[1], cell[0]] = int(CellType.EMPTY)

        def close_barrier(color_idx):
            cells = barrier_cells.get(color_idx, barrier_cells.get(str(color_idx), []))
            for cell in cells:
                sim_terrain[cell[1], cell[0]] = int(CellType.WALL)

        def reachable_from(start):
            """BFS on sim_terrain from start; returns set of (x,y)."""
            from collections import deque

            visited = {start}
            q = deque([start])
            size_h, size_w = sim_terrain.shape
            while q:
                cx, cy = q.popleft()
                for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < size_w and 0 <= ny < size_h:
                        if (nx, ny) not in visited:
                            t = int(sim_terrain[ny, nx])
                            if t not in (int(CellType.WALL), int(CellType.HAZARD)):
                                visited.add((nx, ny))
                                q.append((nx, ny))
            return visited

        current_pos = agent_pos
        for i in range(n):
            sx, sy = switch_positions[i]
            reachable = reachable_from(current_pos)
            if (sx, sy) not in reachable:
                return False
            # Toggle switch i ON
            active.add(i)
            open_barrier(i)
            comp = (i + 1) % n
            close_barrier(comp)
            current_pos = (sx, sy)

        # Check goal is reachable after all switches activated
        reachable = reachable_from(current_pos)
        goal_pos = goal_positions[0]
        return tuple(goal_pos) in reachable

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0
