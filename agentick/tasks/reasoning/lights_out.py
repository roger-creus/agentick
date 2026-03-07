"""LightsOut - Toggle lights to turn them all off.

MECHANICS:
  - N lights (SWITCH objects) placed on the grid
  - Stepping ON a SWITCH toggles it (on→off, off→on)
  - Adjacent lights also toggle (classic Lights Out puzzle)
  - Success = ALL lights are off (no SWITCH objects remain on grid)
  - Agent can move freely; toggles happen by stepping
"""

from __future__ import annotations

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task

# Minimum solution toggles required per difficulty to reject trivial puzzles.
_MIN_SOLUTION_TOGGLES: dict[str, int] = {
    "easy": 2,
    "medium": 4,
    "hard": 6,
    "expert": 8,
}


@register_task("LightsOut-v0", tags=["combinatorial_logic"])
class LightsOutTask(TaskSpec):
    """Toggle all lights off by walking — classic Lights Out puzzle.

    Unlike other tasks, LightsOut switches are walkable. Every time the agent
    steps on a cell, all switches within Manhattan distance 1 (including the
    cell itself) are automatically toggled. No INTERACT needed.
    """

    name = "LightsOut-v0"
    description = "Toggle all lights off"
    capability_tags = ["combinatorial_logic"]

    difficulty_configs = {
        # All levels use adjacent toggle (classic Lights Out mechanic):
        # stepping on any cell toggles it + its 4 neighbors.
        # n_lights: initially-lit switches to turn off.
        # puzzle_size: side length of the NxN light grid embedded in the room.
        # n_decoys: extra isolated switches that add noise but must also be toggled.
        # n_walls: interior obstacles creating longer routing decisions.
        "easy": DifficultyConfig(
            name="easy",
            grid_size=7,
            max_steps=80,
            params={
                "n_lights": 3, "adjacent": True, "n_decoys": 0,
                "n_walls": 0, "puzzle_size": 3,
            },
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=9,
            max_steps=150,
            params={"n_lights": 5, "adjacent": True, "n_decoys": 1, "n_walls": 2, "puzzle_size": 4},
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=11,
            max_steps=250,
            params={"n_lights": 8, "adjacent": True, "n_decoys": 2, "n_walls": 3, "puzzle_size": 4},
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=13,
            max_steps=400,
            params={
                "n_lights": 10, "adjacent": True, "n_decoys": 3,
                "n_walls": 4, "puzzle_size": 5,
            },
        ),
    }

    @staticmethod
    def _count_solution_toggles(
        puzzle_cells: list[tuple[int, int]],
        light_state: dict[tuple[int, int], bool],
        adjacent: bool,
    ) -> int:
        """Return the minimum number of toggles needed to solve the puzzle.

        For non-adjacent (simple) mode each lit cell needs exactly one toggle,
        so the answer is just the number of lit cells.

        For adjacent mode this is the classic Lights Out problem: build the
        toggle matrix over GF(2) and solve via Gaussian elimination to find
        the solution with the fewest 1-bits.  Because every reachable state
        has a *unique* solution (the toggle matrix for an NxN Lights Out grid
        is always full-rank for N <= 5), we just count the 1s in that solution.
        """
        if not adjacent:
            return sum(1 for v in light_state.values() if v)

        n = len(puzzle_cells)
        if n == 0:
            return 0

        idx = {pos: i for i, pos in enumerate(puzzle_cells)}

        # Build the n x n toggle matrix over GF(2).
        # mat[i][j] = 1 iff toggling cell j affects cell i.
        mat = np.zeros((n, n), dtype=np.int8)
        for j, (cx, cy) in enumerate(puzzle_cells):
            neighbors = [
                (cx, cy), (cx + 1, cy), (cx - 1, cy),
                (cx, cy + 1), (cx, cy - 1),
            ]
            for affected in neighbors:
                if affected in idx:
                    mat[idx[affected], j] = 1

        # Build the target vector b: 1 for each lit cell.
        b = np.array(
            [int(light_state.get(pos, False)) for pos in puzzle_cells],
            dtype=np.int8,
        )

        # Gaussian elimination over GF(2) on the augmented matrix.
        aug = np.concatenate([mat, b.reshape(-1, 1)], axis=1)
        pivot_col = 0
        for row in range(n):
            if pivot_col >= n:
                break
            # Find a pivot in this column.
            found = -1
            for k in range(row, n):
                if aug[k, pivot_col]:
                    found = k
                    break
            if found == -1:
                pivot_col += 1
                continue
            if found != row:
                aug[[row, found]] = aug[[found, row]]
            for k in range(n):
                if k != row and aug[k, pivot_col]:
                    aug[k] = (aug[k] ^ aug[row])
            pivot_col += 1

        # The solution vector is the last column (after elimination).
        x = aug[:, -1]
        return int(np.sum(x))

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        n_lights = self.difficulty_config.params.get("n_lights", 3)
        n_decoys = self.difficulty_config.params.get("n_decoys", 0)
        n_walls = self.difficulty_config.params.get("n_walls", 0)
        adjacent = self.difficulty_config.params.get("adjacent", True)
        puzzle_size = self.difficulty_config.params.get("puzzle_size", 3)

        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        # Place agent in top-left area, away from the puzzle grid
        agent_pos = (1, 1)

        # Build a structured NxN light grid in the center of the room
        # Offset so puzzle is centered
        p_off_x = max(1, (size - puzzle_size) // 2)
        p_off_y = max(1, (size - puzzle_size) // 2)
        # Shift puzzle away from agent start if possible
        if p_off_x <= 2:
            p_off_x = min(3, size - puzzle_size - 1)
        if p_off_y <= 2:
            p_off_y = min(3, size - puzzle_size - 1)

        # All puzzle cell positions
        puzzle_cells = [
            (p_off_x + px, p_off_y + py)
            for py in range(puzzle_size)
            for px in range(puzzle_size)
            if 1 <= p_off_x + px < size - 1 and 1 <= p_off_y + py < size - 1
            and (p_off_x + px, p_off_y + py) != agent_pos
        ]

        # Generate a valid solvable puzzle: start with all-off, then apply
        # random toggle moves to create an on-pattern that we know is solvable.
        # Reject trivially easy puzzles that fall below the minimum solution
        # toggle threshold for this difficulty level.
        min_toggles = _MIN_SOLUTION_TOGGLES.get(self.difficulty_config.name, 2)
        n_lit = min(n_lights, len(puzzle_cells))
        max_attempts = 50  # safety cap to avoid infinite loops

        for _attempt in range(max_attempts):
            light_state = {pos: False for pos in puzzle_cells}

            if adjacent:
                # Generate solvable puzzle by applying random moves to all-off state
                # Each move toggles a cell + neighbors — this ensures solvability
                n_moves = max(n_lit, int(rng.integers(n_lit, n_lit * 2 + 1)))
                toggle_positions = list(puzzle_cells)
                for _ in range(n_moves):
                    toggle_pos = toggle_positions[int(rng.integers(len(toggle_positions)))]
                    tx, ty = toggle_pos
                    for affected in [
                        (tx, ty), (tx + 1, ty), (tx - 1, ty), (tx, ty + 1), (tx, ty - 1),
                    ]:
                        if affected in light_state:
                            light_state[affected] = not light_state[affected]
                # Ensure at least n_lit lights are on; if too few, toggle more
                lit_count = sum(1 for v in light_state.values() if v)
                if lit_count == 0:
                    # Force-light some cells
                    for pos in rng.choice(len(puzzle_cells), size=n_lit, replace=False):
                        light_state[puzzle_cells[pos]] = True
            else:
                # Simple mode: randomly pick n_lit cells to be on
                on_cells = [
                    puzzle_cells[i]
                    for i in rng.choice(len(puzzle_cells), size=n_lit, replace=False)
                ]
                for pos in on_cells:
                    light_state[pos] = True

            light_positions = [pos for pos, on in light_state.items() if on]

            # If no lights ended up on, turn some on
            if not light_positions:
                for pos in puzzle_cells[:n_lit]:
                    light_state[pos] = True
                light_positions = puzzle_cells[:n_lit]

            # Check minimum solution complexity
            sol_toggles = self._count_solution_toggles(
                puzzle_cells, light_state, adjacent,
            )
            if sol_toggles >= min_toggles:
                break  # puzzle is complex enough

        used = set(puzzle_cells) | {agent_pos}

        # Decoy SWITCHes outside the puzzle area — agent must toggle all SWITCHes to win
        free = [
            (x, y) for x in range(1, size - 1) for y in range(1, size - 1)
            if (x, y) not in used and (x, y) != agent_pos
        ]
        rng.shuffle(free)
        decoy_positions = []
        for p in free:
            if len(decoy_positions) >= n_decoys:
                break
            decoy_positions.append(p)
            used.add(p)

        # Interior walls — flood-fill to keep all lights reachable
        wall_positions = []
        wall_candidates = [p for p in free if p not in used]
        all_lights = list(puzzle_cells) + decoy_positions
        for p in wall_candidates:
            if len(wall_positions) >= n_walls:
                break
            wx, wy = p
            grid.terrain[wy, wx] = CellType.WALL
            reachable = grid.flood_fill(agent_pos)
            if all(lp in reachable for lp in all_lights):
                wall_positions.append(p)
                used.add(p)
            else:
                grid.terrain[wy, wx] = CellType.EMPTY

        # Place SWITCH at ALL light positions; metadata distinguishes on/off
        # meta=1 (META_LIT) = on/lit, meta=2 (META_LIGHT_POS) = off/unlit
        for pos in puzzle_cells:
            lx, ly = pos
            grid.objects[ly, lx] = ObjectType.SWITCH
            if light_state.get(pos, False):
                grid.metadata[ly, lx] = 1  # lit (on)
            else:
                grid.metadata[ly, lx] = 2  # unlit (off)
        for dx, dy in decoy_positions:
            grid.objects[dy, dx] = ObjectType.SWITCH
            grid.metadata[dy, dx] = 1  # decoys start lit (on)

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [],
            "light_positions": light_positions,
            "decoy_positions": decoy_positions,
            "puzzle_cells": puzzle_cells,  # all cells in the puzzle grid
            "adjacent_toggle": adjacent,
            "max_steps": self.get_max_steps(),
        }

    def can_agent_enter(self, pos, agent, grid) -> bool:
        """All cells are walkable in LightsOut (switches are stepped on)."""
        return True

    # ── Toggle mechanic ───────────────────────────────────────────────────────

    def on_env_reset(self, agent, grid, config):
        """Cache config and count initial lights for reward tracking."""
        self._adjacent_toggle = config.get("adjacent_toggle", False)
        # Build set of all light grid positions (lights + decoys form the toggle grid)
        self._light_grid = set()
        for lx, ly in config.get("light_positions", []):
            self._light_grid.add((lx, ly))
        for dx, dy in config.get("decoy_positions", []):
            self._light_grid.add((dx, dy))
        # Also include all puzzle cells for adjacent toggle mode
        for px, py in config.get("puzzle_cells", []):
            self._light_grid.add((px, py))
        # Count lit lights by metadata (meta==1 means lit/on)
        self._lights_remaining = sum(
            1
            for y in range(grid.height)
            for x in range(grid.width)
            if grid.objects[y, x] == ObjectType.SWITCH
            and int(grid.metadata[y, x]) == 1
        )
        self._lights_remaining_last = self._lights_remaining

    def on_agent_moved(self, pos, agent, grid):
        """Auto-toggle switches when the agent steps on any cell.

        Classic Lights Out: stepping on a cell toggles the cell itself (if a
        switch) AND all adjacent switches. In easy (non-adjacent) mode, only
        the cell the agent steps on is toggled.
        """
        x, y = pos
        self._toggle_cell(x, y, grid)

        # Adjacent toggle mode (medium+): also toggle 4 cardinal neighbors
        if self._adjacent_toggle:
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if 0 < nx < grid.width - 1 and 0 < ny < grid.height - 1:
                    self._toggle_cell(nx, ny, grid)

    def _toggle_cell(self, x, y, grid):
        """Toggle a single cell: ON→OFF or OFF→ON via metadata, SWITCH always stays."""
        if (x, y) not in self._light_grid:
            return
        if grid.objects[y, x] != ObjectType.SWITCH:
            return
        meta = int(grid.metadata[y, x])
        if meta == 1:
            # ON → OFF
            grid.metadata[y, x] = 2
            self._lights_remaining -= 1
        elif meta == 2:
            # OFF → ON
            grid.metadata[y, x] = 1
            self._lights_remaining += 1

    # ── Reward & success ─────────────────────────────────────────────────────

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        # Lights toggled in on_agent_moved — use instance counter
        old_rem = self._lights_remaining_last
        new_rem = self._lights_remaining
        if new_rem < old_rem:
            reward += 0.3 * (old_rem - new_rem)  # reward per light turned off
        self._lights_remaining_last = new_rem
        # Approach shaping: toward nearest remaining lit SWITCH
        if "agent_position" in new_state and "grid" in new_state:
            ax, ay = new_state["agent_position"]
            ox, oy = old_state.get("agent_position", (ax, ay))
            g = new_state["grid"]
            lights = [
                (x, y)
                for y in range(g.height)
                for x in range(g.width)
                if g.objects[y, x] == ObjectType.SWITCH
                and int(g.metadata[y, x]) == 1
            ]
            if lights:
                d_new = min(abs(ax - lx) + abs(ay - ly) for lx, ly in lights)
                d_old = min(abs(ox - lx) + abs(oy - ly) for lx, ly in lights)
                reward += 0.05 * (d_old - d_new)
        if self.check_success(new_state):
            reward += 1.0
        return reward

    def check_success(self, state):
        """All lights must be off (no cell with metadata==1 on grid)."""
        if "grid" not in state:
            return False
        grid = state["grid"]
        for y in range(grid.height):
            for x in range(grid.width):
                if (
                    grid.objects[y, x] == ObjectType.SWITCH
                    and int(grid.metadata[y, x]) == 1
                ):
                    return False
        return True

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0
