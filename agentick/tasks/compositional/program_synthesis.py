"""ProgramSynthesis - Complete a pattern by pushing gems to target positions.

MECHANICS:
  - Fixed pattern blocks (SCROLL objects) show part of a geometric pattern
  - TARGET positions on the floor mark where gems must go to complete the pattern
  - Movable GEM objects must be PUSHED (Sokoban-style) by walking into them
  - Pushing a gem: gem slides 1 cell in push direction if the cell behind is
    empty and walkable; otherwise the push is blocked
  - Success = all TARGET positions have a GEM on them
  - No pick-up/drop: gems are only pushed, never carried
"""

from __future__ import annotations

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task

# Full pattern shapes as (dx, dy) offsets from the pattern center.
# The first portion of each pattern becomes fixed SCROLL blocks;
# the remaining positions are TARGET cells for gems.
_PATTERNS: dict[str, list[tuple[int, int]]] = {
    # Horizontal line: 4 positions total
    "line": [(0, 0), (1, 0), (2, 0), (3, 0)],
    # L-shape: 6 positions total
    "l_shape": [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (2, -1)],
    # T-shape: 8 positions total
    "t_shape": [(-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0), (0, 1), (0, 2), (0, -1)],
    # Cross/plus: 11 positions total
    "cross": [
        (0, 0),
        (1, 0),
        (-1, 0),
        (2, 0),
        (-2, 0),
        (0, 1),
        (0, -1),
        (0, 2),
        (0, -2),
        (1, 1),
        (-1, -1),
    ],
}

# Per-difficulty: (pattern_key, n_fixed, n_gems)
_DIFFICULTY_PATTERNS: dict[str, tuple[str, int, int]] = {
    "easy": ("line", 3, 1),
    "medium": ("l_shape", 4, 2),
    "hard": ("t_shape", 5, 3),
    "expert": ("cross", 7, 4),
}


@register_task("ProgramSynthesis-v0", tags=["reasoning", "planning", "abstraction"])
class ProgramSynthesisTask(TaskSpec):
    """Complete a geometric pattern by pushing gems onto target positions."""

    name = "ProgramSynthesis-v0"
    description = "Push gems to complete a visible geometric pattern"
    capability_tags = ["abstract_reasoning", "planning", "spatial"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=7,
            max_steps=120,
            params={"pattern": "line", "n_fixed": 3, "n_gems": 1},
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=9,
            max_steps=300,
            params={"pattern": "l_shape", "n_fixed": 4, "n_gems": 2},
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=11,
            max_steps=500,
            params={"pattern": "t_shape", "n_fixed": 5, "n_gems": 3},
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=13,
            max_steps=800,
            params={"pattern": "cross", "n_fixed": 7, "n_gems": 4},
        ),
    }

    # ── Generation ────────────────────────────────────────────────────────────

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        params = self.difficulty_config.params
        pattern_key = params.get("pattern", "line")
        n_fixed = params.get("n_fixed", 3)
        n_gems = params.get("n_gems", 1)

        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        agent_pos = (1, 1)

        full_shape = _PATTERNS.get(pattern_key, _PATTERNS["line"])

        # Compute the valid center ranges based on the shape's bounding box.
        # Interior cells: x in [1, size-2] inclusive.  Given offset d from center,
        # center must satisfy 1 <= cx+d <= size-2  →  1-d <= cx <= size-2-d.
        # Combined: cx in [1 - min_dx, size-2 - max_dx].
        min_dx = min(dx for dx, dy in full_shape)
        max_dx = max(dx for dx, dy in full_shape)
        min_dy = min(dy for dx, dy in full_shape)
        max_dy = max(dy for dx, dy in full_shape)

        cx_lo = 1 - min_dx
        cx_hi = (size - 2) - max_dx
        cy_lo = 1 - min_dy
        cy_hi = (size - 2) - max_dy

        pattern_cells: list[tuple[int, int]] = []

        if cx_lo <= cx_hi and cy_lo <= cy_hi:
            # There are valid centers — shuffle the shape to vary fixed vs. target
            # assignment across seeds, then pick a random valid center.
            shuffled_shape = list(full_shape)
            rng.shuffle(shuffled_shape)
            for _ in range(60):
                cx = int(rng.integers(cx_lo, cx_hi + 1))
                cy = int(rng.integers(cy_lo, cy_hi + 1))
                cells = [(cx + dx, cy + dy) for dx, dy in shuffled_shape]
                if agent_pos not in cells:
                    pattern_cells = cells
                    break
            if not pattern_cells:
                # Deterministic fallback: use mid-range center with original shape
                cx = (cx_lo + cx_hi) // 2
                cy = (cy_lo + cy_hi) // 2
                pattern_cells = [(cx + dx, cy + dy) for dx, dy in full_shape]

        if not pattern_cells:
            # Grid too small for this pattern — build a minimal horizontal line
            # from scratch that fits in the available interior.
            total_needed = n_fixed + n_gems
            row_y = size // 2
            start_x = 2
            pattern_cells = [
                (start_x + i, row_y)
                for i in range(total_needed)
                if start_x + i <= size - 2
            ]

        total_needed = n_fixed + n_gems
        # Ensure we have enough cells (extend rightward if necessary)
        if len(pattern_cells) < total_needed:
            rightmost = max(pattern_cells, key=lambda p: p[0])
            rx, ry = rightmost
            for ext in range(1, total_needed - len(pattern_cells) + 2):
                nx, ny = rx + ext, ry
                if 1 <= nx < size - 1 and (nx, ny) not in pattern_cells:
                    pattern_cells.append((nx, ny))
                if len(pattern_cells) >= total_needed:
                    break

        pattern_cells = pattern_cells[:total_needed]

        # First n_fixed become SCROLL (fixed blocks); remainder become TARGETs
        fixed_cells = pattern_cells[:n_fixed]
        target_cells = pattern_cells[n_fixed : n_fixed + n_gems]

        # Place fixed SCROLL blocks
        for x, y in fixed_cells:
            grid.objects[y, x] = ObjectType.SCROLL

        # Place TARGET markers where gems must land
        for x, y in target_cells:
            grid.objects[y, x] = ObjectType.TARGET

        # Build set of occupied cells so gem placement avoids them
        occupied = set(fixed_cells) | set(target_cells) | {agent_pos}
        # Fixed block set used for push-lane validation
        fixed_set = set(fixed_cells)

        def _push_lane_clear(gx, gy, tx, ty):
            """Return True if there is a clear straight push from (gx,gy) to (tx,ty).

            Clear means: same row or column, and every cell between gem and target
            (exclusive of gem, inclusive of target) is free of SCROLL/GEM objects.
            Also, the cell behind the agent (gem_pos - push_dir) must be reachable
            and not a wall (agent must be able to stand there to push).
            """
            if gx == tx:
                # Vertical push
                step = 1 if ty > gy else -1
                for cy in range(gy + step, ty + step, step):
                    if (gx, cy) in fixed_set:
                        return False
                # Agent push position: behind gem in push direction
                push_from_y = gy - step
                if not (0 <= push_from_y < size):
                    return False
                if grid.terrain[push_from_y, gx] == CellType.WALL:
                    return False
                return True
            elif gy == ty:
                # Horizontal push
                step = 1 if tx > gx else -1
                for cx in range(gx + step, tx + step, step):
                    if (cx, gy) in fixed_set:
                        return False
                push_from_x = gx - step
                if not (0 <= push_from_x < size):
                    return False
                if grid.terrain[gy, push_from_x] == CellType.WALL:
                    return False
                return True
            return False

        # Place GEM objects so that each can be pushed to its target via a
        # straight, clear lane.  Prefer placing the gem 1–3 cells away along
        # the same row or column as the target.
        gem_positions: list[tuple[int, int]] = []
        gem_target_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        for tx, ty in target_cells:
            placed = False
            # Build candidates: same row/column as target, various distances
            candidate_offsets = []
            for dx, dy in directions:
                for dist in range(1, size):
                    candidate_offsets.append((dx * dist, dy * dist))
            rng.shuffle(candidate_offsets)

            for cdx, cdy in candidate_offsets:
                gx, gy = tx + cdx, ty + cdy
                if (
                    1 <= gx < size - 1
                    and 1 <= gy < size - 1
                    and (gx, gy) not in occupied
                    and grid.terrain[gy, gx] == CellType.EMPTY
                    and _push_lane_clear(gx, gy, tx, ty)
                ):
                    gem_positions.append((gx, gy))
                    gem_target_pairs.append(((gx, gy), (tx, ty)))
                    occupied.add((gx, gy))
                    placed = True
                    break

            if not placed:
                # Fallback: try any interior cell that has a clear push lane to
                # the target (not necessarily on the same axis — just any cell
                # from which the gem can be pushed in multiple steps).
                # Use a 2-step approach: push to an intermediate cell then to target.
                for fy in range(1, size - 1):
                    for fx in range(1, size - 1):
                        if (
                            (fx, fy) not in occupied
                            and grid.terrain[fy, fx] == CellType.EMPTY
                            and (
                                # Same row or column as target with clear lane
                                (fx == tx and _push_lane_clear(fx, fy, tx, ty))
                                or (fy == ty and _push_lane_clear(fx, fy, tx, ty))
                            )
                        ):
                            gem_positions.append((fx, fy))
                            gem_target_pairs.append(((fx, fy), (tx, ty)))
                            occupied.add((fx, fy))
                            placed = True
                            break
                    if placed:
                        break

            if not placed:
                # Absolute last resort: any free interior cell (level may be harder)
                for fy in range(1, size - 1):
                    for fx in range(1, size - 1):
                        if (fx, fy) not in occupied and grid.terrain[fy, fx] == CellType.EMPTY:
                            gem_positions.append((fx, fy))
                            gem_target_pairs.append(((fx, fy), (tx, ty)))
                            occupied.add((fx, fy))
                            placed = True
                            break
                    if placed:
                        break

        for gx, gy in gem_positions:
            grid.objects[gy, gx] = ObjectType.GEM

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": list(target_cells),
            "fixed_cells": list(fixed_cells),
            "target_positions": list(target_cells),
            "gem_positions": list(gem_positions),
            "gem_target_pairs": [(list(g), list(t)) for g, t in gem_target_pairs],
            "n_gems": n_gems,
            "max_steps": self.get_max_steps(),
        }

    # ── Hooks ─────────────────────────────────────────────────────────────────

    def on_env_reset(self, agent, grid, config):
        """Initialise per-episode state and cache the config."""
        self._current_config = config
        self._gems_on_target = 0
        self._last_gems_on_target = 0
        # Track current gem positions for approach shaping
        self._last_agent_gem_dist = self._min_agent_gem_dist(agent.position, grid)

    def _min_agent_gem_dist(self, agent_pos, grid):
        """Manhattan distance from agent to nearest GEM not on a TARGET."""
        ax, ay = agent_pos
        targets = set(
            tuple(t) for t in self._current_config.get("target_positions", [])
        )
        min_d = None
        for y in range(grid.height):
            for x in range(grid.width):
                if grid.objects[y, x] == ObjectType.GEM and (x, y) not in targets:
                    d = abs(x - ax) + abs(y - ay)
                    if min_d is None or d < min_d:
                        min_d = d
        return min_d

    def _count_gems_on_targets(self, grid) -> int:
        """Count how many TARGET positions currently have a GEM on them."""
        targets = self._current_config.get("target_positions", [])
        count = 0
        for tx, ty in targets:
            if grid.objects[ty, tx] == ObjectType.GEM:
                count += 1
        return count

    def can_agent_enter(self, pos, agent, grid) -> bool:
        """Implement Sokoban-style gem pushing.

        When the agent tries to enter a cell containing a GEM, attempt to slide
        the gem one cell further in the same direction. If the destination is
        blocked (wall, another gem, or SCROLL), the agent cannot enter.
        """
        x, y = pos
        if grid.objects[y, x] != ObjectType.GEM:
            return True

        # Push direction: same as agent's direction of travel
        ax, ay = agent.position
        dx = x - ax
        dy = y - ay
        nx, ny = x + dx, y + dy

        # Destination must be in bounds, not a wall, and not occupied by
        # another GEM or SCROLL (fixed pattern block).
        if (
            0 <= nx < grid.width
            and 0 <= ny < grid.height
            and grid.terrain[ny, nx] not in (CellType.WALL,)
            and grid.objects[ny, nx] not in (ObjectType.GEM, ObjectType.SCROLL)
        ):
            # Clear gem from old cell; restore TARGET marker if it was there
            grid.objects[y, x] = ObjectType.NONE
            target_positions = [
                tuple(t) for t in self._current_config.get("target_positions", [])
            ]
            if (x, y) in target_positions:
                grid.objects[y, x] = ObjectType.TARGET

            # Place gem at new cell; it may land on a TARGET (shown as GEM
            # overlaid on TARGET — check_success reads GEM at TARGET position)
            grid.objects[ny, nx] = ObjectType.GEM

            return True  # agent enters the now-vacated cell

        return False  # push blocked

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01

        if "grid" not in new_state:
            return reward

        grid = new_state["grid"]

        # Bonus when a gem newly lands on a target
        current_on_target = self._count_gems_on_targets(grid)
        if current_on_target > self._last_gems_on_target:
            reward += 0.3 * (current_on_target - self._last_gems_on_target)
        self._last_gems_on_target = current_on_target

        # Approach shaping: reward agent for getting closer to the nearest
        # unpushed gem
        if "agent" in new_state:
            agent_pos = new_state["agent"].position
            new_dist = self._min_agent_gem_dist(agent_pos, grid)
            if new_dist is not None and self._last_agent_gem_dist is not None:
                reward += 0.05 * (self._last_agent_gem_dist - new_dist)
            self._last_agent_gem_dist = new_dist

        if self.check_success(new_state):
            reward += 1.0

        return reward

    def check_success(self, state) -> bool:
        """All TARGET positions must have a GEM on them."""
        if "grid" not in state:
            return False
        grid = state["grid"]
        fallback = self._current_config if hasattr(self, "_current_config") else {}
        config = state.get("config", fallback)
        targets = config.get("target_positions", [])
        if not targets:
            return False
        return all(grid.objects[ty, tx] == ObjectType.GEM for tx, ty in targets)

    def check_done(self, state) -> bool:
        return self.check_success(state)

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0
