"""ProgramSynthesis - Replicate a reference pattern by pushing gems.

MECHANICS:
  - Reference pattern: N SCROLL objects arranged in a specific shape (immovable)
  - Movable gems: N GEM objects scattered around the map
  - Agent must push GEMs (Sokoban-style) so they form the same relative pattern
    as the SCROLLs — anywhere on the map (translation-invariant)
  - Pushing a gem: gem slides 1 cell in push direction if destination is
    empty and walkable; otherwise the push is blocked
  - Success = GEM positions form the same normalized shape as SCROLL positions
  - No pick-up/drop: gems are only pushed, never carried
  - Agent must INFER the pattern from observing the reference group
"""

from __future__ import annotations

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task

# Pattern shapes as (dx, dy) offsets from the first position.
_PATTERNS: dict[str, list[tuple[int, int]]] = {
    "line": [(0, 0), (1, 0), (2, 0)],
    "l_shape": [(0, 0), (1, 0), (2, 0), (2, 1)],
    "t_shape": [(0, 0), (1, 0), (2, 0), (1, 1), (1, -1)],
    "cross": [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1)],
}


@register_task("ProgramSynthesis-v0", tags=["reasoning", "planning", "abstraction"])
class ProgramSynthesisTask(TaskSpec):
    """Replicate a reference pattern by pushing gems to matching positions."""

    name = "ProgramSynthesis-v0"
    description = "Push gems to replicate a reference shape pattern"
    capability_tags = ["abstract_reasoning", "planning", "spatial"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=9,
            max_steps=100,
            params={"pattern": "line", "n_gems": 3},
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=11,
            max_steps=200,
            params={"pattern": "l_shape", "n_gems": 4},
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=13,
            max_steps=350,
            params={"pattern": "t_shape", "n_gems": 5},
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=15,
            max_steps=500,
            params={"pattern": "cross", "n_gems": 6},
        ),
    }

    @staticmethod
    def _normalize_offsets(positions):
        """Compute sorted normalized offsets from the lexicographically first pos."""
        if not positions:
            return []
        pts = sorted(positions)
        x0, y0 = pts[0]
        return sorted((x - x0, y - y0) for x, y in pts)

    def _has_valid_target(self, gem_positions, shape, grid):
        """Check if at least one anchor exists where all pattern offsets land on valid cells."""
        size = grid.width
        offsets = self._normalize_offsets([(dx, dy) for dx, dy in shape])
        # Try every interior position as a potential anchor
        for ax in range(2, size - 2):
            for ay in range(2, size - 2):
                valid = True
                for dx, dy in offsets:
                    tx, ty = ax + dx, ay + dy
                    if not (2 <= tx <= size - 3 and 2 <= ty <= size - 3):
                        valid = False
                        break
                    if grid.terrain[ty, tx] == CellType.WALL:
                        valid = False
                        break
                    if grid.objects[ty, tx] == ObjectType.SCROLL:
                        valid = False
                        break
                if valid:
                    return True
        return False

    def generate(self, seed):
        for attempt in range(50):
            result = self._try_generate(seed + attempt * 1000)
            if result is not None:
                return result
        # Last resort: generate without validation
        return self._try_generate(seed, validate=False)

    def _try_generate(self, seed, validate=True):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        params = self.difficulty_config.params
        pattern_key = params.get("pattern", "line")
        n_gems = params.get("n_gems", 3)

        shape = _PATTERNS.get(pattern_key, _PATTERNS["line"])
        # Trim shape to n_gems if needed
        shape = shape[:n_gems]

        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        corners = [(1, 1), (size - 2, 1), (1, size - 2), (size - 2, size - 2)]
        agent_pos = tuple(corners[int(rng.integers(0, len(corners)))])

        # Place reference SCROLL pattern in left area of grid
        for _attempt in range(60):
            # Random center for reference pattern (left half)
            cx = int(rng.integers(2, max(3, size // 2)))
            cy = int(rng.integers(2, size - 2))
            cells = [(cx + dx, cy + dy) for dx, dy in shape]
            # Check all cells are valid interior positions
            if all(1 <= x < size - 1 and 1 <= y < size - 1 for x, y in cells):
                if agent_pos not in cells:
                    break
        else:
            # Fallback: force a position
            cx, cy = 2, size // 2
            cells = [(cx + dx, cy + dy) for dx, dy in shape]

        scroll_positions = []
        for x, y in cells:
            if 1 <= x < size - 1 and 1 <= y < size - 1:
                grid.objects[y, x] = ObjectType.SCROLL
                scroll_positions.append((x, y))

        # Compute reference offsets for success checking
        reference_offsets = self._normalize_offsets(scroll_positions)

        # Place GEM objects in right area of grid (scattered)
        occupied = set(scroll_positions) | {agent_pos}
        gem_positions = []
        # Prefer right half for gems
        gem_candidates = [
            (x, y)
            for x in range(max(size // 2, 3), size - 2)
            for y in range(2, size - 2)
            if (x, y) not in occupied and grid.terrain[y, x] == CellType.EMPTY
        ]
        if len(gem_candidates) < n_gems:
            gem_candidates = [
                (x, y)
                for x in range(2, size - 2)
                for y in range(2, size - 2)
                if (x, y) not in occupied and grid.terrain[y, x] == CellType.EMPTY
            ]
        rng.shuffle(gem_candidates)
        for gx, gy in gem_candidates[:n_gems]:
            grid.objects[gy, gx] = ObjectType.GEM
            gem_positions.append((gx, gy))
            occupied.add((gx, gy))

        # Validate that at least one valid target configuration exists
        if validate:
            gem_positions_list = list(gem_positions)
            if not self._has_valid_target(gem_positions_list, shape, grid):
                return None

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [],
            "scroll_positions": scroll_positions,
            "gem_positions": gem_positions,
            "reference_offsets": reference_offsets,
            "n_gems": len(scroll_positions),
            "max_steps": self.get_max_steps(),
        }

    def on_env_reset(self, agent, grid, config):
        self._current_config = config
        self._last_match_count = 0

    def _find_gem_positions(self, grid):
        """Find all current GEM positions on the grid."""
        gems = []
        for y in range(grid.height):
            for x in range(grid.width):
                if grid.objects[y, x] == ObjectType.GEM:
                    gems.append((x, y))
        return gems

    def _count_matching_gems(self, grid):
        """Count gems that contribute to a partial pattern match."""
        gem_positions = self._find_gem_positions(grid)
        ref = self._current_config.get("reference_offsets", [])
        if not gem_positions or not ref:
            return 0

        # Try each gem as anchor and count max matching offsets
        best = 0
        for anchor in gem_positions:
            offsets = set()
            for gx, gy in gem_positions:
                offsets.add((gx - anchor[0], gy - anchor[1]))
            matched = sum(1 for o in ref if o in offsets)
            best = max(best, matched)
        return best

    def can_agent_enter(self, pos, agent, grid) -> bool:
        """Sokoban-style gem pushing."""
        x, y = pos
        if grid.objects[y, x] != ObjectType.GEM:
            return True

        ax, ay = agent.position
        dx = x - ax
        dy = y - ay
        nx, ny = x + dx, y + dy

        if (
            0 <= nx < grid.width
            and 0 <= ny < grid.height
            and grid.terrain[ny, nx] not in (CellType.WALL,)
            and grid.objects[ny, nx] not in (ObjectType.GEM, ObjectType.SCROLL)
        ):
            grid.objects[y, x] = ObjectType.NONE
            grid.objects[ny, nx] = ObjectType.GEM
            return True
        return False

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        if "grid" not in new_state:
            return reward

        grid = new_state["grid"]
        current_match = self._count_matching_gems(grid)
        if current_match > self._last_match_count:
            reward += 0.2 * (current_match - self._last_match_count)
        self._last_match_count = current_match

        if self.check_success(new_state):
            reward += 1.0
        return reward

    def check_success(self, state) -> bool:
        """GEM positions must form the same normalized pattern as SCROLLs."""
        if "grid" not in state:
            return False
        grid = state["grid"]
        config = state.get("config", getattr(self, "_current_config", {}))
        ref = config.get("reference_offsets", [])
        if not ref:
            return False

        gem_positions = self._find_gem_positions(grid)
        if len(gem_positions) != len(ref):
            return False

        gem_offsets = self._normalize_offsets(gem_positions)
        return gem_offsets == ref

    def check_done(self, state) -> bool:
        return self.check_success(state)

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0
