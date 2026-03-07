"""SymbolMatching - Match visually distinct colored paired symbols.

MECHANICS:
  - N pairs of symbols using 6 different object types (GEM, POTION, SCROLL, COIN, ORB, LEVER)
  - Each pair has one item (left side) and one matching target (right side), SAME ObjectType
  - The matching is purely visual: same character/sprite = same pair
  - Agent auto-picks up an item by walking onto it (one at a time)
  - Agent delivers to the matching target (same ObjectType) on the other side
  - Placing on wrong type = mismatch penalty, item lost
  - Fake items use types with no matching target (visually distinct, no pair)
  - Metadata tracks match state: <100 = unmatched target, >=100 = matched (old_meta + 100)
  - Success = all pairs matched correctly
"""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task

# Symbol types available for matching pairs
_SYMBOL_TYPES = [
    ObjectType.GEM,
    ObjectType.POTION,
    ObjectType.SCROLL,
    ObjectType.COIN,
    ObjectType.ORB,
]


@register_task("SymbolMatching-v0", tags=["reasoning", "pattern_recognition"])
class SymbolMatchingTask(TaskSpec):
    """Pick up symbol items and deliver to matching targets (same ObjectType) on the other side."""

    name = "SymbolMatching-v0"
    description = "Match symbol items to their matching targets by type"
    capability_tags = ["reasoning", "pattern_recognition"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=7,
            max_steps=100,
            params={"n_pairs": 2, "n_fakes": 0, "n_obstacles": 0},
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=10,
            max_steps=180,
            params={"n_pairs": 3, "n_fakes": 1, "n_obstacles": 3},
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=13,
            max_steps=300,
            params={"n_pairs": 4, "n_fakes": 2, "n_obstacles": 5},
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=15,
            max_steps=500,
            params={"n_pairs": 5, "n_fakes": 3, "n_obstacles": 8},
        ),
    }

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        n = self.difficulty_config.params.get("n_pairs", 2)
        n_fakes = self.difficulty_config.params.get("n_fakes", 0)
        n_obstacles = self.difficulty_config.params.get("n_obstacles", 0)

        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        # Agent starts in the middle
        mid_x = size // 2
        agent_pos = (mid_x, 1)

        # Divide the interior into left and right halves for items and targets
        # Left half: columns 1..mid_x-1  (items to pick up)
        # Right half: columns mid_x+1..size-2  (targets to deliver to)
        left_cells = [
            (x, y)
            for x in range(1, mid_x)
            for y in range(1, size - 1)
            if (x, y) != agent_pos
        ]
        right_cells = [
            (x, y)
            for x in range(mid_x + 1, size - 1)
            for y in range(1, size - 1)
            if (x, y) != agent_pos
        ]
        rng.shuffle(left_cells)
        rng.shuffle(right_cells)

        # Choose n different symbol types for the pairs
        n_types = min(n, len(_SYMBOL_TYPES))
        chosen_indices = list(range(len(_SYMBOL_TYPES)))
        rng.shuffle(chosen_indices)
        pair_types = [_SYMBOL_TYPES[i] for i in chosen_indices[:n_types]]
        # If n > len(_SYMBOL_TYPES), reuse types (shouldn't happen with max 5 pairs)
        while len(pair_types) < n:
            pair_types.append(_SYMBOL_TYPES[int(rng.integers(len(_SYMBOL_TYPES)))])

        # Place items on left, targets on right - both using the SAME ObjectType
        item_positions = left_cells[:n]
        target_positions = right_cells[:n]
        used = {agent_pos} | set(item_positions) | set(target_positions)

        pair_info = []
        target_pos_set = set()
        for i in range(n):
            ix, iy = item_positions[i]
            tx, ty = target_positions[i]
            sym_type = pair_types[i]

            # Place item as its symbol type (left side)
            grid.objects[iy, ix] = sym_type

            # Place target as the SAME symbol type (right side)
            # metadata=0 means unmatched target
            grid.objects[ty, tx] = sym_type
            grid.metadata[ty, tx] = 0

            target_pos_set.add((tx, ty))
            pair_info.append(
                {
                    "item_pos": list(item_positions[i]),
                    "target_pos": list(target_positions[i]),
                    "symbol_type": int(sym_type),
                }
            )

        # Fake items: use symbol types NOT in the current pair set
        # They are visually distinct because no matching target exists for them
        fake_positions = []
        remaining_left = [p for p in left_cells[n:] if p not in used]
        unused_types = [t for t in _SYMBOL_TYPES if t not in pair_types]
        for i in range(min(n_fakes, len(remaining_left), len(unused_types))):
            fx, fy = remaining_left[i]
            fake_type = unused_types[i % len(unused_types)]
            grid.objects[fy, fx] = fake_type
            fake_positions.append(remaining_left[i])
            used.add(remaining_left[i])

        # Obstacle walls with flood-fill reachability check
        wall_positions = []
        all_free = [
            (x, y)
            for x in range(1, size - 1)
            for y in range(1, size - 1)
            if (x, y) not in used
        ]
        rng.shuffle(all_free)
        critical = [agent_pos] + list(item_positions) + list(target_positions)
        for p in all_free:
            if len(wall_positions) >= n_obstacles:
                break
            wx, wy = p
            grid.terrain[wy, wx] = CellType.WALL
            reachable = grid.flood_fill(agent_pos)
            if all(q in reachable for q in critical):
                wall_positions.append(p)
                used.add(p)
            else:
                grid.terrain[wy, wx] = CellType.EMPTY

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": target_positions,
            "pair_info": pair_info,
            "fake_positions": fake_positions,
            "target_positions": [list(p) for p in target_positions],
            "n_pairs": n,
            "max_steps": self.get_max_steps(),
        }

    def on_env_reset(self, agent, grid, config):
        config["_items_placed"] = 0
        config["_carrying"] = None
        config["_mismatches"] = 0
        self._items_placed = 0
        self._carrying = None  # ObjectType int or None
        self._last_placed_for_reward = 0
        # Build a set of target positions for quick lookup
        self._target_pos_set = set()
        for p in config.get("pair_info", []):
            self._target_pos_set.add(tuple(p["target_pos"]))

    def on_agent_moved(self, pos, agent, grid):
        """Auto-pickup symbol item / deliver to matching target (same ObjectType)."""
        x, y = pos
        obj = int(grid.objects[y, x])

        if obj == ObjectType.NONE or grid.metadata[y, x] >= 100:
            return

        is_target = (x, y) in self._target_pos_set

        if self._carrying is None and not is_target and obj in [int(t) for t in _SYMBOL_TYPES]:
            # Pick up an item (not a target)
            grid.objects[y, x] = ObjectType.NONE
            self._carrying = obj

        elif self._carrying is not None and is_target and obj in [int(t) for t in _SYMBOL_TYPES]:
            # Attempting to deliver to a target
            if self._carrying == obj:
                # Correct match: same ObjectType = visual match
                # Remove matched target from grid (pair disappears)
                grid.objects[y, x] = ObjectType.NONE
                grid.metadata[y, x] = 0
                self._target_pos_set.discard((x, y))
                self._carrying = None
                self._items_placed += 1
            else:
                # Wrong match: mismatch penalty, item lost
                self._carrying = None
                config = getattr(self, "_config", {})
                if config:
                    config["_mismatches"] = config.get("_mismatches", 0) + 1

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        old_placed = getattr(self, "_last_placed_for_reward", 0)
        new_placed = self._items_placed
        if new_placed > old_placed:
            reward += 0.5
        self._last_placed_for_reward = new_placed

        # Approach shaping
        if "agent" in new_state and "grid" in new_state:
            ax, ay = new_state["agent"].position
            ox, oy = old_state.get("agent_position", (ax, ay))
            g = new_state["grid"]
            if self._carrying is None:
                # Guide toward nearest uncollected symbol item (not targets, not matched)
                items = [
                    (cx, cy)
                    for cy in range(g.height)
                    for cx in range(g.width)
                    if int(g.objects[cy, cx]) in [int(t) for t in _SYMBOL_TYPES]
                    and (cx, cy) not in self._target_pos_set
                    and int(g.metadata[cy, cx]) < 100
                ]
            else:
                # Guide toward the matching target (same ObjectType, still unmatched)
                items = [
                    (cx, cy)
                    for cy in range(g.height)
                    for cx in range(g.width)
                    if (cx, cy) in self._target_pos_set
                    and int(g.objects[cy, cx]) == self._carrying
                ]
            if items:
                d_new = min(abs(ax - ix) + abs(ay - iy) for ix, iy in items)
                d_old = min(abs(ox - ix) + abs(oy - iy) for ix, iy in items)
                reward += 0.05 * (d_old - d_new)
        if self.check_success(new_state):
            reward += 1.0
        return reward

    def check_success(self, state):
        config = state.get("config", {})
        n = config.get("n_pairs", 1)
        return self._items_placed >= n

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0
