"""SymbolMatching - Match diverse symbol pairs by carrying items to matching targets.

MECHANICS:
  - N pairs of symbols using 6 different object types (GEM, POTION, SCROLL, COIN, ORB, LEVER)
  - Each symbol type has one item and one matching target zone
  - Agent picks up one item at a time, carries it to the matching target
  - Must match by symbol type: GEM→GEM target, POTION→POTION target, etc.
  - Placing wrong type on target = penalty (target remains)
  - Fake items (wrong type with no matching target) add visual noise
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
    ObjectType.LEVER,
]


@register_task("SymbolMatching-v0", tags=["reasoning", "pattern_recognition"])
class SymbolMatchingTask(TaskSpec):
    """Match diverse symbol items to their corresponding target zones."""

    name = "SymbolMatching-v0"
    description = "Match symbol items to their target zones by type"
    capability_tags = ["reasoning", "pattern_recognition"]

    difficulty_configs = {
        "easy":   DifficultyConfig(
            name="easy", grid_size=7, max_steps=100,
            params={"n_pairs": 2, "n_fakes": 0, "n_obstacles": 0},
        ),
        "medium": DifficultyConfig(
            name="medium", grid_size=10, max_steps=180,
            params={"n_pairs": 3, "n_fakes": 1, "n_obstacles": 3},
        ),
        "hard":   DifficultyConfig(
            name="hard", grid_size=13, max_steps=300,
            params={"n_pairs": 4, "n_fakes": 2, "n_obstacles": 5},
        ),
        "expert": DifficultyConfig(
            name="expert", grid_size=15, max_steps=500,
            params={"n_pairs": 5, "n_fakes": 3, "n_obstacles": 8},
        ),
    }

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size        = self.difficulty_config.grid_size
        n           = self.difficulty_config.params.get("n_pairs", 2)
        n_fakes     = self.difficulty_config.params.get("n_fakes", 0)
        n_obstacles = self.difficulty_config.params.get("n_obstacles", 0)

        grid = Grid(size, size)
        grid.terrain[0, :]  = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0]  = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        agent_pos = (1, 1)

        free = [(x, y) for x in range(1, size - 1) for y in range(1, size - 1)
                if (x, y) != agent_pos]
        rng.shuffle(free)

        # Choose n different symbol types for the pairs
        n_types = min(n, len(_SYMBOL_TYPES))
        chosen_indices = list(range(len(_SYMBOL_TYPES)))
        rng.shuffle(chosen_indices)
        pair_types = [_SYMBOL_TYPES[i] for i in chosen_indices[:n_types]]
        # If n > len(_SYMBOL_TYPES), reuse types
        while len(pair_types) < n:
            pair_types.append(_SYMBOL_TYPES[int(rng.integers(len(_SYMBOL_TYPES)))])

        # Place items (the symbol objects) and targets (TARGET with metadata encoding type)
        item_positions = free[:n]
        target_positions = free[n:2 * n]
        used = {agent_pos} | set(item_positions) | set(target_positions)

        # Store pair info: [(item_pos, target_pos, symbol_type), ...]
        pair_info = []
        for i in range(n):
            ix, iy = item_positions[i]
            tx, ty = target_positions[i]
            sym_type = pair_types[i]

            # Place item as its symbol type
            grid.objects[iy, ix] = sym_type
            # Place target as TARGET; encode which symbol type it expects in metadata
            grid.objects[ty, tx] = ObjectType.TARGET
            grid.metadata[ty, tx] = int(sym_type)

            pair_info.append({
                "item_pos": list(item_positions[i]),
                "target_pos": list(target_positions[i]),
                "symbol_type": int(sym_type),
            })

        # Fake items: use symbol types not in the current pair set (or random)
        fake_positions = []
        remaining_free = [p for p in free[2 * n:] if p not in used]
        unused_types = [t for t in _SYMBOL_TYPES if t not in pair_types]
        for i in range(min(n_fakes, len(remaining_free))):
            fx, fy = remaining_free[i]
            if unused_types:
                fake_type = unused_types[i % len(unused_types)]
            else:
                fake_type = _SYMBOL_TYPES[int(rng.integers(len(_SYMBOL_TYPES)))]
            grid.objects[fy, fx] = fake_type
            fake_positions.append(remaining_free[i])
            used.add(remaining_free[i])

        # Obstacle walls with flood-fill check
        wall_positions = []
        wall_candidates = [p for p in free if p not in used]
        critical = [agent_pos] + list(item_positions) + list(target_positions)
        for p in wall_candidates:
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
            "n_pairs": n,
            "max_steps": self.get_max_steps(),
        }

    def on_env_reset(self, agent, grid, config):
        config["_items_placed"] = 0
        config["_carrying"] = None  # ObjectType int or None
        config["_mismatches"] = 0
        self._items_placed = 0
        self._carrying = None
        self._last_placed_for_reward = 0

    def on_agent_moved(self, pos, agent, grid):
        """Pickup symbol item / Place on matching TARGET."""
        x, y = pos
        obj = grid.objects[y, x]

        if self._carrying is None and obj in _SYMBOL_TYPES:
            # Pick up symbol item
            grid.objects[y, x] = ObjectType.NONE
            self._carrying = int(obj)
        elif self._carrying is not None and obj == ObjectType.TARGET:
            expected = int(grid.metadata[y, x])
            if self._carrying == expected:
                # Correct match: mark as completed (GOAL visual)
                grid.objects[y, x] = ObjectType.GOAL
                grid.metadata[y, x] = 0
                self._carrying = None
                self._items_placed += 1
            else:
                # Wrong match: drop the item here (penalty but item returned)
                # Place the carried item back somewhere, or just drop on ground
                grid.objects[y, x] = ObjectType.TARGET  # target stays
                self._carrying = None  # item lost (penalty for mismatch)
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
                # Guide toward nearest uncollected symbol item
                items = [
                    (x, y) for y in range(g.height) for x in range(g.width)
                    if g.objects[y, x] in _SYMBOL_TYPES
                ]
            else:
                # Guide toward the matching TARGET
                items = [
                    (x, y) for y in range(g.height) for x in range(g.width)
                    if g.objects[y, x] == ObjectType.TARGET
                    and int(g.metadata[y, x]) == self._carrying
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

    def get_optimal_return(self, difficulty=None): return 1.0
    def get_random_baseline(self, difficulty=None): return 0.0
