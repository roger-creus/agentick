"""SymbolMatching - Match colored items to their corresponding goal zones.

MECHANICS:
  - N colored items (KEY, BOX, etc.) are placed on the grid
  - N matching target zones (TARGET) at specific positions
  - Agent must push/carry each item to its matching zone
  - Simplified: each KEY must end up at a TARGET cell
  - Agent steps on KEY to auto-carry (only one at a time), then steps on TARGET to place
  - Success = all keys placed on their matching targets
"""

import numpy as np
from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("SymbolMatching-v0", tags=["reasoning", "pattern_recognition"])
class SymbolMatchingTask(TaskSpec):
    """Carry items to their matching goal zones."""

    name = "SymbolMatching-v0"
    description = "Match items to their goal zones"
    capability_tags = ["reasoning", "pattern_recognition"]

    difficulty_configs = {
        "easy":   DifficultyConfig(name="easy",   grid_size=7,  max_steps=100, params={"n_pairs": 2, "n_fakes": 0, "n_obstacles": 0}),
        "medium": DifficultyConfig(name="medium",  grid_size=10, max_steps=180, params={"n_pairs": 3, "n_fakes": 1, "n_obstacles": 3}),
        "hard":   DifficultyConfig(name="hard",    grid_size=13, max_steps=300, params={"n_pairs": 4, "n_fakes": 2, "n_obstacles": 5}),
        "expert": DifficultyConfig(name="expert",  grid_size=15, max_steps=500, params={"n_pairs": 5, "n_fakes": 3, "n_obstacles": 8}),
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

        free = [(x, y) for x in range(1, size-1) for y in range(1, size-1)
                if (x, y) != agent_pos]
        rng.shuffle(free)

        item_positions   = free[:n]
        target_positions = free[n:2*n]
        used = {agent_pos} | set(item_positions) | set(target_positions)

        for ix, iy in item_positions:
            grid.objects[iy, ix] = ObjectType.KEY
        for tx, ty in target_positions:
            grid.objects[ty, tx] = ObjectType.TARGET

        # Fakes: extra KEY objects (look like items but no matching target — mismatches)
        fake_positions = []
        for p in free[2*n:]:
            if len(fake_positions) >= n_fakes:
                break
            if p not in used:
                fx, fy = p
                grid.objects[fy, fx] = ObjectType.KEY
                fake_positions.append(p)
                used.add(p)

        # Obstacle walls — flood-fill check
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
            "item_positions": item_positions,
            "target_positions": target_positions,
            "fake_positions": fake_positions,
            "n_pairs": n,
            "max_steps": self.get_max_steps(),
        }

    def on_env_reset(self, agent, grid, config):
        config["_items_placed"] = 0
        config["_carrying"] = False
        self._items_placed = 0
        self._carrying = False
        self._last_placed_for_reward = 0  # must reset to avoid reward gap at episode start

    def on_agent_moved(self, pos, agent, grid):
        """Pickup KEY / Place on TARGET — fires before reward and success checks."""
        x, y = pos
        obj = grid.objects[y, x]
        if not self._carrying and obj == ObjectType.KEY:
            grid.objects[y, x] = ObjectType.NONE
            self._carrying = True
        elif self._carrying and obj == ObjectType.TARGET:
            grid.objects[y, x] = ObjectType.GOAL  # matched
            self._carrying = False
            self._items_placed += 1

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        old_placed = getattr(self, "_last_placed_for_reward", 0)
        new_placed = self._items_placed
        if new_placed > old_placed:
            reward += 0.5  # reward immediately when item is placed
        self._last_placed_for_reward = new_placed
        # Approach shaping: toward nearest KEY (if not carrying) or TARGET (if carrying)
        if "agent" in new_state and "grid" in new_state:
            ax, ay = new_state["agent"].position
            ox, oy = old_state.get("agent_position", (ax, ay))
            g = new_state["grid"]
            if not self._carrying:
                # Guide toward nearest unplaced KEY
                items = [(x, y) for y in range(g.height) for x in range(g.width)
                         if g.objects[y, x] == ObjectType.KEY]
            else:
                # Guide toward nearest TARGET
                items = [(x, y) for y in range(g.height) for x in range(g.width)
                         if g.objects[y, x] == ObjectType.TARGET]
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
