"""TileSorting - Push colored tiles into matching goal zones (Sokoban-style).

MECHANICS:
  - N BOX tiles placed on the grid (each needs to go to a TARGET zone)
  - Walk into a box to push it one cell in that direction
  - Success = every box is on a TARGET cell
  - Simplified: any box on any target counts (unlabeled matching)
"""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("TileSorting-v0", tags=["combinatorial_logic", "planning"])
class TileSortingTask(TaskSpec):
    """Push all tiles into their matching target zones."""

    name = "TileSorting-v0"
    description = "Push tiles into target zones"
    capability_tags = ["combinatorial_logic", "planning"]

    difficulty_configs = {
        "easy":   DifficultyConfig(name="easy",   grid_size=7,  max_steps=100, params={"n_tiles": 2, "n_distractors": 0, "n_obstacles": 0}),
        "medium": DifficultyConfig(name="medium",  grid_size=9,  max_steps=180, params={"n_tiles": 3, "n_distractors": 1, "n_obstacles": 3}),
        "hard":   DifficultyConfig(name="hard",    grid_size=11, max_steps=300, params={"n_tiles": 4, "n_distractors": 2, "n_obstacles": 5}),
        "expert": DifficultyConfig(name="expert",  grid_size=13, max_steps=500, params={"n_tiles": 5, "n_distractors": 3, "n_obstacles": 8}),
    }

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size          = self.difficulty_config.grid_size
        n             = self.difficulty_config.params.get("n_tiles", 2)
        n_distractors = self.difficulty_config.params.get("n_distractors", 0)
        n_obstacles   = self.difficulty_config.params.get("n_obstacles", 0)

        grid = Grid(size, size)
        grid.terrain[0, :]  = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0]  = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        agent_pos = (1, 1)
        free = [(x, y) for x in range(2, size - 2) for y in range(2, size - 2)
                if (x, y) != agent_pos]
        rng.shuffle(free)

        tile_positions   = free[:n]
        taken = set(tile_positions) | {agent_pos}
        target_positions = [p for p in free[n:] if p not in taken][:n]
        used = taken | set(target_positions)

        for bx, by in tile_positions:
            grid.objects[by, bx] = ObjectType.BOX
        for tx, ty in target_positions:
            grid.objects[ty, tx] = ObjectType.TARGET

        # Distractors: extra SWITCH objects (visual noise, not BOX/TARGET)
        distractor_positions = []
        for p in free[n:]:
            if len(distractor_positions) >= n_distractors:
                break
            if p not in used:
                dx2, dy2 = p
                grid.objects[dy2, dx2] = ObjectType.SWITCH
                distractor_positions.append(p)
                used.add(p)

        # Interior walls — flood-fill check
        wall_positions = []
        wall_candidates = [p for p in free if p not in used]
        critical = [agent_pos] + tile_positions + target_positions
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
            "tile_positions": tile_positions,
            "target_positions": target_positions,
            "distractor_positions": distractor_positions,
            "n_tiles": n,
            "max_steps": self.get_max_steps(),
        }

    def can_agent_enter(self, pos, agent, grid) -> bool:
        """Sokoban push: agent walks into tile → push it one cell forward."""
        x, y = pos
        if grid.objects[y, x] != ObjectType.BOX:
            return True  # passable
        # Compute push direction from agent's current position
        ax, ay = agent.position
        dx, dy = x - ax, y - ay
        nbx, nby = x + dx, y + dy
        # Target cell must be in-bounds, non-wall, and passable
        if not (0 < nbx < grid.width - 1 and 0 < nby < grid.height - 1):
            return False
        if grid.terrain[nby, nbx] == CellType.WALL:
            return False
        dest_obj = grid.objects[nby, nbx]
        if dest_obj not in (ObjectType.NONE, ObjectType.TARGET):
            return False  # blocked by another tile, etc.
        # Push the tile
        grid.objects[y, x] = ObjectType.NONE
        grid.objects[nby, nbx] = ObjectType.BOX  # covers TARGET too
        return True  # agent enters vacated tile cell

    def on_env_reset(self, agent, grid, config):
        agent.inventory.clear()  # prevent accidental TARGET pickup leak
        tiles = config.get("tile_positions", [])
        targets = config.get("target_positions", [])
        if tiles and targets:
            self._last_tile_dist = sum(
                min(abs(tx-ttx)+abs(ty-tty) for ttx,tty in targets) for tx,ty in tiles
            )
        else:
            self._last_tile_dist = None

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        if "grid" not in new_state or "config" not in new_state:
            return reward
        grid = new_state["grid"]
        config = new_state.get("config", {})
        from agentick.core.types import ObjectType
        tiles = [(x, y) for y in range(grid.height) for x in range(grid.width)
                 if grid.objects[y, x] == ObjectType.BOX]
        targets = config.get("target_positions", [])
        if tiles and targets:
            total_d = sum(min(abs(tx2-ttx)+abs(ty2-tty) for ttx,tty in targets)
                         for tx2,ty2 in tiles)
            if self._last_tile_dist is not None and total_d < self._last_tile_dist:
                reward += 0.2 * (self._last_tile_dist - total_d)
            self._last_tile_dist = total_d
            # Approach reward: agent → nearest tile (outweighs step penalty)
            if "agent_position" in new_state:
                ax, ay = new_state["agent_position"]
                ox, oy = old_state.get("agent_position", (ax, ay))
                nb_new = min(abs(ax-tx2)+abs(ay-ty2) for tx2,ty2 in tiles)
                nb_old = min(abs(ox-tx2)+abs(oy-ty2) for tx2,ty2 in tiles)
                reward += 0.05 * (nb_old - nb_new)
        if self.check_success(new_state):
            reward += 1.0
        return reward

    def check_success(self, state):
        """All boxes must be on target cells."""
        if "grid" not in state or "config" not in state:
            return False
        grid = state["grid"]
        targets = set(map(tuple, state["config"].get("target_positions", [])))
        if not targets:
            return False
        # Count remaining TARGET cells (not yet covered by box)
        remaining_targets = sum(
            1 for y in range(grid.height) for x in range(grid.width)
            if grid.objects[y, x] == ObjectType.TARGET
        )
        return remaining_targets == 0

    def get_optimal_return(self, difficulty=None): return 1.0
    def get_random_baseline(self, difficulty=None): return 0.0
