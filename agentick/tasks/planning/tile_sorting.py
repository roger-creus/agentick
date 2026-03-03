"""TileSorting - Sliding puzzle (15-puzzle style).

MECHANICS:
  - A compact NxN grid of numbered tiles with one empty slot.
  - The agent IS the empty slot. Moving the agent toward an adjacent tile
    causes that tile to slide into the empty slot (swap positions).
    For example, if the agent is at (3,3) and moves UP toward tile 5 at (3,2),
    tile 5 slides down to (3,3) and the agent (empty slot) moves to (3,2).
  - Each tile has a unique number (1, 2, 3, ... up to N*N-1).
  - Target positions are marked on the floor in green, showing which numbered
    tile belongs there (e.g., a green "3" means tile 3 should end up there).
  - When a tile is sitting on its correct target, the target marker is hidden.
  - Success = all tiles in their correct (ascending row-major) positions.
  - NOT Sokoban — this is a classic sliding/15-puzzle.
"""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("TileSorting-v0", tags=["combinatorial_logic", "planning"])
class TileSortingTask(TaskSpec):
    """Solve a sliding puzzle by arranging tiles in order."""

    name = "TileSorting-v0"
    description = (
        "Sliding puzzle: move into adjacent numbered tiles to swap them with the "
        "empty slot. Green floor markers show where each tile belongs."
    )
    capability_tags = ["combinatorial_logic", "planning"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy", grid_size=7, max_steps=100, params={"puzzle_size": 2, "n_shuffles": 5}
        ),
        "medium": DifficultyConfig(
            name="medium", grid_size=9, max_steps=200, params={"puzzle_size": 3, "n_shuffles": 15}
        ),
        "hard": DifficultyConfig(
            name="hard", grid_size=11, max_steps=350, params={"puzzle_size": 3, "n_shuffles": 30}
        ),
        "expert": DifficultyConfig(
            name="expert", grid_size=13, max_steps=600, params={"puzzle_size": 4, "n_shuffles": 60}
        ),
    }

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        puzzle_size = self.difficulty_config.params.get("puzzle_size", 3)
        n_shuffles = self.difficulty_config.params.get("n_shuffles", 15)

        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        # Center the puzzle area in the grid
        offset_x = (size - puzzle_size) // 2
        offset_y = (size - puzzle_size) // 2

        # Create solved state: tiles numbered 1..N*N-1, last cell is empty
        n_tiles = puzzle_size * puzzle_size - 1
        # Use metadata to store tile numbers
        goal_positions_map = {}  # maps tile_number -> (gx, gy)
        tile_num = 1
        for py in range(puzzle_size):
            for px in range(puzzle_size):
                gx = offset_x + px
                gy = offset_y + py
                if tile_num <= n_tiles:
                    goal_positions_map[tile_num] = (gx, gy)
                    tile_num += 1

        # Start with solved state and shuffle by legal moves
        # Current tile positions: maps tile_number -> (gx, gy)
        current = {}
        tile_num = 1
        for py in range(puzzle_size):
            for px in range(puzzle_size):
                gx = offset_x + px
                gy = offset_y + py
                if tile_num <= n_tiles:
                    current[tile_num] = (gx, gy)
                    tile_num += 1

        # Empty slot starts at bottom-right of puzzle
        empty = (offset_x + puzzle_size - 1, offset_y + puzzle_size - 1)

        # Shuffle by making random legal moves
        for _ in range(n_shuffles):
            neighbors = []
            ex, ey = empty
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = ex + dx, ey + dy
                if (
                    offset_x <= nx < offset_x + puzzle_size
                    and offset_y <= ny < offset_y + puzzle_size
                ):
                    neighbors.append((nx, ny))
            # Pick random neighbor and swap
            slide_pos = neighbors[int(rng.integers(len(neighbors)))]
            # Find which tile is at slide_pos
            for tn, tp in current.items():
                if tp == slide_pos:
                    current[tn] = empty
                    empty = slide_pos
                    break

        # Build reverse lookup: position -> expected tile number
        goal_pos_to_tile = {pos: tn for tn, pos in goal_positions_map.items()}

        # Place tiles on grid using metadata for tile numbers
        # Use BOX for tiles, TARGET for goal positions (where tiles should go)
        # Metadata = tile_number + 100 when tile is on its correct position (green tint),
        # otherwise just tile_number.
        for tn, (tx, ty) in current.items():
            grid.objects[ty, tx] = ObjectType.BOX
            if goal_positions_map.get(tn) == (tx, ty):
                grid.metadata[ty, tx] = tn + 100  # correct position → visual cue
            else:
                grid.metadata[ty, tx] = tn

        # Mark goal positions with TARGET (where tiles should end up).
        # TARGET shows the expected tile number so the agent can see
        # "green 3 means tile 3 goes here".  Hidden under tiles that
        # are currently occupying the position.
        # Metadata = tile_number + 200 to avoid collision with
        # _META_OBJ_LABELS (ObjectType int values 5, 14-19).
        for tn, (gx, gy) in goal_positions_map.items():
            if grid.objects[gy, gx] == ObjectType.NONE:
                grid.objects[gy, gx] = ObjectType.TARGET
                grid.metadata[gy, gx] = tn + 200

        # Agent starts at the empty slot (agent IS the empty slot)
        agent_pos = empty

        # Wall off the puzzle border (inner walls around puzzle area)
        for y in range(offset_y - 1, offset_y + puzzle_size + 1):
            for x in range(offset_x - 1, offset_x + puzzle_size + 1):
                if 0 < x < size - 1 and 0 < y < size - 1:
                    if not (
                        offset_x <= x < offset_x + puzzle_size
                        and offset_y <= y < offset_y + puzzle_size
                    ):
                        grid.terrain[y, x] = CellType.WALL

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [],
            "puzzle_size": puzzle_size,
            "offset_x": offset_x,
            "offset_y": offset_y,
            "n_tiles": n_tiles,
            "goal_map": {str(k): list(v) for k, v in goal_positions_map.items()},
            # Reverse lookup: "x,y" -> tile_number expected at that position
            "goal_pos_to_tile": {
                f"{pos[0]},{pos[1]}": tn for pos, tn in goal_pos_to_tile.items()
            },
            "max_steps": self.get_max_steps(),
        }

    def can_agent_enter(self, pos, agent, grid) -> bool:
        """Agent moves into a tile position → slide that tile to agent's old position."""
        x, y = pos
        if grid.objects[y, x] == ObjectType.BOX:
            ax, ay = agent.position
            # Move tile from (x,y) to agent's current position (ax,ay)
            raw_meta = int(grid.metadata[y, x])
            tile_num = raw_meta - 100 if raw_meta >= 100 else raw_meta
            grid.objects[y, x] = ObjectType.NONE
            grid.metadata[y, x] = 0
            # Check if target was supposed to be here
            config = getattr(self, "_config", {})
            goal_map = config.get("goal_map", {})
            if str(tile_num) in goal_map:
                gpos = tuple(goal_map[str(tile_num)])
                if gpos == (x, y):
                    grid.objects[y, x] = ObjectType.TARGET
                    grid.metadata[y, x] = tile_num + 200
            # Place tile at agent's old position
            grid.objects[ay, ax] = ObjectType.BOX
            # Check if the tile's new position is its correct goal position
            goal_pos = goal_map.get(str(tile_num))
            if goal_pos is not None and tuple(goal_pos) == (ax, ay):
                grid.metadata[ay, ax] = tile_num + 100  # correct position → green tint
            else:
                grid.metadata[ay, ax] = tile_num
            return True
        return True

    def on_env_reset(self, agent, grid, config):
        self._config = config
        self._last_correct = self._count_correct(grid, config)

    def _count_correct(self, grid, config):
        """Count tiles in their correct positions."""
        goal_map = config.get("goal_map", {})
        correct = 0
        for tn_str, gpos in goal_map.items():
            gx, gy = gpos
            if grid.objects[gy, gx] == ObjectType.BOX:
                raw_meta = int(grid.metadata[gy, gx])
                tile_num = raw_meta - 100 if raw_meta >= 100 else raw_meta
                if tile_num == int(tn_str):
                    correct += 1
        return correct

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        if "grid" not in new_state or "config" not in new_state:
            return reward
        config = new_state["config"]
        n_correct = self._count_correct(new_state["grid"], config)
        if n_correct > self._last_correct:
            reward += 0.3 * (n_correct - self._last_correct)
        elif n_correct < self._last_correct:
            reward -= 0.1 * (self._last_correct - n_correct)
        self._last_correct = n_correct
        if self.check_success(new_state):
            reward += 1.0
        return reward

    def check_success(self, state):
        """All tiles in correct positions."""
        if "grid" not in state or "config" not in state:
            return False
        config = state["config"]
        n_tiles = config.get("n_tiles", 0)
        if n_tiles == 0:
            return False
        return self._count_correct(state["grid"], config) >= n_tiles

    def validate_instance(self, grid, config):
        return True  # puzzle is always solvable by construction

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0
