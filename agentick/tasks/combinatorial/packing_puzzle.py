"""PackingPuzzle - Push boxes to fill all TARGET slots in the packing zone.

MECHANICS:
  - A TARGET zone (row of TARGET cells) must be fully covered by boxes
  - N boxes scattered near agent
  - Push boxes (Sokoban-style) into the target zone
  - Success = all TARGET cells have a BOX on them
"""

import numpy as np
from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("PackingPuzzle-v0", tags=["spatial_reasoning", "planning"])
class PackingPuzzleTask(TaskSpec):
    """Push boxes into all the target slots."""

    name = "PackingPuzzle-v0"
    description = "Push boxes to fill all target slots"
    capability_tags = ["spatial_reasoning", "planning"]

    difficulty_configs = {
        "easy":   DifficultyConfig(name="easy",   grid_size=7,  max_steps=100, params={"n_boxes": 2, "n_distractors": 0, "tight_fit": False}),
        "medium": DifficultyConfig(name="medium",  grid_size=9,  max_steps=180, params={"n_boxes": 3, "n_distractors": 1, "tight_fit": False}),
        "hard":   DifficultyConfig(name="hard",    grid_size=11, max_steps=300, params={"n_boxes": 4, "n_distractors": 2, "tight_fit": True}),
        "expert": DifficultyConfig(name="expert",  grid_size=13, max_steps=500, params={"n_boxes": 5, "n_distractors": 3, "tight_fit": True}),
    }

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size          = self.difficulty_config.grid_size
        n             = self.difficulty_config.params.get("n_boxes", 2)
        n_distractors = self.difficulty_config.params.get("n_distractors", 0)
        tight_fit     = self.difficulty_config.params.get("tight_fit", False)

        grid = Grid(size, size)
        grid.terrain[0, :]  = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0]  = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        # Target zone: a row near the bottom
        target_row = size - 2
        # tight_fit: pack targets tightly starting col 1; normal: spread across row
        target_cols = list(range(1, n + 1))
        if tight_fit and n > 1:
            # Targets placed consecutively (no gaps)
            target_cols = list(range(size // 4, size // 4 + n))
            target_cols = [min(c, size - 2) for c in target_cols]
        target_positions = [(x, target_row) for x in target_cols if x < size - 1][:n]

        for tx, ty in target_positions:
            grid.objects[ty, tx] = ObjectType.TARGET

        agent_pos = (1, 1)
        used = {agent_pos} | set(target_positions)

        box_positions = []
        for i in range(min(n, len(target_positions))):
            bx = target_positions[i][0]
            mid_range = list(range(2, target_row - 1))
            rng.shuffle(mid_range)
            for by in mid_range:
                if (bx, by) not in used:
                    box_positions.append((bx, by))
                    used.add((bx, by))
                    break

        for bx, by in box_positions:
            grid.objects[by, bx] = ObjectType.BOX

        # Distractors: extra BOX objects without matching targets (visual clutter)
        interior = [(x, y) for x in range(1, size - 1) for y in range(2, target_row - 1)
                    if (x, y) not in used]
        rng.shuffle(interior)
        distractor_positions = []
        for p in interior[:n_distractors]:
            dx2, dy2 = p
            grid.objects[dy2, dx2] = ObjectType.SWITCH  # use SWITCH to distinguish from real boxes
            distractor_positions.append(p)
            used.add(p)

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": target_positions,
            "box_positions": box_positions,
            "target_positions": target_positions,
            "distractor_positions": distractor_positions,
            "tight_fit": tight_fit,
            "n_boxes": n,
            "max_steps": self.get_max_steps(),
        }

    def on_env_reset(self, agent, grid, config):
        """Cache config and compute initial box-to-target distance for reward shaping."""
        self._current_config = config
        boxes = config.get("box_positions", [])
        targets = config.get("target_positions", [])
        if boxes and targets:
            self._last_box_dist_pp = sum(
                min(abs(bx-tx)+abs(by-ty) for tx,ty in targets) for bx,by in boxes
            )
        else:
            self._last_box_dist_pp = None

    def can_agent_enter(self, pos, agent, grid) -> bool:
        """Sokoban push: agent walks into box → push it forward one cell."""
        x, y = pos
        if grid.objects[y, x] != ObjectType.BOX:
            return True  # not a box, always passable
        # Compute push direction
        ax, ay = agent.position
        dx, dy = x - ax, y - ay
        nbx, nby = x + dx, y + dy
        # Target cell must be in bounds, non-wall, and empty or TARGET
        if not (0 < nbx < grid.width - 1 and 0 < nby < grid.height - 1):
            return False
        if grid.terrain[nby, nbx] == CellType.WALL:
            return False
        dest_obj = grid.objects[nby, nbx]
        if dest_obj not in (ObjectType.NONE, ObjectType.TARGET):
            return False  # blocked by another box, etc.
        # Push the box
        grid.objects[y, x] = ObjectType.NONE
        grid.objects[nby, nbx] = ObjectType.BOX  # box covers target too
        return True  # agent enters the vacated box cell

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        if "grid" not in new_state or "config" not in new_state:
            return reward
        grid = new_state["grid"]
        config = new_state.get("config", {})
        from agentick.core.types import ObjectType
        boxes = [(x, y) for y in range(grid.height) for x in range(grid.width)
                 if grid.objects[y, x] == ObjectType.BOX]
        targets = config.get("target_positions", [])
        if boxes and targets:
            total_d = sum(min(abs(bx-tx)+abs(by-ty) for tx,ty in targets) for bx,by in boxes)
            if self._last_box_dist_pp is not None and total_d < self._last_box_dist_pp:
                reward += 0.2 * (self._last_box_dist_pp - total_d)
            self._last_box_dist_pp = total_d
            if "agent_position" in new_state:
                ax, ay = new_state["agent_position"]
                ox, oy = old_state.get("agent_position", (ax, ay))
                nb_new = min(abs(ax-bx)+abs(ay-by) for bx,by in boxes)
                nb_old = min(abs(ox-bx)+abs(oy-by) for bx,by in boxes)
                reward += 0.05 * (nb_old - nb_new)  # stronger: outweighs step penalty
        if self.check_success(new_state):
            reward += 1.0
        return reward

    def check_success(self, state):
        """All target cells must be covered by boxes."""
        if "grid" not in state or "config" not in state:
            return False
        grid = state["grid"]
        targets = state["config"].get("target_positions", [])
        if not targets:
            return False
        # Each target position should have a box
        return all(grid.objects[ty, tx] == ObjectType.BOX for tx, ty in targets)

    def get_optimal_return(self, difficulty=None): return 1.0
    def get_random_baseline(self, difficulty=None): return 0.0
