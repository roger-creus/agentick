"""SokobanPush - Push a box onto a target position.

MECHANICS:
  - Box is placed between agent and goal
  - Agent must PUSH the box (walk into it; box slides one step if clear)
  - Pushing box into a wall = stays put (irreversible deadlock warning)
  - Success = box is on TARGET position (not agent on goal)
  - Box is shown as ObjectType.BOX; target shown as ObjectType.TARGET
"""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


def _is_deadlocked(grid, bx, by):
    """Check if a box at (bx, by) is in a corner deadlock.

    A box is deadlocked when two perpendicular neighbors are both blocked
    (WALL, HAZARD, or out of bounds), making it impossible to push the box
    away from the corner.
    """
    blocked = set()
    for dx, dy, label in [
        (0, -1, "up"),
        (0, 1, "down"),
        (-1, 0, "left"),
        (1, 0, "right"),
    ]:
        nx, ny = bx + dx, by + dy
        if (
            nx < 0
            or nx >= grid.width
            or ny < 0
            or ny >= grid.height
            or grid.terrain[ny, nx] in (CellType.WALL, CellType.HAZARD)
        ):
            blocked.add(label)
    # Check all 4 perpendicular pairs
    for v, h in [("up", "left"), ("up", "right"), ("down", "left"), ("down", "right")]:
        if v in blocked and h in blocked:
            return True
    return False


@register_task("SokobanPush-v0", tags=["reasoning", "planning"])
class SokobanPushTask(TaskSpec):
    """Push boxes onto target positions — classic Sokoban mechanics.

    A box is placed in the grid. The agent must push it onto the target
    by walking into it. Boxes can only be pushed, not pulled. If a box
    is pushed against a wall it stays put (potentially creating deadlock).
    """

    name = "SokobanPush-v0"
    description = "Push boxes onto targets"
    capability_tags = ["reasoning", "planning"]
    overrides_walkable = True  # HOLE terrain is walkable at target positions

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=7,
            max_steps=70,
            params={"n_boxes": 1, "n_targets": 1, "n_obstacles": 0, "n_hazards": 0},
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=10,
            max_steps=180,
            params={"n_boxes": 2, "n_targets": 2, "n_obstacles": 3, "n_hazards": 0},
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=13,
            max_steps=350,
            params={"n_boxes": 3, "n_targets": 3, "n_obstacles": 5, "n_hazards": 3},
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=15,
            max_steps=600,
            params={"n_boxes": 4, "n_targets": 4, "n_obstacles": 8, "n_hazards": 5},
        ),
    }

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        n_boxes = self.difficulty_config.params.get("n_boxes", 1)
        n_targets = self.difficulty_config.params.get("n_targets", n_boxes)
        n_obstacles = self.difficulty_config.params.get("n_obstacles", 0)

        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        corners = [(1, 1), (size - 2, 1), (1, size - 2), (size - 2, size - 2)]
        agent_pos = tuple(corners[int(rng.integers(0, len(corners)))])
        # Free cells excluding borders (boxes against outer walls = deadlock)
        # Boxes need at least 2 cells clearance from walls in push directions
        interior_free = [
            (x, y) for x in range(2, size - 2) for y in range(2, size - 2) if (x, y) != agent_pos
        ]
        all_free = [
            (x, y) for x in range(1, size - 1) for y in range(1, size - 1) if (x, y) != agent_pos
        ]
        rng.shuffle(interior_free)
        rng.shuffle(all_free)

        box_positions = []
        target_positions = []
        used = {agent_pos}
        n_pairs = max(n_boxes, n_targets)
        for i in range(n_pairs):
            # Boxes must be in interior (not adjacent to outer walls) and not deadlocked
            bp = next(
                (p for p in interior_free if p not in used and not _is_deadlocked(grid, *p)),
                None,
            )
            if bp is None:
                # Fallback: use any free cell but avoid corners and deadlocks
                bp = next(
                    (
                        p
                        for p in all_free
                        if p not in used
                        and not (p[0] in (1, size - 2) and p[1] in (1, size - 2))
                        and not _is_deadlocked(grid, *p)
                    ),
                    None,
                )
            if bp is None:
                break
            used.add(bp)
            # Targets can be anywhere (including near walls)
            tp = next((p for p in all_free if p not in used), None)
            if tp is None:
                break
            used.add(tp)
            box_positions.append(bp)
            target_positions.append(tp)

        for bx, by in box_positions:
            grid.objects[by, bx] = ObjectType.BOX
        for tx, ty in target_positions:
            grid.objects[ty, tx] = ObjectType.TARGET

        # Interior obstacles — flood-fill check (avoid isolating boxes or targets)
        wall_positions = []
        wall_candidates = [p for p in all_free if p not in used]
        critical = [agent_pos] + box_positions + target_positions
        for p in wall_candidates:
            if len(wall_positions) >= n_obstacles:
                break
            wx, wy = p
            grid.terrain[wy, wx] = CellType.WALL
            reachable = grid.flood_fill(agent_pos)
            if all(q in reachable for q in critical) and not any(
                _is_deadlocked(grid, bx, by) for bx, by in box_positions
            ):
                wall_positions.append(p)
                used.add(p)
            else:
                grid.terrain[wy, wx] = CellType.EMPTY

        # Place hazard terrain (agent loses if stepped on, boxes can't be pushed onto)
        n_hazards = self.difficulty_config.params.get("n_hazards", 0)
        if n_hazards > 0:
            hazard_candidates = [
                p for p in all_free if p not in used and grid.terrain[p[1], p[0]] == CellType.EMPTY
            ]
            placed_h = 0
            for hx, hy in hazard_candidates:
                if placed_h >= n_hazards:
                    break
                grid.terrain[hy, hx] = CellType.HAZARD
                # Verify all critical positions still reachable and no box deadlocked
                reachable = grid.flood_fill(agent_pos)
                if all(q in reachable for q in critical) and not any(
                    _is_deadlocked(grid, bx, by) for bx, by in box_positions
                ):
                    placed_h += 1
                else:
                    grid.terrain[hy, hx] = CellType.EMPTY

        # Set HOLE terrain on target positions AFTER obstacle/hazard placement
        # (flood_fill treats HOLE as non-walkable, so this must come last)
        for tx, ty in target_positions:
            grid.terrain[ty, tx] = CellType.HOLE  # renders as hole_block tile

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": target_positions,
            "box_positions": box_positions,
            "target_positions": target_positions,
            "max_steps": self.get_max_steps(),
        }

    # ── Hooks ──────────────────────────────────────────────────────────────────

    def on_agent_moved(self, pos, agent, grid):
        x, y = pos
        config = getattr(self, "_current_config", {})
        if grid.terrain[y, x] == CellType.HAZARD:
            config["_hazard_hit"] = True
        # Agent falls into uncovered hole (no fixed box on it)
        if grid.terrain[y, x] == CellType.HOLE and int(grid.metadata[y, x]) < 100:
            config["_fell_in_hole"] = True

    def can_agent_enter(self, pos, agent, grid) -> bool:
        """If moving into a box, try to push it one step further."""
        x, y = pos
        # Block walls and hazards (since we override walkable for HOLE terrain)
        if grid.terrain[y, x] in (CellType.WALL, CellType.HAZARD):
            return False
        obj = grid.objects[y, x]
        if obj == ObjectType.BOX:
            # Fixed boxes (on hole/target) cannot be pushed
            if int(grid.metadata[y, x]) >= 100:
                return False
            # Compute push direction (same as agent's direction of travel)
            ax, ay = agent.position
            dx = x - ax
            dy = y - ay
            nx, ny = x + dx, y + dy

            # Can push if next cell is empty (terrain and objects, not wall/hazard)
            if (
                0 <= nx < grid.width
                and 0 <= ny < grid.height
                and grid.terrain[ny, nx] not in (CellType.WALL, CellType.HAZARD)
                and grid.objects[ny, nx] not in (ObjectType.BOX,)
            ):
                # Move box
                grid.objects[y, x] = ObjectType.NONE
                # If target was here, restore it
                config = getattr(self, "_current_config", {})
                if (x, y) in config.get("target_positions", []):
                    grid.objects[y, x] = ObjectType.TARGET
                    grid.terrain[y, x] = CellType.HOLE
                # Place box at new position
                grid.objects[ny, nx] = ObjectType.BOX
                # If box landed on a hole/target, fix it in place (unmovable)
                if grid.terrain[ny, nx] == CellType.HOLE:
                    grid.metadata[ny, nx] = 100
                return True  # agent enters old box cell
            return False  # push blocked (wall or another box)
        return True

    def on_env_reset(self, agent, grid, config):
        """Cache config and compute initial box-to-target distance for reward shaping."""
        config["_hazard_hit"] = False
        config["_fell_in_hole"] = False
        self._current_config = config
        # Pre-compute initial distance so box-progress reward fires from step 1
        boxes = config.get("box_positions", [])
        targets = config.get("target_positions", [])
        if boxes and targets:
            self._last_box_dist = sum(
                min(abs(bx - tx) + abs(by - ty) for tx, ty in targets) for bx, by in boxes
            )
        else:
            self._last_box_dist = None

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        if "grid" not in new_state or "config" not in new_state:
            return reward
        config = new_state.get("config", {})
        if config.get("_fell_in_hole", False):
            return -0.5
        grid = new_state["grid"]
        from agentick.core.types import ObjectType

        boxes = [
            (x, y)
            for y in range(grid.height)
            for x in range(grid.width)
            if grid.objects[y, x] == ObjectType.BOX
        ]
        targets = config.get("target_positions", [])
        if boxes and targets:
            # Shaping 1: box closer to target
            total_d = sum(
                min(abs(bx - tx) + abs(by - ty) for tx, ty in targets) for bx, by in boxes
            )
            if self._last_box_dist is not None and total_d < self._last_box_dist:
                reward += 0.2 * (self._last_box_dist - total_d)
            self._last_box_dist = total_d
            # Shaping 2: agent closer to nearest box (approach reward)
            if "agent_position" in new_state:
                ax, ay = new_state["agent_position"]
                ox, oy = old_state.get("agent_position", (ax, ay))
                nb_new = min(abs(ax - bx) + abs(ay - by) for bx, by in boxes)
                nb_old = min(abs(ox - bx) + abs(oy - by) for bx, by in boxes)
                reward += 0.05 * (nb_old - nb_new)  # stronger: outweighs step penalty
        if self.check_success(new_state):
            reward += 1.0
        return reward

    def check_done(self, state):
        config = state.get("config", {})
        if config.get("_hazard_hit", False):
            return True
        if config.get("_fell_in_hole", False):
            return True
        return self.check_success(state)

    def check_success(self, state):
        """All boxes must be on target positions."""
        config = state.get("config", {})
        if config.get("_hazard_hit", False):
            return False
        if config.get("_fell_in_hole", False):
            return False
        if "grid" not in state or "config" not in state:
            return False
        grid = state["grid"]
        config = state.get("config", {})
        targets = config.get("target_positions", [])

        if not targets:
            return False

        # Success = every target cell has a BOX
        return all(grid.objects[ty, tx] == ObjectType.BOX for tx, ty in targets)

    def validate_instance(self, grid, config):
        """Custom validation: treat HOLE terrain (target positions) as walkable."""
        agent_pos = config.get("agent_start")
        targets = config.get("target_positions", [])
        if not agent_pos or not targets:
            return True
        # Temporarily set HOLE terrain to EMPTY for flood_fill reachability check
        hole_cells = []
        for tx, ty in targets:
            if grid.terrain[ty, tx] == CellType.HOLE:
                grid.terrain[ty, tx] = CellType.EMPTY
                hole_cells.append((tx, ty))
        reachable = grid.flood_fill(agent_pos)
        # Restore HOLE terrain
        for tx, ty in hole_cells:
            grid.terrain[ty, tx] = CellType.HOLE
        return any(t in reachable for t in targets)

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0
