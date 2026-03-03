"""Recursive Rooms - Hierarchical room structure with true recursive subdivision.

MECHANICS:
  - Grid is recursively subdivided into nested quadrants
  - Each subdivision creates 4 sub-rooms separated by walls with doorways
  - Goal is placed in the deepest nested room
  - Agent must navigate through nested doorways to reach the goal
  - Depth parameter controls recursion depth (actual nesting, not just room count)
"""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("RecursiveRooms-v0", tags=["hierarchical", "planning", "composition"])
class RecursiveRoomsTask(TaskSpec):
    """Navigate nested recursive room structures to reach the goal.

    The grid is recursively subdivided into quadrants. Each level of
    recursion creates walls with single doorways, producing rooms nested
    within rooms. The agent must plan at multiple levels of abstraction
    to navigate through the hierarchy and reach the goal in the deepest
    nested room.

    Grid sizes: easy=13x13, medium=19x19, hard=25x25, expert=31x31.
    Recursion depths: easy=2 (4 rooms), medium=3 (16 rooms), hard=4 (64 rooms),
    expert=4 (64 rooms, larger grid).
    """

    name = "RecursiveRooms-v0"
    description = "Navigate nested hierarchical room structure"
    capability_tags = ["hierarchical_planning", "composition", "navigation"]

    difficulty_configs = {
        "easy": DifficultyConfig(name="easy", grid_size=13, max_steps=150, params={"depth": 2}),
        "medium": DifficultyConfig(name="medium", grid_size=19, max_steps=250, params={"depth": 3}),
        "hard": DifficultyConfig(name="hard", grid_size=25, max_steps=350, params={"depth": 4}),
        "expert": DifficultyConfig(name="expert", grid_size=31, max_steps=500, params={"depth": 4}),
    }

    def _subdivide(self, grid, x1, y1, x2, y2, depth, rng, doorways):
        """Recursively subdivide a rectangular region into 4 sub-rooms.

        Creates a horizontal and vertical wall through the region, with
        one doorway in each wall segment (4 wall segments, 3 doorways —
        one segment is left sealed to force non-trivial navigation).

        Args:
            grid: Grid to modify.
            x1, y1: Top-left corner of region (inclusive).
            x2, y2: Bottom-right corner of region (inclusive).
            depth: Remaining recursion depth.
            rng: Random number generator.
            doorways: List to append doorway positions to.

        Returns:
            The deepest sub-room as (x1, y1, x2, y2) tuple for goal placement.
        """
        w = x2 - x1 + 1
        h = y2 - y1 + 1

        if depth <= 0 or w < 5 or h < 5:
            return (x1, y1, x2, y2)

        # Pick split points (avoid edges, ensure min room size of 2)
        mid_x = x1 + int(rng.integers(2, max(3, w - 2)))
        mid_y = y1 + int(rng.integers(2, max(3, h - 2)))

        # Draw horizontal wall
        for x in range(x1, x2 + 1):
            if grid.terrain[mid_y, x] != CellType.WALL:
                grid.terrain[mid_y, x] = CellType.WALL

        # Draw vertical wall
        for y in range(y1, y2 + 1):
            if grid.terrain[y, mid_x] != CellType.WALL:
                grid.terrain[y, mid_x] = CellType.WALL

        # Define 4 quadrants
        quads = [
            (x1, y1, mid_x - 1, mid_y - 1),  # top-left
            (mid_x + 1, y1, x2, mid_y - 1),  # top-right
            (x1, mid_y + 1, mid_x - 1, y2),  # bottom-left
            (mid_x + 1, mid_y + 1, x2, y2),  # bottom-right
        ]

        # Create doorways: 3 of 4 wall segments get a doorway
        # Wall segments: top-half vertical, bottom-half vertical,
        #                left-half horizontal, right-half horizontal
        segments = [
            ("v_top", x1, mid_y - 1, mid_x, y1, mid_y - 1),
            ("v_bot", x1, mid_y + 1, mid_x, mid_y + 1, y2),
            ("h_left", x1, mid_x - 1, mid_y, x1, mid_x - 1),
            ("h_right", mid_x + 1, x2, mid_y, mid_x + 1, x2),
        ]

        # Open 3 doorways (close one randomly to force navigation)
        closed_idx = int(rng.integers(0, 4))
        for i, seg in enumerate(segments):
            if i == closed_idx:
                continue
            kind = seg[0]
            if kind.startswith("v"):
                # Vertical wall doorway: pick y position in the range
                lo_y, hi_y = seg[4], seg[5]
                if hi_y > lo_y:
                    dy = int(rng.integers(lo_y, hi_y + 1))
                    grid.terrain[dy, mid_x] = CellType.EMPTY
                    doorways.append((mid_x, dy))
            else:
                # Horizontal wall doorway: pick x position in the range
                lo_x, hi_x = seg[4], seg[5]
                if hi_x > lo_x:
                    dx = int(rng.integers(lo_x, hi_x + 1))
                    grid.terrain[mid_y, dx] = CellType.EMPTY
                    doorways.append((dx, mid_y))

        # Recurse into all valid quadrants; track the deepest room
        # Goal goes in a random deepest quadrant
        deepest = None
        target_quad = int(rng.integers(0, 4))
        for qi, (qx1, qy1, qx2, qy2) in enumerate(quads):
            if qx2 - qx1 < 2 or qy2 - qy1 < 2:
                continue
            result = self._subdivide(grid, qx1, qy1, qx2, qy2, depth - 1, rng, doorways)
            if qi == target_quad:
                deepest = result

        return deepest if deepest else quads[target_quad % len(quads)]

    def _validate_map(self, grid, agent_pos, goal_pos):
        """Check that map is non-trivial and solvable."""
        # Count empty cells — reject maps that are too sparse
        empty_count = int(np.sum(grid.terrain == CellType.EMPTY))
        min_empty = max(10, grid.width * grid.height // 8)
        if empty_count < min_empty:
            return False
        # Goal must be reachable
        reachable = grid.flood_fill(agent_pos)
        if goal_pos not in reachable:
            return False
        # Goal must be at least half the grid size away (manhattan distance)
        min_dist = max(grid.width, grid.height) // 2
        actual_dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
        if actual_dist < min_dist:
            return False
        return True

    def generate(self, seed):
        size = self.difficulty_config.grid_size
        depth = self.difficulty_config.params.get("depth", 2)

        # Ensure odd size for clean subdivision
        if size % 2 == 0:
            size += 1

        # Retry loop to avoid empty or degenerate maps
        for attempt in range(20):
            attempt_rng = np.random.default_rng(seed + attempt * 1000)
            grid = Grid(size, size)
            # Start with all walls
            grid.terrain[:, :] = CellType.WALL

            # Carve outer room
            for y in range(1, size - 1):
                for x in range(1, size - 1):
                    grid.terrain[y, x] = CellType.EMPTY

            # Recursively subdivide
            doorways = []
            deepest_room = self._subdivide(
                grid, 1, 1, size - 2, size - 2, depth, attempt_rng, doorways
            )

            # Agent starts in top-left area
            agent_pos = (1, 1)
            if grid.terrain[1, 1] == CellType.WALL:
                # Find nearest empty cell
                found = False
                for y in range(1, size - 1):
                    for x in range(1, size - 1):
                        if grid.terrain[y, x] == CellType.EMPTY:
                            agent_pos = (x, y)
                            found = True
                            break
                    if found:
                        break

            # Goal in deepest room
            if deepest_room is None:
                continue
            rx1, ry1, rx2, ry2 = deepest_room
            goal_candidates = [
                (x, y)
                for x in range(rx1, rx2 + 1)
                for y in range(ry1, ry2 + 1)
                if grid.terrain[y, x] == CellType.EMPTY and (x, y) != agent_pos
            ]
            if goal_candidates:
                goal_pos = goal_candidates[int(attempt_rng.integers(len(goal_candidates)))]
            else:
                # Fallback: farthest reachable cell
                reachable = grid.flood_fill(agent_pos)
                reachable_list = list(reachable - {agent_pos})
                if reachable_list:
                    goal_pos = max(
                        reachable_list,
                        key=lambda p: abs(p[0] - agent_pos[0]) + abs(p[1] - agent_pos[1]),
                    )
                else:
                    continue  # retry — no reachable cells

            # Ensure goal is reachable, opening doorways if needed
            reachable = grid.flood_fill(agent_pos)
            if goal_pos not in reachable:
                for dx, dy in doorways:
                    grid.terrain[dy, dx] = CellType.EMPTY
                reachable = grid.flood_fill(agent_pos)
                if goal_pos not in reachable:
                    continue  # retry — still unreachable

            # Validate map quality
            if self._validate_map(grid, agent_pos, goal_pos):
                grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.GOAL
                return grid, {
                    "agent_start": agent_pos,
                    "goal_positions": [goal_pos],
                    "max_steps": self.get_max_steps(),
                    "depth": depth,
                    "n_doorways": len(doorways),
                }

        # Ultimate fallback: simple bordered grid with interior walls
        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL
        # Add a simple cross wall to create rooms
        mid = size // 2
        for x in range(1, size - 1):
            grid.terrain[mid, x] = CellType.WALL
        for y in range(1, size - 1):
            grid.terrain[y, mid] = CellType.WALL
        # Add doorways
        grid.terrain[mid, mid // 2] = CellType.EMPTY
        grid.terrain[mid, mid + mid // 2] = CellType.EMPTY
        grid.terrain[mid // 2, mid] = CellType.EMPTY
        grid.terrain[mid + mid // 2, mid] = CellType.EMPTY
        agent_pos = (1, 1)
        goal_pos = (size - 2, size - 2)
        grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.GOAL
        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [goal_pos],
            "max_steps": self.get_max_steps(),
            "depth": depth,
            "n_doorways": 4,
        }

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        if "agent" in new_state and "config" in new_state:
            config = new_state["config"]
            goal = config.get("goal_positions", [None])[0]
            if goal:
                ax, ay = new_state["agent"].position
                ox, oy = old_state.get("agent_position", (ax, ay))
                old_d = abs(ox - goal[0]) + abs(oy - goal[1])
                new_d = abs(ax - goal[0]) + abs(ay - goal[1])
                reward += 0.05 * (old_d - new_d)
        if self.check_success(new_state):
            reward += 1.0
        return reward

    def check_success(self, state):
        if "grid" not in state or "agent" not in state:
            return False
        x, y = state["agent"].position
        return bool(state["grid"].objects[y, x] == ObjectType.GOAL)

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0
