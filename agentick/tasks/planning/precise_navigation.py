"""PreciseNavigation - Ice Sliding Puzzle.

MECHANICS:
  - Grid filled with ICE terrain; agent slides in move direction until
    hitting WALL, EMPTY cell (stopping point), BOX, or grid edge
  - Interior WALL segments (L-shaped and T-shaped) create corridors and
    deflection points, forcing multi-step sliding paths
  - Scattered EMPTY cells act as "stopping points" (islands in ice)
  - Agent starts on EMPTY, goal on EMPTY; must plan multi-slide trajectories
  - At hard+: BOX objects also slide on ice when pushed
  - Minimum solution length enforced per difficulty (BFS slide count)
  - Difficulties: easy=9x9/4 stops/2 walls, medium=11x11/3 stops/4 walls,
    hard=13x13/3 stops/5 walls/1 box, expert=15x15/3 stops/7 walls/2 boxes
"""

from __future__ import annotations

from collections import deque

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task

# Directions: (dx, dy)
_DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]


def _slide(x, y, dx, dy, grid, box_positions=None):
    """Simulate sliding from (x, y) in direction (dx, dy).

    Returns the final resting position. Stops when the next cell is:
    - Out of bounds
    - WALL terrain
    - EMPTY terrain (stepping stone — agent stops ON the empty cell)
    - A BOX object
    The agent ends up on the last valid ICE cell before the blocker,
    OR on the EMPTY cell that stopped them.
    """
    bset = set(box_positions) if box_positions else set()
    cx, cy = x, y
    while True:
        nx, ny = cx + dx, cy + dy
        # Out of bounds or wall → stop
        if not (0 <= nx < grid.width and 0 <= ny < grid.height):
            break
        if grid.terrain[ny, nx] == CellType.WALL:
            break
        # BOX → stop (don't enter box cell)
        if (nx, ny) in bset:
            break
        # EMPTY cell → we slide INTO this cell and stop here
        if grid.terrain[ny, nx] == CellType.EMPTY:
            cx, cy = nx, ny
            break
        # ICE → keep sliding
        if grid.terrain[ny, nx] == CellType.ICE:
            cx, cy = nx, ny
            continue
        # Any other terrain → stop
        break
    return cx, cy


def _slide_box(bx, by, dx, dy, grid, all_box_positions):
    """Slide a box on ice. Same rules as agent but boxes don't stop on EMPTY
    — they stop when hitting WALL, edge, another BOX, or non-ICE/non-EMPTY."""
    other_boxes = set(all_box_positions) - {(bx, by)}
    cx, cy = bx, by
    while True:
        nx, ny = cx + dx, cy + dy
        if not (0 <= nx < grid.width and 0 <= ny < grid.height):
            break
        if grid.terrain[ny, nx] == CellType.WALL:
            break
        if (nx, ny) in other_boxes:
            break
        t = grid.terrain[ny, nx]
        if t in (CellType.ICE, CellType.EMPTY):
            cx, cy = nx, ny
            continue
        break
    return cx, cy


def _bfs_ice_reachable(start, goal, grid, box_positions=None):
    """BFS over (position) state space using sliding mechanics.

    Returns True if goal is reachable from start.
    For simplicity (validation only), ignores box pushing.
    """
    bset = frozenset(box_positions) if box_positions else frozenset()
    queue = deque([start])
    visited = {start}
    while queue:
        cx, cy = queue.popleft()
        if (cx, cy) == goal:
            return True
        for dx, dy in _DIRS:
            # Must be able to move at least one cell
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < grid.width and 0 <= ny < grid.height):
                continue
            if grid.terrain[ny, nx] == CellType.WALL:
                continue
            if (nx, ny) in bset:
                continue
            # Simulate the slide
            fx, fy = _slide(cx, cy, dx, dy, grid, box_positions)
            if (fx, fy) == (cx, cy):
                continue  # Didn't move
            if (fx, fy) not in visited:
                visited.add((fx, fy))
                queue.append((fx, fy))
    return False


def _bfs_ice_path_length(start, goal, grid, box_positions=None):
    """BFS over slide endpoints. Returns min number of slides to reach goal,
    or -1 if unreachable."""
    bset = frozenset(box_positions) if box_positions else frozenset()
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        (cx, cy), dist = queue.popleft()
        if (cx, cy) == goal:
            return dist
        for dx, dy in _DIRS:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < grid.width and 0 <= ny < grid.height):
                continue
            if grid.terrain[ny, nx] == CellType.WALL:
                continue
            if (nx, ny) in bset:
                continue
            fx, fy = _slide(cx, cy, dx, dy, grid, box_positions)
            if (fx, fy) == (cx, cy):
                continue
            if (fx, fy) not in visited:
                visited.add((fx, fy))
                queue.append(((fx, fy), dist + 1))
    return -1


# ── Wall segment templates ────────────────────────────────────────────────────

# L-shapes and T-shapes as relative (dx, dy) offsets from anchor point
_L_SHAPES = [
    [(0, 0), (1, 0), (2, 0), (2, 1)],           # L right-down
    [(0, 0), (1, 0), (2, 0), (2, -1)],           # L right-up
    [(0, 0), (0, 1), (0, 2), (1, 2)],            # L down-right
    [(0, 0), (0, 1), (0, 2), (-1, 2)],           # L down-left
    [(0, 0), (-1, 0), (-2, 0), (-2, 1)],         # L left-down
    [(0, 0), (-1, 0), (-2, 0), (-2, -1)],        # L left-up
    [(0, 0), (0, -1), (0, -2), (1, -2)],         # L up-right
    [(0, 0), (0, -1), (0, -2), (-1, -2)],        # L up-left
]

_T_SHAPES = [
    [(0, 0), (1, 0), (-1, 0), (0, 1)],           # T down
    [(0, 0), (1, 0), (-1, 0), (0, -1)],          # T up
    [(0, 0), (0, 1), (0, -1), (1, 0)],           # T right
    [(0, 0), (0, 1), (0, -1), (-1, 0)],          # T left
]

_I_SHAPES = [
    [(0, 0), (1, 0), (2, 0)],                    # horizontal bar
    [(0, 0), (0, 1), (0, 2)],                    # vertical bar
]

_ALL_SHAPES = _L_SHAPES + _T_SHAPES + _I_SHAPES


def _place_wall_segment(grid, rng, occupied, size):
    """Try to place a random wall segment in the interior. Returns True on success."""
    shape_idx = rng.integers(0, len(_ALL_SHAPES))
    shape = _ALL_SHAPES[shape_idx]

    # Try random anchor points
    for _ in range(50):
        ax = int(rng.integers(2, size - 2))
        ay = int(rng.integers(2, size - 2))

        cells = [(ax + dx, ay + dy) for dx, dy in shape]

        # Check all cells are valid interior ICE, not occupied, and not adjacent to border
        valid = True
        for cx, cy in cells:
            if not (2 <= cx <= size - 3 and 2 <= cy <= size - 3):
                valid = False
                break
            if (cx, cy) in occupied:
                valid = False
                break
            if grid.terrain[cy, cx] != CellType.ICE:
                valid = False
                break
        if not valid:
            continue

        # Don't place walls that would completely block a row or column
        # (ensure at least one ICE gap on each axis through the wall region)
        for cx, cy in cells:
            grid.terrain[cy, cx] = CellType.WALL
            occupied.add((cx, cy))
        return True

    return False


@register_task("PreciseNavigation-v0", tags=["planning", "spatial_reasoning"])
class PreciseNavigationTask(TaskSpec):
    """Ice sliding puzzle: plan multi-slide trajectories across an icy grid."""

    name = "PreciseNavigation-v0"
    description = "Ice sliding puzzle — slide across ice to reach the goal"
    capability_tags = ["planning", "spatial_reasoning"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=9,
            max_steps=50,
            params={
                "n_stopping_points": 4,
                "n_boxes": 0,
                "n_wall_segments": 2,
                "min_slides": 3,
            },
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=11,
            max_steps=80,
            params={
                "n_stopping_points": 3,
                "n_boxes": 0,
                "n_wall_segments": 4,
                "min_slides": 5,
            },
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=13,
            max_steps=120,
            params={
                "n_stopping_points": 3,
                "n_boxes": 1,
                "n_wall_segments": 5,
                "min_slides": 6,
            },
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=15,
            max_steps=200,
            params={
                "n_stopping_points": 3,
                "n_boxes": 2,
                "n_wall_segments": 7,
                "min_slides": 8,
            },
        ),
    }

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        params = self.difficulty_config.params
        n_stops = params.get("n_stopping_points", 4)
        n_boxes = params.get("n_boxes", 0)
        n_walls = params.get("n_wall_segments", 0)
        min_slides = params.get("min_slides", 3)

        # Try multiple seeds to find a solvable instance with sufficient length
        for attempt in range(500):
            grid = Grid(size, size)

            # Border walls
            grid.terrain[0, :] = CellType.WALL
            grid.terrain[-1, :] = CellType.WALL
            grid.terrain[:, 0] = CellType.WALL
            grid.terrain[:, -1] = CellType.WALL

            # Fill interior with ICE
            for y in range(1, size - 1):
                for x in range(1, size - 1):
                    grid.terrain[y, x] = CellType.ICE

            # Agent start: random corner region
            corners = [(1, 1), (1, size - 2), (size - 2, 1), (size - 2, size - 2)]
            agent_pos = tuple(corners[int(rng.integers(0, 4))])
            grid.terrain[agent_pos[1], agent_pos[0]] = CellType.EMPTY

            occupied = {agent_pos}

            # Place interior wall segments
            for _ in range(n_walls):
                _place_wall_segment(grid, rng, occupied, size)

            # Collect valid interior ICE positions for stopping points
            interior = [
                (x, y)
                for y in range(1, size - 1)
                for x in range(1, size - 1)
                if (x, y) not in occupied and grid.terrain[y, x] == CellType.ICE
            ]
            rng.shuffle(interior)

            # Place stopping points with minimum distance from agent
            stop_positions = []
            for pos in interior:
                if len(stop_positions) >= n_stops:
                    break
                x, y = pos
                # Minimum distance 3 from agent (harder to reach)
                if abs(x - agent_pos[0]) + abs(y - agent_pos[1]) < 3:
                    continue
                stop_positions.append(pos)

            for sx, sy in stop_positions:
                grid.terrain[sy, sx] = CellType.EMPTY

            if not stop_positions:
                continue

            # Goal: pick the farthest stopping point from agent
            stops_by_dist = sorted(
                stop_positions,
                key=lambda p: abs(p[0] - agent_pos[0]) + abs(p[1] - agent_pos[1]),
                reverse=True,
            )
            goal_pos = stops_by_dist[0]
            grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.GOAL

            # Place boxes on ICE cells (hard+)
            box_positions = []
            if n_boxes > 0:
                ice_cells = [
                    (x, y)
                    for y in range(1, size - 1)
                    for x in range(1, size - 1)
                    if grid.terrain[y, x] == CellType.ICE
                    and (x, y) not in occupied
                    and (x, y) != goal_pos
                ]
                rng.shuffle(ice_cells)
                for bp in ice_cells[:n_boxes]:
                    box_positions.append(bp)
                    grid.objects[bp[1], bp[0]] = ObjectType.BOX

            # Validate: BFS with sliding mechanics must reach goal
            # AND minimum solution length must be met
            path_len = _bfs_ice_path_length(
                agent_pos, goal_pos, grid, box_positions
            )
            if path_len >= min_slides:
                return grid, {
                    "agent_start": agent_pos,
                    "goal_positions": [goal_pos],
                    "box_positions": box_positions,
                    "max_steps": self.get_max_steps(),
                }

        # Fallback: minimal solvable instance (should rarely happen)
        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL
        for y in range(1, size - 1):
            for x in range(1, size - 1):
                grid.terrain[y, x] = CellType.ICE
        agent_pos = (1, 1)
        grid.terrain[1, 1] = CellType.EMPTY
        # Place wall in center to force indirect path
        mid = size // 2
        grid.terrain[mid, mid] = CellType.WALL
        grid.terrain[mid, mid + 1] = CellType.WALL
        goal_pos = (size - 2, size - 2)
        grid.terrain[goal_pos[1], goal_pos[0]] = CellType.EMPTY
        grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.GOAL

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [goal_pos],
            "box_positions": [],
            "max_steps": self.get_max_steps(),
        }

    # ── Hooks ────────────────────────────────────────────────────────────────

    def on_env_reset(self, agent, grid, config):
        self._prev_pos = config.get("agent_start", agent.position)
        self._config = config

    def can_agent_enter(self, pos, agent, grid) -> bool:
        """Allow entry to ICE and EMPTY cells. Block WALL."""
        x, y = pos
        t = grid.terrain[y, x]
        if t == CellType.WALL:
            return False
        # Block entry into cells with BOX (agent can't walk onto boxes)
        if grid.objects[y, x] == ObjectType.BOX:
            return False
        return True

    def on_agent_moved(self, pos, agent, grid):
        """Handle ice sliding after agent moves to a new cell.

        If the agent stepped onto ICE, they slide in the movement direction
        until hitting a wall, empty cell, box, or grid edge.
        If sliding into a box, the box also slides.
        """
        config = getattr(self, "_config", {})
        prev = self._prev_pos
        x, y = pos

        # Compute movement direction
        dx = x - prev[0]
        dy = y - prev[1]

        # Clamp to unit direction (should already be, but be safe)
        if dx != 0:
            dx = 1 if dx > 0 else -1
        if dy != 0:
            dy = 1 if dy > 0 else -1

        # If agent is on ICE, slide
        if grid.terrain[y, x] == CellType.ICE and (dx != 0 or dy != 0):
            box_positions = list(config.get("box_positions", []))
            bset = set(box_positions)

            cx, cy = x, y
            while True:
                nx, ny = cx + dx, cy + dy
                # Out of bounds or wall → stop
                if not (0 <= nx < grid.width and 0 <= ny < grid.height):
                    break
                if grid.terrain[ny, nx] == CellType.WALL:
                    break
                # BOX → push the box, agent stops here
                if (nx, ny) in bset:
                    # Slide the box
                    bx, by = nx, ny
                    fbx, fby = _slide_box(bx, by, dx, dy, grid, box_positions)
                    if (fbx, fby) != (bx, by):
                        # Move box on grid
                        grid.objects[by, bx] = ObjectType.NONE
                        grid.objects[fby, fbx] = ObjectType.BOX
                        # Update box positions in config
                        idx = box_positions.index((bx, by))
                        box_positions[idx] = (fbx, fby)
                        config["box_positions"] = box_positions
                        bset = set(box_positions)
                        # Agent moves into the box's old cell
                        cx, cy = nx, ny
                        # If now on ICE, continue sliding
                        if grid.terrain[cy, cx] != CellType.ICE:
                            break
                        continue
                    else:
                        # Box can't move → agent stops
                        break
                # EMPTY cell → slide into it and stop
                if grid.terrain[ny, nx] == CellType.EMPTY:
                    cx, cy = nx, ny
                    break
                # ICE → keep sliding
                if grid.terrain[ny, nx] == CellType.ICE:
                    cx, cy = nx, ny
                    continue
                # Anything else → stop
                break

            # Update agent position if it changed
            if (cx, cy) != (x, y):
                agent.position = (cx, cy)

        # Update previous position for next move
        self._prev_pos = agent.position

    # ── Reward & success ─────────────────────────────────────────────────────

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01  # step penalty

        # Distance shaping toward goal
        goal = new_state.get("config", {}).get("goal_positions", [None])[0]
        if goal and "agent" in new_state:
            ax, ay = new_state["agent"].position
            ox, oy = old_state.get("agent_position", (ax, ay))
            old_d = abs(ox - goal[0]) + abs(oy - goal[1])
            new_d = abs(ax - goal[0]) + abs(ay - goal[1])
            reward += 0.05 * (old_d - new_d)

        if self.check_success(new_state):
            reward += 1.0

        return reward

    def compute_sparse_reward(self, old_state, action, new_state, info):
        if self.check_success(new_state):
            return 1.0
        return 0.0

    def check_success(self, state):
        """Agent on GOAL cell."""
        if "grid" not in state or "agent" not in state:
            return False
        x, y = state["agent"].position
        return bool(state["grid"].objects[y, x] == ObjectType.GOAL)

    def check_done(self, state):
        return self.check_success(state)

    def validate_instance(self, grid, config):
        """Verify goal is reachable via BFS with sliding mechanics."""
        agent_pos = config.get("agent_start")
        goals = config.get("goal_positions", [])
        boxes = config.get("box_positions", [])
        if not agent_pos or not goals:
            return True
        return _bfs_ice_reachable(agent_pos, goals[0], grid, boxes)

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0
