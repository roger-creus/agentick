"""DistributionShift - Multi-phase navigation through shifting mazes.

MECHANICS:
  - Agent must reach 3 goals total across 3 maze phases.
  - After reaching each goal, the maze regenerates with a new layout and a
    new goal position. The agent stays where it is.
  - All 3 maze layouts (DFS-generated) are pre-computed in ``generate()``
    and applied sequentially.
  - At hard+ difficulties, key-door pairs are added in phases 2 and 3.
  - At hard difficulty, action controls flip (UP<->DOWN, LEFT<->RIGHT)
    after the second goal is reached.
  - At expert difficulty, action controls flip after the first goal is reached.
  - Success requires reaching all 3 goals within the step budget.

DIFFICULTY LEVELS:
  - easy:   9x9,  3 goals, no keys, no remap, max_steps=120
  - medium: 11x11, 3 goals, no keys, no remap, max_steps=220
  - hard:   13x13, 3 goals, 1 key-door per phase, action remap after phase 2, max_steps=380
  - expert: 17x17, 3 goals, 2 key-doors per phase, action remap after phase 1, max_steps=550
"""

from __future__ import annotations

from collections import deque

import numpy as np

from agentick.core.entity import Entity
from agentick.core.grid import Grid
from agentick.core.types import ActionType, CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task

# Full action remap: all four cardinal directions swap.
_ACTION_REMAP_FULL = {
    ActionType.MOVE_UP: ActionType.MOVE_DOWN,
    ActionType.MOVE_DOWN: ActionType.MOVE_UP,
    ActionType.MOVE_LEFT: ActionType.MOVE_RIGHT,
    ActionType.MOVE_RIGHT: ActionType.MOVE_LEFT,
}

_DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]


def _generate_maze_dfs(
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a perfect maze using recursive backtracker (DFS).

    The maze uses odd-sized grids: cells live at odd coordinates, walls at
    even coordinates. The border is always WALL.

    Args:
        size: Grid width and height (must be odd and >= 5).
        rng: Seeded random number generator.

    Returns:
        2D int8 array with CellType.WALL / CellType.EMPTY values.
    """
    grid = np.full((size, size), CellType.WALL, dtype=np.int8)

    # Start carving from (1, 1).
    start = (1, 1)
    grid[start[1], start[0]] = CellType.EMPTY

    stack = [start]
    visited = {start}

    while stack:
        x, y = stack[-1]
        neighbors = []
        for dx, dy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
            nx, ny = x + dx, y + dy
            if 1 <= nx < size - 1 and 1 <= ny < size - 1:
                if (nx, ny) not in visited:
                    neighbors.append((nx, ny, x + dx // 2, y + dy // 2))

        if neighbors:
            nx, ny, wx, wy = neighbors[int(rng.integers(len(neighbors)))]
            grid[wy, wx] = CellType.EMPTY
            grid[ny, nx] = CellType.EMPTY
            visited.add((nx, ny))
            stack.append((nx, ny))
        else:
            stack.pop()

    return grid


def _bfs_path(
    terrain: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    """BFS shortest path from *start* to *goal* on a terrain array.

    Returns a list of (x, y) positions from *start* to *goal* (inclusive),
    or an empty list if no path exists.
    """
    if start == goal:
        return [start]
    sx, sy = start
    if terrain[sy, sx] == CellType.WALL:
        return []

    visited: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    queue: deque[tuple[int, int]] = deque([start])

    while queue:
        x, y = queue.popleft()
        for dx, dy in _DIRS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < terrain.shape[1] and 0 <= ny < terrain.shape[0]:
                if (nx, ny) not in visited and terrain[ny, nx] != CellType.WALL:
                    visited[(nx, ny)] = (x, y)
                    if (nx, ny) == goal:
                        # Reconstruct path
                        path = [(nx, ny)]
                        cur = (x, y)
                        while cur is not None:
                            path.append(cur)
                            cur = visited[cur]
                        path.reverse()
                        return path
                    queue.append((nx, ny))
    return []


def _flood_fill(terrain: np.ndarray, start: tuple[int, int]) -> set[tuple[int, int]]:
    """BFS flood fill on a terrain array, returning reachable (x, y) positions.

    Args:
        terrain: 2D int8 array of CellType values.
        start: (x, y) starting position.

    Returns:
        Set of (x, y) positions reachable from *start* via non-WALL cells.
    """
    sx, sy = start
    if terrain[sy, sx] == CellType.WALL:
        return set()

    visited: set[tuple[int, int]] = {start}
    queue: deque[tuple[int, int]] = deque([start])

    while queue:
        x, y = queue.popleft()
        for dx, dy in _DIRS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < terrain.shape[1] and 0 <= ny < terrain.shape[0]:
                if (nx, ny) not in visited and terrain[ny, nx] != CellType.WALL:
                    visited.add((nx, ny))
                    queue.append((nx, ny))

    return visited


def _find_empty_near(
    terrain: np.ndarray, target_x: int, target_y: int, size: int,
) -> tuple[int, int]:
    """Find the nearest EMPTY cell to (target_x, target_y).

    Searches in expanding Manhattan-distance rings from the target.

    Args:
        terrain: 2D terrain array.
        target_x: Desired x coordinate.
        target_y: Desired y coordinate.
        size: Grid dimension.

    Returns:
        (x, y) of the nearest EMPTY cell.
    """
    tx = max(1, min(target_x, size - 2))
    ty = max(1, min(target_y, size - 2))

    if terrain[ty, tx] == CellType.EMPTY:
        return (tx, ty)

    for dist in range(1, size):
        for dy in range(-dist, dist + 1):
            for dx in range(-dist, dist + 1):
                if abs(dx) + abs(dy) != dist:
                    continue
                nx, ny = tx + dx, ty + dy
                if 1 <= nx < size - 1 and 1 <= ny < size - 1:
                    if terrain[ny, nx] == CellType.EMPTY:
                        return (nx, ny)

    return (1, 1)


def _pick_random_empty(
    terrain: np.ndarray,
    size: int,
    rng: np.random.Generator,
    exclude: set[tuple[int, int]] | None = None,
) -> tuple[int, int]:
    """Pick a random EMPTY interior cell, avoiding excluded positions.

    Args:
        terrain: 2D terrain array.
        size: Grid dimension.
        rng: Seeded RNG.
        exclude: Set of (x, y) to avoid.

    Returns:
        (x, y) of a random empty cell.
    """
    if exclude is None:
        exclude = set()
    candidates = [
        (x, y)
        for x in range(1, size - 1)
        for y in range(1, size - 1)
        if terrain[y, x] == CellType.EMPTY and (x, y) not in exclude
    ]
    if candidates:
        return candidates[int(rng.integers(len(candidates)))]
    # Fallback
    return _find_empty_near(terrain, size // 2, size // 2, size)


def _pick_far_empty(
    terrain: np.ndarray,
    size: int,
    rng: np.random.Generator,
    origin: tuple[int, int],
    min_dist: int,
    exclude: set[tuple[int, int]] | None = None,
) -> tuple[int, int]:
    """Pick a random EMPTY cell at least *min_dist* Manhattan distance from *origin*.

    Falls back to _pick_random_empty if no cell meets the distance requirement.

    Args:
        terrain: 2D terrain array.
        size: Grid dimension.
        rng: Seeded RNG.
        origin: (x, y) reference position.
        min_dist: Minimum Manhattan distance from *origin*.
        exclude: Set of (x, y) to avoid.

    Returns:
        (x, y) of a suitable empty cell.
    """
    if exclude is None:
        exclude = set()
    ox, oy = origin
    candidates = [
        (x, y)
        for x in range(1, size - 1)
        for y in range(1, size - 1)
        if terrain[y, x] == CellType.EMPTY
        and (x, y) not in exclude
        and abs(x - ox) + abs(y - oy) >= min_dist
    ]
    if candidates:
        return candidates[int(rng.integers(len(candidates)))]
    # Fallback: no cell far enough — pick any empty cell.
    return _pick_random_empty(terrain, size, rng, exclude)


@register_task("DistributionShift-v0", tags=["adversarial", "adaptation"])
class DistributionShiftTask(TaskSpec):
    """Multi-phase navigation: reach 3 goals across shifting maze layouts."""

    name = "DistributionShift-v0"
    description = "Navigate 3 goals across shifting maze layouts"
    capability_tags = ["adversarial", "adaptation"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=9,
            max_steps=120,
            params={
                "n_keys_per_phase": 0,
                "action_remap_after_phase": -1,  # -1 = never
            },
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=11,
            max_steps=220,
            params={
                "n_keys_per_phase": 0,
                "action_remap_after_phase": -1,
            },
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=13,
            max_steps=380,
            params={
                "n_keys_per_phase": 1,
                "action_remap_after_phase": 2,  # remap after reaching goal 2
            },
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=17,
            max_steps=550,
            params={
                "n_keys_per_phase": 2,
                "action_remap_after_phase": 1,  # remap after reaching goal 1
            },
        ),
    }

    N_GOALS = 3  # Total goals to reach (all difficulties)

    def generate(self, seed: int):
        """Generate a multi-phase shifting-maze instance.

        Pre-generates 3 DFS maze layouts and goal positions. The first maze is
        placed on the grid; subsequent layouts are stored in config and applied
        when the agent reaches each goal.

        For hard/expert: also pre-generates key and door positions for
        phases 2 and 3.

        Args:
            seed: Random seed for reproducible generation.

        Returns:
            Tuple of (grid, config_dict).
        """
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        p = self.difficulty_config.params

        # Ensure odd size for proper maze generation.
        if size % 2 == 0:
            size += 1

        n_keys = p.get("n_keys_per_phase", 0)
        action_remap_after = p.get("action_remap_after_phase", -1)

        # --- Generate 3 DFS mazes ---
        maze_terrains: list[np.ndarray] = []
        goal_positions: list[tuple[int, int]] = []
        key_door_data: list[list[dict]] = []  # per phase: list of {key_pos, door_pos}

        # Agent starts near top-left in maze 0.
        first_terrain = _generate_maze_dfs(size, rng)
        agent_start = _find_empty_near(first_terrain, 1, 1, size)

        min_goal_dist = size // 2

        for phase in range(self.N_GOALS):
            # Generate a unique maze for this phase.
            if phase == 0:
                terrain = first_terrain
            else:
                terrain = _generate_maze_dfs(size, rng)

            # Ensure agent start cell is walkable in all mazes.
            terrain[agent_start[1], agent_start[0]] = CellType.EMPTY
            # For phase 0 only, also clear the immediate neighbors of
            # the agent start for breathing room.  For later phases we
            # preserve the DFS maze structure so that chokepoints exist
            # for door placement.  (The shift code in on_agent_moved
            # clears around the agent's actual position at shift time.)
            if phase == 0:
                for dx, dy in _DIRS:
                    nx, ny = agent_start[0] + dx, agent_start[1] + dy
                    if 1 <= nx < size - 1 and 1 <= ny < size - 1:
                        terrain[ny, nx] = CellType.EMPTY

            # Pick goal position far from agent start, and different from
            # all previous goals (the agent will be standing on the previous
            # goal when the shift happens, so the new goal must be elsewhere).
            exclude_goals = {agent_start} | set(goal_positions)

            # Enforce minimum Manhattan distance from agent_start.
            goal_pos = _pick_far_empty(
                terrain, size, rng, agent_start, min_goal_dist, exclude=exclude_goals,
            )

            # If the pick lands on an excluded position, try harder with a
            # far-corner bias.
            if goal_pos in exclude_goals:
                goal_pos = _find_empty_near(terrain, size - 2, size - 2, size)
            if goal_pos in exclude_goals:
                # Last resort: any empty cell not in exclude.
                for y in range(size - 2, 0, -1):
                    for x in range(size - 2, 0, -1):
                        if terrain[y, x] == CellType.EMPTY and (x, y) not in exclude_goals:
                            goal_pos = (x, y)
                            break
                    if goal_pos not in exclude_goals:
                        break

            # Verify solvability (agent_start -> goal); retry maze if needed.
            for _ in range(20):
                reachable = _flood_fill(terrain, agent_start)
                if goal_pos in reachable and goal_pos not in exclude_goals:
                    break
                terrain = _generate_maze_dfs(size, rng)
                terrain[agent_start[1], agent_start[0]] = CellType.EMPTY
                if phase == 0:
                    for ddx, ddy in _DIRS:
                        nnx, nny = agent_start[0] + ddx, agent_start[1] + ddy
                        if 1 <= nnx < size - 1 and 1 <= nny < size - 1:
                            terrain[nny, nnx] = CellType.EMPTY
                goal_pos = _pick_far_empty(
                    terrain, size, rng, agent_start, min_goal_dist, exclude=exclude_goals,
                )
            else:
                # Ultimate fallback: open grid with border walls only.
                terrain = np.full((size, size), CellType.EMPTY, dtype=np.int8)
                terrain[0, :] = CellType.WALL
                terrain[-1, :] = CellType.WALL
                terrain[:, 0] = CellType.WALL
                terrain[:, -1] = CellType.WALL

            maze_terrains.append(terrain.copy())
            goal_positions.append(goal_pos)

            # --- Key-door data for hard/expert (phases 1 and 2 only) ---
            # Doors MUST physically block the goal.  Since clearing
            # agent_start's neighbors can create alternate routes, we
            # test each candidate to confirm it is a true chokepoint
            # (removing it disconnects start from goal).
            phase_kd: list[dict] = []
            if n_keys > 0 and phase > 0:
                path = _bfs_path(terrain, agent_start, goal_pos)
                used = {agent_start, goal_pos}

                for ki in range(n_keys):
                    door_pos = None

                    # Find a true chokepoint on the path: a cell whose
                    # removal disconnects agent_start from goal_pos.
                    if path and len(path) > 2:
                        # Prefer candidates in the middle portion.
                        mid_start = max(1, len(path) // 4)
                        mid_end = min(len(path) - 1, 3 * len(path) // 4)
                        # Try middle first, then full path.
                        for segment in [
                            path[mid_start:mid_end],
                            path[1:-1],
                        ]:
                            chokepoints = []
                            for p in segment:
                                if p in used:
                                    continue
                                # Test: does removing p disconnect start
                                # from goal?
                                blocked = terrain.copy()
                                blocked[p[1], p[0]] = CellType.WALL
                                if goal_pos not in _flood_fill(
                                    blocked, agent_start
                                ):
                                    chokepoints.append(p)
                            if chokepoints:
                                door_pos = chokepoints[
                                    int(rng.integers(len(chokepoints)))
                                ]
                                break

                    if door_pos is None:
                        # No chokepoint found — fallback: random cell on
                        # the path (best effort).
                        if path and len(path) > 2:
                            cands = [
                                p for p in path[1:-1] if p not in used
                            ]
                            if cands:
                                door_pos = cands[
                                    int(rng.integers(len(cands)))
                                ]

                    if door_pos is None:
                        door_pos = _pick_random_empty(
                            terrain, size, rng, exclude=used,
                        )

                    used.add(door_pos)

                    # Place key on the agent's side of the door: flood-fill
                    # from agent_start treating door_pos as a wall, then
                    # pick a random reachable cell.
                    blocked = terrain.copy()
                    blocked[door_pos[1], door_pos[0]] = CellType.WALL
                    agent_side = _flood_fill(blocked, agent_start)
                    key_candidates = [
                        p for p in agent_side
                        if p not in used
                        and terrain[p[1], p[0]] == CellType.EMPTY
                    ]
                    if key_candidates:
                        key_pos = key_candidates[
                            int(rng.integers(len(key_candidates)))
                        ]
                    else:
                        key_pos = _pick_random_empty(
                            terrain, size, rng, exclude=used,
                        )
                    used.add(key_pos)

                    phase_kd.append({
                        "key_pos": list(key_pos),
                        "door_pos": list(door_pos),
                        "color": ki,
                    })

                    # Recompute path for next door (current door is now
                    # treated as passable since agent will have the key).
                    if ki < n_keys - 1:
                        path = _bfs_path(terrain, agent_start, goal_pos)

            key_door_data.append(phase_kd)

        # --- Build the initial grid from maze 0 ---
        grid = Grid(size, size)
        grid.terrain[:, :] = maze_terrains[0]
        gx, gy = goal_positions[0]
        grid.objects[gy, gx] = ObjectType.GOAL

        config = {
            "agent_start": agent_start,
            "goal_positions": [goal_positions[0]],
            "max_steps": self.get_max_steps(),
            # Pre-computed maze data for all 3 phases.
            "_maze_terrains": [t.tolist() for t in maze_terrains],
            "_goal_positions": [list(g) for g in goal_positions],
            "_key_door_data": key_door_data,
            "_action_remap_after_phase": action_remap_after,
            "_action_remap_enabled": action_remap_after >= 0,
            "_action_remap": {},
            "_rng_seed": int(rng.integers(0, 2**31)),
        }

        return grid, config

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_env_reset(self, agent, grid, config):
        """Reset phase counter and dynamic state."""
        config["_phase"] = 0
        config["_goals_reached"] = 0
        config["_action_remap"] = {}
        config["_shift_rng"] = np.random.default_rng(config.get("_rng_seed", 0))
        agent.inventory.clear()
        self._config = config
        self._prev_goals_reached = 0

    def on_agent_moved(self, pos, agent, grid):
        """Detect goal reached and trigger maze shift to next phase."""
        config = getattr(self, "_config", {})
        x, y = pos

        # Pick up keys.
        if grid.objects[y, x] == ObjectType.KEY:
            color = int(grid.metadata[y, x])
            grid.objects[y, x] = ObjectType.NONE
            grid.metadata[y, x] = 0
            agent.inventory.append(
                Entity(
                    id=f"key_{x}_{y}",
                    entity_type="key",
                    position=pos,
                    properties={"color": color},
                )
            )

        # Check if agent reached the goal.
        if grid.objects[y, x] != ObjectType.GOAL:
            return

        goals_reached = config.get("_goals_reached", 0) + 1
        config["_goals_reached"] = goals_reached

        if goals_reached >= self.N_GOALS:
            # All goals reached — success! Don't shift further.
            return

        # --- Shift to next phase ---
        phase = goals_reached  # 0-indexed: next phase = goals_reached
        config["_phase"] = phase

        size = grid.width
        maze_terrains = config.get("_maze_terrains", [])
        goal_positions = config.get("_goal_positions", [])

        if phase >= len(maze_terrains) or phase >= len(goal_positions):
            return

        # Apply new terrain.
        new_terrain = np.array(maze_terrains[phase], dtype=np.int8)

        # Ensure agent's current position is walkable in the new maze.
        ax, ay = pos
        new_terrain[ay, ax] = CellType.EMPTY
        for dx, dy in _DIRS:
            nx, ny = ax + dx, ay + dy
            if 1 <= nx < size - 1 and 1 <= ny < size - 1:
                new_terrain[ny, nx] = CellType.EMPTY

        grid.terrain[:, :] = new_terrain

        # Clear all objects from old phase.
        grid.objects[:, :] = ObjectType.NONE
        grid.metadata[:, :] = 0

        # Place new goal.
        new_goal = tuple(goal_positions[phase])
        # Ensure goal cell is walkable.
        grid.terrain[new_goal[1], new_goal[0]] = CellType.EMPTY
        grid.objects[new_goal[1], new_goal[0]] = ObjectType.GOAL

        # Update config goal_positions for reward shaping.
        config["goal_positions"] = [new_goal]

        # Clear inventory for new phase.
        agent.inventory.clear()

        # --- Place key-door pairs (hard/expert) ---
        kd_data = config.get("_key_door_data", [])
        if phase < len(kd_data):
            for kd in kd_data[phase]:
                kx, ky = kd["key_pos"]
                dx_d, dy_d = kd["door_pos"]
                color = kd["color"]

                # Ensure positions are walkable.
                grid.terrain[ky, kx] = CellType.EMPTY
                grid.terrain[dy_d, dx_d] = CellType.EMPTY

                grid.objects[ky, kx] = ObjectType.KEY
                grid.metadata[ky, kx] = color

                grid.objects[dy_d, dx_d] = ObjectType.DOOR
                grid.metadata[dy_d, dx_d] = color

        # --- Action remap: toggle after specified phase ---
        remap_after = config.get("_action_remap_after_phase", -1)
        if remap_after >= 0 and goals_reached == remap_after:
            config["_action_remap"] = dict(_ACTION_REMAP_FULL)

        # Verify solvability of new phase from agent's current position.
        reachable = _flood_fill(grid.terrain, pos)
        if new_goal not in reachable:
            # Emergency: open a path by clearing walls between agent and goal.
            self._force_path(grid, pos, new_goal, size)

    def can_agent_enter(self, pos, agent, grid) -> bool:
        """Block closed DOOR cells; open doors (meta >= 10) are passable."""
        x, y = pos
        if grid.objects[y, x] == ObjectType.DOOR:
            return int(grid.metadata[y, x]) >= 10
        return True

    def on_agent_interact(self, pos, agent, grid):
        """INTERACT on a closed door with matching key unlocks it."""
        if not grid.in_bounds(pos):
            return
        x, y = pos
        if grid.objects[y, x] != ObjectType.DOOR:
            return
        door_meta = int(grid.metadata[y, x])
        if door_meta >= 10:
            return
        door_color = door_meta
        matching = next(
            (
                e
                for e in agent.inventory
                if e.entity_type == "key"
                and e.properties.get("color") == door_color
            ),
            None,
        )
        if matching:
            agent.inventory.remove(matching)
            grid.metadata[y, x] = door_color + 10

    # ------------------------------------------------------------------
    # Reward and success
    # ------------------------------------------------------------------

    def compute_dense_reward(self, old_state, action, new_state, info):
        """Step penalty + per-goal bonus + distance shaping + final bonus.

        Returns:
            Float reward:
              -0.01 per step
              +0.3 per intermediate goal reached
              +0.05 * (d_old - d_new) distance shaping toward current goal
              +1.0 on reaching all 3 goals (final success)
        """
        reward = -0.01

        config = new_state.get("config", {})
        goals_reached = config.get("_goals_reached", 0)
        prev = getattr(self, "_prev_goals_reached", 0)

        # Per-goal bonus.
        if goals_reached > prev:
            self._prev_goals_reached = goals_reached
            if goals_reached >= self.N_GOALS:
                # Final goal reached.
                reward += 1.0
            else:
                # Intermediate goal.
                reward += 0.3
            return reward

        # Distance-based shaping toward the current goal.
        goals = config.get("goal_positions", [])
        if goals and "agent" in new_state:
            gx, gy = goals[0]
            ax, ay = new_state["agent"].position
            new_dist = abs(ax - gx) + abs(ay - gy)

            old_pos = old_state.get("agent_position", (ax, ay))
            old_dist = abs(old_pos[0] - gx) + abs(old_pos[1] - gy)

            reward += 0.05 * (old_dist - new_dist)

        return reward

    def compute_sparse_reward(self, old_state, action, new_state, info):
        """Sparse reward: +1.0 on full success, 0.0 otherwise."""
        if self.check_success(new_state):
            return 1.0
        return 0.0

    def check_success(self, state):
        """All 3 goals have been reached."""
        config = state.get("config", {})
        return config.get("_goals_reached", 0) >= self.N_GOALS

    def check_done(self, state):
        """Episode ends on success."""
        return self.check_success(state)

    def validate_instance(self, grid, config):
        """Verify agent can reach the first goal in the initial maze."""
        agent_pos = config.get("agent_start")
        goals = config.get("goal_positions", [])
        if not agent_pos or not goals:
            return True
        reachable = grid.flood_fill(agent_pos)
        return goals[0] in reachable

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _force_path(grid, start, goal, size):
        """Emergency path carving: BFS from start to goal, clearing walls.

        Used as a last resort if a maze shift leaves the goal unreachable.
        """
        sx, sy = start
        gx, gy = goal

        visited = {(sx, sy)}
        queue = deque([(sx, sy, [(sx, sy)])])

        while queue:
            x, y, path = queue.popleft()
            if (x, y) == (gx, gy):
                # Clear all walls along this path.
                for px, py in path:
                    if grid.terrain[py, px] == CellType.WALL:
                        grid.terrain[py, px] = CellType.EMPTY
                return

            for dx, dy in _DIRS:
                nx, ny = x + dx, y + dy
                if 1 <= nx < size - 1 and 1 <= ny < size - 1:
                    if (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny, path + [(nx, ny)]))

        # If BFS through entire grid fails (shouldn't happen), just clear straight line.
        cx, cy = sx, sy
        while (cx, cy) != (gx, gy):
            if cx < gx:
                cx += 1
            elif cx > gx:
                cx -= 1
            elif cy < gy:
                cy += 1
            elif cy > gy:
                cy -= 1
            if 0 < cx < size - 1 and 0 < cy < size - 1:
                grid.terrain[cy, cx] = CellType.EMPTY
