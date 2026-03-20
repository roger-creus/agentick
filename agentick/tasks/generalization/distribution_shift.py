"""DistributionShift - Multi-task sequential episode with shifting mechanics.

MECHANICS:
  - Agent completes N phases. Each phase is a DIFFERENT mini-task type.
  - When the agent finishes a phase (reaches GOAL or triggers success condition),
    the grid reconfigures with a new maze layout and new phase mechanics.
  - Phase types (pool of 5):
      goal_reach   : Navigate DFS maze to GOAL.
      key_door     : Collect KEY (auto-pickup), INTERACT on DOOR, reach GOAL.
      lever_barrier: INTERACT on LEVER to remove wall barrier, reach GOAL.
      collection   : Collect N GEMs (walk over them), GOAL appears after all collected.
      box_push     : Push BOX onto TARGET (walk into box), GOAL appears after box on target.
  - No two consecutive phases share the same type.
  - Action remap (UP<->DOWN, LEFT<->RIGHT) activates at hard/expert after a
    specified phase completion.

DIFFICULTY LEVELS:
  - easy:   9x9,  3 phases, no remap, max_steps=200
  - medium: 11x11, 4 phases, no remap, max_steps=350
  - hard:   13x13, 5 phases, remap after phase 3, max_steps=500
  - expert: 17x17, 6 phases, remap after phase 2, max_steps=700
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

_PHASE_TYPES = ["goal_reach", "key_door", "lever_barrier", "collection"]


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


def _build_phase_config(
    phase_type: str,
    terrain: np.ndarray,
    agent_start: tuple[int, int],
    goal_pos: tuple[int, int],
    size: int,
    rng: np.random.Generator,
    phase_idx: int,
) -> dict:
    """Build per-phase data dict for a given phase type and terrain.

    Args:
        phase_type: One of the five phase type strings.
        terrain: DFS-generated maze terrain (will not be mutated).
        agent_start: Fixed agent start position (guaranteed walkable).
        goal_pos: Target GOAL position (used by goal_reach, key_door, lever_barrier).
        size: Grid dimension.
        rng: Seeded RNG.
        phase_idx: 0-based index of this phase (used for gem count scaling).

    Returns:
        Dict with keys: type, terrain (list), goal_pos (list), and
        phase-specific keys (key_pos, door_pos, lever_pos, barrier_pos,
        gem_positions, n_gems, box_pos, target_pos).
    """
    phase_data: dict = {
        "type": phase_type,
        "terrain": terrain.tolist(),
        "goal_pos": list(goal_pos),
    }

    if phase_type == "key_door":
        # Find a chokepoint on the path from agent_start to goal_pos.
        path = _bfs_path(terrain, agent_start, goal_pos)
        door_pos = None
        if path and len(path) > 2:
            # Try middle third first, then full path (excluding endpoints).
            mid_s = max(1, len(path) // 3)
            mid_e = min(len(path) - 1, 2 * len(path) // 3)
            for segment in [path[mid_s:mid_e], path[1:-1]]:
                chokepoints = []
                for p in segment:
                    if p in (agent_start, goal_pos):
                        continue
                    blocked = terrain.copy()
                    blocked[p[1], p[0]] = CellType.WALL
                    if goal_pos not in _flood_fill(blocked, agent_start):
                        chokepoints.append(p)
                if chokepoints:
                    door_pos = chokepoints[int(rng.integers(len(chokepoints)))]
                    break
        # Fallback: midpoint of path
        if door_pos is None and path and len(path) > 2:
            door_pos = path[len(path) // 2]

        if door_pos is not None:
            # Key goes on the agent's side of the door.
            blocked = terrain.copy()
            blocked[door_pos[1], door_pos[0]] = CellType.WALL
            agent_side = _flood_fill(blocked, agent_start)
            key_cands = [
                p for p in agent_side
                if p not in (agent_start, door_pos)
                and terrain[p[1], p[0]] == CellType.EMPTY
            ]
            if key_cands:
                key_pos = key_cands[int(rng.integers(len(key_cands)))]
            else:
                key_pos = agent_start
            phase_data["key_pos"] = list(key_pos)
            phase_data["door_pos"] = list(door_pos)

    elif phase_type == "lever_barrier":
        # Find a single-cell barrier that disconnects agent from goal.
        path = _bfs_path(terrain, agent_start, goal_pos)
        barrier_pos = None
        if path and len(path) > 2:
            mid_s = max(1, len(path) // 3)
            mid_e = min(len(path) - 1, 2 * len(path) // 3)
            for segment in [path[mid_s:mid_e], path[1:-1]]:
                chokepoints = []
                for p in segment:
                    if p in (agent_start, goal_pos):
                        continue
                    blocked = terrain.copy()
                    blocked[p[1], p[0]] = CellType.WALL
                    if goal_pos not in _flood_fill(blocked, agent_start):
                        chokepoints.append(p)
                if chokepoints:
                    barrier_pos = chokepoints[int(rng.integers(len(chokepoints)))]
                    break
        if barrier_pos is None and path and len(path) > 2:
            barrier_pos = path[len(path) // 2]

        if barrier_pos is not None:
            # Lever goes on the agent's side (reachable without crossing barrier).
            blocked = terrain.copy()
            blocked[barrier_pos[1], barrier_pos[0]] = CellType.WALL
            agent_side = _flood_fill(blocked, agent_start)
            lever_cands = [
                p for p in agent_side
                if p not in (agent_start, barrier_pos)
                and terrain[p[1], p[0]] == CellType.EMPTY
            ]
            if lever_cands:
                lever_pos = lever_cands[int(rng.integers(len(lever_cands)))]
            else:
                lever_pos = agent_start
            phase_data["lever_pos"] = list(lever_pos)
            phase_data["barrier_pos"] = list(barrier_pos)

    elif phase_type == "collection":
        # Scale gem count with phase index, cap at 5.
        n_gems = min(2 + phase_idx, 5)
        reachable = list(_flood_fill(terrain, agent_start) - {agent_start, goal_pos})
        if reachable:
            # rng.shuffle requires array-like; use index permutation.
            idxs = rng.permutation(len(reachable))
            gem_positions = [list(reachable[int(i)]) for i in idxs[:min(n_gems, len(reachable))]]
        else:
            gem_positions = []
        phase_data["gem_positions"] = gem_positions
        phase_data["n_gems"] = len(gem_positions)

    elif phase_type == "box_push":
        # Place box and target on a straight corridor (same row or column)
        # so the oracle only needs to push in a single direction.
        reachable = _flood_fill(terrain, agent_start) - {agent_start, goal_pos}
        reachable_list = list(reachable)
        rng.shuffle(reachable_list)
        best_pair = None
        for p in reachable_list:
            px, py = p
            if not (2 <= px <= size - 3 and 2 <= py <= size - 3):
                continue
            # Look for a target 2-4 cells away in a straight line (no walls)
            for ddx, ddy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                for dist in range(2, 5):
                    tx, ty = px + ddx * dist, py + ddy * dist
                    if not (1 <= tx < size - 1 and 1 <= ty < size - 1):
                        break
                    if terrain[ty, tx] != int(CellType.EMPTY):
                        break
                    # Also check push-from cell is clear
                    pfx, pfy = px - ddx, py - ddy
                    if not (1 <= pfx < size - 1 and 1 <= pfy < size - 1):
                        continue
                    if terrain[pfy, pfx] != int(CellType.EMPTY):
                        continue
                    if (tx, ty) in reachable and (pfx, pfy) in reachable:
                        best_pair = (list(p), [tx, ty])
                        break
                if best_pair:
                    break
            if best_pair:
                break
        if best_pair:
            phase_data["box_pos"] = best_pair[0]
            phase_data["target_pos"] = best_pair[1]
        elif len(reachable_list) >= 2:
            # Fallback: pick any two interior cells
            phase_data["box_pos"] = list(reachable_list[0])
            phase_data["target_pos"] = list(reachable_list[1])

    return phase_data


@register_task("DistributionShift-v0", tags=["adversarial", "adaptation"])
class DistributionShiftTask(TaskSpec):
    """Multi-task sequential episode: agent completes N phases of different mini-tasks.

    Each phase uses a fresh DFS maze and a different mechanic (goal_reach,
    key_door, lever_barrier, collection, or box_push). No two consecutive
    phases share the same type. At hard/expert difficulties, action controls
    flip (UP<->DOWN, LEFT<->RIGHT) after a specified phase completion.
    """

    name = "DistributionShift-v0"
    description = "Complete sequential mini-tasks across shifting maze layouts"
    capability_tags = ["adversarial", "adaptation"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=9,
            max_steps=200,
            params={
                "n_phases": 3,
                "action_remap_after_phase": -1,  # -1 = never
            },
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=11,
            max_steps=350,
            params={
                "n_phases": 4,
                "action_remap_after_phase": -1,
            },
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=13,
            max_steps=500,
            params={
                "n_phases": 5,
                "action_remap_after_phase": 3,  # remap after completing phase 3
            },
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=17,
            max_steps=700,
            params={
                "n_phases": 6,
                "action_remap_after_phase": 2,  # remap after completing phase 2
            },
        ),
    }

    def generate(self, seed: int):
        """Generate a multi-phase sequential episode.

        Pre-generates all phase configurations (terrain + objects) and stores
        them in the config dict. The grid is initialised from phase 0; phase
        transitions happen inside on_agent_moved / on_agent_interact.

        Args:
            seed: Random seed for reproducible generation.

        Returns:
            Tuple of (grid, config_dict).
        """
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        p = self.difficulty_config.params

        # Ensure odd size for proper DFS maze generation.
        if size % 2 == 0:
            size += 1

        n_phases: int = p.get("n_phases", 3)
        action_remap_after: int = p.get("action_remap_after_phase", -1)

        # --- Sample phase types (no consecutive repeats) ---
        selected_phases: list[str] = []
        for _ in range(n_phases):
            available = [t for t in _PHASE_TYPES if not selected_phases or t != selected_phases[-1]]
            selected_phases.append(available[int(rng.integers(len(available)))])

        # --- Agent start: fixed across all phases (random corner of first maze) ---
        first_terrain = _generate_maze_dfs(size, rng)
        corners = [(1, 1), (size - 2, 1), (1, size - 2), (size - 2, size - 2)]
        corner = corners[int(rng.integers(len(corners)))]
        agent_start = _find_empty_near(first_terrain, corner[0], corner[1], size)

        min_goal_dist = size // 2

        # --- Pre-generate each phase ---
        phase_configs: list[dict] = []
        for phase_idx, phase_type in enumerate(selected_phases):
            # Generate or reuse terrain.
            if phase_idx == 0:
                terrain = first_terrain.copy()
            else:
                terrain = _generate_maze_dfs(size, rng)

            # Ensure agent_start is walkable; carve neighbours in phase 0.
            terrain[agent_start[1], agent_start[0]] = CellType.EMPTY
            if phase_idx == 0:
                for dx, dy in _DIRS:
                    nx, ny = agent_start[0] + dx, agent_start[1] + dy
                    if 1 <= nx < size - 1 and 1 <= ny < size - 1:
                        terrain[ny, nx] = CellType.EMPTY

            # Pick goal position far from agent_start.
            exclude = {agent_start}
            goal_pos = _pick_far_empty(terrain, size, rng, agent_start, min_goal_dist, exclude)

            # Verify solvability; retry up to 10 times.
            for _ in range(10):
                reachable = _flood_fill(terrain, agent_start)
                if goal_pos in reachable:
                    break
                terrain = _generate_maze_dfs(size, rng)
                terrain[agent_start[1], agent_start[0]] = CellType.EMPTY
                goal_pos = _pick_far_empty(terrain, size, rng, agent_start, min_goal_dist, exclude)
            else:
                # Ultimate fallback: open-border grid.
                terrain = np.full((size, size), CellType.EMPTY, dtype=np.int8)
                terrain[0, :] = CellType.WALL
                terrain[-1, :] = CellType.WALL
                terrain[:, 0] = CellType.WALL
                terrain[:, -1] = CellType.WALL

            phase_data = _build_phase_config(
                phase_type, terrain, agent_start, goal_pos, size, rng, phase_idx,
            )
            phase_configs.append(phase_data)

        # --- Build initial grid from phase 0 ---
        grid = Grid(size, size)
        phase0 = phase_configs[0]
        grid.terrain[:, :] = np.array(phase0["terrain"], dtype=np.int8)
        self._apply_phase_objects(grid, phase0, agent_start)

        config = {
            "agent_start": agent_start,
            "goal_positions": [tuple(phase0["goal_pos"])],
            "max_steps": self.get_max_steps(),
            # Pre-computed phase data.
            "_phase_configs": [
                {k: v for k, v in pc.items()} for pc in phase_configs
            ],
            "_n_phases": n_phases,
            "_action_remap_after_phase": action_remap_after,
            "_action_remap": {},
            "_rng_seed": int(rng.integers(0, 2**31)),
        }
        return grid, config

    # ------------------------------------------------------------------
    # Phase object layout helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_phase_objects(
        grid: Grid,
        phase_data: dict,
        agent_start: tuple[int, int],
    ) -> None:
        """Place all objects for a phase onto the grid (called during init and transitions).

        Clears the objects and metadata layers, then places phase-specific
        objects according to phase_data. For collection/box_push phases the
        GOAL is intentionally NOT placed — it appears only after the phase
        condition is met.

        Args:
            grid: Live grid to modify in-place.
            phase_data: Phase configuration dict from generate().
            agent_start: Kept walkable; used only for safety (not modified here).
        """
        phase_type = phase_data["type"]
        goal_pos = tuple(phase_data["goal_pos"])
        gx, gy = goal_pos

        if phase_type == "goal_reach":
            # Simply place the GOAL.
            grid.terrain[gy, gx] = CellType.EMPTY
            grid.objects[gy, gx] = ObjectType.GOAL

        elif phase_type == "key_door":
            # GOAL visible from the start.
            grid.terrain[gy, gx] = CellType.EMPTY
            grid.objects[gy, gx] = ObjectType.GOAL
            # Place KEY (walkable auto-pickup).
            key_pos = phase_data.get("key_pos")
            door_pos = phase_data.get("door_pos")
            if key_pos:
                kx, ky = key_pos
                grid.terrain[ky, kx] = CellType.EMPTY
                grid.objects[ky, kx] = ObjectType.KEY
                grid.metadata[ky, kx] = 0  # color 0
            if door_pos:
                dx, dy = door_pos
                grid.terrain[dy, dx] = CellType.EMPTY
                grid.objects[dy, dx] = ObjectType.DOOR
                grid.metadata[dy, dx] = 0  # closed (meta < 10)

        elif phase_type == "lever_barrier":
            # GOAL visible from the start.
            grid.terrain[gy, gx] = CellType.EMPTY
            grid.objects[gy, gx] = ObjectType.GOAL
            lever_pos = phase_data.get("lever_pos")
            barrier_pos = phase_data.get("barrier_pos")
            if lever_pos:
                lx, ly = lever_pos
                grid.terrain[ly, lx] = CellType.EMPTY
                grid.objects[ly, lx] = ObjectType.LEVER
            if barrier_pos:
                bx, by = barrier_pos
                # The barrier is a solid WALL; the lever interaction removes it.
                grid.terrain[by, bx] = CellType.WALL

        elif phase_type == "collection":
            # GOAL is hidden until all gems collected.
            for gem_xy in phase_data.get("gem_positions", []):
                gemx, gemy = gem_xy
                grid.terrain[gemy, gemx] = CellType.EMPTY
                grid.objects[gemy, gemx] = ObjectType.GEM
            # Do NOT place GOAL yet.

        elif phase_type == "box_push":
            # GOAL is hidden until box reaches target.
            box_pos = phase_data.get("box_pos")
            target_pos = phase_data.get("target_pos")
            if box_pos:
                bx, by = box_pos
                grid.terrain[by, bx] = CellType.EMPTY
                grid.objects[by, bx] = ObjectType.BOX
            if target_pos:
                tx, ty = target_pos
                grid.terrain[ty, tx] = CellType.EMPTY
                grid.objects[ty, tx] = ObjectType.TARGET
            # Do NOT place GOAL yet.

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_env_reset(self, agent, grid, config):
        """Reset phase counter and all dynamic state."""
        config["_phases_completed"] = 0
        config["_current_phase_idx"] = 0
        phase_configs = config.get("_phase_configs", [])
        if phase_configs:
            config["_current_phase_type"] = phase_configs[0]["type"]
        else:
            config["_current_phase_type"] = "goal_reach"
        config["_action_remap"] = {}
        config["_gems_collected"] = 0
        config["_gems_needed"] = phase_configs[0].get("n_gems", 0) if phase_configs else 0
        config["_box_on_target"] = False
        # Cache current barrier / target pos for runtime use.
        if phase_configs:
            pc0 = phase_configs[0]
            config["_current_barrier_pos"] = pc0.get("barrier_pos")
            config["_current_target_pos"] = pc0.get("target_pos")
        else:
            config["_current_barrier_pos"] = None
            config["_current_target_pos"] = None
        config["_barrier_opened"] = False
        agent.inventory.clear()
        self._config = config
        self._prev_phases_completed = 0

    def _advance_phase(self, agent, grid, config):
        """Transition the grid to the next phase.

        Increments phases_completed, loads the next phase's terrain and
        objects, clears inventory, updates cached config keys, and triggers
        the action remap if the threshold is reached. If all phases are done,
        marks completion and returns immediately.

        Args:
            agent: Live agent (inventory cleared).
            grid: Live grid (mutated in-place).
            config: Task config dict (mutated in-place).
        """
        phases_completed = config.get("_phases_completed", 0) + 1
        config["_phases_completed"] = phases_completed
        n_phases = config.get("_n_phases", 3)

        if phases_completed >= n_phases:
            # All phases done — do not reconfigure grid.
            return

        next_idx = phases_completed  # next phase index
        config["_current_phase_idx"] = next_idx
        phase_configs = config.get("_phase_configs", [])

        if next_idx >= len(phase_configs):
            return

        next_phase = phase_configs[next_idx]
        size = grid.width
        agent_pos = agent.position

        # Load new terrain, ensure agent position is walkable.
        new_terrain = np.array(next_phase["terrain"], dtype=np.int8)
        new_terrain[agent_pos[1], agent_pos[0]] = CellType.EMPTY
        for dx, dy in _DIRS:
            nx, ny = agent_pos[0] + dx, agent_pos[1] + dy
            if 1 <= nx < size - 1 and 1 <= ny < size - 1:
                new_terrain[ny, nx] = CellType.EMPTY

        grid.terrain[:, :] = new_terrain

        # Clear all objects and metadata from old phase.
        grid.objects[:, :] = ObjectType.NONE
        grid.metadata[:, :] = 0

        # Place next-phase objects.
        agent_start = config.get("agent_start", agent_pos)
        self._apply_phase_objects(grid, next_phase, agent_start)

        # Update config state for new phase.
        config["_current_phase_type"] = next_phase["type"]
        config["goal_positions"] = [tuple(next_phase["goal_pos"])]
        config["_gems_collected"] = 0
        config["_gems_needed"] = next_phase.get("n_gems", 0)
        config["_box_on_target"] = False
        config["_current_barrier_pos"] = next_phase.get("barrier_pos")
        config["_current_target_pos"] = next_phase.get("target_pos")
        config["_barrier_opened"] = False

        # Clear inventory for the new phase.
        agent.inventory.clear()

        # Ensure new goal reachable; emergency path-carve if not.
        goal_pos = tuple(next_phase["goal_pos"])
        # For collection/box_push the GOAL is not placed yet — skip.
        # For lever_barrier the barrier intentionally blocks — skip.
        if next_phase["type"] in ("goal_reach", "key_door"):
            reachable = _flood_fill(grid.terrain, agent_pos)
            if goal_pos not in reachable:
                self._force_path(grid, agent_pos, goal_pos, size)
        elif next_phase["type"] == "lever_barrier":
            # Verify lever is reachable (goal is blocked by design)
            lever_pos = next_phase.get("lever_pos")
            if lever_pos:
                reachable = _flood_fill(grid.terrain, agent_pos)
                if tuple(lever_pos) not in reachable:
                    self._force_path(grid, agent_pos, tuple(lever_pos), size)

        # Action remap: activate after specified phase completion count.
        remap_after = config.get("_action_remap_after_phase", -1)
        if remap_after >= 0 and phases_completed == remap_after:
            config["_action_remap"] = dict(_ACTION_REMAP_FULL)

    # ------------------------------------------------------------------
    # Phase-specific movement and interaction hooks
    # ------------------------------------------------------------------

    def can_agent_enter(self, pos, agent, grid) -> bool:
        """Dispatch to phase-specific entry logic.

        - key_door: DOOR blocks unless metadata >= 10 (open).
        - lever_barrier: LEVER is solid (face + INTERACT required).
        - box_push: Walking into BOX triggers a push attempt.
        - All other phases: terrain decides (default walkable).

        Args:
            pos: (x, y) cell the agent is trying to enter.
            agent: Agent entity.
            grid: Live grid.

        Returns:
            True if the agent may enter the cell, False otherwise.
        """
        x, y = pos
        config = getattr(self, "_config", {})
        phase_type = config.get("_current_phase_type", "goal_reach")

        if phase_type == "key_door":
            if grid.objects[y, x] == ObjectType.DOOR:
                return int(grid.metadata[y, x]) >= 10
            return True

        if phase_type == "lever_barrier":
            # LEVER is physically solid.
            if grid.objects[y, x] == ObjectType.LEVER:
                return False
            return True

        if phase_type == "box_push":
            if grid.objects[y, x] == ObjectType.BOX:
                return self._try_push_box(x, y, agent, grid, config)
            return True

        return True

    def _try_push_box(self, bx: int, by: int, agent, grid, config: dict) -> bool:
        """Attempt to push the BOX at (bx, by) in the agent's direction of travel.

        If the cell beyond the box is clear (not a wall and not another box),
        the box slides one step. If the box lands on TARGET, the goal spawns.

        Args:
            bx: Box x coordinate.
            by: Box y coordinate.
            agent: Agent entity (used for direction).
            grid: Live grid (mutated on successful push).
            config: Task config dict (mutated on success condition).

        Returns:
            True if agent may enter the box's current cell (push succeeded),
            False if push is blocked.
        """
        ax, ay = agent.position
        dx = bx - ax
        dy = by - ay
        nx, ny = bx + dx, by + dy

        if not (0 <= nx < grid.width and 0 <= ny < grid.height):
            return False
        if grid.terrain[ny, nx] == CellType.WALL:
            return False
        if grid.objects[ny, nx] in (ObjectType.BOX, ObjectType.LEVER, ObjectType.DOOR):
            return False

        # Move box.
        grid.objects[by, bx] = ObjectType.NONE

        # If this cell was also the TARGET (box leaving target), restore TARGET.
        target_pos = config.get("_current_target_pos")
        if target_pos and tuple(target_pos) == (bx, by):
            grid.objects[by, bx] = ObjectType.TARGET

        grid.objects[ny, nx] = ObjectType.BOX

        # Check if box landed on target.
        if target_pos and tuple(target_pos) == (nx, ny):
            config["_box_on_target"] = True
            # Spawn goal.
            goal_pos = config.get("goal_positions", [None])[0]
            if goal_pos:
                gx, gy = goal_pos
                grid.terrain[gy, gx] = CellType.EMPTY
                grid.objects[gy, gx] = ObjectType.GOAL
        else:
            config["_box_on_target"] = False

        return True

    def on_agent_interact(self, pos, agent, grid):
        """Dispatch INTERACT to phase-specific handlers.

        - key_door: INTERACT on closed DOOR with matching key opens it.
        - lever_barrier: INTERACT on LEVER opens the wall barrier.

        Args:
            pos: (x, y) cell the agent is facing (interact target).
            agent: Agent entity.
            grid: Live grid (mutated on success).
        """
        if not grid.in_bounds(pos):
            return
        x, y = pos
        config = getattr(self, "_config", {})
        phase_type = config.get("_current_phase_type", "goal_reach")

        if phase_type == "key_door":
            if grid.objects[y, x] != ObjectType.DOOR:
                return
            if int(grid.metadata[y, x]) >= 10:
                return  # already open
            door_color = int(grid.metadata[y, x])
            matching = next(
                (
                    e for e in agent.inventory
                    if e.entity_type == "key"
                    and e.properties.get("color") == door_color
                ),
                None,
            )
            if matching:
                agent.inventory.remove(matching)
                grid.metadata[y, x] = door_color + 10  # open flag

        elif phase_type == "lever_barrier":
            if grid.objects[y, x] != ObjectType.LEVER:
                return
            barrier_pos = config.get("_current_barrier_pos")
            if barrier_pos and not config.get("_barrier_opened", False):
                bx, by = barrier_pos
                grid.terrain[by, bx] = CellType.EMPTY
                config["_barrier_opened"] = True

    def on_agent_moved(self, pos, agent, grid):
        """Handle per-step pickups and GOAL detection.

        - key_door: Pick up KEY on walk.
        - collection: Collect GEM on walk; place GOAL when all gems collected.
        - GOAL reached: advance phase or mark success.

        Args:
            pos: (x, y) cell the agent just moved into.
            agent: Agent entity.
            grid: Live grid (mutated for pickups / goal placement).
        """
        config = getattr(self, "_config", {})
        x, y = pos
        phase_type = config.get("_current_phase_type", "goal_reach")

        # --- Phase-specific on-enter effects ---
        if phase_type == "key_door":
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

        elif phase_type == "collection":
            if grid.objects[y, x] == ObjectType.GEM:
                grid.objects[y, x] = ObjectType.NONE
                collected = config.get("_gems_collected", 0) + 1
                config["_gems_collected"] = collected
                needed = config.get("_gems_needed", 0)
                if collected >= needed:
                    # All gems gathered: spawn GOAL.
                    goal_pos = config.get("goal_positions", [None])[0]
                    if goal_pos:
                        gx, gy = goal_pos
                        grid.terrain[gy, gx] = CellType.EMPTY
                        grid.objects[gy, gx] = ObjectType.GOAL

        # --- GOAL detection: advance phase or complete episode ---
        if grid.objects[y, x] == ObjectType.GOAL:
            self._advance_phase(agent, grid, config)

    def on_env_step(self, agent, grid, config, step_count=0):
        """Called each step; nothing extra needed for this task."""

    # ------------------------------------------------------------------
    # Reward and success
    # ------------------------------------------------------------------

    def compute_dense_reward(self, old_state, action, new_state, info):
        """Stepped penalty + per-phase bonus + distance shaping + final bonus.

        Returns:
            Float reward:
              -0.01 per step
              +0.3 per completed intermediate phase
              +0.05 * (d_old - d_new) distance shaping toward current goal
              +1.0 on completing all phases (final success)
        """
        reward = -0.01

        config = new_state.get("config", {})
        phases_completed = config.get("_phases_completed", 0)
        prev = getattr(self, "_prev_phases_completed", 0)

        if phases_completed > prev:
            self._prev_phases_completed = phases_completed
            n_phases = config.get("_n_phases", 3)
            if phases_completed >= n_phases:
                reward += 1.0
            else:
                reward += 0.3
            return reward

        # Distance-based shaping toward current goal.
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
        """All phases have been completed."""
        config = state.get("config", {})
        n_phases = config.get("_n_phases", 3)
        return config.get("_phases_completed", 0) >= n_phases

    def check_done(self, state):
        """Episode ends on success."""
        return self.check_success(state)

    def validate_instance(self, grid, config):
        """Verify the phase-0 instance is solvable.

        For goal_reach and key_door: goal must be reachable in initial terrain
        (key_door barrier is placed as DOOR object, which is passable in flood_fill).
        For lever_barrier: barrier is a WALL blocking the goal — that is intentional
        (the lever opens it), so we only check that the lever itself is reachable.
        For collection/box_push: goal is not placed initially — always valid.
        """
        agent_pos = config.get("agent_start")
        if not agent_pos:
            return True
        phase_configs = config.get("_phase_configs", [])
        if not phase_configs:
            return True
        phase0 = phase_configs[0]
        phase_type = phase0["type"]

        if phase_type in ("collection", "box_push"):
            # Goal not placed yet; check agent start is on walkable terrain.
            return True

        if phase_type == "lever_barrier":
            # The barrier wall is intentionally blocking; validate that the
            # lever is reachable from agent_start.
            lever_pos = phase0.get("lever_pos")
            if not lever_pos:
                return True
            reachable = grid.flood_fill(agent_pos)
            return tuple(lever_pos) in reachable

        # goal_reach / key_door: check goal reachable (DOOR object doesn't
        # block terrain flood_fill).
        goals = config.get("goal_positions", [])
        if not goals:
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

        Args:
            grid: Live grid (terrain mutated in-place).
            start: (x, y) agent position.
            goal: (x, y) goal position.
            size: Grid dimension (for border guard).
        """
        sx, sy = start
        gx, gy = goal

        visited = {(sx, sy)}
        queue = deque([(sx, sy, [(sx, sy)])])

        while queue:
            x, y, path = queue.popleft()
            if (x, y) == (gx, gy):
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

        # Straight-line fallback (should never reach here with a border-walled grid).
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
