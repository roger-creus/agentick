"""InstructionFollowing - Navigate to unique target object, avoid touching wrong objects.

MECHANICS:
  - ONE unique target object per episode (GEM, SCROLL, ORB, or COIN)
  - Multiple distractor objects of OTHER types scattered on the map
  - Touching ANY distractor = -1.0 reward + episode ends in failure
  - At hard+: keys + doors gate the room containing the target
  - Keys and doors are safe to touch/collect (not penalty objects)
"""

from __future__ import annotations

from collections import deque

import numpy as np

from agentick.core.entity import Entity
from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task

_TARGET_TYPES = [ObjectType.GEM, ObjectType.SCROLL, ObjectType.ORB, ObjectType.COIN]


@register_task("InstructionFollowing-v0", tags=["language", "grounding", "instruction"])
class InstructionFollowingTask(TaskSpec):
    """Navigate to unique target object, avoid touching wrong objects."""

    name = "InstructionFollowing-v0"
    description = "Navigate to unique target object, avoid touching wrong objects"
    capability_tags = ["language", "grounding", "instruction"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=7,
            max_steps=60,
            params={"n_distractors": 3, "n_doors": 0},
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=10,
            max_steps=100,
            params={"n_distractors": 6, "n_doors": 0},
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=13,
            max_steps=180,
            params={"n_distractors": 8, "n_doors": 1},
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=15,
            max_steps=280,
            params={"n_distractors": 12, "n_doors": 2},
        ),
    }

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        n_distractors = self.difficulty_config.params.get("n_distractors", 3)
        n_doors = self.difficulty_config.params.get("n_doors", 0)

        grid = Grid(size, size)
        # Border walls
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        # Randomize agent start from corners
        corners = [(1, 1), (size - 2, 1), (1, size - 2), (size - 2, size - 2)]
        agent_pos = tuple(corners[int(rng.integers(0, len(corners)))])
        used = {agent_pos}

        # Pick target type via RNG
        target_idx = int(rng.integers(0, len(_TARGET_TYPES)))
        target_type = _TARGET_TYPES[target_idx]
        distractor_types = [t for t in _TARGET_TYPES if t != target_type]

        # ----------------------------------------------------------
        # Room layout with doors (hard / expert)
        # ----------------------------------------------------------
        key_positions = []
        door_positions = []

        if n_doors > 0:
            # Strategy: create horizontal wall dividers splitting the grid
            # into (n_doors + 1) horizontal bands.  Each wall has one door.
            # Keys are placed in the band BEFORE the door they open.
            band_height = (size - 2) // (n_doors + 1)  # interior height per band
            for di in range(n_doors):
                wall_y = 1 + band_height * (di + 1)
                if wall_y >= size - 1:
                    wall_y = size - 2
                # Build horizontal wall
                for wx in range(1, size - 1):
                    grid.terrain[wall_y, wx] = CellType.WALL

                # Place door at a random x in this wall
                door_x = int(rng.integers(1, size - 1))
                grid.terrain[wall_y, door_x] = CellType.EMPTY  # clear terrain
                grid.objects[wall_y, door_x] = ObjectType.DOOR
                grid.metadata[wall_y, door_x] = di  # door color index
                door_positions.append((door_x, wall_y))
                used.add((door_x, wall_y))

                # Place key in the band ABOVE this wall (accessible area)
                band_top = 1 if di == 0 else (1 + band_height * di + 1)
                band_bot = wall_y - 1
                band_cells = [
                    (cx, cy)
                    for cx in range(1, size - 1)
                    for cy in range(band_top, band_bot + 1)
                    if (cx, cy) not in used
                    and grid.terrain[cy, cx] == CellType.EMPTY
                    and grid.objects[cy, cx] == ObjectType.NONE
                ]
                if band_cells:
                    kp = band_cells[int(rng.integers(0, len(band_cells)))]
                    grid.objects[kp[1], kp[0]] = ObjectType.KEY
                    grid.metadata[kp[1], kp[0]] = di  # key color matches door
                    key_positions.append(kp)
                    used.add(kp)

        # ----------------------------------------------------------
        # Collect free cells (cells not yet used)
        # ----------------------------------------------------------
        def _free_cells():
            return [
                (x, y)
                for x in range(1, size - 1)
                for y in range(1, size - 1)
                if (x, y) not in used
                and grid.terrain[y, x] == CellType.EMPTY
                and grid.objects[y, x] == ObjectType.NONE
            ]

        # ----------------------------------------------------------
        # Place target object
        # ----------------------------------------------------------
        # If there are doors, place the target behind the LAST door.
        if n_doors > 0:
            last_wall_y = door_positions[-1][1]
            behind_cells = [(x, y) for x, y in _free_cells() if y > last_wall_y]
            if behind_cells:
                target_pos = behind_cells[int(rng.integers(0, len(behind_cells)))]
            else:
                # Fallback: any free cell
                free = _free_cells()
                target_pos = free[int(rng.integers(0, len(free)))]
        else:
            free = _free_cells()
            target_pos = free[int(rng.integers(0, len(free)))]

        grid.objects[target_pos[1], target_pos[0]] = int(target_type)
        used.add(target_pos)

        # ----------------------------------------------------------
        # Place distractor objects (with path-safety retry)
        # ----------------------------------------------------------
        # Build a buffer zone around doors, keys, and agent start so
        # distractors never block critical paths.
        no_distractor = set()
        for bx, by in list(door_positions) + list(key_positions) + [agent_pos]:
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    no_distractor.add((bx + dx, by + dy))

        def _distractor_cells():
            return [(x, y) for x, y in _free_cells() if (x, y) not in no_distractor]

        distractor_positions = []
        for _attempt in range(20):
            # Clear any previously placed distractors
            for dp in distractor_positions:
                grid.objects[dp[1], dp[0]] = ObjectType.NONE
                used.discard(dp)
            distractor_positions = []

            # Place distractors
            for _ in range(n_distractors):
                free = _distractor_cells()
                if not free:
                    # Fall back to any free cell if we run out of non-buffer cells
                    free = _free_cells()
                if not free:
                    break
                dp = free[int(rng.integers(0, len(free)))]
                dt = distractor_types[int(rng.integers(0, len(distractor_types)))]
                grid.objects[dp[1], dp[0]] = int(dt)
                distractor_positions.append(dp)
                used.add(dp)

            # Validate that distractor-free paths exist
            blocked = set(distractor_positions)
            paths_ok = True

            if n_doors == 0:
                # Simple case: agent must reach target avoiding distractors
                if not self._bfs_reachable(grid, agent_pos, target_pos, blocked):
                    paths_ok = False
            else:
                # Key/door case: check sequential reachability
                # Agent → key[0], then key[0] → door[0] adjacent, ...
                # then final position → target
                current = agent_pos
                for i, kp in enumerate(key_positions):
                    if not self._bfs_reachable(grid, current, kp, blocked):
                        paths_ok = False
                        break
                    current = kp
                    # After picking up key, need to reach a cell adjacent to
                    # door[i] (doors are solid, agent stands next to them)
                    if i < len(door_positions):
                        dp = door_positions[i]
                        # Check that at least one adjacent cell of the door is
                        # reachable without crossing distractors
                        door_adj_ok = False
                        for ddx, ddy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                            adj = (dp[0] + ddx, dp[1] + ddy)
                            if (
                                grid.in_bounds(adj)
                                and grid.terrain[adj[1], adj[0]] == CellType.EMPTY
                                and adj not in blocked
                            ):
                                if self._bfs_reachable(grid, current, adj, blocked):
                                    current = adj
                                    door_adj_ok = True
                                    break
                        if not door_adj_ok:
                            paths_ok = False
                            break
                if paths_ok:
                    # After all doors, check target reachability.
                    # Target is behind the last door, so we need to account
                    # for the opened door passage. Use a simple BFS that
                    # treats door cells as passable (they will be opened).
                    door_cells = set(door_positions)
                    if not self._bfs_reachable(
                        grid, current, target_pos, blocked, passable=door_cells,
                    ):
                        paths_ok = False

            if paths_ok:
                break
        else:
            # All 20 attempts failed — use last placement anyway
            pass

        # ----------------------------------------------------------
        # Validate reachability
        # ----------------------------------------------------------
        # For no-door layouts, basic flood_fill suffices.
        # For door layouts, we do a key-aware reachability check.
        if n_doors == 0:
            reachable = grid.flood_fill(agent_pos)
            if target_pos not in reachable:
                # Retry with a different seed — should be extremely rare
                return self.generate(seed + 1000)
        else:
            if not self._validate_key_door_path(
                grid, agent_pos, target_pos, key_positions, door_positions
            ):
                return self.generate(seed + 1000)

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [target_pos],
            "target_type": int(target_type),
            "distractor_positions": distractor_positions,
            "key_positions": key_positions,
            "door_positions": door_positions,
            "max_steps": self.get_max_steps(),
        }

    # ------------------------------------------------------------------
    # Key-aware reachability validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_key_door_path(grid, start, target, key_positions, door_positions):
        """Check that agent can reach all keys and the target sequentially.

        Simulates: from start, reach key_0 (open door_0), reach key_1
        (open door_1), ..., then reach target.
        """
        current = start
        opened_doors = set()

        def _reachable_from(pos):
            """Flood-fill that treats opened doors as passable."""
            visited = {pos}
            q = deque([pos])
            while q:
                cx, cy = q.popleft()
                for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                    nx, ny = cx + dx, cy + dy
                    if (nx, ny) in visited:
                        continue
                    if not grid.in_bounds((nx, ny)):
                        continue
                    if grid.terrain[ny, nx] == CellType.WALL:
                        # Allow if it is an opened door
                        continue
                    # Check for closed door
                    if grid.objects[ny, nx] == ObjectType.DOOR:
                        if (nx, ny) not in opened_doors:
                            continue
                    visited.add((nx, ny))
                    q.append((nx, ny))
            return visited

        for i, kp in enumerate(key_positions):
            reachable = _reachable_from(current)
            if kp not in reachable:
                return False
            current = kp
            # Opening door i
            if i < len(door_positions):
                opened_doors.add(door_positions[i])

        # Finally, check target is reachable
        reachable = _reachable_from(current)
        return target in reachable

    @staticmethod
    def _bfs_reachable(grid, start, end, blocked, passable=None):
        """BFS from *start* to *end* treating *blocked* cells as walls.

        Args:
            grid: The Grid object.
            start: (x, y) start position.
            end: (x, y) target position.
            blocked: Set of (x, y) positions to treat as impassable.
            passable: Optional set of (x, y) positions to treat as passable
                even if they contain blocking objects (e.g. door cells that
                will be opened).

        Returns:
            True if *end* is reachable from *start* without touching *blocked*.
        """
        if start == end:
            return True
        passable = passable or set()
        visited = {start}
        q = deque([start])
        while q:
            cx, cy = q.popleft()
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = cx + dx, cy + dy
                npos = (nx, ny)
                if npos in visited or npos in blocked:
                    continue
                if not grid.in_bounds(npos):
                    continue
                if grid.terrain[ny, nx] == CellType.WALL:
                    continue
                # Allow door cells if they are in the passable set
                if (
                    grid.objects[ny, nx] == ObjectType.DOOR
                    and npos not in passable
                ):
                    continue
                visited.add(npos)
                if npos == end:
                    return True
                q.append(npos)
        return False

    # ------------------------------------------------------------------
    # Runtime hooks
    # ------------------------------------------------------------------

    def on_env_reset(self, agent, grid, config):
        """Reset wrong-touch flag and config reference. Clear inventory."""
        config["_wrong_touch"] = False
        self._config = config
        agent.inventory.clear()

    def on_agent_moved(self, pos, agent, grid):
        """Handle stepping on objects: keys auto-pickup, distractors penalize."""
        x, y = pos
        config = getattr(self, "_config", {})
        obj = grid.objects[y, x]

        if obj == ObjectType.NONE:
            return

        target_type = config.get("target_type")
        target_pos = config.get("goal_positions", [None])[0]

        # Key pickup (safe)
        if obj == ObjectType.KEY:
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
            return

        # Target object reached — remove from grid so check_success sees
        # the agent at the target position
        if target_pos is not None and (x, y) == tuple(target_pos) and obj == target_type:
            grid.objects[y, x] = ObjectType.NONE
            return

        # Any of the four collectible types that is NOT the target is a
        # distractor.  Stepping on it ends the episode with a penalty.
        if obj in (ObjectType.GEM, ObjectType.SCROLL, ObjectType.ORB, ObjectType.COIN):
            if obj != target_type or (x, y) != tuple(target_pos or ()):
                config["_wrong_touch"] = True
            return

    def can_agent_enter(self, pos, agent, grid) -> bool:
        """Block closed doors; allow open doors and all other cells."""
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
                if e.entity_type == "key" and e.properties.get("color") == door_color
            ),
            None,
        )
        if matching:
            agent.inventory.remove(matching)
            grid.metadata[y, x] = door_color + 10

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def compute_sparse_reward(self, old_state, action, new_state, info):
        if new_state.get("config", {}).get("_wrong_touch", False):
            return -1.0
        if self.check_success(new_state):
            return 1.0
        return 0.0

    def compute_dense_reward(self, old_state, action, new_state, info):
        config = new_state.get("config", {})
        if config.get("_wrong_touch", False):
            return -1.0

        reward = -0.01  # step penalty

        # Approach shaping toward target
        goal = config.get("goal_positions", [None])[0]
        if goal and "agent" in new_state:
            ax, ay = new_state["agent"].position
            ox, oy = old_state.get("agent_position", new_state["agent"].position)
            old_d = abs(ox - goal[0]) + abs(oy - goal[1])
            new_d = abs(ax - goal[0]) + abs(ay - goal[1])
            reward += 0.05 * (old_d - new_d)

        if self.check_success(new_state):
            reward += 1.0

        return reward

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def check_success(self, state):
        config = state.get("config", {})
        if config.get("_wrong_touch", False):
            return False
        if "agent" not in state:
            return False
        target_pos = config.get("goal_positions", [None])[0]
        if target_pos is None:
            return False
        x, y = state["agent"].position
        return (x, y) == tuple(target_pos)

    def check_done(self, state):
        if self.check_success(state):
            return True
        config = state.get("config", {})
        return config.get("_wrong_touch", False)

    # ------------------------------------------------------------------
    # Validation & baselines
    # ------------------------------------------------------------------

    def validate_instance(self, grid, config):
        agent_start = config.get("agent_start")
        goal = config.get("goal_positions", [None])[0]
        if agent_start is None or goal is None:
            return True
        n_doors = len(config.get("door_positions", []))
        if n_doors == 0:
            reachable = grid.flood_fill(agent_start)
            return goal in reachable
        return self._validate_key_door_path(
            grid,
            agent_start,
            goal,
            config.get("key_positions", []),
            config.get("door_positions", []),
        )

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0
