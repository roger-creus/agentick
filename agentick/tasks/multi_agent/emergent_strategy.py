"""EmergentStrategy - Exploit NPC behaviors to solve pressure-plate puzzles.

MECHANICS:
  - The GOAL is behind one or more WALL barriers (full-width horizontal rows)
  - Each barrier is opened by placing an NPC on the matching pressure plate (TARGET)
  - Four NPC types, ALL rendered as ObjectType.NPC with metadata-based colors:
    * Follower (meta=1, cyan): BFS toward cell adjacent to agent (stays 1 tile behind)
    * Fearful (meta=3, green): flees from agent when within Manhattan distance 3
    * Mirror (meta=4, purple): moves toward the mirrored position of the agent
    * Contrarian (meta=5, gold): moves opposite to agent's last action
  - Zone-gated: NPCs only react when the agent is in the same zone (no closed
    barrier between them). No idle wandering — purely agent-controlled.
  - NPCs only move as a direct reaction to agent proximity/actions
  - Agent cannot push NPCs directly -- must exploit their AI behaviors
  - Plates LOCK: NPC on TARGET = NPC locks in place permanently, barrier stays open
  - Each plate has ≥2 adjacent walkable cells so NPCs can approach from multiple dirs

EMERGENT STRATEGY:
  The "emergent" aspect: the agent must DISCOVER that NPC behaviors can be
  exploited to solve the puzzle. The strategy of "use the follower NPC as a
  weight for the pressure plate" is not explicitly given -- it EMERGES from
  understanding NPC behavior.

DIFFICULTY AXES:
  - easy: 1 follower, 1 plate, 9x9 grid
  - medium: 1 follower + 1 fearful, 2 plates, 12x12 grid
  - hard: 1 follower + 1 fearful + 1 contrarian, 3 plates, 15x15 grid
  - expert: 1 follower + 1 fearful + 1 contrarian + 1 mirror, 4 plates, 18x18 grid
"""

from __future__ import annotations

from collections import deque

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task

# All NPCs use ObjectType.NPC; metadata encodes behavior type for color coding
_NPC_OBJ_TYPE = int(ObjectType.NPC)
_NPC_META: dict[str, int] = {
    "follower": 1,    # cyan
    "fearful": 3,     # green
    "mirror": 4,      # purple
    "contrarian": 5,  # gold
}

_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


@register_task("EmergentStrategy-v0", tags=["skill_composition", "long_horizon"])
class EmergentStrategyTask(TaskSpec):
    """Exploit NPC behaviors to activate pressure plates and reach the goal."""

    name = "EmergentStrategy-v0"
    description = "Lure, scare, or exploit NPCs onto pressure plates to open barriers"
    capability_tags = ["skill_composition", "long_horizon"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=9,
            max_steps=100,
            params={
                "n_plates": 1,
                "npc_types": ["follower"],
            },
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=12,
            max_steps=200,
            params={
                "n_plates": 2,
                "npc_types": ["follower", "fearful"],
            },
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=15,
            max_steps=350,
            params={
                "n_plates": 3,
                "npc_types": ["follower", "fearful", "contrarian"],
            },
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=18,
            max_steps=500,
            params={
                "n_plates": 4,
                "npc_types": ["follower", "fearful", "contrarian", "mirror"],
            },
        ),
    }

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, seed: int):  # noqa: C901
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        p = self.difficulty_config.params
        n_plates: int = p["n_plates"]
        npc_types: list[str] = list(p["npc_types"])

        grid = Grid(size, size)

        # Outer walls
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        # -- Barrier rows ------------------------------------------------
        # Divide the interior into (n_plates + 1) zones with horizontal
        # WALL rows. Zone 0 is where the agent starts; zone n_plates has
        # the goal.
        interior_h = size - 2  # rows 1 .. size-2
        # Evenly space barrier rows
        barrier_rows: list[int] = []
        for i in range(1, n_plates + 1):
            row = 1 + int(i * interior_h / (n_plates + 1))
            # Clamp to interior
            row = max(2, min(size - 3, row))
            barrier_rows.append(row)
        # Deduplicate (can happen on very small grids)
        barrier_rows = sorted(set(barrier_rows))
        # Trim to n_plates
        barrier_rows = barrier_rows[:n_plates]

        # Build barrier cells per index (full-width wall, no gap)
        barrier_cells: dict[int, list[tuple[int, int]]] = {}
        for idx, br in enumerate(barrier_rows):
            cells = []
            for x in range(1, size - 1):
                grid.terrain[br, x] = CellType.WALL
                cells.append((x, br))
            barrier_cells[idx] = cells

        # -- Compute zone y-ranges ----------------------------------------
        # zone_ranges[i] = (y_min, y_max) inclusive for zone i
        zone_boundaries = [1] + [br for br in barrier_rows] + [size - 2]
        zone_ranges: list[tuple[int, int]] = []
        for i in range(len(zone_boundaries) - 1):
            y_min = zone_boundaries[i]
            y_max = zone_boundaries[i + 1]
            if i > 0:
                y_min += 1  # skip the barrier row itself
            zone_ranges.append((y_min, y_max))

        def _zone_cells(zi: int) -> list[tuple[int, int]]:
            y_lo, y_hi = zone_ranges[zi]
            cells = []
            for y in range(y_lo, y_hi + 1):
                for x in range(1, size - 1):
                    if grid.terrain[y, x] == CellType.EMPTY:
                        cells.append((x, y))
            return cells

        # -- Place pressure plates (TARGET) in zones BEFORE their barrier --
        plate_positions: list[tuple[int, int]] = []
        plate_barrier_map: list[int] = []  # plate index -> barrier index
        for idx in range(n_plates):
            zone_idx = idx  # plate for barrier idx goes in zone idx
            cells = _zone_cells(zone_idx)
            # Remove cells where objects already placed
            cells = [
                c for c in cells
                if grid.objects[c[1], c[0]] == ObjectType.NONE
            ]
            if not cells:
                # Fallback: place in center of zone
                y_lo, y_hi = zone_ranges[zone_idx]
                mid_y = (y_lo + y_hi) // 2
                cells = [(size // 2, mid_y)]
            # Prefer cells with ≥2 adjacent walkable neighbors (not wall corridor)
            good_cells = []
            for c in cells:
                adj_count = sum(
                    1 for dx, dy in _DIRS
                    if (
                        grid.in_bounds((c[0] + dx, c[1] + dy))
                        and grid.terrain[c[1] + dy, c[0] + dx] == CellType.EMPTY
                    )
                )
                if adj_count >= 2:
                    good_cells.append(c)
            if good_cells:
                cells = good_cells
            pos = cells[int(rng.integers(len(cells)))]
            plate_positions.append(pos)
            plate_barrier_map.append(idx)
            grid.objects[pos[1], pos[0]] = ObjectType.TARGET
            grid.metadata[pos[1], pos[0]] = idx  # barrier index

        # -- Place agent in zone 0 ----------------------------------------
        zone0 = _zone_cells(0)
        # Remove plate positions from available
        occupied = set(plate_positions)
        zone0 = [c for c in zone0 if c not in occupied]
        if not zone0:
            y_lo, y_hi = zone_ranges[0]
            zone0 = [(1, (y_lo + y_hi) // 2)]
        agent_pos = zone0[int(rng.integers(len(zone0)))]
        occupied.add(agent_pos)

        # -- Place GOAL in last zone ---------------------------------------
        last_zone = len(zone_ranges) - 1
        last_cells = _zone_cells(last_zone)
        last_cells = [c for c in last_cells if c not in occupied]
        if not last_cells:
            y_lo, y_hi = zone_ranges[last_zone]
            last_cells = [(size // 2, (y_lo + y_hi) // 2)]
        goal_pos = last_cells[int(rng.integers(len(last_cells)))]
        grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.GOAL
        occupied.add(goal_pos)

        # -- Place NPCs ----------------------------------------------------
        # Assign each NPC to a zone, close to its plate (within ~3 cells)
        npc_data: list[dict] = []

        for i, ntype in enumerate(npc_types):
            # Place NPC in the same zone as the plate it should service
            zone_idx = min(i, len(zone_ranges) - 2)
            plate_x, plate_y = plate_positions[min(i, len(plate_positions) - 1)]

            # Prefer cells within Manhattan distance 3 of the plate
            cells = _zone_cells(zone_idx)
            cells = [c for c in cells if c not in occupied]
            near_cells = [
                c for c in cells
                if abs(c[0] - plate_x) + abs(c[1] - plate_y) <= 3
            ]
            if near_cells:
                cells = near_cells
            elif cells:
                # Sort by distance to plate, pick from closest quarter
                cells.sort(key=lambda c: abs(c[0] - plate_x) + abs(c[1] - plate_y))
                cells = cells[:max(1, len(cells) // 4)]

            if not cells:
                y_lo, y_hi = zone_ranges[zone_idx]
                cells = [(2, (y_lo + y_hi) // 2)]
                cells = [c for c in cells if c not in occupied]
                if not cells:
                    cells = [(2, y_lo)]
            npc_pos = cells[int(rng.integers(len(cells)))]
            occupied.add(npc_pos)

            # Place NPC on grid (all NPCs use same ObjectType, metadata = behavior)
            grid.objects[npc_pos[1], npc_pos[0]] = _NPC_OBJ_TYPE
            grid.metadata[npc_pos[1], npc_pos[0]] = _NPC_META[ntype]

            npc_entry: dict = {
                "pos": list(npc_pos),
                "type": ntype,
                "locked": False,
            }
            npc_data.append(npc_entry)

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [goal_pos],
            "max_steps": self.get_max_steps(),
            "npc_data": npc_data,
            "npc_types": [nd["type"] for nd in npc_data],
            "plate_positions": plate_positions,
            "plate_barrier_map": plate_barrier_map,
            "barrier_cells": {
                str(k): v for k, v in barrier_cells.items()
            },
            "barrier_rows": barrier_rows,
            "n_plates": n_plates,
            "grid_width": size,
            "grid_height": size,
            "_rng_seed": int(rng.integers(0, 2**31)),
        }

    # ------------------------------------------------------------------
    # Validation -- barriers start closed so flood_fill won't find goal
    # ------------------------------------------------------------------

    def validate_instance(self, grid, config):
        """Always valid: barriers are designed to open via NPC manipulation."""
        return True

    # ------------------------------------------------------------------
    # Reset / step hooks
    # ------------------------------------------------------------------

    def on_env_reset(self, agent, grid, config):
        # Rebuild NPC data from config
        npc_data = config.get("npc_data", [])
        config["_npc_positions"] = [tuple(nd["pos"]) for nd in npc_data]
        config["_npc_locked"] = [nd.get("locked", False) for nd in npc_data]
        config["_barrier_open"] = {
            str(i): False for i in range(config["n_plates"])
        }
        config["_npc_rng"] = np.random.default_rng(config.get("_rng_seed", 0))
        config["_barriers_opened_ever"] = set()
        config["_last_agent_action"] = None  # for contrarian NPC
        config["_prev_agent_pos"] = config.get("agent_start", (1, 1))
        self._config = config

        # Ensure NPCs are drawn on grid (all ObjectType.NPC, metadata = behavior)
        for i, (px, py) in enumerate(config["_npc_positions"]):
            ntype = config["npc_types"][i]
            grid.objects[py, px] = _NPC_OBJ_TYPE
            grid.metadata[py, px] = _NPC_META[ntype]

        # Ensure plates are drawn
        for px, py in config["plate_positions"]:
            if grid.objects[py, px] == ObjectType.NONE:
                grid.objects[py, px] = ObjectType.TARGET
        # Ensure barriers are closed (WALL)
        for idx_str, cells in config["barrier_cells"].items():
            for bx, by in cells:
                grid.terrain[by, bx] = CellType.WALL

    @staticmethod
    def _same_zone(ay: int, ny: int, barrier_rows: list[int],
                   barrier_open: dict[str, bool]) -> bool:
        """Check if agent (at row ay) and NPC (at row ny) share a zone.

        Returns True if no closed barrier separates them vertically.
        """
        lo, hi = min(ay, ny), max(ay, ny)
        for idx, br in enumerate(barrier_rows):
            if lo < br < hi and not barrier_open.get(str(idx), False):
                return False
        return True

    def on_env_step(self, agent, grid, config, step_count):  # noqa: C901
        ax, ay = agent.position
        # Check goal reached -- skip NPC logic
        if grid.objects[ay, ax] == ObjectType.GOAL:
            return

        npc_positions = config["_npc_positions"]
        npc_types = config["npc_types"]
        npc_locked = config["_npc_locked"]
        rng = config["_npc_rng"]
        gw = config.get("grid_width", grid.width)
        gh = config.get("grid_height", grid.height)
        last_action = config.get("_last_agent_action")
        prev_agent = config.get("_prev_agent_pos", (ax, ay))
        barrier_rows = config.get("barrier_rows", [])
        barrier_open = config.get("_barrier_open", {})

        # Build set of occupied cells (agent + all NPCs) for collision
        occupied: set[tuple[int, int]] = {(ax, ay)}
        for pos in npc_positions:
            occupied.add(pos)

        new_positions: list[tuple[int, int]] = []

        for i, (nx, ny) in enumerate(npc_positions):
            ntype = npc_types[i]
            occupied.discard((nx, ny))

            # Skip locked NPCs entirely -- they stay on their plate
            if npc_locked[i]:
                new_positions.append((nx, ny))
                occupied.add((nx, ny))
                continue

            new_pos = (nx, ny)

            # NPCs only react when agent is in the same zone (no closed
            # barrier between them). This prevents accidental drifting
            # while the agent works on NPCs in other zones.
            in_zone = self._same_zone(ay, ny, barrier_rows, barrier_open)

            if not in_zone:
                new_positions.append((nx, ny))
                occupied.add((nx, ny))
                continue

            if ntype == "follower":
                new_pos = self._move_follower(
                    nx, ny, ax, ay, prev_agent, grid, occupied, rng
                )
            elif ntype == "fearful":
                new_pos = self._move_fearful(
                    nx, ny, ax, ay, grid, occupied, rng
                )
            elif ntype == "mirror":
                new_pos = self._move_mirror(
                    nx, ny, ax, ay, gw, gh, grid, occupied, rng
                )
            elif ntype == "contrarian":
                new_pos = self._move_contrarian(
                    nx, ny, ax, ay, last_action, grid, occupied, rng
                )

            # Update grid: clear old, set new (all NPCs use ObjectType.NPC)
            meta_val = _NPC_META.get(ntype, 1)
            if grid.objects[ny, nx] == _NPC_OBJ_TYPE:
                grid.objects[ny, nx] = ObjectType.NONE
                grid.metadata[ny, nx] = 0
            grid.objects[new_pos[1], new_pos[0]] = _NPC_OBJ_TYPE
            grid.metadata[new_pos[1], new_pos[0]] = meta_val

            new_positions.append(new_pos)
            occupied.add(new_pos)

        config["_npc_positions"] = new_positions

        # Track agent state for next step
        config["_prev_agent_pos"] = (ax, ay)

        # -- Check pressure plates (LOCKING) ----------------------------
        npc_set = set(new_positions)
        for plate_idx, (px, py) in enumerate(config["plate_positions"]):
            barrier_idx = config["plate_barrier_map"][plate_idx]
            idx_str = str(barrier_idx)
            cells = config["barrier_cells"].get(idx_str, [])

            if (px, py) in npc_set:
                # NPC on plate -> lock it permanently
                if not config["_barrier_open"].get(idx_str, False):
                    config["_barrier_open"][idx_str] = True
                    config["_barriers_opened_ever"].add(barrier_idx)
                    for bx, by in cells:
                        grid.terrain[by, bx] = CellType.EMPTY
                # Find which NPC is on this plate and lock it
                for ni, npos in enumerate(new_positions):
                    if npos == (px, py) and not config["_npc_locked"][ni]:
                        config["_npc_locked"][ni] = True
            # NOTE: No closing logic. Once a plate is activated, the NPC
            # is locked on it and the barrier stays open permanently.

    def on_agent_moved(self, new_pos, agent, grid):
        """Track agent's last action direction for contrarian NPC."""
        config = self._config
        prev = config.get("_prev_agent_pos", new_pos)
        ox, oy = prev
        nx, ny = new_pos
        dx, dy = nx - ox, ny - oy
        if dx == 1:
            config["_last_agent_action"] = "right"
        elif dx == -1:
            config["_last_agent_action"] = "left"
        elif dy == 1:
            config["_last_agent_action"] = "down"
        elif dy == -1:
            config["_last_agent_action"] = "up"
        else:
            config["_last_agent_action"] = None

    # ------------------------------------------------------------------
    # NPC movement helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _move_follower(
        nx: int, ny: int, ax: int, ay: int,
        prev_agent: tuple[int, int],
        grid: Grid, occupied: set[tuple[int, int]],
        rng: np.random.Generator,
    ) -> tuple[int, int]:
        """Follower: BFS toward cell adjacent to agent (stays 1 tile behind).

        Target is the cell the agent just left (prev_agent position). If that
        cell is not walkable, fall back to any cell adjacent to the agent.
        """
        dist = abs(nx - ax) + abs(ny - ay)
        if dist <= 4:
            # Primary target: the cell the agent just left
            pax, pay = prev_agent
            target = (pax, pay)

            # If prev_agent == current agent pos (no movement), or if
            # prev_agent is not walkable, find best adjacent cell
            if target == (ax, ay) or not _is_walkable(target, grid, occupied):
                # Find adjacent cell to agent that is closest to NPC
                best = (nx, ny)
                best_d = float("inf")
                for ddx, ddy in _DIRS:
                    cx, cy = ax + ddx, ay + ddy
                    if _is_walkable((cx, cy), grid, occupied) or (cx, cy) == (nx, ny):
                        d = abs(cx - nx) + abs(cy - ny)
                        if d < best_d:
                            best_d = d
                            best = (cx, cy)
                target = best

            if target == (nx, ny):
                return (nx, ny)

            step = _bfs_one_step(nx, ny, target[0], target[1], grid, occupied)
            if step != (nx, ny):
                return step

        # Agent is far away — stay put (no idle wander)
        return (nx, ny)

    @staticmethod
    def _move_fearful(
        nx: int, ny: int, ax: int, ay: int,
        grid: Grid, occupied: set[tuple[int, int]],
        rng: np.random.Generator,
    ) -> tuple[int, int]:
        """Fearful: if agent within Manhattan distance 3, flee (maximize distance).

        No idle wander — only moves as a direct reaction to agent proximity.
        """
        dist = abs(nx - ax) + abs(ny - ay)
        if dist <= 3:
            best = (nx, ny)
            best_d = dist
            candidates = [(nx + dx, ny + dy) for dx, dy in _DIRS]
            rng.shuffle(candidates)
            for cx, cy in candidates:
                if not grid.in_bounds((cx, cy)):
                    continue
                if grid.terrain[cy, cx] != CellType.EMPTY:
                    continue
                if (cx, cy) in occupied:
                    continue
                if grid.objects[cy, cx] in (
                    ObjectType.GOAL, ObjectType.KEY
                ):
                    continue
                d = abs(cx - ax) + abs(cy - ay)
                if d > best_d:
                    best_d = d
                    best = (cx, cy)
            return best
        # No idle wander — stays put until agent approaches
        return (nx, ny)

    @staticmethod
    def _move_mirror(
        nx: int, ny: int, ax: int, ay: int,
        grid_width: int, grid_height: int,
        grid: Grid, occupied: set[tuple[int, int]],
        rng: np.random.Generator,
    ) -> tuple[int, int]:
        """Mirror: BFS one step toward the mirrored position of the agent.

        Mirror position: mirror_x = (grid_width - 1) - agent_x,
                         mirror_y = (grid_height - 1) - agent_y.
        If target is blocked, stays put.
        """
        mirror_x = (grid_width - 1) - ax
        mirror_y = (grid_height - 1) - ay
        # Clamp to interior
        mirror_x = max(1, min(grid_width - 2, mirror_x))
        mirror_y = max(1, min(grid_height - 2, mirror_y))

        if (nx, ny) == (mirror_x, mirror_y):
            return (nx, ny)

        step = _bfs_one_step(nx, ny, mirror_x, mirror_y, grid, occupied)
        if step != (nx, ny):
            return step

        # BFS failed — stay put (no idle wander)
        return (nx, ny)

    @staticmethod
    def _move_contrarian(
        nx: int, ny: int, ax: int, ay: int,
        last_action: str | None,
        grid: Grid, occupied: set[tuple[int, int]],
        rng: np.random.Generator,
    ) -> tuple[int, int]:
        """Contrarian: moves OPPOSITE to agent's last move.

        No idle wander — purely agent-controlled. Zone-gating in on_env_step
        prevents drift while the agent is in a different zone.
        """
        if last_action is None:
            return (nx, ny)

        # Opposite direction map
        opposite = {
            "up": (0, 1),      # agent up -> contrarian down
            "down": (0, -1),   # agent down -> contrarian up
            "left": (1, 0),    # agent left -> contrarian right
            "right": (-1, 0),  # agent right -> contrarian left
        }
        dx, dy = opposite.get(last_action, (0, 0))
        if dx == 0 and dy == 0:
            return (nx, ny)

        cx, cy = nx + dx, ny + dy
        if (
            grid.in_bounds((cx, cy))
            and grid.terrain[cy, cx] == CellType.EMPTY
            and (cx, cy) not in occupied
            and grid.objects[cy, cx] not in (ObjectType.GOAL, ObjectType.KEY)
        ):
            return (cx, cy)

        # Target blocked, stay put
        return (nx, ny)

    # ------------------------------------------------------------------
    # Agent entry / movement
    # ------------------------------------------------------------------

    def can_agent_enter(self, pos, agent, grid):
        """Block agent from walking through NPCs."""
        x, y = pos
        if int(grid.objects[y, x]) == _NPC_OBJ_TYPE:
            return False
        return True

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def compute_dense_reward(self, old_state, action, new_state, info):
        config = new_state.get("config", {})
        reward = -0.01  # step penalty

        # Bonus for newly opened barriers (permanent)
        opened_ever = config.get("_barriers_opened_ever", set())
        old_opened = old_state.get("config", {}).get(
            "_barriers_opened_ever", set()
        )
        newly_opened = opened_ever - old_opened
        reward += 0.2 * len(newly_opened)

        # Approach shaping toward goal
        if "agent" in new_state and "agent" in old_state:
            new_ax, new_ay = new_state["agent"].position
            old_ax, old_ay = old_state["agent"].position
            goal = config.get("goal_positions", [None])[0]
            if goal:
                gx, gy = goal
                old_d = abs(old_ax - gx) + abs(old_ay - gy)
                new_d = abs(new_ax - gx) + abs(new_ay - gy)
                reward += 0.02 * (old_d - new_d)

        if self.check_success(new_state):
            reward += 1.0

        return reward

    def compute_sparse_reward(self, old_state, action, new_state, info):
        if self.check_success(new_state):
            return 1.0
        return 0.0

    # ------------------------------------------------------------------
    # Done / success
    # ------------------------------------------------------------------

    def check_success(self, state):
        if "grid" not in state or "agent" not in state:
            return False
        x, y = state["agent"].position
        return bool(state["grid"].objects[y, x] == ObjectType.GOAL)

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0


# ======================================================================
# Module-level helpers (used by NPC movement)
# ======================================================================


def _is_walkable(
    pos: tuple[int, int],
    grid: Grid,
    occupied: set[tuple[int, int]],
) -> bool:
    """Check if a cell is in-bounds, empty terrain, and not occupied."""
    x, y = pos
    if not grid.in_bounds(pos):
        return False
    if grid.terrain[y, x] != CellType.EMPTY:
        return False
    if pos in occupied:
        return False
    return True


def _bfs_one_step(
    sx: int, sy: int, gx: int, gy: int,
    grid: Grid, occupied: set[tuple[int, int]],
) -> tuple[int, int]:
    """BFS from (sx, sy) toward (gx, gy); return the first step position."""
    if (sx, sy) == (gx, gy):
        return (sx, sy)
    queue: deque[tuple[tuple[int, int], tuple[int, int]]] = deque()
    visited: set[tuple[int, int]] = {(sx, sy)}
    for dx, dy in _DIRS:
        nx, ny = sx + dx, sy + dy
        pos = (nx, ny)
        if pos == (gx, gy):
            return pos
        if not grid.in_bounds(pos):
            continue
        if grid.terrain[ny, nx] != CellType.EMPTY:
            continue
        if pos in occupied:
            continue
        visited.add(pos)
        queue.append((pos, pos))  # (current, first_step)
    while queue:
        (cx, cy), first = queue.popleft()
        for dx, dy in _DIRS:
            nx, ny = cx + dx, cy + dy
            pos = (nx, ny)
            if pos in visited:
                continue
            if pos == (gx, gy):
                return first
            if not grid.in_bounds(pos):
                continue
            if grid.terrain[ny, nx] != CellType.EMPTY:
                continue
            if pos in occupied:
                continue
            visited.add(pos)
            queue.append((pos, first))
    return (sx, sy)  # no path found


def _random_step(
    nx: int, ny: int,
    grid: Grid, occupied: set[tuple[int, int]],
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Take a random walkable step."""
    candidates = [(nx + dx, ny + dy) for dx, dy in _DIRS]
    valid = [
        (cx, cy) for cx, cy in candidates
        if (
            grid.in_bounds((cx, cy))
            and grid.terrain[cy, cx] == CellType.EMPTY
            and (cx, cy) not in occupied
            and grid.objects[cy, cx] not in (ObjectType.GOAL, ObjectType.KEY)
        )
    ]
    if valid:
        return valid[int(rng.integers(len(valid)))]
    return (nx, ny)
