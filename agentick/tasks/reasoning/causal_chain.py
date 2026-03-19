"""CausalChain - Open-arena lever combination puzzle.

MECHANICS:
  - Open arena: ALL levers reachable from the start (no physical gating)
  - M barriers (wall segments) block the path to the goal zone
  - Each barrier gap starts as WALL (closed)
  - Each lever TOGGLES ON/OFF via INTERACT (face-then-INTERACT pattern)
  - Each lever has effects: opens some barriers, closes others
  - A SOLUTION COMBINATION exists: a specific ON/OFF assignment that opens ALL
    barriers simultaneously, giving the agent access to the goal
  - Decoy levers toggle a patch of ICE terrain but never affect barriers
  - Agent must experiment to discover the correct combination

DIFFICULTY:
  - easy:   9x9,  2 levers, 0 decoys, 2 barriers, max_steps=150
  - medium: 11x11, 3 levers, 1 decoy,  3 barriers, max_steps=280
  - hard:   13x13, 4 levers, 2 decoys, 4 barriers, max_steps=450
  - expert: 15x15, 5 levers, 3 decoys, 5 barriers, max_steps=650
"""

from __future__ import annotations

from collections import deque

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task

# ---------------------------------------------------------------------------
# Helper: BFS over (position, lever_bitmask) to check solvability
# ---------------------------------------------------------------------------

def _bfs_solvable(
    terrain_base: np.ndarray,
    agent_pos: tuple[int, int],
    goal_pos: tuple[int, int],
    lever_positions: list[tuple[int, int]],
    switch_effects: list[dict],
    barriers: list[dict],
    solution_mask: int,
    size: int,
) -> bool:
    """BFS over (grid_position, lever_bitmask) state space.

    Returns True if there exists a reachable state where all barriers are
    open (bitmask == solution_mask) and the agent can then walk to the goal.
    """
    n = len(lever_positions)
    n_barriers = len(barriers)

    def barrier_states_for_mask(mask: int) -> list[bool]:
        """Return list of per-barrier open flags for a given lever bitmask."""
        lever_on = [(mask >> i) & 1 for i in range(n)]
        barrier_has_opener = [False] * n_barriers
        barrier_has_closer = [False] * n_barriers
        for li, is_on in enumerate(lever_on):
            if not is_on or li >= len(switch_effects):
                continue
            eff = switch_effects[li]
            for b in eff.get("opens", []):
                if b < n_barriers:
                    barrier_has_opener[b] = True
            for b in eff.get("closes", []):
                if b < n_barriers:
                    barrier_has_closer[b] = True
        return [
            barrier_has_opener[b] and not barrier_has_closer[b]
            for b in range(n_barriers)
        ]

    def terrain_for_mask(mask: int) -> np.ndarray:
        t = terrain_base.copy()
        states = barrier_states_for_mask(mask)
        for b_idx, barrier in enumerate(barriers):
            is_open = states[b_idx] if b_idx < len(states) else False
            for cell in barrier.get("cells", []):
                cx, cy = cell[0], cell[1]
                t[cy, cx] = int(CellType.EMPTY) if is_open else int(CellType.WALL)
        return t

    def flood(terrain: np.ndarray, start: tuple[int, int]) -> set[tuple[int, int]]:
        visited: set[tuple[int, int]] = {start}
        q: deque[tuple[int, int]] = deque([start])
        while q:
            cx, cy = q.popleft()
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if (nx, ny) not in visited and 0 <= nx < size and 0 <= ny < size:
                    if int(terrain[ny, nx]) not in (int(CellType.WALL), int(CellType.HAZARD)):
                        visited.add((nx, ny))
                        q.append((nx, ny))
        return visited

    # State: (lever_bitmask,).  We track which masks we've explored.
    # For each mask, compute what's reachable from agent, which levers can be
    # toggled (they are adjacent to and can be interacted with).
    visited_masks: set[int] = {0}
    q_masks: deque[int] = deque([0])

    while q_masks:
        mask = q_masks.popleft()
        terrain = terrain_for_mask(mask)
        reachable = flood(terrain, agent_pos)

        # Check: with all barriers open (solution_mask), can we reach goal?
        if mask == solution_mask:
            if goal_pos in reachable:
                return True

        # Try toggling each reachable lever (agent needs to stand adjacent)
        for li, lpos in enumerate(lever_positions):
            lx, ly = lpos
            # Lever is solid; agent must be adjacent to interact
            adjacent = {
                (lx + dx, ly + dy) for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]
            }
            if not (adjacent & reachable):
                continue
            new_mask = mask ^ (1 << li)
            if new_mask not in visited_masks:
                visited_masks.add(new_mask)
                q_masks.append(new_mask)

    return False


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

@register_task("CausalChain-v0", tags=["reasoning", "causal_reasoning"])
class CausalChainTask(TaskSpec):
    """Open-arena lever combination puzzle.

    All levers are accessible from the start. The agent must discover which
    combination of levers (ON/OFF) opens all barriers simultaneously to reach
    the goal. Decoy levers create visible ICE terrain changes but never affect
    barriers.
    """

    name = "CausalChain-v0"
    description = (
        "Discover the correct lever combination that opens all barriers to reach the goal"
    )
    capability_tags = ["reasoning", "causal_reasoning"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=9,
            max_steps=150,
            params={"n_switches": 2, "n_decoys": 0},
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=11,
            max_steps=280,
            params={"n_switches": 3, "n_decoys": 1},
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=13,
            max_steps=450,
            params={"n_switches": 4, "n_decoys": 2},
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=15,
            max_steps=650,
            params={"n_switches": 5, "n_decoys": 3},
        ),
    }

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, seed: int):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        n = self.difficulty_config.params.get("n_switches", 2)
        n_decoys = self.difficulty_config.params.get("n_decoys", 0)
        n_barriers = n  # one barrier per real lever

        for _attempt in range(30):
            result = self._try_generate(rng, size, n, n_decoys, n_barriers)
            if result is not None:
                return result

        # Fallback: guaranteed simple layout
        return self._fallback_generate(rng, size, n, n_decoys, n_barriers)

    def _try_generate(
        self,
        rng: np.random.Generator,
        size: int,
        n: int,
        n_decoys: int,
        n_barriers: int,
    ):
        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        # N horizontal barrier rows partition the grid into N+1 zones.
        # Zone 0 (top) holds agent + levers, zone N (bottom) holds goal.
        # Each barrier row spans x=1..size-2 with ONE gap cell (starts WALL).
        interior_height = size - 2
        # Evenly space N barriers across the interior height, leaving at
        # least 2 rows for zone 0 (lever area) and 1 for goal zone.
        min_zone0 = max(3, (n + n_decoys + 2) // (size - 2) + 2)
        barrier_ys: list[int] = []
        usable = interior_height - min_zone0
        for i in range(n_barriers):
            by = min_zone0 + 1 + i * max(1, usable // max(n_barriers, 1))
            if by < size - 2:
                barrier_ys.append(by)
        if len(barrier_ys) < n_barriers:
            return None

        barriers: list[dict] = []
        for by in barrier_ys:
            for x in range(1, size - 1):
                grid.terrain[by, x] = CellType.WALL
            gap_x = int(rng.integers(1, size - 1))
            barriers.append({"cells": [[int(gap_x), int(by)]], "open": False})

        # Agent start — random position in zone 0 (rows 1..barrier_ys[0]-1)
        zone0_y_max = barrier_ys[0] - 1
        agent_x = int(rng.integers(1, size - 1))
        agent_y = int(rng.integers(1, zone0_y_max + 1))
        agent_pos: tuple[int, int] = (agent_x, agent_y)

        # Goal — random position in zone N (rows below last barrier)
        last_barrier_y = barrier_ys[-1]
        goal_y = int(rng.integers(last_barrier_y + 1, size - 1))
        goal_x = int(rng.integers(1, size - 1))
        goal_pos: tuple[int, int] = (goal_x, goal_y)
        grid.objects[goal_y, goal_x] = ObjectType.GOAL

        # Lever effects
        _hint_mask = int(rng.integers(0, 2**n))
        switch_effects, solution_mask = self._build_effects(n, n_barriers, _hint_mask, rng)

        # Place levers in zone 0
        arena_cells = [
            (x, y)
            for x in range(1, size - 1)
            for y in range(1, zone0_y_max + 1)
            if (x, y) != agent_pos and grid.terrain[y, x] == CellType.EMPTY
        ]
        if len(arena_cells) < n + n_decoys + 1:
            return None
        rng.shuffle(arena_cells)

        lever_positions: list[tuple[int, int]] = []
        used: set[tuple[int, int]] = {agent_pos, goal_pos}
        for cell in arena_cells:
            if len(lever_positions) >= n:
                break
            if cell in used:
                continue
            cx, cy = cell
            adj_free = [
                (cx + dx, cy + dy)
                for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]
                if 1 <= cx + dx <= size - 2
                and 1 <= cy + dy <= zone0_y_max
                and (cx + dx, cy + dy) not in used
                and grid.terrain[cy + dy, cx + dx] == CellType.EMPTY
            ]
            if not adj_free:
                continue
            lever_positions.append(cell)
            used.add(cell)
        if len(lever_positions) < n:
            return None

        decoy_positions: list[tuple[int, int]] = []
        for cell in arena_cells:
            if len(decoy_positions) >= n_decoys:
                break
            if cell in used:
                continue
            cx, cy = cell
            adj_free = [
                (cx + dx, cy + dy)
                for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]
                if 1 <= cx + dx <= size - 2
                and 1 <= cy + dy <= zone0_y_max
                and (cx + dx, cy + dy) not in used
                and grid.terrain[cy + dy, cx + dx] == CellType.EMPTY
            ]
            if not adj_free:
                continue
            decoy_positions.append(cell)
            used.add(cell)

        # Place lever objects
        for lx, ly in lever_positions:
            grid.objects[ly, lx] = ObjectType.LEVER
        for dx, dy in decoy_positions:
            grid.objects[dy, dx] = ObjectType.LEVER

        # Verify all levers reachable from agent (BFS around solid levers)
        all_levers = set(lever_positions) | set(decoy_positions)
        reachable: set[tuple[int, int]] = {agent_pos}
        bfs_q: deque[tuple[int, int]] = deque([agent_pos])
        while bfs_q:
            cx, cy = bfs_q.popleft()
            for ddx, ddy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nxx, nyy = cx + ddx, cy + ddy
                if (nxx, nyy) not in reachable and 0 <= nxx < size and 0 <= nyy < size:
                    if int(grid.terrain[nyy, nxx]) not in (
                        int(CellType.WALL), int(CellType.HAZARD)
                    ) and not grid.is_object_blocking((nxx, nyy)):
                        reachable.add((nxx, nyy))
                        bfs_q.append((nxx, nyy))
        for lp in all_levers:
            lx, ly = lp
            if not any(
                (lx + ddx, ly + ddy) in reachable
                for ddx, ddy in [(0, -1), (1, 0), (0, 1), (-1, 0)]
            ):
                for px, py in all_levers:
                    grid.objects[py, px] = ObjectType.NONE
                return None

        # Decoy ICE patches — exclude barrier rows
        arena_y_range = range(1, zone0_y_max + 1)
        decoy_ice_patches: list[list[int]] = []
        ice_used: set[tuple[int, int]] = set(used)
        for by in barrier_ys:
            for x in range(1, size - 1):
                ice_used.add((x, by))
        for _ in decoy_positions:
            patch = self._find_ice_patch(rng, size, arena_y_range, ice_used)
            decoy_ice_patches.append(patch)

        # Validate solvability
        terrain_base = grid.terrain.copy()
        for barrier in barriers:
            for cell in barrier["cells"]:
                cx, cy = cell[0], cell[1]
                terrain_base[cy, cx] = int(CellType.WALL)

        solvable = _bfs_solvable(
            terrain_base, agent_pos, goal_pos,
            lever_positions, switch_effects, barriers, solution_mask, size,
        )
        if not solvable:
            return None

        # Verify goal unreachable with all barriers closed
        initial_reachable: set[tuple[int, int]] = {agent_pos}
        bfs_q2: deque[tuple[int, int]] = deque([agent_pos])
        while bfs_q2:
            cx, cy = bfs_q2.popleft()
            for ddx, ddy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nxx, nyy = cx + ddx, cy + ddy
                if (nxx, nyy) not in initial_reachable and 0 <= nxx < size and 0 <= nyy < size:
                    if int(terrain_base[nyy, nxx]) not in (
                        int(CellType.WALL), int(CellType.HAZARD)
                    ):
                        initial_reachable.add((nxx, nyy))
                        bfs_q2.append((nxx, nyy))
        if goal_pos in initial_reachable:
            return None

        all_lever_positions = lever_positions + decoy_positions
        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [goal_pos],
            "switch_positions": [list(p) for p in lever_positions],
            "decoy_positions": [list(p) for p in decoy_positions],
            "all_lever_positions": [list(p) for p in all_lever_positions],
            "switch_effects": switch_effects,
            "barriers": barriers,
            "_solution_mask": solution_mask,
            "_lever_states": [False] * n,
            "_decoy_states": [False] * n_decoys,
            "_decoy_ice_patches": decoy_ice_patches,
            "_barriers_open_count": 0,
            "n_switches": n,
            "n_decoys": n_decoys,
            "max_steps": self.get_max_steps(),
        }

    def _fallback_generate(
        self,
        rng: np.random.Generator,
        size: int,
        n: int,
        n_decoys: int,
        n_barriers: int,
    ):
        """Simple guaranteed fallback: horizontal barrier rows."""
        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        agent_pos: tuple[int, int] = (1, 1)

        # Place N horizontal barrier rows
        barrier_ys: list[int] = []
        spacing = max(2, (size - 4) // (n_barriers + 1))
        for i in range(n_barriers):
            by = 3 + i * spacing
            if by < size - 2:
                barrier_ys.append(by)

        barriers: list[dict] = []
        for by in barrier_ys:
            for x in range(1, size - 1):
                grid.terrain[by, x] = CellType.WALL
            gap_x = size // 2
            barriers.append({"cells": [[gap_x, by]], "open": False})

        _hint_mask = int(rng.integers(1, 2**n))
        switch_effects, solution_mask = self._build_effects(n, n_barriers, _hint_mask, rng)

        # Place levers in zone 0 (rows 1..barrier_ys[0]-1)
        zone0_y_max = barrier_ys[0] - 1 if barrier_ys else size - 2
        lever_positions: list[tuple[int, int]] = []
        for i in range(n):
            lx = 2 + i * 2
            ly = 2
            if lx < size - 1 and ly <= zone0_y_max:
                lever_positions.append((lx, ly))
                grid.objects[ly, lx] = ObjectType.LEVER

        decoy_positions: list[tuple[int, int]] = []
        for i in range(n_decoys):
            dx_pos = size - 2 - i * 2
            dy_pos = 2
            if dx_pos > 0 and (dx_pos, dy_pos) not in lever_positions and dy_pos <= zone0_y_max:
                decoy_positions.append((dx_pos, dy_pos))
                grid.objects[dy_pos, dx_pos] = ObjectType.LEVER

        last_by = barrier_ys[-1] if barrier_ys else size // 2
        goal_pos: tuple[int, int] = (size // 2, min(last_by + 2, size - 2))
        grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.GOAL

        arena_y_range = range(1, zone0_y_max + 1)
        decoy_ice_patches: list[list[int]] = []
        ice_used: set[tuple[int, int]] = set(lever_positions + decoy_positions)
        ice_used.add(agent_pos)
        ice_used.add(goal_pos)
        for by in barrier_ys:
            for x in range(1, size - 1):
                ice_used.add((x, by))
        for _ in decoy_positions:
            patch = self._find_ice_patch(rng, size, arena_y_range, ice_used)
            decoy_ice_patches.append(patch)

        all_lever_positions = lever_positions + decoy_positions

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [goal_pos],
            "switch_positions": [list(p) for p in lever_positions],
            "decoy_positions": [list(p) for p in decoy_positions],
            "all_lever_positions": [list(p) for p in all_lever_positions],
            "switch_effects": switch_effects,
            "barriers": barriers,
            "_solution_mask": solution_mask,
            "_lever_states": [False] * n,
            "_decoy_states": [False] * n_decoys,
            "_decoy_ice_patches": decoy_ice_patches,
            "_barriers_open_count": 0,
            "n_switches": n,
            "n_decoys": n_decoys,
            "max_steps": self.get_max_steps(),
        }

    # ------------------------------------------------------------------
    # Effect building
    # ------------------------------------------------------------------

    def _build_effects(
        self,
        n: int,
        n_barriers: int,
        solution_mask: int,
        rng: np.random.Generator,
    ) -> tuple[list[dict], int]:
        """Build lever effects and derive the true solution_mask.

        Design:
        - Randomly assign each barrier j an "opener lever" from the set of n
          levers.  Opener levers MUST be ON in the solution.
        - The remaining levers are "saboteur" levers that MUST be OFF; when
          turned ON they close one of the barriers.
        - solution_mask is then: bit i = 1 iff lever i is an opener.

        This guarantees every barrier is opened by exactly one ON lever and
        that the returned solution_mask is self-consistent.

        Returns:
            Tuple (effects, true_solution_mask) where effects[i] is
            {"opens": [...], "closes": [...]}.
        """
        # Assign openers: for each barrier j pick one lever uniformly at
        # random.  Multiple barriers can share an opener (that opener opens
        # them all).
        barrier_opener = [int(rng.integers(0, n)) for _ in range(n_barriers)]

        # Build solution_mask: lever i must be ON iff it is an opener.
        true_mask = 0
        for j in range(n_barriers):
            true_mask |= 1 << barrier_opener[j]

        # Build opens lists
        effects: list[dict] = [{"opens": [], "closes": []} for _ in range(n)]
        for j, opener_li in enumerate(barrier_opener):
            effects[opener_li]["opens"].append(j)

        # Saboteur levers (sol bit == 0) close a random ON-lever's barrier.
        # This forces the agent to keep them OFF.
        opener_levers = [i for i in range(n) if (true_mask >> i) & 1]
        for i in range(n):
            if (true_mask >> i) & 1:
                # This lever is an opener; no close-effect
                continue
            # Pick a barrier opened by some opener lever to close
            if opener_levers:
                victim_opener = int(rng.choice(opener_levers))
                victims = effects[victim_opener]["opens"]
                if victims:
                    close_b = victims[int(rng.integers(0, len(victims)))]
                    if close_b not in effects[i]["closes"]:
                        effects[i]["closes"].append(close_b)

        # For hard/expert (n >= 4): saboteur levers also close a second barrier
        if n >= 4:
            for i in range(n):
                if (true_mask >> i) & 1:
                    continue
                # Pick a second barrier to close (different from first)
                if opener_levers:
                    victim_opener = int(rng.choice(opener_levers))
                    victims = effects[victim_opener]["opens"]
                    for close_b in victims:
                        if close_b not in effects[i]["closes"]:
                            effects[i]["closes"].append(close_b)
                            break

        return effects, true_mask

    # ------------------------------------------------------------------
    # Barrier state helpers
    # ------------------------------------------------------------------

    def _compute_barrier_states(
        self,
        lever_states: list[bool],
        switch_effects: list[dict],
        n_barriers: int,
    ) -> list[bool]:
        """Return per-barrier open flags given current lever states.

        A barrier is open iff at least one lever that opens it is ON,
        AND no lever that closes it is ON.
        """
        has_opener = [False] * n_barriers
        has_closer = [False] * n_barriers

        for li, is_on in enumerate(lever_states):
            if not is_on or li >= len(switch_effects):
                continue
            eff = switch_effects[li]
            for b in eff.get("opens", []):
                if b < n_barriers:
                    has_opener[b] = True
            for b in eff.get("closes", []):
                if b < n_barriers:
                    has_closer[b] = True

        return [has_opener[b] and not has_closer[b] for b in range(n_barriers)]

    def _apply_barriers(self, grid: Grid, barriers: list[dict], states: list[bool]) -> None:
        """Write barrier states to the grid terrain."""
        for b_idx, barrier in enumerate(barriers):
            is_open = states[b_idx] if b_idx < len(states) else False
            barrier["open"] = is_open
            for cell in barrier.get("cells", []):
                cx, cy = cell[0], cell[1]
                grid.terrain[cy, cx] = CellType.EMPTY if is_open else CellType.WALL
                if not is_open:
                    grid.metadata[cy, cx] = b_idx

    # ------------------------------------------------------------------
    # ICE patch helpers (decoys)
    # ------------------------------------------------------------------

    def _find_ice_patch(
        self,
        rng: np.random.Generator,
        size: int,
        arena_y_range: range,
        used: set[tuple[int, int]],
    ) -> list[int]:
        """Find a single cell for a decoy ICE patch, not overlapping used."""
        candidates = [
            [x, y]
            for x in range(1, size - 1)
            for y in arena_y_range
            if (x, y) not in used
        ]
        if not candidates:
            return [1, 1]
        idx = int(rng.integers(0, len(candidates)))
        chosen = candidates[idx]
        used.add((chosen[0], chosen[1]))
        return chosen

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def on_env_reset(self, agent, grid, config: dict) -> None:
        """Initialise runtime state: all levers OFF, all barriers closed."""
        n = config.get("n_switches", 0)
        n_decoys = config.get("n_decoys", 0)
        config["_lever_states"] = [False] * n
        config["_decoy_states"] = [False] * n_decoys
        config["_barriers_open_count"] = 0

        # Close all barriers on grid
        barriers = config.get("barriers", [])
        switch_effects = config.get("switch_effects", [])
        states = self._compute_barrier_states([False] * n, switch_effects, len(barriers))
        self._apply_barriers(grid, barriers, states)

        # Restore lever objects (may have been cleared on prev episode)
        for lx, ly in [tuple(p) for p in config.get("switch_positions", [])]:
            grid.objects[ly, lx] = ObjectType.LEVER
            grid.metadata[ly, lx] = 0  # 0 = OFF

        for dx, dy in [tuple(p) for p in config.get("decoy_positions", [])]:
            grid.objects[dy, dx] = ObjectType.LEVER
            grid.metadata[dy, dx] = 0

        # Clear any lingering ICE from decoys
        for patch in config.get("_decoy_ice_patches", []):
            px, py = patch[0], patch[1]
            if 0 < px < grid.width - 1 and 0 < py < grid.height - 1:
                grid.terrain[py, px] = CellType.EMPTY

        # Place goal
        goal_x, goal_y = config["goal_positions"][0]
        grid.objects[goal_y, goal_x] = ObjectType.GOAL

        self._config = config
        self._prev_open_count = 0

    def on_agent_interact(self, pos: tuple[int, int], agent, grid: Grid) -> None:
        """Toggle lever at *pos* (face-then-INTERACT pattern)."""
        config = getattr(self, "_config", {})
        x, y = pos

        if grid.objects[y, x] != ObjectType.LEVER:
            return

        lever_positions = [tuple(p) for p in config.get("switch_positions", [])]
        decoy_positions = [tuple(p) for p in config.get("decoy_positions", [])]
        switch_effects = config.get("switch_effects", [])
        barriers = config.get("barriers", [])
        lever_states: list[bool] = config.get("_lever_states", [])
        decoy_states: list[bool] = config.get("_decoy_states", [])
        ice_patches = config.get("_decoy_ice_patches", [])

        pos_tuple = (x, y)

        if pos_tuple in decoy_positions:
            # Decoy: toggle ICE patch
            di = decoy_positions.index(pos_tuple)
            decoy_states[di] = not decoy_states[di]
            config["_decoy_states"] = decoy_states
            grid.metadata[y, x] = 100 if decoy_states[di] else 0

            if di < len(ice_patches):
                px, py = ice_patches[di][0], ice_patches[di][1]
                if 0 < px < grid.width - 1 and 0 < py < grid.height - 1:
                    current = int(grid.terrain[py, px])
                    grid.terrain[py, px] = (
                        CellType.ICE if current == int(CellType.EMPTY) else CellType.EMPTY
                    )
            return

        if pos_tuple in lever_positions:
            li = lever_positions.index(pos_tuple)
            lever_states[li] = not lever_states[li]
            config["_lever_states"] = lever_states
            grid.metadata[y, x] = 100 if lever_states[li] else 0

            # Recompute and apply barrier states
            n_barriers = len(barriers)
            new_states = self._compute_barrier_states(lever_states, switch_effects, n_barriers)
            self._apply_barriers(grid, barriers, new_states)
            config["_barriers_open_count"] = sum(new_states)

    def can_agent_enter(self, pos: tuple[int, int], agent, grid: Grid) -> bool:
        """LEVERs are solid; use default blocking for all other objects."""
        return not grid.is_object_blocking(pos)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def compute_dense_reward(
        self,
        old_state: dict,
        action,
        new_state: dict,
        info: dict,
    ) -> float:
        reward = -0.01  # step penalty

        if self.check_success(new_state):
            reward += 1.0
            return reward

        config = new_state.get("config", {})
        open_count = config.get("_barriers_open_count", 0)
        prev = getattr(self, "_prev_open_count", 0)

        # +0.3 per newly opened barrier
        delta = open_count - prev
        if delta > 0:
            reward += 0.3 * delta
        self._prev_open_count = open_count

        # Approach shaping: guide toward nearest lever or goal
        if "agent" in new_state:
            ax, ay = new_state["agent"].position
            ox, oy = old_state.get("agent_position", (ax, ay))

            n_barriers = len(config.get("barriers", []))
            all_open = open_count == n_barriers

            if all_open:
                goal = config.get("goal_positions", [None])[0]
                if goal:
                    tx, ty = goal
                    reward += 0.05 * (
                        abs(ox - tx) + abs(oy - ty) - abs(ax - tx) - abs(ay - ty)
                    )
            else:
                lever_positions = config.get("switch_positions", [])
                if lever_positions:
                    best = min(
                        (abs(ax - lp[0]) + abs(ay - lp[1]), lp)
                        for lp in lever_positions
                    )
                    tx, ty = best[1]
                    reward += 0.05 * (
                        abs(ox - tx) + abs(oy - ty) - abs(ax - tx) - abs(ay - ty)
                    )

        return reward

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def check_success(self, state: dict) -> bool:
        if "grid" not in state or "agent" not in state:
            return False
        x, y = state["agent"].position
        return bool(state["grid"].objects[y, x] == ObjectType.GOAL)

    def check_done(self, state: dict) -> bool:
        return self.check_success(state)

    def validate_instance(self, grid: Grid, config: dict) -> bool:
        return True  # barriers are dynamic; solvability checked during generate()

    def get_optimal_return(self, difficulty: str | None = None) -> float:
        return 1.0

    def get_random_baseline(self, difficulty: str | None = None) -> float:
        return 0.0
