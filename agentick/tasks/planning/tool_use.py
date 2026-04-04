"""ToolUse - Scroll-Combination Discovery.

MECHANICS:
  - Goal is behind a river (horizontal band of WATER terrain).
  - Agent CAN cross the river, but crossing without the ORB tool means
    the agent only gets partial reward (0.2) on reaching the goal.
  - SCROLL objects are scattered on the agent's side of the river.
    Stepping on a scroll collects it (removes from grid).
  - Once ALL required scrolls are collected, an ORB spawns at a designated
    location on the agent's side.
  - Picking up the ORB (stepping on it) grants the "tool". Crossing the
    river and reaching the goal WITH the orb gives full reward (1.0).
  - Higher difficulties: more scrolls, decoy COINs, wider rivers,
    multiple river crossings, scrolls placed behind obstacles.
  - The "emergent" insight: figure out that scrolls create the ORB tool.
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


def _bfs_reachable(
    grid: Grid,
    start: tuple[int, int],
    passable_terrain: set[int] | None = None,
) -> set[tuple[int, int]]:
    """BFS reachability from *start*, treating only the given terrain as passable.

    If *passable_terrain* is ``None`` the default walkable check is used
    (everything except WALL and HOLE).
    """
    if passable_terrain is None:
        return grid.flood_fill(start)

    visited: set[tuple[int, int]] = set()
    sx, sy = start
    if not grid.in_bounds(start):
        return visited
    if int(grid.terrain[sy, sx]) not in passable_terrain:
        return visited
    visited.add(start)
    queue: deque[tuple[int, int]] = deque([start])
    while queue:
        x, y = queue.popleft()
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            nx, ny = x + dx, y + dy
            nb = (nx, ny)
            if nb in visited or not grid.in_bounds(nb):
                continue
            if int(grid.terrain[ny, nx]) in passable_terrain:
                visited.add(nb)
                queue.append(nb)
    return visited


@register_task("ToolUse-v0", tags=["tool_use", "discovery"])
class ToolUseTask(TaskSpec):
    """Collect all scrolls to spawn an ORB; use the ORB to cross the river
    and reach the goal for full reward."""

    name = "ToolUse-v0"
    description = (
        "Collect scattered scrolls to spawn an orb tool, then cross "
        "the river to reach the goal for full reward"
    )
    capability_tags = ["tool_use", "discovery"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=9,
            max_steps=100,
            params={
                "n_scrolls": 2,
                "n_decoys": 0,
                "n_rivers": 1,
                "river_width": 2,
            },
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=11,
            max_steps=180,
            params={
                "n_scrolls": 3,
                "n_decoys": 1,
                "n_rivers": 1,
                "river_width": 3,
            },
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=13,
            max_steps=300,
            params={
                "n_scrolls": 4,
                "n_decoys": 2,
                "n_rivers": 2,
                "river_width": 2,
            },
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=15,
            max_steps=450,
            params={
                "n_scrolls": 5,
                "n_decoys": 3,
                "n_rivers": 2,
                "river_width": 3,
            },
        ),
    }

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, seed: int) -> tuple[Grid, dict]:
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        p = self.difficulty_config.params
        n_scrolls: int = p["n_scrolls"]
        n_decoys: int = p["n_decoys"]
        n_rivers: int = p["n_rivers"]
        river_width: int = p["river_width"]

        for _attempt in range(200):
            grid, cfg = self._try_generate(
                rng, size, n_scrolls, n_decoys, n_rivers, river_width,
            )
            if cfg is not None:
                return grid, cfg

        # Fallback: relax constraints and retry
        return self._try_generate(
            rng, size, n_scrolls, n_decoys, n_rivers, river_width,
            relaxed=True,
        )

    def _try_generate(
        self,
        rng: np.random.Generator,
        size: int,
        n_scrolls: int,
        n_decoys: int,
        n_rivers: int,
        river_width: int,
        *,
        relaxed: bool = False,
    ) -> tuple[Grid, dict | None]:
        grid = Grid(size, size)

        # Border walls
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        # --- River layout ---------------------------------------------------
        # We place river bands horizontally.  With 1 river the map is split
        # into an "agent zone" (top) and a "goal zone" (bottom).  With 2
        # rivers there's an intermediate zone as well.
        #
        # Each river band is *river_width* rows of WATER.  We space them
        # evenly in the interior (rows 1..size-2).
        inner_h = size - 2  # rows available (1 to size-2)
        river_rows: list[list[int]] = []
        if n_rivers == 1:
            mid = size // 2
            rows = list(range(mid, mid + river_width))
            # Clamp to interior
            rows = [r for r in rows if 1 <= r <= size - 2]
            river_rows.append(rows)
        else:
            # 2 rivers: place at ~1/3 and ~2/3
            y1 = 1 + inner_h // 3
            y2 = 1 + 2 * inner_h // 3
            for start_y in (y1, y2):
                rows = list(range(start_y, start_y + river_width))
                rows = [r for r in rows if 1 <= r <= size - 2]
                river_rows.append(rows)

        all_river_set: set[tuple[int, int]] = set()
        for band in river_rows:
            for ry in band:
                for rx in range(1, size - 1):
                    grid.terrain[ry, rx] = CellType.WATER
                    all_river_set.add((rx, ry))

        # --- Agent & goal positions -----------------------------------------
        # Agent always starts in the top-left area (above first river).
        agent_pos = (1, 1)

        # Goal in the bottom area (below last river).
        last_river_bottom = river_rows[-1][-1]
        goal_y = last_river_bottom + 1
        if goal_y > size - 2:
            goal_y = size - 2
        goal_x = size - 2
        goal_pos = (goal_x, goal_y)

        # Make sure goal cell is empty terrain
        grid.terrain[goal_pos[1], goal_pos[0]] = CellType.EMPTY
        grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.GOAL

        # --- Identify safe zone (agent side, above first river) -------------
        safe_passable = {int(CellType.EMPTY)}
        safe_cells = _bfs_reachable(grid, agent_pos, safe_passable)
        safe_cells.discard(agent_pos)
        safe_cells.discard(goal_pos)

        # Remove any cells that are on a river row (shouldn't happen, but
        # be defensive)
        safe_cells -= all_river_set

        if len(safe_cells) < n_scrolls + n_decoys + 1:
            if not relaxed:
                return grid, None
            # In relaxed mode just take whatever we have
            safe_list = sorted(safe_cells)
        else:
            safe_list = sorted(safe_cells)

        rng.shuffle(safe_list)

        used: set[tuple[int, int]] = {agent_pos, goal_pos}

        # --- Place scrolls ---------------------------------------------------
        scroll_positions: list[tuple[int, int]] = []
        for pos in safe_list:
            if len(scroll_positions) >= n_scrolls:
                break
            if pos in used:
                continue
            sx, sy = pos
            grid.objects[sy, sx] = ObjectType.SCROLL
            scroll_positions.append(pos)
            used.add(pos)

        if len(scroll_positions) < n_scrolls and not relaxed:
            return grid, None

        # --- Place decoys (COIN objects) -------------------------------------
        decoy_positions: list[tuple[int, int]] = []
        for pos in safe_list:
            if len(decoy_positions) >= n_decoys:
                break
            if pos in used:
                continue
            dx, dy = pos
            grid.objects[dy, dx] = ObjectType.COIN
            decoy_positions.append(pos)
            used.add(pos)

        # --- ORB spawn position (safe side, not on scroll/decoy) -------------
        orb_spawn_pos: tuple[int, int] | None = None
        for pos in safe_list:
            if pos in used:
                continue
            orb_spawn_pos = pos
            used.add(pos)
            break
        if orb_spawn_pos is None:
            # Fallback: place orb adjacent to agent start
            for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
                cand = (agent_pos[0] + dx, agent_pos[1] + dy)
                cx, cy = cand
                if (
                    grid.in_bounds(cand)
                    and grid.terrain[cy, cx] == CellType.EMPTY
                    and cand not in used
                ):
                    orb_spawn_pos = cand
                    used.add(cand)
                    break
        if orb_spawn_pos is None:
            if not relaxed:
                return grid, None
            orb_spawn_pos = (2, 1)  # last resort

        # --- Validate reachability -------------------------------------------
        # All scrolls, decoys, and orb_spawn must be reachable from agent_pos
        # without crossing water.
        full_reachable = grid.flood_fill(agent_pos)
        for sp in scroll_positions:
            if sp not in full_reachable and not relaxed:
                return grid, None
        if orb_spawn_pos not in full_reachable and not relaxed:
            return grid, None

        # Goal must be reachable when crossing water is allowed
        full_reach_with_water = grid.flood_fill(agent_pos)
        if goal_pos not in full_reach_with_water and not relaxed:
            return grid, None

        config = {
            "agent_start": agent_pos,
            "goal_positions": [goal_pos],
            "max_steps": self.get_max_steps(),
            "scroll_positions": [list(p) for p in scroll_positions],
            "decoy_positions": [list(p) for p in decoy_positions],
            "orb_spawn_pos": list(orb_spawn_pos),
            "n_scrolls_required": len(scroll_positions),
            # Runtime state (reset in on_env_reset)
            "_scrolls_collected": set(),
            "_has_orb": False,
            "_crossed_water_without_orb": False,
            "_orb_spawned": False,
        }
        return grid, config

    # ------------------------------------------------------------------
    # Runtime hooks
    # ------------------------------------------------------------------

    def on_env_reset(self, agent, grid, config):
        agent.inventory.clear()
        config["_scrolls_collected"] = set()
        config["_has_orb"] = False
        config["_crossed_water_without_orb"] = False
        config["_orb_spawned"] = False
        self._config = config

    def on_agent_moved(self, pos, agent, grid):
        config = getattr(self, "_config", {})
        x, y = pos
        obj = grid.objects[y, x]
        terrain = grid.terrain[y, x]

        # Collect scroll
        if obj == ObjectType.SCROLL:
            grid.objects[y, x] = ObjectType.NONE
            scrolls_collected: set = config.setdefault(
                "_scrolls_collected", set(),
            )
            scrolls_collected.add((x, y))

            # Check if all scrolls collected => spawn ORB
            n_required = config.get("n_scrolls_required", 0)
            if (
                len(scrolls_collected) >= n_required
                and not config.get("_orb_spawned", False)
            ):
                orb_pos = config.get("orb_spawn_pos", None)
                if orb_pos is not None:
                    ox, oy = orb_pos
                    grid.objects[oy, ox] = ObjectType.ORB
                    config["_orb_spawned"] = True

        # Collect decoy (COIN) — just remove it, no effect
        elif obj == ObjectType.COIN:
            grid.objects[y, x] = ObjectType.NONE

        # Pick up ORB
        elif obj == ObjectType.ORB:
            grid.objects[y, x] = ObjectType.NONE
            config["_has_orb"] = True
            agent.inventory.append(
                Entity(
                    id=f"orb_{x}_{y}",
                    entity_type="orb",
                    position=pos,
                )
            )

        # Track water crossing without orb
        if terrain == CellType.WATER and not config.get("_has_orb", False):
            config["_crossed_water_without_orb"] = True

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def compute_dense_reward(self, old_state, action, new_state, info):
        config = new_state.get("config", {})
        reward = -0.01  # step penalty

        # --- Scroll collection bonus ---
        old_collected = len(
            old_state.get("config", {}).get("_scrolls_collected", set())
        )
        new_collected = len(config.get("_scrolls_collected", set()))
        scrolls_just_found = new_collected - old_collected
        if scrolls_just_found > 0:
            reward += 0.15 * scrolls_just_found

        # --- ORB pickup bonus ---
        old_orb = old_state.get("config", {}).get("_has_orb", False)
        new_orb = config.get("_has_orb", False)
        if new_orb and not old_orb:
            reward += 0.3

        # --- Goal reached ---
        if self.check_success(new_state):
            if config.get("_has_orb", False):
                reward += 1.0
            else:
                reward += 0.2

        # --- Approach shaping toward next objective -----------------------
        # If orb not yet obtained, shape toward scrolls/orb.
        # If orb obtained, shape toward goal.
        if "agent" in new_state and "agent" in old_state:
            ax, ay = new_state["agent"].position
            ox, oy = old_state["agent"].position

            if config.get("_has_orb", False):
                # Shape toward goal
                goal = config.get("goal_positions", [None])[0]
                if goal:
                    old_dist = abs(ox - goal[0]) + abs(oy - goal[1])
                    new_dist = abs(ax - goal[0]) + abs(ay - goal[1])
                    reward += 0.02 * (old_dist - new_dist)
            elif config.get("_orb_spawned", False):
                # Shape toward orb
                orb_pos = config.get("orb_spawn_pos")
                if orb_pos:
                    old_dist = abs(ox - orb_pos[0]) + abs(oy - orb_pos[1])
                    new_dist = abs(ax - orb_pos[0]) + abs(ay - orb_pos[1])
                    reward += 0.02 * (old_dist - new_dist)

        return reward

    def compute_sparse_reward(self, old_state, action, new_state, info):
        if self.check_success(new_state):
            config = new_state.get("config", {})
            if config.get("_has_orb", False):
                return 1.0
            return 0.2
        return 0.0

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def check_success(self, state):
        if "grid" not in state or "agent" not in state:
            return False
        x, y = state["agent"].position
        return bool(state["grid"].objects[y, x] == ObjectType.GOAL)

    def check_done(self, state):
        return self.check_success(state)

    # ------------------------------------------------------------------
    # Baselines / validation
    # ------------------------------------------------------------------

    def validate_instance(self, grid, config):
        agent_pos = tuple(config.get("agent_start", (1, 1)))
        # Scrolls must be reachable WITHOUT crossing water
        safe_reach = _bfs_reachable(grid, agent_pos, {int(CellType.EMPTY)})
        for sp in config.get("scroll_positions", []):
            if tuple(sp) not in safe_reach:
                return False
        # Goal must be reachable (including through water)
        full_reach = grid.flood_fill(agent_pos)
        for gp in config.get("goal_positions", []):
            if tuple(gp) not in full_reach:
                return False
        return True

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0
