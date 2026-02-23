"""ToolUse - Navigate a winding path; use tools as shortcuts to reach the goal faster.

MECHANICS:
  - Tool-obstacle pairs (visually obvious):
    - GEM (torch): pass through HAZARD (lava) safely
    - KEY (bridge): cross WATER cells safely
    - TOOL (hammer): break through a cracked WALL obstacle
  - Agent walks into tool object → auto-pickup
  - Agent attempts to enter barrier shortcut → tool is consumed for that passage
  - Tools have limited durability (N uses)
  - Without correct tool: HAZARD kills → episode fail, WATER/WALL blocks

LEVEL DESIGN (not random!):
  - The map is a serpentine (zigzag) corridor from start (1,1) to goal (size-2, size-2).
  - At each shortcut point a 1-2 cell barrier (WATER / HAZARD / WALL) bridges two
    parallel sections of the zigzag, offering a shorter path through if the matching
    tool is held.
  - The matching tool is placed on the long path BEFORE the shortcut.
  - Using shortcuts saves many steps → the step penalty makes shortcuts strictly better.

LAYOUT SKETCH (easy, 9x9 grid):
  S → → → → → → → ↓
                   ↓
  ← ← ← ← [WTR] ←←  ← long detour starts here
  ↓
  → → → → → → → → ↓
                   ↓
  ← ← ← ← ← ← ← ← G

  The WATER column is 1 cell; walking east through it (with KEY) skips the left-side
  detour leg.  Without key the agent must walk all the way west and wrap around south.
"""

from __future__ import annotations

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task

_TOOL_OBSTACLES = {
    "torch": CellType.HAZARD,
    "bridge": CellType.WATER,
    "hammer": CellType.WALL,
}
_TOOL_OBJECT = {
    "torch": ObjectType.GEM,    # GEM = torch (visual: bright gem)
    "bridge": ObjectType.KEY,   # KEY = bridge (visual: key shape)
    "hammer": ObjectType.TOOL,  # TOOL = hammer (visual: tool)
}

# Shortcut assignment per difficulty: list of tool names in the order the
# shortcuts appear top-to-bottom in the zigzag.
_DIFFICULTY_SHORTCUTS: dict[str, list[str]] = {
    "easy":   ["bridge"],
    "medium": ["bridge", "torch"],
    "hard":   ["bridge", "torch", "hammer"],
    "expert": ["bridge", "torch", "hammer"],
}


def _build_zigzag_layout(size: int, n_shortcuts: int) -> dict:
    """Return layout information for a zigzag corridor with shortcut slots.

    The zigzag uses every other interior row as a horizontal corridor.
    Rows that carry the corridor:  1, 3, 5, 7, ...  (up to size-2)
    Vertical connectors: at column (size-2) linking row k to row k+2 going south,
                         at column 1 linking row k+2 to row k+4 going south, etc.

    Shortcut slots are placed between consecutive corridor rows.

    Returns:
        dict with:
            corridor_rows  – list of y values of horizontal corridors
            connector_cols – list of x values of each vertical connector
            shortcut_cols  – list of (x, y1, y2) for shortcut barrier position
                             between corridor row y1 and the next corridor row y2
            path_long      – list of (x,y) cells in traversal order (long route)
    """
    # Corridor rows: 1, 3, 5, ... up to size-2 (inclusive if possible)
    corridor_rows: list[int] = []
    row = 1
    while row <= size - 2:
        corridor_rows.append(row)
        row += 2

    # We need at least 2 corridor rows (for at least one shortcut slot)
    # n_shortcuts is capped at len(corridor_rows) - 1
    n_shortcuts = min(n_shortcuts, len(corridor_rows) - 1)

    # Connector columns alternate: east side (size-2), west side (1)
    # connector_cols[i] is the x-column of the vertical link between
    # corridor_rows[i] and corridor_rows[i+1]
    connector_cols: list[int] = []
    for i in range(len(corridor_rows) - 1):
        # Even index → connector at east (size-2); odd index → at west (1)
        if i % 2 == 0:
            connector_cols.append(size - 2)
        else:
            connector_cols.append(1)

    # Direction of each corridor row: even index → east (left to right),
    # odd index → west (right to left).
    # corridor_rows[0] goes east, [1] goes west, [2] east, ...

    # Shortcut columns: placed in the middle of the grid horizontally.
    # A shortcut between corridor_rows[i] and corridor_rows[i+1] is a single
    # cell at (shortcut_x, rows[i]+1) and (shortcut_x, rows[i+2]-1) i.e. the
    # intermediate wall row. Actually the barrier is 1 cell at
    # (shortcut_x, mid_row) where mid_row = corridor_rows[i] + 1 (==
    # corridor_rows[i+1] - 1 because rows are 2 apart).
    # The shortcut_x should not be the connector column for this segment
    # (which is already open as a vertical connector).
    mid_x = size // 2  # preferred x for shortcuts
    shortcut_slots: list[tuple[int, int]] = []  # (barrier_x, barrier_y)
    for i in range(n_shortcuts):
        y_top = corridor_rows[i]
        y_mid = y_top + 1  # the wall row between them (rows are always 2 apart)
        # Choose x: midpoint, but avoid connector column for this segment
        conn_col = connector_cols[i]
        sx = mid_x
        if sx == conn_col:
            sx = mid_x + 1 if mid_x + 1 < size - 1 else mid_x - 1
        # Also keep away from outer walls
        sx = max(2, min(size - 3, sx))
        shortcut_slots.append((sx, y_mid))

    # Build the long path in traversal order.
    # Start at (1, 1). The long path does NOT use shortcuts.
    path_long: list[tuple[int, int]] = []
    for i, row in enumerate(corridor_rows):
        if i % 2 == 0:
            # Going east: from col 1 to col size-2
            xs = range(1, size - 1) if i == 0 else range(1, size - 1)
            path_long.extend((x, row) for x in xs)
        else:
            # Going west: from col size-2 to col 1
            path_long.extend((x, row) for x in range(size - 2, 0, -1))
        # Add vertical connector cells (if not last row)
        if i < len(corridor_rows) - 1:
            conn_x = connector_cols[i]
            y_start = corridor_rows[i] + 1
            y_end = corridor_rows[i + 1]  # exclusive — corridor row already in next iter
            path_long.extend((conn_x, y) for y in range(y_start, y_end))

    return {
        "corridor_rows": corridor_rows,
        "connector_cols": connector_cols,
        "shortcut_slots": shortcut_slots,  # list of (x, y) for barrier cells
        "path_long": path_long,
    }


@register_task("ToolUse-v0", tags=["compositional_logic", "planning"])
class ToolUseTask(TaskSpec):
    """Navigate a winding corridor; use tools as shortcuts to reach goal faster."""

    name = "ToolUse-v0"
    description = "Use the right tool to take shortcuts on a long winding path"
    capability_tags = ["compositional_logic", "planning"]
    overrides_walkable = True

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=9,
            max_steps=80,
            params={
                "shortcuts": ["bridge"],
                "tool_durability": 3,
            },
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=11,
            max_steps=160,
            params={
                "shortcuts": ["bridge", "torch"],
                "tool_durability": 2,
            },
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=14,
            max_steps=280,
            params={
                "shortcuts": ["bridge", "torch", "hammer"],
                "tool_durability": 2,
            },
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=16,
            max_steps=450,
            params={
                "shortcuts": ["bridge", "torch", "hammer"],
                "tool_durability": 1,
            },
        ),
    }

    # ------------------------------------------------------------------
    # Level generation
    # ------------------------------------------------------------------

    def generate(self, seed):  # noqa: ARG002  (seed unused – designed levels)
        size = self.difficulty_config.grid_size
        shortcuts_order: list[str] = self.difficulty_config.params.get("shortcuts", ["bridge"])
        durability: int = self.difficulty_config.params.get("tool_durability", 2)
        n_shortcuts = len(shortcuts_order)

        # --- 1. Start from all-WALL interior, border already WALL by default ---
        grid = Grid(size, size)
        # Set entire grid to WALL terrain
        grid.terrain[:, :] = CellType.WALL

        # --- 2. Compute zigzag layout ---
        layout = _build_zigzag_layout(size, n_shortcuts)
        corridor_rows: list[int] = layout["corridor_rows"]
        connector_cols: list[int] = layout["connector_cols"]
        shortcut_slots: list[tuple[int, int]] = layout["shortcut_slots"]

        # --- 3. Carve horizontal corridors (set to EMPTY) ---
        for row in corridor_rows:
            for x in range(1, size - 1):
                grid.terrain[row, x] = CellType.EMPTY

        # --- 4. Carve vertical connectors ---
        for i, conn_x in enumerate(connector_cols):
            y_start = corridor_rows[i] + 1
            y_end = corridor_rows[i + 1]  # end is next corridor row, already EMPTY
            for y in range(y_start, y_end):
                grid.terrain[y, conn_x] = CellType.EMPTY

        # --- 5. Place shortcut barriers ---
        # shortcut_slots is aligned with shortcuts_order
        barrier_info: dict[str, dict] = {}
        for i, tool_name in enumerate(shortcuts_order):
            if i >= len(shortcut_slots):
                break
            sx, sy = shortcut_slots[i]
            obs_type = _TOOL_OBSTACLES[tool_name]
            grid.terrain[sy, sx] = obs_type
            barrier_info[tool_name] = {
                "cells": [[sx, sy]],
                "obstacle_type": int(obs_type),
            }

        # --- 6. Place tools along the long path, BEFORE their shortcut ---
        # The long path visits cells in order.  For each barrier (shortcut), find
        # the index in path_long of the LAST cell that is still in the corridor row
        # adjacent to the barrier AND has not yet passed the barrier column.
        # The tool is then placed midway between the previous barrier zone and that index.
        path_long: list[tuple[int, int]] = layout["path_long"]
        tool_positions: dict[str, list[int]] = {}

        # For each shortcut, compute the corridor-row index where the agent is
        # about to reach the shortcut column.  The barrier cell is at (sx, sy)
        # where sy is a "wall row" between two corridor rows.  The corridor row
        # *above* the barrier is sy-1.  When the long path traverses that row,
        # it will reach sx at some index.  All path cells in that row with index
        # BEFORE reaching sx are valid tool-placement zones.
        barrier_path_indices: dict[str, int] = {}
        for tool_name, binfo in barrier_info.items():
            bx, by = binfo["cells"][0]
            target_row = by - 1  # corridor row above the barrier
            # Walk path_long and find all indices where py == target_row and
            # the cell is *before* the shortcut column on that row.
            # We want the LAST such index (farthest along the path that is still
            # before the shortcut entry point).
            # Also accept the cell AT bx (that is the entry point itself).
            last_before = None
            for pidx, (px, py) in enumerate(path_long):
                if py == target_row:
                    # Accept cells up to and including bx
                    if px <= bx:
                        last_before = pidx
                    else:
                        # Once we pass bx on an east-going row, stop searching
                        break
            if last_before is None:
                # Row may go west; accept cells at or after bx (approaching from east)
                for pidx, (px, py) in enumerate(path_long):
                    if py == target_row and px >= bx:
                        last_before = pidx
                        break
            if last_before is None:
                # Absolute fallback: first cell of target_row in path
                for pidx, (px, py) in enumerate(path_long):
                    if py == target_row:
                        last_before = pidx
                        break
            if last_before is None:
                last_before = len(path_long) // 2
            barrier_path_indices[tool_name] = last_before

        used_positions: set[tuple[int, int]] = {(1, 1)}  # agent start
        # Place goal first so we don't overwrite it
        goal_pos = (size - 2, size - 2)
        used_positions.add(goal_pos)

        # Sort tools by their barrier index (earlier barriers → earlier tools)
        sorted_tools = sorted(shortcuts_order, key=lambda t: barrier_path_indices.get(t, 0))

        prev_end_idx = 0
        for tool_name in sorted_tools:
            barrier_idx = barrier_path_indices[tool_name]
            # Include cells from prev_end_idx+1 up to (not including) barrier_idx.
            # Use +1 offset from prev to avoid agent start and be clear of last tool.
            zone_start = max(1, prev_end_idx + 1)
            zone_end = barrier_idx  # exclusive, so tool lands at or before barrier entry
            candidates = [
                path_long[j]
                for j in range(zone_start, zone_end)
                if path_long[j] not in used_positions
                and int(grid.terrain[path_long[j][1], path_long[j][0]]) == int(CellType.EMPTY)
            ]
            if candidates:
                # Place tool at the midpoint of the candidate zone for visibility
                pos = candidates[len(candidates) // 2]
                tx, ty = pos
                obj_type = _TOOL_OBJECT[tool_name]
                grid.objects[ty, tx] = obj_type
                tool_positions[tool_name] = [tx, ty]
                used_positions.add(pos)
            prev_end_idx = barrier_idx + 1

        # --- 7. Place GOAL at end of last corridor row ---
        gx, gy = goal_pos
        # Ensure goal cell is EMPTY (it's on the last corridor row)
        grid.terrain[gy, gx] = CellType.EMPTY
        grid.objects[gy, gx] = ObjectType.GOAL

        # Ensure agent start is EMPTY
        grid.terrain[1, 1] = CellType.EMPTY

        return grid, {
            "agent_start": (1, 1),
            "goal_positions": [goal_pos],
            "shortcuts": shortcuts_order,
            "tool_positions": tool_positions,
            "barrier_info": {
                k: {"cells": [list(c) for c in v["cells"]], "obstacle_type": v["obstacle_type"]}
                for k, v in barrier_info.items()
            },
            "tool_durability": durability,
            "max_steps": self.get_max_steps(),
        }

    # ------------------------------------------------------------------
    # Environment hooks
    # ------------------------------------------------------------------

    def on_env_reset(self, agent, grid, config):
        agent.inventory.clear()
        config["_tool_uses"] = {}
        config["_lava_death"] = False
        self._config = config

    def can_agent_enter(self, pos, agent, grid):
        x, y = pos
        terrain = grid.terrain[y, x]
        config = getattr(self, "_config", {})
        durability = config.get("tool_durability", 2)

        if terrain == CellType.HAZARD:
            tool = next((e for e in agent.inventory if e.entity_type == "torch"), None)
            if tool is None:
                return False
            uses = config.get("_tool_uses", {})
            if durability > 0 and uses.get(tool.id, 0) >= durability:
                return False
            return True
        elif terrain == CellType.WATER:
            tool = next((e for e in agent.inventory if e.entity_type == "bridge"), None)
            if tool is None:
                return False
            uses = config.get("_tool_uses", {})
            if durability > 0 and uses.get(tool.id, 0) >= durability:
                return False
            return True
        elif terrain == CellType.WALL:
            tool = next((e for e in agent.inventory if e.entity_type == "hammer"), None)
            if tool is None:
                return False
            uses = config.get("_tool_uses", {})
            if durability > 0 and uses.get(tool.id, 0) >= durability:
                return False
            return True
        return True

    def on_agent_moved(self, pos, agent, grid):
        from agentick.core.entity import Entity

        config = getattr(self, "_config", {})
        x, y = pos
        terrain = grid.terrain[y, x]
        durability = config.get("tool_durability", 2)

        # Auto-pickup tools
        obj = grid.objects[y, x]
        tool_map = {
            ObjectType.GEM: "torch",
            ObjectType.KEY: "bridge",
            ObjectType.TOOL: "hammer",
        }
        if obj in tool_map:
            tool_name = tool_map[obj]
            grid.objects[y, x] = ObjectType.NONE
            agent.inventory.append(
                Entity(id=f"{tool_name}_{x}_{y}", entity_type=tool_name, position=pos)
            )

        # Consume tool use when crossing a barrier cell
        if terrain == CellType.HAZARD:
            tool = next((e for e in agent.inventory if e.entity_type == "torch"), None)
            if tool is None:
                config["_lava_death"] = True
                return
            uses = config.setdefault("_tool_uses", {})
            uses[tool.id] = uses.get(tool.id, 0) + 1
            if durability > 0 and uses[tool.id] >= durability:
                agent.inventory.remove(tool)

        elif terrain == CellType.WATER:
            tool = next((e for e in agent.inventory if e.entity_type == "bridge"), None)
            if tool:
                uses = config.setdefault("_tool_uses", {})
                uses[tool.id] = uses.get(tool.id, 0) + 1
                if durability > 0 and uses[tool.id] >= durability:
                    agent.inventory.remove(tool)

        elif terrain == CellType.WALL:
            tool = next((e for e in agent.inventory if e.entity_type == "hammer"), None)
            if tool:
                grid.terrain[y, x] = CellType.EMPTY
                uses = config.setdefault("_tool_uses", {})
                uses[tool.id] = uses.get(tool.id, 0) + 1
                if durability > 0 and uses[tool.id] >= durability:
                    agent.inventory.remove(tool)

    # ------------------------------------------------------------------
    # Reward and termination
    # ------------------------------------------------------------------

    def compute_dense_reward(self, old_state, action, new_state, info):
        config = new_state.get("config", {})
        if config.get("_lava_death", False):
            return -1.0
        reward = -0.01  # step penalty encourages shortcuts
        if "agent" in new_state:
            goal = config.get("goal_positions", [None])[0]
            if goal:
                ax, ay = new_state["agent"].position
                ox, oy = old_state.get("agent_position", new_state["agent"].position)
                reward += 0.05 * (
                    abs(ox - goal[0]) + abs(oy - goal[1]) - abs(ax - goal[0]) - abs(ay - goal[1])
                )
        if self.check_success(new_state):
            reward += 1.0
        return reward

    def compute_sparse_reward(self, old_state, action, new_state, info):
        config = new_state.get("config", {})
        if config.get("_lava_death", False):
            return -1.0
        if self.check_success(new_state):
            return 1.0
        return 0.0

    def check_done(self, state):
        config = state.get("config", {})
        if config.get("_lava_death", False):
            return True
        return self.check_success(state)

    def check_success(self, state):
        config = state.get("config", {})
        if config.get("_lava_death", False):
            return False
        if "grid" not in state or "agent" not in state:
            return False
        x, y = state["agent"].position
        return bool(state["grid"].objects[y, x] == ObjectType.GOAL)

    def validate_instance(self, grid, config):
        return True

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0
