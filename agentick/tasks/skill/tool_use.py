"""ToolUse - Use specific tools to bypass specific obstacle types and reach the goal.

MECHANICS:
  - Tool-obstacle pairs (visually obvious):
    - GEM (torch): pass through HAZARD (lava) safely
    - KEY (bridge): cross WATER cells safely
    - TOOL (hammer): break through a cracked WALL obstacle
  - Agent walks into tool object → auto-pickup
  - Agent attempts to enter obstacle → tool is consumed for that passage
  - Tools have limited durability (N uses)
  - Without correct tool: HAZARD kills → episode fail, WATER blocks
  - GOAL is behind a sequence of obstacles

LEVEL DESIGN (not random!):
  - Each level has a clear designed layout:
    - Easy:   one obstacle type, tool placed visibly before obstacle, straight path to goal
    - Medium: two obstacle types, tools on separate sides, must plan order
    - Hard:   three obstacle types, decoy tools, tool chaining needed
    - Expert: all three, tools with durability 1, multiple obstacle barriers

VISIBILITY:
  - Obstacles are full rows/columns, not scattered patches
  - Tools are placed in the quadrant BEFORE their matching obstacle
  - Agent can clearly see: obstacle → find tool → use tool → proceed
"""

import numpy as np

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


@register_task("ToolUse-v0", tags=["compositional_logic", "planning"])
class ToolUseTask(TaskSpec):
    """Collect specific tools to bypass different obstacle types and reach goal."""

    name = "ToolUse-v0"
    description = "Use the right tool to bypass each obstacle type"
    capability_tags = ["compositional_logic", "planning"]
    overrides_walkable = True

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=9,
            max_steps=80,
            params={
                "tools": ["torch"],
                "tool_durability": 3,
                "n_decoys": 0,
                "gap_width": 1,  # width of obstacle barrier gap (0=solid row)
            },
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=11,
            max_steps=160,
            params={
                "tools": ["torch", "bridge"],
                "tool_durability": 2,
                "n_decoys": 1,
                "gap_width": 0,
            },
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=14,
            max_steps=280,
            params={
                "tools": ["torch", "bridge", "hammer"],
                "tool_durability": 2,
                "n_decoys": 2,
                "gap_width": 0,
            },
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=16,
            max_steps=450,
            params={
                "tools": ["torch", "bridge", "hammer"],
                "tool_durability": 1,
                "n_decoys": 3,
                "gap_width": 0,
            },
        ),
    }

    def _place_barrier(self, grid, row, tool_name, col_range, gap_col=None):
        """Place a full horizontal barrier of the obstacle type for this tool."""
        obs_type = _TOOL_OBSTACLES[tool_name]
        barrier_cells = []
        for x in col_range:
            if gap_col is not None and x == gap_col:
                continue  # leave a gap in the barrier (requires tool to pass anywhere else)
            if obs_type == CellType.WALL:
                # For walls: mark as WALL terrain, but won't block tool user
                grid.terrain[row, x] = obs_type
            else:
                grid.terrain[row, x] = obs_type
            barrier_cells.append((x, row))
        return barrier_cells

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        tools = self.difficulty_config.params.get("tools", ["torch"])
        durability = self.difficulty_config.params.get("tool_durability", 2)
        n_decoys = self.difficulty_config.params.get("n_decoys", 0)

        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        agent_pos = (1, 1)
        goal_pos = (size - 2, size - 2)
        grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.GOAL
        used = {agent_pos, goal_pos}

        n_barriers = len(tools)
        barrier_info = {}

        # Place barrier rows evenly spaced from top to bottom
        barrier_rows = []
        for i in range(n_barriers):
            row = 2 + (i + 1) * (size - 4) // (n_barriers + 1)
            row = max(2, min(size - 3, row))
            barrier_rows.append(row)

        # Place barriers: full row except column 0 and size-1 (walls)
        for i, tool_name in enumerate(tools):
            row = barrier_rows[i]
            obs_type = _TOOL_OBSTACLES[tool_name]
            cells = []
            for x in range(1, size - 1):
                grid.terrain[row, x] = obs_type
                cells.append((x, row))
            barrier_info[tool_name] = {
                "row": row,
                "cells": cells,
                "obstacle_type": int(obs_type),
            }

        # Place tools: in the region ABOVE their barrier (visually before the obstacle)
        # Tool appears in the left half of the area above the barrier
        tool_positions = {}
        for i, tool_name in enumerate(tools):
            obj_type = _TOOL_OBJECT[tool_name]
            barrier_row = barrier_rows[i]
            prev_row = barrier_rows[i - 1] if i > 0 else 0

            # Tool candidates: in the zone between previous barrier and this one
            candidates = [
                (x, y)
                for x in range(1, size - 1)
                for y in range(prev_row + 1, barrier_row)
                if grid.terrain[y, x] == CellType.EMPTY and (x, y) not in used
            ]
            if not candidates:
                candidates = [
                    (x, y)
                    for x in range(1, size - 1)
                    for y in range(1, size - 1)
                    if grid.terrain[y, x] == CellType.EMPTY and (x, y) not in used
                ]
            if candidates:
                # Pick a visible position (not in corner, roughly centered x)
                cx_target = size // 3
                candidates.sort(key=lambda p: abs(p[0] - cx_target))
                pos = candidates[0]
                tx, ty = pos
                grid.objects[ty, tx] = obj_type
                tool_positions[tool_name] = pos
                used.add(pos)

        # Decoys: wrong tool types (useless for any barrier)
        decoy_positions = []
        decoy_types = [ObjectType.SCROLL, ObjectType.ORB, ObjectType.LEVER]
        free = [
            (x, y)
            for x in range(1, size - 1)
            for y in range(1, size - 1)
            if grid.terrain[y, x] == CellType.EMPTY and (x, y) not in used
        ]
        rng.shuffle(free)
        for i in range(min(n_decoys, len(free))):
            dx, dy = free[i]
            dt = decoy_types[i % len(decoy_types)]
            grid.objects[dy, dx] = dt
            decoy_positions.append(free[i])
            used.add(free[i])

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [goal_pos],
            "tools": tools,
            "tool_positions": {k: list(v2) for k, v2 in tool_positions.items()},
            "barrier_info": {
                k: {"row": v["row"], "cells": [list(c) for c in v["cells"]],
                    "obstacle_type": v["obstacle_type"]}
                for k, v in barrier_info.items()
            },
            "tool_durability": durability,
            "decoy_positions": decoy_positions,
            "max_steps": self.get_max_steps(),
        }

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

    def compute_dense_reward(self, old_state, action, new_state, info):
        config = new_state.get("config", {})
        if config.get("_lava_death", False):
            return -1.0
        reward = -0.01
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
