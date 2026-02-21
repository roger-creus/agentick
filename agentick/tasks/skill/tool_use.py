"""ToolUse - Pick up specific tools to bypass different obstacle types.

MECHANICS:
  - Multiple tool types with different effects:
    - TOOL (hammer): breaks through WALL obstacles (one use)
    - KEY (bridge): crosses WATER cells safely
    - GEM (torch): illuminates and passes through HAZARD (lava) safely
  - Each obstacle type requires the matching tool
  - Without correct tool: HAZARD kills, WATER blocks, WALL blocks
  - Tool durability: each tool has limited uses before breaking
  - Must plan which tools to collect and in what order
  - Success = reach GOAL using appropriate tools
"""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task

# Tool type → obstacle it can bypass
_TOOL_OBSTACLES = {
    "hammer": CellType.WALL,  # TOOL object → breaks walls
    "bridge": CellType.WATER,  # KEY object → crosses water
    "torch": CellType.HAZARD,  # GEM object → passes lava
}

_TOOL_OBJECT = {
    "hammer": ObjectType.TOOL,
    "bridge": ObjectType.KEY,
    "torch": ObjectType.GEM,
}


@register_task("ToolUse-v0", tags=["compositional_logic", "planning"])
class ToolUseTask(TaskSpec):
    """Collect specific tools to bypass different obstacle types and reach the goal."""

    name = "ToolUse-v0"
    description = "Use different tools to bypass different obstacles"
    capability_tags = ["compositional_logic", "planning"]
    overrides_walkable = True  # hammer breaks WALLs, handled in can_agent_enter

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=7,
            max_steps=80,
            params={
                "tools": ["torch"],
                "tool_durability": 3,
                "n_decoys": 0,
            },
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=10,
            max_steps=150,
            params={
                "tools": ["torch", "bridge"],
                "tool_durability": 2,
                "n_decoys": 1,
            },
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=13,
            max_steps=250,
            params={
                "tools": ["torch", "bridge", "hammer"],
                "tool_durability": 2,
                "n_decoys": 2,
            },
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=15,
            max_steps=400,
            params={
                "tools": ["torch", "bridge", "hammer"],
                "tool_durability": 1,
                "n_decoys": 3,
            },
        ),
    }

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

        # Place obstacle barriers for each tool type
        barrier_info = {}
        n_barriers = len(tools)
        for i, tool_name in enumerate(tools):
            obstacle_type = _TOOL_OBSTACLES[tool_name]
            # Place a horizontal row of obstacles
            row = 2 + (i + 1) * (size - 4) // (n_barriers + 1)
            row = max(2, min(size - 3, row))
            barrier_cells = []
            for x in range(1, size - 1):
                if grid.terrain[row, x] == CellType.EMPTY:
                    grid.terrain[row, x] = obstacle_type
                    barrier_cells.append((x, row))
            barrier_info[tool_name] = {"row": row, "cells": barrier_cells}

        # Place tools above their corresponding barriers
        tool_positions = {}
        for i, tool_name in enumerate(tools):
            obj_type = _TOOL_OBJECT[tool_name]
            barrier_row = barrier_info[tool_name]["row"]
            # Tool goes between agent area and its barrier
            tool_candidates = [
                (x, y)
                for x in range(1, size - 1)
                for y in range(1, barrier_row)
                if grid.terrain[y, x] == CellType.EMPTY and (x, y) not in used
            ]
            if not tool_candidates:
                tool_candidates = [
                    (x, y)
                    for x in range(1, size - 1)
                    for y in range(1, size - 1)
                    if grid.terrain[y, x] == CellType.EMPTY and (x, y) not in used
                ]
            if tool_candidates:
                pos = tool_candidates[int(rng.integers(len(tool_candidates)))]
                tx, ty = pos
                grid.objects[ty, tx] = obj_type
                tool_positions[tool_name] = pos
                used.add(pos)

        # Decoys: wrong tool types (can't help with any barrier)
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
            "tool_positions": {k: list(v) for k, v in tool_positions.items()},
            "barrier_info": {
                k: {"row": v["row"], "cells": [list(c) for c in v["cells"]]}
                for k, v in barrier_info.items()
            },
            "tool_durability": durability,
            "decoy_positions": decoy_positions,
            "max_steps": self.get_max_steps(),
        }

    def on_env_reset(self, agent, grid, config):
        agent.inventory.clear()
        config["_tool_uses"] = {}  # tool_id -> uses count
        config["_lava_death"] = False
        self._config = config

    def can_agent_enter(self, pos, agent, grid):
        x, y = pos
        terrain = grid.terrain[y, x]
        config = getattr(self, "_config", {})
        durability = config.get("tool_durability", 2)

        if terrain == CellType.HAZARD:
            # Need torch (GEM in inventory)
            tool = next((e for e in agent.inventory if e.entity_type == "torch"), None)
            if tool is None:
                return False
            uses = config.get("_tool_uses", {})
            if durability > 0 and uses.get(tool.id, 0) >= durability:
                return False
            return True
        elif terrain == CellType.WATER:
            # Need bridge (KEY in inventory)
            tool = next((e for e in agent.inventory if e.entity_type == "bridge"), None)
            if tool is None:
                return False
            uses = config.get("_tool_uses", {})
            if durability > 0 and uses.get(tool.id, 0) >= durability:
                return False
            return True
        elif terrain == CellType.WALL:
            # Need hammer (TOOL in inventory) — breaks through
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

        # Pick up tools
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
                Entity(
                    id=f"{tool_name}_{x}_{y}",
                    entity_type=tool_name,
                    position=pos,
                )
            )

        # Use tools on obstacles
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
                # Break the wall permanently
                grid.terrain[y, x] = CellType.EMPTY
                uses = config.setdefault("_tool_uses", {})
                uses[tool.id] = uses.get(tool.id, 0) + 1
                if durability > 0 and uses[tool.id] >= durability:
                    agent.inventory.remove(tool)

    def compute_sparse_reward(self, old_state, action, new_state, info):
        config = new_state.get("config", {})
        if config.get("_lava_death", False):
            return -1.0
        if self.check_success(new_state):
            return 1.0
        return 0.0

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
        return True  # dynamic obstacle passability

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0
