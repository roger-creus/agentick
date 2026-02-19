"""ToolUse - Pick up a tool to bypass an obstacle, then reach the goal.

MECHANICS:
  - A KEY (tool) is placed on one side of the grid
  - A row of HAZARD obstacles blocks the direct path to the goal
  - WITH the key in inventory, agent can pass through hazard (can_agent_enter)
  - WITHOUT the key, hazard ends episode
  - Success = reach GOAL with key in inventory
"""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("ToolUse-v0", tags=["compositional_logic", "planning"])
class ToolUseTask(TaskSpec):
    """Collect the tool to bypass obstacles and reach the goal."""

    name = "ToolUse-v0"
    description = "Pick up tool to bypass obstacles and reach goal"
    capability_tags = ["compositional_logic", "planning"]

    difficulty_configs = {
        "easy":   DifficultyConfig(
            name="easy", grid_size=7, max_steps=80,
            params={
                "n_tools": 1, "n_obstacles": 0, "n_decoys": 0,
                "n_barriers": 1, "tool_durability": 0,
            },
        ),
        "medium": DifficultyConfig(
            name="medium", grid_size=10, max_steps=150,
            params={
                "n_tools": 2, "n_obstacles": 3, "n_decoys": 1,
                "n_barriers": 1, "tool_durability": 0,
            },
        ),
        "hard":   DifficultyConfig(
            name="hard", grid_size=13, max_steps=250,
            params={
                "n_tools": 2, "n_obstacles": 5, "n_decoys": 2,
                "n_barriers": 2, "tool_durability": 3,
            },
        ),
        "expert": DifficultyConfig(
            name="expert", grid_size=15, max_steps=400,
            params={
                "n_tools": 3, "n_obstacles": 8, "n_decoys": 3,
                "n_barriers": 3, "tool_durability": 2,
            },
        ),
    }

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size           = self.difficulty_config.grid_size
        n_tools        = self.difficulty_config.params.get("n_tools", 1)
        n_obstacles    = self.difficulty_config.params.get("n_obstacles", 0)
        n_decoys       = self.difficulty_config.params.get("n_decoys", 0)
        n_barriers     = self.difficulty_config.params.get("n_barriers", 1)
        tool_durability = self.difficulty_config.params.get(
            "tool_durability", 0
        )

        grid = Grid(size, size)
        grid.terrain[0, :]  = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0]  = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        # Place multiple horizontal hazard barriers evenly spaced
        interior = size - 2
        barrier_rows = []
        for i in range(n_barriers):
            row = 1 + (i + 1) * interior // (n_barriers + 1)
            row = max(2, min(size - 3, row))
            barrier_rows.append(row)
            for x in range(1, size - 1):
                grid.terrain[row, x] = CellType.HAZARD

        agent_pos = (1, 1)
        goal_pos  = (size - 2, size - 2)
        grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.GOAL

        first_barrier = barrier_rows[0]
        top_free = [
            (x, y) for x in range(1, size - 1)
            for y in range(1, first_barrier)
            if (x, y) not in {agent_pos, goal_pos}
            and grid.terrain[y, x] == CellType.EMPTY
        ]
        rng.shuffle(top_free)
        used = {agent_pos, goal_pos}

        tool_positions = []
        for p in top_free[:n_tools]:
            tx, ty = p
            grid.objects[ty, tx] = ObjectType.KEY
            tool_positions.append(p)
            used.add(p)

        last_barrier = barrier_rows[-1]
        bottom_free = [
            (x, y) for x in range(1, size - 1)
            for y in range(last_barrier + 1, size - 1)
            if (x, y) not in used
            and grid.terrain[y, x] == CellType.EMPTY
        ]
        rng.shuffle(bottom_free)
        decoy_positions = []
        for p in bottom_free[:n_decoys]:
            dx2, dy2 = p
            grid.objects[dy2, dx2] = ObjectType.TARGET
            decoy_positions.append(p)
            used.add(p)

        wall_positions = []
        wall_candidates = [
            p for p in top_free[n_tools:] if p not in used
        ]
        for p in wall_candidates:
            if len(wall_positions) >= n_obstacles:
                break
            wx, wy = p
            grid.terrain[wy, wx] = CellType.WALL
            reachable = grid.flood_fill(agent_pos)
            if all(tp in reachable for tp in tool_positions):
                wall_positions.append(p)
                used.add(p)
            else:
                grid.terrain[wy, wx] = CellType.EMPTY

        tool_pos = tool_positions[0] if tool_positions else (2, 1)

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [goal_pos],
            "tool_pos": tool_pos,
            "tool_positions": tool_positions,
            "decoy_positions": decoy_positions,
            "barrier_rows": barrier_rows,
            "tool_durability": tool_durability,
            "max_steps": self.get_max_steps(),
        }

    def on_env_reset(self, agent, grid, config):
        agent.inventory.clear()
        config["_tool_uses"] = {}
        self._config = config

    def can_agent_enter(self, pos, agent, grid):
        x, y = pos
        if grid.terrain[y, x] == CellType.HAZARD:
            tool = next(
                (e for e in agent.inventory if e.entity_type == "tool"),
                None,
            )
            if tool is None:
                return False
            config = getattr(self, "_config", {})
            durability = config.get("tool_durability", 0)
            if durability > 0:
                uses = config.get("_tool_uses", {})
                if uses.get(tool.id, 0) >= durability:
                    return False
            return True
        return True

    def on_agent_moved(self, pos, agent, grid):
        from agentick.core.entity import Entity
        config = getattr(self, "_config", {})
        x, y = pos
        if grid.objects[y, x] == ObjectType.KEY:
            grid.objects[y, x] = ObjectType.NONE
            agent.inventory.append(
                Entity(
                    id=f"tool_{x}_{y}",
                    entity_type="tool",
                    position=pos,
                )
            )
        if grid.terrain[y, x] == CellType.HAZARD:
            durability = config.get("tool_durability", 0)
            if durability > 0:
                tool = next(
                    (e for e in agent.inventory
                     if e.entity_type == "tool"),
                    None,
                )
                if tool:
                    uses = config.setdefault("_tool_uses", {})
                    uses[tool.id] = uses.get(tool.id, 0) + 1
                    if uses[tool.id] >= durability:
                        agent.inventory.remove(tool)

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        config = new_state.get("config", {})
        agent = new_state.get("agent")
        if agent:
            has_tool = any(e.entity_type == "tool" for e in agent.inventory)
            if not has_tool:
                # Reward getting to tool
                tool = config.get("tool_pos")
                if tool:
                    ax, ay = agent.position
                    ox, oy = old_state.get('agent_position', new_state['agent'].position)
                    reward += 0.05 * (abs(ox-tool[0])+abs(oy-tool[1]) -
                                      abs(ax-tool[0])-abs(ay-tool[1]))
            else:
                goal = config.get("goal_positions", [None])[0]
                if goal:
                    ax, ay = agent.position
                    ox, oy = old_state.get('agent_position', new_state['agent'].position)
                    reward += 0.05 * (abs(ox-goal[0])+abs(oy-goal[1]) -
                                      abs(ax-goal[0])-abs(ay-goal[1]))
        if self.check_success(new_state):
            reward += 1.0
        return reward

    def check_success(self, state):
        if "grid" not in state or "agent" not in state:
            return False
        x, y = state["agent"].position
        return bool(state["grid"].objects[y, x] == ObjectType.GOAL)

    def validate_instance(self, grid, config):
        return True  # dynamic hazard passability

    def get_optimal_return(self, difficulty=None): return 1.0
    def get_random_baseline(self, difficulty=None): return 0.0
