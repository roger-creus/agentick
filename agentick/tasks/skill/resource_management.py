"""ResourceManagement - Collect resources and spend them to unlock paths.

MECHANICS:
  - RESOURCE items (KEY objects) scattered on grid
  - Locked doors (DOOR) require N resources to pass through
  - Agent must plan: collect enough resources, then spend to traverse doors
  - Auto-collect resources, auto-spend at doors (if enough in inventory)
  - Success = reach GOAL
"""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task

_RM_DIRS = [(0,-1),(0,1),(-1,0),(1,0)]


@register_task("ResourceManagement-v0", tags=["planning", "resource_allocation"])
class ResourceManagementTask(TaskSpec):
    """Collect resources and spend them to unlock doors; reach the goal."""

    name = "ResourceManagement-v0"
    description = "Collect resources to unlock doors, reach goal"
    capability_tags = ["planning", "resource_allocation"]

    difficulty_configs = {
        "easy":   DifficultyConfig(name="easy",   grid_size=9,  max_steps=100, params={"n_resources": 2, "door_cost": 1, "n_traps": 0, "n_guards": 0}),
        "medium": DifficultyConfig(name="medium",  grid_size=12, max_steps=180, params={"n_resources": 4, "door_cost": 2, "n_traps": 2, "n_guards": 0}),
        "hard":   DifficultyConfig(name="hard",    grid_size=15, max_steps=300, params={"n_resources": 6, "door_cost": 3, "n_traps": 4, "n_guards": 1}),
        "expert": DifficultyConfig(name="expert",  grid_size=18, max_steps=500, params={"n_resources": 8, "door_cost": 4, "n_traps": 6, "n_guards": 2}),
    }

    _DIRS = _RM_DIRS

    def generate(self, seed):
        rng    = np.random.default_rng(seed)
        size   = self.difficulty_config.grid_size
        n_res  = self.difficulty_config.params.get("n_resources", 2)
        cost   = self.difficulty_config.params.get("door_cost", 1)
        n_traps   = self.difficulty_config.params.get("n_traps", 0)
        n_guards  = self.difficulty_config.params.get("n_guards", 0)

        grid = Grid(size, size)
        grid.terrain[0, :]  = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0]  = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        mid = size // 2
        for y in range(1, size-1):
            grid.terrain[y, mid] = CellType.WALL
        door_y = size // 2
        grid.terrain[door_y, mid] = CellType.EMPTY

        agent_pos = (1, 1)
        goal_pos  = (size-2, size-2)
        grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.GOAL

        left_free = [(x, y) for x in range(1, mid) for y in range(1, size-1)
                     if (x, y) != agent_pos]
        rng.shuffle(left_free)
        resource_positions = left_free[:n_res]
        used = {agent_pos, goal_pos, (mid, door_y)} | set(resource_positions)
        for rx, ry in resource_positions:
            grid.objects[ry, rx] = ObjectType.KEY

        grid.objects[door_y, mid] = ObjectType.SWITCH

        # Traps: HAZARD terrain on left side (avoid resources and agent start)
        trap_positions = []
        trap_candidates = [p for p in left_free[n_res:] if p not in used]
        for p in trap_candidates:
            if len(trap_positions) >= n_traps:
                break
            tx, ty = p
            grid.terrain[ty, tx] = CellType.HAZARD
            reachable = grid.flood_fill(agent_pos)
            if all(q in reachable for q in resource_positions):
                trap_positions.append(p)
                used.add(p)
            else:
                grid.terrain[ty, tx] = CellType.EMPTY

        # Guards: NPC on left side, distant from agent
        right_free = [(x, y) for x in range(mid+1, size-1) for y in range(1, size-1)
                      if (x, y) != goal_pos]
        rng.shuffle(right_free)
        guard_candidates = [p for p in right_free if p not in used
                            and abs(p[0]-agent_pos[0])+abs(p[1]-agent_pos[1]) > 2]
        guard_positions = guard_candidates[:n_guards]
        for gx, gy in guard_positions:
            grid.objects[gy, gx] = ObjectType.NPC
            used.add((gx, gy))

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [goal_pos],
            "resource_positions": resource_positions,
            "door_pos": (mid, door_y),
            "door_cost": cost,
            "trap_positions": trap_positions,
            "_guard_positions": guard_positions,
            "_guard_dirs": [int(rng.integers(0, 4)) for _ in guard_positions],
            "_guard_seed": int(rng.integers(0, 2**31)),
            "max_steps": self.get_max_steps(),
        }

    def on_env_reset(self, agent, grid, config):
        config["_resources_held"] = 0
        config["_guard_collision"] = False
        config["_guard_rng"] = np.random.default_rng(config.get("_guard_seed", 0))
        self._last_resources_held = 0
        self._config = config

    def on_env_step(self, agent, grid, config, step_count):
        ax, ay = agent.position
        # Collect resource
        if grid.objects[ay, ax] == ObjectType.KEY:
            grid.objects[ay, ax] = ObjectType.NONE
            config["_resources_held"] = config.get("_resources_held", 0) + 1
        # Move guards and check collision
        guards = config.get("_guard_positions", [])
        dirs   = config.get("_guard_dirs", [])
        rng    = config.get("_guard_rng")
        if not guards or rng is None:
            return
        for gx, gy in guards:
            if grid.objects[gy, gx] == ObjectType.NPC:
                grid.objects[gy, gx] = ObjectType.NONE
        new_g, new_d = [], []
        for i, (gx, gy) in enumerate(guards):
            d = dirs[i]; ddx, ddy = self._DIRS[d]; nx, ny = gx+ddx, gy+ddy
            if (0 < nx < grid.width-1 and 0 < ny < grid.height-1
                    and grid.terrain[ny, nx] == CellType.EMPTY
                    and grid.objects[ny, nx] not in (ObjectType.GOAL, ObjectType.SWITCH)):
                new_g.append((nx, ny))
            else:
                d = int(rng.integers(0, 4)); new_g.append((gx, gy))
            new_d.append(d)
            if new_g[-1] == (ax, ay):
                config["_guard_collision"] = True
        config["_guard_positions"] = new_g
        config["_guard_dirs"] = new_d
        for gx, gy in new_g:
            if grid.terrain[gy, gx] == CellType.EMPTY:
                grid.objects[gy, gx] = ObjectType.NPC

    def can_agent_enter(self, pos, agent, grid):
        """Agent can enter door gap only if they have enough resources."""
        x, y = pos
        if grid.objects[y, x] == ObjectType.SWITCH:
            config = getattr(self, "_config", {})
            held = config.get("_resources_held", 0)
            cost = config.get("door_cost", 1)
            return held >= cost
        return True

    def on_agent_moved(self, pos, agent, grid):
        x, y = pos
        config = getattr(self, "_config", {})
        # Spend resources when entering the door cell
        if grid.objects[y, x] == ObjectType.SWITCH:
            cost = config.get("door_cost", 1)
            config["_resources_held"] = config.get("_resources_held", 0) - cost
            grid.objects[y, x] = ObjectType.NONE  # door unlocked
        if grid.objects[y, x] == ObjectType.NPC:
            config["_guard_collision"] = True

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        config = new_state.get("config", {})
        new_r = config.get("_resources_held", 0)
        cost = config.get("door_cost", 1)
        if new_r > self._last_resources_held:
            reward += 0.2 * (new_r - self._last_resources_held)
        self._last_resources_held = new_r
        # Approach shaping: toward resources (if need more) → door (if unlocked) → goal
        if "agent_position" in new_state and "grid" in new_state:
            ax, ay = new_state["agent_position"]
            ox, oy = old_state.get("agent_position", (ax, ay))
            g = new_state["grid"]
            if new_r < cost:
                # Guide toward nearest resource (KEY object)
                from agentick.core.types import ObjectType as OT
                resources = [(x, y) for y in range(g.height) for x in range(g.width)
                             if g.objects[y, x] == OT.KEY]
                if resources:
                    d_new = min(abs(ax-rx)+abs(ay-ry) for rx,ry in resources)
                    d_old = min(abs(ox-rx)+abs(oy-ry) for rx,ry in resources)
                    reward += 0.05 * (d_old - d_new)
            else:
                # Have enough resources — guide toward goal
                goal = config.get("goal_positions", [None])[0]
                if goal:
                    old_d = abs(ox-goal[0]) + abs(oy-goal[1])
                    new_d = abs(ax-goal[0]) + abs(ay-goal[1])
                    reward += 0.05 * (old_d - new_d)
        if self.check_success(new_state):
            reward += 1.0
        return reward

    def check_success(self, state):
        config = state.get("config", {})
        if config.get("_guard_collision", False):
            return False
        if "grid" not in state or "agent" not in state:
            return False
        x, y = state["agent"].position
        return bool(state["grid"].objects[y, x] == ObjectType.GOAL)

    def check_done(self, state):
        config = state.get("config", {})
        if config.get("_guard_collision", False):
            return True
        return self.check_success(state)

    def validate_instance(self, grid, config):
        return True

    def get_optimal_return(self, difficulty=None): return 1.0
    def get_random_baseline(self, difficulty=None): return 0.0
