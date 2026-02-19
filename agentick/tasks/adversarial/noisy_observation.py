"""NoisyObservation - Navigate to the true goal among decoy targets.

MECHANICS:
  - One TRUE goal (G) placed at a random position
  - N decoy targets (T) placed at random positions — look similar to goal
  - Random wall obstacles scattered throughout
  - Agent must find and reach the TRUE goal, ignoring decoys
  - Decoys give 0 reward; only the true goal gives +1
  - Tests robustness to distracting/misleading observations
"""

import numpy as np
from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task

_NO_DIRS = [(0,-1),(0,1),(-1,0),(1,0)]


@register_task("NoisyObservation-v0", tags=["robustness", "navigation", "noise"])
class NoisyObservationTask(TaskSpec):
    """Navigate to the true goal; ignore decoy targets."""

    name = "NoisyObservation-v0"
    description = "Find true goal among noisy decoy targets"
    capability_tags = ["robustness", "navigation", "noise"]

    difficulty_configs = {
        "easy":   DifficultyConfig(name="easy",   grid_size=7,  max_steps=100, params={"n_decoys": 2, "n_obstacles": 3, "n_guards": 0}),
        "medium": DifficultyConfig(name="medium",  grid_size=9,  max_steps=150, params={"n_decoys": 4, "n_obstacles": 5, "n_guards": 1}),
        "hard":   DifficultyConfig(name="hard",    grid_size=11, max_steps=220, params={"n_decoys": 6, "n_obstacles": 8, "n_guards": 2}),
        "expert": DifficultyConfig(name="expert",  grid_size=13, max_steps=320, params={"n_decoys": 8, "n_obstacles": 12, "n_guards": 3}),
    }

    _DIRS = _NO_DIRS

    def generate(self, seed):
        rng      = np.random.default_rng(seed)
        size     = self.difficulty_config.grid_size
        n_decoys = self.difficulty_config.params.get("n_decoys", 2)
        n_obs    = self.difficulty_config.params.get("n_obstacles", 3)
        n_guards = self.difficulty_config.params.get("n_guards", 0)

        grid = Grid(size, size)
        grid.terrain[0, :]  = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0]  = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        agent_pos = (1, 1)
        interior = [(x, y) for x in range(1, size-1) for y in range(1, size-1)
                    if (x, y) != agent_pos]
        rng.shuffle(interior)

        non_corner = [(x, y) for (x, y) in interior
                      if not (x == size-2 and y == size-2)]
        goal_pos = non_corner[0]
        used = {goal_pos, agent_pos}

        decoys = []
        for p in non_corner[1:]:
            if p not in used and len(decoys) < n_decoys:
                decoys.append(p)
                used.add(p)

        walls = []
        candidates = [p for p in interior if p not in used]
        for p in candidates:
            if len(walls) >= n_obs:
                break
            wx, wy = p
            grid.terrain[wy, wx] = CellType.WALL
            reachable = grid.flood_fill(agent_pos)
            if goal_pos not in reachable:
                grid.terrain[wy, wx] = CellType.EMPTY
            else:
                walls.append(p)
                used.add(p)

        for (dx2, dy2) in decoys:
            grid.objects[dy2, dx2] = ObjectType.TARGET
        grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.GOAL

        # Guards: NPC objects on reachable cells, distant from agent
        reachable = grid.flood_fill(agent_pos)
        guard_candidates = [p for p in reachable if p not in used
                            and p != goal_pos
                            and abs(p[0]-agent_pos[0])+abs(p[1]-agent_pos[1]) > 2]
        rng.shuffle(guard_candidates)
        guard_positions = guard_candidates[:n_guards]
        for gx, gy in guard_positions:
            grid.objects[gy, gx] = ObjectType.NPC
            used.add((gx, gy))

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [goal_pos],
            "decoy_positions": decoys,
            "_guard_positions": guard_positions,
            "_guard_dirs": [int(rng.integers(0, 4)) for _ in guard_positions],
            "_guard_seed": int(rng.integers(0, 2**31)),
            "max_steps": self.get_max_steps(),
        }

    def on_env_reset(self, agent, grid, config):
        config["_guard_collision"] = False
        config["_guard_rng"] = np.random.default_rng(config.get("_guard_seed", 0))
        self._config = config  # cache for on_agent_moved

    def on_agent_moved(self, pos, agent, grid):
        x, y = pos
        config = getattr(self, "_config", {})
        if grid.objects[y, x] == ObjectType.NPC:
            config["_guard_collision"] = True

    def on_env_step(self, agent, grid, config, step_count):
        guards = config.get("_guard_positions", [])
        dirs   = config.get("_guard_dirs", [])
        rng    = config.get("_guard_rng")
        ax, ay = agent.position
        if not guards or rng is None:
            return
        for gx, gy in guards:
            if grid.objects[gy, gx] == ObjectType.NPC:
                grid.objects[gy, gx] = ObjectType.NONE
        new_g, new_d = [], []
        for i, (gx, gy) in enumerate(guards):
            d = dirs[i]; dx, dy = self._DIRS[d]; nx, ny = gx+dx, gy+dy
            if (0 < nx < grid.width-1 and 0 < ny < grid.height-1
                    and grid.terrain[ny, nx] == CellType.EMPTY
                    and grid.objects[ny, nx] not in (ObjectType.GOAL, ObjectType.TARGET)):
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

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        config = new_state.get("config", {})
        goal = config.get("goal_positions", [None])[0]
        if goal and "agent" in new_state:
            ax, ay = new_state["agent"].position
            ox, oy = old_state.get('agent_position', new_state['agent'].position)
            reward += 0.05 * (abs(ox-goal[0]) + abs(oy-goal[1])
                              - abs(ax-goal[0]) - abs(ay-goal[1]))
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

    def get_optimal_return(self, difficulty=None): return 1.0
    def get_random_baseline(self, difficulty=None): return 0.0
