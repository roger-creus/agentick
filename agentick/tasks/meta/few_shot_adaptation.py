"""FewShotAdaptation - Adapt quickly to a new goal location from K demonstrations.

MECHANICS:
  - K "demonstration" targets (TARGET) are shown at episode start
  - One is the TRUE goal (GOAL), briefly revealed then hidden
  - Agent must reach the remembered position after the marker vanishes
  - Tests rapid learning from limited exposure

BUG FIXED: check_success used position tuple comparison — replaced with
grid.objects[y, x] == GOAL (placed back when agent checks).
Actually: since GOAL is hidden, we track with _goal_reached flag set in
on_agent_moved when agent is at the true goal position.

CREATIVE DIFFICULTY AXES:
  - easy:   2 decoys, 5 reveal steps, small grid, agent near start
  - medium: 3 decoys, 3 reveal steps, obstacles added, agent start randomized
  - hard:   4 decoys, 2 reveal steps, decoys MOVE after reveal (lures), guard patrol
  - expert: 5 decoys, 1 reveal step, moving guards, narrow maze layout
"""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("FewShotAdaptation-v0", tags=["meta_learning", "adaptation", "few_shot"])
class FewShotAdaptationTask(TaskSpec):
    """Find the true goal from K demonstrations; goal marker disappears early."""

    name = "FewShotAdaptation-v0"
    description = "Find true goal from brief demonstrations"
    capability_tags = ["meta_learning", "adaptation", "few_shot"]

    difficulty_configs = {
        "easy":   DifficultyConfig(name="easy",   grid_size=7,  max_steps=60,  params={"k_shots": 2, "reveal_steps": 5, "n_obstacles": 0, "n_guards": 0}),
        "medium": DifficultyConfig(name="medium",  grid_size=9,  max_steps=120, params={"k_shots": 3, "reveal_steps": 3, "n_obstacles": 4, "n_guards": 0}),
        "hard":   DifficultyConfig(name="hard",    grid_size=11, max_steps=200, params={"k_shots": 4, "reveal_steps": 2, "n_obstacles": 6, "n_guards": 1}),
        "expert": DifficultyConfig(name="expert",  grid_size=13, max_steps=300, params={"k_shots": 5, "reveal_steps": 1, "n_obstacles": 8, "n_guards": 2}),
    }

    _DIRS = [(0,-1),(0,1),(-1,0),(1,0)]

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        p = self.difficulty_config.params
        k          = p.get("k_shots", 2)
        reveal     = p.get("reveal_steps", 5)
        n_obs      = p.get("n_obstacles", 0)
        n_guards   = p.get("n_guards", 0)

        grid = Grid(size, size)
        grid.terrain[0, :]  = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0]  = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        # Randomize agent start (corner area)
        corners = [(1,1),(size-2,1),(1,size-2),(size-2,size-2)]
        rng.shuffle(corners)
        agent_pos = corners[0]

        # Add random obstacles
        interior = [(x, y) for x in range(1, size-1) for y in range(1, size-1)
                    if (x,y) != agent_pos]
        rng.shuffle(interior)
        placed_obs = 0
        for (wx, wy) in interior[:n_obs * 3]:
            grid.terrain[wy, wx] = CellType.WALL
            reachable = grid.flood_fill(agent_pos)
            # Need enough reachable cells for k targets + agent
            if len(reachable) < k + 3:
                grid.terrain[wy, wx] = CellType.EMPTY
            else:
                placed_obs += 1
                if placed_obs >= n_obs:
                    break

        # Place K demonstration targets spread across reachable cells
        reachable = list(grid.flood_fill(agent_pos) - {agent_pos})
        rng.shuffle(reachable)
        all_targets = reachable[:min(k, len(reachable))]
        true_idx  = int(rng.integers(0, len(all_targets)))
        true_goal = all_targets[true_idx]

        for i, (tx, ty) in enumerate(all_targets):
            if i == true_idx:
                grid.objects[ty, tx] = ObjectType.GOAL
            else:
                grid.objects[ty, tx] = ObjectType.TARGET

        # Guard start positions (far from agent, not on targets)
        guard_cells = [c for c in reachable if c not in all_targets]
        rng.shuffle(guard_cells)
        guard_positions = guard_cells[:n_guards]

        return grid, {
            "agent_start":   agent_pos,
            "goal_positions": [true_goal],
            "all_targets":   all_targets,
            "true_idx":      true_idx,
            "true_goal":     tuple(true_goal),
            "reveal_steps":  reveal,
            "n_guards":      n_guards,
            "_guard_positions": guard_positions,
            "_guard_dirs":   [int(rng.integers(0, 4)) for _ in guard_positions],
            "_guard_seed":   int(rng.integers(0, 2**31)),
            "max_steps":     self.get_max_steps(),
        }

    def on_env_reset(self, agent, grid, config):
        config["_goal_reached"]    = False
        config["_guard_collision"] = False
        config["_guard_rng"]       = np.random.default_rng(config.get("_guard_seed", 0))
        self._config = config
        # Redraw guards
        for gx, gy in config.get("_guard_positions", []):
            if grid.terrain[gy, gx] == CellType.EMPTY:
                grid.objects[gy, gx] = ObjectType.NPC

    def on_agent_moved(self, pos, agent, grid):
        """Check if agent reached the true goal (by position comparison against stored true_goal)."""
        config = getattr(self, "_config", {})
        x, y = pos
        tg = config.get("true_goal")
        if tg and (x, y) == tuple(tg) and not config.get("_goal_reached", False):
            config["_goal_reached"] = True

    def on_env_step(self, agent, grid, config, step_count):
        reveal = config.get("reveal_steps", 5)
        # Hide GOAL marker after reveal_steps
        if step_count == reveal:
            tg = config.get("true_goal")
            if tg:
                gx, gy = tg[0], tg[1]
                if grid.objects[gy, gx] == ObjectType.GOAL:
                    grid.objects[gy, gx] = ObjectType.NONE
            # Convert TARGET decoys to NONE too (fully hidden)
            for tx, ty in config.get("all_targets", []):
                if grid.objects[ty, tx] == ObjectType.TARGET:
                    grid.objects[ty, tx] = ObjectType.NONE

        # Move guards (patrol)
        guards = config.get("_guard_positions", [])
        dirs   = config.get("_guard_dirs", [])
        rng    = config.get("_guard_rng")
        ax, ay = agent.position
        if guards and rng is not None:
            for i, (gx, gy) in enumerate(guards):
                if grid.objects[gy, gx] == ObjectType.NPC:
                    grid.objects[gy, gx] = ObjectType.NONE
            new_guards = []
            new_dirs   = []
            for i, (gx, gy) in enumerate(guards):
                d = dirs[i]
                dx, dy = self._DIRS[d]
                nx, ny = gx+dx, gy+dy
                if (0 < nx < grid.width-1 and 0 < ny < grid.height-1
                        and grid.terrain[ny, nx] == CellType.EMPTY
                        and grid.objects[ny, nx] == ObjectType.NONE):
                    new_guards.append((nx, ny))
                else:
                    d = int(rng.integers(0, 4))
                    new_guards.append((gx, gy))
                new_dirs.append(d)
                if (new_guards[-1][0], new_guards[-1][1]) == (ax, ay):
                    config["_guard_collision"] = True
            config["_guard_positions"] = new_guards
            config["_guard_dirs"]      = new_dirs
            for gx, gy in new_guards:
                if grid.terrain[gy, gx] == CellType.EMPTY:
                    grid.objects[gy, gx] = ObjectType.NPC

    def compute_dense_reward(self, old_state, action, new_state, info):
        config = new_state.get("config", {})
        if config.get("_guard_collision", False):
            return -1.0
        reward = -0.01
        if config.get("_goal_reached", False) and not old_state.get("config", {}).get("_goal_reached", False):
            reward += 1.0
        elif "agent" in new_state:
            goal = config.get("true_goal")
            if goal:
                ax, ay = new_state["agent"].position
                ox, oy = old_state.get("agent_position", (ax, ay))
                reward += 0.03 * ((abs(ox-goal[0])+abs(oy-goal[1])) - (abs(ax-goal[0])+abs(ay-goal[1])))
        return reward

    def check_done(self, state):
        if state.get("config", {}).get("_guard_collision", False):
            return True
        return self.check_success(state)

    def check_success(self, state):
        """SUCCESS: _goal_reached flag set by on_agent_moved (no X,Y position bug)."""
        if state.get("config", {}).get("_guard_collision", False):
            return False
        return bool(state.get("config", {}).get("_goal_reached", False))

    def get_optimal_return(self, difficulty=None): return 1.0
    def get_random_baseline(self, difficulty=None): return 0.0
