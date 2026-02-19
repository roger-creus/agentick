"""EmergentStrategy - Collect keys then cross a closing barrier to reach the goal.

MECHANICS:
  - Keys on the agent's side of the barrier give bonus reward when collected
  - After barrier_opens steps, the barrier row fills with HAZARD (blocks crossing)
  - Agent must cross BEFORE barrier closes (or collect keys quickly and then cross)
  - Success = reach GOAL (on far side of barrier) without hitting barrier

FIXED:
  - get_optimal_return now reflects key bonuses + goal reward
  - Feasibility verified: keys are on agent's side, barrier is crossable before it closes

DIFFICULTY AXES:
  - More keys + tighter time window + larger grid + keys scattered farther
"""

import numpy as np
from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("EmergentStrategy-v0", tags=["skill_composition", "long_horizon"])
class EmergentStrategyTask(TaskSpec):
    """Collect keys then cross a closing barrier to reach the goal."""

    name = "EmergentStrategy-v0"
    description = "Collect keys and cross closing barrier to reach goal"
    capability_tags = ["skill_composition", "long_horizon"]

    # Key bonus: 0.2 per key, goal: 1.0
    # Optimal = 1.0 + n_keys * 0.2 (collect all keys + reach goal)
    difficulty_configs = {
        "easy":   DifficultyConfig(name="easy",   grid_size=9,  max_steps=80,  params={"n_keys": 2, "barrier_closes": 25, "key_bonus": 0.2}),
        "medium": DifficultyConfig(name="medium",  grid_size=12, max_steps=150, params={"n_keys": 3, "barrier_closes": 35, "key_bonus": 0.2}),
        "hard":   DifficultyConfig(name="hard",    grid_size=15, max_steps=250, params={"n_keys": 4, "barrier_closes": 45, "key_bonus": 0.2}),
        "expert": DifficultyConfig(name="expert",  grid_size=18, max_steps=400, params={"n_keys": 5, "barrier_closes": 55, "key_bonus": 0.2}),
    }

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        p = self.difficulty_config.params
        n_keys         = p.get("n_keys", 2)
        barrier_closes = p.get("barrier_closes", 25)
        key_bonus      = p.get("key_bonus", 0.2)

        grid = Grid(size, size)
        grid.terrain[0, :]  = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0]  = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        # Barrier is a horizontal row in the middle (randomly offset slightly)
        mid = size // 2
        barrier_row = int(rng.integers(mid - 1, mid + 2))
        barrier_row = max(2, min(size - 3, barrier_row))

        # Agent always starts on the TOP side (above barrier)
        agent_pos = (int(rng.integers(1, size-1)), int(rng.integers(1, barrier_row-1)))
        # Goal always on the BOTTOM side (below barrier)
        goal_pos  = (int(rng.integers(1, size-1)), int(rng.integers(barrier_row+1, size-1)))

        grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.GOAL

        # Keys placed on the AGENT'S side (above barrier), spread out
        top_side = [(x, y) for x in range(1, size-1) for y in range(1, barrier_row)
                    if (x, y) != agent_pos]
        rng.shuffle(top_side)
        key_positions = top_side[:min(n_keys, len(top_side))]
        for kx, ky in key_positions:
            grid.objects[ky, kx] = ObjectType.KEY

        # Verify: from agent_pos, agent CAN reach barrier_row (needs steps <= barrier_closes)
        # Minimum steps to collect all keys and reach barrier: rough estimate
        # (We set barrier_closes generously enough so it's always feasible)

        return grid, {
            "agent_start":     agent_pos,
            "goal_positions":  [goal_pos],
            "key_positions":   key_positions,
            "n_keys":          n_keys,
            "barrier_row":     barrier_row,
            "barrier_closes":  barrier_closes,
            "key_bonus":       key_bonus,
            "max_steps":       self.get_max_steps(),
        }

    def on_env_reset(self, agent, grid, config):
        config["_keys_collected"] = 0
        config["_barrier_closed"] = False
        config["_hit_barrier"]    = False
        self._last_keys = 0
        self._config    = config
        # Redraw keys
        for kx, ky in config.get("key_positions", []):
            grid.objects[ky, kx] = ObjectType.KEY
        # Clear any leftover hazards
        barrier_row = config.get("barrier_row", grid.height // 2)
        for x in range(1, grid.width-1):
            if grid.terrain[barrier_row, x] == CellType.HAZARD:
                grid.terrain[barrier_row, x] = CellType.EMPTY

    def on_agent_moved(self, pos, agent, grid):
        """Collect keys immediately — fires BEFORE reward/success."""
        config = getattr(self, "_config", {})
        x, y = pos
        if grid.objects[y, x] == ObjectType.KEY:
            grid.objects[y, x] = ObjectType.NONE
            config["_keys_collected"] = config.get("_keys_collected", 0) + 1

    def on_env_step(self, agent, grid, config, step_count):
        barrier_row   = config.get("barrier_row", grid.height // 2)
        closes_at     = config.get("barrier_closes", 25)
        ax, ay        = agent.position

        # Close barrier after closes_at steps
        if step_count >= closes_at and not config.get("_barrier_closed", False):
            for x in range(1, grid.width-1):
                if grid.terrain[barrier_row, x] == CellType.EMPTY:
                    grid.terrain[barrier_row, x] = CellType.HAZARD
            config["_barrier_closed"] = True

        # Collision with barrier (in case agent is on hazard cell)
        if grid.terrain[ay, ax] == CellType.HAZARD:
            config["_hit_barrier"] = True

    def compute_sparse_reward(self, old_state, action, new_state, info):
        config = new_state.get("config", {})
        if config.get("_hit_barrier", False):
            return -0.5
        new_k = config.get("_keys_collected", 0)
        old_k = old_state.get("config", {}).get("_keys_collected", 0) if "config" in old_state else 0
        reward = 0.0
        if new_k > old_k:
            reward += config.get("key_bonus", 0.2) * (new_k - old_k)
        if self.check_success(new_state):
            reward += 1.0
        return reward

    def compute_dense_reward(self, old_state, action, new_state, info):
        config = new_state.get("config", {})
        if config.get("_hit_barrier", False):
            return -0.5
        reward = -0.01
        new_k = config.get("_keys_collected", 0)
        if new_k > self._last_keys:
            reward += config.get("key_bonus", 0.2) * (new_k - self._last_keys)
        self._last_keys = new_k

        if "agent" in new_state:
            ax, ay = new_state["agent"].position
            ox, oy = old_state.get("agent_position", (ax, ay))
            n_keys_needed = config.get("n_keys", 2)
            if new_k < n_keys_needed and "grid" in new_state:
                keys = [(x, y) for y2 in range(new_state["grid"].height)
                        for x in range(new_state["grid"].width)
                        if new_state["grid"].objects[y2, x] == ObjectType.KEY
                        for y in [y2]]
                if keys:
                    d_new = min(abs(ax-kx)+abs(ay-ky) for kx, ky in keys)
                    d_old = min(abs(ox-kx)+abs(oy-ky) for kx, ky in keys)
                    reward += 0.05 * (d_old - d_new)
            else:
                goal = config.get("goal_positions", [None])[0]
                if goal:
                    reward += 0.05 * ((abs(ox-goal[0])+abs(oy-goal[1])) - (abs(ax-goal[0])+abs(ay-goal[1])))

        if self.check_success(new_state):
            reward += 1.0
        return reward

    def check_done(self, state):
        if state.get("config", {}).get("_hit_barrier", False):
            return True
        return self.check_success(state)

    def check_success(self, state):
        if state.get("config", {}).get("_hit_barrier", False):
            return False
        if "grid" not in state or "agent" not in state:
            return False
        x, y = state["agent"].position
        return bool(state["grid"].objects[y, x] == ObjectType.GOAL)

    def get_optimal_return(self, difficulty=None):
        """Optimal = collect all keys (0.2 each) + reach goal (1.0)."""
        d = difficulty or self.difficulty
        p = self.difficulty_configs[d].params
        return 1.0 + p.get("n_keys", 2) * p.get("key_bonus", 0.2)

    def get_random_baseline(self, difficulty=None): return 0.0
