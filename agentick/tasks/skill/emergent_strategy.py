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
        "easy": DifficultyConfig(
            name="easy",
            grid_size=9,
            max_steps=80,
            params={
                "n_keys": 2,
                "barrier_closes": 25,
                "key_bonus": 0.2,
                "n_barriers": 1,
                "barrier_warning": 0,
            },
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=12,
            max_steps=150,
            params={
                "n_keys": 3,
                "barrier_closes": 35,
                "key_bonus": 0.2,
                "n_barriers": 1,
                "barrier_warning": 0,
            },
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=15,
            max_steps=250,
            params={
                "n_keys": 4,
                "barrier_closes": 45,
                "key_bonus": 0.2,
                "n_barriers": 2,
                "barrier_warning": 3,
            },
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=18,
            max_steps=400,
            params={
                "n_keys": 5,
                "barrier_closes": 55,
                "key_bonus": 0.2,
                "n_barriers": 3,
                "barrier_warning": 5,
            },
        ),
    }

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        p = self.difficulty_config.params
        n_keys = p.get("n_keys", 2)
        barrier_closes = p.get("barrier_closes", 25)
        key_bonus = p.get("key_bonus", 0.2)
        n_barriers = p.get("n_barriers", 1)
        barrier_warning = p.get("barrier_warning", 0)

        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        interior = size - 2
        barrier_rows = []
        barrier_close_times = []
        for i in range(n_barriers):
            row = 1 + (i + 1) * interior // (n_barriers + 1)
            row = max(2, min(size - 3, row))
            if row in barrier_rows:
                row = min(size - 3, row + 1)
            barrier_rows.append(row)
            offset = i * 10
            barrier_close_times.append(barrier_closes + offset)

        first_barrier = barrier_rows[0]
        agent_pos = (
            int(rng.integers(1, size - 1)),
            int(rng.integers(1, max(2, first_barrier - 1))),
        )
        last_barrier = barrier_rows[-1]
        goal_pos = (
            int(rng.integers(1, size - 1)),
            int(rng.integers(min(last_barrier + 1, size - 2), size - 1)),
        )

        grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.GOAL

        top_side = [
            (x, y)
            for x in range(1, size - 1)
            for y in range(1, first_barrier)
            if (x, y) != agent_pos
        ]
        rng.shuffle(top_side)
        key_positions = top_side[: min(n_keys, len(top_side))]
        for kx, ky in key_positions:
            grid.objects[ky, kx] = ObjectType.KEY

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [goal_pos],
            "key_positions": key_positions,
            "n_keys": n_keys,
            "barrier_row": barrier_rows[0],
            "barrier_rows": barrier_rows,
            "barrier_close_times": barrier_close_times,
            "barrier_closes": barrier_closes,
            "barrier_warning": barrier_warning,
            "key_bonus": key_bonus,
            "max_steps": self.get_max_steps(),
        }

    def on_env_reset(self, agent, grid, config):
        config["_keys_collected"] = 0
        config["_barrier_closed"] = False
        config["_hit_barrier"] = False
        rows = config.get("barrier_rows", [config.get("barrier_row")])
        config["_barriers_closed"] = [False] * len(rows)
        config["_barriers_warned"] = [False] * len(rows)
        self._last_keys = 0
        self._config = config
        for kx, ky in config.get("key_positions", []):
            grid.objects[ky, kx] = ObjectType.KEY
        for row in rows:
            for x in range(1, grid.width - 1):
                if grid.terrain[row, x] == CellType.HAZARD:
                    grid.terrain[row, x] = CellType.EMPTY

    def on_agent_moved(self, pos, agent, grid):
        """Collect keys immediately — fires BEFORE reward/success."""
        config = getattr(self, "_config", {})
        x, y = pos
        if grid.objects[y, x] == ObjectType.KEY:
            grid.objects[y, x] = ObjectType.NONE
            config["_keys_collected"] = config.get("_keys_collected", 0) + 1

    def on_env_step(self, agent, grid, config, step_count):
        # If agent already reached goal, don't process barrier logic
        x_a, y_a = agent.position
        if grid.objects[y_a, x_a] == ObjectType.GOAL:
            return

        rows = config.get(
            "barrier_rows",
            [config.get("barrier_row", grid.height // 2)],
        )
        close_times = config.get(
            "barrier_close_times",
            [config.get("barrier_closes", 25)],
        )
        warning = config.get("barrier_warning", 0)
        closed_flags = config.get("_barriers_closed", [False] * len(rows))
        warned_flags = config.get("_barriers_warned", [False] * len(rows))

        for idx, (row, closes_at) in enumerate(zip(rows, close_times)):
            if warning > 0 and not warned_flags[idx]:
                if step_count >= closes_at - warning:
                    for x in range(1, grid.width - 1):
                        if grid.terrain[row, x] == CellType.EMPTY:
                            grid.terrain[row, x] = CellType.WATER
                    warned_flags[idx] = True

            if step_count >= closes_at and not closed_flags[idx]:
                for x in range(1, grid.width - 1):
                    t = grid.terrain[row, x]
                    if t in (CellType.EMPTY, CellType.WATER):
                        grid.terrain[row, x] = CellType.HAZARD
                closed_flags[idx] = True

        config["_barriers_closed"] = closed_flags
        config["_barriers_warned"] = warned_flags
        if any(closed_flags):
            config["_barrier_closed"] = True

        if grid.terrain[y_a, x_a] == CellType.HAZARD:
            config["_hit_barrier"] = True

    def compute_sparse_reward(self, old_state, action, new_state, info):
        config = new_state.get("config", {})
        if config.get("_hit_barrier", False):
            return -0.5
        new_k = config.get("_keys_collected", 0)
        old_k = (
            old_state.get("config", {}).get("_keys_collected", 0) if "config" in old_state else 0
        )
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
                keys = [
                    (x, y)
                    for y2 in range(new_state["grid"].height)
                    for x in range(new_state["grid"].width)
                    if new_state["grid"].objects[y2, x] == ObjectType.KEY
                    for y in [y2]
                ]
                if keys:
                    d_new = min(abs(ax - kx) + abs(ay - ky) for kx, ky in keys)
                    d_old = min(abs(ox - kx) + abs(oy - ky) for kx, ky in keys)
                    reward += 0.05 * (d_old - d_new)
            else:
                goal = config.get("goal_positions", [None])[0]
                if goal:
                    reward += 0.05 * (
                        (abs(ox - goal[0]) + abs(oy - goal[1]))
                        - (abs(ax - goal[0]) + abs(ay - goal[1]))
                    )

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

    def get_random_baseline(self, difficulty=None):
        return 0.0
