"""EmergentStrategy - Exploit NPC flee behavior to clear a path through a barrier.

MECHANICS:
  - A horizontal barrier (WALL row with gaps) separates the agent from the GOAL
  - SHEEP NPCs block the barrier gaps — the agent cannot walk through a gap
    occupied by a SHEEP
  - SHEEP flee from the agent when the agent is within Manhattan distance 2,
    moving to the adjacent cell that maximises distance from the agent
  - The agent must position themselves to scare the SHEEP away from the gap,
    then rush through before the barrier fills with HAZARD
  - Keys on the agent's side give bonus reward when collected
  - After barrier_closes steps the open gaps fill with HAZARD (permanently blocked)
  - Success = reach GOAL on the far side of the barrier

EMERGENT STRATEGY:
  The agent must *exploit* the SHEEP's flee AI to clear the barrier gap.
  Approaching from the correct side causes the SHEEP to run away from the gap,
  opening a path.  This is the "emergent strategy" the task tests.

DIFFICULTY AXES:
  - More keys + tighter time window + larger grid
  - More gaps blocked by more sheep (harder difficulties)
  - Barrier warning (WATER visual cue) at hard/expert
"""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("EmergentStrategy-v0", tags=["skill_composition", "long_horizon"])
class EmergentStrategyTask(TaskSpec):
    """Exploit NPC SHEEP flee behavior to clear barrier gaps and reach the goal."""

    name = "EmergentStrategy-v0"
    description = "Scare sheep away from barrier gaps, collect keys, and reach goal"
    capability_tags = ["skill_composition", "long_horizon"]

    # Key bonus: 0.2 per key, goal: 1.0
    # Optimal = 1.0 + n_keys * 0.2 (collect all keys + reach goal)
    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=9,
            max_steps=100,
            params={
                "n_keys": 1,
                "barrier_closes": 40,
                "key_bonus": 0.2,
                "n_gaps": 1,
                "n_sheep": 1,
                "barrier_warning": 0,
            },
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=12,
            max_steps=180,
            params={
                "n_keys": 2,
                "barrier_closes": 50,
                "key_bonus": 0.2,
                "n_gaps": 1,
                "n_sheep": 1,
                "barrier_warning": 0,
            },
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=15,
            max_steps=300,
            params={
                "n_keys": 3,
                "barrier_closes": 60,
                "key_bonus": 0.2,
                "n_gaps": 2,
                "n_sheep": 2,
                "barrier_warning": 5,
            },
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=18,
            max_steps=450,
            params={
                "n_keys": 4,
                "barrier_closes": 70,
                "key_bonus": 0.2,
                "n_gaps": 3,
                "n_sheep": 3,
                "barrier_warning": 8,
            },
        ),
    }

    _DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        p = self.difficulty_config.params
        n_keys = p.get("n_keys", 1)
        barrier_closes = p.get("barrier_closes", 40)
        key_bonus = p.get("key_bonus", 0.2)
        n_gaps = p.get("n_gaps", 1)
        n_sheep = p.get("n_sheep", 1)
        barrier_warning = p.get("barrier_warning", 0)

        grid = Grid(size, size)
        # Outer walls
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        # Barrier row in the middle third of the grid
        barrier_row = size // 2
        # Fill entire barrier row with WALL
        for x in range(1, size - 1):
            grid.terrain[barrier_row, x] = CellType.WALL

        # Punch gaps in the barrier — these are the only crossings
        interior_xs = list(range(2, size - 2))
        rng.shuffle(interior_xs)
        actual_n_gaps = min(n_gaps, len(interior_xs))
        gap_xs = sorted(interior_xs[:actual_n_gaps])

        for gx in gap_xs:
            grid.terrain[barrier_row, gx] = CellType.EMPTY

        # Agent starts on the top side (above barrier)
        agent_y = int(rng.integers(1, barrier_row))
        agent_x = int(rng.integers(1, size - 1))
        agent_pos = (agent_x, agent_y)

        # Goal on the bottom side (below barrier), always visible
        goal_y = int(rng.integers(barrier_row + 1, size - 1))
        goal_x = int(rng.integers(1, size - 1))
        goal_pos = (goal_x, goal_y)
        grid.objects[goal_y, goal_x] = ObjectType.GOAL

        # Place one SHEEP in each gap (blocking it)
        # If n_sheep > n_gaps, extra sheep go on the top side near the barrier
        sheep_positions = []
        for i in range(min(n_sheep, actual_n_gaps)):
            sx = gap_xs[i]
            sy = barrier_row  # sheep sits IN the gap
            sheep_positions.append((sx, sy))
            grid.objects[sy, sx] = ObjectType.SHEEP

        # Extra sheep placed one row above random gaps (additional blockers)
        for i in range(actual_n_gaps, n_sheep):
            gx = gap_xs[i % actual_n_gaps]
            sy = barrier_row - 1
            if (
                grid.terrain[sy, gx] == CellType.EMPTY
                and grid.objects[sy, gx] == ObjectType.NONE
                and (gx, sy) != agent_pos
            ):
                sheep_positions.append((gx, sy))
                grid.objects[sy, gx] = ObjectType.SHEEP

        # Place keys on the agent's side (above barrier)
        top_cells = [
            (x, y)
            for x in range(1, size - 1)
            for y in range(1, barrier_row)
            if (x, y) != agent_pos and grid.objects[y, x] == ObjectType.NONE
        ]
        rng.shuffle(top_cells)
        key_positions = top_cells[: min(n_keys, len(top_cells))]
        for kx, ky in key_positions:
            grid.objects[ky, kx] = ObjectType.KEY

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [goal_pos],
            "key_positions": key_positions,
            "n_keys": n_keys,
            "barrier_row": barrier_row,
            "gap_xs": gap_xs,
            "barrier_closes": barrier_closes,
            "barrier_warning": barrier_warning,
            "key_bonus": key_bonus,
            "sheep_positions": sheep_positions,
            "max_steps": self.get_max_steps(),
            "_rng_seed": int(rng.integers(0, 2**31)),
        }

    # ── Reset / step hooks ─────────────────────────────────────────────────

    def on_env_reset(self, agent, grid, config):
        config["_keys_collected"] = 0
        config["_barrier_closed"] = False
        config["_hit_barrier"] = False
        config["_barrier_warned"] = False
        config["_live_sheep"] = list(config.get("sheep_positions", []))
        config["_sheep_rng"] = np.random.default_rng(config.get("_rng_seed", 0))
        self._last_keys = 0
        self._config = config

        # Ensure keys are drawn
        for kx, ky in config.get("key_positions", []):
            grid.objects[ky, kx] = ObjectType.KEY

        # Ensure barrier row is correct (WALL with gaps)
        barrier_row = config["barrier_row"]
        gap_xs = set(config["gap_xs"])
        for x in range(1, grid.width - 1):
            if x in gap_xs:
                grid.terrain[barrier_row, x] = CellType.EMPTY
            else:
                grid.terrain[barrier_row, x] = CellType.WALL

        # Draw sheep
        for sx, sy in config["_live_sheep"]:
            grid.objects[sy, sx] = ObjectType.SHEEP

    def on_agent_moved(self, pos, agent, grid):
        """Collect keys immediately — fires BEFORE reward/success."""
        config = getattr(self, "_config", {})
        x, y = pos
        if grid.objects[y, x] == ObjectType.KEY:
            grid.objects[y, x] = ObjectType.NONE
            config["_keys_collected"] = config.get("_keys_collected", 0) + 1

    def can_agent_enter(self, pos, agent, grid):
        """SHEEP block the agent — cannot walk into a cell occupied by SHEEP."""
        x, y = pos
        if grid.objects[y, x] == ObjectType.SHEEP:
            return False
        return True

    def on_env_step(self, agent, grid, config, step_count):
        # If agent already reached goal, skip all logic
        x_a, y_a = agent.position
        if grid.objects[y_a, x_a] == ObjectType.GOAL:
            return

        # ── Move SHEEP (flee from agent) ──────────────────────────────────
        sheep = config.get("_live_sheep", [])
        rng = config.get("_sheep_rng", np.random.default_rng(0))
        ax, ay = agent.position
        barrier_row = config["barrier_row"]

        # Erase old sheep positions
        for sx, sy in sheep:
            if 0 <= sx < grid.width and 0 <= sy < grid.height:
                if grid.objects[sy, sx] == ObjectType.SHEEP:
                    grid.objects[sy, sx] = ObjectType.NONE

        occupied = {(sx, sy) for sx, sy in sheep}
        new_sheep = []
        for sx, sy in sheep:
            occupied.discard((sx, sy))
            dist_agent = abs(sx - ax) + abs(sy - ay)

            if dist_agent <= 2:
                # Flee: pick adjacent cell that maximises distance from agent
                best = (sx, sy)
                best_d = dist_agent
                candidates = [(sx + dx, sy + dy) for dx, dy in self._DIRS]
                rng.shuffle(candidates)
                for nx, ny in candidates:
                    if not (0 < nx < grid.width - 1 and 0 < ny < grid.height - 1):
                        continue
                    # Sheep can walk on EMPTY terrain only (not WALL, HAZARD, etc.)
                    if grid.terrain[ny, nx] not in (CellType.EMPTY, CellType.WATER):
                        continue
                    # Don't collide with other sheep or the agent
                    if (nx, ny) in occupied or (nx, ny) == (ax, ay):
                        continue
                    # Don't walk onto keys or goal
                    if grid.objects[ny, nx] in (ObjectType.KEY, ObjectType.GOAL):
                        continue
                    d = abs(nx - ax) + abs(ny - ay)
                    if d > best_d:
                        best_d = d
                        best = (nx, ny)
                new_sheep.append(best)
            else:
                # Idle: small random movement
                if rng.random() < 0.15:
                    candidates = [(sx + dx, sy + dy) for dx, dy in self._DIRS]
                    valid = [
                        (nx, ny)
                        for nx, ny in candidates
                        if (
                            0 < nx < grid.width - 1
                            and 0 < ny < grid.height - 1
                            and grid.terrain[ny, nx] in (CellType.EMPTY, CellType.WATER)
                            and (nx, ny) not in occupied
                            and (nx, ny) != (ax, ay)
                            and grid.objects[ny, nx]
                            not in (ObjectType.KEY, ObjectType.GOAL)
                        )
                    ]
                    if valid:
                        new_sheep.append(valid[int(rng.integers(len(valid)))])
                    else:
                        new_sheep.append((sx, sy))
                else:
                    new_sheep.append((sx, sy))

            occupied.add(new_sheep[-1])

        config["_live_sheep"] = new_sheep

        # Draw new sheep positions
        for sx, sy in new_sheep:
            if 0 <= sx < grid.width and 0 <= sy < grid.height:
                grid.objects[sy, sx] = ObjectType.SHEEP

        # ── Barrier closing logic ─────────────────────────────────────────
        closes_at = config.get("barrier_closes", 40)
        warning = config.get("barrier_warning", 0)

        if warning > 0 and not config.get("_barrier_warned", False):
            if step_count >= closes_at - warning:
                for gx in config.get("gap_xs", []):
                    if grid.terrain[barrier_row, gx] == CellType.EMPTY:
                        # Only warn if no sheep sitting there
                        if grid.objects[barrier_row, gx] != ObjectType.SHEEP:
                            grid.terrain[barrier_row, gx] = CellType.WATER
                config["_barrier_warned"] = True

        if step_count >= closes_at and not config.get("_barrier_closed", False):
            for gx in config.get("gap_xs", []):
                t = grid.terrain[barrier_row, gx]
                if t in (CellType.EMPTY, CellType.WATER):
                    grid.terrain[barrier_row, gx] = CellType.HAZARD
            config["_barrier_closed"] = True

        # Check if agent is standing on hazard
        if grid.terrain[y_a, x_a] == CellType.HAZARD:
            config["_hit_barrier"] = True

    # ── Reward ────────────────────────────────────────────────────────────

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

            # Check if any gap is blocked by sheep
            sheep_set = set(map(tuple, config.get("_live_sheep", [])))
            barrier_row = config.get("barrier_row", 0)
            gap_xs = config.get("gap_xs", [])
            gaps_blocked = [
                gx for gx in gap_xs if (gx, barrier_row) in sheep_set
            ]
            gaps_clear = [gx for gx in gap_xs if gx not in gaps_blocked]

            if gaps_blocked and not gaps_clear:
                # All gaps blocked: reward for approaching the nearest sheep
                # (to scare it away)
                nearest_sheep = config.get("_live_sheep", [])
                if nearest_sheep:
                    d_new = min(abs(ax - sx) + abs(ay - sy) for sx, sy in nearest_sheep)
                    d_old = min(abs(ox - sx) + abs(oy - sy) for sx, sy in nearest_sheep)
                    reward += 0.03 * (d_old - d_new)
            elif gaps_clear:
                # At least one gap is clear: reward for approaching goal
                goal = config.get("goal_positions", [None])[0]
                if goal:
                    reward += 0.05 * (
                        (abs(ox - goal[0]) + abs(oy - goal[1]))
                        - (abs(ax - goal[0]) + abs(ay - goal[1]))
                    )

        if self.check_success(new_state):
            reward += 1.0
        return reward

    # ── Done / success ────────────────────────────────────────────────────

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
        return 1.0 + p.get("n_keys", 1) * p.get("key_bonus", 0.2)

    def get_random_baseline(self, difficulty=None):
        return 0.0
