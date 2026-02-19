"""BacktrackPuzzle - Reach the switch PAST the goal, then backtrack to claim it.

MECHANICS:
  - The agent must walk PAST the goal (which starts locked) to reach the switch
  - Activating the switch unlocks the goal
  - Agent must BACKTRACK to the now-unlocked goal
  - True backtracking: agent always overshoots the goal position

PROCEDURAL DIVERSITY:
  - Random corridor orientations and room counts per seed
  - Switch and goal placed so agent ALWAYS passes goal before switch
  - Multiple rooms / branches at higher difficulties
  - Solvability verified via flood-fill
"""

import numpy as np
from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("BacktrackPuzzle-v0", tags=["memory", "planning"])
class BacktrackPuzzleTask(TaskSpec):
    """Activate a switch PAST the goal, then backtrack to claim it."""

    name = "BacktrackPuzzle-v0"
    description = "Overshoot goal to hit switch, then backtrack"
    capability_tags = ["memory", "planning"]

    difficulty_configs = {
        "easy":   DifficultyConfig(name="easy",   grid_size=9,  max_steps=80,  params={"n_switches": 1}),
        "medium": DifficultyConfig(name="medium",  grid_size=11, max_steps=150, params={"n_switches": 2}),
        "hard":   DifficultyConfig(name="hard",    grid_size=13, max_steps=250, params={"n_switches": 3}),
        "expert": DifficultyConfig(name="expert",  grid_size=15, max_steps=400, params={"n_switches": 4}),
    }

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        n_sw = self.difficulty_config.params.get("n_switches", 1)

        # DUAL-CORRIDOR BYPASS LAYOUT (always solvable):
        #
        #  bypass_row: [A][.][.][.][.][.][S]   — agent can walk here to reach switch
        #  main_row:   [A][.][G][.][.][.][.]   — goal locked behind gate
        #               gate^   goal^
        #
        # Agent connects from main_row to bypass_row at left end (col 1).
        # Agent path: main→bypass via (1,bypass), walk right to switch (size-2),
        # activate switch, gate opens, backtrack to goal on main_row.

        # Randomize which two rows and where the gate/goal sit
        mid = size // 2
        main_row_offset   = int(rng.integers(0, 2))  # 0 or 1 row offset from mid
        main_row   = max(2, min(size-3, mid + main_row_offset))
        bypass_row = main_row - 1 if main_row > 2 else main_row + 1

        grid = Grid(size, size)
        for y in range(size):
            for x in range(size):
                grid.terrain[y, x] = CellType.WALL

        # Open both corridors fully
        for x in range(1, size-1):
            grid.terrain[main_row, x]   = CellType.EMPTY
            grid.terrain[bypass_row, x] = CellType.EMPTY

        # Random goal position (middle-left area of main corridor)
        goal_col = int(rng.integers(size // 3, size // 2 + 1))
        goal_col = max(2, min(size-3, goal_col))
        goal_pos = (goal_col, main_row)

        # Gate: one cell LEFT of goal on main_row, blocking direct access
        gate_col = goal_col - 1
        gate_col = max(2, gate_col)
        gate_pos = (gate_col, main_row)
        grid.terrain[main_row, gate_col] = CellType.WALL

        # Switch positions: far RIGHT of bypass_row (past the goal area)
        switch_positions = []
        for i in range(n_sw):
            sc = max(size - 2 - i, goal_col + 2)
            sc = min(sc, size-2)
            # Put first switch on bypass_row, extra switches on main_row past gate
            if i == 0:
                sw_pos = (sc, bypass_row)
            else:
                sw_pos = (sc, main_row)
            switch_positions.append(sw_pos)

        agent_pos = (1, main_row)

        # Verify: agent can reach all switches
        reachable = grid.flood_fill(agent_pos)
        for sw in switch_positions:
            if sw not in reachable:
                # Fallback: ensure switch row is open
                grid.terrain[sw[1], sw[0]] = CellType.EMPTY

        # Re-verify
        reachable = grid.flood_fill(agent_pos)
        if not all(sw in reachable for sw in switch_positions):
            # Ultimate fallback: simple guaranteed layout
            grid = Grid(size, size)
            for y in range(size):
                for x in range(size):
                    grid.terrain[y, x] = CellType.WALL
            row = mid
            byp = row - 1
            for x in range(1, size-1):
                grid.terrain[row, x] = CellType.EMPTY
                grid.terrain[byp, x] = CellType.EMPTY
            gate_col = size // 3
            goal_col = gate_col + 1
            grid.terrain[row, gate_col] = CellType.WALL
            agent_pos       = (1, row)
            goal_pos        = (goal_col, row)
            gate_pos        = (gate_col, row)
            switch_positions = [(size-2, byp)]
            bypass_row       = byp

        # Place switch objects
        for sx, sy in switch_positions:
            grid.objects[sy, sx] = ObjectType.SWITCH

        return grid, {
            "agent_start":      agent_pos,
            "goal_positions":   [goal_pos],
            "switch_positions": switch_positions,
            "switch_pos":       switch_positions[0] if switch_positions else None,
            "gate_pos":         gate_pos,
            "n_switches":       n_sw,
            "max_steps":        self.get_max_steps(),
        }

    # ── Hooks ────────────────────────────────────────────────────────────────

    def on_env_reset(self, agent, grid, config):
        config["_switches_activated"] = 0
        config["_all_activated"] = False
        # Goal is LOCKED initially — no GOAL object, and terrain around gate is WALL
        self._switch_milestone_given = False
        self._config = config

    def on_agent_moved(self, pos, agent, grid):
        """Activate switches immediately when stepped on."""
        config = getattr(self, "_config", {})
        ax, ay = pos
        switches = config.get("switch_positions", [])

        for sw in list(switches):
            sx, sy = sw
            if (ax, ay) == (sx, sy) and grid.objects[sy, sx] == ObjectType.SWITCH:
                grid.objects[sy, sx] = ObjectType.NONE
                config["_switches_activated"] = config.get("_switches_activated", 0) + 1
                config["_switch_activated"] = True  # backward-compat alias for tests

        # When ALL switches activated, open gate and place GOAL
        n_needed = config.get("n_switches", 1)
        if (config.get("_switches_activated", 0) >= n_needed
                and not config.get("_all_activated", False)):
            config["_all_activated"] = True
            gx, gy = config["gate_pos"]
            grid.terrain[gy, gx] = CellType.EMPTY   # open gate
            gx2, gy2 = config["goal_positions"][0]
            grid.objects[gy2, gx2] = ObjectType.GOAL  # unlock goal

    # ── Reward & success ─────────────────────────────────────────────────────

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        config = new_state.get("config", {})
        activated = config.get("_switches_activated", 0)
        n_needed  = config.get("n_switches", 1)

        # Milestone: each switch activated
        if activated > 0 and not self._switch_milestone_given:
            reward += 0.3 * activated
            self._switch_milestone_given = True

        if not config.get("_all_activated", False):
            # Guide toward next switch (first unactivated)
            switches = config.get("switch_positions", [])
            remaining = [sw for sw in switches
                         if "grid" not in new_state or
                         new_state["grid"].objects[sw[1], sw[0]] == ObjectType.SWITCH]
            if remaining and "agent" in new_state:
                ax, ay = new_state["agent"].position
                ox, oy = old_state.get("agent_position", (ax, ay))
                tgt = remaining[0]
                d_new = abs(ax - tgt[0]) + abs(ay - tgt[1])
                d_old = abs(ox - tgt[0]) + abs(oy - tgt[1])
                reward += 0.05 * (d_old - d_new)
        else:
            # All switches done → guide to goal
            goal = config.get("goal_positions", [None])[0]
            if goal and "agent" in new_state:
                ax, ay = new_state["agent"].position
                ox, oy = old_state.get("agent_position", (ax, ay))
                d_new = abs(ax - goal[0]) + abs(ay - goal[1])
                d_old = abs(ox - goal[0]) + abs(oy - goal[1])
                reward += 0.05 * (d_old - d_new)

        if self.check_success(new_state):
            reward += 1.0
        return reward

    def check_success(self, state):
        config = state.get("config", {})
        if not config.get("_all_activated", False):
            return False  # must activate ALL switches first
        if "grid" not in state or "agent" not in state:
            return False
        x, y = state["agent"].position
        return bool(state["grid"].objects[y, x] == ObjectType.GOAL)

    def validate_instance(self, grid, config):
        return True  # gate is dynamic

    def get_optimal_return(self, difficulty=None): return 1.0
    def get_random_baseline(self, difficulty=None): return 0.0
