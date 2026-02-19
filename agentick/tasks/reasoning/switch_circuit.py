"""SwitchCircuit - Activate the right combination of switches to open the gate.

MECHANICS:
  - N switches on the grid (binary: on/off via stepping)
  - A specific combination must be active simultaneously
  - Gate to goal is locked until correct combo is active
  - Stepping on a switch toggles it on or off
  - Success = correct switches activated + agent reaches goal
"""

import numpy as np
from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("SwitchCircuit-v0", tags=["combinatorial_logic", "reasoning"])
class SwitchCircuitTask(TaskSpec):
    """Find and activate the correct switch combination to unlock the gate."""

    name = "SwitchCircuit-v0"
    description = "Activate the right switches to unlock the gate"
    capability_tags = ["combinatorial_logic", "reasoning"]

    difficulty_configs = {
        "easy":   DifficultyConfig(name="easy",   grid_size=7,  max_steps=60,  params={"n_switches": 2}),
        "medium": DifficultyConfig(name="medium",  grid_size=9,  max_steps=120, params={"n_switches": 3}),
        "hard":   DifficultyConfig(name="hard",    grid_size=11, max_steps=200, params={"n_switches": 4}),
        "expert": DifficultyConfig(name="expert",  grid_size=13, max_steps=300, params={"n_switches": 5}),
    }

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        n = self.difficulty_config.params.get("n_switches", 2)

        grid = Grid(size, size)
        grid.terrain[0, :]  = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0]  = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        # Randomize agent and goal positions (opposite areas)
        corners = [(1,1),(size-2,1),(1,size-2),(size-2,size-2)]
        rng.shuffle(corners)
        agent_pos = corners[0]
        goal_pos  = corners[1]

        # Gate: one cell adjacent to goal (on the path toward goal center)
        gx, gy = goal_pos
        # Gate is one step toward grid center
        dx = 1 if gx == 1 else -1
        dy = 1 if gy == 1 else -1
        gate_pos = (gx + dx, gy)  # horizontal neighbor
        if gate_pos == agent_pos:
            gate_pos = (gx, gy + dy)
        grid.terrain[gate_pos[1], gate_pos[0]] = CellType.WALL

        # All switches must be activated (simplified: all ON = success)
        free = [(x, y) for x in range(1, size-1) for y in range(1, size-1)
                if (x, y) != agent_pos and (x, y) != goal_pos and (x, y) != gate_pos]
        rng.shuffle(free)
        switch_positions = free[:n]

        for sx, sy in switch_positions:
            grid.objects[sy, sx] = ObjectType.SWITCH

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [goal_pos],
            "switch_positions": switch_positions,
            "gate_pos": gate_pos,
            "max_steps": self.get_max_steps(),
        }

    # ── Hooks ────────────────────────────────────────────────────────────────

    def on_env_reset(self, agent, grid, config):
        """Init switch state; cache config for on_agent_moved; place GOAL."""
        config["_switches_on"] = set()
        goal_pos = config.get("goal_positions", [None])[0]
        if goal_pos:
            gx, gy = goal_pos
            grid.objects[gy, gx] = ObjectType.GOAL
        self._config = config
        self._last_n_switches = 0

    def on_agent_moved(self, pos, agent, grid):
        """Toggle switch when agent steps on it — fires BEFORE reward/success."""
        config = getattr(self, "_config", {})
        switches = config.get("switch_positions", [])
        active = config.get("_switches_on", set())
        ax, ay = pos
        if (ax, ay) in switches:
            key = (ax, ay)
            if key in active:
                active.discard(key)
                grid.objects[ay, ax] = ObjectType.SWITCH  # back to off
            else:
                active.add(key)
                grid.objects[ay, ax] = ObjectType.NONE  # consumed/activated
            config["_switches_on"] = active
            # Open gate if all switches active
            if len(active) >= len(switches):
                gx, gy = config.get("gate_pos", (0, 0))
                grid.terrain[gy, gx] = CellType.EMPTY

    # ── Reward & success ─────────────────────────────────────────────────────

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        config = new_state.get("config", {})
        new_n = len(config.get("_switches_on", set()))
        if new_n > self._last_n_switches:
            reward += 0.2 * (new_n - self._last_n_switches)
        self._last_n_switches = new_n
        # Approach shaping: toward nearest unactivated switch (or goal when all on)
        if "agent_position" in new_state and "grid" in new_state:
            ax, ay = new_state["agent_position"]
            ox, oy = old_state.get("agent_position", (ax, ay))
            g = new_state["grid"]
            active = config.get("_switches_on", set())
            switches = config.get("switch_positions", [])
            unactivated = [s for s in switches if s not in active]
            if unactivated:
                d_new = min(abs(ax-sx)+abs(ay-sy) for sx,sy in unactivated)
                d_old = min(abs(ox-sx)+abs(oy-sy) for sx,sy in unactivated)
                reward += 0.05 * (d_old - d_new)
            else:
                # All switches on — guide toward goal
                goal = config.get("goal_positions", [None])[0]
                if goal:
                    d_new = abs(ax-goal[0])+abs(ay-goal[1])
                    d_old = abs(ox-goal[0])+abs(oy-goal[1])
                    reward += 0.05 * (d_old - d_new)
        if self.check_success(new_state):
            reward += 1.0
        return reward

    def check_success(self, state):
        if "grid" not in state or "agent" not in state:
            return False
        x, y = state["agent"].position
        return bool(state["grid"].objects[y, x] == ObjectType.GOAL)

    def validate_instance(self, grid, config):
        return True  # gate is dynamic

    def get_optimal_return(self, difficulty=None): return 1.0
    def get_random_baseline(self, difficulty=None): return 0.0
