"""SwitchCircuit - Activate switches in dependency order to open the gate.

MECHANICS:
  - N switches on the grid with dependency relationships (circuit graph)
  - Switch B only activates if its prerequisite switch A is already ON
  - Stepping on a switch toggles it (on/off), but only if prerequisites are met
  - Gate opens when all required switches are active simultaneously
  - At higher difficulties: some switches are inverters (toggling deactivates another)
  - Success = all required switches active + agent reaches goal
"""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("SwitchCircuit-v0", tags=["combinatorial_logic", "reasoning"])
class SwitchCircuitTask(TaskSpec):
    """Activate switches respecting dependency order to unlock the gate."""

    name = "SwitchCircuit-v0"
    description = "Activate switches in circuit dependency order"
    capability_tags = ["combinatorial_logic", "reasoning"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy", grid_size=7, max_steps=80,
            params={"n_switches": 2, "n_inverters": 0},
        ),
        "medium": DifficultyConfig(
            name="medium", grid_size=9, max_steps=150,
            params={"n_switches": 3, "n_inverters": 0},
        ),
        "hard": DifficultyConfig(
            name="hard", grid_size=11, max_steps=250,
            params={"n_switches": 4, "n_inverters": 1},
        ),
        "expert": DifficultyConfig(
            name="expert", grid_size=13, max_steps=400,
            params={"n_switches": 5, "n_inverters": 2},
        ),
    }

    def _build_circuit(self, n_switches, n_inverters, rng):
        """Build a dependency DAG for switches.

        Returns:
            deps: dict mapping switch_idx -> list of prerequisite switch indices
            inverters: set of switch indices that are inverters
                       (toggling deactivates the next switch in chain)
        """
        deps = {i: [] for i in range(n_switches)}

        # Build a chain: switch i depends on switch i-1
        # This creates a linear dependency: must activate 0, then 1, then 2, etc.
        for i in range(1, n_switches):
            deps[i].append(i - 1)

        # Add some cross-dependencies at higher counts for complexity
        if n_switches >= 4:
            # Switch 3 also depends on switch 0 (skip dependency)
            if 0 not in deps[3]:
                deps[3].append(0)

        # Designate inverter switches (toggling these deactivates their dependent)
        inverter_set = set()
        if n_inverters > 0 and n_switches >= 3:
            # Pick switches from the middle of the chain as inverters
            candidates = list(range(1, n_switches - 1))
            rng.shuffle(candidates)
            for idx in candidates[:n_inverters]:
                inverter_set.add(idx)

        return deps, inverter_set

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        n = self.difficulty_config.params.get("n_switches", 2)
        n_inv = self.difficulty_config.params.get("n_inverters", 0)

        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        # Randomize agent and goal positions
        corners = [(1, 1), (size - 2, 1), (1, size - 2), (size - 2, size - 2)]
        rng.shuffle(corners)
        agent_pos = corners[0]
        goal_pos = corners[1]

        # Gate adjacent to goal
        gx, gy = goal_pos
        dx = 1 if gx == 1 else -1
        dy = 1 if gy == 1 else -1
        gate_pos = (gx + dx, gy)
        if gate_pos == agent_pos:
            gate_pos = (gx, gy + dy)
        grid.terrain[gate_pos[1], gate_pos[0]] = CellType.WALL

        # Build circuit
        deps, inverters = self._build_circuit(n, n_inv, rng)

        # Place switches
        free = [
            (x, y) for x in range(1, size - 1) for y in range(1, size - 1)
            if (x, y) != agent_pos and (x, y) != goal_pos and (x, y) != gate_pos
        ]
        rng.shuffle(free)
        switch_positions = free[:n]

        for sx, sy in switch_positions:
            grid.objects[sy, sx] = ObjectType.SWITCH

        # Serialize deps for config
        deps_serialized = {i: sorted(deps[i]) for i in range(n)}

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [goal_pos],
            "switch_positions": switch_positions,
            "gate_pos": gate_pos,
            "circuit_deps": deps_serialized,
            "inverter_switches": sorted(inverters),
            "max_steps": self.get_max_steps(),
        }

    def on_env_reset(self, agent, grid, config):
        config["_switches_on"] = set()
        goal_pos = config.get("goal_positions", [None])[0]
        if goal_pos:
            gx, gy = goal_pos
            grid.objects[gy, gx] = ObjectType.GOAL
        self._config = config
        self._last_n_switches = 0

    def on_agent_moved(self, pos, agent, grid):
        config = getattr(self, "_config", {})
        switches = config.get("switch_positions", [])
        active = config.get("_switches_on", set())
        deps = config.get("circuit_deps", {})
        inverters = set(config.get("inverter_switches", []))
        ax, ay = pos

        if (ax, ay) in switches:
            idx = switches.index((ax, ay))
            key = idx

            if key in active:
                # Toggle OFF
                active.discard(key)
                grid.objects[ay, ax] = ObjectType.SWITCH

                # If this is an inverter, toggling OFF re-enables its dependent
                # (no special action needed — dependent just needs prereqs met)
            else:
                # Check prerequisites
                prereqs = deps.get(idx, deps.get(str(idx), []))
                prereqs_met = all(p in active for p in prereqs)

                if prereqs_met:
                    active.add(key)
                    grid.objects[ay, ax] = ObjectType.NONE  # activated

                    # If this is an inverter, deactivate its dependent switch
                    if idx in inverters:
                        # Find switches that depend on this one
                        for other_idx, other_deps in deps.items():
                            if isinstance(other_idx, str):
                                other_idx = int(other_idx)
                            if idx in other_deps and other_idx in active:
                                active.discard(other_idx)
                                ox, oy = switches[other_idx]
                                grid.objects[oy, ox] = ObjectType.SWITCH

            config["_switches_on"] = active

            # Open gate if all switches active
            if len(active) >= len(switches):
                gx, gy = config.get("gate_pos", (0, 0))
                grid.terrain[gy, gx] = CellType.EMPTY
            else:
                # Close gate if not all active
                gx, gy = config.get("gate_pos", (0, 0))
                if grid.terrain[gy, gx] == CellType.EMPTY:
                    grid.terrain[gy, gx] = CellType.WALL

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        config = new_state.get("config", {})
        new_n = len(config.get("_switches_on", set()))

        if new_n > self._last_n_switches:
            reward += 0.2 * (new_n - self._last_n_switches)
        elif new_n < self._last_n_switches:
            reward -= 0.1 * (self._last_n_switches - new_n)
        self._last_n_switches = new_n

        # Approach shaping
        if "agent_position" in new_state and "grid" in new_state:
            ax, ay = new_state["agent_position"]
            ox, oy = old_state.get("agent_position", (ax, ay))
            active = config.get("_switches_on", set())
            switches = config.get("switch_positions", [])

            # Find next activatable switch (first with met prereqs that isn't on)
            deps = config.get("circuit_deps", {})
            target_switches = []
            for i, s in enumerate(switches):
                if i not in active:
                    prereqs = deps.get(i, deps.get(str(i), []))
                    if all(p in active for p in prereqs):
                        target_switches.append(s)

            if target_switches:
                d_new = min(abs(ax - sx) + abs(ay - sy) for sx, sy in target_switches)
                d_old = min(abs(ox - sx) + abs(oy - sy) for sx, sy in target_switches)
                reward += 0.05 * (d_old - d_new)
            elif len(active) >= len(switches):
                goal = config.get("goal_positions", [None])[0]
                if goal:
                    d_new = abs(ax - goal[0]) + abs(ay - goal[1])
                    d_old = abs(ox - goal[0]) + abs(oy - goal[1])
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
        return True

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0
