"""SwitchCircuit - Activate switches with dependency order and physical consequences.

MECHANICS:
  - N switches on the grid with a dependency DAG (circuit graph)
  - Switch B only activates if prerequisite switch A is already ON
  - Each switch activation has a visible physical consequence:
    - Opens a wall barrier (removes wall cells)
    - Deactivates a hazard zone (hazard → empty)
    - Creates a bridge over water (water → empty)
  - Stepping on a switch toggles it (on/off), but only if prerequisites are met
  - Gate to goal chamber opens when all switches are active simultaneously
  - Inverter switches (at hard+): toggling deactivates a dependent switch
  - Success = all switches active + agent at goal in chamber
"""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("SwitchCircuit-v0", tags=["combinatorial_logic", "reasoning"])
class SwitchCircuitTask(TaskSpec):
    """Activate switches respecting dependencies; each causes physical grid changes."""

    name = "SwitchCircuit-v0"
    description = "Activate circuit switches with physical consequences"
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
        """Build dependency DAG for switches."""
        deps = {i: [] for i in range(n_switches)}
        for i in range(1, n_switches):
            deps[i].append(i - 1)
        if n_switches >= 4:
            if 0 not in deps[3]:
                deps[3].append(0)
        inverter_set = set()
        if n_inverters > 0 and n_switches >= 3:
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

        agent_pos = (1, 1)

        # Build goal chamber in bottom-right corner
        cx, cy = size - 3, size - 3
        for wx in range(cx - 1, min(cx + 3, size)):
            for wy in range(cy - 1, min(cy + 3, size)):
                if 0 < wx < size - 1 and 0 < wy < size - 1:
                    grid.terrain[wy, wx] = CellType.WALL
        goal_pos = (cx, cy)
        grid.terrain[cy, cx] = CellType.EMPTY
        gate_pos = (cx - 1, cy)
        if gate_pos[0] < 1:
            gate_pos = (cx, cy - 1)
        grid.terrain[gate_pos[1], gate_pos[0]] = CellType.WALL

        # Build circuit
        deps, inverters = self._build_circuit(n, n_inv, rng)

        # Chamber cells to exclude
        chamber_cells = set()
        for wx in range(cx - 1, min(cx + 3, size)):
            for wy in range(cy - 1, min(cy + 3, size)):
                chamber_cells.add((wx, wy))
        chamber_cells.add(gate_pos)

        # Place switches
        free = [
            (x, y) for x in range(1, size - 1) for y in range(1, size - 1)
            if (x, y) != agent_pos and (x, y) not in chamber_cells
            and grid.terrain[y, x] == CellType.EMPTY
        ]
        rng.shuffle(free)
        switch_positions = free[:n]

        for sx, sy in switch_positions:
            grid.objects[sy, sx] = ObjectType.SWITCH

        # Create physical effects for each switch:
        # Each switch, when activated, changes terrain near its position
        switch_effects = []
        effect_types = [CellType.WALL, CellType.HAZARD, CellType.WATER]

        for i in range(n):
            sx, sy = switch_positions[i]
            effect_type = effect_types[i % len(effect_types)]
            # Place obstacle cluster near (but not on) the switch
            effect_cells = []
            candidates = [
                (sx + dx, sy + dy) for dx in range(-2, 3) for dy in range(-2, 3)
                if abs(dx) + abs(dy) == 2  # distance 2 from switch
            ]
            for ex, ey in candidates:
                if (1 <= ex < size - 1 and 1 <= ey < size - 1
                        and (ex, ey) not in chamber_cells
                        and (ex, ey) not in set(switch_positions)
                        and (ex, ey) != agent_pos
                        and grid.terrain[ey, ex] == CellType.EMPTY
                        and grid.objects[ey, ex] == ObjectType.NONE):
                    grid.terrain[ey, ex] = effect_type
                    effect_cells.append((ex, ey))
                    if len(effect_cells) >= 2:
                        break

            switch_effects.append({
                "cells": [list(c) for c in effect_cells],
                "original_type": int(effect_type),
            })

        # Verify agent can reach first switch
        reachable = grid.flood_fill(agent_pos)
        if switch_positions and switch_positions[0] not in reachable:
            # Remove obstacles blocking path
            for y in range(1, size - 1):
                for x in range(1, size - 1):
                    if (grid.terrain[y, x] in (CellType.HAZARD, CellType.WATER)
                            and (x, y) not in chamber_cells):
                        grid.terrain[y, x] = CellType.EMPTY

        deps_serialized = {i: sorted(deps[i]) for i in range(n)}

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [goal_pos],
            "switch_positions": switch_positions,
            "switch_effects": switch_effects,
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
        effects = config.get("switch_effects", [])
        ax, ay = pos

        if (ax, ay) in switches:
            idx = switches.index((ax, ay))
            key = idx

            if key in active:
                # Toggle OFF
                active.discard(key)
                grid.objects[ay, ax] = ObjectType.SWITCH

                # Reverse physical effect: restore obstacle
                if idx < len(effects):
                    eff = effects[idx]
                    orig_type = CellType(eff["original_type"])
                    for cx, cy in eff["cells"]:
                        if grid.terrain[cy, cx] == CellType.EMPTY:
                            grid.terrain[cy, cx] = orig_type
            else:
                # Check prerequisites
                prereqs = deps.get(idx, deps.get(str(idx), []))
                prereqs_met = all(p in active for p in prereqs)

                if prereqs_met:
                    active.add(key)
                    grid.objects[ay, ax] = ObjectType.NONE  # activated visual

                    # Physical effect: clear obstacle
                    if idx < len(effects):
                        eff = effects[idx]
                        for cx, cy in eff["cells"]:
                            grid.terrain[cy, cx] = CellType.EMPTY

                    # Inverter: deactivate dependent switch
                    if idx in inverters:
                        for other_idx, other_deps in deps.items():
                            if isinstance(other_idx, str):
                                other_idx = int(other_idx)
                            if idx in other_deps and other_idx in active:
                                active.discard(other_idx)
                                ox, oy = switches[other_idx]
                                grid.objects[oy, ox] = ObjectType.SWITCH
                                # Restore that switch's effect
                                if other_idx < len(effects):
                                    oeff = effects[other_idx]
                                    orig = CellType(oeff["original_type"])
                                    for ecx, ecy in oeff["cells"]:
                                        if grid.terrain[ecy, ecx] == CellType.EMPTY:
                                            grid.terrain[ecy, ecx] = orig

            config["_switches_on"] = active

            # Open/close gate based on all switches active
            if len(active) >= len(switches):
                gx, gy = config.get("gate_pos", (0, 0))
                grid.terrain[gy, gx] = CellType.EMPTY
            else:
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
        if "agent" in new_state:
            ax, ay = new_state["agent"].position
            ox, oy = old_state.get("agent_position", (ax, ay))
            active = config.get("_switches_on", set())
            switches = config.get("switch_positions", [])
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
        config = state.get("config", {})
        active = config.get("_switches_on", set())
        switches = config.get("switch_positions", [])
        if len(active) < len(switches):
            return False
        x, y = state["agent"].position
        return bool(state["grid"].objects[y, x] == ObjectType.GOAL)

    def validate_instance(self, grid, config):
        return True

    def get_optimal_return(self, difficulty=None): return 1.0
    def get_random_baseline(self, difficulty=None): return 0.0
