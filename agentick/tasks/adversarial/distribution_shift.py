"""DistributionShift - Navigate when the environment shifts mid-episode.

PROCEDURAL DIVERSITY + CREATIVE DIFFICULTY AXES:
  - easy:   1 shift, goal moves once, fake goal visible from start
  - medium: 2 shifts, goal moves twice, multiple fake goals
  - hard:   3 shifts, faster shifts, terrain type changes, walls change
  - expert: 4 shifts, 2x shift speed, 4 terrain flips, short windows

MECHANICS:
  - Phase 0: Real goal at pos A (GOAL object), fake goals as TARGET objects
  - At shift_step T: real goal moves to pos B, old GOAL becomes TARGET (fake)
  - Agent must detect shift and navigate to new real goal
  - Fake goals look identical to real goal until reached (luring agent)
  - Walls may appear/disappear at shifts (harder difficulties)
"""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("DistributionShift-v0", tags=["generalization", "ood", "robustness"])
class DistributionShiftTask(TaskSpec):
    """Goal shifts mid-episode; agent must detect and adapt."""

    name = "DistributionShift-v0"
    description = "Navigate when goal moves mid-episode"
    capability_tags = ["generalization", "ood", "robustness"]

    difficulty_configs = {
        "easy":   DifficultyConfig(
            name="easy", grid_size=7, max_steps=80,
            params={
                "n_shifts": 1, "n_fakes": 1,
                "walls_change": False,
                "shift_speed": 1.0, "terrain_changes": 0,
            },
        ),
        "medium": DifficultyConfig(
            name="medium", grid_size=9, max_steps=130,
            params={
                "n_shifts": 2, "n_fakes": 2,
                "walls_change": False,
                "shift_speed": 1.0, "terrain_changes": 0,
            },
        ),
        "hard":   DifficultyConfig(
            name="hard", grid_size=11, max_steps=200,
            params={
                "n_shifts": 3, "n_fakes": 3,
                "walls_change": True,
                "shift_speed": 1.5, "terrain_changes": 2,
            },
        ),
        "expert": DifficultyConfig(
            name="expert", grid_size=13, max_steps=280,
            params={
                "n_shifts": 4, "n_fakes": 4,
                "walls_change": True,
                "shift_speed": 2.0, "terrain_changes": 4,
            },
        ),
    }

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        p = self.difficulty_config.params
        n_shifts        = p.get("n_shifts", 1)
        n_fakes         = p.get("n_fakes", 1)
        walls_change    = p.get("walls_change", False)
        shift_speed     = p.get("shift_speed", 1.0)
        terrain_changes = p.get("terrain_changes", 0)

        grid = Grid(size, size)
        grid.terrain[0, :]  = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0]  = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        agent_pos = (1, 1)
        interior  = [(x, y) for x in range(1, size-1) for y in range(1, size-1)
                     if (x, y) != agent_pos]
        rng.shuffle(interior)

        # N+1 goal positions (one per phase + initial)
        n_goal_positions = n_shifts + 1
        goal_sequence    = interior[:n_goal_positions]
        # Fake goals (placed at start, stay throughout as visual lures)
        fake_goals       = interior[n_goal_positions: n_goal_positions + n_fakes]

        # Interval between shifts — shift_speed > 1 makes shifts
        # come faster, compressing the time the agent has to react
        total_steps   = self.get_max_steps()
        shift_interval = max(
            1, int(total_steps / ((n_shifts + 1) * shift_speed)),
        )
        shift_steps = [
            shift_interval * (i + 1) for i in range(n_shifts)
        ]

        # Optional: wall patches to add/remove at each shift
        wall_patches = []
        if walls_change and n_shifts > 0:
            remaining = interior[n_goal_positions + n_fakes:]
            chunk = max(1, len(remaining) // n_shifts)
            for i in range(n_shifts):
                patch = remaining[i*chunk:(i+1)*chunk][:3]
                wall_patches.append(list(patch))
        else:
            wall_patches = [[] for _ in range(n_shifts)]

        # Terrain change cells: cells that flip between EMPTY
        # and HAZARD/WATER at each shift (hard/expert only)
        terrain_cells = []
        if terrain_changes > 0:
            tc_pool = [
                pos for pos in interior[n_goal_positions + n_fakes:]
                if pos not in {
                    p for patch in wall_patches for p in patch
                }
            ]
            rng.shuffle(tc_pool)
            terrain_cells = tc_pool[:terrain_changes]

        # Place initial goal and fake goals
        if goal_sequence:
            gx, gy = goal_sequence[0]
            grid.objects[gy, gx] = ObjectType.GOAL
        for fx, fy in fake_goals:
            grid.objects[fy, fx] = ObjectType.TARGET

        return grid, {
            "agent_start":    agent_pos,
            "goal_positions": [goal_sequence[-1]] if goal_sequence else [interior[0]],
            "goal_sequence":  goal_sequence,
            "fake_goals":     fake_goals,
            "shift_steps":    shift_steps,
            "wall_patches":   wall_patches,
            "n_shifts":       n_shifts,
            "shift_speed":    shift_speed,
            "terrain_changes": terrain_changes,
            "_terrain_cells": terrain_cells,
            "max_steps":      self.get_max_steps(),
            # backward-compat keys for tests
            "goal_a":         goal_sequence[0] if goal_sequence else None,
            "goal_b":         goal_sequence[-1] if len(goal_sequence) > 1 else (goal_sequence[0] if goal_sequence else None),
            "shift_step":     shift_steps[0] if shift_steps else total_steps // 2,
        }

    def on_env_reset(self, agent, grid, config):
        config["_phase"]    = 0   # which goal is currently real
        config["_shifted"]  = 0   # how many shifts have happened
        config["_fake_visit_penalty"] = False
        self._last_phase = 0
        self._config = config
        # Redraw initial state
        seq = config.get("goal_sequence", [])
        if seq:
            gx, gy = seq[0]
            grid.objects[gy, gx] = ObjectType.GOAL
        for fx, fy in config.get("fake_goals", []):
            if grid.objects[fy, fx] != ObjectType.GOAL:
                grid.objects[fy, fx] = ObjectType.TARGET

    def on_agent_moved(self, pos, agent, grid):
        """Penalize reaching a fake goal (lure)."""
        config = getattr(self, "_config", {})
        x, y = pos
        if grid.objects[y, x] == ObjectType.TARGET:
            # Agent reached a fake goal — small penalty, episode continues
            config["_fake_visit_penalty"] = True
            grid.objects[y, x] = ObjectType.NONE  # remove the fake (used up)

    def on_env_step(self, agent, grid, config, step_count):
        """Trigger goal shifts at scheduled steps."""
        shift_steps = config.get("shift_steps", [])
        shifted     = config.get("_shifted", 0)
        seq         = config.get("goal_sequence", [])

        if shifted < len(shift_steps) and step_count >= shift_steps[shifted]:
            phase = shifted  # current phase (0-indexed goal)
            # Remove old GOAL marker (it becomes a fake/lure)
            if phase < len(seq):
                ox, oy = seq[phase]
                if grid.objects[oy, ox] == ObjectType.GOAL:
                    grid.objects[oy, ox] = ObjectType.TARGET  # becomes fake

            # Place new GOAL at next position
            new_phase = phase + 1
            if new_phase < len(seq):
                nx, ny = seq[new_phase]
                grid.objects[ny, nx] = ObjectType.GOAL

            # Optional: toggle wall patch for this shift
            patches = config.get("wall_patches", [])
            if shifted < len(patches):
                for wx, wy in patches[shifted]:
                    if 0 < wx < grid.width-1 and 0 < wy < grid.height-1:
                        if grid.terrain[wy, wx] == CellType.EMPTY:
                            grid.terrain[wy, wx] = CellType.WALL
                        else:
                            grid.terrain[wy, wx] = CellType.EMPTY

            # Terrain type changes: flip designated cells
            # between EMPTY and HAZARD at each shift
            for tx, ty in config.get("_terrain_cells", []):
                if 0 < tx < grid.width-1 and 0 < ty < grid.height-1:
                    if grid.terrain[ty, tx] == CellType.EMPTY:
                        grid.terrain[ty, tx] = CellType.HAZARD
                    elif grid.terrain[ty, tx] == CellType.HAZARD:
                        grid.terrain[ty, tx] = CellType.EMPTY

            config["_shifted"] = shifted + 1
            config["_phase"]   = new_phase
            config["_fake_visit_penalty"] = False  # reset per phase

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        config = new_state.get("config", {})
        if config.get("_fake_visit_penalty", False):
            reward -= 0.3  # penalty for going to a fake goal
        if self.check_success(new_state):
            reward += 1.0
        elif "agent" in new_state:
            ax, ay = new_state["agent"].position
            ox, oy = old_state.get("agent_position", (ax, ay))
            seq     = config.get("goal_sequence", [])
            phase   = config.get("_phase", 0)
            if phase < len(seq):
                tgt = seq[phase]
                reward += 0.05 * ((abs(ox-tgt[0])+abs(oy-tgt[1])) - (abs(ax-tgt[0])+abs(ay-tgt[1])))
        return reward

    def check_success(self, state):
        """Success = agent at the CURRENT real GOAL object."""
        if "grid" not in state or "agent" not in state:
            return False
        x, y = state["agent"].position
        return bool(state["grid"].objects[y, x] == ObjectType.GOAL)

    def get_optimal_return(self, difficulty=None): return 1.0
    def get_random_baseline(self, difficulty=None): return 0.0
