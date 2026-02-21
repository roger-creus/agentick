"""DistributionShift - Navigate when the environment shifts mid-episode.

MECHANICS:
  - Phase 0: Real goal at pos A (GOAL object), fake goals as TARGET objects
  - At shift_step T: real goal moves to pos B, old GOAL becomes TARGET (fake)
  - Mid-episode RULE CHANGES at each shift:
    - Goal position changes
    - Wall patches toggle (appear/disappear)
    - Terrain cells flip (EMPTY↔HAZARD)
    - Action remapping: movement directions swap (UP↔DOWN or LEFT↔RIGHT)
  - Agent must detect each shift and adapt its behavior
  - Fake goals look identical to real goal until reached (luring agent)

DIFFICULTY AXES:
  - easy:   1 shift, goal moves once, no action remap
  - medium: 2 shifts, goal moves twice, partial remap (L↔R at shift 1)
  - hard:   3 shifts, terrain flips, full remap (U↔D at shift 1, L↔R at shift 2)
  - expert: 4 shifts, fast shifts, all remaps stack
"""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import ActionType, CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task

# Pre-defined action remap patterns
_REMAP_UD = {
    ActionType.MOVE_UP: ActionType.MOVE_DOWN,
    ActionType.MOVE_DOWN: ActionType.MOVE_UP,
}
_REMAP_LR = {
    ActionType.MOVE_LEFT: ActionType.MOVE_RIGHT,
    ActionType.MOVE_RIGHT: ActionType.MOVE_LEFT,
}


@register_task("DistributionShift-v0", tags=["generalization", "ood", "robustness"])
class DistributionShiftTask(TaskSpec):
    """Goal shifts mid-episode; agent must detect and adapt to rule changes."""

    name = "DistributionShift-v0"
    description = "Navigate when goals and rules change mid-episode"
    capability_tags = ["generalization", "ood", "robustness"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=7,
            max_steps=80,
            params={
                "n_shifts": 1,
                "n_fakes": 1,
                "walls_change": False,
                "shift_speed": 1.0,
                "terrain_changes": 0,
                "action_remaps": [],
            },
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=9,
            max_steps=130,
            params={
                "n_shifts": 2,
                "n_fakes": 2,
                "walls_change": False,
                "shift_speed": 1.0,
                "terrain_changes": 0,
                "action_remaps": ["lr"],  # L↔R swap at shift 1
            },
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=11,
            max_steps=200,
            params={
                "n_shifts": 3,
                "n_fakes": 3,
                "walls_change": True,
                "shift_speed": 1.5,
                "terrain_changes": 2,
                "action_remaps": ["ud", "lr"],  # U↔D at shift 1, L↔R at shift 2
            },
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=13,
            max_steps=280,
            params={
                "n_shifts": 4,
                "n_fakes": 4,
                "walls_change": True,
                "shift_speed": 2.0,
                "terrain_changes": 4,
                "action_remaps": ["ud", "lr", "ud", "lr"],
            },
        ),
    }

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        p = self.difficulty_config.params
        n_shifts = p.get("n_shifts", 1)
        n_fakes = p.get("n_fakes", 1)
        walls_change = p.get("walls_change", False)
        shift_speed = p.get("shift_speed", 1.0)
        terrain_changes = p.get("terrain_changes", 0)
        action_remaps = p.get("action_remaps", [])

        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        agent_pos = (1, 1)
        interior = [
            (x, y) for x in range(1, size - 1) for y in range(1, size - 1) if (x, y) != agent_pos
        ]
        rng.shuffle(interior)

        # N+1 goal positions (one per phase + initial)
        n_goal_positions = n_shifts + 1
        goal_sequence = interior[:n_goal_positions]
        fake_goals = interior[n_goal_positions : n_goal_positions + n_fakes]

        total_steps = self.get_max_steps()
        shift_interval = max(
            1,
            int(total_steps / ((n_shifts + 1) * shift_speed)),
        )
        shift_steps = [shift_interval * (i + 1) for i in range(n_shifts)]

        # Wall patches to add/remove at each shift
        wall_patches = []
        if walls_change and n_shifts > 0:
            remaining = interior[n_goal_positions + n_fakes :]
            chunk = max(1, len(remaining) // n_shifts)
            for i in range(n_shifts):
                patch = remaining[i * chunk : (i + 1) * chunk][:3]
                wall_patches.append(list(patch))
        else:
            wall_patches = [[] for _ in range(n_shifts)]

        # Terrain change cells
        terrain_cells = []
        if terrain_changes > 0:
            tc_pool = [
                pos
                for pos in interior[n_goal_positions + n_fakes :]
                if pos not in {q for patch in wall_patches for q in patch}
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
            "agent_start": agent_pos,
            "goal_positions": [goal_sequence[-1]] if goal_sequence else [interior[0]],
            "goal_sequence": goal_sequence,
            "fake_goals": fake_goals,
            "shift_steps": shift_steps,
            "wall_patches": wall_patches,
            "n_shifts": n_shifts,
            "shift_speed": shift_speed,
            "terrain_changes": terrain_changes,
            "_terrain_cells": terrain_cells,
            "_action_remaps_schedule": action_remaps,
            "max_steps": self.get_max_steps(),
            # backward-compat keys for tests
            "goal_a": goal_sequence[0] if goal_sequence else None,
            "goal_b": (
                goal_sequence[-1]
                if len(goal_sequence) > 1
                else (goal_sequence[0] if goal_sequence else None)
            ),
            "shift_step": shift_steps[0] if shift_steps else total_steps // 2,
        }

    def on_env_reset(self, agent, grid, config):
        config["_phase"] = 0
        config["_shifted"] = 0
        config["_fake_visit_penalty"] = False
        config["_action_remap"] = {}  # no remap initially
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
            config["_fake_visit_penalty"] = True
            grid.objects[y, x] = ObjectType.NONE

    def on_env_step(self, agent, grid, config, step_count):
        """Trigger goal shifts and rule changes at scheduled steps."""
        shift_steps = config.get("shift_steps", [])
        shifted = config.get("_shifted", 0)
        seq = config.get("goal_sequence", [])

        if shifted < len(shift_steps) and step_count >= shift_steps[shifted]:
            phase = shifted
            # Remove old GOAL marker (it becomes a fake/lure)
            if phase < len(seq):
                ox, oy = seq[phase]
                if grid.objects[oy, ox] == ObjectType.GOAL:
                    grid.objects[oy, ox] = ObjectType.TARGET

            # Place new GOAL at next position
            new_phase = phase + 1
            if new_phase < len(seq):
                nx, ny = seq[new_phase]
                grid.objects[ny, nx] = ObjectType.GOAL

            # Toggle wall patch
            patches = config.get("wall_patches", [])
            if shifted < len(patches):
                for wx, wy in patches[shifted]:
                    if 0 < wx < grid.width - 1 and 0 < wy < grid.height - 1:
                        if grid.terrain[wy, wx] == CellType.EMPTY:
                            grid.terrain[wy, wx] = CellType.WALL
                        else:
                            grid.terrain[wy, wx] = CellType.EMPTY

            # Terrain type changes
            for tx, ty in config.get("_terrain_cells", []):
                if 0 < tx < grid.width - 1 and 0 < ty < grid.height - 1:
                    if grid.terrain[ty, tx] == CellType.EMPTY:
                        grid.terrain[ty, tx] = CellType.HAZARD
                    elif grid.terrain[ty, tx] == CellType.HAZARD:
                        grid.terrain[ty, tx] = CellType.EMPTY

            # Action remapping: apply the scheduled remap for this shift
            schedule = config.get("_action_remaps_schedule", [])
            if shifted < len(schedule):
                remap_type = schedule[shifted]
                current = dict(config.get("_action_remap", {}))
                if remap_type == "ud":
                    # Toggle UP↔DOWN
                    for k, v in _REMAP_UD.items():
                        if k in current:
                            del current[k]  # undo previous remap
                        else:
                            current[k] = v
                elif remap_type == "lr":
                    # Toggle LEFT↔RIGHT
                    for k, v in _REMAP_LR.items():
                        if k in current:
                            del current[k]
                        else:
                            current[k] = v
                config["_action_remap"] = current

            config["_shifted"] = shifted + 1
            config["_phase"] = new_phase
            config["_fake_visit_penalty"] = False

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        config = new_state.get("config", {})
        if config.get("_fake_visit_penalty", False):
            reward -= 0.3
        if self.check_success(new_state):
            reward += 1.0
        elif "agent" in new_state:
            ax, ay = new_state["agent"].position
            ox, oy = old_state.get("agent_position", (ax, ay))
            seq = config.get("goal_sequence", [])
            phase = config.get("_phase", 0)
            if phase < len(seq):
                tgt = seq[phase]
                reward += 0.05 * (
                    (abs(ox - tgt[0]) + abs(oy - tgt[1])) - (abs(ax - tgt[0]) + abs(ay - tgt[1]))
                )
        return reward

    def check_success(self, state):
        """Success = agent at the CURRENT real GOAL object."""
        if "grid" not in state or "agent" not in state:
            return False
        x, y = state["agent"].position
        return bool(state["grid"].objects[y, x] == ObjectType.GOAL)

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0
