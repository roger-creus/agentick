"""DistributionShift - Navigate when the environment dramatically shifts mid-episode.

MECHANICS:
  - Phase 1 (first ~40% of steps): NO goal visible; agent must explore & collect COINs
    (the goal position exists but is BLOCKED or absent so oracle CANNOT reach it early)
  - At shift_step: DRAMATIC VISIBLE CHANGE — new GOAL appears, terrain mutates,
    walls toggle, colours of objects change (TARGET→GOAL swaps, hazards flip)
  - Phase 2 (remaining ~60%): Goal is now reachable; agent must find and reach it
  - Agent must DETECT that the shift happened and ADAPT its strategy
  - Easy:   1 shift, obvious flashing warning (BLOCKER objects appear 5 steps before),
            one region changes
  - Medium: 2 shifts, moderate warning (3 steps), two regions change
  - Hard:   3 shifts, minimal warning (1 step), terrain type swaps, action remap
  - Expert: 4 shifts, no warning, multiple simultaneous changes, rapid shifts

PHASE 1 OBJECTIVE:
  Collecting coins gives reward so oracle is drawn to coins, not to absent goal.
  The goal cell starts as BLOCKER terrain (impassable) and only becomes walkable
  on the shift — oracle cannot reach it in Phase 1.
"""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import ActionType, CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task

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
    """Goal and rules shift mid-episode; agent must detect and adapt."""

    name = "DistributionShift-v0"
    description = "Navigate when goals and rules change mid-episode"
    capability_tags = ["generalization", "ood", "robustness"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=7,
            max_steps=120,
            params={
                "n_shifts": 1,
                "n_coins": 3,
                "walls_change": False,
                "terrain_changes": 0,
                "action_remaps": [],
                "warning_steps": 5,
                "shift_frac": 0.4,
            },
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=9,
            max_steps=200,
            params={
                "n_shifts": 2,
                "n_coins": 4,
                "walls_change": True,
                "terrain_changes": 2,
                "action_remaps": ["lr"],
                "warning_steps": 3,
                "shift_frac": 0.35,
            },
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=11,
            max_steps=300,
            params={
                "n_shifts": 3,
                "n_coins": 5,
                "walls_change": True,
                "terrain_changes": 4,
                "action_remaps": ["ud", "lr"],
                "warning_steps": 1,
                "shift_frac": 0.30,
            },
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=13,
            max_steps=450,
            params={
                "n_shifts": 4,
                "n_coins": 6,
                "walls_change": True,
                "terrain_changes": 6,
                "action_remaps": ["ud", "lr", "ud", "lr"],
                "warning_steps": 0,
                "shift_frac": 0.25,
            },
        ),
    }

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        p = self.difficulty_config.params
        n_shifts = p.get("n_shifts", 1)
        n_coins = p.get("n_coins", 3)
        walls_change = p.get("walls_change", False)
        terrain_changes = p.get("terrain_changes", 0)
        action_remaps = p.get("action_remaps", [])
        warning_steps = p.get("warning_steps", 5)
        shift_frac = p.get("shift_frac", 0.4)

        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        agent_pos = (1, 1)
        interior = [
            (x, y) for x in range(1, size - 1) for y in range(1, size - 1)
            if (x, y) != agent_pos
        ]
        rng.shuffle(interior)

        total_steps = self.get_max_steps()

        # First shift_step ends Phase 1; subsequent shifts happen in Phase 2
        first_shift = int(total_steps * shift_frac)
        if n_shifts == 1:
            shift_steps = [first_shift]
        else:
            interval = (total_steps - first_shift) // n_shifts
            shift_steps = [first_shift] + [
                first_shift + interval * (i + 1) for i in range(n_shifts - 1)
            ]

        # Goal positions: one per shift (first goal appears AT first shift, then changes)
        n_goal_positions = n_shifts + 1
        goal_sequence = interior[:n_goal_positions]

        # Phase 1: goal cell blocked so oracle cannot reach it
        # Use a BLOCKER (impassable) object at the first goal position in Phase 1
        if goal_sequence:
            gx, gy = goal_sequence[0]
            grid.objects[gy, gx] = ObjectType.BLOCKER  # blocks until shift

        # Coins for Phase 1 objective (give oracle something to do in Phase 1)
        coin_pool = interior[n_goal_positions:]
        rng.shuffle(coin_pool)
        coin_positions = coin_pool[:n_coins]
        for cx, cy in coin_positions:
            grid.objects[cy, cx] = ObjectType.COIN

        used = set(goal_sequence) | set(coin_positions) | {agent_pos}

        # Wall patches to toggle at each shift
        wall_patches = []
        if walls_change and n_shifts > 0:
            wall_cands = [p for p in interior if p not in used]
            rng.shuffle(wall_cands)
            chunk = max(1, min(3, len(wall_cands) // max(1, n_shifts)))
            for i in range(n_shifts):
                patch = wall_cands[i * chunk : (i + 1) * chunk]
                wall_patches.append(list(patch))
        else:
            wall_patches = [[] for _ in range(n_shifts)]

        # Terrain change cells (EMPTY↔HAZARD at each shift)
        terrain_cells = []
        if terrain_changes > 0:
            tc_pool = [
                p for p in interior
                if p not in used and p not in {q for patch in wall_patches for q in patch}
            ]
            rng.shuffle(tc_pool)
            terrain_cells = tc_pool[:terrain_changes]

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [goal_sequence[-1]] if goal_sequence else [],
            "goal_sequence": goal_sequence,
            "shift_steps": shift_steps,
            "wall_patches": wall_patches,
            "n_shifts": n_shifts,
            "terrain_changes": terrain_changes,
            "_terrain_cells": terrain_cells,
            "_action_remaps_schedule": action_remaps,
            "n_coins": n_coins,
            "_coin_positions": list(coin_positions),
            "warning_steps": warning_steps,
            "shift_frac": shift_frac,
            "max_steps": self.get_max_steps(),
            # backward-compat
            "goal_a": goal_sequence[0] if goal_sequence else None,
            "goal_b": goal_sequence[-1] if goal_sequence else None,
            "shift_step": shift_steps[0] if shift_steps else total_steps // 2,
        }

    def on_env_reset(self, agent, grid, config):
        config["_phase"] = 0
        config["_shifted"] = 0
        config["_action_remap"] = {}
        config["_coins_collected"] = 0
        self._config = config
        # Redraw initial state: blocker at first goal
        seq = config.get("goal_sequence", [])
        if seq:
            gx, gy = seq[0]
            if grid.objects[gy, gx] != ObjectType.BLOCKER:
                grid.objects[gy, gx] = ObjectType.BLOCKER

    def on_env_step(self, agent, grid, config, step_count):
        """Trigger shifts, place warning beacons, collect coins."""
        shift_steps = config.get("shift_steps", [])
        shifted = config.get("_shifted", 0)
        seq = config.get("goal_sequence", [])
        warning_steps = config.get("warning_steps", 5)

        # Check coin collection at agent position each step
        ax, ay = agent.position
        if grid.objects[ay, ax] == ObjectType.COIN:
            grid.objects[ay, ax] = ObjectType.NONE
            config["_coins_collected"] = config.get("_coins_collected", 0) + 1

        # Warning: place BLOCKER→TARGET flicker markers before each shift
        if warning_steps > 0 and shifted < len(shift_steps):
            steps_to_shift = shift_steps[shifted] - step_count
            if 0 < steps_to_shift <= warning_steps:
                # Flash: alternate BLOCKER/NONE at goal pos (visual warning)
                new_phase_idx = shifted + 1
                if new_phase_idx < len(seq):
                    nx, ny = seq[new_phase_idx]
                    if grid.objects[ny, nx] == ObjectType.NONE:
                        grid.objects[ny, nx] = ObjectType.TARGET  # "incoming goal" warning
                elif seq:
                    nx, ny = seq[-1]
                    if grid.objects[ny, nx] == ObjectType.NONE:
                        grid.objects[ny, nx] = ObjectType.TARGET

        # Trigger shifts
        if shifted < len(shift_steps) and step_count >= shift_steps[shifted]:
            phase = shifted

            # Remove old blocker / goal marker
            if phase < len(seq):
                ox, oy = seq[phase]
                grid.objects[oy, ox] = ObjectType.NONE

            # Place new GOAL
            new_phase = phase + 1
            if new_phase < len(seq):
                nx, ny = seq[new_phase]
                grid.objects[ny, nx] = ObjectType.GOAL
            elif seq:
                nx, ny = seq[-1]
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

            # Terrain type changes (EMPTY↔HAZARD)
            for tx, ty in config.get("_terrain_cells", []):
                if 0 < tx < grid.width - 1 and 0 < ty < grid.height - 1:
                    if grid.terrain[ty, tx] == CellType.EMPTY:
                        grid.terrain[ty, tx] = CellType.HAZARD
                    elif grid.terrain[ty, tx] == CellType.HAZARD:
                        grid.terrain[ty, tx] = CellType.EMPTY

            # Action remapping
            schedule = config.get("_action_remaps_schedule", [])
            if shifted < len(schedule):
                remap_type = schedule[shifted]
                current = dict(config.get("_action_remap", {}))
                if remap_type == "ud":
                    for k, v in _REMAP_UD.items():
                        if k in current:
                            del current[k]
                        else:
                            current[k] = v
                elif remap_type == "lr":
                    for k, v in _REMAP_LR.items():
                        if k in current:
                            del current[k]
                        else:
                            current[k] = v
                config["_action_remap"] = current

            config["_shifted"] = shifted + 1
            config["_phase"] = new_phase

    def on_agent_moved(self, pos, agent, grid):
        """Collect coins on move."""
        config = getattr(self, "_config", {})
        x, y = pos
        if grid.objects[y, x] == ObjectType.COIN:
            grid.objects[y, x] = ObjectType.NONE
            config["_coins_collected"] = config.get("_coins_collected", 0) + 1

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        config = new_state.get("config", {})

        old_coins = old_state.get("config", {}).get("_coins_collected", 0)
        new_coins = config.get("_coins_collected", 0)
        if new_coins > old_coins:
            reward += 0.3 * (new_coins - old_coins)

        if self.check_success(new_state):
            reward += 1.0
        elif "agent" in new_state:
            ax, ay = new_state["agent"].position
            ox, oy = old_state.get("agent_position", (ax, ay))
            seq = config.get("goal_sequence", [])
            phase = config.get("_phase", 0)
            if phase < len(seq):
                tgt = seq[phase]
                if config.get("_shifted", 0) > 0:
                    # Phase 2+: shape toward goal
                    reward += 0.05 * (
                        (abs(ox - tgt[0]) + abs(oy - tgt[1]))
                        - (abs(ax - tgt[0]) + abs(ay - tgt[1]))
                    )

        return reward

    def check_success(self, state):
        """Success = agent at the current real GOAL object."""
        if "grid" not in state or "agent" not in state:
            return False
        x, y = state["agent"].position
        return bool(state["grid"].objects[y, x] == ObjectType.GOAL)

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0
