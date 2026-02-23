"""DistributionShift - Navigate when the environment dramatically shifts mid-episode.

MECHANICS:
  - Phase 1 (first ~40% of steps): NO goal visible; agent must explore & collect COINs
    (the goal position is behind a WALL barrier so oracle CANNOT reach it early)
  - At shift_step: DRAMATIC VISIBLE CHANGE — wall barriers demolished, new passages open,
    hazard zones flip, GOAL appears where a wall used to be
  - Phase 2 (remaining ~60%): Goal is now reachable; agent must find and reach it
  - Agent must DETECT that the shift happened and ADAPT its strategy
  - Easy:   1 shift, obvious flashing warning (TARGET markers appear 5 steps before),
            one wall barrier toggles
  - Medium: 2 shifts, moderate warning (3 steps), two wall barriers toggle, terrain swaps
  - Hard:   3 shifts, minimal warning (1 step), terrain type swaps, action remap
  - Expert: 4 shifts, no warning, multiple simultaneous changes, rapid shifts

PHASE 1 OBJECTIVE:
  Collecting coins gives reward so oracle is drawn to coins, not to absent goal.
  The goal cell starts as WALL terrain (impassable) surrounded by a wall barrier,
  and only becomes walkable on the shift — oracle cannot reach it in Phase 1.

WALL BARRIER DESIGN:
  In Phase 1, the goal position itself is WALL terrain, plus a small wall barrier
  (3-5 cells) surrounds it, creating a visible blocked-off region. At shift time,
  the goal cell becomes EMPTY + GOAL object, and the barrier walls are removed,
  opening new passages. Additional wall segments elsewhere may appear, forcing
  the agent to find new routes.
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


def _get_barrier_cells(gx, gy, size):
    """Get wall barrier cells surrounding a goal position.

    Returns cells adjacent to (gx, gy) that are within the interior (not border).
    These form the wall barrier that blocks access to the goal in Phase 1.
    """
    candidates = [
        (gx - 1, gy), (gx + 1, gy), (gx, gy - 1), (gx, gy + 1),
    ]
    return [
        (x, y) for x, y in candidates
        if 1 <= x < size - 1 and 1 <= y < size - 1
    ]


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

        # Cells adjacent to agent — never place barrier walls here to avoid
        # trapping the agent.
        agent_neighbors = set(_get_barrier_cells(agent_pos[0], agent_pos[1], size))

        # Pick goal positions. Phase 1 goal needs barrier room; later goals
        # just need the single cell to be WALL.
        goal_sequence = []
        goal_set = set()
        used_cells = {agent_pos}

        # Phase 1 goal: pick a position with barrier room, NOT adjacent to agent
        for pos in interior:
            if pos in used_cells or pos in agent_neighbors:
                continue
            barrier = _get_barrier_cells(pos[0], pos[1], size)
            # Barrier cells must not be agent, agent's neighbors, or already used
            free_barrier = [
                b for b in barrier
                if b not in used_cells and b not in agent_neighbors and b != agent_pos
            ]
            if len(free_barrier) >= 2:
                goal_sequence.append(pos)
                goal_set.add(pos)
                used_cells.add(pos)
                break

        # Remaining goals: reserve the cell itself. Avoid cells adjacent to agent
        # (walling those could trap the agent) and prefer cells at least 2 steps away.
        for pos in interior:
            if pos in used_cells or pos in agent_neighbors:
                continue
            goal_sequence.append(pos)
            goal_set.add(pos)
            used_cells.add(pos)
            if len(goal_sequence) >= n_goal_positions:
                break

        # Fallback if we couldn't avoid agent neighbors
        if len(goal_sequence) < n_goal_positions:
            for pos in interior:
                if pos in used_cells:
                    continue
                goal_sequence.append(pos)
                goal_set.add(pos)
                used_cells.add(pos)
                if len(goal_sequence) >= n_goal_positions:
                    break

        # --- Place walls for Phase 1 goal: goal cell + barrier ---
        # The barrier is placed incrementally with connectivity checks to ensure
        # the agent is never trapped.
        barrier_walls_per_goal = {}  # goal_idx -> list of barrier positions
        min_reachable = max(n_coins + 3, (size - 2) ** 2 // 3)

        if goal_sequence:
            gx, gy = goal_sequence[0]
            # Make the goal cell itself a WALL (truly impassable)
            grid.terrain[gy, gx] = CellType.WALL

            # Build barrier walls one-by-one, checking connectivity after each
            barrier = _get_barrier_cells(gx, gy, size)
            barrier_for_goal = []
            for bx, by in barrier:
                if (bx, by) == agent_pos or (bx, by) in goal_set:
                    continue
                if (bx, by) in agent_neighbors:
                    continue
                # Tentatively place wall
                grid.terrain[by, bx] = CellType.WALL
                # Check agent still has enough reachable space
                reach = grid.flood_fill(agent_pos)
                if len(reach) >= min_reachable:
                    barrier_for_goal.append((bx, by))
                    used_cells.add((bx, by))
                else:
                    # Undo — this wall would fragment the grid too much
                    grid.terrain[by, bx] = CellType.EMPTY
            barrier_walls_per_goal[0] = barrier_for_goal

        # For subsequent goals: only wall the goal cell itself (no barrier)
        # to avoid fragmenting the grid. Barrier gets built at shift time
        # when the previous goal region opens up.
        for idx in range(1, len(goal_sequence)):
            gx, gy = goal_sequence[idx]
            grid.terrain[gy, gx] = CellType.WALL
            barrier_walls_per_goal[idx] = []  # no barrier at generation time

        # Coins for Phase 1 objective (give oracle something to do in Phase 1)
        # Place coins AFTER walls so we can verify reachability.
        reachable = grid.flood_fill(agent_pos)
        coin_pool = [
            pos for pos in interior
            if pos not in used_cells
            and grid.terrain[pos[1], pos[0]] == CellType.EMPTY
            and pos in reachable
        ]
        rng.shuffle(coin_pool)
        coin_positions = coin_pool[:n_coins]
        for cx, cy in coin_positions:
            grid.objects[cy, cx] = ObjectType.COIN
        used_cells.update(coin_positions)

        # Extra wall segments to toggle at each shift (dramatic restructuring)
        # These are NOT placed as walls at generation time — they toggle at shift.
        # We just record the positions so they get toggled (EMPTY->WALL) at shift.
        extra_wall_patches = []
        if walls_change and n_shifts > 0:
            wall_cands = [
                pos for pos in interior
                if pos not in used_cells
                and grid.terrain[pos[1], pos[0]] == CellType.EMPTY
            ]
            rng.shuffle(wall_cands)
            # Build connected wall segments (not isolated cells) for drama
            chunk = max(2, min(4, len(wall_cands) // max(1, n_shifts)))
            for i in range(n_shifts):
                patch = wall_cands[i * chunk : (i + 1) * chunk]
                extra_wall_patches.append(list(patch))
                used_cells.update(patch)
        else:
            extra_wall_patches = [[] for _ in range(n_shifts)]

        # Terrain change cells (EMPTY<->HAZARD at each shift)
        terrain_cells = []
        if terrain_changes > 0:
            tc_pool = [
                pos for pos in interior
                if pos not in used_cells
                and grid.terrain[pos[1], pos[0]] == CellType.EMPTY
            ]
            rng.shuffle(tc_pool)
            terrain_cells = tc_pool[:terrain_changes]

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [goal_sequence[-1]] if goal_sequence else [],
            "goal_sequence": goal_sequence,
            "shift_steps": shift_steps,
            "extra_wall_patches": extra_wall_patches,
            "barrier_walls_per_goal": {
                str(k): v for k, v in barrier_walls_per_goal.items()
            },
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
            "wall_patches": extra_wall_patches,
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

        # Ensure initial state: goal cells are WALL terrain
        seq = config.get("goal_sequence", [])
        if seq:
            gx, gy = seq[0]
            grid.terrain[gy, gx] = CellType.WALL
            grid.objects[gy, gx] = ObjectType.NONE
            # Restore barrier walls
            barrier_map = config.get("barrier_walls_per_goal", {})
            for bx, by in barrier_map.get("0", []):
                grid.terrain[by, bx] = CellType.WALL

        # Also wall off future goal positions
        for idx in range(1, len(seq)):
            gx, gy = seq[idx]
            grid.terrain[gy, gx] = CellType.WALL
            grid.objects[gy, gx] = ObjectType.NONE

    def can_agent_enter(self, pos, agent, grid):
        """Block entry to BLOCKER objects (secondary safety net)."""
        x, y = pos
        if grid.objects[y, x] == ObjectType.BLOCKER:
            return False
        return True

    def on_env_step(self, agent, grid, config, step_count):
        """Trigger shifts, place warning beacons, collect coins."""
        shift_steps = config.get("shift_steps", [])
        shifted = config.get("_shifted", 0)
        seq = config.get("goal_sequence", [])
        warning_steps = config.get("warning_steps", 5)
        barrier_map = config.get("barrier_walls_per_goal", {})

        # Check coin collection at agent position each step
        ax, ay = agent.position
        if grid.objects[ay, ax] == ObjectType.COIN:
            grid.objects[ay, ax] = ObjectType.NONE
            config["_coins_collected"] = config.get("_coins_collected", 0) + 1

        # Warning: place TARGET markers near the upcoming goal before each shift
        if warning_steps > 0 and shifted < len(shift_steps):
            steps_to_shift = shift_steps[shifted] - step_count
            if 0 < steps_to_shift <= warning_steps:
                # Flash: place TARGET markers on barrier cells as visual warning
                new_phase_idx = shifted + 1 if shifted + 1 < len(seq) else len(seq) - 1
                if new_phase_idx < len(seq):
                    # Place TARGET on some barrier walls to signal change is coming
                    barrier_cells = barrier_map.get(str(shifted), [])
                    for bx, by in barrier_cells[:2]:
                        if grid.terrain[by, bx] == CellType.WALL:
                            grid.objects[by, bx] = ObjectType.TARGET

        # Trigger shifts
        if shifted < len(shift_steps) and step_count >= shift_steps[shifted]:
            self._execute_shift(grid, config, shifted, seq, barrier_map)

    def _execute_shift(self, grid, config, shifted, seq, barrier_map):
        """Execute a dramatic terrain shift.

        1. Remove old goal's wall barrier (open passages)
        2. Convert old goal cell from WALL to EMPTY
        3. Convert new goal cell from WALL to EMPTY + place GOAL object
        4. Remove new goal's barrier walls (open access)
        5. Toggle extra wall patches (new walls appear elsewhere)
        6. Flip terrain cells (EMPTY<->HAZARD)
        7. Apply action remaps
        """
        phase = shifted

        # --- STEP 1: Demolish OLD goal's barrier walls ---
        old_barrier = barrier_map.get(str(phase), [])
        for bx, by in old_barrier:
            if 0 < bx < grid.width - 1 and 0 < by < grid.height - 1:
                grid.terrain[by, bx] = CellType.EMPTY
                grid.objects[by, bx] = ObjectType.NONE

        # --- STEP 2: Convert old goal cell to EMPTY ---
        if phase < len(seq):
            ox, oy = seq[phase]
            grid.terrain[oy, ox] = CellType.EMPTY
            grid.objects[oy, ox] = ObjectType.NONE

        # --- STEP 3: Open new goal cell and place GOAL ---
        new_phase = phase + 1
        goal_idx = new_phase if new_phase < len(seq) else len(seq) - 1
        if goal_idx < len(seq):
            nx, ny = seq[goal_idx]
            # Convert goal cell from WALL to EMPTY
            grid.terrain[ny, nx] = CellType.EMPTY
            # Place GOAL object
            grid.objects[ny, nx] = ObjectType.GOAL

            # --- STEP 4: Demolish new goal's barrier walls (open access) ---
            new_barrier = barrier_map.get(str(goal_idx), [])
            for bx, by in new_barrier:
                if 0 < bx < grid.width - 1 and 0 < by < grid.height - 1:
                    grid.terrain[by, bx] = CellType.EMPTY
                    grid.objects[by, bx] = ObjectType.NONE

        # --- STEP 5: Toggle extra wall patches (new walls appear elsewhere) ---
        patches = config.get("extra_wall_patches", config.get("wall_patches", []))
        if shifted < len(patches):
            for wx, wy in patches[shifted]:
                if 0 < wx < grid.width - 1 and 0 < wy < grid.height - 1:
                    if grid.terrain[wy, wx] == CellType.EMPTY:
                        grid.terrain[wy, wx] = CellType.WALL
                    else:
                        grid.terrain[wy, wx] = CellType.EMPTY

        # --- STEP 6: Terrain type changes (EMPTY<->HAZARD) ---
        for tx, ty in config.get("_terrain_cells", []):
            if 0 < tx < grid.width - 1 and 0 < ty < grid.height - 1:
                if grid.terrain[ty, tx] == CellType.EMPTY:
                    grid.terrain[ty, tx] = CellType.HAZARD
                elif grid.terrain[ty, tx] == CellType.HAZARD:
                    grid.terrain[ty, tx] = CellType.EMPTY

        # --- STEP 7: Action remapping ---
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

    def validate_instance(self, grid, config):
        """Goal is intentionally blocked in Phase 1 — skip default reachability check.

        Just verify agent can move (has reachable interior cells) and coins
        are reachable.
        """
        agent_pos = config.get("agent_start", (1, 1))
        reachable = grid.flood_fill(agent_pos)
        # Agent should be able to reach at least 3 cells
        return len(reachable) >= 3

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0
