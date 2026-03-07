"""RuleInduction - Pattern Discovery: infer which switches are real from terrain cues.

MECHANICS:
  - Grid has N switches (SWITCH objects): some REAL, some DECOY.
  - Activating all real switches (any order) opens a barrier DOOR to the GOAL.
  - The PATTERN distinguishing real from decoy:
      Real switches: the cell directly SOUTH has ICE terrain.
      Decoys: NO ICE directly south.
  - Easy: real switches also have a GEM placed 1 cell north (obvious cue).
  - Medium+: only ICE cue (south cell).
  - Agent uses INTERACT to toggle switches. Wrong switch (decoy) gives a heavy
    penalty AND deactivates the most recently activated real switch (progress
    reset). Decoys are single-use: after one wrong activation they become inert.
  - Barrier: a row of WALLs across the middle with one DOOR that opens when
    all real switches are activated.

DIFFICULTY:
  - easy:   9x9, 2 real + 1 decoy, GEM hints + ICE, max_steps=120
  - medium: 11x11, 3 real + 2 decoy, ICE only, max_steps=220
  - hard:   13x13, 4 real + 3 decoy, ICE only, max_steps=380
  - expert: 15x15, 5 real + 4 decoy, ICE only, max_steps=550
"""

from __future__ import annotations

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("RuleInduction-v0", tags=["reasoning", "rule_learning"])
class RuleInductionTask(TaskSpec):
    """Discover which switches are real via terrain cues, activate them all."""

    name = "RuleInduction-v0"
    description = "Infer which switches are real from terrain patterns and activate them all"
    capability_tags = ["reasoning", "rule_learning"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=9,
            max_steps=120,
            params={
                "n_real": 2,
                "n_decoy": 1,
                "gem_hints": True,
            },
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=11,
            max_steps=220,
            params={
                "n_real": 3,
                "n_decoy": 2,
                "gem_hints": False,
            },
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=13,
            max_steps=380,
            params={
                "n_real": 4,
                "n_decoy": 3,
                "gem_hints": False,
            },
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=15,
            max_steps=550,
            params={
                "n_real": 5,
                "n_decoy": 4,
                "gem_hints": False,
            },
        ),
    }

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #

    def generate(self, seed: int):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        params = self.difficulty_config.params
        n_real = params["n_real"]
        n_decoy = params["n_decoy"]
        gem_hints = params.get("gem_hints", False)
        n_switches = n_real + n_decoy

        grid = Grid(size, size)

        # Border walls
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        # ---- Barrier row ------------------------------------------------
        # Place barrier wall across the middle of the grid.
        # The switch area is the top portion (rows 1..barrier_y-1),
        # the goal area is the bottom portion (rows barrier_y+1..size-2).
        barrier_y = size // 2
        for x in range(1, size - 1):
            grid.terrain[barrier_y, x] = CellType.WALL

        # One door in the barrier (random x position)
        door_x = int(rng.integers(2, size - 2))
        grid.terrain[barrier_y, door_x] = CellType.EMPTY
        grid.objects[barrier_y, door_x] = ObjectType.DOOR
        # metadata 0 = locked; we will set >= 10 to mark open
        grid.metadata[barrier_y, door_x] = 0
        barrier_door_pos = (door_x, barrier_y)

        # ---- Goal placement (below barrier) -----------------------------
        goal_cells = [
            (x, y)
            for x in range(1, size - 1)
            for y in range(barrier_y + 1, size - 1)
            if grid.terrain[y, x] == CellType.EMPTY
        ]
        rng.shuffle(goal_cells)
        goal_pos = goal_cells[0]
        grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.GOAL

        # ---- Switch placement (above barrier) ---------------------------
        # Switches need y such that y+1 < barrier_y (room for ICE south).
        # With gem_hints, also need y-1 >= 1 (room for GEM north).
        min_switch_y = 2 if gem_hints else 1
        max_switch_y = barrier_y - 2  # south cell must be < barrier_y

        switch_candidates = [
            (x, y)
            for x in range(1, size - 1)
            for y in range(min_switch_y, max_switch_y + 1)
            if grid.terrain[y, x] == CellType.EMPTY and grid.terrain[y + 1, x] == CellType.EMPTY
        ]
        rng.shuffle(switch_candidates)

        # Pick switch positions ensuring minimum spacing of 2
        all_switch_positions: list[tuple[int, int]] = []
        used: set[tuple[int, int]] = set()

        for pos in switch_candidates:
            if len(all_switch_positions) >= n_switches:
                break
            px, py = pos
            # Ensure the south cell and (optionally) north cell are free
            south = (px, py + 1)
            north = (px, py - 1) if gem_hints else None
            if south in used or pos in used:
                continue
            if north is not None and north in used:
                continue
            # Minimum spacing of 2 from other switches
            too_close = False
            for sx, sy in all_switch_positions:
                if abs(px - sx) + abs(py - sy) < 2:
                    too_close = True
                    break
            if too_close:
                continue
            all_switch_positions.append(pos)
            used.add(pos)
            used.add(south)
            if north is not None:
                used.add(north)

        # If we couldn't place enough, relax spacing
        if len(all_switch_positions) < n_switches:
            for pos in switch_candidates:
                if len(all_switch_positions) >= n_switches:
                    break
                if pos in used:
                    continue
                px, py = pos
                south = (px, py + 1)
                if south in used:
                    continue
                all_switch_positions.append(pos)
                used.add(pos)
                used.add(south)

        # Shuffle and split into real / decoy
        rng.shuffle(all_switch_positions)
        real_switch_positions = list(all_switch_positions[:n_real])
        decoy_positions = list(all_switch_positions[n_real : n_real + n_decoy])

        # Assign activation order to real switches (random permutation)
        switch_order = list(range(n_real))
        rng.shuffle(switch_order)
        # switch_order[i] = the activation rank of real_switch_positions[i]
        # Invert: ordered_real[rank] = position
        ordered_real = [None] * n_real
        for i, rank in enumerate(switch_order):
            ordered_real[rank] = real_switch_positions[i]

        # Place SWITCH objects
        for pos in all_switch_positions:
            grid.objects[pos[1], pos[0]] = ObjectType.SWITCH

        # ---- ICE cues for real switches ---------------------------------
        for pos in real_switch_positions:
            sx, sy = pos
            # ICE directly south
            grid.terrain[sy + 1, sx] = CellType.ICE

        # ---- GEM hints for easy difficulty ------------------------------
        if gem_hints:
            for pos in real_switch_positions:
                sx, sy = pos
                north_y = sy - 1
                if north_y >= 1 and grid.objects[north_y, sx] == ObjectType.NONE:
                    grid.objects[north_y, sx] = ObjectType.GEM

        # ---- Agent start (top-left corner of switch area) ---------------
        agent_pos = (1, 1)
        # Make sure agent can reach all switches above barrier
        reachable = set()
        from collections import deque

        queue = deque([agent_pos])
        reachable.add(agent_pos)
        while queue:
            cx, cy = queue.popleft()
            for ddx, ddy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx2, ny2 = cx + ddx, cy + ddy
                if (
                    (nx2, ny2) not in reachable
                    and 0 < nx2 < size - 1
                    and 0 < ny2 < size - 1
                    and grid.terrain[ny2, nx2] not in (CellType.WALL,)
                    and grid.objects[ny2, nx2] != ObjectType.DOOR
                ):
                    reachable.add((nx2, ny2))
                    queue.append((nx2, ny2))

        # Verify all switch positions reachable from agent
        all_reachable = all(p in reachable for p in all_switch_positions)
        if not all_reachable:
            # Fallback: clear any blocking ICE (ICE is walkable anyway)
            pass

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [goal_pos],
            "real_switch_positions": [list(p) for p in ordered_real],
            "decoy_positions": [list(p) for p in decoy_positions],
            "switch_order": switch_order,
            "barrier_door_pos": list(barrier_door_pos),
            "n_real": n_real,
            "n_decoy": n_decoy,
            "max_steps": self.get_max_steps(),
        }

    # ------------------------------------------------------------------ #
    # Runtime hooks
    # ------------------------------------------------------------------ #

    def on_env_reset(self, agent, grid, config):
        n_real = config["n_real"]
        config["_activated"] = [False] * n_real
        config["_activation_order"] = []  # tracks order real switches were activated
        config["_wrong_attempts"] = 0
        config["_door_opened"] = False
        config["_goal_reached"] = False
        self._prev_activated_count = 0
        self._prev_wrong = 0
        self._config = config

    def on_agent_interact(self, pos, agent, grid):
        """Handle INTERACT on switch positions."""
        config = getattr(self, "_config", {})
        x, y = pos

        # Only process if standing on a SWITCH
        if grid.objects[y, x] != ObjectType.SWITCH:
            return

        # Single-use decoys: metadata==99 means already used, ignore
        if int(grid.metadata[y, x]) == 99:
            return

        real_positions = [tuple(p) for p in config.get("real_switch_positions", [])]
        decoy_positions = [tuple(p) for p in config.get("decoy_positions", [])]
        activated = config.get("_activated", [])
        activation_order = config.get("_activation_order", [])

        pos_tuple = (x, y)

        if pos_tuple in decoy_positions:
            # ---- Decoy switch activated: heavy penalty + progress reset ----
            # 1. Mark decoy as used (metadata=99) so it cannot be re-activated
            grid.metadata[y, x] = 99

            # 2. Find most recently activated real switch and deactivate it
            if activation_order:
                last_idx = activation_order.pop()
                activated[last_idx] = False
                config["_activated"] = activated

                # Reset the deactivated switch's visual metadata back to its index
                rpos = real_positions[last_idx]
                rx, ry = rpos
                grid.metadata[ry, rx] = last_idx

                # 3. If barrier was open and not all switches active, close it
                if config.get("_door_opened", False) and not all(activated):
                    self._close_barrier(grid, config)

            # 4. Increment wrong attempts
            config["_wrong_attempts"] = config.get("_wrong_attempts", 0) + 1
            return

        if pos_tuple in real_positions:
            idx = real_positions.index(pos_tuple)

            # Already activated -- no-op
            if activated[idx]:
                return

            # Correct activation (any order is valid)
            activated[idx] = True
            config["_activated"] = activated
            activation_order.append(idx)
            config["_activation_order"] = activation_order
            grid.metadata[y, x] = 100  # mark as activated

            # Check if all real switches are now active
            if all(activated):
                self._open_barrier(grid, config)

    def _open_barrier(self, grid, config):
        """Open the barrier door."""
        dx, dy = config["barrier_door_pos"]
        # Mark door as open via metadata >= 10
        grid.metadata[dy, dx] = 10
        config["_door_opened"] = True

    def _close_barrier(self, grid, config):
        """Close the barrier door (after a decoy penalty deactivated a real switch)."""
        dx, dy = config["barrier_door_pos"]
        grid.metadata[dy, dx] = 0
        config["_door_opened"] = False

    def can_agent_enter(self, pos, agent, grid) -> bool:
        """Block passage through locked door and non-walkable objects."""
        x, y = pos
        if grid.objects[y, x] == ObjectType.DOOR:
            return int(grid.metadata[y, x]) >= 10
        if grid.is_object_blocking(pos):
            return False
        return True

    def on_agent_moved(self, pos, agent, grid):
        """Check if agent reached the goal."""
        config = getattr(self, "_config", {})
        x, y = pos
        if grid.objects[y, x] == ObjectType.GOAL and config.get("_door_opened", False):
            config["_goal_reached"] = True

    # ------------------------------------------------------------------ #
    # Reward
    # ------------------------------------------------------------------ #

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01  # step penalty
        config = new_state.get("config", {})

        activated = config.get("_activated", [])
        n_activated = sum(activated)
        wrong = config.get("_wrong_attempts", 0)

        # +0.3 per newly activated real switch
        delta_activated = n_activated - self._prev_activated_count
        if delta_activated > 0:
            reward += 0.3 * delta_activated
        elif delta_activated < 0:
            # A real switch was deactivated by decoy penalty — no extra penalty
            # here (the -0.5 wrong penalty below covers it), but update tracker.
            pass
        self._prev_activated_count = n_activated

        # -0.5 per new wrong attempt (decoy activation)
        delta_wrong = wrong - self._prev_wrong
        if delta_wrong > 0:
            reward -= 0.5 * delta_wrong
        self._prev_wrong = wrong

        # Approach shaping
        if "agent" in new_state:
            ax, ay = new_state["agent"].position
            ox, oy = old_state.get("agent_position", (ax, ay))

            door_opened = config.get("_door_opened", False)

            if not door_opened:
                # Guide toward the next real switch to activate
                real_positions = config.get("real_switch_positions", [])
                next_target = None
                for i, done in enumerate(activated):
                    if not done:
                        next_target = tuple(real_positions[i])
                        break
                if next_target is not None:
                    tx, ty = next_target
                    d_new = abs(ax - tx) + abs(ay - ty)
                    d_old = abs(ox - tx) + abs(oy - ty)
                    reward += 0.05 * (d_old - d_new)
            else:
                # Guide toward the goal
                goal_positions = config.get("goal_positions", [])
                if goal_positions:
                    gx, gy = goal_positions[0]
                    d_new = abs(ax - gx) + abs(ay - gy)
                    d_old = abs(ox - gx) + abs(oy - gy)
                    reward += 0.05 * (d_old - d_new)

        # Success bonus
        if self.check_success(new_state):
            reward += 1.0

        return reward

    # ------------------------------------------------------------------ #
    # Termination
    # ------------------------------------------------------------------ #

    def check_success(self, state):
        config = state.get("config", {})
        return bool(config.get("_goal_reached", False))

    def check_done(self, state):
        return self.check_success(state)

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0
