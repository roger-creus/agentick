"""LightsOut - Toggle lights to turn them all off.

MECHANICS:
  - N lights (SWITCH objects) placed on the grid
  - Stepping ON a SWITCH toggles it (on→off, off→on)
  - Adjacent lights also toggle (classic Lights Out puzzle)
  - Success = ALL lights are off (no SWITCH objects remain on grid)
  - Agent can move freely; toggles happen by stepping
"""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("LightsOut-v0", tags=["combinatorial_logic"])
class LightsOutTask(TaskSpec):
    """Toggle all lights off — stepping on a light toggles it and its neighbors."""

    name = "LightsOut-v0"
    description = "Toggle all lights off"
    capability_tags = ["combinatorial_logic"]

    difficulty_configs = {
        # n_lights: lit switches to turn off | adjacent: toggling propagates to neighbors
        # n_decoys: extra switches that look lit but reset if not in sequence | walls: obstacles
        "easy": DifficultyConfig(
            name="easy",
            grid_size=7,
            max_steps=60,
            params={"n_lights": 3, "adjacent": False, "n_decoys": 0, "n_walls": 0},
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=9,
            max_steps=100,
            params={"n_lights": 5, "adjacent": True, "n_decoys": 1, "n_walls": 2},
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=11,
            max_steps=160,
            params={"n_lights": 7, "adjacent": True, "n_decoys": 2, "n_walls": 4},
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=13,
            max_steps=250,
            params={"n_lights": 9, "adjacent": True, "n_decoys": 3, "n_walls": 6},
        ),
    }

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        n_lights = self.difficulty_config.params.get("n_lights", 3)
        n_decoys = self.difficulty_config.params.get("n_decoys", 0)
        n_walls = self.difficulty_config.params.get("n_walls", 0)
        adjacent = self.difficulty_config.params.get("adjacent", False)

        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        agent_pos = (1, 1)

        free = [
            (x, y) for x in range(1, size - 1) for y in range(1, size - 1) if (x, y) != agent_pos
        ]
        rng.shuffle(free)
        light_positions = free[: min(n_lights, len(free))]
        used = set(light_positions) | {agent_pos}

        # Decoy SWITCHes: extra lights — agent must toggle all SWITCHes to win
        decoy_positions = []
        for p in free[n_lights:]:
            if len(decoy_positions) >= n_decoys:
                break
            if p not in used:
                decoy_positions.append(p)
                used.add(p)

        # Interior walls — flood-fill to keep all lights reachable
        wall_positions = []
        wall_candidates = [p for p in free if p not in used]
        for p in wall_candidates:
            if len(wall_positions) >= n_walls:
                break
            wx, wy = p
            grid.terrain[wy, wx] = CellType.WALL
            reachable = grid.flood_fill(agent_pos)
            all_reach = all(lp in reachable for lp in light_positions + decoy_positions)
            if all_reach:
                wall_positions.append(p)
                used.add(p)
            else:
                grid.terrain[wy, wx] = CellType.EMPTY

        for lx, ly in light_positions:
            grid.objects[ly, lx] = ObjectType.SWITCH
        for dx, dy in decoy_positions:
            grid.objects[dy, dx] = ObjectType.SWITCH  # decoys look identical to lights

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [],
            "light_positions": light_positions,
            "decoy_positions": decoy_positions,
            "adjacent_toggle": adjacent,
            "max_steps": self.get_max_steps(),
        }

    # ── Toggle mechanic ───────────────────────────────────────────────────────

    def on_env_reset(self, agent, grid, config):
        """Cache config and count initial lights for reward tracking."""
        self._adjacent_toggle = config.get("adjacent_toggle", False)
        # Build set of all light grid positions (lights + decoys form the toggle grid)
        self._light_grid = set()
        for lx, ly in config.get("light_positions", []):
            self._light_grid.add((lx, ly))
        for dx, dy in config.get("decoy_positions", []):
            self._light_grid.add((dx, dy))
        self._lights_remaining = sum(
            1
            for y in range(grid.height)
            for x in range(grid.width)
            if grid.objects[y, x] == ObjectType.SWITCH
        )
        self._lights_remaining_last = self._lights_remaining
        # Set metadata for visual rendering: lit cells = 1 (bright yellow)
        self._update_light_metadata(grid)

    def _update_light_metadata(self, grid):
        """Update metadata layer for lit/unlit rendering."""
        for lx, ly in self._light_grid:
            if grid.objects[ly, lx] == ObjectType.SWITCH:
                grid.metadata[ly, lx] = 1  # META_LIT: bright yellow
            else:
                grid.metadata[ly, lx] = 2  # META_LIGHT_POS: dark gray (unlit)

    def on_agent_moved(self, pos, agent, grid):
        """Toggle lights immediately on step — fires BEFORE reward computation."""
        x, y = pos
        self._toggle_cell(x, y, grid)

        # Adjacent toggle mode
        if self._adjacent_toggle:
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if 0 < nx < grid.width - 1 and 0 < ny < grid.height - 1:
                    self._toggle_cell(nx, ny, grid)

        # Update visual metadata after toggling
        self._update_light_metadata(grid)

    def _toggle_cell(self, x, y, grid):
        """Toggle a single cell: ON→OFF or OFF→ON, but only at light grid positions."""
        if grid.objects[y, x] == ObjectType.SWITCH:
            grid.objects[y, x] = ObjectType.NONE
            self._lights_remaining -= 1
        elif (x, y) in self._light_grid and grid.objects[y, x] == ObjectType.NONE:
            grid.objects[y, x] = ObjectType.SWITCH
            self._lights_remaining += 1

    # ── Reward & success ─────────────────────────────────────────────────────

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        # Lights toggled in on_agent_moved (before this runs) — use instance counter
        old_rem = self._lights_remaining_last
        new_rem = self._lights_remaining
        if new_rem < old_rem:
            reward += 0.3 * (old_rem - new_rem)  # reward per light turned off
        self._lights_remaining_last = new_rem
        # Approach shaping: toward nearest remaining SWITCH
        if "agent_position" in new_state and "grid" in new_state:
            ax, ay = new_state["agent_position"]
            ox, oy = old_state.get("agent_position", (ax, ay))
            g = new_state["grid"]
            lights = [
                (x, y)
                for y in range(g.height)
                for x in range(g.width)
                if g.objects[y, x] == ObjectType.SWITCH
            ]
            if lights:
                d_new = min(abs(ax - lx) + abs(ay - ly) for lx, ly in lights)
                d_old = min(abs(ox - lx) + abs(oy - ly) for lx, ly in lights)
                reward += 0.05 * (d_old - d_new)
        if self.check_success(new_state):
            reward += 1.0
        return reward

    def check_success(self, state):
        """All lights must be off (no SWITCH objects on grid)."""
        if "grid" not in state:
            return False
        grid = state["grid"]
        for y in range(grid.height):
            for x in range(grid.width):
                if grid.objects[y, x] == ObjectType.SWITCH:
                    return False
        return True

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0
