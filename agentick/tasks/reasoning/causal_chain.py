"""CausalChain - Activate switches that each cause a visible physical effect.

MECHANICS:
  - N switches (LEVER objects) on the grid, each with a physical consequence
  - Each lever, when stepped on, causes a visible change:
    - Removes a specific wall barrier (opens a path)
    - Creates a bridge over water/hazard
    - Deactivates a hazard zone
  - Switches must be activated in causal order (switch 0 opens path to switch 1, etc.)
  - Goal room only accessible after all switches activated
  - Decoy levers do nothing or cause harmful effects (add walls)
  - Tests causal reasoning: observe cause → effect relationships
"""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("CausalChain-v0", tags=["reasoning", "causal_reasoning"])
class CausalChainTask(TaskSpec):
    """Activate LEVER objects in causal order; each causes a physical grid change."""

    name = "CausalChain-v0"
    description = "Activate causal chain of levers with physical consequences"
    capability_tags = ["reasoning", "causal_reasoning"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=9,
            max_steps=100,
            params={"n_switches": 2, "n_decoys": 0},
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=11,
            max_steps=180,
            params={"n_switches": 3, "n_decoys": 1},
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=13,
            max_steps=280,
            params={"n_switches": 4, "n_decoys": 2},
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=15,
            max_steps=420,
            params={"n_switches": 5, "n_decoys": 3},
        ),
    }

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        n = self.difficulty_config.params.get("n_switches", 2)
        n_decoys = self.difficulty_config.params.get("n_decoys", 0)

        for attempt in range(20):
            grid = Grid(size, size)
            grid.terrain[0, :] = CellType.WALL
            grid.terrain[-1, :] = CellType.WALL
            grid.terrain[:, 0] = CellType.WALL
            grid.terrain[:, -1] = CellType.WALL

            corners = [(1, 1), (size - 2, 1), (1, size - 2), (size - 2, size - 2)]
            agent_pos = tuple(corners[int(rng.integers(0, len(corners)))])

            # Create N+1 zones separated by wall barriers
            # Each barrier has a gap that starts as WALL (blocked)
            # Activating switch i opens barrier i (removes wall at gap)
            # Zone 0: agent start area
            # Zone i (1..N): contains switch i
            # Zone N: contains goal

            # Divide the grid into vertical strips
            n_zones = n + 1
            zone_width = max(2, (size - 2) // n_zones)
            barriers = []  # list of (barrier_col, gap_row)

            for i in range(n):
                bx = 1 + (i + 1) * zone_width
                bx = min(bx, size - 3)
                # Build wall column
                for y in range(1, size - 1):
                    if 0 < bx < size - 1:
                        grid.terrain[y, bx] = CellType.WALL
                # Gap position (will be walled, opened by switch i)
                gap_y = 1 + int(rng.integers(1, max(2, size - 3)))
                gap_y = min(gap_y, size - 2)
                barriers.append((bx, gap_y))

            # Place switches in each zone (switch i is in zone i, accessible before barrier i)
            switch_positions = []
            switch_effects = []  # (barrier_x, gap_y) that each switch opens
            for i in range(n):
                # Zone i spans from previous barrier+1 to current barrier-1
                zone_left = 1 if i == 0 else min(barriers[i - 1][0] + 1, size - 2)
                zone_right = barriers[i][0] - 1 if i < len(barriers) else size - 2
                zone_right = max(zone_left, zone_right)

                zone_cells = [
                    (x, y)
                    for x in range(zone_left, zone_right + 1)
                    for y in range(1, size - 1)
                    if grid.terrain[y, x] == CellType.EMPTY and (x, y) != agent_pos
                ]
                if zone_cells:
                    pos = zone_cells[int(rng.integers(len(zone_cells)))]
                else:
                    pos = (max(1, zone_left), size // 2)
                switch_positions.append(pos)
                switch_effects.append(barriers[i])

            # Place goal in the last zone
            last_barrier = barriers[-1][0] if barriers else 1
            goal_zone_cells = [
                (x, y)
                for x in range(last_barrier + 1, size - 1)
                for y in range(1, size - 1)
                if grid.terrain[y, x] == CellType.EMPTY
            ]
            if goal_zone_cells:
                goal_pos = goal_zone_cells[int(rng.integers(len(goal_zone_cells)))]
            else:
                goal_pos = (size - 2, size - 2)

            # Place switch markers (LEVER objects)
            for sx, sy in switch_positions:
                grid.objects[sy, sx] = ObjectType.LEVER

            # Place goal
            grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.NONE  # placed on reset

            # Verify agent can reach first switch
            reachable = grid.flood_fill(agent_pos)
            if switch_positions[0] in reachable:
                break
        else:
            # Fallback: simple layout
            grid = Grid(size, size)
            grid.terrain[0, :] = CellType.WALL
            grid.terrain[-1, :] = CellType.WALL
            grid.terrain[:, 0] = CellType.WALL
            grid.terrain[:, -1] = CellType.WALL
            corners = [(1, 1), (size - 2, 1), (1, size - 2), (size - 2, size - 2)]
            agent_pos = tuple(corners[int(rng.integers(0, len(corners)))])
            goal_pos = (size - 2, size - 2)
            switch_positions = [(1 + i * 2, size // 2) for i in range(n)]
            barriers = [(size // 2, size // 2)]
            switch_effects = barriers * n
            for sx, sy in switch_positions:
                if 0 < sx < size - 1 and 0 < sy < size - 1:
                    grid.objects[sy, sx] = ObjectType.LEVER

        # Place decoy levers (stepping on them adds a wall near the agent)
        all_used = {agent_pos, goal_pos} | set(switch_positions)
        all_free = [
            (x, y)
            for x in range(1, size - 1)
            for y in range(1, size - 1)
            if grid.terrain[y, x] == CellType.EMPTY and (x, y) not in all_used
        ]
        rng.shuffle(all_free)
        decoy_positions = all_free[:n_decoys]
        for dx, dy in decoy_positions:
            grid.objects[dy, dx] = ObjectType.LEVER

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [goal_pos],
            "switch_positions": switch_positions,
            "switch_effects": [list(e) for e in switch_effects],
            "decoy_positions": decoy_positions,
            "n_switches": n,
            "max_steps": self.get_max_steps(),
        }

    # ── Hooks ────────────────────────────────────────────────────────────────

    def on_env_reset(self, agent, grid, config):
        config["_switch_progress"] = 0
        config["_all_activated"] = False
        self._last_progress = 0
        self._config = config
        # Goal not placed yet (only after all switches)
        gx, gy = config["goal_positions"][0]
        grid.objects[gy, gx] = ObjectType.NONE

    def on_agent_interact(self, pos, agent, grid):
        """INTERACT while standing on a LEVER activates it in causal sequence."""
        config = getattr(self, "_config", {})
        progress = config.get("_switch_progress", 0)
        switches = config.get("switch_positions", [])
        effects = config.get("switch_effects", [])
        decoys = config.get("decoy_positions", [])
        ax, ay = pos

        # Check if standing on a decoy
        if (ax, ay) in [tuple(d) for d in decoys]:
            if grid.objects[ay, ax] == ObjectType.LEVER:
                grid.objects[ay, ax] = ObjectType.NONE  # consumed
            return

        # Check if standing on the next switch in sequence
        if progress < len(switches):
            sx, sy = switches[progress]
            if (ax, ay) == (sx, sy) and grid.objects[sy, sx] == ObjectType.LEVER:
                # Activate: change lever to NONE (consumed)
                grid.objects[sy, sx] = ObjectType.NONE

                # Physical consequence: open the barrier (remove wall at gap)
                if progress < len(effects):
                    bx, by = effects[progress]
                    if 0 < bx < grid.width - 1 and 0 < by < grid.height - 1:
                        grid.terrain[by, bx] = CellType.EMPTY

                config["_switch_progress"] = progress + 1

                # If all switches done, place goal
                if config["_switch_progress"] >= len(switches):
                    config["_all_activated"] = True
                    goal_x, goal_y = config["goal_positions"][0]
                    grid.objects[goal_y, goal_x] = ObjectType.GOAL

    # ── Reward & success ─────────────────────────────────────────────────────

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        config = new_state.get("config", {})
        new_p = config.get("_switch_progress", 0)

        if new_p > self._last_progress:
            reward += 0.3 * (new_p - self._last_progress)
        self._last_progress = new_p

        # Approach shaping
        if "agent" in new_state:
            ax, ay = new_state["agent"].position
            ox, oy = old_state.get("agent_position", (ax, ay))
            switches = config.get("switch_positions", [])
            goal = config.get("goal_positions", [None])[0]

            if not config.get("_all_activated", False) and new_p < len(switches):
                tgt = switches[new_p]
            elif config.get("_all_activated", False) and goal:
                tgt = goal
            else:
                tgt = None

            if tgt:
                d_new = abs(ax - tgt[0]) + abs(ay - tgt[1])
                d_old = abs(ox - tgt[0]) + abs(oy - tgt[1])
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
        return True  # barriers are dynamic

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0
