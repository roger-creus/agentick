"""TreasureHunt task - Find invisible treasures by reading directional scroll clues."""

from __future__ import annotations

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task

# Direction constants matching Direction enum: 0=N, 1=E, 2=S, 3=W
_DIR_DELTAS = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}


@register_task("TreasureHunt-v0", tags=["exploration", "memory", "reasoning"])
class TreasureHuntTask(TaskSpec):
    """Find invisible treasures by reading directional SCROLL clues.

    Treasures are hidden (not rendered on the grid). SCROLL objects are
    scattered around the map. Stepping on a scroll reads it (consuming it)
    and reveals a directional clue — the scroll's metadata encodes the
    direction (N/E/S/W) and distance to the nearest hidden treasure as
    ``direction * 10 + min(distance, 9)``.

    At harder difficulties some scrolls are *misleading*: they point in a
    random wrong direction.

    The agent must read clues, triangulate treasure positions, and step on
    the exact cell to collect each hidden treasure.
    """

    name = "TreasureHunt-v0"
    description = "Find invisible treasures by reading directional scroll clues"
    capability_tags = ["exploration", "memory", "reasoning"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=9,
            max_steps=60,
            params={
                "n_treasures": 2,
                "n_clues": 6,
                "n_misleading": 0,
                "wall_density": 0.05,
            },
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=12,
            max_steps=80,
            params={
                "n_treasures": 3,
                "n_clues": 8,
                "n_misleading": 1,
                "wall_density": 0.08,
            },
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=15,
            max_steps=120,
            params={
                "n_treasures": 4,
                "n_clues": 10,
                "n_misleading": 3,
                "wall_density": 0.10,
            },
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=18,
            max_steps=180,
            params={
                "n_treasures": 5,
                "n_clues": 12,
                "n_misleading": 5,
                "wall_density": 0.12,
            },
        ),
    }

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        p = self.difficulty_config.params or {}
        n_treasures = p.get("n_treasures", 2)
        n_clues = p.get("n_clues", 6)
        n_misleading = p.get("n_misleading", 0)
        wall_density = p.get("wall_density", 0.05)

        for _attempt in range(40):
            grid = Grid(size, size)
            # Border walls
            grid.terrain[0, :] = CellType.WALL
            grid.terrain[-1, :] = CellType.WALL
            grid.terrain[:, 0] = CellType.WALL
            grid.terrain[:, -1] = CellType.WALL

            # Random interior walls
            n_walls = int((size - 2) ** 2 * wall_density)
            for _ in range(n_walls):
                wx = int(rng.integers(1, size - 1))
                wy = int(rng.integers(1, size - 1))
                grid.terrain[wy, wx] = CellType.WALL

            # Agent in a random corner
            corners = [(1, 1), (size - 2, 1), (1, size - 2), (size - 2, size - 2)]
            rng.shuffle(corners)
            agent_pos = tuple(int(c) for c in corners[0])
            grid.terrain[agent_pos[1], agent_pos[0]] = CellType.EMPTY

            # Reachable cells
            reachable = grid.flood_fill(agent_pos)
            reachable_list = list(reachable - {agent_pos})
            if len(reachable_list) < n_treasures + n_clues + 5:
                continue

            rng.shuffle(reachable_list)

            # Place invisible treasures (positions only, nothing on grid)
            treasure_positions = []
            for i in range(n_treasures):
                tx, ty = int(reachable_list[i][0]), int(reachable_list[i][1])
                treasure_positions.append((tx, ty))

            # Place SCROLL clue objects on remaining reachable cells
            clue_candidates = reachable_list[n_treasures:]
            # Exclude treasure positions from clue placement
            treasure_set = set(treasure_positions)
            clue_candidates = [c for c in clue_candidates if tuple(c) not in treasure_set]

            if len(clue_candidates) < n_clues:
                continue

            clue_info = {}  # (x, y) -> {direction, distance, nearest_treasure, misleading}
            misleading_indices = set()
            if n_misleading > 0:
                misleading_indices = set(
                    int(i) for i in rng.choice(n_clues, size=min(n_misleading, n_clues), replace=False)
                )

            for ci in range(n_clues):
                cx, cy = int(clue_candidates[ci][0]), int(clue_candidates[ci][1])
                grid.objects[cy, cx] = ObjectType.SCROLL

                # Find nearest treasure
                best_dist = float("inf")
                best_treasure = treasure_positions[0]
                for tx, ty in treasure_positions:
                    d = abs(cx - tx) + abs(cy - ty)
                    if d < best_dist:
                        best_dist = d
                        best_treasure = (tx, ty)

                tx, ty = best_treasure
                dx_raw = tx - cx
                dy_raw = ty - cy

                # Determine primary direction toward nearest treasure
                if abs(dy_raw) > abs(dx_raw):
                    direction = 0 if dy_raw < 0 else 2  # N or S
                elif abs(dx_raw) > 0:
                    direction = 1 if dx_raw > 0 else 3  # E or W
                else:
                    direction = 0  # fallback (same cell — shouldn't happen)

                is_misleading = ci in misleading_indices
                if is_misleading:
                    # Pick a random *wrong* direction
                    wrong_dirs = [d for d in range(4) if d != direction]
                    direction = int(rng.choice(wrong_dirs))

                distance = int(min(best_dist, 9))

                # Encode in metadata: direction * 10 + distance
                grid.metadata[cy, cx] = direction * 10 + distance

                clue_info[(cx, cy)] = {
                    "direction": direction,
                    "distance": distance,
                    "nearest_treasure": best_treasure,
                    "misleading": is_misleading,
                }

            return grid, {
                "agent_start": agent_pos,
                "goal_positions": list(treasure_positions),
                "_treasure_positions": list(treasure_positions),
                "_collected_treasures": [],
                "_clues_read": [],
                "_clue_info": clue_info,
                "max_steps": self.get_max_steps(),
            }

        # Fallback: minimal open grid
        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL
        agent_pos = (1, 1)
        treasure_positions = [(size - 2, size - 2)]
        # Place one scroll clue near center
        mid = size // 2
        grid.objects[mid, mid] = ObjectType.SCROLL
        grid.metadata[mid, mid] = 2 * 10 + min(abs(mid - (size - 2)) + abs(mid - (size - 2)), 9)
        return grid, {
            "agent_start": agent_pos,
            "goal_positions": treasure_positions,
            "_treasure_positions": treasure_positions,
            "_collected_treasures": [],
            "_clues_read": [],
            "_clue_info": {
                (mid, mid): {
                    "direction": 2,
                    "distance": min(abs(mid - (size - 2)) + abs(mid - (size - 2)), 9),
                    "nearest_treasure": treasure_positions[0],
                    "misleading": False,
                },
            },
            "max_steps": self.get_max_steps(),
        }

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_env_reset(self, agent, grid, config):
        config["_collected_treasures"] = []
        config["_clues_read"] = []

    def on_env_step(self, agent, grid, config, step_count):
        self._config = config

    def on_agent_moved(self, pos, agent, grid):
        """Handle scroll reading and treasure discovery when agent moves."""
        config = getattr(self, "_config", None)
        if config is None:
            return

        x, y = pos

        # Check if agent stepped on a SCROLL: read and consume it
        if grid.objects[y, x] == ObjectType.SCROLL:
            grid.objects[y, x] = ObjectType.NONE
            # Keep metadata so language renderer can reference it if needed,
            # but clear it to avoid confusion
            grid.metadata[y, x] = 0
            clues_read = config.get("_clues_read", [])
            if (x, y) not in clues_read:
                clues_read.append((x, y))
            config["_clues_read"] = clues_read

        # Check if agent is standing on an uncollected treasure position
        treasure_positions = config.get("_treasure_positions", [])
        collected = config.get("_collected_treasures", [])
        if (x, y) in treasure_positions and (x, y) not in collected:
            collected.append((x, y))
            config["_collected_treasures"] = collected
            # Visual feedback: place GOAL marker at discovered treasure position
            grid.objects[y, x] = ObjectType.GOAL

    # ------------------------------------------------------------------
    # Success / reward
    # ------------------------------------------------------------------

    def check_success(self, state):
        if "config" not in state:
            return False
        config = state["config"]
        treasures = config.get("_treasure_positions", [])
        collected = config.get("_collected_treasures", [])
        return len(collected) >= len(treasures) and len(treasures) > 0

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01  # step penalty

        old_config = old_state.get("config", {})
        new_config = new_state.get("config", {})

        # Clue reading bonus
        old_clues = len(old_config.get("_clues_read", []))
        new_clues = len(new_config.get("_clues_read", []))
        if new_clues > old_clues:
            reward += 0.05

        # Treasure collection bonus
        old_collected = len(old_config.get("_collected_treasures", []))
        new_collected = len(new_config.get("_collected_treasures", []))
        if new_collected > old_collected:
            reward += 0.3

        # Success bonus
        if self.check_success(new_state):
            reward += 1.0

        return reward

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0
