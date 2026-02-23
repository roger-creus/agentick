"""TreasureHunt task - Explore to find hidden treasures using proximity clues."""

from __future__ import annotations

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("TreasureHunt-v0", tags=["exploration", "memory", "reasoning"])
class TreasureHuntTask(TaskSpec):
    """Find hidden treasure chests using proximity clues on the floor.

    The agent explores a grid containing hidden TREASURE objects (GEM).
    Floor metadata encodes Manhattan distance to the nearest treasure
    as visible clue markers (TARGET objects at distances 1-3). The agent
    must interpret "warmer/colder" clues to locate and collect all
    treasures. Collected treasures reveal the next set of clues.

    Difficulty scales grid size, treasure count, wall density, and clue
    radius (smaller radius = harder).
    """

    name = "TreasureHunt-v0"
    description = "Find hidden treasures using proximity floor clues"
    capability_tags = ["exploration", "memory", "reasoning"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=9,
            max_steps=120,
            params={"n_treasures": 2, "clue_radius": 3, "wall_density": 0.05},
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=11,
            max_steps=200,
            params={"n_treasures": 3, "clue_radius": 3, "wall_density": 0.10},
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=14,
            max_steps=350,
            params={"n_treasures": 4, "clue_radius": 2, "wall_density": 0.15},
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=16,
            max_steps=500,
            params={"n_treasures": 5, "clue_radius": 1, "wall_density": 0.18},
        ),
    }

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        p = self.difficulty_config.params or {}
        n_treasures = p.get("n_treasures", 2)
        clue_radius = p.get("clue_radius", 3)
        wall_density = p.get("wall_density", 0.05)

        for _attempt in range(20):
            grid = Grid(size, size)
            # Outer walls
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

            # Agent in random corner
            corners = [(1, 1), (size - 2, 1), (1, size - 2), (size - 2, size - 2)]
            rng.shuffle(corners)
            agent_pos = tuple(int(c) for c in corners[0])
            # Ensure agent pos is empty
            grid.terrain[agent_pos[1], agent_pos[0]] = CellType.EMPTY

            # Find reachable cells
            reachable = grid.flood_fill(agent_pos)
            reachable_list = list(reachable - {agent_pos})
            if len(reachable_list) < n_treasures + 5:
                continue

            rng.shuffle(reachable_list)

            # Place treasures as GEM objects
            treasure_positions = []
            for i in range(n_treasures):
                tx, ty = reachable_list[i]
                tx, ty = int(tx), int(ty)
                grid.objects[ty, tx] = ObjectType.GEM
                treasure_positions.append((tx, ty))

            # Place proximity clues around treasures
            self._place_clues(grid, treasure_positions, clue_radius)

            return grid, {
                "agent_start": agent_pos,
                "goal_positions": treasure_positions,
                "_treasure_positions": treasure_positions,
                "_collected": [],
                "_clue_radius": clue_radius,
                "max_steps": self.get_max_steps(),
            }

        # Fallback: open grid
        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL
        agent_pos = (1, 1)
        grid.objects[size - 2, size - 2] = ObjectType.GEM
        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [(size - 2, size - 2)],
            "_treasure_positions": [(size - 2, size - 2)],
            "_collected": [],
            "_clue_radius": 3,
            "max_steps": self.get_max_steps(),
        }

    @staticmethod
    def _place_clues(grid, treasure_positions, clue_radius):
        """Place TARGET clue markers around uncollected treasures."""
        for tx, ty in treasure_positions:
            if grid.objects[ty, tx] != ObjectType.GEM:
                continue  # already collected
            for dy in range(-clue_radius, clue_radius + 1):
                for dx in range(-clue_radius, clue_radius + 1):
                    nx, ny = tx + dx, ty + dy
                    dist = abs(dx) + abs(dy)
                    if dist == 0 or dist > clue_radius:
                        continue
                    if 0 <= nx < grid.width and 0 <= ny < grid.height:
                        if (
                            grid.terrain[ny, nx] == CellType.EMPTY
                            and grid.objects[ny, nx] == ObjectType.NONE
                        ):
                            # Closer = brighter clue (lower metadata value)
                            grid.objects[ny, nx] = ObjectType.TARGET
                            grid.metadata[ny, nx] = max(
                                1, min(dist, int(grid.metadata[ny, nx]) or 99)
                            )

    def on_env_reset(self, agent, grid, config):
        config["_collected"] = []

    def on_agent_moved(self, pos, agent, grid):
        """Collect treasure when agent steps on it."""
        x, y = pos
        if grid.objects[y, x] == ObjectType.GEM:
            grid.objects[y, x] = ObjectType.NONE
            config = getattr(self, "_config", None)
            if config is not None:
                collected = config.get("_collected", [])
                collected.append((x, y))
                config["_collected"] = collected
                # Refresh clues for remaining treasures
                self._clear_and_refresh_clues(grid, config)

    def _clear_and_refresh_clues(self, grid, config):
        """Remove old clues and place new ones for remaining treasures."""
        # Clear all TARGET clue markers
        for cy in range(grid.height):
            for cx in range(grid.width):
                if grid.objects[cy, cx] == ObjectType.TARGET:
                    grid.objects[cy, cx] = ObjectType.NONE
                    grid.metadata[cy, cx] = 0
        # Re-place clues for uncollected treasures
        treasures = config.get("_treasure_positions", [])
        collected = config.get("_collected", [])
        remaining = [t for t in treasures if t not in collected]
        clue_radius = config.get("_clue_radius", 3)
        self._place_clues(grid, remaining, clue_radius)

    def on_env_step(self, agent, grid, config, step_count):
        self._config = config

    def check_success(self, state):
        if "grid" not in state or "agent" not in state:
            return False
        config = state.get("config", {})
        treasures = config.get("_treasure_positions", [])
        collected = config.get("_collected", [])
        return len(collected) >= len(treasures) and len(treasures) > 0

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        # Treasure collection bonus
        old_collected = len(old_state.get("config", {}).get("_collected", []))
        new_collected = len(new_state.get("config", {}).get("_collected", []))
        if new_collected > old_collected:
            reward += 0.3

        # Distance shaping to nearest uncollected treasure
        if "agent" in new_state and "config" in new_state:
            config = new_state["config"]
            treasures = config.get("_treasure_positions", [])
            collected = config.get("_collected", [])
            remaining = [t for t in treasures if t not in collected]
            if remaining:
                ax, ay = new_state["agent"].position
                ox, oy = old_state.get("agent_position", (ax, ay))
                old_min = min(abs(ox - t[0]) + abs(oy - t[1]) for t in remaining)
                new_min = min(abs(ax - t[0]) + abs(ay - t[1]) for t in remaining)
                reward += 0.03 * (old_min - new_min)

        if self.check_success(new_state):
            reward += 1.0
        return reward

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0
