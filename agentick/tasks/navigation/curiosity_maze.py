"""CuriosityMaze task - Coverage-based maze exploration.

The agent must explore a procedurally generated maze and visit a target
percentage of all reachable cells within a step budget. No explicit targets
or landmarks are placed on the grid — success is purely coverage-driven.

SUCCESS METRIC:
  Success = visited_cells >= 0.9 * upper_bound_coverage
  upper_bound_coverage is the best achievable coverage within the step budget,
  estimated via a greedy nearest-unvisited BFS simulation during generation.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task

_DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]


def _bfs_path_len(terrain, start, goal, height, width):
    """BFS shortest path length from start to goal on terrain array."""
    if start == goal:
        return 0
    visited = {start}
    q = deque([(start, 0)])
    while q:
        (cx, cy), d = q.popleft()
        for dx, dy in _DIRS:
            nx, ny = cx + dx, cy + dy
            if (nx, ny) == goal:
                return d + 1
            if (nx, ny) in visited or not (0 <= nx < width and 0 <= ny < height):
                continue
            if terrain[ny, nx] == CellType.WALL:
                continue
            visited.add((nx, ny))
            q.append(((nx, ny), d + 1))
    return -1  # unreachable


def _simulate_greedy_coverage(terrain, start, max_steps, reachable, height, width):
    """Simulate greedy nearest-unvisited exploration and return coverage count."""
    visited = {start}
    pos = start
    steps_used = 0
    unvisited = set(reachable) - visited

    while steps_used < max_steps and unvisited:
        # Find nearest unvisited cell via BFS
        bfs_visited = {pos}
        q = deque([(pos, 0)])
        nearest = None
        nearest_dist = -1

        while q:
            (cx, cy), d = q.popleft()
            if (cx, cy) in unvisited:
                nearest = (cx, cy)
                nearest_dist = d
                break
            for dx, dy in _DIRS:
                nx, ny = cx + dx, cy + dy
                if (nx, ny) in bfs_visited or not (0 <= nx < width and 0 <= ny < height):
                    continue
                if terrain[ny, nx] == CellType.WALL:
                    continue
                bfs_visited.add((nx, ny))
                q.append(((nx, ny), d + 1))

        if nearest is None:
            break

        if steps_used + nearest_dist > max_steps:
            # Can't reach the nearest unvisited; count cells found during BFS
            visited |= bfs_visited & unvisited
            break

        steps_used += nearest_dist
        visited.add(nearest)
        unvisited.discard(nearest)
        pos = nearest

    return len(visited)


@register_task("CuriosityMaze-v0", tags=["exploration", "memory", "navigation"])
class CuriosityMazeTask(TaskSpec):
    """Explore a maze and visit a target percentage of all reachable cells.

    The agent is placed in a procedurally generated maze with border walls
    and random interior walls. There are no explicit targets or landmarks.
    Success is achieved by visiting at least ``coverage_threshold`` percent
    of all reachable cells within the step budget.

    Visited cells are tracked internally but are NOT visually marked on the
    grid — the agent must remember where it has been.

    Rewards:
      - +0.02 for each newly visited cell
      - -0.01 per-step penalty
      - +1.0 bonus on success

    Difficulty scales grid size, step budget, coverage threshold, and wall
    density.
    """

    name = "CuriosityMaze-v0"
    description = "Explore maze to cover a target percentage of reachable cells"
    capability_tags = ["exploration", "memory", "navigation"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=9,
            max_steps=80,
            params={"wall_density": 0.15},
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=13,
            max_steps=150,
            params={"wall_density": 0.20},
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=17,
            max_steps=280,
            params={"wall_density": 0.25},
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=21,
            max_steps=450,
            params={"wall_density": 0.28},
        ),
    }

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        p = self.difficulty_config.params or {}
        wall_density = p.get("wall_density", 0.15)

        for _attempt in range(20):
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

            # Agent at center
            cx, cy = size // 2, size // 2
            grid.terrain[cy, cx] = CellType.EMPTY
            agent_pos = (cx, cy)

            # Compute reachable cells via flood fill
            reachable = grid.flood_fill(agent_pos)
            reachable_count = len(reachable)

            # Need enough reachable cells for the task to be meaningful
            min_cells = max(5, int((size - 2) ** 2 * 0.3))
            if reachable_count < min_cells:
                continue

            # Compute upper-bound coverage via greedy BFS simulation
            max_steps = self.get_max_steps()
            upper_bound = _simulate_greedy_coverage(
                grid.terrain, agent_pos, max_steps, reachable, size, size,
            )
            # Reject if budget is too tight for meaningful exploration
            if upper_bound < 0.5 * reachable_count:
                continue

            return grid, {
                "agent_start": agent_pos,
                "goal_positions": [],
                "_reachable_count": reachable_count,
                "_upper_bound_coverage": upper_bound,
                "_visited_cells": set(),
                "max_steps": max_steps,
            }

        # Fallback: open grid with border walls only
        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL
        agent_pos = (size // 2, size // 2)
        reachable = grid.flood_fill(agent_pos)
        max_steps = self.get_max_steps()
        upper_bound = _simulate_greedy_coverage(
            grid.terrain, agent_pos, max_steps, reachable, size, size,
        )
        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [],
            "_reachable_count": len(reachable),
            "_upper_bound_coverage": upper_bound,
            "_visited_cells": set(),
            "max_steps": max_steps,
        }

    def on_env_reset(self, agent, grid, config):
        """Initialize visited cells with just the agent start position."""
        config["_visited_cells"] = {tuple(agent.position)}
        self._config = config

    def on_agent_moved(self, pos, agent, grid):
        """Track each cell the agent visits."""
        config = getattr(self, "_config", None)
        if config is None:
            return
        visited = config.get("_visited_cells", set())
        visited.add(tuple(pos))
        config["_visited_cells"] = visited

    def on_env_step(self, agent, grid, config, step_count):
        self._config = config

    def check_success(self, state):
        if "config" not in state:
            return False
        config = state["config"]
        visited = config.get("_visited_cells", set())
        upper_bound = config.get("_upper_bound_coverage", config.get("_reachable_count", 1))
        if upper_bound <= 0:
            return False
        # Success = visited at least 90% of what's theoretically achievable
        return len(visited) >= 0.9 * upper_bound

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01  # step penalty

        # Bonus for visiting new cells
        old_visited = len(old_state.get("config", {}).get("_visited_cells", set()))
        new_visited = len(new_state.get("config", {}).get("_visited_cells", set()))
        new_cells = new_visited - old_visited
        if new_cells > 0:
            reward += 0.02 * new_cells

        # Success bonus
        if self.check_success(new_state):
            reward += 1.0

        return reward

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0
