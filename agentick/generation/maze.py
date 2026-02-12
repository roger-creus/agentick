"""Maze generation algorithms for procedural level generation.

This module implements multiple maze generation algorithms, each producing
different maze "flavors" suitable for different task types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from agentick.core.types import CellType


@dataclass
class MazeConfig:
    """Configuration for maze generation."""

    algorithm: Literal[
        "recursive_backtracker",
        "prims",
        "kruskals",
        "binary_tree",
        "recursive_division",
    ] = "recursive_backtracker"
    corridor_width: int = 1  # Width of corridors (1 = single cell)
    room_frequency: float = 0.0  # 0-1, how often to create rooms instead of corridors
    dead_end_density: float = 0.5  # 0-1, how many dead ends to keep (vs removing)
    loop_frequency: float = 0.1  # 0-1, how often to add loops (remove walls)
    wall_thickness: int = 1  # Thickness of walls


class MazeGenerator:
    """Generate mazes using various algorithms."""

    def __init__(self, rng: np.random.Generator | None = None):
        """
        Initialize maze generator.

        Args:
            rng: Random number generator (if None, creates a new one)
        """
        self.rng = rng or np.random.default_rng()

    def generate(
        self,
        width: int,
        height: int,
        config: MazeConfig | None = None,
    ) -> np.ndarray:
        """
        Generate a maze using the specified algorithm.

        Args:
            width: Maze width
            height: Maze height
            config: Maze configuration

        Returns:
            2D array with CellType values (EMPTY for corridors, WALL for walls)
        """
        config = config or MazeConfig()

        if config.algorithm == "recursive_backtracker":
            grid = recursive_backtracker(width, height, self.rng)
        elif config.algorithm == "prims":
            grid = prims_maze(width, height, self.rng)
        elif config.algorithm == "kruskals":
            grid = kruskals_maze(width, height, self.rng)
        elif config.algorithm == "binary_tree":
            grid = binary_tree_maze(width, height, self.rng)
        elif config.algorithm == "recursive_division":
            grid = recursive_division(width, height, self.rng)
        else:
            raise ValueError(f"Unknown algorithm: {config.algorithm}")

        # Post-process based on config
        if config.loop_frequency > 0:
            grid = self._add_loops(grid, config.loop_frequency)

        if config.dead_end_density < 1.0:
            grid = self._remove_dead_ends(grid, 1.0 - config.dead_end_density)

        return grid

    def _add_loops(self, grid: np.ndarray, frequency: float) -> np.ndarray:
        """Add loops by removing random walls."""
        height, width = grid.shape
        grid = grid.copy()

        # Find all interior walls
        walls = []
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if grid[y, x] == CellType.WALL:
                    # Check if removing this wall would connect two corridors
                    neighbors = [
                        grid[y - 1, x],
                        grid[y + 1, x],
                        grid[y, x - 1],
                        grid[y, x + 1],
                    ]
                    if neighbors.count(CellType.EMPTY) >= 2:
                        walls.append((y, x))

        # Remove random walls
        num_to_remove = int(len(walls) * frequency)
        if num_to_remove > 0:
            indices = self.rng.choice(len(walls), size=num_to_remove, replace=False)
            for idx in indices:
                y, x = walls[idx]
                grid[y, x] = CellType.EMPTY

        return grid

    def _remove_dead_ends(self, grid: np.ndarray, removal_rate: float) -> np.ndarray:
        """Remove dead ends by extending corridors."""
        height, width = grid.shape
        grid = grid.copy()

        # Find all dead ends
        dead_ends = []
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if grid[y, x] == CellType.EMPTY:
                    # Count empty neighbors
                    neighbors = [
                        grid[y - 1, x],
                        grid[y + 1, x],
                        grid[y, x - 1],
                        grid[y, x + 1],
                    ]
                    if neighbors.count(CellType.EMPTY) == 1:
                        dead_ends.append((y, x))

        # Remove random dead ends
        num_to_remove = int(len(dead_ends) * removal_rate)
        if num_to_remove > 0:
            indices = self.rng.choice(len(dead_ends), size=num_to_remove, replace=False)
            for idx in indices:
                y, x = dead_ends[idx]
                # Extend corridor by removing a neighboring wall
                wall_neighbors = []
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 < ny < height - 1 and 0 < nx < width - 1:
                        if grid[ny, nx] == CellType.WALL:
                            wall_neighbors.append((ny, nx))

                if wall_neighbors:
                    wy, wx = wall_neighbors[self.rng.integers(len(wall_neighbors))]
                    grid[wy, wx] = CellType.EMPTY

        return grid


def recursive_backtracker(
    width: int,
    height: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate maze using recursive backtracking (DFS).

    Produces mazes with long, winding corridors and few branches.

    Args:
        width: Maze width
        height: Maze height
        rng: Random number generator

    Returns:
        2D array with maze (EMPTY for corridors, WALL for walls)
    """
    rng = rng or np.random.default_rng()

    # Initialize grid with all walls
    grid = np.full((height, width), CellType.WALL, dtype=np.int32)

    # Start at random position
    start_x = rng.integers(1, width - 1)
    start_y = rng.integers(1, height - 1)

    stack = [(start_x, start_y)]
    visited = {(start_x, start_y)}
    grid[start_y, start_x] = CellType.EMPTY

    while stack:
        x, y = stack[-1]

        # Get unvisited neighbors (2 cells away)
        neighbors = []
        for dx, dy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
            nx, ny = x + dx, y + dy
            if 0 < nx < width - 1 and 0 < ny < height - 1:
                if (nx, ny) not in visited:
                    neighbors.append((nx, ny, dx // 2, dy // 2))

        if neighbors:
            # Choose random unvisited neighbor
            nx, ny, wall_dx, wall_dy = neighbors[rng.integers(len(neighbors))]

            # Carve path to neighbor
            grid[y + wall_dy, x + wall_dx] = CellType.EMPTY
            grid[ny, nx] = CellType.EMPTY

            visited.add((nx, ny))
            stack.append((nx, ny))
        else:
            # Backtrack
            stack.pop()

    return grid


def prims_maze(
    width: int,
    height: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate maze using Prim's algorithm.

    Produces mazes with many short corridors and frequent branches.

    Args:
        width: Maze width
        height: Maze height
        rng: Random number generator

    Returns:
        2D array with maze
    """
    rng = rng or np.random.default_rng()

    # Initialize grid with all walls
    grid = np.full((height, width), CellType.WALL, dtype=np.int32)

    # Start at random position
    start_x = rng.integers(1, width - 1)
    start_y = rng.integers(1, height - 1)
    grid[start_y, start_x] = CellType.EMPTY

    # Add walls to frontier
    frontier = []
    for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        nx, ny = start_x + dx, start_y + dy
        if 0 < nx < width - 1 and 0 < ny < height - 1:
            frontier.append((nx, ny, start_x, start_y))

    while frontier:
        # Choose random wall from frontier
        idx = rng.integers(len(frontier))
        wx, wy, px, py = frontier[idx]
        frontier.pop(idx)

        # Check if wall separates corridor from unvisited cell
        if grid[wy, wx] == CellType.WALL:
            # Find the cell on the other side
            dx = wx - px
            dy = wy - py
            nx = wx + dx
            ny = wy + dy

            if 0 < nx < width - 1 and 0 < ny < height - 1:
                if grid[ny, nx] == CellType.WALL:
                    # Carve path
                    grid[wy, wx] = CellType.EMPTY
                    grid[ny, nx] = CellType.EMPTY

                    # Add new walls to frontier
                    for ddx, ddy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                        nnx, nny = nx + ddx, ny + ddy
                        if 0 < nnx < width - 1 and 0 < nny < height - 1:
                            if grid[nny, nnx] == CellType.WALL:
                                frontier.append((nnx, nny, nx, ny))

    return grid


def kruskals_maze(
    width: int,
    height: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate maze using Kruskal's algorithm (randomized).

    Produces mazes with many small trees that gradually merge.

    Args:
        width: Maze width
        height: Maze height
        rng: Random number generator

    Returns:
        2D array with maze
    """
    rng = rng or np.random.default_rng()

    # Initialize grid with all walls
    grid = np.full((height, width), CellType.WALL, dtype=np.int32)

    # Create cells (only odd positions)
    cells = []
    cell_to_set = {}
    set_id = 0

    for y in range(1, height - 1, 2):
        for x in range(1, width - 1, 2):
            grid[y, x] = CellType.EMPTY
            cells.append((x, y))
            cell_to_set[(x, y)] = set_id
            set_id += 1

    # Create list of walls between cells
    walls = []
    for x, y in cells:
        # Check right neighbor
        if x + 2 < width - 1:
            walls.append((x, y, x + 2, y, x + 1, y))
        # Check down neighbor
        if y + 2 < height - 1:
            walls.append((x, y, x, y + 2, x, y + 1))

    # Shuffle walls
    rng.shuffle(walls)

    # Process walls
    for x1, y1, x2, y2, wx, wy in walls:
        set1 = cell_to_set.get((x1, y1))
        set2 = cell_to_set.get((x2, y2))

        # If cells are in different sets, connect them
        if set1 != set2:
            grid[wy, wx] = CellType.EMPTY

            # Merge sets
            for cell, s in cell_to_set.items():
                if s == set2:
                    cell_to_set[cell] = set1

    return grid


def binary_tree_maze(
    width: int,
    height: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate maze using binary tree algorithm.

    Produces mazes with a bias toward two directions (typically north and east).
    Very fast but has obvious patterns.

    Args:
        width: Maze width
        height: Maze height
        rng: Random number generator

    Returns:
        2D array with maze
    """
    rng = rng or np.random.default_rng()

    # Initialize grid with all walls
    grid = np.full((height, width), CellType.WALL, dtype=np.int32)

    # Process each cell
    for y in range(1, height - 1, 2):
        for x in range(1, width - 1, 2):
            grid[y, x] = CellType.EMPTY

            # Carve north or east (randomly)
            directions = []
            if y > 1:  # Can carve north
                directions.append((0, -2, 0, -1))
            if x < width - 3:  # Can carve east
                directions.append((2, 0, 1, 0))

            if directions:
                dx, dy, wall_dx, wall_dy = directions[rng.integers(len(directions))]
                nx, ny = x + dx, y + dy
                wx, wy = x + wall_dx, y + wall_dy
                if 0 < nx < width - 1 and 0 < ny < height - 1:
                    grid[wy, wx] = CellType.EMPTY

    return grid


def recursive_division(
    width: int,
    height: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate maze using recursive division.

    Produces mazes with large open rooms divided by walls with passages.

    Args:
        width: Maze width
        height: Maze height
        rng: Random number generator

    Returns:
        2D array with maze
    """
    rng = rng or np.random.default_rng()

    # Initialize grid with all empty (opposite of other algorithms)
    grid = np.full((height, width), CellType.EMPTY, dtype=np.int32)

    # Add border walls
    grid[0, :] = CellType.WALL
    grid[-1, :] = CellType.WALL
    grid[:, 0] = CellType.WALL
    grid[:, -1] = CellType.WALL

    def divide(x, y, w, h):
        """Recursively divide chamber with walls."""
        if w < 2 or h < 2:
            return

        # Choose orientation (horizontal or vertical wall)
        if w > h:
            horizontal = False
        elif h > w:
            horizontal = True
        else:
            horizontal = rng.random() < 0.5

        if horizontal:
            # Place horizontal wall
            if h < 3:
                return
            wall_y = y + rng.integers(1, h - 1)
            passage_x = x + rng.integers(0, w)

            for i in range(w):
                if x + i != passage_x:
                    grid[wall_y, x + i] = CellType.WALL

            # Recurse on sub-chambers
            divide(x, y, w, wall_y - y)
            divide(x, wall_y + 1, w, h - (wall_y - y) - 1)
        else:
            # Place vertical wall
            if w < 3:
                return
            wall_x = x + rng.integers(1, w - 1)
            passage_y = y + rng.integers(0, h)

            for i in range(h):
                if y + i != passage_y:
                    grid[y + i, wall_x] = CellType.WALL

            # Recurse on sub-chambers
            divide(x, y, wall_x - x, h)
            divide(wall_x + 1, y, w - (wall_x - x) - 1, h)

    # Start division
    divide(1, 1, width - 2, height - 2)

    return grid
