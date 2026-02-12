"""Tests for Grid class."""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType


def test_grid_initialization():
    """Test grid is properly initialized."""
    grid = Grid(10, 15)
    assert grid.height == 10
    assert grid.width == 15
    assert grid.terrain.shape == (10, 15)
    assert grid.objects.shape == (10, 15)
    assert grid.agents.shape == (10, 15)
    assert grid.metadata.shape == (10, 15)
    assert np.all(grid.terrain == 0)


def test_grid_copy():
    """Test grid deep copy."""
    grid = Grid(5, 5)
    grid.terrain[2, 3] = CellType.WALL
    grid.objects[1, 1] = ObjectType.GOAL

    copied = grid.copy()
    assert copied == grid
    assert copied is not grid
    assert copied.terrain is not grid.terrain

    # Modify copy shouldn't affect original
    copied.terrain[0, 0] = CellType.HAZARD
    assert grid.terrain[0, 0] != CellType.HAZARD


def test_grid_equality():
    """Test grid equality comparison."""
    grid1 = Grid(5, 5)
    grid2 = Grid(5, 5)
    assert grid1 == grid2

    grid1.terrain[2, 2] = CellType.WALL
    assert grid1 != grid2

    grid2.terrain[2, 2] = CellType.WALL
    assert grid1 == grid2


def test_grid_hash():
    """Test grid hashing."""
    grid1 = Grid(3, 3)
    grid2 = Grid(3, 3)

    # Same state should have same hash
    assert hash(grid1) == hash(grid2)

    # Can be used in sets
    grid_set = {grid1, grid2}
    assert len(grid_set) == 1


def test_in_bounds():
    """Test bounds checking."""
    grid = Grid(5, 5)

    assert grid.in_bounds((0, 0))
    assert grid.in_bounds((4, 4))
    assert grid.in_bounds((2, 3))

    assert not grid.in_bounds((-1, 0))
    assert not grid.in_bounds((0, -1))
    assert not grid.in_bounds((5, 0))
    assert not grid.in_bounds((0, 5))


def test_is_walkable():
    """Test walkability checking."""
    grid = Grid(5, 5)

    # Empty cells are walkable
    assert grid.is_walkable((2, 2))

    # Walls are not walkable
    grid.terrain[3, 3] = CellType.WALL
    assert not grid.is_walkable((3, 3))

    # Holes are not walkable
    grid.terrain[1, 1] = CellType.HOLE
    assert not grid.is_walkable((1, 1))

    # Out of bounds not walkable
    assert not grid.is_walkable((10, 10))


def test_get_neighbors():
    """Test getting neighbor positions."""
    grid = Grid(5, 5)

    # Middle cell has 4 neighbors
    neighbors = grid.get_neighbors((2, 2))
    assert len(neighbors) == 4
    assert set(neighbors) == {(2, 1), (3, 2), (2, 3), (1, 2)}

    # Corner cell has 2 neighbors
    neighbors = grid.get_neighbors((0, 0))
    assert len(neighbors) == 2
    assert set(neighbors) == {(1, 0), (0, 1)}

    # With diagonals
    neighbors = grid.get_neighbors((2, 2), include_diagonal=True)
    assert len(neighbors) == 8


def test_bfs():
    """Test BFS pathfinding."""
    grid = Grid(5, 5)

    # Simple path
    path = grid.bfs((0, 0), (4, 4))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (4, 4)
    assert len(path) == 9  # Manhattan path length

    # Path blocked by wall
    grid.terrain[:, 2] = CellType.WALL  # Vertical wall at column 2
    path = grid.bfs((1, 1), (3, 1))
    assert path is None  # No path possible

    # Create opening in wall
    grid.terrain[1, 2] = CellType.EMPTY
    path = grid.bfs((1, 1), (3, 1))
    assert path is not None
    assert len(path) == 3  # (1,1) -> (2,1) -> (3,1)


def test_flood_fill():
    """Test flood fill reachability."""
    grid = Grid(5, 5)

    # All cells reachable from (0,0) in empty grid
    reachable = grid.flood_fill((0, 0))
    assert len(reachable) == 25

    # Wall divides grid
    grid.terrain[2, :] = CellType.WALL
    reachable = grid.flood_fill((0, 0))
    assert len(reachable) == 10  # 2 rows * 5 cols


def test_line_of_sight():
    """Test line of sight checking."""
    grid = Grid(10, 10)

    # Clear line of sight
    assert grid.line_of_sight((0, 0), (5, 5))

    # Wall blocks sight
    grid.terrain[3, 3] = CellType.WALL
    assert not grid.line_of_sight((0, 0), (5, 5))


def test_manhattan_distance():
    """Test Manhattan distance calculation."""
    grid = Grid(10, 10)

    assert grid.manhattan_distance((0, 0), (0, 0)) == 0
    assert grid.manhattan_distance((0, 0), (3, 4)) == 7
    assert grid.manhattan_distance((5, 5), (2, 3)) == 5


def test_serialization():
    """Test grid serialization and deserialization."""
    grid = Grid(3, 4)
    grid.terrain[0, 0] = CellType.WALL
    grid.objects[1, 1] = ObjectType.GOAL
    grid.metadata[2, 2] = 42

    # Dict serialization
    data = grid.to_dict()
    restored = Grid.from_dict(data)
    assert restored == grid

    # JSON serialization
    json_str = grid.to_json()
    restored = Grid.from_json(json_str)
    assert restored == grid
