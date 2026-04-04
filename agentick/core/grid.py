"""Grid class for 2D gridworld representation."""

from __future__ import annotations

import json
from collections import deque
from typing import Any

import numpy as np

from agentick.core.types import NON_WALKABLE_OBJECTS, CellType, ObjectType, Position


class Grid:
    """
    Multi-layer grid representation for gridworld environments.

    Layers:
        - terrain: Base terrain type (empty, wall, hazard, etc.)
        - objects: Collectible/interactive objects (keys, goals, etc.)
        - agents: Agent positions
        - metadata: Extra per-cell data (colors, IDs, states)
    """

    __slots__ = ("height", "width", "terrain", "objects", "agents", "metadata")

    def __init__(self, height: int, width: int):
        """
        Initialize an empty grid.

        Args:
            height: Grid height
            width: Grid width
        """
        self.height = height
        self.width = width

        # Layer arrays
        self.terrain = np.zeros((height, width), dtype=np.int8)
        self.objects = np.zeros((height, width), dtype=np.int8)
        self.agents = np.zeros((height, width), dtype=np.int8)
        self.metadata = np.zeros((height, width), dtype=np.int16)

    def copy(self) -> Grid:
        """Create a deep copy of this grid."""
        new_grid = Grid(self.height, self.width)
        new_grid.terrain = self.terrain.copy()
        new_grid.objects = self.objects.copy()
        new_grid.agents = self.agents.copy()
        new_grid.metadata = self.metadata.copy()
        return new_grid

    def __eq__(self, other: object) -> bool:
        """Check equality with another grid."""
        if not isinstance(other, Grid):
            return False
        return (
            self.height == other.height
            and self.width == other.width
            and np.array_equal(self.terrain, other.terrain)
            and np.array_equal(self.objects, other.objects)
            and np.array_equal(self.agents, other.agents)
            and np.array_equal(self.metadata, other.metadata)
        )

    def __hash__(self) -> int:
        """Hash grid state for use in sets/dicts."""
        return hash(
            (
                self.terrain.tobytes(),
                self.objects.tobytes(),
                self.agents.tobytes(),
                self.metadata.tobytes(),
            )
        )

    def in_bounds(self, pos: Position) -> bool:
        """Check if position is within grid bounds."""
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def is_walkable(self, pos: Position) -> bool:
        """Check if position can be walked on."""
        if not self.in_bounds(pos):
            return False
        x, y = pos
        return self.terrain[y, x] not in (CellType.WALL, CellType.HOLE)

    def is_object_blocking(self, pos: Position) -> bool:
        """Check if a non-walkable object blocks movement at *pos*.

        DOOR, LEVER, SWITCH are solid. However, open doors (metadata >= 10)
        are passable.
        """
        if not self.in_bounds(pos):
            return False
        x, y = pos
        obj = ObjectType(self.objects[y, x])
        if obj not in NON_WALKABLE_OBJECTS:
            return False
        # Open doors are passable
        if obj == ObjectType.DOOR and int(self.metadata[y, x]) >= 10:
            return False
        return True

    def get_neighbors(self, pos: Position, include_diagonal: bool = False) -> list[Position]:
        """
        Get neighboring positions.

        Args:
            pos: Center position
            include_diagonal: Whether to include diagonal neighbors

        Returns:
            List of valid neighboring positions
        """
        x, y = pos
        neighbors = []

        # Cardinal directions
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if self.in_bounds((nx, ny)):
                neighbors.append((nx, ny))

        # Diagonal directions
        if include_diagonal:
            for dx, dy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
                nx, ny = x + dx, y + dy
                if self.in_bounds((nx, ny)):
                    neighbors.append((nx, ny))

        return neighbors

    def line_of_sight(self, from_pos: Position, to_pos: Position) -> bool:
        """
        Check if there's a clear line of sight between two positions.

        Uses Bresenham's line algorithm to check for walls.

        Args:
            from_pos: Starting position
            to_pos: Target position

        Returns:
            True if line of sight is clear
        """
        x0, y0 = from_pos
        x1, y1 = to_pos

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0

        while True:
            if not self.is_walkable((x, y)):
                return False

            if x == x1 and y == y1:
                return True

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def flood_fill(
        self, start_pos: Position, check_objects: bool = False
    ) -> set[Position]:
        """
        Get all positions reachable from start_pos.

        Args:
            start_pos: Starting position
            check_objects: If True, also treat blocking objects (closed doors,
                levers, switches) as impassable. Default False for backward
                compatibility with generation code.

        Returns:
            Set of reachable positions
        """
        if not self.is_walkable(start_pos):
            return set()
        if check_objects and self.is_object_blocking(start_pos):
            return set()

        visited = {start_pos}
        queue = deque([start_pos])

        while queue:
            pos = queue.popleft()
            for neighbor in self.get_neighbors(pos):
                if neighbor not in visited and self.is_walkable(neighbor):
                    if check_objects and self.is_object_blocking(neighbor):
                        continue
                    visited.add(neighbor)
                    queue.append(neighbor)

        return visited

    def bfs(
        self,
        start: Position,
        goal: Position,
        check_objects: bool = False,
    ) -> list[Position] | None:
        """
        Find shortest path from start to goal using BFS.

        Args:
            start: Start position
            goal: Goal position
            check_objects: If True, also treat blocking objects (closed doors,
                levers, switches) as impassable. Default False for backward
                compatibility with generation code.

        Returns:
            List of positions forming the path, or None if no path exists
        """
        if not self.is_walkable(start) or not self.is_walkable(goal):
            return None
        if check_objects and (
            self.is_object_blocking(start) or self.is_object_blocking(goal)
        ):
            return None

        if start == goal:
            return [start]

        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            pos, path = queue.popleft()

            for neighbor in self.get_neighbors(pos):
                if neighbor not in visited and self.is_walkable(neighbor):
                    if check_objects and self.is_object_blocking(neighbor):
                        continue
                    if neighbor == goal:
                        return path + [neighbor]
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    def manhattan_distance(self, pos1: Position, pos2: Position) -> int:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def to_dict(self) -> dict[str, Any]:
        """Serialize grid to dictionary."""
        return {
            "height": self.height,
            "width": self.width,
            "terrain": self.terrain.tolist(),
            "objects": self.objects.tolist(),
            "agents": self.agents.tolist(),
            "metadata": self.metadata.tolist(),
        }

    def to_json(self) -> str:
        """Serialize grid to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Grid:
        """Deserialize grid from dictionary."""
        grid = cls(data["height"], data["width"])
        grid.terrain = np.array(data["terrain"], dtype=np.int8)
        grid.objects = np.array(data["objects"], dtype=np.int8)
        grid.agents = np.array(data["agents"], dtype=np.int8)
        grid.metadata = np.array(data["metadata"], dtype=np.int16)
        return grid

    @classmethod
    def from_json(cls, json_str: str) -> Grid:
        """Deserialize grid from JSON string."""
        return cls.from_dict(json.loads(json_str))
