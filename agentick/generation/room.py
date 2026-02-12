"""Room-based level generation with BSP and corridor connections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType


@dataclass
class Room:
    """Represents a rectangular room."""

    x: int
    y: int
    width: int
    height: int

    def center(self) -> tuple[int, int]:
        """Get room center coordinates."""
        return (self.x + self.width // 2, self.y + self.height // 2)

    def intersects(self, other: Room) -> bool:
        """Check if this room intersects another."""
        return not (
            self.x + self.width < other.x
            or other.x + other.width < self.x
            or self.y + self.height < other.y
            or other.y + other.height < self.y
        )


class RoomGenerator:
    """Generate room-based levels."""

    def __init__(self, rng: np.random.Generator | None = None):
        """
        Initialize room generator.

        Args:
            rng: Random number generator
        """
        self.rng = rng or np.random.default_rng()

    def generate_bsp(
        self,
        width: int,
        height: int,
        min_room_size: int = 4,
        max_depth: int = 4,
    ) -> tuple[np.ndarray, list[Room]]:
        """
        Generate rooms using Binary Space Partitioning.

        Args:
            width: Grid width
            height: Grid height
            min_room_size: Minimum room dimension
            max_depth: Maximum BSP tree depth

        Returns:
            Tuple of (grid, rooms_list)
        """
        # Initialize grid with all walls
        grid_array = np.full((height, width), CellType.WALL, dtype=np.int32)

        # BSP tree to partition space
        rooms = []

        def split_node(x, y, w, h, depth):
            """Recursively split space."""
            if depth >= max_depth or w < min_room_size * 2 or h < min_room_size * 2:
                # Create a room in this space
                room_w = self.rng.integers(min_room_size, w + 1)
                room_h = self.rng.integers(min_room_size, h + 1)
                room_x = x + self.rng.integers(0, w - room_w + 1)
                room_y = y + self.rng.integers(0, h - room_h + 1)

                # Carve room
                grid_array[room_y : room_y + room_h, room_x : room_x + room_w] = CellType.EMPTY

                room = Room(room_x, room_y, room_w, room_h)
                rooms.append(room)
                return [room]

            # Split horizontally or vertically
            if w > h:
                # Vertical split
                split_pos = self.rng.integers(min_room_size, w - min_room_size + 1)
                left_rooms = split_node(x, y, split_pos, h, depth + 1)
                right_rooms = split_node(x + split_pos, y, w - split_pos, h, depth + 1)

                # Connect rooms with corridor
                if left_rooms and right_rooms:
                    room1 = left_rooms[self.rng.integers(len(left_rooms))]
                    room2 = right_rooms[self.rng.integers(len(right_rooms))]
                    self._carve_corridor(grid_array, room1.center(), room2.center())

                return left_rooms + right_rooms
            else:
                # Horizontal split
                split_pos = self.rng.integers(min_room_size, h - min_room_size + 1)
                top_rooms = split_node(x, y, w, split_pos, depth + 1)
                bottom_rooms = split_node(x, y + split_pos, w, h - split_pos, depth + 1)

                # Connect rooms with corridor
                if top_rooms and bottom_rooms:
                    room1 = top_rooms[self.rng.integers(len(top_rooms))]
                    room2 = bottom_rooms[self.rng.integers(len(bottom_rooms))]
                    self._carve_corridor(grid_array, room1.center(), room2.center())

                return top_rooms + bottom_rooms

        # Start BSP
        split_node(0, 0, width, height, 0)

        return grid_array, rooms

    def generate_random_rooms(
        self,
        width: int,
        height: int,
        num_rooms: int = 10,
        min_size: int = 4,
        max_size: int = 10,
    ) -> tuple[np.ndarray, list[Room]]:
        """
        Generate random rooms connected by corridors.

        Args:
            width: Grid width
            height: Grid height
            num_rooms: Number of rooms to attempt
            min_size: Minimum room dimension
            max_size: Maximum room dimension

        Returns:
            Tuple of (grid, rooms_list)
        """
        # Initialize grid with all walls
        grid_array = np.full((height, width), CellType.WALL, dtype=np.int32)

        rooms = []

        # Try to place rooms
        for _ in range(num_rooms):
            # Random room size
            room_w = self.rng.integers(min_size, max_size + 1)
            room_h = self.rng.integers(min_size, max_size + 1)

            # Random position
            room_x = self.rng.integers(1, width - room_w - 1)
            room_y = self.rng.integers(1, height - room_h - 1)

            room = Room(room_x, room_y, room_w, room_h)

            # Check if overlaps with existing rooms
            overlaps = any(room.intersects(r) for r in rooms)

            if not overlaps:
                # Carve room
                grid_array[room_y : room_y + room_h, room_x : room_x + room_w] = CellType.EMPTY

                # Connect to previous room
                if rooms:
                    prev_room = rooms[-1]
                    self._carve_corridor(grid_array, prev_room.center(), room.center())

                rooms.append(room)

        return grid_array, rooms

    def _carve_corridor(
        self,
        grid: np.ndarray,
        pos1: tuple[int, int],
        pos2: tuple[int, int],
    ):
        """Carve L-shaped corridor between two positions."""
        x1, y1 = pos1
        x2, y2 = pos2

        # Randomly choose horizontal-first or vertical-first
        if self.rng.random() < 0.5:
            # Horizontal then vertical
            for x in range(min(x1, x2), max(x1, x2) + 1):
                if 0 <= y1 < grid.shape[0] and 0 <= x < grid.shape[1]:
                    grid[y1, x] = CellType.EMPTY
            for y in range(min(y1, y2), max(y1, y2) + 1):
                if 0 <= y < grid.shape[0] and 0 <= x2 < grid.shape[1]:
                    grid[y, x2] = CellType.EMPTY
        else:
            # Vertical then horizontal
            for y in range(min(y1, y2), max(y1, y2) + 1):
                if 0 <= y < grid.shape[0] and 0 <= x1 < grid.shape[1]:
                    grid[y, x1] = CellType.EMPTY
            for x in range(min(x1, x2), max(x1, x2) + 1):
                if 0 <= y2 < grid.shape[0] and 0 <= x < grid.shape[1]:
                    grid[y2, x] = CellType.EMPTY


def bsp_rooms(
    width: int,
    height: int,
    rng: np.random.Generator | None = None,
    **kwargs,
) -> tuple[np.ndarray, list[Room]]:
    """
    Generate rooms using BSP algorithm.

    Args:
        width: Grid width
        height: Grid height
        rng: Random number generator
        **kwargs: Additional arguments for BSP generation

    Returns:
        Tuple of (grid, rooms)
    """
    generator = RoomGenerator(rng)
    return generator.generate_bsp(width, height, **kwargs)


def random_rooms_with_corridors(
    width: int,
    height: int,
    rng: np.random.Generator | None = None,
    **kwargs,
) -> tuple[np.ndarray, list[Room]]:
    """
    Generate random rooms with corridors.

    Args:
        width: Grid width
        height: Grid height
        rng: Random number generator
        **kwargs: Additional arguments

    Returns:
        Tuple of (grid, rooms)
    """
    generator = RoomGenerator(rng)
    return generator.generate_random_rooms(width, height, **kwargs)


def place_key_door_sequence(
    grid: Grid,
    rooms: list[Room],
    num_keys: int,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """
    Place keys and doors with proper reachability ordering.

    Args:
        grid: Grid to modify
        rooms: List of rooms
        num_keys: Number of keys to place
        rng: Random number generator

    Returns:
        Dict with key and door positions
    """
    rng = rng or np.random.default_rng()

    if len(rooms) < num_keys + 1:
        # Not enough rooms
        return {"keys": {}, "doors": {}}

    # Shuffle rooms
    shuffled_rooms = rooms.copy()
    rng.shuffle(shuffled_rooms)

    keys = {}
    doors = {}
    colors = ["red", "blue", "green", "yellow", "purple"]

    for i in range(min(num_keys, len(colors))):
        color = colors[i]

        # Place key in room i
        key_room = shuffled_rooms[i]
        kx, ky = key_room.center()
        grid.objects[ky, kx] = ObjectType.KEY
        keys[color] = (kx, ky)

        # Place door between room i and room i+1
        if i + 1 < len(shuffled_rooms):
            door_room = shuffled_rooms[i + 1]
            # Place door near entrance to next room
            dx, dy = door_room.center()
            grid.objects[dy, dx] = ObjectType.DOOR
            doors[color] = (dx, dy)

    return {"keys": keys, "doors": doors}
