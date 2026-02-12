"""Solvability validation and optimal path finding.

This module provides algorithms to verify that generated levels are solvable
and to compute optimal solution paths.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType


@dataclass
class ValidationResult:
    """Result of solvability validation."""

    solvable: bool
    optimal_path: list[tuple[int, int]] | None
    optimal_length: int
    reason: str = ""


class SolvabilityValidator:
    """Validate solvability of generated levels."""

    def __init__(self):
        """Initialize validator."""
        pass

    def validate(
        self,
        grid: Grid,
        start_pos: tuple[int, int],
        goal_positions: list[tuple[int, int]],
        config: dict[str, Any] | None = None,
    ) -> ValidationResult:
        """
        Validate that level is solvable.

        Args:
            grid: Grid to validate
            start_pos: Starting position
            goal_positions: List of goal positions (any one counts)
            config: Additional configuration (keys, doors, boxes, etc.)

        Returns:
            ValidationResult with solvability info and optimal path
        """
        config = config or {}

        # Check if any goal is reachable
        if not goal_positions:
            return ValidationResult(
                solvable=False,
                optimal_path=None,
                optimal_length=0,
                reason="No goal positions specified",
            )

        # Simple case: no keys/doors/boxes
        if not config.get("keys") and not config.get("doors") and not config.get("boxes"):
            return self._validate_simple_navigation(grid, start_pos, goal_positions)

        # Complex case: handle keys, doors, boxes, switches
        if config.get("switches"):
            return self._validate_with_switches(grid, start_pos, goal_positions, config)

        if config.get("inventory"):
            return self._validate_with_inventory(grid, start_pos, goal_positions, config)

        if config.get("keys") or config.get("doors"):
            return self._validate_with_keys_doors(grid, start_pos, goal_positions, config)

        if config.get("boxes"):
            return self._validate_with_boxes(grid, start_pos, goal_positions, config)

        # Default to simple validation
        return self._validate_simple_navigation(grid, start_pos, goal_positions)

    def _validate_simple_navigation(
        self,
        grid: Grid,
        start_pos: tuple[int, int],
        goal_positions: list[tuple[int, int]],
    ) -> ValidationResult:
        """Validate simple navigation (no obstacles besides walls)."""
        # BFS to find shortest path to any goal
        queue = deque([(start_pos, [start_pos])])
        visited = {start_pos}

        while queue:
            pos, path = queue.popleft()
            x, y = pos

            # Check if reached goal
            if pos in goal_positions:
                return ValidationResult(
                    solvable=True,
                    optimal_path=path,
                    optimal_length=len(path) - 1,
                )

            # Explore neighbors
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = x + dx, y + dy

                if not grid.in_bounds((nx, ny)):
                    continue

                if (nx, ny) in visited:
                    continue

                # Check if passable
                if grid.terrain[ny, nx] == CellType.WALL:
                    continue

                # Check for locked doors (if no keys, doors are impassable)
                if grid.objects[ny, nx] == ObjectType.DOOR:
                    continue

                visited.add((nx, ny))
                queue.append(((nx, ny), path + [(nx, ny)]))

        return ValidationResult(
            solvable=False,
            optimal_path=None,
            optimal_length=0,
            reason="No path found to any goal",
        )

    def _validate_with_keys_doors(
        self,
        grid: Grid,
        start_pos: tuple[int, int],
        goal_positions: list[tuple[int, int]],
        config: dict[str, Any],
    ) -> ValidationResult:
        """Validate with key-door mechanics."""
        # State: (position, keys_held)
        # keys_held is a frozenset of key colors

        initial_state = (start_pos, frozenset())
        queue = deque([(initial_state, [start_pos])])
        visited = {initial_state}

        # Get key and door info from config
        key_positions = config.get("keys", {})  # {color: position}
        door_positions = config.get("doors", {})  # {color: position}

        while queue:
            state, path = queue.popleft()
            pos, keys = state
            x, y = pos

            # Check if reached goal
            if pos in goal_positions:
                return ValidationResult(
                    solvable=True,
                    optimal_path=path,
                    optimal_length=len(path) - 1,
                )

            # Explore neighbors
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = x + dx, y + dy

                if not grid.in_bounds((nx, ny)):
                    continue

                # Check if passable
                if grid.terrain[ny, nx] == CellType.WALL:
                    continue

                # Check if this is a key
                new_keys = keys
                for color, key_pos in key_positions.items():
                    if (nx, ny) == key_pos and color not in keys:
                        new_keys = keys | {color}

                # Check if this is a door
                door_blocks = False
                for color, door_pos in door_positions.items():
                    if (nx, ny) == door_pos:
                        if color not in new_keys:
                            # Can't pass through locked door
                            door_blocks = True
                            break
                if not door_blocks:
                    # No door blocking or have key
                    new_state = ((nx, ny), new_keys)

                    if new_state not in visited:
                        visited.add(new_state)
                        queue.append((new_state, path + [(nx, ny)]))

        return ValidationResult(
            solvable=False,
            optimal_path=None,
            optimal_length=0,
            reason="No valid path found (keys/doors may block all paths)",
        )

    def _validate_with_boxes(
        self,
        grid: Grid,
        start_pos: tuple[int, int],
        goal_positions: list[tuple[int, int]],
        config: dict[str, Any],
    ) -> ValidationResult:
        """Validate Sokoban-style box pushing with full state tracking."""
        # State: (player_pos, box_positions)
        box_positions = set(config.get("boxes", []))
        target_positions = set(config.get("targets", goal_positions))

        initial_state = (start_pos, frozenset(box_positions))
        queue = deque([(initial_state, [start_pos])])
        visited = {initial_state}

        max_iterations = 10000  # Prevent infinite loops
        iterations = 0

        while queue and iterations < max_iterations:
            iterations += 1
            state, path = queue.popleft()
            pos, boxes = state
            x, y = pos

            # Check if all boxes are on targets
            if boxes.issubset(target_positions):
                return ValidationResult(
                    solvable=True,
                    optimal_path=path,
                    optimal_length=len(path) - 1,
                )

            # Try each direction
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = x + dx, y + dy

                if not grid.in_bounds((nx, ny)):
                    continue

                if grid.terrain[ny, nx] == CellType.WALL:
                    continue

                # Check if there's a box in the way
                if (nx, ny) in boxes:
                    # Try to push box
                    bnx, bny = nx + dx, ny + dy

                    if not grid.in_bounds((bnx, bny)):
                        continue

                    if grid.terrain[bny, bnx] == CellType.WALL:
                        continue

                    if (bnx, bny) in boxes:
                        # Can't push into another box
                        continue

                    # Push box
                    new_boxes = (boxes - {(nx, ny)}) | {(bnx, bny)}
                    new_state = ((nx, ny), new_boxes)

                    if new_state not in visited:
                        visited.add(new_state)
                        queue.append((new_state, path + [(nx, ny)]))
                else:
                    # Just move
                    new_state = ((nx, ny), boxes)

                    if new_state not in visited:
                        visited.add(new_state)
                        queue.append((new_state, path + [(nx, ny)]))

        if iterations >= max_iterations:
            return ValidationResult(
                solvable=False,
                optimal_path=None,
                optimal_length=0,
                reason="Validation exceeded maximum iterations (too complex)",
            )

        return ValidationResult(
            solvable=False,
            optimal_path=None,
            optimal_length=0,
            reason="No solution found for box puzzle",
        )

    def _validate_with_switches(
        self,
        grid: Grid,
        start_pos: tuple[int, int],
        goal_positions: list[tuple[int, int]],
        config: dict[str, Any],
    ) -> ValidationResult:
        """Validate with switch mechanics."""
        # State: (position, switch_states)
        # switch_states is a frozenset of activated switch positions

        switch_positions = set(config.get("switches", []))
        required_switches = config.get("required_switches", switch_positions)

        initial_state = (start_pos, frozenset())
        queue = deque([(initial_state, [start_pos])])
        visited = {initial_state}

        while queue:
            state, path = queue.popleft()
            pos, activated = state
            x, y = pos

            # Check if reached goal with all required switches activated
            if pos in goal_positions and required_switches.issubset(activated):
                return ValidationResult(
                    solvable=True,
                    optimal_path=path,
                    optimal_length=len(path) - 1,
                )

            # Explore neighbors
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = x + dx, y + dy

                if not grid.in_bounds((nx, ny)):
                    continue

                if grid.terrain[ny, nx] == CellType.WALL:
                    continue

                # Check if this position has a switch
                new_activated = activated
                if (nx, ny) in switch_positions and (nx, ny) not in activated:
                    new_activated = activated | {(nx, ny)}

                new_state = ((nx, ny), new_activated)

                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, path + [(nx, ny)]))

        return ValidationResult(
            solvable=False,
            optimal_path=None,
            optimal_length=0,
            reason="No valid path found (switches may not be reachable or goal unreachable)",
        )

    def _validate_with_inventory(
        self,
        grid: Grid,
        start_pos: tuple[int, int],
        goal_positions: list[tuple[int, int]],
        config: dict[str, Any],
    ) -> ValidationResult:
        """Validate with inventory mechanics (collect items to reach goal)."""
        # State: (position, inventory)
        # inventory is a frozenset of collected item positions

        item_positions = set(config.get("items", []))
        required_items = config.get("required_items", item_positions)

        initial_state = (start_pos, frozenset())
        queue = deque([(initial_state, [start_pos])])
        visited = {initial_state}

        while queue:
            state, path = queue.popleft()
            pos, inventory = state
            x, y = pos

            # Check if reached goal with all required items
            if pos in goal_positions and required_items.issubset(inventory):
                return ValidationResult(
                    solvable=True,
                    optimal_path=path,
                    optimal_length=len(path) - 1,
                )

            # Explore neighbors
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = x + dx, y + dy

                if not grid.in_bounds((nx, ny)):
                    continue

                if grid.terrain[ny, nx] == CellType.WALL:
                    continue

                # Check if this position has an item
                new_inventory = inventory
                if (nx, ny) in item_positions and (nx, ny) not in inventory:
                    new_inventory = inventory | {(nx, ny)}

                new_state = ((nx, ny), new_inventory)

                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, path + [(nx, ny)]))

        return ValidationResult(
            solvable=False,
            optimal_path=None,
            optimal_length=0,
            reason="No valid path found (items may not be reachable or goal unreachable)",
        )


def verify_solvable(
    grid: Grid,
    start_pos: tuple[int, int],
    goal_positions: list[tuple[int, int]],
    config: dict[str, Any] | None = None,
) -> bool:
    """
    Quick verification that level is solvable.

    Args:
        grid: Grid to validate
        start_pos: Starting position
        goal_positions: Goal positions
        config: Additional configuration

    Returns:
        True if solvable, False otherwise
    """
    validator = SolvabilityValidator()
    result = validator.validate(grid, start_pos, goal_positions, config)
    return result.solvable


def find_optimal_path(
    grid: Grid,
    start_pos: tuple[int, int],
    goal_positions: list[tuple[int, int]],
    config: dict[str, Any] | None = None,
) -> tuple[list[tuple[int, int]] | None, int]:
    """
    Find optimal path to goal.

    Args:
        grid: Grid to search
        start_pos: Starting position
        goal_positions: Goal positions
        config: Additional configuration

    Returns:
        Tuple of (path, length) where path is None if unsolvable
    """
    validator = SolvabilityValidator()
    result = validator.validate(grid, start_pos, goal_positions, config)

    if result.solvable:
        return result.optimal_path, result.optimal_length
    else:
        return None, 0


def compute_solution_stats(
    grid: Grid,
    start_pos: tuple[int, int],
    goal_positions: list[tuple[int, int]],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Compute detailed statistics about solution.

    Args:
        grid: Grid to analyze
        start_pos: Starting position
        goal_positions: Goal positions
        config: Additional configuration

    Returns:
        Dictionary with solution statistics
    """
    validator = SolvabilityValidator()
    result = validator.validate(grid, start_pos, goal_positions, config)

    if not result.solvable:
        return {
            "solvable": False,
            "reason": result.reason,
        }

    # Compute additional stats
    path = result.optimal_path
    path_length = result.optimal_length

    # Branching factor (average number of choices at each step)
    total_neighbors = 0
    for pos in path[:-1]:  # Exclude goal
        x, y = pos
        neighbors = 0
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = x + dx, y + dy
            if grid.in_bounds((nx, ny)) and grid.terrain[ny, nx] != CellType.WALL:
                neighbors += 1
        total_neighbors += neighbors

    avg_branching = total_neighbors / len(path) if path else 0

    # Number of turns in optimal path
    turns = 0
    if len(path) > 2:
        for i in range(1, len(path) - 1):
            prev_dx = path[i][0] - path[i - 1][0]
            prev_dy = path[i][1] - path[i - 1][1]
            next_dx = path[i + 1][0] - path[i][0]
            next_dy = path[i + 1][1] - path[i][1]
            if (prev_dx, prev_dy) != (next_dx, next_dy):
                turns += 1

    return {
        "solvable": True,
        "optimal_length": path_length,
        "branching_factor": avg_branching,
        "turns_in_path": turns,
        "path_straightness": 1.0 - (turns / path_length) if path_length > 0 else 1.0,
    }
