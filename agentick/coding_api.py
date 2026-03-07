"""Privileged programmatic interface for all Agentick environments.

Provides spatial queries, entity queries, pathfinding, and high-level action
primitives. Designed for hand-coded bots, oracle solvers, and code-generating
AI agents.

Example::

    import agentick
    from agentick.coding_api import AgentickAPI

    env = agentick.make("GoToGoal-v0")
    api = AgentickAPI(env)

    obs, info = env.reset(seed=42)
    api.update(obs, info)

    actions = api.go_to_nearest("goal")
    for action in actions:
        obs, reward, done, trunc, info = env.step(action)
        api.update(obs, info)
        if done or trunc:
            break
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from agentick.core.types import CellType, Direction, ObjectType


@dataclass
class EntityInfo:
    """Lightweight entity descriptor returned by API queries.

    Attributes:
        entity_type: Type string (e.g. "goal", "key", "door").
        position: ``(x, y)`` position on the grid.
        distance: Manhattan distance from agent (filled by queries).
        properties: Extra properties dict from the underlying Entity.
    """

    entity_type: str
    position: tuple[int, int]
    distance: int = 0
    properties: dict[str, Any] | None = None


# Mapping from ObjectType int to human-readable name
_OBJECT_NAMES: dict[int, str] = {int(v): v.name.lower() for v in ObjectType}
_CELL_NAMES: dict[int, str] = {int(v): v.name.lower() for v in CellType}

# Reverse: name → ObjectType int
_NAME_TO_OBJECT: dict[str, int] = {v.name.lower(): int(v) for v in ObjectType}

# Direction names
_DIR_NAMES: dict[Direction, str] = {
    Direction.NORTH: "north",
    Direction.EAST: "east",
    Direction.SOUTH: "south",
    Direction.WEST: "west",
}


class AgentickAPI:
    """Privileged programmatic wrapper for any Agentick environment.

    The API reads directly from the environment's internal state (grid layers,
    agent object, entities list) to expose a rich set of queries. It does NOT
    call ``env.step()`` itself — high-level action methods return lists of
    action integers that the caller must execute.

    All positions use the ``(x, y)`` convention matching the codebase, where
    ``x`` is the column and ``y`` is the row. Grid array indexing is ``[y, x]``.

    Args:
        env: An Agentick environment (``TaskEnv`` or ``AgentickEnv``).
    """

    def __init__(self, env: Any) -> None:
        self._env = env
        self._obs: Any = None
        self._info: dict[str, Any] = {}
        self._total_reward: float = 0.0
        self._is_done: bool = False
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # State update
    # ------------------------------------------------------------------

    def update(self, obs: Any = None, info: dict[str, Any] | None = None) -> None:
        """Refresh cached state from latest observation and info.

        Must be called after ``env.reset()`` and after every ``env.step()``.
        """
        if obs is not None:
            self._obs = obs
        if info is not None:
            self._info = info
        self._step_count = self._env.step_count
        self._total_reward = self._env.episode_reward
        self._is_done = self._env.done

    # ------------------------------------------------------------------
    # Spatial queries
    # ------------------------------------------------------------------

    @property
    def agent_position(self) -> tuple[int, int]:
        """Agent ``(x, y)`` position on the grid."""
        return self._env.agent.position

    @property
    def agent_direction(self) -> str:
        """Agent facing direction as a string (``"north"``/``"south"``/…)."""
        return _DIR_NAMES[self._env.agent.orientation]

    @property
    def grid_size(self) -> tuple[int, int]:
        """Grid dimensions as ``(width, height)``."""
        return (self._env.grid.width, self._env.grid.height)

    # ------------------------------------------------------------------
    # Entity queries
    # ------------------------------------------------------------------

    def get_entities(self) -> list[EntityInfo]:
        """Return all non-NONE objects on the grid plus tracked entities.

        Scans the ``objects`` layer and the env's entity list.
        """
        entities: list[EntityInfo] = []
        grid = self._env.grid
        ax, ay = self.agent_position

        # Scan objects layer
        for y in range(grid.height):
            for x in range(grid.width):
                obj = int(grid.objects[y, x])
                if obj != int(ObjectType.NONE):
                    name = _OBJECT_NAMES.get(obj, f"object_{obj}")
                    dist = abs(x - ax) + abs(y - ay)
                    entities.append(EntityInfo(name, (x, y), dist))

        # Also include entities tracked in env.entities (NPCs etc.)
        for ent in self._env.entities:
            dist = abs(ent.position[0] - ax) + abs(ent.position[1] - ay)
            entities.append(EntityInfo(ent.entity_type, ent.position, dist, ent.properties))

        return entities

    def get_entities_of_type(self, entity_type: str) -> list[EntityInfo]:
        """Return all entities matching the given type string."""
        entity_type = entity_type.lower()
        return [e for e in self.get_entities() if e.entity_type == entity_type]

    def get_entity_at(self, x: int, y: int) -> EntityInfo | None:
        """Return the entity at ``(x, y)``, or ``None``."""
        grid = self._env.grid
        if not grid.in_bounds((x, y)):
            return None
        obj = int(grid.objects[y, x])
        if obj != int(ObjectType.NONE):
            name = _OBJECT_NAMES.get(obj, f"object_{obj}")
            ax, ay = self.agent_position
            dist = abs(x - ax) + abs(y - ay)
            return EntityInfo(name, (x, y), dist)
        # Check entity list
        for ent in self._env.entities:
            if ent.position == (x, y):
                ax, ay = self.agent_position
                dist = abs(x - ax) + abs(y - ay)
                return EntityInfo(ent.entity_type, ent.position, dist, ent.properties)
        return None

    def get_nearest(self, entity_type: str) -> EntityInfo | None:
        """Return the nearest entity of the given type, or ``None``."""
        matches = self.get_entities_of_type(entity_type)
        if not matches:
            return None
        return min(matches, key=lambda e: e.distance)

    def get_all_positions(self, entity_type: str) -> list[tuple[int, int]]:
        """Return positions of all entities of the given type."""
        return [e.position for e in self.get_entities_of_type(entity_type)]

    def count(self, entity_type: str) -> int:
        """Count entities of the given type currently on the map."""
        return len(self.get_entities_of_type(entity_type))

    def get_inventory(self) -> list[EntityInfo]:
        """Return items in the agent's inventory."""
        return [
            EntityInfo(item.entity_type, item.position, 0, item.properties)
            for item in self._env.agent.inventory
        ]

    def has_in_inventory(self, entity_type: str) -> bool:
        """Check whether the agent is carrying an item of the given type."""
        return self._env.agent.has_item(entity_type)

    # ------------------------------------------------------------------
    # Distance & pathfinding
    # ------------------------------------------------------------------

    def distance_to(self, x: int, y: int) -> int:
        """Manhattan distance from the agent to ``(x, y)``."""
        ax, ay = self.agent_position
        return abs(x - ax) + abs(y - ay)

    def direction_to(self, x: int, y: int) -> str | None:
        """Primary cardinal direction from agent toward ``(x, y)``.

        Returns ``"north"``/``"south"``/``"east"``/``"west"`` or ``None``
        if the agent is already at ``(x, y)``.
        """
        ax, ay = self.agent_position
        dx = x - ax
        dy = y - ay
        if dx == 0 and dy == 0:
            return None
        # Prefer the axis with the larger absolute difference
        if abs(dy) >= abs(dx):
            return "north" if dy < 0 else "south"
        return "east" if dx > 0 else "west"

    def path_to(self, x: int, y: int) -> list[int]:
        """BFS shortest-path from agent to ``(x, y)`` as action integers.

        Returns an empty list if the target is unreachable.
        """
        ax, ay = self.agent_position
        if (ax, ay) == (x, y):
            return []
        positions = self._bfs_path((ax, ay), (x, y))
        if positions is None:
            return []
        return self._positions_to_actions(positions)

    def is_reachable(self, x: int, y: int) -> bool:
        """Check whether the agent can reach ``(x, y)`` via walkable cells."""
        ax, ay = self.agent_position
        if (ax, ay) == (x, y):
            return True
        return self._bfs_path((ax, ay), (x, y)) is not None

    def is_adjacent(self, x: int, y: int) -> bool:
        """Check whether ``(x, y)`` is cardinally adjacent to the agent."""
        ax, ay = self.agent_position
        return abs(x - ax) + abs(y - ay) == 1

    def neighbors(self, x: int, y: int) -> list[tuple[int, int]]:
        """Return walkable cardinal neighbors of ``(x, y)``.

        A cell is walkable if its terrain is passable AND no solid object
        (closed DOOR, LEVER, SWITCH) blocks it.
        """
        result: list[tuple[int, int]] = []
        grid = self._env.grid
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            pos = (nx, ny)
            if grid.in_bounds(pos) and grid.is_walkable(pos) and not grid.is_object_blocking(pos):
                result.append(pos)
        return result

    # ------------------------------------------------------------------
    # Map queries
    # ------------------------------------------------------------------

    def get_cell(self, x: int, y: int) -> str:
        """Return terrain type name at ``(x, y)``."""
        grid = self._env.grid
        if not grid.in_bounds((x, y)):
            return "out_of_bounds"
        t = int(grid.terrain[y, x])
        return _CELL_NAMES.get(t, f"terrain_{t}")

    def get_object(self, x: int, y: int) -> str:
        """Return object type name at ``(x, y)``."""
        grid = self._env.grid
        if not grid.in_bounds((x, y)):
            return "out_of_bounds"
        o = int(grid.objects[y, x])
        return _OBJECT_NAMES.get(o, f"object_{o}")

    def is_walkable(self, x: int, y: int) -> bool:
        """Check if the agent can step on ``(x, y)``.

        Returns ``False`` if terrain is impassable (WALL/HOLE) OR a solid
        object (closed DOOR, LEVER, SWITCH) blocks the cell.
        """
        grid = self._env.grid
        pos = (x, y)
        return grid.is_walkable(pos) and not grid.is_object_blocking(pos)

    def get_walkable_cells(self) -> list[tuple[int, int]]:
        """Return all walkable ``(x, y)`` positions.

        Excludes cells blocked by solid objects (closed DOORs, etc.).
        """
        grid = self._env.grid
        cells: list[tuple[int, int]] = []
        for y in range(grid.height):
            for x in range(grid.width):
                pos = (x, y)
                if grid.is_walkable(pos) and not grid.is_object_blocking(pos):
                    cells.append(pos)
        return cells

    def get_walls(self) -> list[tuple[int, int]]:
        """Return all wall positions."""
        grid = self._env.grid
        walls: list[tuple[int, int]] = []
        for y in range(grid.height):
            for x in range(grid.width):
                if int(grid.terrain[y, x]) == int(CellType.WALL):
                    walls.append((x, y))
        return walls

    def get_terrain_type(self, x: int, y: int) -> str:
        """Alias for :meth:`get_cell`."""
        return self.get_cell(x, y)

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    @property
    def current_step(self) -> int:
        """Current timestep in the episode."""
        return self._env.step_count

    @property
    def max_steps(self) -> int:
        """Episode step limit."""
        return self._env.max_steps

    @property
    def total_reward(self) -> float:
        """Accumulated reward so far."""
        return self._env.episode_reward

    @property
    def is_done(self) -> bool:
        """Whether the episode has ended."""
        return self._env.done

    @property
    def valid_actions(self) -> list[int]:
        """List of currently valid action integers."""
        mask = self._env.get_valid_actions()
        return [i for i in range(len(mask)) if mask[i]]

    @property
    def action_names(self) -> dict[int, str]:
        """Mapping from action integer to name."""
        space = self._env.action_space_obj
        return {i: space.get_action_name(i) for i in range(space.n_actions)}

    @property
    def action_name_to_int(self) -> dict[str, int]:
        """Reverse mapping from action name to integer."""
        return {name: idx for idx, name in self.action_names.items()}

    @property
    def task_config(self) -> dict[str, Any]:
        """Task configuration dictionary (privileged access)."""
        return getattr(self._env, "task_config", {})

    @property
    def grid(self):
        """Direct access to the Grid object."""
        return self._env.grid

    @property
    def agent(self):
        """Direct access to the Agent object."""
        return self._env.agent

    # ------------------------------------------------------------------
    # High-level actions (return action sequences)
    # ------------------------------------------------------------------

    def move_to(self, x: int, y: int) -> list[int]:
        """Pathfind and return action sequence to reach ``(x, y)``."""
        return self.path_to(x, y)

    def move_toward(self, x: int, y: int) -> list[int]:
        """Return a single action that moves the agent closer to ``(x, y)``.

        Uses BFS if a path exists, otherwise falls back to Manhattan direction.
        """
        path_actions = self.path_to(x, y)
        if path_actions:
            return [path_actions[0]]
        # Fallback: try cardinal direction
        direction = self.direction_to(x, y)
        if direction is None:
            return []
        name_map = self.action_name_to_int
        dir_to_action = {
            "north": "move_up",
            "south": "move_down",
            "east": "move_right",
            "west": "move_left",
        }
        action_name = dir_to_action.get(direction, "noop")
        if action_name in name_map:
            return [name_map[action_name]]
        return []

    def pickup_nearest(self, entity_type: str) -> list[int]:
        """Move to nearest entity of type, then pick it up."""
        nearest = self.get_nearest(entity_type)
        if nearest is None:
            return []
        actions = self.move_to(*nearest.position)
        name_map = self.action_name_to_int
        if "pickup" in name_map:
            actions.append(name_map["pickup"])
        return actions

    def go_to_nearest(self, entity_type: str) -> list[int]:
        """Return action sequence to reach the nearest entity of given type."""
        nearest = self.get_nearest(entity_type)
        if nearest is None:
            return []
        return self.move_to(*nearest.position)

    def interact_with(self, x: int, y: int) -> list[int]:
        """Move to ``(x, y)`` and interact."""
        actions = self.move_to(x, y)
        name_map = self.action_name_to_int
        if "interact" in name_map:
            actions.append(name_map["interact"])
        return actions

    def flee_from(self, x: int, y: int) -> list[int]:
        """Return a single action that moves the agent away from ``(x, y)``."""
        ax, ay = self.agent_position
        dx = ax - x
        dy = ay - y

        # Determine the best direction to flee
        candidates: list[tuple[str, int, int]] = []
        if abs(dx) >= abs(dy):
            if dx > 0:
                candidates = [("move_right", 1, 0), ("move_up", 0, -1), ("move_down", 0, 1)]
            else:
                candidates = [("move_left", -1, 0), ("move_up", 0, -1), ("move_down", 0, 1)]
        else:
            if dy > 0:
                candidates = [("move_down", 0, 1), ("move_left", -1, 0), ("move_right", 1, 0)]
            else:
                candidates = [("move_up", 0, -1), ("move_left", -1, 0), ("move_right", 1, 0)]

        name_map = self.action_name_to_int
        grid = self._env.grid
        for action_name, ddx, ddy in candidates:
            nx, ny = ax + ddx, ay + ddy
            pos = (nx, ny)
            if (
                grid.is_walkable(pos)
                and not grid.is_object_blocking(pos)
                and action_name in name_map
            ):
                return [name_map[action_name]]
        return []

    # ------------------------------------------------------------------
    # Convenience helpers for oracle bots
    # ------------------------------------------------------------------

    def action_int(self, name: str) -> int:
        """Convert action name to integer. Raises KeyError if unknown."""
        return self.action_name_to_int[name]

    def step_action(self, dx: int, dy: int) -> int | None:
        """Return the action integer for a single ``(dx, dy)`` movement.

        Returns ``None`` if no matching movement action exists.
        """
        delta_to_name = {
            (0, -1): "move_up",
            (0, 1): "move_down",
            (-1, 0): "move_left",
            (1, 0): "move_right",
        }
        name = delta_to_name.get((dx, dy))
        if name is None:
            return None
        return self.action_name_to_int.get(name)

    def bfs_path_positions(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        extra_passable: set[tuple[int, int]] | None = None,
        avoid: set[tuple[int, int]] | None = None,
        terrain_ok: set[int] | None = None,
    ) -> list[tuple[int, int]] | None:
        """BFS from *start* to *goal* returning the list of positions.

        ``extra_passable`` lets callers mark cells as walkable even if the
        terrain says otherwise (useful for doors the oracle knows it can open).
        ``avoid`` marks cells that the BFS should treat as impassable.
        ``terrain_ok`` if provided, restricts walkable terrain to these types.
        """
        return self._bfs_path(
            start,
            goal,
            extra_passable=extra_passable,
            avoid=avoid,
            terrain_ok=terrain_ok,
        )

    def positions_to_actions(self, positions: list[tuple[int, int]]) -> list[int]:
        """Convert a sequence of ``(x, y)`` waypoints to action integers."""
        return self._positions_to_actions(positions)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bfs_path(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        extra_passable: set[tuple[int, int]] | None = None,
        avoid: set[tuple[int, int]] | None = None,
        terrain_ok: set[int] | None = None,
    ) -> list[tuple[int, int]] | None:
        """BFS shortest path returning list of positions (including start).

        A cell is passable when its terrain is walkable AND no solid object
        blocks it (closed DOOR, LEVER, SWITCH).  ``extra_passable`` overrides
        both terrain and object checks for the listed positions.
        """
        grid = self._env.grid

        def _ok(pos: tuple[int, int]) -> bool:
            if avoid and pos in avoid:
                return False
            if extra_passable and pos in extra_passable:
                return True
            if terrain_ok is not None:
                if not grid.in_bounds(pos):
                    return False
                if int(grid.terrain[pos[1], pos[0]]) not in terrain_ok:
                    return False
                # Still check blocking objects even with terrain_ok
                return not grid.is_object_blocking(pos)
            return grid.is_walkable(pos) and not grid.is_object_blocking(pos)

        if not _ok(start):
            return None
        # Goal may not be walkable (e.g. door), but we still want to reach it
        if not grid.in_bounds(goal):
            return None
        if start == goal:
            return [start]

        queue: deque[tuple[tuple[int, int], list[tuple[int, int]]]] = deque()
        queue.append((start, [start]))
        visited: set[tuple[int, int]] = {start}

        while queue:
            pos, path = queue.popleft()
            for ddx, ddy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = pos[0] + ddx, pos[1] + ddy
                npos = (nx, ny)
                if npos in visited:
                    continue
                if npos == goal:
                    return path + [npos]
                if not _ok(npos):
                    continue
                visited.add(npos)
                queue.append((npos, path + [npos]))

        return None

    def _positions_to_actions(self, positions: list[tuple[int, int]]) -> list[int]:
        """Convert a list of sequential positions to action integers."""
        actions: list[int] = []
        name_map = self.action_name_to_int
        delta_to_name = {
            (0, -1): "move_up",
            (0, 1): "move_down",
            (-1, 0): "move_left",
            (1, 0): "move_right",
        }
        for i in range(len(positions) - 1):
            cx, cy = positions[i]
            nx, ny = positions[i + 1]
            dx, dy = nx - cx, ny - cy
            action_name = delta_to_name.get((dx, dy))
            if action_name and action_name in name_map:
                actions.append(name_map[action_name])
        return actions
