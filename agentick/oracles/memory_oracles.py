"""Oracle bots for memory tasks."""

from __future__ import annotations

from agentick.core.types import CellType, ObjectType
from agentick.oracles.base import OracleAgent
from agentick.oracles.registry import register_oracle


@register_oracle("KeyDoorPuzzle-v0")
class KeyDoorPuzzleOracle(OracleAgent):
    """Collect color-coded keys to unlock matching doors, then reach goal.

    Strategy: if holding a key, find the door with matching color metadata.
    If not holding a key, find the nearest reachable key.
    At hard+, avoids NPC guards with prediction.
    """

    def _get_door_positions(self):
        """Return set of all closed DOOR object positions on the grid.

        Open doors (metadata >= 10) are excluded since they are passable.
        """
        grid = self.api.grid
        doors = set()
        for y in range(grid.height):
            for x in range(grid.width):
                if grid.objects[y, x] == ObjectType.DOOR:
                    if int(grid.metadata[y, x]) < 10:
                        doors.add((x, y))
        return doors

    def _get_door_color(self, pos):
        """Return color metadata of a door at position (x, y).

        For open doors (meta >= 10), returns the base color (meta - 10).
        """
        x, y = pos
        meta = int(self.api.grid.metadata[y, x])
        if meta >= 10:
            return meta - 10
        return meta

    def _get_guard_avoidance(self):
        """Return (wide_avoid, exact_avoid) sets for guard positions.

        wide_avoid includes guard positions + adjacent cells + predicted positions.
        exact_avoid includes only current guard positions (no prediction).
        """
        avoid_wide = set()
        avoid_exact = set()
        grid = self.api.grid
        config = self.api.task_config
        guards = config.get("_guard_positions", [])
        dirs = config.get("_guard_dirs", [])

        for i, (gx, gy) in enumerate(guards):
            avoid_exact.add((gx, gy))
            avoid_wide.add((gx, gy))
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                avoid_wide.add((gx + dx, gy + dy))
            # Predict next position (guard moves in current direction)
            if i < len(dirs):
                d = dirs[i]
                ddx, ddy = [(0, -1), (0, 1), (-1, 0), (1, 0)][d]
                nx, ny = gx + ddx, gy + ddy
                if (
                    0 < nx < grid.width - 1
                    and 0 < ny < grid.height - 1
                    and int(grid.terrain[ny, nx]) == int(CellType.EMPTY)
                ):
                    avoid_wide.add((nx, ny))
                    for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                        avoid_wide.add((nx + dx, ny + dy))

        # Also include NPC entities from grid scan
        for e in self.api.get_entities():
            if e.entity_type in ("npc", "enemy"):
                avoid_exact.add(e.position)
                avoid_wide.add(e.position)
                for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                    avoid_wide.add((e.position[0] + dx, e.position[1] + dy))

        return avoid_wide, avoid_exact

    def plan(self):
        ax, ay = self.api.agent_position
        avoid_wide, avoid_exact = self._get_guard_avoidance()

        # If carrying a key, find the door whose color matches the key's color
        inv_keys = [e for e in self.api.agent.inventory if e.entity_type == "key"]
        if inv_keys:
            held_key = inv_keys[0]
            key_color = held_key.properties.get("color", 0)

            # Find matching door
            doors = self.api.get_entities_of_type("door")
            matching_door = None
            for d in doors:
                if self._get_door_color(d.position) == key_color:
                    matching_door = d
                    break
            if matching_door is None and doors:
                matching_door = min(doors, key=lambda d: d.distance)

            if matching_door:
                dp = matching_door.position
                for avoid in [avoid_wide - {dp}, avoid_exact - {dp}, set()]:
                    path = self.api.bfs_path_positions(
                        (ax, ay),
                        dp,
                        extra_passable={dp},
                        avoid=avoid,
                    )
                    if path:
                        actions = self.api.positions_to_actions(path)
                        if actions:
                            self.action_queue = [actions[0]]
                            return
                self.action_queue = self.api.move_toward(*dp)
                return

        # Look for keys — avoid doors (impassable without a key)
        door_cells = self._get_door_positions()
        keys = self.api.get_entities_of_type("key")
        if keys:
            # Try each key, sorted by distance, picking the nearest reachable one
            # Always avoid doors (can't enter without a key)
            sorted_keys = sorted(keys, key=lambda k: k.distance)
            for key_ent in sorted_keys:
                kp = key_ent.position
                base_avoid = door_cells - {kp}
                for avoid in [
                    (avoid_wide | base_avoid) - {kp},
                    (avoid_exact | base_avoid) - {kp},
                    base_avoid - {kp},
                ]:
                    path = self.api.bfs_path_positions(
                        (ax, ay),
                        kp,
                        avoid=avoid,
                    )
                    if path:
                        actions = self.api.positions_to_actions(path)
                        if actions:
                            self.action_queue = [actions[0]]
                            return
            # No key reachable avoiding doors -- try without door avoidance
            # (in case a key IS behind a door that we need to find another key for)
            for key_ent in sorted_keys:
                kp = key_ent.position
                for avoid in [avoid_wide - {kp}, avoid_exact - {kp}, set()]:
                    path = self.api.bfs_path_positions(
                        (ax, ay),
                        kp,
                        avoid=avoid,
                    )
                    if path:
                        actions = self.api.positions_to_actions(path)
                        if actions:
                            self.action_queue = [actions[0]]
                            return
            # Fallback
            self.action_queue = self.api.move_toward(*sorted_keys[0].position)
            return

        # No keys, no doors - head to goal (through opened door positions)
        goal = self.api.get_nearest("goal")
        if goal:
            gp = goal.position
            for avoid in [avoid_wide - {gp}, avoid_exact - {gp}, set()]:
                path = self.api.bfs_path_positions(
                    (ax, ay),
                    gp,
                    avoid=avoid,
                )
                if path:
                    actions = self.api.positions_to_actions(path)
                    if actions:
                        self.action_queue = [actions[0]]
                        return
            self.action_queue = self.api.move_toward(*gp)


@register_oracle("SequenceMemory-v0")
class SequenceMemoryOracle(OracleAgent):
    """Privileged access: read sequence from config, visit in order."""

    def __init__(self, env):
        super().__init__(env)
        self._sequence = []
        self._progress = 0

    def reset(self, obs, info):
        self._sequence = list(self.api.task_config.get("sequence", []))
        self._progress = 0
        super().reset(obs, info)

    def plan(self):
        config = self.api.task_config
        phase = config.get("_phase", "show")

        if phase == "show":
            self.action_queue = [0]
            return

        progress = config.get("_seq_progress", 0)
        if progress < len(self._sequence):
            target = self._sequence[progress]
            self.action_queue = self.api.move_to(*target)


@register_oracle("BreadcrumbTrail-v0")
class BreadcrumbTrailOracle(OracleAgent):
    """Visit crumbs in order, then go to goal."""

    def __init__(self, env):
        super().__init__(env)
        self._crumbs = []

    def reset(self, obs, info):
        self._crumbs = list(self.api.task_config.get("crumb_positions", []))
        super().reset(obs, info)

    def plan(self):
        config = self.api.task_config
        next_idx = config.get("_next_crumb", 0)
        all_collected = config.get("_all_collected", False)

        if not all_collected and next_idx < len(self._crumbs):
            target = self._crumbs[next_idx]
            self.action_queue = self.api.move_to(*target)
        else:
            goal = self.api.get_nearest("goal")
            if goal:
                self.action_queue = self.api.move_to(*goal.position)


@register_oracle("BacktrackPuzzle-v0")
class BacktrackPuzzleOracle(OracleAgent):
    """Activate all switches, then go to goal."""

    def plan(self):
        config = self.api.task_config
        all_activated = config.get("_all_activated", False)

        if not all_activated:
            switches = self.api.get_entities_of_type("switch")
            # Filter out already-activated switches (metadata >= 100)
            grid = self.api.grid
            unactivated = [
                s for s in switches
                if int(grid.metadata[s.position[1], s.position[0]]) < 100
            ]
            if unactivated:
                nearest = min(unactivated, key=lambda s: s.distance)
                self.action_queue = self.api.move_to(*nearest.position)
                return

        goal = self.api.get_nearest("goal")
        if goal:
            self.action_queue = self.api.move_to(*goal.position)


@register_oracle("DelayedGratification-v0")
class DelayedGratificationOracle(OracleAgent):
    """Go directly to goal, avoiding decoy KEY objects and HAZARD terrain.

    Decoys are KEY objects - stepping on them ends the episode.
    HAZARD terrain is also fatal.
    Custom BFS that avoids both KEY cells and HAZARD cells.
    """

    def plan(self):
        goal = self.api.get_nearest("goal")
        if not goal:
            return

        grid = self.api.grid
        ax, ay = self.api.agent_position
        gx, gy = goal.position

        # Build avoid set: all KEY positions + HAZARD terrain
        avoid = set()
        for y in range(grid.height):
            for x in range(grid.width):
                if grid.objects[y, x] == ObjectType.KEY:
                    avoid.add((x, y))
                if int(grid.terrain[y, x]) == int(CellType.HAZARD):
                    avoid.add((x, y))

        path = self.api.bfs_path_positions(
            (ax, ay),
            (gx, gy),
            avoid=avoid,
        )
        if path:
            self.action_queue = self.api.positions_to_actions(path)
            return

        # Fallback: avoid only KEYs (walk through hazards if needed)
        key_avoid = set()
        for y in range(grid.height):
            for x in range(grid.width):
                if grid.objects[y, x] == ObjectType.KEY:
                    key_avoid.add((x, y))

        path = self.api.bfs_path_positions(
            (ax, ay),
            (gx, gy),
            avoid=key_avoid,
        )
        if path:
            self.action_queue = self.api.positions_to_actions(path)
            return

        # Last fallback: direct path (might hit decoy but better than nothing)
        self.action_queue = self.api.move_to(gx, gy)
