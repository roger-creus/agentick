"""Oracle bots for memory tasks."""

from __future__ import annotations

from agentick.core.types import CellType, ObjectType
from agentick.oracles.base import OracleAgent
from agentick.oracles.registry import register_oracle


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


@register_oracle("FogOfWarExploration-v0")
class FogOfWarOracle(OracleAgent):
    """BFS to goal (oracle has full grid access despite fog)."""

    def plan(self):
        goal = self.api.get_nearest("goal")
        if goal:
            self.action_queue = self.api.move_to(*goal.position)


@register_oracle("TreasureHunt-v0")
class TreasureHuntOracle(OracleAgent):
    """Read scroll clues, triangulate hidden treasure positions, collect all.

    The oracle has privileged access to task_config which contains the hidden
    treasure positions.  Strategy:
    1. Read nearby scrolls first (for the +0.05 clue bonus).
    2. Navigate to each uncollected treasure position, closest first.
    Re-plans after each action queue is exhausted so that the remaining
    target list stays up-to-date.
    """

    def plan(self):
        config = self.api.task_config
        treasure_positions = [
            tuple(t) for t in config.get("_treasure_positions", [])
        ]
        collected = {
            tuple(c) for c in config.get("_collected_treasures", [])
        }
        remaining = [t for t in treasure_positions if t not in collected]

        if not remaining:
            return

        ax, ay = self.api.agent_position

        # Optionally read a nearby scroll first (greedy: closest scroll)
        scrolls = [
            e for e in self.api.get_entities() if e.entity_type == "scroll"
        ]
        if scrolls:
            scrolls.sort(
                key=lambda e: abs(e.position[0] - ax) + abs(e.position[1] - ay)
            )
            nearest_scroll = scrolls[0]
            # Only detour to scroll if it is reasonably close (within 6 steps)
            if nearest_scroll.distance <= 6:
                actions = self.api.move_to(*nearest_scroll.position)
                if actions:
                    self.action_queue.extend(actions)
                    return  # after reading the scroll, re-plan to pick next target

        # Navigate to the closest uncollected treasure
        remaining.sort(key=lambda t: abs(t[0] - ax) + abs(t[1] - ay))
        for target in remaining:
            actions = self.api.move_to(*target)
            if actions:
                self.action_queue.extend(actions)
                return  # navigate to one treasure, then re-plan
