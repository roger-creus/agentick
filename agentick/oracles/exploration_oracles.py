"""Oracle bots for exploration tasks."""

from __future__ import annotations

from agentick.core.types import ObjectType
from agentick.oracles.base import OracleAgent
from agentick.oracles.registry import register_oracle


@register_oracle("FogOfWarExploration-v0")
class FogOfWarOracle(OracleAgent):
    """BFS to goal (oracle has full grid access despite fog)."""

    def plan(self):
        goal = self.api.get_nearest("goal")
        if goal:
            self.action_queue = self.api.move_to(*goal.position)


@register_oracle("TreasureHunt-v0")
class TreasureHuntOracle(OracleAgent):
    """Visit all GEM treasures using BFS, closest first."""

    def plan(self):
        gems = [e for e in self.api.get_entities() if e.entity_type == "gem"]
        if not gems:
            return
        # Sort by distance, visit closest first
        ax, ay = self.api.agent_position
        gems.sort(key=lambda e: abs(e.position[0] - ax) + abs(e.position[1] - ay))
        for gem in gems:
            actions = self.api.move_to(*gem.position)
            self.action_queue.extend(actions)


@register_oracle("CuriosityMaze-v0")
class CuriosityMazeOracle(OracleAgent):
    """Visit all landmark objects using BFS, closest first."""

    _LANDMARK_TYPES = {"switch", "scroll", "orb", "coin"}

    def plan(self):
        landmarks = [
            e for e in self.api.get_entities()
            if e.entity_type in self._LANDMARK_TYPES
        ]
        if not landmarks:
            # Check if GOAL appeared (all landmarks visited)
            goal = self.api.get_nearest("goal")
            if goal:
                self.action_queue = self.api.move_to(*goal.position)
            return
        # Sort by distance, visit closest first
        ax, ay = self.api.agent_position
        landmarks.sort(
            key=lambda e: abs(e.position[0] - ax) + abs(e.position[1] - ay)
        )
        for lm in landmarks:
            actions = self.api.move_to(*lm.position)
            self.action_queue.extend(actions)
