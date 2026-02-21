"""Oracle bots for adversarial tasks."""

from __future__ import annotations

from agentick.core.types import CellType
from agentick.oracles.base import OracleAgent
from agentick.oracles.registry import register_oracle


@register_oracle("DeceptiveReward-v0")
class DeceptiveRewardOracle(OracleAgent):
    """BFS to real goal, avoiding HAZARD traps.

    Uses terrain_ok to only walk EMPTY cells. Re-plans each step
    for moving traps at expert difficulty. Predicts moving trap
    positions to avoid walking into their path.
    """

    _SAFE_TERRAIN = {int(CellType.EMPTY)}
    _DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    def plan(self):
        goal = self.api.get_nearest("goal")
        if not goal:
            return
        ax, ay = self.api.agent_position
        grid = self.api.grid
        config = self.api.task_config

        # Predict where moving traps will be after on_env_step
        mt_pos = config.get("_mt_positions", [])
        mt_dirs = config.get("_mt_dirs", [])
        predicted_traps = set()
        for i, (tx, ty) in enumerate(mt_pos):
            d = mt_dirs[i] if i < len(mt_dirs) else 0
            dx, dy = self._DIRS[d]
            nx, ny = tx + dx, ty + dy
            if 1 <= nx < grid.width - 1 and 1 <= ny < grid.height - 1:
                predicted_traps.add((nx, ny))
            else:
                predicted_traps.add((tx, ty))

        # Agent may be on HAZARD (trap moved under us) — need extra_passable
        extra = set()
        if int(grid.terrain[ay, ax]) != int(CellType.EMPTY):
            extra.add((ax, ay))

        # BFS avoiding HAZARD terrain + predicted trap positions
        path = self.api.bfs_path_positions(
            (ax, ay),
            goal.position,
            terrain_ok=self._SAFE_TERRAIN,
            extra_passable=extra or None,
            avoid=predicted_traps - {(ax, ay)} or None,
        )
        if path:
            actions = self.api.positions_to_actions(path)
            if actions:
                self.action_queue = [actions[0]]
                return

        # Try without predicted avoidance (but still safe terrain)
        path = self.api.bfs_path_positions(
            (ax, ay),
            goal.position,
            terrain_ok=self._SAFE_TERRAIN,
            extra_passable=extra or None,
        )
        if path:
            actions = self.api.positions_to_actions(path)
            if actions:
                self.action_queue = [actions[0]]
                return

        # Wait for traps to move rather than walk into them
        self.action_queue = [0]


@register_oracle("DistributionShift-v0")
class DistributionShiftOracle(OracleAgent):
    """Navigate to current goal, re-planning each step for rule shifts."""

    def plan(self):
        goal = self.api.get_nearest("goal")
        if goal:
            self.action_queue = self.api.move_toward(*goal.position)


@register_oracle("NoisyObservation-v0")
class NoisyObservationOracle(OracleAgent):
    """Oracle has perfect state access - just go to real goal."""

    def plan(self):
        goal = self.api.get_nearest("goal")
        if goal:
            self.action_queue = self.api.move_to(*goal.position)
