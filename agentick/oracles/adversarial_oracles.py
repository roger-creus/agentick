"""Oracle bots for adversarial tasks."""

from __future__ import annotations

from agentick.core.types import ActionType, CellType
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
    """Phase 1: collect coins; Phase 2: navigate to current goal after shift.

    In Phase 1, no goal exists (blocked by BLOCKER). Collects coins for reward.
    After the first shift, the real GOAL appears and oracle navigates to it.
    Re-plans each step to handle terrain changes and action remaps.

    Crucially, when an action remap is active (e.g. hard difficulty swaps lr+ud),
    the oracle applies the remap to its planned action integers before queuing them.
    Since the remap is self-inverse (left<->right, up<->down), applying it to the
    planned actions causes the env's remap to restore the intended movement.
    """

    def _apply_remap(self, actions: list[int]) -> list[int]:
        """Apply the current action remap to a list of action integers.

        The env applies _action_remap at step time, so we pre-apply the same
        remap so the env's remap cancels it out — net result is intended movement.
        The remap is self-inverse (e.g. left<->right), so applying it once
        before the env applies it once yields the original intended action.
        """
        config = self.api.task_config
        remap = config.get("_action_remap", {})
        if not remap:
            return actions
        # Build int->int remap table from ActionType->ActionType dict
        int_remap = {int(k): int(v) for k, v in remap.items()}
        return [int_remap.get(a, a) for a in actions]

    def plan(self):
        config = self.api.task_config
        shifted = config.get("_shifted", 0)

        if shifted == 0:
            # Phase 1: no goal yet, collect coins
            coins = self.api.get_entities_of_type("coin")
            if coins:
                nearest = min(coins, key=lambda c: c.distance)
                raw = self.api.move_to(*nearest.position)
                self.action_queue = self._apply_remap(raw)
                return
            # No coins or already collected — wait near center
            self.action_queue = [0]
            return

        # Phase 2+: navigate to current GOAL using full BFS path (not just one step)
        goal = self.api.get_nearest("goal")
        if goal:
            raw = self.api.move_to(*goal.position)
            if not raw:
                # Fallback to single-step if move_to fails
                raw = self.api.move_toward(*goal.position)
            self.action_queue = self._apply_remap(raw)


@register_oracle("NoisyObservation-v0")
class NoisyObservationOracle(OracleAgent):
    """Oracle has perfect state access - just go to real goal."""

    def plan(self):
        goal = self.api.get_nearest("goal")
        if goal:
            self.action_queue = self.api.move_to(*goal.position)
