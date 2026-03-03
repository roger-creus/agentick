"""Oracle bots for generalization tasks."""

from __future__ import annotations

from agentick.core.types import ObjectType
from agentick.oracles.base import OracleAgent
from agentick.oracles.registry import register_oracle


@register_oracle("FewShotAdaptation-v0")
class FewShotAdaptationOracle(OracleAgent):
    """Oracle knows the hidden rule and trial structure from config.

    Demo trials auto-advance, so oracle waits (noop) during those.
    During the test trial, navigates directly to the correct target.
    """

    def plan(self):
        config = self.api.task_config
        trials = config.get("trials", [])
        current = config.get("_current_trial", 0)
        ax, ay = self.api.agent_position

        if current >= len(trials):
            # All trials done — wait
            self.action_queue = [0]
            return

        trial = trials[current]

        # Demo trials auto-advance — just wait
        if not trial.get("is_test", False):
            self.action_queue = [0]
            return

        # Test trial — navigate to the correct candidate
        correct_idx = trial.get("correct_idx", 0)
        positions = trial.get("positions", [])
        if correct_idx < len(positions):
            tx, ty = positions[correct_idx]
            path = self.api.bfs_path_positions((ax, ay), (tx, ty))
            if path:
                actions = self.api.positions_to_actions(path)
                if actions:
                    self.action_queue = [actions[0]]
                    return
            self.action_queue = self.api.move_toward(tx, ty)
            return

        self.action_queue = [0]


@register_oracle("DistributionShift-v0")
class DistributionShiftOracle(OracleAgent):
    """Multi-phase maze oracle: BFS to each GOAL across 3 shifting phases.

    Re-plans every step. For hard/expert, collects keys before navigating
    to the goal (doors block the path without matching keys).

    When an action remap is active (expert after first goal), the oracle
    pre-applies the same remap so the env's remap cancels it out.
    """

    def _apply_remap(self, actions: list[int]) -> list[int]:
        """Pre-apply the env's action remap so movements are correct.

        The env applies _action_remap at step time. Since the remap is
        self-inverse (UP<->DOWN, LEFT<->RIGHT), applying it once before
        the env applies it once yields the original intended action.
        """
        config = self.api.task_config
        remap = config.get("_action_remap", {})
        if not remap:
            return actions
        int_remap = {int(k): int(v) for k, v in remap.items()}
        return [int_remap.get(a, a) for a in actions]

    def plan(self):
        grid = self.api.grid
        ax, ay = self.api.agent_position

        # Identify closed door cells (meta < 10) and open door cells.
        closed_doors = set()
        all_doors = set()
        for y in range(grid.height):
            for x in range(grid.width):
                if int(grid.objects[y, x]) == int(ObjectType.DOOR):
                    all_doors.add((x, y))
                    if int(grid.metadata[y, x]) < 10:
                        closed_doors.add((x, y))

        # Head to goal.
        goal = self.api.get_nearest("goal")
        if not goal:
            self.action_queue = [0]
            return

        # First, try to reach the goal WITHOUT passing through any closed
        # doors.  If a path exists, doors are irrelevant — skip keys.
        path_no_doors = self.api.bfs_path_positions(
            (ax, ay),
            goal.position,
            avoid=closed_doors or None,
            extra_passable=(all_doors - closed_doors) or None,
        )
        if path_no_doors:
            actions = self.api.positions_to_actions(path_no_doors)
            if actions:
                self.action_queue = self._apply_remap([actions[0]])
                return

        # Goal is blocked by closed doors — collect nearest key first.
        keys = self.api.get_entities_of_type("key")
        if keys:
            nearest_key = min(keys, key=lambda k: k.distance)
            path = self.api.bfs_path_positions(
                (ax, ay),
                nearest_key.position,
                extra_passable=all_doors or None,
            )
            if path:
                actions = self.api.positions_to_actions(path)
                if actions:
                    self.action_queue = self._apply_remap([actions[0]])
                    return

        # Try to reach goal, treating all doors as passable (we will
        # open them on the way).
        path = self.api.bfs_path_positions(
            (ax, ay),
            goal.position,
            extra_passable=all_doors or None,
        )
        if path:
            actions = self.api.positions_to_actions(path)
            if actions:
                self.action_queue = self._apply_remap([actions[0]])
                return

        # Fallback: move toward goal heuristically.
        raw = self.api.move_toward(*goal.position)
        self.action_queue = self._apply_remap(raw)


@register_oracle("NoisyObservation-v0")
class NoisyObservationOracle(OracleAgent):
    """Oracle has perfect state access - just go to real goal."""

    def plan(self):
        goal = self.api.get_nearest("goal")
        if goal:
            self.action_queue = self.api.move_to(*goal.position)
