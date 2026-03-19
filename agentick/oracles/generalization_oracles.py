"""Oracle bots for generalization tasks."""

from __future__ import annotations

from agentick.core.types import ObjectType
from agentick.oracles.base import OracleAgent
from agentick.oracles.helpers import interact_adjacent
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
    """Multi-task sequential oracle: dispatches to phase-type-specific logic.

    Phase types: goal_reach, key_door, lever_barrier, collection, box_push.
    Re-plans every step. Pre-applies action remap when active.
    """

    def _apply_remap(self, actions: list[int]) -> list[int]:
        config = self.api.task_config
        remap = config.get("_action_remap", {})
        if not remap:
            return actions
        int_remap = {int(k): int(v) for k, v in remap.items()}
        return [int_remap.get(a, a) for a in actions]

    def plan(self):
        config = self.api.task_config
        phase_type = config.get("_current_phase_type", "goal_reach")

        if phase_type == "goal_reach":
            self._plan_goal_reach()
        elif phase_type == "key_door":
            self._plan_key_door()
        elif phase_type == "lever_barrier":
            self._plan_lever_barrier()
        elif phase_type == "collection":
            self._plan_collection()
        elif phase_type == "box_push":
            self._plan_box_push()
        else:
            self._plan_goal_reach()

    def _plan_goal_reach(self):
        goal = self.api.get_nearest("goal")
        if goal:
            actions = self.api.move_to(*goal.position)
            self.action_queue = self._apply_remap(actions)
        else:
            self.action_queue = [0]

    def _plan_key_door(self):
        grid = self.api.grid
        ax, ay = self.api.agent_position

        closed_doors = set()
        open_doors = set()
        for y in range(grid.height):
            for x in range(grid.width):
                if int(grid.objects[y, x]) == int(ObjectType.DOOR):
                    if int(grid.metadata[y, x]) < 10:
                        closed_doors.add((x, y))
                    else:
                        open_doors.add((x, y))

        goal = self.api.get_nearest("goal")
        if not goal:
            self.action_queue = [0]
            return

        # Try direct path to goal
        path = self.api.bfs_path_positions(
            (ax, ay), goal.position,
            avoid=closed_doors or None,
            extra_passable=open_doors or None,
        )
        if path:
            actions = self.api.positions_to_actions(path)
            if actions:
                self.action_queue = self._apply_remap([actions[0]])
                return

        has_key = self.api.has_in_inventory("key")
        if has_key and closed_doors:
            nearest_door = min(
                closed_doors, key=lambda d: abs(d[0] - ax) + abs(d[1] - ay),
            )
            agent_ori = self.api.agent.orientation
            actions = interact_adjacent(
                (ax, ay), agent_ori, nearest_door, grid, self.api,
            )
            if actions:
                self.action_queue = self._apply_remap(actions)
                return

        if not has_key:
            keys = self.api.get_entities_of_type("key")
            if keys:
                nearest_key = min(keys, key=lambda k: k.distance)
                path = self.api.bfs_path_positions(
                    (ax, ay), nearest_key.position,
                    avoid=closed_doors or None,
                    extra_passable=open_doors or None,
                )
                if path:
                    actions = self.api.positions_to_actions(path)
                    if actions:
                        self.action_queue = self._apply_remap([actions[0]])
                        return

        raw = self.api.move_toward(*goal.position)
        self.action_queue = self._apply_remap(raw)

    def _plan_lever_barrier(self):
        config = self.api.task_config
        grid = self.api.grid
        ax, ay = self.api.agent_position

        barrier_opened = config.get("_barrier_opened", False)

        if barrier_opened:
            # Barrier open, go to goal
            goal = self.api.get_nearest("goal")
            if goal:
                actions = self.api.move_to(*goal.position)
                self.action_queue = self._apply_remap(actions)
            else:
                self.action_queue = [0]
            return

        # Find lever and interact with it
        levers = self.api.get_entities_of_type("lever")
        if levers:
            lever = levers[0]
            agent_ori = self.api.agent.orientation
            actions = interact_adjacent(
                (ax, ay), agent_ori, lever.position, grid, self.api,
            )
            if actions:
                self.action_queue = self._apply_remap(actions)
                return

        self.action_queue = [0]

    def _plan_collection(self):
        ax, ay = self.api.agent_position

        # Check if all gems collected (goal should be visible)
        goal = self.api.get_nearest("goal")
        if goal:
            actions = self.api.move_to(*goal.position)
            self.action_queue = self._apply_remap(actions)
            return

        # Collect nearest gem
        gems = self.api.get_entities_of_type("gem")
        if gems:
            nearest = min(gems, key=lambda g: g.distance)
            actions = self.api.move_to(*nearest.position)
            self.action_queue = self._apply_remap(actions)
        else:
            self.action_queue = [0]

    def _plan_box_push(self):
        grid = self.api.grid
        ax, ay = self.api.agent_position

        # If goal is visible, box is on target — go to goal
        goal = self.api.get_nearest("goal")
        if goal:
            actions = self.api.move_to(*goal.position)
            self.action_queue = self._apply_remap(actions)
            return

        # Find box and target positions
        box_pos = None
        target_pos = None
        for y in range(grid.height):
            for x in range(grid.width):
                if int(grid.objects[y, x]) == int(ObjectType.BOX):
                    box_pos = (x, y)
                if int(grid.objects[y, x]) == int(ObjectType.TARGET):
                    target_pos = (x, y)

        if not box_pos or not target_pos:
            self.action_queue = [0]
            return

        bx, by = box_pos
        tx, ty = target_pos

        # Determine push direction: box → target
        dx = 0 if tx == bx else (1 if tx > bx else -1)
        dy = 0 if ty == by else (1 if ty > by else -1)
        # Prefer axis with larger distance
        if abs(tx - bx) >= abs(ty - by) and dx != 0:
            dy = 0
        elif dy != 0:
            dx = 0
        else:
            dx = 1 if tx >= bx else -1

        # Push-from position: opposite side of box from push direction
        push_from = (bx - dx, by - dy)

        # Navigate to push-from position
        if (ax, ay) == push_from:
            # Push! Move toward box
            move_action = self._dir_to_action(dx, dy)
            self.action_queue = self._apply_remap([move_action])
        else:
            # Avoid walking through box
            path = self.api.bfs_path_positions(
                (ax, ay), push_from, avoid={box_pos},
            )
            if path:
                actions = self.api.positions_to_actions(path)
                if actions:
                    self.action_queue = self._apply_remap([actions[0]])
                    return
            # Fallback
            actions = self.api.move_toward(*push_from)
            self.action_queue = self._apply_remap(actions)

    @staticmethod
    def _dir_to_action(dx, dy):
        if dy < 0:
            return 1  # MOVE_UP
        if dy > 0:
            return 2  # MOVE_DOWN
        if dx < 0:
            return 3  # MOVE_LEFT
        if dx > 0:
            return 4  # MOVE_RIGHT
        return 0


@register_oracle("NoisyObservation-v0")
class NoisyObservationOracle(OracleAgent):
    """Oracle has perfect state access - just go to real goal."""

    def plan(self):
        goal = self.api.get_nearest("goal")
        if goal:
            self.action_queue = self.api.move_to(*goal.position)
