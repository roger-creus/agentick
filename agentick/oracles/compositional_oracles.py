"""Oracle bots for compositional tasks."""

from __future__ import annotations

from agentick.core.types import ObjectType
from agentick.oracles.base import OracleAgent
from agentick.oracles.registry import register_oracle


def _get_guard_cells(api):
    """Return set of NPC/enemy positions + adjacent cells for avoidance."""
    avoid = set()
    for e in api.get_entities():
        if e.entity_type in ("npc", "enemy"):
            avoid.add(e.position)
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                avoid.add((e.position[0] + dx, e.position[1] + dy))
    return avoid


def _get_guard_exact(api):
    """Return set of NPC/enemy positions only."""
    avoid = set()
    for e in api.get_entities():
        if e.entity_type in ("npc", "enemy"):
            avoid.add(e.position)
    return avoid


@register_oracle("InstructionFollowing-v0")
class InstructionFollowingOracle(OracleAgent):
    """Follow instruction: hit switches, visit waypoints, reach correct goal.

    Uses config to know the true goal position. Visits conditional switches,
    then waypoint zones, then the true goal. Avoids NPC guards with fallback.
    Steps ONE at a time for reactivity.
    """

    def __init__(self, env):
        super().__init__(env)
        self._wait_counter = 0

    def reset(self, obs, info):
        self._wait_counter = 0
        super().reset(obs, info)

    def plan(self):
        config = self.api.task_config
        ax, ay = self.api.agent_position
        grid = self.api.grid

        # Build guard avoidance sets
        avoid_wide = _get_guard_cells(self.api)
        avoid_exact = _get_guard_exact(self.api)

        # True goal from config
        goal_positions = config.get("goal_positions", [])
        true_goal = tuple(goal_positions[0]) if goal_positions else None

        # Build set of wrong TARGET zones to avoid (stepping on them = instant fail).
        # Waypoint zones are TARGET cells the agent must visit for multi_step,
        # so exclude them from the wrong-zone set.
        wp_zones = {tuple(w) for w in config.get("_waypoint_zones", [])}
        wrong_zones = set()
        for y in range(grid.height):
            for x in range(grid.width):
                if grid.objects[y, x] == ObjectType.TARGET:
                    if true_goal is None or (x, y) != true_goal:
                        if (x, y) not in wp_zones:
                            wrong_zones.add((x, y))

        # Step 1: Hit conditional switches
        conds = config.get("_conditional_positions", [])
        hits = config.get("_switches_hit", set())
        for cpos in conds:
            cpos = tuple(cpos)
            if cpos not in hits:
                return self._navigate_to(ax, ay, cpos, avoid_wide, avoid_exact, wrong_zones)

        # Step 2: Visit waypoint zones
        wp_zones_list = config.get("_waypoint_zones", [])
        wp_visited = config.get("_waypoints_visited", set())
        for wp in wp_zones_list:
            wp = tuple(wp)
            if wp not in wp_visited:
                return self._navigate_to(ax, ay, wp, avoid_wide, avoid_exact, wrong_zones)

        # Step 3: Go to true goal
        if true_goal:
            return self._navigate_to(ax, ay, true_goal, avoid_wide, avoid_exact, wrong_zones)

    def _navigate_to(self, ax, ay, target, avoid_wide, avoid_exact, wrong_zones):
        """Navigate to target with multi-level guard avoidance fallback."""
        # Combine guard avoidance with wrong zone avoidance
        avoid_w = (avoid_wide | wrong_zones) - {target}
        avoid_e = (avoid_exact | wrong_zones) - {target}
        avoid_z = wrong_zones - {target}

        action = None

        # Try with wide avoidance (guards + adjacent + wrong zones)
        path = self.api.bfs_path_positions((ax, ay), target, avoid=avoid_w)
        if path:
            actions = self.api.positions_to_actions(path)
            if actions:
                action = actions[0]

        # Try with exact avoidance (guard positions + wrong zones)
        if action is None:
            path = self.api.bfs_path_positions((ax, ay), target, avoid=avoid_e)
            if path:
                actions = self.api.positions_to_actions(path)
                if actions:
                    action = actions[0]

        # Try with just wrong zone avoidance
        if action is None:
            path = self.api.bfs_path_positions((ax, ay), target, avoid=avoid_z)
            if path:
                actions = self.api.positions_to_actions(path)
                if actions:
                    action = actions[0]

        # No avoidance at all
        if action is None:
            path = self.api.bfs_path_positions((ax, ay), target)
            if path:
                actions = self.api.positions_to_actions(path)
                if actions:
                    action = actions[0]

        if action is None:
            self.action_queue = self.api.move_toward(*target)
            return

        # Safety check: if the resulting position would be adjacent to a
        # guard, the guard may step onto us next turn. Wait (noop) instead,
        # unless we've been waiting too long (avoid infinite loops).
        # Also check if waiting at our current position would be dangerous.
        _ACTION_DELTAS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
        guard_exact = _get_guard_exact(self.api)
        if action in _ACTION_DELTAS and guard_exact and self._wait_counter < 8:
            dx, dy = _ACTION_DELTAS[action]
            next_pos = (ax + dx, ay + dy)
            # Check if any guard is adjacent to or at our next position
            # (meaning the guard could step onto us after we move)
            move_danger = False
            for gx, gy in guard_exact:
                if abs(gx - next_pos[0]) + abs(gy - next_pos[1]) <= 1:
                    move_danger = True
                    break
            # Check if staying put is also dangerous
            stay_danger = False
            for gx, gy in guard_exact:
                if abs(gx - ax) + abs(gy - ay) <= 1:
                    stay_danger = True
                    break
            if move_danger and not stay_danger:
                # Wait instead of moving into danger
                self.action_queue = [0]
                self._wait_counter += 1
                return

        self._wait_counter = 0
        self.action_queue = [action]


@register_oracle("ProgramSynthesis-v0")
class ProgramSynthesisOracle(OracleAgent):
    """Pick up ORB items and place on predicted TARGET outputs.

    Alternates between picking up nearest ORB and delivering to nearest TARGET.
    """

    def __init__(self, env):
        super().__init__(env)
        self._carrying = False
        self._last_orb_count = 0

    def reset(self, obs, info):
        self._carrying = False
        self._last_orb_count = 0
        super().reset(obs, info)

    def plan(self):
        grid = self.api.grid

        # Count current ORBs on grid
        orb_count = 0
        for y in range(grid.height):
            for x in range(grid.width):
                if grid.objects[y, x] == ObjectType.ORB:
                    orb_count += 1

        if orb_count < self._last_orb_count:
            self._carrying = True
        self._last_orb_count = orb_count

        if not self._carrying:
            orbs = self.api.get_entities_of_type("orb")
            if orbs:
                nearest = min(orbs, key=lambda o: o.distance)
                self.action_queue = self.api.move_to(*nearest.position)
                return
        else:
            targets = self.api.get_entities_of_type("target")
            if targets:
                nearest = min(targets, key=lambda t: t.distance)
                path = self.api.move_to(*nearest.position)
                if path:
                    self.action_queue = path
                    self._carrying = False
                    return

        goal = self.api.get_nearest("goal")
        if goal:
            self.action_queue = self.api.move_to(*goal.position)


@register_oracle("RecursiveRooms-v0")
class RecursiveRoomsOracle(OracleAgent):
    """Navigate nested rooms to reach goal in deepest room."""

    def plan(self):
        goal = self.api.get_nearest("goal")
        if goal:
            path = self.api.path_to(*goal.position)
            if path:
                self.action_queue = path
                return

        if self.api.has_in_inventory("key"):
            doors = self.api.get_entities_of_type("door")
            if doors:
                nearest = min(doors, key=lambda d: d.distance)
                path = self.api.bfs_path_positions(
                    self.api.agent_position,
                    nearest.position,
                    extra_passable={nearest.position},
                )
                if path:
                    self.action_queue = self.api.positions_to_actions(path)
                    return

        keys = self.api.get_entities_of_type("key")
        if keys:
            nearest = min(keys, key=lambda k: k.distance)
            self.action_queue = self.api.move_to(*nearest.position)
            return

        if goal:
            self.action_queue = self.api.move_toward(*goal.position)
