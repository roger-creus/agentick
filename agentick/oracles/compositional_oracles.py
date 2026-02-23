"""Oracle bots for compositional tasks."""

from __future__ import annotations

from agentick.core.types import CellType, ObjectType
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
                if grid.objects[y, x] in (ObjectType.TARGET, ObjectType.LEVER):
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
    """Push GEM objects onto TARGET positions to complete a geometric pattern.

    Strategy: align gem on one axis first (push horizontally until same column
    as target OR vertically until same row), then push along the other axis.
    Uses config gem_target_pairs for assignment. One action per call.
    """

    def plan(self):
        cfg = self.api.task_config
        all_targets = {tuple(t) for t in cfg.get("target_positions", [])}
        grid = self.api.grid
        ax, ay = self.api.agent_position

        # Find current gem positions and which targets are satisfied
        gems = []
        for y in range(grid.height):
            for x in range(grid.width):
                if grid.objects[y, x] == ObjectType.GEM:
                    gems.append((x, y))

        unsatisfied = [t for t in all_targets if grid.objects[t[1], t[0]] != ObjectType.GEM]
        if not unsatisfied:
            self.action_queue = [0]
            return

        free_gems = [g for g in gems if g not in all_targets]
        settled = {t for t in all_targets if grid.objects[t[1], t[0]] == ObjectType.GEM}
        if not free_gems:
            self.action_queue = [0]
            return

        # Pick the gem-target pair with smallest Manhattan distance
        # Prefer aligned pairs (same row or column) for easier pushing
        best_gem, best_target, best_score = None, None, float("inf")
        for tpos in unsatisfied:
            for gpos in free_gems:
                d = abs(gpos[0] - tpos[0]) + abs(gpos[1] - tpos[1])
                # Give bonus to aligned pairs (same row/col = easier push)
                aligned = 1 if gpos[0] == tpos[0] or gpos[1] == tpos[1] else 0
                score = d - aligned * 100  # heavily prefer aligned
                if score < best_score:
                    best_score = score
                    best_gem = gpos
                    best_target = tpos

        if not best_gem or not best_target:
            self.action_queue = [0]
            return

        gx, gy = best_gem
        tx, ty = best_target

        # Determine push direction: prefer aligning on one axis first
        # If same column, push vertically toward target
        # If same row, push horizontally toward target
        # Otherwise, push horizontally first to align columns
        if gx == tx:
            push_dir = (0, 1) if ty > gy else (0, -1)
        elif gy == ty:
            push_dir = (1, 0) if tx > gx else (-1, 0)
        else:
            # Not aligned — push along horizontal axis first
            push_dir = (1, 0) if tx > gx else (-1, 0)

        push_pos = (gx - push_dir[0], gy - push_dir[1])

        # Verify push is valid: push_pos must be walkable AND the gem's
        # destination (gem + push_dir) must also be walkable/empty.
        def _push_ok(d):
            pp = (gx - d[0], gy - d[1])  # agent stands here
            dest = (gx + d[0], gy + d[1])  # gem lands here
            if not (0 <= pp[0] < grid.width and 0 <= pp[1] < grid.height):
                return False
            if not (0 <= dest[0] < grid.width and 0 <= dest[1] < grid.height):
                return False
            if int(grid.terrain[pp[1], pp[0]]) == int(CellType.WALL):
                return False
            if grid.objects[pp[1], pp[0]] in (ObjectType.SCROLL, ObjectType.GEM):
                return False
            if int(grid.terrain[dest[1], dest[0]]) == int(CellType.WALL):
                return False
            if grid.objects[dest[1], dest[0]] in (ObjectType.SCROLL, ObjectType.GEM):
                return False
            return True

        if not _push_ok(push_dir):
            # Try all 4 directions, prefer ones that reduce distance to target
            all_dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            best_alt = None
            best_alt_score = float("inf")
            for d in all_dirs:
                if d == push_dir:
                    continue
                if _push_ok(d):
                    # Score: Manhattan distance from gem's new position to target
                    new_gx, new_gy = gx + d[0], gy + d[1]
                    score = abs(new_gx - tx) + abs(new_gy - ty)
                    if score < best_alt_score:
                        best_alt_score = score
                        best_alt = d
            if best_alt:
                push_dir = best_alt
                push_pos = (gx - best_alt[0], gy - best_alt[1])

        # Navigate to push position or push
        # Avoid ALL gems (not just the target one) to prevent accidental pushes
        all_gem_positions = set(gems)
        avoid = all_gem_positions | settled
        if (ax, ay) == push_pos:
            # We're in position — push the gem
            dx, dy = gx - ax, gy - ay
            delta_map = {(0, -1): "move_up", (0, 1): "move_down",
                         (-1, 0): "move_left", (1, 0): "move_right"}
            aname = delta_map.get((dx, dy))
            if aname:
                a = self.api.action_name_to_int.get(aname)
                if a is not None:
                    self.action_queue = [a]
                    return

        # Navigate to push position (avoid all gems to prevent accidental pushes)
        path = self.api.bfs_path_positions((ax, ay), push_pos, avoid=avoid)
        if not path:
            # Relax: only avoid non-target gems
            path = self.api.bfs_path_positions(
                (ax, ay), push_pos, avoid=all_gem_positions - {best_gem}
            )
        if not path:
            path = self.api.bfs_path_positions((ax, ay), push_pos)
        if path:
            actions = self.api.positions_to_actions(path)
            if actions:
                self.action_queue = [actions[0]]
                return

        # Fallback: move toward gem
        self.action_queue = self.api.move_toward(*best_gem) or [0]


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
