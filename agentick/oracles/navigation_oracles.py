"""Oracle bots for navigation tasks."""

from __future__ import annotations

from agentick.core.types import CellType, ObjectType
from agentick.oracles.base import OracleAgent
from agentick.oracles.registry import register_oracle


def _get_hazard_cells(api):
    """Return set of all HAZARD terrain positions."""
    grid = api.grid
    hazards = set()
    for y in range(grid.height):
        for x in range(grid.width):
            if int(grid.terrain[y, x]) == int(CellType.HAZARD):
                hazards.add((x, y))
    return hazards


def _get_npc_cells(api):
    """Return set of cells occupied by NPCs/enemies/blockers + adjacent + predicted.

    Also predicts the next guard position using guard direction from task config.
    """
    avoid = set()
    grid = api.grid
    config = api.task_config
    guards = config.get("_guard_positions", [])
    dirs = config.get("_guard_dirs", [])

    for e in api.get_entities():
        if e.entity_type in ("npc", "enemy", "blocker"):
            avoid.add(e.position)
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                avoid.add((e.position[0] + dx, e.position[1] + dy))

    # Add predicted next positions for guards
    _DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    for i, (gx, gy) in enumerate(guards):
        if i < len(dirs):
            d = dirs[i]
            ddx, ddy = _DIRS[d]
            nx, ny = gx + ddx, gy + ddy
            if (
                0 < nx < grid.width - 1
                and 0 < ny < grid.height - 1
                and int(grid.terrain[ny, nx]) == int(CellType.EMPTY)
            ):
                avoid.add((nx, ny))
                for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                    avoid.add((nx + dx, ny + dy))

    return avoid


def _get_npc_exact(api):
    """Return set of exact NPC/enemy/blocker positions (no prediction)."""
    avoid = set()
    for e in api.get_entities():
        if e.entity_type in ("npc", "enemy", "blocker"):
            avoid.add(e.position)
    return avoid


def _navigate_with_fallback(api, ax, ay, goal_pos, avoid_wide, avoid_exact):
    """BFS with multi-level fallback: wide avoidance → exact → no avoidance.

    Always avoids HAZARD terrain at every fallback level (hazards are fatal).
    After finding a path, verifies the first step won't land on a guard's
    current or predicted position. If it would, tries alternative paths or waits.
    """
    hazards = _get_hazard_cells(api)
    guard_current = _get_npc_exact(api)

    def _first_step_safe(path):
        """Check if the first step of the path avoids current guard positions."""
        if not path or len(path) < 2:
            return True
        next_pos = path[1]
        return next_pos not in guard_current

    def _try_level(avoid_set):
        """Try BFS with given avoid set. Return first action if safe."""
        avoid = avoid_set - {goal_pos}
        path = api.bfs_path_positions((ax, ay), goal_pos, avoid=avoid)
        if path:
            actions = api.positions_to_actions(path)
            if actions and _first_step_safe(path):
                return [actions[0]]
        return None

    # Wide avoidance (guards + adjacent + predicted + hazards)
    result = _try_level(avoid_wide | hazards)
    if result:
        return result

    # Exact NPC avoidance + hazards
    result = _try_level(avoid_exact | hazards)
    if result:
        return result

    # Hazards only (no NPC avoidance)
    result = _try_level(hazards)
    if result:
        return result

    # No avoidance at all (last resort)
    path = api.bfs_path_positions((ax, ay), goal_pos)
    if path:
        actions = api.positions_to_actions(path)
        if actions:
            if _first_step_safe(path):
                return [actions[0]]
            # First step is unsafe -- wait (noop) unless at goal
            if (ax, ay) == goal_pos:
                return [actions[0]]
            return [0]

    return api.move_toward(*goal_pos)


@register_oracle("GoToGoal-v0")
class GoToGoalOracle(OracleAgent):
    """BFS to the goal, avoiding NPC/guard cells. Re-plans every step."""

    def plan(self):
        goal = self.api.get_nearest("goal")
        if not goal:
            return
        ax, ay = self.api.agent_position
        avoid_wide = _get_npc_cells(self.api)
        avoid_exact = _get_npc_exact(self.api)
        self.action_queue = _navigate_with_fallback(
            self.api,
            ax,
            ay,
            goal.position,
            avoid_wide,
            avoid_exact,
        )


@register_oracle("MazeNavigation-v0")
class MazeNavigationOracle(OracleAgent):
    """BFS to the goal through the maze, avoiding guards. Re-plans each step."""

    def plan(self):
        goal = self.api.get_nearest("goal")
        if not goal:
            return
        ax, ay = self.api.agent_position
        avoid_wide = _get_npc_cells(self.api)
        avoid_exact = _get_npc_exact(self.api)
        self.action_queue = _navigate_with_fallback(
            self.api,
            ax,
            ay,
            goal.position,
            avoid_wide,
            avoid_exact,
        )


@register_oracle("MultiGoalRoute-v0")
class MultiGoalRouteOracle(OracleAgent):
    """Greedily visit the nearest unvisited goal."""

    def plan(self):
        goals = self.api.get_entities_of_type("goal")
        if not goals:
            return
        nearest = min(goals, key=lambda g: g.distance)
        self.action_queue = self.api.move_to(*nearest.position)


_MOVE_DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
_ACTION_DELTAS = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]


def _predict_pursuing_positions(obstacles, agent_pos, grid, pursuing):
    """Predict where pursuing obstacles will move toward *agent_pos*."""
    predicted = set()
    for ox, oy in obstacles:
        best_obs = (ox, oy)
        if pursuing:
            best_d = abs(ox - agent_pos[0]) + abs(oy - agent_pos[1])
            for ddx, ddy in _MOVE_DIRS:
                nx, ny = ox + ddx, oy + ddy
                if (
                    0 < nx < grid.width - 1
                    and 0 < ny < grid.height - 1
                    and int(grid.terrain[ny, nx]) == int(CellType.EMPTY)
                    and int(grid.objects[ny, nx]) != int(ObjectType.GOAL)
                ):
                    dist = abs(nx - agent_pos[0]) + abs(ny - agent_pos[1])
                    if dist < best_d:
                        best_d = dist
                        best_obs = (nx, ny)
        predicted.add(best_obs)
    return predicted


@register_oracle("DynamicObstacles-v0")
class DynamicObstaclesOracle(OracleAgent):
    """Move toward goal while predicting pursuing-obstacle positions.

    For each candidate move, simulates where pursuing obstacles would chase
    (assuming worst-case pursuit) and rejects moves that would collide with
    predicted positions.  Among safe moves, picks the one closest to the goal.
    Falls back to standard BFS avoidance when obstacles are not pursuing.
    """

    def plan(self):
        goal = self.api.get_nearest("goal")
        if not goal:
            return
        ax, ay = self.api.agent_position
        config = self.api.task_config
        pursuing = config.get("_pursuing", False)

        if not pursuing:
            # Non-pursuing: standard avoidance is fine
            avoid_wide = _get_npc_cells(self.api)
            avoid_exact = _get_npc_exact(self.api)
            self.action_queue = _navigate_with_fallback(
                self.api,
                ax,
                ay,
                goal.position,
                avoid_wide,
                avoid_exact,
            )
            return

        # Pursuing mode: predictive avoidance
        grid = self.api.grid
        obstacles = config.get("_live_obstacles", [])
        goal_pos = goal.position
        current_positions = set(obstacles)

        candidates = []
        for action_id, (dx, dy) in enumerate(_ACTION_DELTAS):
            new_ax, new_ay = ax + dx, ay + dy
            if not (0 < new_ax < grid.width - 1 and 0 < new_ay < grid.height - 1):
                continue
            if int(grid.terrain[new_ay, new_ax]) == int(CellType.WALL):
                continue

            # Skip if current obstacle sits on candidate cell
            if (new_ax, new_ay) in current_positions:
                continue

            # Predict worst-case pursuit toward this candidate position
            predicted = _predict_pursuing_positions(
                obstacles,
                (new_ax, new_ay),
                grid,
                pursuing,
            )
            if (new_ax, new_ay) in predicted:
                continue  # would collide with a pursuing obstacle

            dist_to_goal = abs(new_ax - goal_pos[0]) + abs(new_ay - goal_pos[1])
            candidates.append((action_id, dist_to_goal))

        if candidates:
            candidates.sort(key=lambda c: c[1])
            self.action_queue = [candidates[0][0]]
        else:
            # All moves predicted to collide — fall back to BFS ignoring obstacles
            path = self.api.bfs_path_positions((ax, ay), goal_pos)
            if path:
                actions = self.api.positions_to_actions(path)
                if actions:
                    self.action_queue = [actions[0]]
                    return
            self.action_queue = self.api.move_toward(*goal_pos)
