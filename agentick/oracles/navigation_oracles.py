"""Oracle bots for navigation tasks."""

from __future__ import annotations

from collections import deque

from agentick.core.types import CellType, ObjectType
from agentick.oracles.base import OracleAgent
from agentick.oracles.helpers import interact_adjacent
from agentick.oracles.registry import register_oracle


def _door_interact_action(api, door_pos, avoid=None):
    """Return a single action to progress toward unlocking a solid door.

    Delegates to the shared ``interact_adjacent`` helper which handles:
    - Adjacent + facing → INTERACT
    - Adjacent + not facing → move toward door (sets orientation, blocked by door)
    - Not adjacent → BFS to nearest walkable cell adjacent to the door

    Args:
        avoid: Optional set of positions the BFS should treat as impassable
            (e.g. distractor cells the agent must not step on).

    Returns a list with one action, or an empty list if unreachable.
    """
    return interact_adjacent(
        api.agent_position,
        api.agent.orientation,
        door_pos,
        api.grid,
        api,
        avoid=avoid,
    )


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
    dir_deltas = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    for i, (gx, gy) in enumerate(guards):
        if i < len(dirs):
            d = dirs[i]
            ddx, ddy = dir_deltas[d]
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

    # Also predict where guards will be after this step
    guard_predicted = set(guard_current)  # start with current positions
    config = api.task_config
    guards = config.get("_guard_positions", [])
    dirs = config.get("_guard_dirs", [])
    dir_deltas = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    grid = api.grid
    for i, (gx, gy) in enumerate(guards):
        if i < len(dirs):
            d = dirs[i]
            ddx, ddy = dir_deltas[d]
            nx, ny = gx + ddx, gy + ddy
            if (
                0 < nx < grid.width - 1
                and 0 < ny < grid.height - 1
                and int(grid.terrain[ny, nx]) == int(CellType.EMPTY)
            ):
                guard_predicted.add((nx, ny))

    def _first_step_safe(path):
        """Check if the first step avoids current and predicted guard positions."""
        if not path or len(path) < 2:
            return True
        next_pos = path[1]
        return next_pos not in guard_predicted

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
    """BFS to the goal through the maze, handling key/door mechanics.

    Strategy (re-planned every step):
    1. Try BFS to the goal.  If reachable -> go.
    2. Otherwise there must be a closed door in the way.
       a. Find a REACHABLE closed door we hold a key for -> approach, face, INTERACT.
       b. If no such door, find and collect a reachable key for any locked door.
       c. If the needed key is behind another door, open that door first.
    """

    def _is_door_reachable(self, agent_pos, door_pos):
        """Check if at least one adjacent walkable cell of *door_pos* is reachable."""
        grid = self.api.grid
        for ddx, ddy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            nx, ny = door_pos[0] + ddx, door_pos[1] + ddy
            pos = (nx, ny)
            if (
                grid.in_bounds(pos)
                and grid.is_walkable(pos)
                and not grid.is_object_blocking(pos)
            ):
                path = self.api.bfs_path_positions(agent_pos, pos)
                if path:
                    return True
        return False

    def plan(self):
        ax, ay = self.api.agent_position
        grid = self.api.grid
        agent = self.api.agent
        avoid_wide = _get_npc_cells(self.api)
        avoid_exact = _get_npc_exact(self.api)

        goal = self.api.get_nearest("goal")
        if not goal:
            return

        # --- 1. Try direct path to goal (BFS respects blocking objects) ---
        direct_path = self.api.bfs_path_positions((ax, ay), goal.position)
        if direct_path:
            self.action_queue = _navigate_with_fallback(
                self.api, ax, ay, goal.position, avoid_wide, avoid_exact,
            )
            return

        # --- 2. Goal unreachable — handle doors ---
        # Identify all closed doors and their colors
        doors = self.api.get_entities_of_type("door")
        locked_colors: dict[int, tuple[int, int]] = {}
        for d in doors:
            dx, dy = d.position
            meta = int(grid.metadata[dy, dx])
            if meta < 10:
                locked_colors[meta] = d.position

        if not locked_colors:
            # No doors but goal unreachable — fallback
            self.action_queue = self.api.move_toward(*goal.position) or [0]
            return

        # Which key colors do we hold?
        held_colors = {
            e.properties.get("color") for e in agent.inventory if e.entity_type == "key"
        }

        # 2a. Try to open a REACHABLE closed door that we have a key for
        for color in held_colors:
            if color in locked_colors:
                door_pos = locked_colors[color]
                if self._is_door_reachable((ax, ay), door_pos):
                    self.action_queue = _door_interact_action(self.api, door_pos)
                    return

        # 2b. Collect a reachable key for any locked door
        keys_on_grid = self.api.get_entities_of_type("key")
        closed_door_cells = set(locked_colors.values())

        # Sort by distance so we try the nearest key first
        needed = []
        for color, door_pos in locked_colors.items():
            if color in held_colors:
                continue
            for k in keys_on_grid:
                kx, ky = k.position
                if int(grid.metadata[ky, kx]) == color:
                    needed.append((k.distance, color, k))

        needed.sort(key=lambda t: t[0])

        for _, color, k in needed:
            # Try BFS to this key, avoiding closed doors
            path = self.api.bfs_path_positions(
                (ax, ay), k.position, avoid=closed_door_cells,
            )
            if path:
                actions = self.api.positions_to_actions(path)
                if actions:
                    self.action_queue = [actions[0]]
                    return
            # Try without avoiding doors (key reachable directly)
            path = self.api.bfs_path_positions((ax, ay), k.position)
            if path:
                actions = self.api.positions_to_actions(path)
                if actions:
                    self.action_queue = [actions[0]]
                    return

        # Fallback: move toward goal heuristically
        self.action_queue = self.api.move_toward(*goal.position) or [0]


@register_oracle("ShortestPath-v0")
class ShortestPathOracle(OracleAgent):
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


@register_oracle("CuriosityMaze-v0")
class CuriosityMazeOracle(OracleAgent):
    """Coverage-based exploration: visit all reachable cells efficiently.

    Uses a nearest-unvisited greedy strategy: at each planning step, BFS to
    the closest reachable cell not yet visited. This produces a near-optimal
    traversal that minimises revisits.
    """

    def __init__(self, env):
        super().__init__(env)
        self._visited: set[tuple[int, int]] = set()

    def reset(self, obs, info):
        """Reset visited tracking on episode start."""
        super().reset(obs, info)
        self._visited = {tuple(self.api.agent_position)}

    def act(self, obs, info):
        """Track visited cells and delegate to base."""
        self.api.update(obs, info)
        self._visited.add(tuple(self.api.agent_position))
        if not self.action_queue:
            self.plan()
        if self.action_queue:
            return self.action_queue.pop(0)
        return 0

    def plan(self):
        """Greedily BFS to the nearest unvisited reachable cell."""
        start = tuple(self.api.agent_position)
        self._visited.add(start)

        # Get all walkable cells
        walkable = set(self.api.get_walkable_cells())

        # Find unvisited walkable cells
        unvisited = walkable - self._visited

        if not unvisited:
            return  # Nothing left to visit

        # BFS from current position to find nearest unvisited cell
        target = self._nearest_unvisited_bfs(start, unvisited)
        if target is None:
            return

        actions = self.api.move_to(*target)
        self.action_queue.extend(actions)

    def _nearest_unvisited_bfs(
        self,
        start: tuple[int, int],
        unvisited: set[tuple[int, int]],
    ) -> tuple[int, int] | None:
        """BFS from start, return the first unvisited cell found."""
        queue: deque[tuple[int, int]] = deque([start])
        seen: set[tuple[int, int]] = {start}
        grid = self.api.grid

        while queue:
            pos = queue.popleft()
            if pos in unvisited:
                return pos
            x, y = pos
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                npos = (x + dx, y + dy)
                if npos not in seen and grid.in_bounds(npos) and grid.is_walkable(npos):
                    seen.add(npos)
                    queue.append(npos)

        return None


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
                # Doors are solid — approach, face, and INTERACT to unlock
                self.action_queue = _door_interact_action(self.api, nearest.position)
                if self.action_queue:
                    return

        keys = self.api.get_entities_of_type("key")
        if keys:
            nearest = min(keys, key=lambda k: k.distance)
            self.action_queue = self.api.move_to(*nearest.position)
            return

        if goal:
            self.action_queue = self.api.move_toward(*goal.position)


@register_oracle("TimingChallenge-v0")
class TimingChallengeOracle(OracleAgent):
    """Navigate past moving blockers by predicting their positions.

    Uses config's _bx_i and _bdir_i to predict where blockers will be
    after on_env_step, and only crosses gaps when safe.
    """

    def plan(self):
        config = self.api.task_config
        goal = self.api.get_nearest("goal")
        if not goal:
            return

        ax, ay = self.api.agent_position
        gx, gy = goal.position
        grid = self.api.grid
        mid_row = config.get("mid_row", grid.height // 2)
        gap_cols = config.get("gap_cols", [config.get("gap_col", grid.width // 2)])
        specs = config.get("_blocker_specs", [])

        # Predict where all blockers will be AFTER the next on_env_step
        next_step = self.api.current_step + 1
        danger_cells = set()
        current_blocker_cells = set()

        for i, s in enumerate(specs):
            speed = s.get("speed", 1)
            bx = config.get(f"_bx_{i}", s["x"])
            d = config.get(f"_bdir_{i}", 1)
            by = s["row"]
            p0, p1 = s["p0"], s["p1"]
            current_blocker_cells.add((bx, by))

            if next_step % speed == 0:
                # Blocker will move
                new_x = bx + d
                if new_x > p1:
                    new_x = bx - 1
                elif new_x < p0:
                    new_x = bx + 1
                new_x = max(p0, min(p1, new_x))
                danger_cells.add((new_x, by))
            else:
                # Blocker stays put
                danger_cells.add((bx, by))

        # Do we need to cross the barrier?
        need_cross = (ay < mid_row and gy > mid_row) or (ay > mid_row and gy < mid_row)

        if need_cross:
            nearest_gap = min(gap_cols, key=lambda gc: abs(gc - ax))

            # Are we adjacent to the gap?
            if ax == nearest_gap and abs(ay - mid_row) == 1:
                cross_target = (nearest_gap, mid_row)
                if cross_target not in danger_cells:
                    # Safe to cross!
                    dy_step = 1 if gy > ay else -1
                    step = self.api.step_action(0, dy_step)
                    if step is not None:
                        self.action_queue = [step]
                        return
                # Not safe - wait
                self.action_queue = [0]
                return

            # Are we ON the barrier row at a gap?
            if ay == mid_row and ax in gap_cols:
                dy_step = 1 if gy > ay else -1
                next_pos = (ax, ay + dy_step)
                if next_pos not in danger_cells:
                    step = self.api.step_action(0, dy_step)
                    if step is not None:
                        self.action_queue = [step]
                        return
                self.action_queue = [0]
                return

            # Navigate to the cell adjacent to the gap
            target_y = mid_row - 1 if ay < mid_row else mid_row + 1
            target_y = max(1, min(grid.height - 2, target_y))

            # Avoid all current blocker cells
            path = self.api.bfs_path_positions(
                (ax, ay),
                (nearest_gap, target_y),
                avoid=current_blocker_cells,
            )
            if path:
                actions = self.api.positions_to_actions(path)
                if actions:
                    # Check first step isn't dangerous
                    first_pos = path[1] if len(path) > 1 else path[0]
                    if first_pos not in danger_cells:
                        self.action_queue = [actions[0]]
                        return
                    # First step is dangerous - wait
                    self.action_queue = [0]
                    return

            # Fallback: try without avoidance
            path = self.api.bfs_path_positions(
                (ax, ay),
                (nearest_gap, target_y),
            )
            if path:
                actions = self.api.positions_to_actions(path)
                if actions:
                    self.action_queue = [actions[0]]
                    return

        # After crossing or no barrier - go to goal avoiding blockers
        avoid = danger_cells | current_blocker_cells
        path = self.api.bfs_path_positions((ax, ay), (gx, gy), avoid=avoid)
        if path:
            actions = self.api.positions_to_actions(path)
            if actions:
                first_pos = path[1] if len(path) > 1 else path[0]
                if first_pos not in danger_cells:
                    self.action_queue = [actions[0]]
                    return
                self.action_queue = [0]
                return

        # Fallback without avoidance
        path = self.api.bfs_path_positions((ax, ay), (gx, gy))
        if path:
            actions = self.api.positions_to_actions(path)
            if actions:
                self.action_queue = [actions[0]]
                return

        self.action_queue = self.api.move_toward(gx, gy)


@register_oracle("InstructionFollowing-v0")
class InstructionFollowingOracle(OracleAgent):
    """Navigate to the unique target object while avoiding distractors.

    Strategy:
    1. Collect keys (if any) to unlock doors gating the target.
    2. Navigate to the target position, treating all distractor positions
       as impassable so the agent never steps on them.
    One action per call for reactivity.
    """

    def plan(self):
        config = self.api.task_config
        ax, ay = self.api.agent_position
        grid = self.api.grid

        target_pos = tuple(config.get("goal_positions", [None])[0] or ())
        target_type = config.get("target_type")

        # Build distractor avoidance set: every cell that holds one of the
        # four collectible types that is NOT the target type.
        distractor_types = {
            int(ObjectType.GEM),
            int(ObjectType.SCROLL),
            int(ObjectType.ORB),
            int(ObjectType.COIN),
        }
        if target_type is not None:
            distractor_types.discard(int(target_type))
        distractor_cells = set()
        for y in range(grid.height):
            for x in range(grid.width):
                if int(grid.objects[y, x]) in distractor_types:
                    distractor_cells.add((x, y))

        avoid = distractor_cells

        # Step 1: If there are doors, collect keys first (in order).
        key_positions = config.get("key_positions", [])
        door_positions = config.get("door_positions", [])
        agent_inv = self.api.agent.inventory
        has_key = any(e.entity_type == "key" for e in agent_inv)

        # Check which doors are still closed
        for i, dp in enumerate(door_positions):
            dpx, dpy = dp
            door_meta = int(grid.metadata[dpy, dpx])
            if door_meta < 10:
                # Door i is still closed -- need key i
                if not has_key:
                    # Navigate to the matching key (keys are walkable, auto-pickup)
                    if i < len(key_positions):
                        kp = tuple(key_positions[i])
                        # Check key is still on grid
                        if grid.objects[kp[1], kp[0]] == ObjectType.KEY:
                            path = self.api.bfs_path_positions((ax, ay), kp, avoid=avoid)
                            if path:
                                actions = self.api.positions_to_actions(path)
                                if actions:
                                    self.action_queue = [actions[0]]
                                    return
                            # Fallback without avoidance
                            path = self.api.bfs_path_positions((ax, ay), kp)
                            if path:
                                actions = self.api.positions_to_actions(path)
                                if actions:
                                    self.action_queue = [actions[0]]
                                    return

                # Have key (or already holding) — approach door, face, INTERACT
                door_pos = tuple(dp)
                self.action_queue = _door_interact_action(self.api, door_pos, avoid=avoid)
                if self.action_queue:
                    return
                break  # Must deal with this door before proceeding

        # Step 2: Navigate to target
        if not target_pos:
            self.action_queue = [0]
            return

        path = self.api.bfs_path_positions((ax, ay), target_pos, avoid=avoid)
        if path:
            actions = self.api.positions_to_actions(path)
            if actions:
                self.action_queue = [actions[0]]
                return

        # Fallback: no avoidance
        path = self.api.bfs_path_positions((ax, ay), target_pos)
        if path:
            actions = self.api.positions_to_actions(path)
            if actions:
                self.action_queue = [actions[0]]
                return

        # Last resort: greedy move toward target
        self.action_queue = self.api.move_toward(*target_pos) or [0]
