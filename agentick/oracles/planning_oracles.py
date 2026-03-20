"""Oracle bots for planning tasks."""

from __future__ import annotations

from collections import deque

from agentick.core.types import CellType, ObjectType
from agentick.oracles.base import OracleAgent
from agentick.oracles.helpers import interact_adjacent as _interact_adjacent
from agentick.oracles.registry import register_oracle


@register_oracle("SokobanPush-v0")
class SokobanPushOracle(OracleAgent):
    """Push boxes onto targets using privileged knowledge of positions.

    Strategy: assign each unplaced box to the best free target, then for
    the most promising box find the push position, navigate there avoiding
    hazards and other boxes, then push. Re-plan after each step.

    Key improvements over naive approach:
    - Avoids HAZARD terrain during BFS navigation (hazards kill the agent).
    - Skips boxes that are completely deadlocked.
    - Tries all 4 push directions (not just the 2 toward target).
    - Detects corner and wall-line deadlocks to avoid creating them.
    - Uses greedy box-target assignment instead of first-fit.
    """

    def _get_hazard_cells(self, grid):
        """Return set of all HAZARD terrain positions."""
        hazards = set()
        for y in range(grid.height):
            for x in range(grid.width):
                if grid.terrain[y, x] == CellType.HAZARD:
                    hazards.add((x, y))
        return hazards

    def _get_uncovered_holes(self, grid):
        """Return set of HOLE positions not yet covered by a box (metadata < 100)."""
        holes = set()
        for y in range(grid.height):
            for x in range(grid.width):
                if (
                    int(grid.terrain[y, x]) == int(CellType.HOLE)
                    and int(grid.metadata[y, x]) < 100
                ):
                    holes.add((x, y))
        return holes

    def _is_blocked(self, x, y, grid):
        """Check if position is impassable (wall, hazard, or border)."""
        if not (0 < x < grid.width - 1 and 0 < y < grid.height - 1):
            return True
        t = int(grid.terrain[y, x])
        return t in (int(CellType.WALL), int(CellType.HAZARD))

    def _is_corner_deadlock(self, x, y, grid, target_set):
        """Check if a box at (x,y) would be in a corner deadlock.

        A corner deadlock means the box is against walls/hazards/borders
        on two perpendicular sides and is not on a target.
        """
        if (x, y) in target_set:
            return False
        left = self._is_blocked(x - 1, y, grid)
        right = self._is_blocked(x + 1, y, grid)
        up = self._is_blocked(x, y - 1, grid)
        down = self._is_blocked(x, y + 1, grid)
        return (left or right) and (up or down)

    def _is_wall_deadlock(self, x, y, grid, target_set):
        """Check if a box at (x,y) is deadlocked against a wall line.

        A box against a wall with no target reachable along that wall
        segment can never be moved to any target.
        """
        if (x, y) in target_set:
            return False
        # Check horizontal wall contact (top or bottom blocked)
        for wall_dy in [-1, 1]:
            if self._is_blocked(x, y + wall_dy, grid):
                # Scan the contiguous wall-adjacent corridor for a target
                has_target = False
                # Scan left
                cx = x
                while 0 < cx < grid.width - 1 and not self._is_blocked(cx, y, grid):
                    if (cx, y) in target_set:
                        has_target = True
                        break
                    cx -= 1
                if not has_target:
                    # Scan right
                    cx = x + 1
                    while 0 < cx < grid.width - 1 and not self._is_blocked(cx, y, grid):
                        if (cx, y) in target_set:
                            has_target = True
                            break
                        cx += 1
                if not has_target:
                    return True
        # Check vertical wall contact (left or right blocked)
        for wall_dx in [-1, 1]:
            if self._is_blocked(x + wall_dx, y, grid):
                has_target = False
                cy = y
                while 0 < cy < grid.height - 1 and not self._is_blocked(x, cy, grid):
                    if (x, cy) in target_set:
                        has_target = True
                        break
                    cy -= 1
                if not has_target:
                    cy = y + 1
                    while 0 < cy < grid.height - 1 and not self._is_blocked(x, cy, grid):
                        if (x, cy) in target_set:
                            has_target = True
                            break
                        cy += 1
                if not has_target:
                    return True
        return False

    def _is_border_adjacent(self, x, y, grid):
        """Check if (x,y) is adjacent to the outer border wall."""
        return x <= 1 or x >= grid.width - 2 or y <= 1 or y >= grid.height - 2

    def _would_deadlock(self, lx, ly, grid, target_set):
        """Check if pushing a box to (lx, ly) would cause a deadlock.

        Always checks corner deadlocks (two perpendicular walls).
        Also checks wall-line deadlocks for border-adjacent positions,
        since border walls are permanent and a box against the border
        can never be pulled off.
        """
        if (lx, ly) in target_set:
            return False
        if self._is_corner_deadlock(lx, ly, grid, target_set):
            return True
        # Wall-line deadlock is only reliable near border walls
        if self._is_border_adjacent(lx, ly, grid):
            if self._is_wall_deadlock(lx, ly, grid, target_set):
                return True
        return False

    def _can_push_any_direction(self, bx, by, grid, box_set, target_set):
        """Check if a box can be pushed in at least one non-deadlocking dir."""
        for pdx, pdy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            push_from = (bx - pdx, by - pdy)
            land = (bx + pdx, by + pdy)
            pfx, pfy = push_from
            lx, ly = land
            if not (0 < pfx < grid.width - 1 and 0 < pfy < grid.height - 1):
                continue
            if self._is_blocked(pfx, pfy, grid):
                continue
            if push_from in box_set:
                continue
            if not (0 < lx < grid.width - 1 and 0 < ly < grid.height - 1):
                continue
            if self._is_blocked(lx, ly, grid):
                continue
            if land in box_set:
                continue
            return True
        return False

    def _assign_boxes_to_targets(self, boxes, targets, box_set, grid, target_set):
        """Greedy assignment of boxes to targets, skipping deadlocked boxes.

        Returns list of (box_pos, target_pos) pairs.
        """
        # Identify which targets are already occupied by boxes ON them
        occupied_targets = set()
        for bx, by in boxes:
            if (bx, by) in target_set:
                occupied_targets.add((bx, by))

        # Unplaced boxes that can still be pushed
        unplaced = []
        for bx, by in boxes:
            if (bx, by) in target_set:
                continue
            if not self._can_push_any_direction(bx, by, grid, box_set, target_set):
                continue
            unplaced.append((bx, by))

        free_targets = [tuple(t) for t in targets if tuple(t) not in occupied_targets]

        # Greedy assignment: repeatedly pick the closest (box, target) pair
        assignments = []
        remaining_boxes = list(unplaced)
        remaining_targets = list(free_targets)
        while remaining_boxes and remaining_targets:
            best_dist = 999999
            best_pair = None
            for b in remaining_boxes:
                for t in remaining_targets:
                    d = abs(b[0] - t[0]) + abs(b[1] - t[1])
                    if d < best_dist:
                        best_dist = d
                        best_pair = (b, t)
            if best_pair is None:
                break
            assignments.append(best_pair)
            remaining_boxes.remove(best_pair[0])
            remaining_targets.remove(best_pair[1])

        return assignments

    def _try_push(self, bx, by, tx, ty, ax, ay, grid, box_set, hazards, target_set, holes=None):
        """Try to find a valid push for box at (bx,by) toward target (tx,ty).

        Tries direct pushes first, then perpendicular pushes as fallback.
        Rejects pushes that would create deadlocks. Returns a single action
        integer if a valid push (or navigation step) is found, else None.
        """
        # Build ordered push directions: direct toward target first,
        # then all remaining directions
        push_attempts = []
        if abs(tx - bx) >= abs(ty - by):
            if tx != bx:
                push_attempts.append((1 if tx > bx else -1, 0))
            if ty != by:
                push_attempts.append((0, 1 if ty > by else -1))
        else:
            if ty != by:
                push_attempts.append((0, 1 if ty > by else -1))
            if tx != bx:
                push_attempts.append((1 if tx > bx else -1, 0))

        # Add all other directions not already listed
        for d in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            if d not in push_attempts:
                push_attempts.append(d)

        for pdx, pdy in push_attempts:
            push_from = (bx - pdx, by - pdy)
            pfx, pfy = push_from

            if not (0 < pfx < grid.width - 1 and 0 < pfy < grid.height - 1):
                continue
            if self._is_blocked(pfx, pfy, grid):
                continue
            if push_from in box_set:
                continue

            land = (bx + pdx, by + pdy)
            lx, ly = land
            if not (0 < lx < grid.width - 1 and 0 < ly < grid.height - 1):
                continue
            if self._is_blocked(lx, ly, grid):
                continue
            if land in box_set:
                continue

            # Avoid creating deadlocks (unless landing on a target)
            if self._would_deadlock(lx, ly, grid, target_set):
                continue

            # One-step look-ahead: can the box continue toward target from landing?
            if (lx, ly) != (tx, ty):  # Not already at target
                can_progress = False
                for ndx, ndy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    next_land = (lx + ndx, ly + ndy)
                    nlx, nly = next_land
                    # Check if next push would reduce distance to target
                    current_dist = abs(lx - tx) + abs(ly - ty)
                    next_dist = abs(nlx - tx) + abs(nly - ty)
                    if next_dist >= current_dist:
                        continue
                    # Check if next push is physically possible
                    next_push_from = (lx - ndx, ly - ndy)
                    npfx, npfy = next_push_from
                    if not (0 < npfx < grid.width - 1 and 0 < npfy < grid.height - 1):
                        continue
                    if self._is_blocked(npfx, npfy, grid):
                        continue
                    if not (0 < nlx < grid.width - 1 and 0 < nly < grid.height - 1):
                        continue
                    if self._is_blocked(nlx, nly, grid):
                        continue
                    can_progress = True
                    break
                if not can_progress:
                    continue  # Skip this push direction, try next

            if (ax, ay) == push_from:
                step = self.api.step_action(pdx, pdy)
                if step is not None:
                    return step
            else:
                # Navigate to push position avoiding boxes, hazards, and uncovered holes
                avoid = box_set | hazards | (holes or set())
                # HOLE terrain at target positions is passable
                target_passable = set(
                    tuple(t)
                    for t in self.api.task_config.get("target_positions", [])
                )
                path = self.api.bfs_path_positions(
                    (ax, ay),
                    push_from,
                    avoid=avoid,
                    extra_passable=target_passable,
                )
                if path:
                    actions = self.api.positions_to_actions(path)
                    if actions:
                        return actions[0]

        return None

    def plan(self):
        grid = self.api.grid
        config = self.api.task_config
        targets = config.get("target_positions", [])

        # Find current box positions from grid, skipping boxes already fixed
        # on a target hole (metadata >= 100 means covered/locked in place).
        boxes = []
        for y in range(grid.height):
            for x in range(grid.width):
                if grid.objects[y, x] == ObjectType.BOX:
                    if int(grid.metadata[y, x]) >= 100:
                        continue  # already fixed on target, nothing to do
                    boxes.append((x, y))

        if not boxes or not targets:
            return

        ax, ay = self.api.agent_position
        box_set = set(boxes)
        target_set = set(tuple(t) for t in targets)
        hazards = self._get_hazard_cells(grid)
        holes = self._get_uncovered_holes(grid)

        # Assign boxes to targets, skipping deadlocked boxes
        assignments = self._assign_boxes_to_targets(
            boxes,
            targets,
            box_set,
            grid,
            target_set,
        )
        if not assignments:
            return

        # Try each assigned (box, target) pair in order
        for (bx, by), (tx, ty) in assignments:
            if (bx, by) == (tx, ty):
                continue

            action = self._try_push(
                bx,
                by,
                tx,
                ty,
                ax,
                ay,
                grid,
                box_set,
                hazards,
                target_set,
                holes=holes,
            )
            if action is not None:
                self.action_queue = [action]
                return


@register_oracle("KeyDoorPuzzle-v0")
class KeyDoorPuzzleOracle(OracleAgent):
    """Collect color-coded keys to unlock matching doors, then reach goal.

    Strategy with 1-key inventory limit:
    1. Determine the next closed door in sequence.
    2. If holding the correct key, navigate adjacent to that door, face it,
       then INTERACT to open it.
    3. If holding the wrong key or no key, go pick up the correct key
       (keys are walkable -- auto-pickup on walk-over).
    4. After all doors are open, head to goal.

    Doors are solid (non-walkable). Keys are walkable.
    """

    def _get_closed_doors(self):
        """Return dict of {(x, y): color} for all closed doors on the grid."""
        grid = self.api.grid
        doors = {}
        for y in range(grid.height):
            for x in range(grid.width):
                if grid.objects[y, x] == ObjectType.DOOR:
                    meta = int(grid.metadata[y, x])
                    if meta < 10:
                        doors[(x, y)] = meta
        return doors

    def _get_keys_on_grid(self):
        """Return dict of {(x, y): color} for all keys on the grid."""
        grid = self.api.grid
        keys = {}
        for y in range(grid.height):
            for x in range(grid.width):
                if grid.objects[y, x] == ObjectType.KEY:
                    keys[(x, y)] = int(grid.metadata[y, x])
        return keys

    def _navigate_to_walkable(self, target, avoid=None):
        """BFS to a walkable target (like a key or goal). Returns True on success."""
        ax, ay = self.api.agent_position
        av = avoid or set()
        path = self.api.bfs_path_positions(
            (ax, ay),
            target,
            avoid=av - {target},
        )
        if path:
            actions = self.api.positions_to_actions(path)
            if actions:
                self.action_queue = [actions[0]]
                return True
        # Fallback without avoidance
        path = self.api.bfs_path_positions((ax, ay), target)
        if path:
            actions = self.api.positions_to_actions(path)
            if actions:
                self.action_queue = [actions[0]]
                return True
        self.action_queue = self.api.move_toward(*target)
        return True

    def plan(self):
        ax, ay = self.api.agent_position
        grid = self.api.grid

        inv_keys = [e for e in self.api.agent.inventory if e.entity_type == "key"]
        held_color = inv_keys[0].properties.get("color", 0) if inv_keys else None
        closed_doors = self._get_closed_doors()
        keys_on_grid = self._get_keys_on_grid()
        all_key_pos = set(keys_on_grid.keys())

        if closed_doors:
            next_door_pos = min(closed_doors, key=lambda p: closed_doors[p])
            next_door_color = closed_doors[next_door_pos]

            if held_color == next_door_color:
                # Go open door -- door is solid, use face-and-interact
                agent_ori = self.api.agent.orientation
                actions = _interact_adjacent(
                    (ax, ay), agent_ori, next_door_pos, grid, self.api,
                )
                if actions:
                    self.action_queue = actions
                return

            # Find the correct key on the grid
            target_key_pos = None
            for kp, kc in keys_on_grid.items():
                if kc == next_door_color:
                    target_key_pos = kp
                    break

            if target_key_pos is not None:
                # Key at our feet -- step off first
                if target_key_pos == (ax, ay):
                    for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                        nx, ny = ax + dx, ay + dy
                        if (
                            0 <= nx < grid.width
                            and 0 <= ny < grid.height
                            and int(grid.terrain[ny, nx]) == int(CellType.EMPTY)
                            and not grid.is_object_blocking((nx, ny))
                            and int(grid.objects[ny, nx]) != int(ObjectType.KEY)
                        ):
                            self.action_queue = self.api.positions_to_actions(
                                [(nx, ny)]
                            )
                            return
                    for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                        nx, ny = ax + dx, ay + dy
                        if (
                            0 <= nx < grid.width
                            and 0 <= ny < grid.height
                            and int(grid.terrain[ny, nx]) == int(CellType.EMPTY)
                            and not grid.is_object_blocking((nx, ny))
                        ):
                            self.action_queue = self.api.positions_to_actions(
                                [(nx, ny)]
                            )
                            return
                    return

                # Navigate to key -- keys are walkable, avoid other keys and
                # closed doors (doors are solid so BFS won't cross them anyway,
                # but adding to avoid set ensures no path through them)
                other_keys = all_key_pos - {target_key_pos}
                door_avoid = set(closed_doors.keys())
                self._navigate_to_walkable(
                    target_key_pos,
                    avoid=other_keys | door_avoid,
                )
                return

            # No matching key -- try nearest available
            if keys_on_grid:
                nearest = min(
                    keys_on_grid,
                    key=lambda p: abs(p[0] - ax) + abs(p[1] - ay),
                )
                self._navigate_to_walkable(nearest)
                return

        # All doors open -- head to goal
        goal = self.api.get_nearest("goal")
        if goal:
            self._navigate_to_walkable(goal.position)


@register_oracle("BacktrackPuzzle-v0")
class BacktrackPuzzleOracle(OracleAgent):
    """Activate all switches, then go to goal.

    Switches are solid (non-walkable). The agent must stand adjacent,
    face the switch, then INTERACT. Activated switches become passable
    so the agent can walk through them to reach the next switch.
    """

    def _activated_switch_positions(self, grid, config):
        """Return set of switch positions that have been activated (passable)."""
        activated = set()
        for sw in config.get("switch_positions", []):
            sx, sy = sw
            if int(grid.metadata[sy, sx]) >= 100:
                activated.add((sx, sy))
        return activated

    def _find_reachable_switch(self, unactivated, passable, agent_pos, grid):
        """Find the nearest unactivated switch reachable via BFS.

        Checks each unactivated switch's adjacent walkable cells and picks
        the one with the shortest BFS path from the agent. Uses *passable*
        (activated switch positions) as extra_passable for BFS.
        """
        best_switch = None
        best_dist = float("inf")
        ep = passable or None

        for sw in unactivated:
            sx, sy = sw.position
            # Find walkable cells adjacent to this switch
            for ddx, ddy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = sx + ddx, sy + ddy
                adj = (nx, ny)
                if not grid.in_bounds(adj):
                    continue
                if not grid.is_walkable(adj):
                    continue
                if grid.is_object_blocking(adj) and (ep is None or adj not in ep):
                    continue
                path = self.api.bfs_path_positions(
                    agent_pos, adj, extra_passable=ep,
                )
                if path and len(path) < best_dist:
                    best_dist = len(path)
                    best_switch = sw
        return best_switch

    def plan(self):
        config = self.api.task_config
        all_activated = config.get("_all_activated", False)

        if not all_activated:
            switches = self.api.get_entities_of_type("switch")
            grid = self.api.grid
            # Filter out already-activated switches (metadata >= 100)
            unactivated = [
                s for s in switches
                if int(grid.metadata[s.position[1], s.position[0]]) < 100
            ]
            if unactivated:
                agent_pos = self.api.agent_position
                agent_ori = self.api.agent.orientation
                # Activated switches are passable — include them as
                # extra_passable so BFS can route through them.
                passable = self._activated_switch_positions(grid, config)

                # Pick the nearest REACHABLE switch (not just Manhattan nearest)
                target = self._find_reachable_switch(
                    unactivated, passable, agent_pos, grid,
                )
                if target is None:
                    # Fallback: try Manhattan nearest
                    target = min(unactivated, key=lambda s: s.distance)

                actions = _interact_adjacent(
                    agent_pos, agent_ori, target.position, grid, self.api,
                    extra_passable=passable or None,
                )
                if actions:
                    self.action_queue = actions
                return

        # All switches activated — navigate to goal.
        # Activated switches are passable in the task's can_agent_enter,
        # so we pass them as extra_passable for BFS too.
        goal = self.api.get_nearest("goal")
        if goal:
            grid = self.api.grid
            passable = self._activated_switch_positions(grid, config)
            ax, ay = self.api.agent_position
            gx, gy = goal.position
            path = self.api.bfs_path_positions(
                (ax, ay), (gx, gy), extra_passable=passable or None,
            )
            if path:
                actions = self.api.positions_to_actions(path)
                if actions:
                    self.action_queue = [actions[0]]
                    return
            self.action_queue = self.api.move_to(gx, gy)


def _solve_sliding_puzzle(n, board, blank_pos):
    """Solve an NxN sliding puzzle using IDA* with Manhattan distance heuristic.

    Args:
        n: Puzzle dimension (e.g. 3 for 8-puzzle, 4 for 15-puzzle).
        board: Tuple of length n*n. board[row*n + col] = tile number (1..n*n-1)
               or 0 for the blank.
        blank_pos: Index of the blank in board (row*n + col).

    Returns:
        List of blank-move directions as (dr, dc) tuples, where each move
        swaps the blank with the tile at blank + (dr, dc). Returns empty
        list if already solved.
    """
    goal = tuple(list(range(1, n * n)) + [0])
    if board == goal:
        return []

    # Precompute goal positions: goal_row[tile], goal_col[tile]
    goal_row = [0] * (n * n)
    goal_col = [0] * (n * n)
    for i in range(n * n):
        tile = goal[i]
        goal_row[tile] = i // n
        goal_col[tile] = i % n

    def manhattan(state):
        h = 0
        for i in range(n * n):
            tile = state[i]
            if tile == 0:
                continue
            r, c = i // n, i % n
            h += abs(r - goal_row[tile]) + abs(c - goal_col[tile])
        return h

    def linear_conflict(state):
        """Manhattan distance + linear conflict heuristic."""
        h = 0
        conflict = 0
        for i in range(n * n):
            tile = state[i]
            if tile == 0:
                continue
            r, c = i // n, i % n
            h += abs(r - goal_row[tile]) + abs(c - goal_col[tile])

        # Row conflicts
        for row in range(n):
            for i in range(n):
                ti = state[row * n + i]
                if ti == 0 or goal_row[ti] != row:
                    continue
                for j in range(i + 1, n):
                    tj = state[row * n + j]
                    if tj == 0 or goal_row[tj] != row:
                        continue
                    if goal_col[ti] > goal_col[tj]:
                        conflict += 2

        # Column conflicts
        for col in range(n):
            for i in range(n):
                ti = state[i * n + col]
                if ti == 0 or goal_col[ti] != col:
                    continue
                for j in range(i + 1, n):
                    tj = state[j * n + col]
                    if tj == 0 or goal_col[tj] != col:
                        continue
                    if goal_row[ti] > goal_row[tj]:
                        conflict += 2

        return h + conflict

    heuristic = linear_conflict
    moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    threshold = heuristic(board)
    path = []  # list of (dr, dc) moves

    def search(state, blank, g, bound, last_move):
        f = g + heuristic(state)
        if f > bound:
            return f
        if state == goal:
            return -1  # found

        min_t = float("inf")
        for dr, dc in moves:
            # Don't undo the previous move
            if last_move is not None and (dr + last_move[0] == 0 and dc + last_move[1] == 0):
                continue
            br, bc = blank // n, blank % n
            nr, nc = br + dr, bc + dc
            if 0 <= nr < n and 0 <= nc < n:
                new_blank = nr * n + nc

                # Swap
                lst = list(state)
                lst[blank], lst[new_blank] = lst[new_blank], lst[blank]
                new_state = tuple(lst)

                path.append((dr, dc))
                t = search(new_state, new_blank, g + 1, bound, (dr, dc))
                if t == -1:
                    return -1
                if t < min_t:
                    min_t = t
                path.pop()

        return min_t

    # IDA* loop
    while True:
        t = search(board, blank_pos, 0, threshold, None)
        if t == -1:
            return list(path)
        if t == float("inf"):
            return []  # unsolvable
        threshold = t


@register_oracle("TileSorting-v0")
class TileSortingOracle(OracleAgent):
    """Sliding puzzle solver using IDA* search.

    Agent IS the empty slot. Moving into a BOX tile slides it to agent's
    old position. Uses IDA* with Manhattan distance + linear conflict
    heuristic to find optimal solutions.
    """

    def __init__(self, env):
        super().__init__(env)
        self._solution = []  # Precomputed full solution as action ints

    def reset(self, obs, info):
        self._solution = []
        super().reset(obs, info)

    def plan(self):
        # If we have a precomputed solution, use it
        if self._solution:
            self.action_queue = self._solution
            self._solution = []
            return

        config = self.api.task_config
        goal_map = config.get("goal_map", {})
        puzzle_size = config.get("puzzle_size", 3)
        offset_x = config.get("offset_x", 0)
        offset_y = config.get("offset_y", 0)
        grid = self.api.grid
        ax, ay = self.api.agent_position

        # Build abstract puzzle board from current grid state
        n = puzzle_size
        board = [0] * (n * n)
        blank_idx = -1

        for py in range(n):
            for px in range(n):
                gx = offset_x + px
                gy = offset_y + py
                idx = py * n + px
                if (gx, gy) == (ax, ay):
                    board[idx] = 0
                    blank_idx = idx
                elif grid.objects[gy, gx] == ObjectType.BOX:
                    raw_tn = int(grid.metadata[gy, gx])
                    tn = raw_tn - 100 if raw_tn >= 100 else raw_tn
                    board[idx] = tn
                else:
                    # Cell should have a tile or be the blank
                    # If it's a TARGET marker, it's the blank equivalent
                    board[idx] = 0
                    if blank_idx == -1:
                        blank_idx = idx

        if blank_idx == -1:
            self.action_queue = [0]
            return

        # Remap tile numbers to canonical 1..n*n-1 based on goal positions
        # goal_map maps str(tile_num) -> [gx, gy]
        # We need: goal index = (gy - offset_y) * n + (gx - offset_x)
        # and the tile at that goal index should be goal_idx + 1 in canonical form
        tile_to_canonical = {}
        for tn_str, gpos in goal_map.items():
            tn = int(tn_str)
            gx, gy = gpos
            goal_idx = (gy - offset_y) * n + (gx - offset_x)
            canonical = goal_idx + 1  # 1-based; last cell (n*n) wraps to 0 (blank)
            if canonical == n * n:
                canonical = 0
            tile_to_canonical[tn] = canonical

        # Remap the board
        canonical_board = []
        for val in board:
            if val == 0:
                canonical_board.append(0)
            else:
                canonical_board.append(tile_to_canonical.get(val, val))

        board_tuple = tuple(canonical_board)

        # Solve with IDA*
        moves = _solve_sliding_puzzle(n, board_tuple, blank_idx)

        if not moves:
            # Already solved or unsolvable (shouldn't happen)
            self.action_queue = [0]
            return

        # Convert abstract moves to agent actions
        # Each move (dr, dc) means the blank moves by (dr, dc) in the abstract grid
        # In the real grid, dr maps to dy, dc maps to dx
        actions = []
        for dr, dc in moves:
            # blank moves by (dc, dr) in grid coordinates (dx, dy)
            dx, dy = dc, dr
            step = self.api.step_action(dx, dy)
            if step is not None:
                actions.append(step)

        self._solution = actions[1:] if len(actions) > 1 else []
        self.action_queue = [actions[0]] if actions else [0]


@register_oracle("PackingPuzzle-v0")
class PackingPuzzleOracle(OracleAgent):
    """Push typed pieces into matching target slots.

    Strategy per piece — vertical-first to avoid wrong-type targets:
      1. Align the piece vertically with its target row.
      2. Push the piece horizontally into its target column.

    Pieces already vertically aligned (only needing horizontal push)
    are processed first.  When a push is blocked, the oracle tries
    to clear the blocking piece or attempts a perpendicular detour.
    """

    _PIECE_TYPES = {5, 14, 19, 17}  # BOX, GEM, ORB, SCROLL

    def plan(self):  # noqa: C901
        grid = self.api.grid
        ax, ay = self.api.agent_position

        # ------------------------------------------------------------------
        # Scan grid
        # ------------------------------------------------------------------
        pieces: list[tuple[int, int, int]] = []
        targets: list[tuple[int, int, int]] = []
        for y in range(grid.height):
            for x in range(grid.width):
                obj = int(grid.objects[y, x])
                if obj in self._PIECE_TYPES:
                    pieces.append((x, y, obj))
                elif obj == int(ObjectType.TARGET):
                    meta = int(grid.metadata[y, x])
                    targets.append((x, y, meta))

        if not pieces or not targets:
            self.action_queue = [0]
            return

        piece_set = {(px, py) for px, py, _ in pieces}
        piece_at: dict[tuple[int, int], int] = {(px, py): pt for px, py, pt in pieces}

        def _ib(x, y):
            return 0 < x < grid.width - 1 and 0 < y < grid.height - 1

        # ------------------------------------------------------------------
        # _can_push: matches task's can_agent_enter exactly.
        #   Landing allowed on: NONE, TARGET (any type).
        #   Landing blocked on: GOAL, pieces, walls, OOB.
        #   When strict=True (default), also blocks wrong-type TARGET.
        # ------------------------------------------------------------------
        def _can_push(px, py, ptype, ddx, ddy, strict=True):
            pfx, pfy = px - ddx, py - ddy
            if not _ib(pfx, pfy):
                return False
            if int(grid.terrain[pfy, pfx]) != 0:
                return False
            if (pfx, pfy) in piece_at:
                return False
            lx, ly = px + ddx, py + ddy
            if not _ib(lx, ly):
                return False
            if int(grid.terrain[ly, lx]) == int(CellType.WALL):
                return False
            if (lx, ly) in piece_at:
                return False
            land_obj = int(grid.objects[ly, lx])
            if land_obj == int(ObjectType.TARGET):
                if strict and int(grid.metadata[ly, lx]) != int(ptype):
                    return False
            elif land_obj != 0:  # only NONE (0) is passable
                return False
            return True

        # ------------------------------------------------------------------
        # _blocker_at: which piece blocks a push?
        # ------------------------------------------------------------------
        def _blocker_at(px, py, ddx, ddy):
            pfx, pfy = px - ddx, py - ddy
            if _ib(pfx, pfy) and (pfx, pfy) in piece_at:
                return (pfx, pfy)
            lx, ly = px + ddx, py + ddy
            if _ib(lx, ly) and (lx, ly) in piece_at:
                return (lx, ly)
            return None

        # ------------------------------------------------------------------
        # _try_execute: navigate to push-from and push
        # ------------------------------------------------------------------
        def _try_execute(px, py, ddx, ddy):
            push_from = (px - ddx, py - ddy)
            if (ax, ay) == push_from:
                step = self.api.step_action(ddx, ddy)
                if step is not None:
                    self.action_queue = [step]
                    return True
            else:
                path = self.api.bfs_path_positions(
                    (ax, ay),
                    push_from,
                    avoid=piece_set,
                )
                if not path:
                    # Relax avoidance: allow walking through any cell except
                    # the piece we are about to push
                    path = self.api.bfs_path_positions(
                        (ax, ay),
                        push_from,
                        avoid={(px, py)},
                    )
                if path:
                    actions = self.api.positions_to_actions(path)
                    if actions:
                        self.action_queue = [actions[0]]
                        return True
            return False

        # ------------------------------------------------------------------
        # _try_clear_blocker: push blocking piece in any safe direction.
        # If the blocker itself is stuck, recursively clear *its*
        # blocker (up to ``depth`` levels).
        # ------------------------------------------------------------------
        def _try_clear_blocker(bx, by, depth=2):
            btype = piece_at.get((bx, by))
            if btype is None:
                return False
            for ddx, ddy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                if _can_push(bx, by, btype, ddx, ddy):
                    if _try_execute(bx, by, ddx, ddy):
                        return True
            # Try relaxed pushes (allow wrong-type target landing)
            for ddx, ddy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                if _can_push(bx, by, btype, ddx, ddy, strict=False):
                    if _try_execute(bx, by, ddx, ddy):
                        return True
            if depth > 0:
                for ddx, ddy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    bb = _blocker_at(bx, by, ddx, ddy)
                    if bb and bb != (bx, by):
                        if _try_clear_blocker(*bb, depth=depth - 1):
                            return True
            return False

        # ------------------------------------------------------------------
        # Build piece->target assignments (greedy, no duplicate targets)
        # ------------------------------------------------------------------
        assignments: list[tuple[int, int, int, int, int]] = []
        claimed: set[tuple[int, int]] = set()
        pdists = []
        for px, py, ptype in pieces:
            matching = [(tx, ty) for tx, ty, tt in targets if tt == ptype]
            if not matching:
                continue
            best = min(matching, key=lambda t: abs(t[0] - px) + abs(t[1] - py))
            pdists.append((abs(best[0] - px) + abs(best[1] - py), px, py, ptype))
        pdists.sort()
        for _, px, py, ptype in pdists:
            avail = [(tx, ty) for tx, ty, tt in targets
                     if tt == ptype and (tx, ty) not in claimed]
            if not avail:
                continue
            tx, ty = min(avail, key=lambda t: abs(t[0] - px) + abs(t[1] - py))
            if (px, py) == (tx, ty):
                continue
            assignments.append((px, py, ptype, tx, ty))
            claimed.add((tx, ty))

        if not assignments:
            self.action_queue = [0]
            return

        # ------------------------------------------------------------------
        # Sort: already-aligned first (only horizontal push needed),
        #       then by total Manhattan distance
        # ------------------------------------------------------------------
        def _sort_key(a):
            px, py, ptype, tx, ty = a
            needs_vert = 0 if py == ty else 1
            return (needs_vert, abs(px - tx) + abs(py - ty))

        assignments.sort(key=_sort_key)

        # ------------------------------------------------------------------
        # Main loop: vertical-first push strategy
        # ------------------------------------------------------------------
        for px, py, ptype, tx, ty in assignments:
            # --- Phase 1: Vertical alignment ---
            if py != ty:
                ddy = 1 if ty > py else -1
                if _can_push(px, py, ptype, 0, ddy):
                    if _try_execute(px, py, 0, ddy):
                        return
                else:
                    bpos = _blocker_at(px, py, 0, ddy)
                    if bpos and _try_clear_blocker(*bpos):
                        return

            # --- Phase 2: Horizontal alignment ---
            if px != tx:
                ddx = 1 if tx > px else -1
                if _can_push(px, py, ptype, ddx, 0):
                    if _try_execute(px, py, ddx, 0):
                        return
                else:
                    bpos = _blocker_at(px, py, ddx, 0)
                    if bpos and _try_clear_blocker(*bpos):
                        return

            # --- Phase 3: Perpendicular detour ---
            # If vertical push was blocked, try pushing piece horizontally
            # (detour) to open up the vertical path from a new column.
            if py != ty:
                ddy = 1 if ty > py else -1
                for ddx_det in [1, -1]:
                    if _can_push(px, py, ptype, ddx_det, 0):
                        if _try_execute(px, py, ddx_det, 0):
                            return

            # If horizontal push was blocked, try pushing piece vertically
            # to open up a horizontal path from a new row.
            if px != tx and py == ty:
                for ddy_det in [1, -1]:
                    if _can_push(px, py, ptype, 0, ddy_det):
                        if _try_execute(px, py, 0, ddy_det):
                            return

            # --- Phase 4: Relaxed pushes (allow wrong-type target) ---
            if py != ty:
                ddy = 1 if ty > py else -1
                if _can_push(px, py, ptype, 0, ddy, strict=False):
                    if _try_execute(px, py, 0, ddy):
                        return
            if px != tx:
                ddx = 1 if tx > px else -1
                if _can_push(px, py, ptype, ddx, 0, strict=False):
                    if _try_execute(px, py, ddx, 0):
                        return

        # ------------------------------------------------------------------
        # Fallback: move toward nearest unfinished piece
        # ------------------------------------------------------------------
        if assignments:
            self.action_queue = self.api.move_toward(
                assignments[0][0],
                assignments[0][1],
            )
            return
        if pieces:
            self.action_queue = self.api.move_toward(pieces[0][0], pieces[0][1])


@register_oracle("PreciseNavigation-v0")
class PreciseNavigationOracle(OracleAgent):
    """Ice-sliding puzzle oracle: BFS over slide endpoints to reach goal.

    The grid is mostly ICE — agent slides until hitting WALL/EMPTY/edge.
    EMPTY cells are stopping points. BFS explores (direction, endpoint) pairs.
    """

    _DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right
    _DIR_ACTIONS = [1, 2, 3, 4]  # ActionType: UP=1, DOWN=2, LEFT=3, RIGHT=4

    def _simulate_slide(self, x, y, dx, dy):
        """Simulate sliding from (x,y) in direction (dx,dy). Return endpoint."""
        grid = self.api.grid
        cx, cy = x, y
        while True:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < grid.width and 0 <= ny < grid.height):
                break  # hit edge
            terrain = int(grid.terrain[ny, nx])
            if terrain == int(CellType.WALL):
                break  # hit wall
            obj = int(grid.objects[ny, nx])
            if obj == int(ObjectType.BOX):
                break  # hit box
            if terrain == int(CellType.EMPTY) or terrain == 0:
                # EMPTY is a stopping point — land on it if different from start
                if (nx, ny) != (x, y):
                    return (nx, ny)
                break
            # ICE — keep sliding
            cx, cy = nx, ny
        # Stopped on last valid cell
        if (cx, cy) != (x, y):
            return (cx, cy)
        return None  # didn't move

    def plan(self):
        ax, ay = self.api.agent_position
        goal = self.api.get_nearest("goal")
        if not goal:
            return

        gx, gy = goal.position

        # BFS over slide-reachable positions
        visited = {(ax, ay)}
        queue = deque([(ax, ay, [])])  # (x, y, action_sequence)

        while queue:
            cx, cy, actions = queue.popleft()
            for i, (dx, dy) in enumerate(self._DIRS):
                endpoint = self._simulate_slide(cx, cy, dx, dy)
                if endpoint is None:
                    continue
                ex, ey = endpoint
                if endpoint == (gx, gy):
                    # Found path to goal
                    final_actions = actions + [self._DIR_ACTIONS[i]]
                    self.action_queue = [final_actions[0]]
                    return
                if endpoint not in visited:
                    visited.add(endpoint)
                    queue.append((ex, ey, actions + [self._DIR_ACTIONS[i]]))

        # Fallback: move toward goal directly
        self.action_queue = self.api.move_toward(gx, gy)


@register_oracle("RecipeAssembly-v0")
class RecipeAssemblyOracle(OracleAgent):
    """Follow recipe: collect ingredients in correct order, deliver to station.

    Strategy:
      1. Read current recipe step from config["_step"]
      2. If holding the needed ingredient → navigate to station (NPC position)
      3. Otherwise → navigate to nearest ingredient of needed type
      4. Repeat until all recipe steps are complete
    """

    _INGREDIENT_TYPE_NAMES = {
        14: "gem",    # GEM/herb
        17: "scroll", # SCROLL/mushroom
        19: "orb",    # ORB/crystal
        18: "coin",   # COIN/reagent
    }

    def _nav_to(self, tx, ty):
        """Navigate to target using BFS, fallback to move_toward."""
        ax, ay = self.api.agent_position
        path = self.api.bfs_path_positions((ax, ay), (tx, ty))
        if path:
            actions = self.api.positions_to_actions(path)
            if actions:
                self.action_queue = [actions[0]]
                return
        self.action_queue = self.api.move_toward(tx, ty)

    def plan(self):
        config = self.api.task_config
        recipe = config.get("recipe", [])
        step = config.get("_step", 0)
        ax, ay = self.api.agent_position

        if step >= len(recipe):
            self.action_queue = [0]
            return

        needed_type = recipe[step]
        entity_type_name = self._INGREDIENT_TYPE_NAMES.get(needed_type)

        # Check if holding the needed ingredient
        agent = self.api.agent
        holding_needed = any(
            e.entity_type == "ingredient"
            and e.properties.get("ing_type") == needed_type
            for e in agent.inventory
        )

        if holding_needed:
            # Navigate to station (NPC chef) — NOT "goal" which is recipe zone
            station = config.get("station_pos")
            if station:
                sx, sy = station
                self._nav_to(sx, sy)
            return

        # Find nearest ingredient of needed type on the grid
        if entity_type_name:
            ings = self.api.get_entities_of_type(entity_type_name)
            # Filter out recipe zone entities (TARGET/GOAL at y=1, right side)
            recipe_zone = set(tuple(p) for p in config.get("recipe_zone", []))
            ings = [e for e in ings if e.position not in recipe_zone]
            if ings:
                nearest = min(ings, key=lambda e: e.distance)
                self._nav_to(*nearest.position)
                return

        # Fallback: go to station
        station = config.get("station_pos")
        if station:
            sx, sy = station
            self._nav_to(sx, sy)


@register_oracle("ToolUse-v0")
class ToolUseOracle(OracleAgent):
    """Collect all scrolls to spawn the ORB, pick it up, then cross river to goal.

    Strategy:
      1. Collect all SCROLL objects (ignore COIN decoys).
      2. Once the ORB spawns, navigate to it and pick it up.
      3. Cross the river and reach the GOAL for full reward.

    All pickups are automatic on walk-over, so the oracle just needs to
    navigate to each target in the correct order.
    """

    def plan(self):
        config = self.api.task_config
        ax, ay = self.api.agent_position
        has_orb = config.get("_has_orb", False)
        orb_spawned = config.get("_orb_spawned", False)

        # --- Phase 3: Have the orb — head to goal ---
        if has_orb:
            goal = self.api.get_nearest("goal")
            if goal:
                path = self.api.bfs_path_positions(
                    (ax, ay), goal.position,
                )
                if path:
                    actions = self.api.positions_to_actions(path)
                    if actions:
                        self.action_queue = [actions[0]]
                        return
                self.action_queue = self.api.move_toward(*goal.position)
            return

        # --- Phase 2: All scrolls collected, orb spawned — go get the orb ---
        if orb_spawned:
            orbs = self.api.get_entities_of_type("orb")
            if orbs:
                nearest_orb = min(orbs, key=lambda e: e.distance)
                path = self.api.bfs_path_positions(
                    (ax, ay), nearest_orb.position,
                )
                if path:
                    actions = self.api.positions_to_actions(path)
                    if actions:
                        self.action_queue = [actions[0]]
                        return
                self.action_queue = self.api.move_toward(
                    *nearest_orb.position,
                )
            return

        # --- Phase 1: Collect scrolls (ignore coins) ---
        scrolls = self.api.get_entities_of_type("scroll")
        if scrolls:
            nearest_scroll = min(scrolls, key=lambda e: e.distance)
            path = self.api.bfs_path_positions(
                (ax, ay), nearest_scroll.position,
            )
            if path:
                actions = self.api.positions_to_actions(path)
                if actions:
                    self.action_queue = [actions[0]]
                    return
            self.action_queue = self.api.move_toward(
                *nearest_scroll.position,
            )
            return

        # Fallback: head to goal directly (may not have orb)
        goal = self.api.get_nearest("goal")
        if goal:
            self.action_queue = self.api.move_toward(*goal.position)


@register_oracle("ResourceManagement-v0")
class ResourceManagementOracle(OracleAgent):
    """Keep all energy stations alive for the full episode.

    No goal — success is surviving max_steps without any station dying.
    Oracle monitors energy levels and always visits the most critical station.
    """

    def _get_station_levels(self) -> list[tuple[int, int, int]]:
        """Return list of (energy_level, x, y) for all RESOURCE objects."""
        from agentick.core.types import ObjectType
        grid = self.api.grid
        stations = []
        for y in range(grid.height):
            for x in range(grid.width):
                if int(grid.objects[y, x]) == int(ObjectType.RESOURCE):
                    level = int(grid.metadata[y, x])
                    stations.append((level, x, y))
        return stations

    def plan(self):
        stations = self._get_station_levels()
        if not stations:
            return

        # Always go to the most critical station (lowest energy)
        stations.sort(key=lambda s: s[0])
        most_critical_level, cx, cy = stations[0]
        self.action_queue = self.api.move_to(cx, cy)
