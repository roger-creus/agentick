"""Oracle bots for combinatorial tasks."""

from __future__ import annotations

from collections import deque

from agentick.core.types import CellType, ObjectType
from agentick.oracles.base import OracleAgent
from agentick.oracles.registry import register_oracle


@register_oracle("GraphColoring-v0")
class GraphColoringOracle(OracleAgent):
    """Color graph nodes via INTERACT so no adjacent nodes share a color.

    Redesigned mechanic: no color stations. Agent walks to a node,
    then uses INTERACT (action 8) to cycle its color until it matches
    the target coloring. Greedy coloring computed from config adjacency.
    """

    def __init__(self, env):
        super().__init__(env)
        self._coloring = {}
        self._plan_computed = False

    def reset(self, obs, info):
        self._coloring = {}
        self._plan_computed = False
        super().reset(obs, info)

    def _compute_coloring(self):
        config = self.api.task_config
        nodes = config.get("node_positions", [])
        adj = config.get("adjacency", {})
        n = len(nodes)

        colors = {}
        for i in range(n):
            neighbors = adj.get(i, adj.get(str(i), []))
            used = set()
            for nb in neighbors:
                nb = int(nb)
                if nb in colors:
                    used.add(colors[nb])
            c = 1  # colors are 1-based in new design (0 = uncolored)
            while c in used:
                c += 1
            colors[i] = c

        self._coloring = colors
        self._plan_computed = True

    def plan(self):
        config = self.api.task_config
        if not self._plan_computed:
            self._compute_coloring()

        nodes = config.get("node_positions", [])
        n_colors = config.get("n_colors", 2)
        ax, ay = self.api.agent_position
        grid = self.api.grid

        # Find the next node that needs color adjustment
        for i, (nx, ny) in enumerate(nodes):
            target_color = self._coloring.get(i, 1)  # 1-based target color
            current_color = int(grid.metadata[ny, nx])  # 0=uncolored, 1-N=colors

            if current_color == target_color:
                continue  # already correct

            # Navigate to this node
            if (ax, ay) == (nx, ny):
                # We're at the node — INTERACT to cycle color
                # INTERACT action = 8
                interact_action = 8
                # We may need multiple INTERACTs to reach target_color
                # From current_color, cycles: current → (current+1)%(n_colors+1)
                # Count how many cycles needed
                cycles_needed = (target_color - current_color) % (n_colors + 1)
                if cycles_needed == 0:
                    cycles_needed = n_colors + 1  # full cycle
                self.action_queue = [interact_action] * cycles_needed
                return
            else:
                # Navigate to node
                path = self.api.bfs_path_positions((ax, ay), (nx, ny))
                if path:
                    actions = self.api.positions_to_actions(path)
                    if actions:
                        self.action_queue = [actions[0]]
                        return
                self.action_queue = self.api.move_toward(nx, ny)
                return

        # All nodes correctly colored
        self.action_queue = [0]


@register_oracle("LightsOut-v0")
class LightsOutOracle(OracleAgent):
    """Toggle switches to turn all lights off.

    For easy (no adjacent toggle): visit each lit switch to toggle it off.
    For medium+ (adjacent toggle): use simulation-based greedy planning
    that queues a full multi-leg route at once.

    Key mechanic: ``on_agent_moved`` fires on EVERY step.  In adjacent-toggle
    mode, stepping on *any* cell toggles the cell itself AND its 4 neighbours
    (if they are light-grid positions).  The oracle must account for **all**
    incidental toggles along the entire walk, not just at the destination.
    """

    def __init__(self, env):
        super().__init__(env)
        self._light_positions = set()
        self._all_light_positions = set()
        self._adjacent_toggle = False

    def reset(self, obs, info):
        config = self.api.task_config
        self._adjacent_toggle = config.get("adjacent_toggle", False)
        self._light_positions = set()
        self._all_light_positions = set()
        # Initially lit positions
        for p in config.get("light_positions", []):
            self._light_positions.add(tuple(p))
            self._all_light_positions.add(tuple(p))
        # Decoy positions (also part of toggle grid)
        for p in config.get("decoy_positions", []):
            self._all_light_positions.add(tuple(p))
        # ALL puzzle cells (structured grid) — in adjacent toggle mode,
        # every puzzle cell can be toggled by stepping near it
        for p in config.get("puzzle_cells", []):
            self._all_light_positions.add(tuple(p))
        super().reset(obs, info)

    def _get_lit_positions(self):
        """Return set of currently lit positions."""
        grid = self.api.grid
        lit = set()
        for pos in self._all_light_positions:
            px, py = pos
            if grid.objects[py, px] == ObjectType.SWITCH:
                lit.add(pos)
        return lit

    def _adjacency_zone(self, positions):
        """Return ``positions`` plus all 4-neighbours of each position."""
        zone = set(positions)
        for px, py in positions:
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                zone.add((px + dx, py + dy))
        return zone

    def _solve_gf2(self, lit_override=None):
        """Solve which light-grid positions to toggle to clear all lights.

        If *lit_override* is given it is used instead of reading the grid.
        """
        all_pos = sorted(self._all_light_positions)
        n = len(all_pos)
        if n == 0:
            return set()
        pos_to_idx = {pos: i for i, pos in enumerate(all_pos)}

        lit_set = lit_override if lit_override is not None else self._get_lit_positions()

        A = [[0] * n for _ in range(n)]
        for i, pos in enumerate(all_pos):
            px, py = pos
            A[i][i] = 1
            if self._adjacent_toggle:
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    npos = (px + dx, py + dy)
                    if npos in pos_to_idx:
                        A[pos_to_idx[npos]][i] = 1

        b = [1 if pos in lit_set else 0 for pos in all_pos]
        M = [A[r][:] + [b[r]] for r in range(n)]

        pivot_col = [None] * n
        row = 0
        for col in range(n):
            found = -1
            for r in range(row, n):
                if M[r][col] == 1:
                    found = r
                    break
            if found == -1:
                continue
            M[row], M[found] = M[found], M[row]
            pivot_col[col] = row
            for r in range(n):
                if r != row and M[r][col] == 1:
                    for j in range(n + 1):
                        M[r][j] ^= M[row][j]
            row += 1

        x = [0] * n
        for col in range(n):
            if pivot_col[col] is not None:
                x[col] = M[pivot_col[col]][n]

        return {all_pos[i] for i in range(n) if x[i] == 1}

    def _simulate_path_toggles(self, path, lit):
        """Simulate walking *path* and return the resulting lit set.

        Returns a new ``set`` -- *lit* is not mutated.
        """
        lit = set(lit)
        for pos in path[1:]:
            affected = {pos} if pos in self._all_light_positions else set()
            if self._adjacent_toggle:
                px, py = pos
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nb = (px + dx, py + dy)
                    if nb in self._all_light_positions:
                        affected.add(nb)
            for a in affected:
                lit.symmetric_difference_update({a})
        return lit

    def _candidate_paths(self, start, target, solution):
        """Return candidate BFS paths to *target* using tiered avoidance."""
        non_sol = self._all_light_positions - solution
        danger = self._adjacency_zone(non_sol)

        avoid_tiers = [
            (danger | self._all_light_positions) - {target},
            danger - {target} - solution,
            self._all_light_positions - {target},
            non_sol,
            None,
        ]
        paths = []
        seen = set()
        for avoid in avoid_tiers:
            if avoid is not None and not avoid:
                avoid = None
            path = self.api.bfs_path_positions(start, target, avoid=avoid)
            if path:
                key = tuple(path)
                if key not in seen:
                    seen.add(key)
                    paths.append(path)
        return paths

    def _pick_best_leg(self, start, targets, lit):
        """Pick the target + path that minimises remaining lit lights.

        Always returns *some* path if one exists, even if it temporarily
        increases the lit count (multi-leg routes may recover later).
        """
        best_path = None
        best_remaining = float("inf")
        best_lit = lit
        for target in sorted(targets):
            for path in self._candidate_paths(start, target, targets):
                result = self._simulate_path_toggles(path, lit)
                if len(result) < best_remaining:
                    best_remaining = len(result)
                    best_path = path
                    best_lit = result
        return best_path, best_lit

    def _cells_that_toggle(self, light_pos):
        """Return all walkable cells whose step toggles *light_pos*.

        A cell toggles a light if the cell IS the light, or in adjacent
        mode if the cell is a cardinal neighbour of the light.
        """
        grid = self.api.grid
        lx, ly = light_pos
        cells = set()
        if grid.is_walkable(light_pos):
            cells.add(light_pos)
        if self._adjacent_toggle:
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                c = (lx + dx, ly + dy)
                if grid.in_bounds(c) and grid.is_walkable(c):
                    cells.add(c)
        return cells

    def _find_toggle_path(self, start, lit):
        """Try to find a single-leg path that reduces *lit*.

        Unlike ``_pick_best_leg`` (which only targets GF(2) solution
        cells), this considers EVERY walkable cell adjacent to a
        remaining lit light as a potential destination.
        """
        candidate_cells = set()
        for lp in lit:
            candidate_cells |= self._cells_that_toggle(lp)
        candidate_cells.discard(start)
        if not candidate_cells:
            return None, lit

        best_path = None
        best_remaining = len(lit)
        best_lit = lit

        for cell in sorted(candidate_cells):
            path = self.api.bfs_path_positions(start, cell)
            if not path:
                continue
            result = self._simulate_path_toggles(path, lit)
            if len(result) < best_remaining:
                best_remaining = len(result)
                best_path = path
                best_lit = result
        return best_path, best_lit

    def _bfs_light_states(self, start, lit, max_depth=4):
        """BFS over (position, lit-set) states to find a multi-leg route
        that reaches an empty lit set.

        Each "move" is a BFS path to a cell that toggles at least one
        remaining light.  Returns a list of paths (legs) or ``None``.
        """
        initial = (start, frozenset(lit))
        queue = deque()
        queue.append((initial, []))
        visited = {initial}

        while queue:
            (pos, flit), legs = queue.popleft()
            if len(legs) >= max_depth:
                continue
            cur_lit = set(flit)
            # Candidate cells: any cell that toggles a remaining light
            candidate_cells = set()
            for lp in cur_lit:
                candidate_cells |= self._cells_that_toggle(lp)
            candidate_cells.discard(pos)

            for cell in sorted(candidate_cells):
                path = self.api.bfs_path_positions(pos, cell)
                if not path:
                    continue
                result = self._simulate_path_toggles(path, cur_lit)
                new_state = (cell, frozenset(result))
                if new_state in visited:
                    continue
                visited.add(new_state)
                new_legs = legs + [path]
                if not result:
                    return new_legs  # solved!
                queue.append((new_state, new_legs))

        # No solution found within depth limit.
        return None

    def _greedy_full_route(self, start, lit):
        """Build a full action sequence visiting targets greedily.

        After each leg, re-solve GF(2) on the *simulated* lit state.
        All actions are queued at once to avoid single-step oscillation.
        A visited-states set prevents infinite cycling.
        When the greedy search gets stuck, falls back to a BFS over
        light-state space to find a multi-step solution.
        """
        actions = []
        current = start
        sim_lit = set(lit)
        visited_states = set()
        max_legs = 3 * len(self._all_light_positions) + 10

        for _ in range(max_legs):
            if not sim_lit:
                break

            state_key = (current, frozenset(sim_lit))
            if state_key in visited_states:
                break  # cycle detected
            visited_states.add(state_key)

            # First try GF(2) solution targets.
            targets = self._solve_gf2(lit_override=sim_lit)
            path = None
            if targets:
                targets.discard(current)
            if targets:
                path, new_lit = self._pick_best_leg(
                    current,
                    targets,
                    sim_lit,
                )

            # If GF(2) targets didn't help, try any cell that toggles a
            # remaining light.
            if not path:
                path, new_lit = self._find_toggle_path(current, sim_lit)

            if not path:
                break

            sim_lit = new_lit
            path_actions = self.api.positions_to_actions(path)
            if path_actions:
                actions.extend(path_actions)
            current = path[-1]

        # If lights remain, try a BFS over light-state space to find a
        # multi-leg solution that the greedy approach missed.
        if sim_lit:
            bfs_legs = self._bfs_light_states(current, sim_lit)
            if bfs_legs:
                for leg in bfs_legs:
                    leg_actions = self.api.positions_to_actions(leg)
                    if leg_actions:
                        actions.extend(leg_actions)

        return actions

    def plan(self):  # noqa: C901
        lit = self._get_lit_positions()
        if not lit:
            self.action_queue = [0]
            return

        ax, ay = self.api.agent_position

        if not self._adjacent_toggle:
            # Simple mode: visit nearest lit switch
            nearest = min(lit, key=lambda p: abs(p[0] - ax) + abs(p[1] - ay))
            if abs(nearest[0] - ax) + abs(nearest[1] - ay) == 0:
                others = [p for p in lit if p != (ax, ay)]
                if others:
                    nearest = min(
                        others,
                        key=lambda p: abs(p[0] - ax) + abs(p[1] - ay),
                    )
                else:
                    self.action_queue = [0]
                    return

            unlit_lights = self._all_light_positions - lit
            path = self.api.bfs_path_positions(
                (ax, ay),
                nearest,
                avoid=unlit_lights - {nearest},
            )
            if path:
                actions = self.api.positions_to_actions(path)
                if actions:
                    self.action_queue = [actions[0]]
                    return
            path = self.api.bfs_path_positions((ax, ay), nearest)
            if path:
                actions = self.api.positions_to_actions(path)
                if actions:
                    self.action_queue = [actions[0]]
                    return
            self.action_queue = self.api.move_toward(*nearest)
        else:
            # ----- Adjacent toggle mode -----
            full_actions = self._greedy_full_route((ax, ay), lit)
            if full_actions:
                self.action_queue = full_actions
                return

            # Fallback: move toward nearest lit cell
            nearest = min(
                lit,
                key=lambda p: abs(p[0] - ax) + abs(p[1] - ay),
            )
            self.action_queue = self.api.move_toward(*nearest)


@register_oracle("PackingPuzzle-v0")
class PackingPuzzleOracle(OracleAgent):
    """Push typed pieces into matching target slots.

    Two-phase strategy per piece:
      1. Align the piece horizontally with its target column.
      2. Push the piece straight down into its target.

    When a piece is blocked, skip to another piece rather than
    making a random sideways push.  Pieces that need horizontal
    alignment are processed before vertically-aligned ones so
    blocking pieces are cleared out of the way first.
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
        # _can_push: single push is valid and avoids wrong-type targets
        # ------------------------------------------------------------------
        def _can_push(px, py, ptype, ddx, ddy):
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
                if int(grid.metadata[ly, lx]) != int(ptype):
                    return False
            elif land_obj not in (0, int(ObjectType.GOAL)):
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
            avail = [(tx, ty) for tx, ty, tt in targets if tt == ptype and (tx, ty) not in claimed]
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
        # Sort: horizontal-misaligned first, then by distance
        # ------------------------------------------------------------------
        def _sort_key(a):
            px, py, ptype, tx, ty = a
            horiz = 0 if px == tx else 1
            return (-horiz, abs(px - tx) + abs(py - ty))

        assignments.sort(key=_sort_key)

        # ------------------------------------------------------------------
        # Main loop: push or clear blockers
        # ------------------------------------------------------------------
        for px, py, ptype, tx, ty in assignments:
            if px != tx:
                ddx = 1 if tx > px else -1
                if _can_push(px, py, ptype, ddx, 0):
                    if _try_execute(px, py, ddx, 0):
                        return
                else:
                    bpos = _blocker_at(px, py, ddx, 0)
                    if bpos and _try_clear_blocker(*bpos):
                        return
                continue

            if py != ty:
                ddy = 1 if ty > py else -1
                if _can_push(px, py, ptype, 0, ddy):
                    if _try_execute(px, py, 0, ddy):
                        return
                else:
                    bpos = _blocker_at(px, py, 0, ddy)
                    if bpos and _try_clear_blocker(*bpos):
                        return
                continue

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
                    tn = int(grid.metadata[gy, gx])
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
