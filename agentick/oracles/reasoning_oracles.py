"""Oracle bots for reasoning tasks."""

from __future__ import annotations

from collections import deque

from agentick.core.types import ActionType, CellType, Direction, ObjectType
from agentick.oracles.base import OracleAgent
from agentick.oracles.helpers import interact_adjacent
from agentick.oracles.registry import register_oracle

# ---------------------------------------------------------------------------
# Shared helper: adjacent-interact pattern for solid SWITCH / LEVER objects
# ---------------------------------------------------------------------------

# Map from delta (dx, dy) to the move action that faces that direction.
# Moving toward a solid object is blocked but still updates orientation.
_DELTA_TO_MOVE = {
    (0, -1): int(ActionType.MOVE_UP),     # face NORTH
    (0, 1): int(ActionType.MOVE_DOWN),    # face SOUTH
    (1, 0): int(ActionType.MOVE_RIGHT),   # face EAST
    (-1, 0): int(ActionType.MOVE_LEFT),   # face WEST
}

_DELTA_TO_DIR = {
    (0, -1): Direction.NORTH,
    (0, 1): Direction.SOUTH,
    (1, 0): Direction.EAST,
    (-1, 0): Direction.WEST,
}

_INTERACT = int(ActionType.INTERACT)  # 5


def _blocking_positions(grid):
    """Return the set of all positions with blocking objects (SWITCH, LEVER, DOOR)."""
    blocked = set()
    h, w = grid.terrain.shape
    for y in range(h):
        for x in range(w):
            if grid.is_object_blocking((x, y)):
                blocked.add((x, y))
    return blocked


def _interact_with_target(api, target, extra_passable=None, avoid=None):
    """Return a list of actions to interact with a solid object at *target*.

    The object (SWITCH, LEVER, etc.) is non-walkable so the agent must:
    1. Navigate to a walkable cell adjacent to *target*.
    2. Face *target* (move toward it; the move is blocked but sets orientation).
    3. Use INTERACT (action 5), which targets the faced cell.

    Returns a list with **one** action (the next step), or an empty list if
    no adjacent cell is reachable. The caller should call this each turn
    until INTERACT is emitted.

    Blocking objects (SWITCH, LEVER, closed DOOR) are automatically added to
    the BFS avoid set so the agent doesn't try to path through them.

    Special case: if the agent is standing ON the target (e.g. placed there
    during generation), the function moves the agent to an adjacent walkable
    cell first, so it can then face the target and INTERACT.
    """
    ax, ay = api.agent_position
    tx, ty = target

    dx, dy = tx - ax, ty - ay

    # Case 0: agent is ON the target — move to an adjacent walkable cell first
    if dx == 0 and dy == 0:
        grid = api.grid
        for ddx, ddy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            nx, ny = ax + ddx, ay + ddy
            if (
                grid.in_bounds((nx, ny))
                and grid.is_walkable((nx, ny))
                and not grid.is_object_blocking((nx, ny))
            ):
                return [_DELTA_TO_MOVE[(ddx, ddy)]]
        return []

    # Case 1: already adjacent and facing → INTERACT
    if (dx, dy) in _DELTA_TO_DIR:
        agent_orientation = api._env.agent.orientation
        if agent_orientation == _DELTA_TO_DIR[(dx, dy)]:
            return [_INTERACT]
        # Case 2: adjacent but not facing → move toward target to set orientation
        return [_DELTA_TO_MOVE[(dx, dy)]]

    # Case 3: not adjacent → BFS to the best adjacent walkable cell
    grid = api.grid

    # Build avoid set: merge caller's avoid with all blocking object positions.
    # This prevents the BFS from pathing through solid objects like SWITCHes.
    blocking = _blocking_positions(grid)
    full_avoid = set(blocking)
    if avoid:
        full_avoid |= avoid

    # If the agent is currently on a blocking cell (placed during generation),
    # remove it from the avoid set AND add it to extra_passable so BFS can
    # start from there (avoid is checked before extra_passable in BFS _ok).
    agent_pos = (ax, ay)
    ep = set(extra_passable) if extra_passable else set()
    if grid.is_object_blocking(agent_pos):
        full_avoid.discard(agent_pos)
        blocking.discard(agent_pos)
        ep.add(agent_pos)
    passable = ep or None

    adj_cells = []
    for ddx, ddy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
        nx, ny = tx + ddx, ty + ddy
        if grid.in_bounds((nx, ny)) and grid.is_walkable((nx, ny)):
            if not grid.is_object_blocking((nx, ny)):
                adj_cells.append((nx, ny))

    if not adj_cells:
        return []

    # Pick the adjacent cell with shortest BFS path
    best_path = None
    for cell in adj_cells:
        if cell in full_avoid:
            continue
        path = api.bfs_path_positions(
            (ax, ay), cell, extra_passable=passable, avoid=full_avoid,
        )
        if path and (best_path is None or len(path) < len(best_path)):
            best_path = path

    # Retry without caller's avoid if nothing found (but keep blocking avoid)
    if best_path is None:
        for cell in adj_cells:
            if cell in blocking:
                continue
            path = api.bfs_path_positions(
                (ax, ay), cell, extra_passable=passable, avoid=blocking,
            )
            if path and (best_path is None or len(path) < len(best_path)):
                best_path = path

    if best_path is None:
        return []

    if len(best_path) <= 1:
        # Already at an adjacent cell — face the target
        fdx, fdy = tx - best_path[0][0], ty - best_path[0][1]
        if (fdx, fdy) in _DELTA_TO_MOVE:
            agent_orientation = api._env.agent.orientation
            if agent_orientation == _DELTA_TO_DIR[(fdx, fdy)]:
                return [_INTERACT]
            return [_DELTA_TO_MOVE[(fdx, fdy)]]
        return []

    # Return just the first step of the path
    actions = api.positions_to_actions(best_path)
    if actions:
        return [actions[0]]
    return []


@register_oracle("SwitchCircuit-v0")
class SwitchCircuitOracle(OracleAgent):
    """Plan switch toggle sequence in room-based layout to reach the goal.

    The task uses a chain of rooms (R0..R(N-1)) connected by single-cell
    doors, with GoalRoom hanging off Hub (R0). Dual switches open the next
    door but close the previous, forcing toggle cycles (ON->OFF) to unwind
    back to Hub.

    Strategy:
    1. BFS over (position_index, switch_bitmask) state space to find the
       shortest toggle sequence that makes the goal reachable.
    2. Follow the toggle plan: navigate to each switch, INTERACT to toggle.
    3. Handles dual switches by planning toggle-ON/OFF sequences with
       forced backtracking through the room chain.

    One step is queued at a time so the plan is re-evaluated each turn.
    """

    def _get_open_barrier_cells(self, config):
        """Return the set of barrier cells that are currently EMPTY (open)."""
        barriers = config.get("barriers", [])
        open_cells: set[tuple[int, int]] = set()
        for barrier in barriers:
            if barrier.get("open", False):
                for cell in barrier.get("cells", []):
                    open_cells.add((cell[0], cell[1]))
        return open_cells

    def _navigate_to(self, target, extra, avoid):
        """Try BFS navigation to target with decreasing strictness."""
        ax, ay = self.api.agent_position

        # Primary: avoid active switches, use open barriers as extra passable
        path = self.api.bfs_path_positions(
            (ax, ay), target, extra_passable=extra, avoid=avoid,
        )
        if path:
            actions = self.api.positions_to_actions(path)
            if actions:
                self.action_queue = [actions[0]]
                return True

        # Retry without avoid
        path = self.api.bfs_path_positions(
            (ax, ay), target, extra_passable=extra,
        )
        if path:
            actions = self.api.positions_to_actions(path)
            if actions:
                self.action_queue = [actions[0]]
                return True

        # Plain BFS
        path = self.api.bfs_path_positions((ax, ay), target)
        if path:
            actions = self.api.positions_to_actions(path)
            if actions:
                self.action_queue = [actions[0]]
                return True

        return False

    def _is_reachable(self, target, extra, avoid=None):
        """Check if target is reachable using current open barriers."""
        ax, ay = self.api.agent_position
        path = self.api.bfs_path_positions(
            (ax, ay), target, extra_passable=extra, avoid=avoid,
        )
        return path is not None

    def _compute_barrier_states(self, switch_states, switch_effects, n_barriers):
        """Compute which barriers are open given switch states.

        A barrier is open if at least one opener is ON and no closer is ON.
        """
        barrier_open = [False] * n_barriers
        barrier_has_closer = [False] * n_barriers

        for sw_idx, is_on in enumerate(switch_states):
            if not is_on or sw_idx >= len(switch_effects):
                continue
            effects = switch_effects[sw_idx]
            for b_idx in effects.get("opens", []):
                if b_idx < n_barriers:
                    barrier_open[b_idx] = True
            for b_idx in effects.get("closes", []):
                if b_idx < n_barriers:
                    barrier_has_closer[b_idx] = True

        return [
            barrier_open[i] and not barrier_has_closer[i]
            for i in range(n_barriers)
        ]

    def _find_toggle_plan(self, config):
        """BFS over (position_index, switch_bitmask) to find optimal toggle plan.

        Returns a list of (switch_index, target_state_bool) tuples representing
        the sequence of toggles needed, or None if unsolvable. Each entry means
        "navigate to switch_index and INTERACT to set it to target_state_bool".
        """
        from collections import deque

        switches = config.get("switch_positions", [])
        barriers = config.get("barriers", [])
        switch_effects = config.get("switch_effects", [])
        goal_positions = config.get("goal_positions", [])
        n = len(switches)

        if n == 0 or not goal_positions:
            return []

        goal_pos = tuple(goal_positions[0])
        n_barriers = len(barriers)
        grid = self.api.grid
        size = grid.height

        barrier_cell_sets = []
        for b in barriers:
            barrier_cell_sets.append([(c[0], c[1]) for c in b.get("cells", [])])

        # Current state
        switch_states = list(config.get("switch_states", [False] * n))
        current_mask = 0
        for i, s in enumerate(switch_states):
            if s:
                current_mask |= (1 << i)

        agent_pos = self.api.agent_position

        # Key positions: 0=agent, 1..n=switches, n+1=goal
        key_positions = [agent_pos] + [tuple(s) for s in switches] + [goal_pos]

        def _terrain_for_mask(mask):
            states = [(mask >> i) & 1 for i in range(n)]
            b_open = self._compute_barrier_states(states, switch_effects, n_barriers)
            t = grid.terrain.copy()
            for b_idx, cells in enumerate(barrier_cell_sets):
                is_open = b_open[b_idx] if b_idx < len(b_open) else False
                for cx, cy in cells:
                    t[cy, cx] = (
                        int(CellType.EMPTY) if is_open else int(CellType.WALL)
                    )
            return t

        def _bfs_reach(terrain, start):
            visited = {start}
            q = deque([start])
            while q:
                cx, cy = q.popleft()
                for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < size and 0 <= ny < size and (nx, ny) not in visited:
                        t = int(terrain[ny, nx])
                        if t not in (int(CellType.WALL), int(CellType.HAZARD)):
                            visited.add((nx, ny))
                            q.append((nx, ny))
            return visited

        # BFS: state = (pos_idx, mask), parent tracks for path reconstruction
        start_state = (0, current_mask)
        visited = {start_state}
        # parent[state] = (prev_state, switch_index_toggled)
        parent = {start_state: None}
        q = deque([start_state])

        found = None
        while q:
            pos_idx, mask = q.popleft()
            pos = key_positions[pos_idx]
            terrain = _terrain_for_mask(mask)
            reachable = _bfs_reach(terrain, pos)

            # Check if goal reachable from current state
            if goal_pos in reachable:
                found = (pos_idx, mask)
                break

            # Try toggling each reachable switch
            for sw_i in range(n):
                sw_pos = key_positions[sw_i + 1]
                if sw_pos not in reachable:
                    continue
                new_mask = mask ^ (1 << sw_i)
                new_state = (sw_i + 1, new_mask)
                if new_state not in visited:
                    visited.add(new_state)
                    parent[new_state] = ((pos_idx, mask), sw_i)
                    q.append(new_state)

        if found is None:
            return None

        # Reconstruct toggle sequence
        toggles = []
        state = found
        while parent[state] is not None:
            prev_state, sw_i = parent[state]
            # The target state of the switch after toggle
            _, mask_after = state
            target_on = bool((mask_after >> sw_i) & 1)
            toggles.append((sw_i, target_on))
            state = prev_state
        toggles.reverse()
        return toggles

    def plan(self):
        config = self.api.task_config
        switches = config.get("switch_positions", [])
        switch_states = config.get("switch_states", [])
        ax, ay = self.api.agent_position

        n = len(switches)
        if n == 0:
            return

        extra = self._get_open_barrier_cells(config)

        # PRIORITY 1: If the goal is reachable RIGHT NOW, go there.
        goal = self.api.get_nearest("goal")
        if goal:
            goal_pos = goal.position
            if self._is_reachable(goal_pos, extra):
                if self._navigate_to(goal_pos, extra, set()):
                    return

        # PRIORITY 2: Compute BFS toggle plan and follow it
        toggle_plan = self._find_toggle_plan(config)

        if toggle_plan:
            # Find next toggle action in the plan
            for sw_idx, target_on in toggle_plan:
                # Skip toggles that are already in the target state
                if switch_states[sw_idx] == target_on:
                    continue
                sw_pos = (switches[sw_idx][0], switches[sw_idx][1])
                # Switches are solid — use adjacent-interact pattern
                avoid = set()
                for i in range(n):
                    if i != sw_idx:
                        pos = (switches[i][0], switches[i][1])
                        if pos != (ax, ay):
                            avoid.add(pos)
                actions = _interact_with_target(
                    self.api, sw_pos, extra_passable=extra, avoid=avoid,
                )
                if actions:
                    self.action_queue = actions
                    return
                # Fallback without avoidance
                actions = _interact_with_target(
                    self.api, sw_pos, extra_passable=extra,
                )
                if actions:
                    self.action_queue = actions
                    return
                fallback = self.api.move_toward(*sw_pos)
                if fallback:
                    self.action_queue = fallback
                return

        # PRIORITY 3: No toggle plan needed — go to goal
        if goal:
            if self._navigate_to(goal.position, extra, set()):
                return
            fallback = self.api.move_toward(*goal.position)
            if fallback:
                self.action_queue = fallback


@register_oracle("SymbolMatching-v0")
class SymbolMatchingOracle(OracleAgent):
    """Pick up symbol items and deliver to matching targets (same ObjectType).

    Items are on the left side, targets on the right side. Both use the same
    ObjectType per pair (e.g., GEM item -> GEM target). The oracle uses
    pair_info from config to know which positions are items vs targets.
    Fake items use types with no matching target.
    """

    _SYMBOL_TYPES = [
        ObjectType.GEM,
        ObjectType.POTION,
        ObjectType.SCROLL,
        ObjectType.COIN,
        ObjectType.ORB,
        ObjectType.LEVER,
    ]
    _SYMBOL_SET = {int(t) for t in _SYMBOL_TYPES}

    def _get_task_target_pos_set(self):
        """Get the set of target positions from the task object."""
        task = getattr(self.api._env, "task", None)
        if task is not None:
            return getattr(task, "_target_pos_set", set())
        # Fallback: build from config
        config = self.api.task_config
        target_pos_set = set()
        for p in config.get("pair_info", []):
            target_pos_set.add(tuple(p["target_pos"]))
        return target_pos_set

    def _get_task_carrying(self):
        """Read carrying state directly from the task object."""
        task = getattr(self.api._env, "task", None)
        if task is None:
            return None
        carrying = getattr(task, "_carrying", None)
        return carrying

    def plan(self):
        grid = self.api.grid
        config = self.api.task_config
        pair_info = config.get("pair_info", [])
        ax, ay = self.api.agent_position

        # Read carrying state from the task directly (most reliable)
        carrying = self._get_task_carrying()

        # Build target position set (unmatched targets still have symbol ObjectType)
        target_pos_set = self._get_task_target_pos_set()

        # Build set of all symbol positions on the grid (items + unmatched targets)
        all_symbol_positions = set()
        for y in range(grid.height):
            for x in range(grid.width):
                if int(grid.objects[y, x]) in self._SYMBOL_SET:
                    all_symbol_positions.add((x, y))

        # Separate into item positions and target positions
        all_targets = all_symbol_positions & target_pos_set
        all_items = all_symbol_positions - target_pos_set

        # Build set of legitimate pair item positions
        pair_item_positions = set()
        for p in pair_info:
            ix, iy = p["item_pos"]
            if int(grid.objects[iy, ix]) == p["symbol_type"]:
                pair_item_positions.add((ix, iy))

        # Fake items = items not in pair_item_positions
        fake_positions = all_items - pair_item_positions

        if carrying is None:
            # Not carrying — find an uncollected item to pick up
            best_pair = None
            best_dist = 999
            for p in pair_info:
                ix, iy = p["item_pos"]
                sym_type = p["symbol_type"]
                tx, ty = p["target_pos"]
                # Item still present and target still unmatched
                if (
                    int(grid.objects[iy, ix]) == sym_type
                    and (tx, ty) in target_pos_set
                    and int(grid.objects[ty, tx]) == sym_type
                ):
                    d = abs(ix - ax) + abs(iy - ay)
                    if d < best_dist:
                        best_dist = d
                        best_pair = p
            if best_pair:
                ix, iy = best_pair["item_pos"]
                # Avoid: all targets + fake items (stepping on them picks them up)
                avoid = (all_targets | fake_positions) - {(ix, iy)}
                path = self.api.bfs_path_positions(
                    (ax, ay),
                    (ix, iy),
                    avoid=avoid,
                )
                if path:
                    actions = self.api.positions_to_actions(path)
                    if actions:
                        self.action_queue = [actions[0]]
                        return
                # Try avoiding only fakes (not targets)
                avoid2 = fake_positions - {(ix, iy)}
                path = self.api.bfs_path_positions(
                    (ax, ay),
                    (ix, iy),
                    avoid=avoid2,
                )
                if path:
                    actions = self.api.positions_to_actions(path)
                    if actions:
                        self.action_queue = [actions[0]]
                        return
                # Fallback without avoidance
                path = self.api.path_to(ix, iy)
                if path:
                    self.action_queue = [path[0]]
                    return
                self.action_queue = self.api.move_toward(ix, iy)
                return
            self.action_queue = [0]
        else:
            # Carrying — deliver to the matching target (same ObjectType)
            correct_target = None
            for p in pair_info:
                if p["symbol_type"] == carrying:
                    tx, ty = p["target_pos"]
                    if (tx, ty) in target_pos_set and int(grid.objects[ty, tx]) == carrying:
                        correct_target = (tx, ty)
                        break

            if correct_target:
                # Avoid: all OTHER targets + all remaining items (don't pick up more)
                wrong_targets = all_targets - {correct_target}
                avoid = wrong_targets | all_items
                path = self.api.bfs_path_positions(
                    (ax, ay),
                    correct_target,
                    avoid=avoid,
                )
                if path:
                    actions = self.api.positions_to_actions(path)
                    if actions:
                        self.action_queue = [actions[0]]
                        return
                # Try avoiding only wrong targets
                path = self.api.bfs_path_positions(
                    (ax, ay),
                    correct_target,
                    avoid=wrong_targets,
                )
                if path:
                    actions = self.api.positions_to_actions(path)
                    if actions:
                        self.action_queue = [actions[0]]
                        return
                # Fallback without avoidance
                path = self.api.path_to(*correct_target)
                if path:
                    self.action_queue = [path[0]]
                    return
                self.action_queue = self.api.move_toward(*correct_target)
                return

            # Carrying a fake or unknown item — dump on any remaining target
            if all_targets:
                nearest_target = min(
                    all_targets, key=lambda t: abs(t[0] - ax) + abs(t[1] - ay)
                )
                path = self.api.bfs_path_positions(
                    (ax, ay),
                    nearest_target,
                )
                if path:
                    actions = self.api.positions_to_actions(path)
                    if actions:
                        self.action_queue = [actions[0]]
                        return
            self.action_queue = [0]


@register_oracle("RuleInduction-v0")
class RuleInductionOracle(OracleAgent):
    """Combination-discovery oracle for RuleInduction-v0.

    Strategy (with full knowledge of the rule table):
    1. If _target_crafted: walk onto the target object to collect it.
    2. If not carrying: find a rule that directly produces target_type, pick up
       input A while avoiding all other object positions so no accidental pickup
       occurs en route.
    3. If carrying A: navigate to input B of the chosen rule (avoid other
       objects to prevent accidental pickups); stepping on B triggers the combo.
    4. If no valid ground object exists for the carried type, idle until trial
       budget resets the grid.
    """

    def __init__(self, env):
        super().__init__(env)
        # Persistent planning state across plan() calls within a trial
        self._chosen_rule: tuple[int, int, int] | None = None  # (a, b, c)

    def _scan_objects(self, grid) -> list[tuple[int, int, int]]:
        """Return list of (x, y, obj_type_int) for all non-NONE objects."""
        results = []
        for cy in range(grid.height):
            for cx in range(grid.width):
                gt = int(grid.objects[cy, cx])
                if gt != 0:
                    results.append((cx, cy, gt))
        return results

    def _find_rule_for_target(
        self,
        rule_table: dict[tuple[int, int], int],
        target_type: int,
        objects_on_grid: list[tuple[int, int, int]],
    ) -> tuple[int, int, int] | None:
        """Find a rule (a, b, c) where c==target_type and both a and b exist on grid."""
        grid_types = {gt for _, _, gt in objects_on_grid}
        for (a, b), c in rule_table.items():
            if c == target_type and a in grid_types and b in grid_types:
                return (a, b, c)
        # Fall back: any rule where result is target_type (even if B not on grid yet)
        for (a, b), c in rule_table.items():
            if c == target_type and a in grid_types:
                return (a, b, c)
        return None

    def plan(self):  # noqa: C901
        config = self.api.task_config
        grid = self.api.grid
        agent = self.api.agent
        ax, ay = self.api.agent_position

        # Already succeeded
        if config.get("_target_collected", False):
            self.action_queue = [0]
            return

        target_type = config.get("_target_type", -1)
        rule_table_list = config.get("_rule_table_list", [])
        rule_table = {(int(r[0]), int(r[1])): int(r[2]) for r in rule_table_list}
        objects_on_grid = self._scan_objects(grid)
        all_obj_positions = {(ox, oy) for ox, oy, _ in objects_on_grid}

        # ------------------------------------------------------------------ #
        # Phase 1: collect crafted target
        # ------------------------------------------------------------------ #
        if config.get("_target_crafted", False):
            for ox, oy, gt in objects_on_grid:
                if gt == target_type:
                    # Special case: target was placed on the agent's current cell
                    # (combination result lands where agent stands).  Step away
                    # first, then return.
                    if (ox, oy) == (ax, ay):
                        for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
                            nx, ny = ax + dx, ay + dy
                            if (
                                grid.in_bounds((nx, ny))
                                and int(grid.terrain[ny, nx]) != 1  # not WALL
                                and (nx, ny) not in all_obj_positions
                            ):
                                actions = self.api.positions_to_actions(
                                    [(ax, ay), (nx, ny)]
                                )
                                self.action_queue = actions if actions else [0]
                                return
                        self.action_queue = [0]
                        return
                    # Navigate to target; avoid other objects
                    avoid = all_obj_positions - {(ox, oy)}
                    path = self.api.bfs_path_positions((ax, ay), (ox, oy), avoid=avoid)
                    if not path:
                        path = self.api.bfs_path_positions((ax, ay), (ox, oy))
                    if path:
                        actions = self.api.positions_to_actions(path)
                        if actions:
                            self.action_queue = [actions[0]]
                            return
            self.action_queue = [0]
            return

        # ------------------------------------------------------------------ #
        # Phase 2: choose a rule to execute
        # ------------------------------------------------------------------ #
        # If inventory changed (trial reset cleared it), discard stale plan
        carrying_type: int | None = None
        if len(agent.inventory) > 0:
            carrying_type = agent.inventory[0].properties.get("object_type", None)

        if carrying_type is None:
            # Reset chosen rule so we re-evaluate
            self._chosen_rule = None

        if self._chosen_rule is None:
            self._chosen_rule = self._find_rule_for_target(
                rule_table, target_type, objects_on_grid,
            )

        if self._chosen_rule is None:
            # No rule found — idle until trial resets
            self.action_queue = [0]
            return

        chosen_a, chosen_b, chosen_c = self._chosen_rule

        # ------------------------------------------------------------------ #
        # Phase 3: pick up input A (if not already holding it)
        # ------------------------------------------------------------------ #
        if carrying_type is None:
            # Navigate to nearest object of type chosen_a, avoiding all others
            pickup_pos: tuple[int, int] | None = None
            for ox, oy, gt in objects_on_grid:
                if gt == chosen_a:
                    pickup_pos = (ox, oy)
                    break

            if pickup_pos is None:
                # chosen_a no longer on grid (used up); reset plan
                self._chosen_rule = None
                self.action_queue = [0]
                return

            # Avoid all other objects on the path to pickup_pos
            avoid = all_obj_positions - {pickup_pos}
            path = self.api.bfs_path_positions((ax, ay), pickup_pos, avoid=avoid)
            if not path:
                # Relax: allow passing through other objects
                path = self.api.bfs_path_positions((ax, ay), pickup_pos)
            if path:
                actions = self.api.positions_to_actions(path)
                if actions:
                    self.action_queue = [actions[0]]
                    return
            self.action_queue = [0]
            return

        # ------------------------------------------------------------------ #
        # Phase 4: carrying something — navigate to input B to combine
        # ------------------------------------------------------------------ #
        # Validate we're still carrying the right type
        if carrying_type != chosen_a:
            # Carrying something unexpected; find any valid ground for it
            for ox, oy, gt in objects_on_grid:
                if rule_table.get((carrying_type, gt)) is not None:
                    avoid = all_obj_positions - {(ox, oy)}
                    path = self.api.bfs_path_positions((ax, ay), (ox, oy), avoid=avoid)
                    if not path:
                        path = self.api.bfs_path_positions((ax, ay), (ox, oy))
                    if path:
                        actions = self.api.positions_to_actions(path)
                        if actions:
                            self.action_queue = [actions[0]]
                            return
            # Nothing to combine with — idle
            self.action_queue = [0]
            return

        # Find input B on the grid
        ground_pos: tuple[int, int] | None = None
        # Prefer the exact B type from chosen_rule
        for ox, oy, gt in objects_on_grid:
            if gt == chosen_b:
                ground_pos = (ox, oy)
                break

        if ground_pos is None:
            # B not available; try any valid ground for carrying_type
            for ox, oy, gt in objects_on_grid:
                if rule_table.get((carrying_type, gt)) is not None:
                    ground_pos = (ox, oy)
                    break

        if ground_pos is None:
            # No valid ground at all — idle until trial resets
            self.action_queue = [0]
            return

        # Navigate to ground_pos, avoiding all other objects along the path
        avoid = all_obj_positions - {ground_pos}
        path = self.api.bfs_path_positions((ax, ay), ground_pos, avoid=avoid)
        if not path:
            path = self.api.bfs_path_positions((ax, ay), ground_pos)
        if path:
            actions = self.api.positions_to_actions(path)
            if actions:
                self.action_queue = [actions[0]]
                return
        self.action_queue = [0]


@register_oracle("GraphColoring-v0")
class GraphColoringOracle(OracleAgent):
    """Color graph nodes via INTERACT so no adjacent nodes share a color.

    Uses backtracking graph coloring to find a valid coloring with exactly
    n_colors (the chromatic number). Greedy coloring can fail when the
    number of available colors equals the exact chromatic number, so
    backtracking is required for correctness.
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
        """Compute a valid coloring using backtracking (not greedy).

        Colors are 1-based (1..n_colors) because metadata 0 = uncolored.
        """
        config = self.api.task_config
        nodes = config.get("node_positions", [])
        adj = config.get("adjacency", {})
        n_colors = config.get("n_colors", 2)
        n = len(nodes)

        # Build adjacency as sets of ints for fast lookup
        adj_sets = {}
        for i in range(n):
            neighbors = adj.get(i, adj.get(str(i), []))
            adj_sets[i] = {int(nb) for nb in neighbors}

        # Backtracking solver (0-based colors internally)
        assignment = [-1] * n

        def _bt(node):
            if node == n:
                return True
            used = {assignment[nb] for nb in adj_sets.get(node, set())
                    if assignment[nb] >= 0}
            for c in range(n_colors):
                if c not in used:
                    assignment[node] = c
                    if _bt(node + 1):
                        return True
                    assignment[node] = -1
            return False

        _bt(0)

        # Convert to 1-based colors for the task (metadata: 0=uncolored, 1..n=colors)
        self._coloring = {i: assignment[i] + 1 for i in range(n)}
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

            # Nodes are solid SWITCH objects — use adjacent-interact pattern.
            # Check if we're already adjacent and facing the node.
            fdx, fdy = nx - ax, ny - ay
            if (fdx, fdy) in _DELTA_TO_DIR:
                agent_orientation = self.api._env.agent.orientation
                if agent_orientation == _DELTA_TO_DIR[(fdx, fdy)]:
                    # Adjacent and facing — queue all needed INTERACTs
                    cycles_needed = (target_color - current_color) % (n_colors + 1)
                    if cycles_needed == 0:
                        cycles_needed = n_colors + 1  # full cycle
                    self.action_queue = [_INTERACT] * cycles_needed
                    return
                # Adjacent but not facing — face the node
                self.action_queue = [_DELTA_TO_MOVE[(fdx, fdy)]]
                return

            # Not adjacent — navigate toward the node
            actions = _interact_with_target(self.api, (nx, ny))
            if actions:
                self.action_queue = actions
                return
            self.action_queue = self.api.move_toward(nx, ny)
            return

        # All nodes correctly colored
        self.action_queue = [0]


@register_oracle("LightsOut-v0")
class LightsOutOracle(OracleAgent):
    """Toggle switches to turn all lights off by walking.

    Classic Lights Out: stepping on any cell auto-toggles it (if switch) +
    adjacent switches. No INTERACT needed — just walk to the right cells.

    For easy (no adjacent toggle): walk onto each lit switch to toggle it.
    For medium+ (adjacent toggle): walk-simulation planning accounts for
    cascading toggles from walking through the switch grid.
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
        for p in config.get("light_positions", []):
            self._light_positions.add(tuple(p))
            self._all_light_positions.add(tuple(p))
        for p in config.get("decoy_positions", []):
            self._all_light_positions.add(tuple(p))
        for p in config.get("puzzle_cells", []):
            self._all_light_positions.add(tuple(p))
        super().reset(obs, info)

    def _get_lit_positions(self):
        """Return set of currently lit positions (metadata == 1)."""
        grid = self.api.grid
        lit = set()
        for pos in self._all_light_positions:
            px, py = pos
            if (
                grid.objects[py, px] == ObjectType.SWITCH
                and int(grid.metadata[py, px]) == 1
            ):
                lit.add(pos)
        return lit

    def _bfs(self, start, goal, avoid=None):
        """BFS wrapper that marks all light-grid positions as passable."""
        return self.api.bfs_path_positions(
            start, goal,
            extra_passable=self._all_light_positions,
            avoid=avoid,
        )

    def _simulate_step_at(self, pos, lit):
        """Simulate stepping on *pos*: toggle self + adjacent switches.

        Returns a new set — *lit* is not mutated.
        """
        lit = set(lit)
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

    def _simulate_walk(self, path, lit):
        """Simulate walking along *path*, toggling at each step.

        Returns the resulting lit set after walking the full path.
        """
        lit = set(lit)
        for pos in path:
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

    def _solve_gf2(self, lit_override=None):
        """Solve which cells to step on to clear all lights (GF(2))."""
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

    def _cells_that_toggle(self, light_pos):
        """Return all walkable cells where stepping would toggle *light_pos*."""
        grid = self.api.grid
        lx, ly = light_pos
        cells = set()
        if grid.is_walkable(light_pos) or light_pos in self._all_light_positions:
            cells.add(light_pos)
        if self._adjacent_toggle:
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                c = (lx + dx, ly + dy)
                if grid.in_bounds(c) and (
                    grid.is_walkable(c) or c in self._all_light_positions
                ):
                    cells.add(c)
        return cells

    def _greedy_full_route(self, start, lit):
        """Build a full action sequence, walking to targets greedily.

        Walking auto-toggles, so we must simulate toggles along the path.
        After each leg, re-solve GF(2) on the simulated lit state.
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
                break
            visited_states.add(state_key)

            targets = self._solve_gf2(lit_override=sim_lit)
            best_path = None
            best_remaining = float("inf")
            best_lit = sim_lit

            # Try each GF(2) target; simulate the FULL walk including
            # intermediate toggles
            if targets:
                targets.discard(current)
            candidates = targets if targets else set()

            # Also consider any cell that toggles a remaining light
            if not candidates:
                for lp in sim_lit:
                    candidates |= self._cells_that_toggle(lp)
                candidates.discard(current)

            for target in sorted(candidates):
                path = self._bfs(current, target)
                if not path:
                    continue
                # Skip first element if it equals current position
                walk = path if path[0] != current else path[1:]
                result = self._simulate_walk(walk, sim_lit)
                if len(result) < best_remaining:
                    best_remaining = len(result)
                    best_path = path
                    best_lit = result

            if not best_path:
                break

            sim_lit = best_lit
            path_actions = self.api.positions_to_actions(best_path)
            if path_actions:
                actions.extend(path_actions)
            current = best_path[-1]

        # If still lit, try BFS over light-state space
        if sim_lit:
            bfs_legs = self._bfs_light_states(current, sim_lit)
            if bfs_legs:
                for leg in bfs_legs:
                    leg_actions = self.api.positions_to_actions(leg)
                    if leg_actions:
                        actions.extend(leg_actions)

        return actions

    def _bfs_light_states(self, start, lit, max_depth=4):
        """BFS over (position, lit-set) states to find a multi-leg route."""
        initial = (start, frozenset(lit))
        queue = deque()
        queue.append((initial, []))
        visited = {initial}

        while queue:
            (pos, flit), legs = queue.popleft()
            if len(legs) >= max_depth:
                continue
            cur_lit = set(flit)
            candidate_cells = set()
            for lp in cur_lit:
                candidate_cells |= self._cells_that_toggle(lp)
            candidate_cells.discard(pos)

            for cell in sorted(candidate_cells):
                path = self._bfs(pos, cell)
                if not path:
                    continue
                walk = path if path[0] != pos else path[1:]
                result = self._simulate_walk(walk, cur_lit)
                new_state = (cell, frozenset(result))
                if new_state in visited:
                    continue
                visited.add(new_state)
                new_legs = legs + [path]
                if not result:
                    return new_legs
                queue.append((new_state, new_legs))

        return None

    def plan(self):  # noqa: C901
        lit = self._get_lit_positions()
        if not lit:
            self.action_queue = [0]
            return

        ax, ay = self.api.agent_position

        # Both modes use greedy full-route planning because walking through
        # switch cells auto-toggles them, requiring simulation of all
        # intermediate toggles.
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


@register_oracle("ProgramSynthesis-v0")
class ProgramSynthesisOracle(OracleAgent):
    """Push GEM objects to replicate reference SCROLL pattern (translation-invariant).

    Strategy — Sokoban-style push planning with gem-path BFS:
    1. Identify corner-deadlocked gems and compute feasible target positions
    2. Assign gems to targets using optimal matching (brute-force permutations)
    3. Compute push ordering: gems whose targets are "deeper" (adjacent to more
       other targets) are pushed first, so they don't get blocked later
    4. For each gem needing to move, use BFS in gem-position-space to find the
       sequence of pushes that routes the gem to its target (around obstacles)
    5. When blocked by a gem at its target, temporarily displace the blocker
    6. Stuck detection: if state cycles for 12+ steps, skip current gem
    """

    def __init__(self, env):
        super().__init__(env)
        self._cached_targets: list[tuple[int, int]] | None = None
        self._gem_push_plan: list[tuple[int, int]] | None = None
        self._gem_push_target: tuple[int, int] | None = None
        self._skip_gems: set[tuple[int, int]] = set()
        self._state_history: list[tuple] = []
        self._stuck_threshold = 12
        self._current_gem: tuple[int, int] | None = None
        self._target_attempts = 0
        self._clearance_plan: list[int] | None = None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _find_gems(self):
        grid = self.api.grid
        gems = []
        for y in range(grid.height):
            for x in range(grid.width):
                if grid.objects[y, x] == ObjectType.GEM:
                    gems.append((x, y))
        return gems

    def _is_blocked(self, x, y, grid):
        """Check if cell (x,y) is wall or out of bounds."""
        if not (0 <= x < grid.width and 0 <= y < grid.height):
            return True
        return int(grid.terrain[y, x]) == int(CellType.WALL)

    def _is_corner_deadlocked(self, gx, gy, grid):
        """Check if gem is stuck: walls on two perpendicular sides."""
        for dx1, dy1, dx2, dy2 in [
            (1, 0, 0, 1), (1, 0, 0, -1), (-1, 0, 0, 1), (-1, 0, 0, -1)
        ]:
            if self._is_blocked(gx + dx1, gy + dy1, grid) and self._is_blocked(
                gx + dx2, gy + dy2, grid
            ):
                return True
        return False

    def _static_obstacles(self, grid):
        """Return set of cells that are walls or contain SCROLL objects."""
        obs = set()
        for y in range(grid.height):
            for x in range(grid.width):
                if int(grid.terrain[y, x]) == int(CellType.WALL):
                    obs.add((x, y))
                elif grid.objects[y, x] == ObjectType.SCROLL:
                    obs.add((x, y))
        return obs

    def _bfs_gem_path(self, start, goal, grid, other_gems=None):
        """BFS in gem-position-space to find push sequence from start to goal.

        Each state is a gem position (x, y). Transitions are pushes in the
        4 cardinal directions. A push (dx, dy) requires:
          - push_pos (gx-dx, gy-dy) is not wall (agent stands there)
          - dest (gx+dx, gy+dy) is not wall, not scroll, not other gem
          - push_pos is not an other-gem position

        Returns list of push directions [(dx, dy), ...] or None if unreachable.
        """
        from collections import deque

        static_obs = self._static_obstacles(grid)
        other = set(other_gems) if other_gems else set()

        if start == goal:
            return []

        visited = {start}
        queue = deque([(start, [])])

        while queue:
            (gx, gy), pushes = queue.popleft()

            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                px, py = gx - dx, gy - dy  # push position (agent)
                dest_x, dest_y = gx + dx, gy + dy  # gem destination

                # Push position must be walkable and not another gem
                if self._is_blocked(px, py, grid):
                    continue
                if (px, py) in other:
                    continue

                # Destination must be walkable, not scroll, not another gem
                if (dest_x, dest_y) in static_obs:
                    continue
                if (dest_x, dest_y) in other:
                    continue
                if self._is_blocked(dest_x, dest_y, grid):
                    continue

                if (dest_x, dest_y) in visited:
                    continue

                new_pushes = pushes + [(dx, dy)]
                if (dest_x, dest_y) == goal:
                    return new_pushes

                visited.add((dest_x, dest_y))
                queue.append(((dest_x, dest_y), new_pushes))

        return None

    def _gem_reachable_set(self, start, grid):
        """BFS flood-fill in gem-push-space from start position.

        Returns the set of all positions reachable by the gem via pushes,
        considering only static obstacles (walls, scrolls). Other gems are
        NOT considered obstacles (they can be moved).
        """
        from collections import deque

        static_obs = self._static_obstacles(grid)
        visited = {start}
        queue = deque([start])

        while queue:
            gx, gy = queue.popleft()
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                px, py = gx - dx, gy - dy
                dest_x, dest_y = gx + dx, gy + dy
                if self._is_blocked(px, py, grid):
                    continue
                if (dest_x, dest_y) in static_obs:
                    continue
                if self._is_blocked(dest_x, dest_y, grid):
                    continue
                if (dest_x, dest_y) in visited:
                    continue
                visited.add((dest_x, dest_y))
                queue.append((dest_x, dest_y))

        return visited

    def _compute_targets(self, immovable_gems=None):
        """Find the best anchor so targets minimise total assignment cost.

        Pre-computes reachability set for each gem (positions reachable via
        pushes around static obstacles). Then checks assignments using fast
        set-membership tests instead of per-pair BFS.
        """
        from itertools import permutations

        cfg = self.api.task_config
        ref_offsets = cfg.get("reference_offsets", [])
        grid = self.api.grid
        if not ref_offsets:
            return []

        scroll_set = {tuple(s) for s in cfg.get("scroll_positions", [])}
        gems = self._find_gems()
        n = len(ref_offsets)
        if len(gems) != n:
            return []

        immovable = set(immovable_gems) if immovable_gems else set()

        # Pre-compute reachability for each gem
        gem_reachable = []
        gem_is_deadlocked = []
        for g in gems:
            if self._is_corner_deadlocked(g[0], g[1], grid):
                gem_reachable.append({g})  # can only stay in place
                gem_is_deadlocked.append(True)
            else:
                gem_reachable.append(self._gem_reachable_set(g, grid))
                gem_is_deadlocked.append(False)

        best_targets = None
        best_cost = float("inf")

        for ox in range(1, grid.width - 1):
            for oy in range(1, grid.height - 1):
                targets = [(ox + dx, oy + dy) for dx, dy in ref_offsets]
                target_set = set(targets)

                # All target cells must be interior, non-wall, non-scroll
                ok = True
                for tx, ty in targets:
                    if not (1 <= tx < grid.width - 1 and 1 <= ty < grid.height - 1):
                        ok = False
                        break
                    if int(grid.terrain[ty, tx]) == int(CellType.WALL):
                        ok = False
                        break
                    if (tx, ty) in scroll_set:
                        ok = False
                        break
                if not ok:
                    continue

                # Immovable gems must each sit on a target
                if immovable and not immovable.issubset(target_set):
                    continue

                # Find optimal feasible assignment via brute-force permutations.
                # Feasibility: each gem must be able to reach its assigned target
                # (checked via pre-computed reachability sets — O(1) per check).
                for perm in permutations(range(n)):
                    cost = sum(
                        abs(gems[i][0] - targets[perm[i]][0])
                        + abs(gems[i][1] - targets[perm[i]][1])
                        for i in range(n)
                    )
                    if cost >= best_cost:
                        continue

                    feasible = True
                    for i in range(n):
                        t = targets[perm[i]]
                        if gems[i] == t:
                            continue
                        # Deadlocked gems that need to move: infeasible
                        if gem_is_deadlocked[i]:
                            feasible = False
                            break
                        # Check reachability via pre-computed set
                        if t not in gem_reachable[i]:
                            feasible = False
                            break

                    if feasible:
                        best_cost = cost
                        best_targets = targets

        return best_targets or []

    def _compute_assignment(self, gems, targets):
        """Return dict mapping gem_pos -> target_pos (optimal assignment).

        When multiple permutations have the same cost, prefer the one where
        more gems can reach their targets (using pre-computed reachability
        sets that ignore other gems but respect walls and scrolls).
        """
        from itertools import permutations

        n = len(gems)
        if n == 0:
            return {}

        grid = self.api.grid

        # Pre-compute static reachability (ignoring other gems)
        reach_sets = []
        for g in gems:
            if self._is_corner_deadlocked(g[0], g[1], grid):
                reach_sets.append({g})
            else:
                reach_sets.append(self._gem_reachable_set(g, grid))

        # Also pre-compute BFS reachability WITH other gems as obstacles
        gem_set = set(gems)
        reach_with_others = []
        for i, g in enumerate(gems):
            others = gem_set - {g}
            reachable = set()
            for j, t in enumerate(targets):
                if g == t:
                    reachable.add(t)
                elif self._bfs_gem_path(g, t, grid, others) is not None:
                    reachable.add(t)
            reach_with_others.append(reachable)

        best_perm = None
        best_cost = float("inf")
        best_bfs_reach = -1

        for perm in permutations(range(n)):
            cost = sum(
                abs(gems[i][0] - targets[perm[i]][0])
                + abs(gems[i][1] - targets[perm[i]][1])
                for i in range(n)
            )
            if cost > best_cost:
                continue

            if cost < best_cost:
                # Strictly better cost — use without checking reachability
                bfs_reach = sum(
                    1 for i in range(n)
                    if targets[perm[i]] in reach_with_others[i]
                    or gems[i] == targets[perm[i]]
                )
                best_cost = cost
                best_perm = perm
                best_bfs_reach = bfs_reach
            else:
                # Same cost — prefer better BFS reachability
                bfs_reach = sum(
                    1 for i in range(n)
                    if targets[perm[i]] in reach_with_others[i]
                    or gems[i] == targets[perm[i]]
                )
                if bfs_reach > best_bfs_reach:
                    best_perm = perm
                    best_bfs_reach = bfs_reach

        if best_perm is None:
            return {}
        return {gems[i]: targets[best_perm[i]] for i in range(n)}

    def _check_stuck(self, gems):
        """Track (agent_pos, gem_positions) state. Return True if cycling.

        Detects both short cycles (agent oscillating in place) and longer
        cycles (gem positions oscillating between 2 states over many steps).
        """
        pos = self.api.agent_position
        gem_state = tuple(sorted(gems))
        state = (pos, gem_state)
        self._state_history.append(state)

        # Keep longer history for detecting oscillations
        max_hist = self._stuck_threshold * 4
        if len(self._state_history) > max_hist:
            self._state_history = self._state_history[-max_hist:]

        # Check for short cycles (same full state repeating within threshold)
        if len(self._state_history) >= self._stuck_threshold:
            recent = self._state_history[-self._stuck_threshold:]
            if len(set(recent)) <= 3:
                return True

        # Check for gem-position oscillations: if gem positions only take
        # 2 values over the last 30 steps, we're oscillating
        if len(self._state_history) >= 30:
            recent_gems = [s[1] for s in self._state_history[-30:]]
            unique_gem_states = set(recent_gems)
            if len(unique_gem_states) <= 2:
                return True

        return False

    def _can_push_now(self, gx, gy, dx, dy, grid, gem_set):
        """Check if gem at (gx,gy) can be pushed in direction (dx,dy) right now.

        Considers walls, scrolls, and other gems as obstacles.
        """
        px, py = gx - dx, gy - dy
        dest_x, dest_y = gx + dx, gy + dy
        if self._is_blocked(px, py, grid):
            return False
        if (px, py) in gem_set:
            return False
        if self._is_blocked(dest_x, dest_y, grid):
            return False
        if (dest_x, dest_y) in gem_set:
            return False
        if grid.objects[dest_y, dest_x] == ObjectType.SCROLL:
            return False
        return True

    def _bfs_agent_to(self, start, goal, gem_set, grid):
        """BFS path for agent from start to goal, avoiding all gem cells.

        The agent CAN walk on scrolls (they are walkable reference objects).
        Returns list of positions or None.
        """
        avoid = set(gem_set) - {start, goal}
        path = self.api.bfs_path_positions(start, goal, avoid=avoid)
        if path:
            return path
        return self.api.bfs_path_positions(start, goal)

    def _compute_push_order(self, assignment, targets, grid):
        """Determine the order to push gems so earlier gems don't block later ones.

        Gems whose targets have more adjacent targets (i.e., "inner" positions
        in the pattern) should be pushed first, because they will be harder to
        reach after surrounding gems are placed.

        Returns sorted list of (gem, target) pairs.
        """
        target_set = set(targets)
        needs_move = [(g, t) for g, t in assignment.items() if g != t]
        if not needs_move:
            return []

        def target_adjacency(t):
            """Count how many neighbors of target t are also targets."""
            tx, ty = t
            count = 0
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                if (tx + dx, ty + dy) in target_set:
                    count += 1
            return count

        def wall_adjacency(t):
            """Count how many neighbors of target t are walls/OOB."""
            tx, ty = t
            count = 0
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                if self._is_blocked(tx + dx, ty + dy, grid):
                    count += 1
            return count

        # Sort by: more adjacent targets first (inner positions), then by
        # more wall adjacency (harder to reach later), then by distance
        needs_move.sort(
            key=lambda gt: (
                -target_adjacency(gt[1]),
                -wall_adjacency(gt[1]),
                abs(gt[0][0] - gt[1][0]) + abs(gt[0][1] - gt[1][1]),
            )
        )
        return needs_move

    def _try_clearance_push(self, gem_pos, target_pos, grid, gem_set, assignment):
        """When gem's push to target is blocked by another gem, try to
        temporarily displace the blocker.

        Handles two cases:
        1. Destination blocked: another gem sits where the current gem needs to go
        2. Push position blocked: another gem sits where the agent needs to stand

        Returns action list to execute, or None if clearance not possible.
        """
        ax, ay = self.api.agent_position

        # Find the push plan ignoring other gems
        plan = self._bfs_gem_path(gem_pos, target_pos, grid)
        if not plan:
            return None

        # Walk through the plan to find the first blocked push
        gx, gy = gem_pos
        for push_idx, (pdx, pdy) in enumerate(plan):
            dest_x, dest_y = gx + pdx, gy + pdy
            push_x, push_y = gx - pdx, gy - pdy

            # Identify the blocker — either at dest or at push position
            blocker = None
            if (dest_x, dest_y) in gem_set and (dest_x, dest_y) != gem_pos:
                blocker = (dest_x, dest_y)
            elif (push_x, push_y) in gem_set and (push_x, push_y) != gem_pos:
                blocker = (push_x, push_y)

            if blocker:
                # Try to push the blocker out of the way
                static_obs = self._static_obstacles(grid)
                other_gems_no_blocker = gem_set - {blocker}

                for cdx, cdy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    clear_dest = (blocker[0] + cdx, blocker[1] + cdy)
                    clear_push = (blocker[0] - cdx, blocker[1] - cdy)

                    # Clearance dest must be valid
                    if self._is_blocked(clear_dest[0], clear_dest[1], grid):
                        continue
                    if clear_dest in static_obs:
                        continue
                    if clear_dest in other_gems_no_blocker:
                        continue
                    # Don't push to where the current gem is
                    if clear_dest == gem_pos:
                        continue

                    # Clearance push position must be valid
                    if self._is_blocked(clear_push[0], clear_push[1], grid):
                        continue
                    if clear_push in other_gems_no_blocker:
                        continue
                    # Agent can't push from a position with our gem
                    if clear_push == gem_pos:
                        continue

                    # Don't create corner deadlock for blocker
                    if self._is_corner_deadlocked(
                        clear_dest[0], clear_dest[1], grid
                    ):
                        continue

                    # Navigate agent to push position for blocker
                    path = self._bfs_agent_to(
                        (ax, ay), clear_push, gem_set, grid
                    )
                    if path and len(path) >= 2:
                        actions = self.api.positions_to_actions(path)
                        # Add the push action for the clearance
                        push_action = self.api.step_action(cdx, cdy)
                        if actions and push_action is not None:
                            return actions + [push_action]

                return None

            # Advance simulated gem position
            gx, gy = dest_x, dest_y

        return None

    # ------------------------------------------------------------------
    # main planning
    # ------------------------------------------------------------------

    def _recompute_targets_with_deadlocks(self):
        """Identify immovable gems and compute targets that respect them."""
        grid = self.api.grid
        gems = self._find_gems()

        immovable = set()
        for gx, gy in gems:
            if self._is_corner_deadlocked(gx, gy, grid):
                immovable.add((gx, gy))

        targets = self._compute_targets(immovable_gems=immovable)

        if not targets and immovable:
            targets = self._compute_targets()

        return targets

    def _check_all_reachable(self, gems, targets, grid):
        """Check if all gems can still reach at least one target via pushes."""
        from itertools import permutations

        n = len(gems)
        if n != len(targets):
            return False

        gem_reach = []
        for g in gems:
            if self._is_corner_deadlocked(g[0], g[1], grid):
                gem_reach.append({g})
            else:
                gem_reach.append(self._gem_reachable_set(g, grid))

        # Check if any valid assignment exists
        for perm in permutations(range(n)):
            feasible = True
            for i in range(n):
                t = targets[perm[i]]
                if gems[i] == t:
                    continue
                if t not in gem_reach[i]:
                    feasible = False
                    break
            if feasible:
                return True
        return False

    def plan(self):
        grid = self.api.grid
        ax, ay = self.api.agent_position
        gems = self._find_gems()
        gem_set = set(gems)

        # If we have a clearance plan in progress, execute it
        if self._clearance_plan:
            self.action_queue = [self._clearance_plan.pop(0)]
            if not self._clearance_plan:
                self._clearance_plan = None
                self._gem_push_plan = None  # Recompute after clearance
            return

        # Check stuck
        is_stuck = self._check_stuck(gems)

        # Compute targets (with deadlock/feasibility awareness)
        if self._cached_targets is None:
            self._cached_targets = self._recompute_targets_with_deadlocks()
            self._target_attempts += 1
        targets = self._cached_targets

        # Verify targets are still reachable from current gem positions.
        # Gems may have been pushed into wall-locked positions during execution.
        if targets and self._target_attempts < 5:
            if not self._check_all_reachable(gems, targets, grid):
                self._cached_targets = None
                self._skip_gems.clear()
                self._gem_push_plan = None
                self._current_gem = None
                self._state_history.clear()
                self.action_queue = [0]
                return

        if not targets:
            self.action_queue = [0]
            return

        # Recompute assignment based on current gem positions
        assignment = self._compute_assignment(gems, targets)

        # Find gems that still need to move, in optimal push order
        ordered_moves = self._compute_push_order(assignment, targets, grid)

        if not ordered_moves:
            self.action_queue = [0]
            return

        # If stuck, skip the gem we were targeting and clear push plan
        if is_stuck and self._current_gem is not None:
            self._skip_gems.add(self._current_gem)
            self._state_history.clear()
            self._gem_push_plan = None
            # After skipping multiple gems, try recomputing targets
            if len(self._skip_gems) >= 2 and self._target_attempts < 5:
                self._cached_targets = None
                self._skip_gems.clear()
                self._gem_push_plan = None
                self._current_gem = None
                self.action_queue = [0]
                return

        # If we have an active push plan, check if it's still valid
        if self._gem_push_plan and self._current_gem:
            # Check gem is still at expected position
            if self._current_gem not in gem_set:
                self._gem_push_plan = None

        # Filter out skipped gems
        available = [(g, t) for g, t in ordered_moves if g not in self._skip_gems]
        if not available:
            self._skip_gems.clear()
            available = ordered_moves

        # If we have an active push plan for a gem, continue it
        if (
            self._gem_push_plan
            and self._current_gem in gem_set
            and self._current_gem not in self._skip_gems
        ):
            gx, gy = self._current_gem
            pdx, pdy = self._gem_push_plan[0]
            push_pos = (gx - pdx, gy - pdy)

            if self._can_push_now(gx, gy, pdx, pdy, grid, gem_set - {(gx, gy)}):
                if (ax, ay) == push_pos:
                    # Execute the push
                    a = self.api.step_action(pdx, pdy)
                    if a is not None:
                        self._gem_push_plan = self._gem_push_plan[1:]
                        if not self._gem_push_plan:
                            self._gem_push_plan = None
                        self.action_queue = [a]
                        return

                # Navigate to push position
                path = self._bfs_agent_to(
                    (ax, ay), push_pos, gem_set, grid
                )
                if path and len(path) >= 2:
                    actions = self.api.positions_to_actions(path)
                    if actions:
                        self.action_queue = [actions[0]]
                        return

            # Push blocked — try clearance (temporarily move blocking gem)
            target_for_gem = assignment.get(self._current_gem)
            if target_for_gem:
                clearance = self._try_clearance_push(
                    self._current_gem, target_for_gem, grid, gem_set, assignment
                )
                if clearance:
                    self._clearance_plan = clearance[1:]
                    self.action_queue = [clearance[0]]
                    return

            # Push plan invalid, recompute
            self._gem_push_plan = None

        # Pick the gem-target pair to work on.
        # Use the computed push order (inner targets first).
        # For each, try BFS with other gems as obstacles first, then without.
        best_pair = None
        best_plan = None
        fallback_pair = None
        fallback_plan = None

        for g, t in available:
            # Try BFS with other gems as obstacles
            others = gem_set - {g}
            plan = self._bfs_gem_path(g, t, grid, others)
            if plan is not None:
                best_pair = (g, t)
                best_plan = plan
                break
            else:
                # Try without other gems (need clearance later)
                plan = self._bfs_gem_path(g, t, grid)
                if plan is not None and fallback_pair is None:
                    fallback_pair = (g, t)
                    fallback_plan = plan

        if best_pair is None and fallback_pair is None:
            # No gem can reach its target at all — try clearance for first one
            for g, t in available:
                clearance = self._try_clearance_push(
                    g, t, grid, gem_set, assignment
                )
                if clearance:
                    self._current_gem = g
                    self._clearance_plan = clearance[1:]
                    self.action_queue = [clearance[0]]
                    return

            # Recompute targets if we haven't tried too many times
            if self._target_attempts < 5:
                self._cached_targets = None
                self._skip_gems.clear()
                self._gem_push_plan = None
                self._current_gem = None
                self._state_history.clear()
                self.action_queue = [0]
                return

            self._skip_gems.clear()
            self.action_queue = [0]
            return

        if best_pair is not None:
            gem_pos, target_pos = best_pair
            push_plan = best_plan
        else:
            gem_pos, target_pos = fallback_pair
            push_plan = fallback_plan

        self._current_gem = gem_pos
        gx, gy = gem_pos
        self._gem_push_plan = push_plan

        # Execute first push in the plan
        pdx, pdy = push_plan[0]
        push_pos = (gx - pdx, gy - pdy)

        if self._can_push_now(gx, gy, pdx, pdy, grid, gem_set - {gem_pos}):
            if (ax, ay) == push_pos:
                a = self.api.step_action(pdx, pdy)
                if a is not None:
                    self._gem_push_plan = self._gem_push_plan[1:]
                    if not self._gem_push_plan:
                        self._gem_push_plan = None
                    self.action_queue = [a]
                    return

            path = self._bfs_agent_to((ax, ay), push_pos, gem_set, grid)
            if path and len(path) >= 2:
                actions = self.api.positions_to_actions(path)
                if actions:
                    self.action_queue = [actions[0]]
                    return

        # Blocked — try clearance before skipping
        clearance = self._try_clearance_push(
            gem_pos, target_pos, grid, gem_set, assignment
        )
        if clearance:
            self._clearance_plan = clearance[1:]
            self.action_queue = [clearance[0]]
            return

        # Can't execute planned push right now (blocked by dynamic obstacle).
        # Skip this gem temporarily — another gem likely needs to move first.
        self._gem_push_plan = None
        self._skip_gems.add(gem_pos)
        self.action_queue = [0]


@register_oracle("TaskInterference-v0")
class TaskInterferenceOracle(OracleAgent):
    """Balance RED and BLUE meters by alternating collection.

    Strategy:
    - Alternate between collecting GEM (red) and ORB (blue) items.
    - Read last collected color from task config to decide next target.
    - Among candidates of the desired color, pick the nearest.
    - If the desired color is exhausted, pick the other color.
    - On hold-items, stay put until collected.
    - Re-evaluate target every step (items flee and change position).
    """

    def _nav_to(self, target):
        ax, ay = self.api.agent_position
        path = self.api.bfs_path_positions((ax, ay), target)
        if path:
            acts = self.api.positions_to_actions(path)
            if acts:
                self.action_queue = [acts[0]]
                return
        self.action_queue = self.api.move_toward(*target)

    def plan(self):
        config = self.api.task_config
        ax, ay = self.api.agent_position

        # Check if we're standing on a hold-item and should stay
        hold_items = config.get("_hold_items", set())
        grid = self.api.grid
        from agentick.core.types import ObjectType

        if isinstance(hold_items, set):
            on_hold = (ax, ay) in hold_items
        else:
            on_hold = tuple((ax, ay)) in {tuple(p) for p in hold_items}

        if on_hold:
            obj = int(grid.objects[ay, ax])
            if obj in (int(ObjectType.GEM), int(ObjectType.ORB)):
                self.action_queue = [0]
                return

        # Read live item positions from config (updated after flee)
        live_red = [tuple(p) for p in config.get("_live_red", [])]
        live_blue = [tuple(p) for p in config.get("_live_blue", [])]

        if not live_red and not live_blue:
            self.action_queue = [0]
            return

        # Alternate: pick the opposite color from last *actually* collected
        # (read from task config, not self-tracked, to avoid desync)
        last_color = config.get("_last_collected_color")
        want_color = None
        if last_color == "red" and live_blue:
            want_color = "blue"
        elif last_color == "blue" and live_red:
            want_color = "red"

        # Build candidate list for the wanted color
        if want_color == "red":
            candidates = live_red
        elif want_color == "blue":
            candidates = live_blue
        else:
            # No preference — pick nearest overall
            candidates = live_red + live_blue

        if not candidates:
            # Desired color exhausted, pick remaining
            candidates = live_red + live_blue

        if not candidates:
            self.action_queue = [0]
            return

        # Pick nearest candidate
        best_pos = min(
            candidates, key=lambda p: abs(p[0] - ax) + abs(p[1] - ay)
        )
        self._nav_to(best_pos)


def _interact_with_door(api, door_pos, avoid=None):
    """Return a single action to progress toward opening a solid door.

    Uses interact_adjacent from helpers.py which handles the full
    adjacent+face+INTERACT pattern.  If the door is not yet adjacent,
    BFS routes around coins (via *avoid*).
    """
    return interact_adjacent(
        api.agent_position,
        api.agent.orientation,
        door_pos,
        api.grid,
        api,
    )


@register_oracle("DeceptiveReward-v0")
class DeceptiveRewardOracle(OracleAgent):
    """Ignore all coins and decoy targets; BFS straight to true GOAL.

    For hard+ difficulties the true path is behind key+door pairs.
    Keys are walkable (auto-pickup on step). Doors are SOLID and must
    be opened by standing adjacent, facing the door, then INTERACTing
    with the matching key in inventory.

    Strategy (re-planned every step):
    1. Try BFS to goal (avoiding coins/targets). If reachable → go.
    2. If blocked by closed door and we hold a matching key → approach
       door, face it, INTERACT to open.
    3. If we need a key → BFS to the nearest key (avoiding coins).
    """

    def plan(self):
        grid = self.api.grid
        ax, ay = self.api.agent_position
        agent = self.api.agent

        # Build set of coin + target positions to avoid (stepping on coins
        # closes the gate to the true path).
        coin_cells = set()
        for y in range(grid.height):
            for x in range(grid.width):
                if int(grid.objects[y, x]) == int(ObjectType.COIN):
                    coin_cells.add((x, y))
                if int(grid.objects[y, x]) == int(ObjectType.TARGET):
                    coin_cells.add((x, y))
        avoid = coin_cells - {(ax, ay)} or None

        # --- 1. Try direct BFS to goal (respects blocking objects) ---
        goal = self.api.get_nearest("goal")
        if not goal:
            self.action_queue = [0]
            return

        path = self.api.bfs_path_positions((ax, ay), goal.position, avoid=avoid)
        if path:
            actions = self.api.positions_to_actions(path)
            if actions:
                self.action_queue = [actions[0]]
                return

        # --- 2. Goal unreachable — handle closed doors ---
        # Identify all closed doors and their colors
        closed_doors: dict[int, tuple[int, int]] = {}
        open_door_cells: set[tuple[int, int]] = set()
        for y in range(grid.height):
            for x in range(grid.width):
                if int(grid.objects[y, x]) == int(ObjectType.DOOR):
                    meta = int(grid.metadata[y, x])
                    if meta < 10:
                        closed_doors[meta] = (x, y)
                    else:
                        open_door_cells.add((x, y))

        # Which key colors do we hold?
        held_colors = {
            e.properties.get("color") for e in agent.inventory if e.entity_type == "key"
        }

        # 2a. Have a matching key for a closed door → approach and INTERACT
        for color in held_colors:
            if color in closed_doors:
                door_pos = closed_doors[color]
                self.action_queue = _interact_with_door(
                    self.api, door_pos, avoid=avoid,
                )
                return

        # 2b. Need a key → navigate to nearest key (avoiding coins)
        keys = self.api.get_entities_of_type("key")
        if keys:
            nearest_key = min(keys, key=lambda k: k.distance)
            path = self.api.bfs_path_positions(
                (ax, ay), nearest_key.position, avoid=avoid,
            )
            if path:
                actions = self.api.positions_to_actions(path)
                if actions:
                    self.action_queue = [actions[0]]
                    return
            # Try without coin avoidance as fallback
            path = self.api.bfs_path_positions((ax, ay), nearest_key.position)
            if path:
                actions = self.api.positions_to_actions(path)
                if actions:
                    self.action_queue = [actions[0]]
                    return

        # Fallback
        self.action_queue = self.api.move_toward(*goal.position) or [0]
