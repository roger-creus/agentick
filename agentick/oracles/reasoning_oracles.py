"""Oracle bots for reasoning tasks."""

from __future__ import annotations

from agentick.core.types import CellType, ObjectType
from agentick.oracles.base import OracleAgent
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

    def _try_push(self, bx, by, tx, ty, ax, ay, grid, box_set, hazards, target_set):
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
            if not grid.is_walkable((pfx, pfy)):
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

            if (ax, ay) == push_from:
                step = self.api.step_action(pdx, pdy)
                if step is not None:
                    return step
            else:
                # Navigate to push position avoiding boxes AND hazards
                avoid = box_set | hazards
                path = self.api.bfs_path_positions(
                    (ax, ay),
                    push_from,
                    avoid=avoid,
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

        # Find current box positions from grid
        boxes = []
        for y in range(grid.height):
            for x in range(grid.width):
                if grid.objects[y, x] == ObjectType.BOX:
                    boxes.append((x, y))

        if not boxes or not targets:
            return

        ax, ay = self.api.agent_position
        box_set = set(boxes)
        target_set = set(tuple(t) for t in targets)
        hazards = self._get_hazard_cells(grid)

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
            )
            if action is not None:
                self.action_queue = [action]
                return


@register_oracle("CausalChain-v0")
class CausalChainOracle(OracleAgent):
    """Activate levers in causal order, then reach goal."""

    def plan(self):
        config = self.api.task_config
        all_activated = config.get("_all_activated", False)

        if not all_activated:
            levers = self.api.get_entities_of_type("lever")
            if levers:
                for lev in sorted(levers, key=lambda l: l.distance):
                    path = self.api.path_to(*lev.position)
                    if path:
                        self.action_queue = path
                        return
            return

        goal = self.api.get_nearest("goal")
        if goal:
            self.action_queue = self.api.move_to(*goal.position)


@register_oracle("SwitchCircuit-v0")
class SwitchCircuitOracle(OracleAgent):
    """Activate switches in dependency order, then reach goal chamber.

    Effect cells are initially walls/hazards. They become EMPTY only when
    their switch is activated. Only use extra_passable for already-activated
    switch effects. Takes ONE step at a time.
    """

    def plan(self):
        config = self.api.task_config
        active = config.get("_switches_on", set())
        switches = config.get("switch_positions", [])
        deps = config.get("circuit_deps", {})
        effects = config.get("switch_effects", [])
        gate_pos = config.get("gate_pos")
        ax, ay = self.api.agent_position

        # Build passable cells from ALREADY-activated effects only
        extra = set()
        for i in active:
            if i < len(effects):
                for cell in effects[i].get("cells", []):
                    extra.add(tuple(cell))

        if gate_pos and len(active) >= len(switches):
            extra.add(tuple(gate_pos))

        # Build set of active switch positions to avoid when not targeting
        # them. Stepping on an active switch toggles it OFF, which breaks
        # the circuit and causes infinite re-activation loops.
        # Exclude the agent's current position — it is already there and
        # leaving it will not toggle anything (only *entering* triggers).
        active_switch_positions = set()
        for i in active:
            if i < len(switches):
                pos = tuple(switches[i])
                if pos != (ax, ay):
                    active_switch_positions.add(pos)

        # Find next activatable switch
        for i, spos in enumerate(switches):
            sx, sy = spos
            if i in active:
                continue
            prereqs = deps.get(i, deps.get(str(i), []))
            if all(p in active for p in prereqs):
                # Avoid other active switches while navigating to this one
                avoid = active_switch_positions - {(sx, sy)}
                # Try with currently-active effects as passable
                path = self.api.bfs_path_positions(
                    (ax, ay),
                    (sx, sy),
                    extra_passable=extra,
                    avoid=avoid,
                )
                if path:
                    actions = self.api.positions_to_actions(path)
                    if actions:
                        self.action_queue = [actions[0]]
                        return
                # Retry without avoid (in case avoiding creates no path)
                path = self.api.bfs_path_positions(
                    (ax, ay),
                    (sx, sy),
                    extra_passable=extra,
                )
                if path:
                    actions = self.api.positions_to_actions(path)
                    if actions:
                        self.action_queue = [actions[0]]
                        return
                # Try plain BFS (no extra passable)
                path = self.api.bfs_path_positions((ax, ay), (sx, sy))
                if path:
                    actions = self.api.positions_to_actions(path)
                    if actions:
                        self.action_queue = [actions[0]]
                        return
                # Switch unreachable — skip to next or noop
                continue

        # All switches active — go to goal through gate, avoiding switches
        if len(active) >= len(switches):
            goal = self.api.get_nearest("goal")
            if goal:
                path = self.api.bfs_path_positions(
                    (ax, ay),
                    goal.position,
                    extra_passable=extra,
                    avoid=active_switch_positions,
                )
                if path:
                    actions = self.api.positions_to_actions(path)
                    if actions:
                        self.action_queue = [actions[0]]
                        return
                # Fallback: try without avoid (better than stuck forever)
                path = self.api.bfs_path_positions(
                    (ax, ay),
                    goal.position,
                    extra_passable=extra,
                )
                if path:
                    actions = self.api.positions_to_actions(path)
                    if actions:
                        self.action_queue = [actions[0]]
                        return
                self.action_queue = self.api.move_toward(*goal.position)


@register_oracle("SymbolMatching-v0")
class SymbolMatchingOracle(OracleAgent):
    """Pick up symbol items and deliver to matching targets.

    Uses config pair_info for oracle knowledge of which item matches which target.
    Reads task internal _carrying state directly for reliable carrying detection.
    Avoids fake items and wrong targets during navigation.
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

    def _get_all_target_positions(self):
        """Return set of all TARGET cell positions on grid."""
        grid = self.api.grid
        targets = set()
        for y in range(grid.height):
            for x in range(grid.width):
                if grid.objects[y, x] == ObjectType.TARGET:
                    targets.add((x, y))
        return targets

    def _get_all_symbol_positions(self):
        """Return set of all symbol item positions on the grid."""
        grid = self.api.grid
        symbols = set()
        for y in range(grid.height):
            for x in range(grid.width):
                if int(grid.objects[y, x]) in self._SYMBOL_SET:
                    symbols.add((x, y))
        return symbols

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

        # Build set of all target positions for avoidance when navigating
        all_targets = self._get_all_target_positions()
        # Build set of all symbol positions (for avoiding fakes)
        all_symbols = self._get_all_symbol_positions()

        # Build set of pair item positions (legitimate items)
        pair_item_positions = set()
        for p in pair_info:
            ix, iy = p["item_pos"]
            if grid.objects[iy, ix] == ObjectType(p["symbol_type"]):
                pair_item_positions.add((ix, iy))

        # Fake items = all symbols NOT in pair_item_positions
        fake_positions = all_symbols - pair_item_positions

        if carrying is None:
            # Not carrying — find an uncollected item to pick up
            best_pair = None
            best_dist = 999
            for p in pair_info:
                ix, iy = p["item_pos"]
                sym_type = p["symbol_type"]
                tx, ty = p["target_pos"]
                if (
                    grid.objects[iy, ix] == ObjectType(sym_type)
                    and grid.objects[ty, tx] == ObjectType.TARGET
                ):
                    d = abs(ix - ax) + abs(iy - ay)
                    if d < best_dist:
                        best_dist = d
                        best_pair = p
            if best_pair:
                ix, iy = best_pair["item_pos"]
                # Avoid: all targets + all fake symbol positions
                # (stepping on fakes picks them up, breaking the plan)
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
            # Carrying — deliver to the CORRECT matching target
            correct_target = None
            for p in pair_info:
                if p["symbol_type"] == carrying:
                    tx, ty = p["target_pos"]
                    if grid.objects[ty, tx] == ObjectType.TARGET:
                        correct_target = (tx, ty)
                        break

            if correct_target:
                # Avoid: all OTHER targets + all symbol items (don't pick up more)
                wrong_targets = all_targets - {correct_target}
                avoid = wrong_targets | all_symbols
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

            # Carrying a fake or unknown item — try to drop on any target
            # or just move toward a pair item (fake will be lost on mismatch)
            if all_targets:
                nearest_target = min(all_targets, key=lambda t: abs(t[0] - ax) + abs(t[1] - ay))
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
    """Visit switch to reveal rule, then go to correct target."""

    def plan(self):
        config = self.api.task_config
        revealed = config.get("_rule_revealed", False)

        if not revealed:
            switches = self.api.get_entities_of_type("switch")
            if switches:
                self.action_queue = self.api.move_to(*switches[0].position)
                return

        goal = self.api.get_nearest("goal")
        if goal:
            self.action_queue = self.api.move_to(*goal.position)
