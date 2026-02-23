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
    """Toggle color-coded switches in order 0..N-1, then reach the goal.

    Strategy:
    1. Find the next switch to activate (lowest color index not yet ON).
    2. Navigate to it via BFS; the currently-open barrier cells are treated
       as extra_passable so the oracle can walk through them.
    3. After all switches are ON, navigate to the goal.

    The oracle avoids stepping on already-active switches because doing so
    would toggle them OFF and close previously opened barriers. The agent's
    current position is excluded from the avoidance set (already standing
    on a switch does not re-trigger it on the same step).

    One step is queued at a time so the plan is re-evaluated each turn,
    which lets the oracle react to barrier state changes immediately.
    """

    def _get_open_barrier_cells(self, config):
        """Return the set of barrier cells that are currently EMPTY (open).

        A barrier cell belonging to color i is open when switch i is ON
        (the toggle turned it EMPTY). We read the live grid terrain rather
        than inferring from the logical switch state to stay in sync with
        whatever physical state the grid is actually in.
        """
        grid = self.api.grid
        barrier_cells_cfg = config.get("barrier_cells", {})
        open_cells: set[tuple[int, int]] = set()
        for _color_str, cells in barrier_cells_cfg.items():
            for cell in cells:
                cx, cy = cell[0], cell[1]
                if int(grid.terrain[cy, cx]) != int(CellType.WALL):
                    open_cells.add((cx, cy))
        return open_cells

    def plan(self):
        config = self.api.task_config
        active = config.get("_switches_on", set())
        switches = config.get("switch_positions", [])
        ax, ay = self.api.agent_position

        n = len(switches)
        if n == 0:
            return

        # Cells that are physically open right now (passable despite being
        # barrier cells) — treat them as extra_passable in BFS.
        extra = self._get_open_barrier_cells(config)

        # Build the set of active-switch positions to avoid while routing.
        # Stepping on an active switch toggles it OFF, which re-closes its
        # barrier and opens the complementary one — disrupting the plan.
        # Exclude the agent's own cell; standing there already is harmless.
        active_switch_positions: set[tuple[int, int]] = set()
        for i in active:
            if i < n:
                pos = tuple(switches[i])
                if pos != (ax, ay):
                    active_switch_positions.add(pos)

        # Determine the next switch that needs to be activated.
        # The canonical order is 0, 1, 2, ..., N-1.
        next_target: tuple[int, int] | None = None
        next_idx: int = -1
        for i in range(n):
            if i not in active:
                next_target = tuple(switches[i])
                next_idx = i
                break

        if next_target is not None:
            sx, sy = next_target
            avoid = active_switch_positions - {(sx, sy)}

            # Primary attempt: avoid active switches, use open barriers.
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

            # Retry without avoid constraint (path may route through active
            # switch positions if there is no other way).
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

            # Last resort: plain BFS ignoring extra_passable hints.
            path = self.api.bfs_path_positions((ax, ay), (sx, sy))
            if path:
                actions = self.api.positions_to_actions(path)
                if actions:
                    self.action_queue = [actions[0]]
                    return

            # Switch is unreachable — fall back to move_toward.
            fallback = self.api.move_toward(sx, sy)
            if fallback:
                self.action_queue = fallback
            return

        # All switches are ON — navigate to the goal.
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

            # Retry without avoid (in case all routes pass through a switch).
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

            # Final fallback.
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
