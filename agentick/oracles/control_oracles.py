"""Oracle bots for control tasks."""

from __future__ import annotations

from agentick.core.types import CellType
from agentick.oracles.base import OracleAgent
from agentick.oracles.registry import register_oracle


@register_oracle("ChaseEvade-v0")
class ChaseEvadeOracle(OracleAgent):
    """Evade enemies to survive for the required number of steps.

    Strategy: Read true enemy positions from config (not grid objects layer,
    which loses overlapping enemies). Account for enemy_speed (enemies may
    move multiple cells per turn). Use deeper lookahead proportional to
    enemy speed. Simulate enemy-enemy blocking (enemies cannot overlap).
    Use freeze time to create distance. Use walls as shields.
    """

    _DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    def plan(self):
        grid = self.api.grid
        ax, ay = self.api.agent_position
        config = self.api.task_config

        # Read TRUE enemy list from config (grid.objects loses overlaps)
        enemy_positions = list(config.get("_enemies", []))
        if not enemy_positions:
            self.action_queue = [0]
            return

        enemy_speed = config.get("enemy_speed", 1)
        freeze_remaining = config.get("_freeze_remaining", 0)

        name_map = self.api.action_name_to_int
        deltas = [(0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)]
        move_names = ["move_up", "move_down", "move_left", "move_right", "noop"]

        def _walkable(x, y):
            if not (0 < x < grid.width - 1 and 0 < y < grid.height - 1):
                return False
            t = int(grid.terrain[y, x])
            return t in (0, 3, 4)  # EMPTY, WATER, ICE

        def _simulate_enemies(enemies, tx, ty, speed):
            """Simulate all enemies chasing (tx, ty) for `speed` sub-steps.

            Accounts for enemy-enemy blocking: enemies cannot move onto
            cells occupied by other enemies.
            """
            positions = list(enemies)
            for _ in range(speed):
                occupied = set(positions)
                new_positions = []
                for i, (ex, ey) in enumerate(positions):
                    best, best_d = (ex, ey), abs(ex - tx) + abs(ey - ty)
                    for ddx, ddy in self._DIRS:
                        nnx, nny = ex + ddx, ey + ddy
                        if _walkable(nnx, nny) and (nnx, nny) not in occupied:
                            d = abs(nnx - tx) + abs(nny - ty)
                            if d < best_d:
                                best_d = d
                                best = (nnx, nny)
                    # Reserve the cell so later enemies can't also move there
                    if best != (ex, ey):
                        occupied.discard((ex, ey))
                        occupied.add(best)
                    new_positions.append(best)
                positions = new_positions
            return positions

        # Lookahead depth: moderate for speed-1 (too deep causes pessimism),
        # deeper for speed-2+ enemies where immediate threats need detection
        depth = 4 if enemy_speed <= 1 else enemy_speed * 3

        # If enemies are frozen, use time to maximize distance from enemies
        if freeze_remaining > 0:
            # Still consider potion pickup during freeze for chaining
            potions = self.api.get_entities_of_type("potion")
            if potions:
                nearest_p = min(potions, key=lambda p: p.distance)
                if nearest_p.distance <= freeze_remaining:
                    px, py = nearest_p.position
                    safe = not any((px, py) == ep for ep in enemy_positions)
                    if safe:
                        self.action_queue = self.api.move_toward(px, py)
                        return

            # Maximize distance from enemies while frozen
            best_action = 0
            best_score = -99999
            for i, (dx, dy) in enumerate(deltas):
                if move_names[i] not in name_map:
                    continue
                nx, ny = ax + dx, ay + dy
                if dx == 0 and dy == 0:
                    nx, ny = ax, ay
                elif not _walkable(nx, ny):
                    continue
                total_d = sum(abs(nx - ex) + abs(ny - ey) for ex, ey in enemy_positions)
                min_d = min(abs(nx - ex) + abs(ny - ey) for ex, ey in enemy_positions)
                escape_count = sum(1 for ddx, ddy in self._DIRS if _walkable(nx + ddx, ny + ddy))
                cx, cy = grid.width / 2, grid.height / 2
                center_bonus = -(abs(nx - cx) + abs(ny - cy))
                score = min_d * 200 + total_d * 10 + escape_count * 15 + center_bonus
                if score > best_score:
                    best_score = score
                    best_action = name_map[move_names[i]]
            self.action_queue = [best_action]
            return

        # Collect potion only if safe
        potions = self.api.get_entities_of_type("potion")
        if potions:
            nearest_p = min(potions, key=lambda p: p.distance)
            min_enemy_dist = min(abs(ax - ex) + abs(ay - ey) for ex, ey in enemy_positions)
            # Only go for potion if we have a comfortable lead on enemies
            # Need more margin for faster enemies
            margin = 2 + enemy_speed
            if nearest_p.distance <= 3 and min_enemy_dist > nearest_p.distance + margin:
                self.action_queue = self.api.move_toward(*nearest_p.position)
                return

        # Evaluate each initial move with deep lookahead
        best_action = 0
        best_score = -99999

        for i, (dx, dy) in enumerate(deltas):
            if move_names[i] not in name_map:
                continue
            nx, ny = ax + dx, ay + dy
            if dx == 0 and dy == 0:
                nx, ny = ax, ay
            elif not _walkable(nx, ny):
                continue

            # Hard reject: stepping onto current enemy position
            if any((nx, ny) == ep for ep in enemy_positions):
                continue

            score = self._evaluate_deep(
                nx,
                ny,
                enemy_positions,
                enemy_speed,
                grid,
                _walkable,
                _simulate_enemies,
                depth,
            )

            if score > best_score:
                best_score = score
                best_action = name_map[move_names[i]]

        self.action_queue = [best_action]

    def _evaluate_deep(
        self,
        ax,
        ay,
        enemies,
        speed,
        grid,
        walkable_fn,
        sim_fn,
        depth,
    ):
        """Evaluate a position with minimax-style lookahead.

        At each level: simulate enemy movement (adversary), then pick
        the best agent move (maximizer). Returns the leaf score of the
        best play sequence.
        """
        if depth == 0 or not enemies:
            return self._score_position(ax, ay, enemies, grid, walkable_fn)

        # Simulate enemies chasing toward agent
        next_enemies = sim_fn(enemies, ax, ay, speed)

        # Check if any enemy caught us
        if any((ex, ey) == (ax, ay) for ex, ey in next_enemies):
            return -100000 * depth  # earlier death is worse

        enemy_set = set(next_enemies)

        # Try all agent moves, pick best
        best_child = -100000
        for ddx, ddy in [(0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)]:
            nx, ny = ax + ddx, ay + ddy
            if ddx == 0 and ddy == 0:
                nx, ny = ax, ay
            elif not walkable_fn(nx, ny):
                continue
            if (nx, ny) in enemy_set:
                continue

            child_score = self._evaluate_deep(
                nx,
                ny,
                next_enemies,
                speed,
                grid,
                walkable_fn,
                sim_fn,
                depth - 1,
            )
            best_child = max(best_child, child_score)

        return best_child

    def _score_position(self, ax, ay, enemies, grid, walkable_fn):
        """Score a leaf position based on distance, escape routes, centrality."""
        if not enemies:
            return 99999

        dists = [abs(ax - ex) + abs(ay - ey) for ex, ey in enemies]
        min_d = min(dists)
        total_d = sum(dists)

        # Count escape routes and best 2-step min distance
        escape_count = 0
        best_2step_min = 0
        for ddx, ddy in self._DIRS:
            fx, fy = ax + ddx, ay + ddy
            if walkable_fn(fx, fy):
                escape_count += 1
                md = min(abs(fx - ex) + abs(fy - ey) for ex, ey in enemies)
                best_2step_min = max(best_2step_min, md)

        # Prefer center of grid (more escape routes, fewer dead ends)
        cx, cy = grid.width / 2, grid.height / 2
        center_bonus = -(abs(ax - cx) + abs(ay - cy))

        # Heavy penalty for dead ends
        trap_penalty = 0
        if escape_count <= 1:
            trap_penalty = -1000
        elif escape_count == 2:
            trap_penalty = -200

        # Encirclement penalty: count how many quadrants have enemies
        quadrants = set()
        for ex, ey in enemies:
            qx = 1 if ex >= ax else -1
            qy = 1 if ey >= ay else -1
            quadrants.add((qx, qy))
        encircle_penalty = -100 * max(0, len(quadrants) - 2)

        return (
            min_d * 300
            + total_d * 20
            + best_2step_min * 30
            + escape_count * 25
            + center_bonus * 3
            + trap_penalty
            + encircle_penalty
        )


@register_oracle("Herding-v0")
class HerdingOracle(OracleAgent):
    """Herd sheep into pen using flee-physics simulation.

    Two-phase approach:
    Phase A (approach): Navigate to the best push position for the current
        target sheep, avoiding other sheep to not spook them.
    Phase B (push): When within flee range of target sheep, use per-step
        greedy evaluation that simulates flee physics on ALL nearby sheep
        to pick the best single move. Avoids pushing penned sheep out.

    Sheep are worked on one at a time (closest to pen first). A stuck
    counter forces target switching when progress stalls.
    """

    _FLEE_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    _MOVE_DELTAS = [(0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)]
    _MOVE_NAMES = ["move_up", "move_down", "move_left", "move_right", "noop"]

    def __init__(self, env):
        super().__init__(env)
        self._pen_cells: set[tuple[int, int]] = set()
        self._pen_cx = 0.0
        self._pen_cy = 0.0
        self._target_idx: int | None = None
        self._stuck_counter = 0
        self._prev_out_count: int | None = None

    def reset(self, obs, info):
        config = self.api.task_config
        self._pen_cells = set(map(tuple, config.get("pen_cells", [])))
        if self._pen_cells:
            self._pen_cx = sum(p[0] for p in self._pen_cells) / len(self._pen_cells)
            self._pen_cy = sum(p[1] for p in self._pen_cells) / len(self._pen_cells)
        self._target_idx = None
        self._stuck_counter = 0
        self._prev_out_count = None
        super().reset(obs, info)

    # -- Flee simulation (mirrors HerdingTask exactly) -----------------

    def _simulate_flee(self, sx, sy, ax, ay, grid):
        """Predict where sheep at (sx,sy) flees when agent is at (ax,ay)."""
        dist = abs(sx - ax) + abs(sy - ay)
        if dist > 2:
            return (sx, sy)
        best = (sx, sy)
        best_d = dist
        for dx, dy in self._FLEE_DIRS:
            nx, ny = sx + dx, sy + dy
            d = abs(nx - ax) + abs(ny - ay)
            if (
                0 < nx < grid.width - 1
                and 0 < ny < grid.height - 1
                and int(grid.terrain[ny, nx]) == CellType.EMPTY
                and d > best_d
            ):
                best_d = d
                best = (nx, ny)
        return best

    def _pen_dist(self, x, y):
        return abs(x - self._pen_cx) + abs(y - self._pen_cy)

    def _walkable(self, x, y, grid):
        return (
            0 < x < grid.width - 1
            and 0 < y < grid.height - 1
            and int(grid.terrain[y, x]) == CellType.EMPTY
        )

    # -- Best push position for a sheep --------------------------------

    def _find_best_push(self, sx, sy, grid, agent_pos=None):
        """Find best agent position to push sheep toward pen.

        Returns (agent_target, flee_result) or None.
        """
        current_pd = self._pen_dist(sx, sy)
        if (sx, sy) in self._pen_cells:
            return None

        candidates = []
        for r in range(1, 3):
            for dx in range(-r, r + 1):
                dy_abs = r - abs(dx)
                for dy in [-dy_abs, dy_abs] if dy_abs > 0 else [0]:
                    cax, cay = sx + dx, sy + dy
                    if not self._walkable(cax, cay, grid):
                        continue
                    fx, fy = self._simulate_flee(sx, sy, cax, cay, grid)
                    new_pd = self._pen_dist(fx, fy)
                    improvement = current_pd - new_pd
                    if improvement > 0:
                        agent_dist = 0
                        if agent_pos:
                            agent_dist = abs(cax - agent_pos[0]) + abs(cay - agent_pos[1])
                        score = improvement * 100 - agent_dist
                        candidates.append((score, (cax, cay), (fx, fy)))

        if candidates:
            candidates.sort(key=lambda c: -c[0])
            return (candidates[0][1], candidates[0][2])

        # Fallback: push toward pen's corner
        target_x = 1 if self._pen_cx < grid.width / 2 else grid.width - 2
        target_y = 1 if self._pen_cy < grid.height / 2 else grid.height - 2
        best_score = -999.0
        best_result = None

        for r in range(1, 3):
            for dx in range(-r, r + 1):
                dy_abs = r - abs(dx)
                for dy in [-dy_abs, dy_abs] if dy_abs > 0 else [0]:
                    cax, cay = sx + dx, sy + dy
                    if not self._walkable(cax, cay, grid):
                        continue
                    fx, fy = self._simulate_flee(sx, sy, cax, cay, grid)
                    if (fx, fy) == (sx, sy):
                        continue
                    old_wd = abs(sx - target_x) + abs(sy - target_y)
                    new_wd = abs(fx - target_x) + abs(fy - target_y)
                    improvement = old_wd - new_wd
                    if improvement > 0:
                        agent_dist = 0
                        if agent_pos:
                            agent_dist = abs(cax - agent_pos[0]) + abs(cay - agent_pos[1])
                        score = improvement * 100 - agent_dist
                        if score > best_score:
                            best_score = score
                            best_result = ((cax, cay), (fx, fy))

        return best_result

    # -- Target selection ----------------------------------------------

    def _pick_target(self, sheep, ax, ay):
        """Pick which sheep to herd. Prefer closest to pen, but also
        consider distance from agent to the sheep's push position."""
        out = [i for i, s in enumerate(sheep) if s not in self._pen_cells]
        if not out:
            return None
        # Keep current target if still outside pen
        if self._target_idx is not None and self._target_idx in out:
            return self._target_idx
        # Pick sheep closest to pen
        return min(out, key=lambda i: self._pen_dist(*sheep[i]))

    # -- Greedy move evaluation ----------------------------------------

    def _score_move(self, nax, nay, sheep, grid):
        """Score agent position (nax, nay) by simulating flee of all
        nearby sheep. Returns total pen-distance improvement.
        Heavily penalises pushing penned sheep out.
        """
        score = 0.0
        for sx, sy in sheep:
            dist = abs(sx - nax) + abs(sy - nay)
            if dist > 2:
                continue
            fx, fy = self._simulate_flee(sx, sy, nax, nay, grid)
            if (fx, fy) == (sx, sy):
                continue
            old_pd = self._pen_dist(sx, sy)
            new_pd = self._pen_dist(fx, fy)
            improvement = old_pd - new_pd

            in_pen = (sx, sy) in self._pen_cells
            flee_in_pen = (fx, fy) in self._pen_cells

            if in_pen and not flee_in_pen:
                score -= 50  # very bad: pushing out of pen
            elif not in_pen and flee_in_pen:
                score += 30  # great: pushing into pen
            elif not in_pen:
                score += improvement * 8
        return score

    # -- Main plan -----------------------------------------------------

    def plan(self):
        config = self.api.task_config
        grid = self.api.grid
        ax, ay = self.api.agent_position

        sheep_raw = config.get("_live_sheep", [])
        if not sheep_raw or not self._pen_cells:
            self.action_queue = [0]
            return

        sheep = [(int(s[0]), int(s[1])) for s in sheep_raw]
        out_count = sum(1 for s in sheep if s not in self._pen_cells)
        if out_count == 0:
            self.action_queue = [0]
            return

        # Stuck detection
        if self._prev_out_count is not None:
            if out_count >= self._prev_out_count:
                self._stuck_counter += 1
            else:
                self._stuck_counter = 0
        self._prev_out_count = out_count

        # Pick target
        tidx = self._pick_target(sheep, ax, ay)
        if tidx is None:
            self.action_queue = [0]
            return

        # Force-switch target if stuck
        if self._stuck_counter > 25:
            out_idx = [i for i, s in enumerate(sheep) if s not in self._pen_cells]
            others = [i for i in out_idx if i != tidx]
            if others:
                tidx = others[0]
                self._stuck_counter = 0

        self._target_idx = tidx
        sx, sy = sheep[tidx]
        dist_s = abs(ax - sx) + abs(ay - sy)

        # PHASE B: If within flee range of target sheep, use greedy eval
        if dist_s <= 2:
            self._do_greedy_move(ax, ay, sheep, grid, sx, sy, tidx)
            return

        # PHASE A: Far from target. Navigate to push position.
        push = self._find_best_push(sx, sy, grid, agent_pos=(ax, ay))
        if push is None:
            self.action_queue = self.api.move_toward(sx, sy)
            return

        target_pos = push[0]
        if (ax, ay) == target_pos:
            # At push position but sheep is >2 away; approach sheep
            self.action_queue = self.api.move_toward(sx, sy)
            return

        # Build penned-sheep protection zone: ALWAYS avoid spooking
        # sheep that are already in the pen
        penned_zone = set()
        for i, s in enumerate(sheep):
            if s in self._pen_cells:
                for fdx in range(-2, 3):
                    for fdy in range(-2, 3):
                        if abs(fdx) + abs(fdy) <= 2 and (fdx, fdy) != (0, 0):
                            penned_zone.add((s[0] + fdx, s[1] + fdy))

        # Navigate with decreasing avoidance strictness
        other_positions = set()
        for i, s in enumerate(sheep):
            if i != tidx:
                other_positions.add(s)

        for level in range(4):
            avoid = set(penned_zone)  # always protect penned sheep
            if level == 0:
                # Full buffer around all other sheep
                for s in other_positions:
                    avoid.add(s)
                    for ddx, ddy in self._FLEE_DIRS:
                        avoid.add((s[0] + ddx, s[1] + ddy))
            elif level == 1:
                # Only other sheep positions
                avoid |= other_positions
            elif level == 2:
                pass  # just penned zone
            else:
                avoid = set()  # no avoidance at all

            avoid.discard(target_pos)
            avoid.discard((ax, ay))

            path = self.api.bfs_path_positions(
                (ax, ay),
                target_pos,
                avoid=avoid if avoid else None,
            )
            if path and len(path) > 1:
                acts = self.api.positions_to_actions(path)
                if acts:
                    self.action_queue = [acts[0]]
                    return

        self.action_queue = self.api.move_toward(target_pos[0], target_pos[1])

    def _do_greedy_move(self, ax, ay, sheep, grid, sx, sy, tidx):
        """Pick best single move using greedy flee simulation.

        Evaluates all 5 actions (4 moves + noop). For each, simulates
        flee responses of all nearby sheep. Also adds:
        - An approach incentive toward the target push position
        - A small penalty for noop to prevent indefinite stalling
        """
        name_map = self.api.action_name_to_int
        push = self._find_best_push(sx, sy, grid, agent_pos=(ax, ay))
        tp = push[0] if push else None

        best_act = 0  # noop default
        best_score = -9999.0

        for i, (dx, dy) in enumerate(self._MOVE_DELTAS):
            if self._MOVE_NAMES[i] not in name_map:
                continue
            is_noop = dx == 0 and dy == 0
            if is_noop:
                nx, ny = ax, ay
            else:
                nx, ny = ax + dx, ay + dy
                if not self._walkable(nx, ny, grid):
                    continue

            score = self._score_move(nx, ny, sheep, grid)

            # Approach incentive: prefer moves that bring us closer to
            # the target push position
            if tp is not None:
                cur_d = abs(ax - tp[0]) + abs(ay - tp[1])
                new_d = abs(nx - tp[0]) + abs(ny - tp[1])
                score += (cur_d - new_d) * 1.5

            # Small penalty for noop to keep the agent moving when
            # nothing beneficial is happening
            if is_noop:
                # But NOT if we already have good flee angle
                fx, fy = self._simulate_flee(sx, sy, ax, ay, grid)
                if self._pen_dist(fx, fy) < self._pen_dist(sx, sy):
                    score += 0.5  # small bonus for waiting with good angle
                else:
                    score -= 0.5  # penalty for idle with bad angle

            if score > best_score:
                best_score = score
                best_act = name_map[self._MOVE_NAMES[i]]

        self.action_queue = [best_act]


@register_oracle("PreciseNavigation-v0")
class PreciseNavigationOracle(OracleAgent):
    """Navigate narrow hazard corridors, collect waypoints, reach goal.

    Must restrict BFS to EMPTY cells only - HAZARD is fatal.
    """

    _EMPTY_ONLY = {0}

    def plan(self):
        config = self.api.task_config
        remaining = config.get("_waypoints_remaining", [])
        ax, ay = self.api.agent_position

        if remaining:
            nearest_wp = min(remaining, key=lambda w: abs(w[0] - ax) + abs(w[1] - ay))
            path = self.api.bfs_path_positions(
                (ax, ay),
                tuple(nearest_wp),
                terrain_ok=self._EMPTY_ONLY,
            )
            if path:
                actions = self.api.positions_to_actions(path)
                if actions:
                    self.action_queue = [actions[0]]
                    return

        goal = self.api.get_nearest("goal")
        if goal:
            path = self.api.bfs_path_positions(
                (ax, ay),
                goal.position,
                terrain_ok=self._EMPTY_ONLY,
            )
            if path:
                self.action_queue = self.api.positions_to_actions(path)
                return


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
