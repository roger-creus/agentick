"""Oracle bots for multi-agent tasks."""

from __future__ import annotations

from collections import deque

from agentick.core.types import CellType, ObjectType
from agentick.oracles.base import OracleAgent
from agentick.oracles.helpers import interact_adjacent
from agentick.oracles.registry import register_oracle


@register_oracle("TagHunt-v0")
class TagHuntOracle(OracleAgent):
    """Tag enemy NPCs by moving onto them.

    Strategy: chase nearest enemy via BFS. If a freeze SWITCH is nearby and
    closer than the nearest enemy, activate it first (navigate adjacent, face,
    INTERACT) to freeze all NPCs for 5 steps, then chase enemies while frozen.

    SWITCHes are solid — the agent must stand adjacent, face, and INTERACT.
    """

    def plan(self):
        enemies = self.api.get_entities_of_type("enemy")
        if not enemies:
            self.action_queue = [0]
            return

        ax, ay = self.api.agent_position
        config = self.api.task_config
        nearest = min(enemies, key=lambda e: e.distance)
        dist_to_enemy = nearest.distance

        # Consider activating a freeze switch if one is nearby
        active_switches = config.get("_active_switches", [])
        freeze_remaining = config.get("_freeze_remaining", 0)
        if active_switches and freeze_remaining == 0:
            # Find nearest switch
            best_sw, best_sw_dist = None, float("inf")
            for sx, sy in active_switches:
                d = abs(ax - sx) + abs(ay - sy)
                if d < best_sw_dist:
                    best_sw_dist, best_sw = d, (sx, sy)
            # If switch is closer than enemy, activate it via adjacent INTERACT
            if best_sw is not None and best_sw_dist < dist_to_enemy:
                grid = self.api.grid
                agent_ori = self.api.agent.orientation
                actions = interact_adjacent(
                    (ax, ay), agent_ori, best_sw, grid, self.api,
                )
                if actions:
                    self.action_queue = actions
                    return

        # Chase nearest enemy
        path = self.api.bfs_path_positions(
            (ax, ay),
            nearest.position,
        )
        if path:
            actions = self.api.positions_to_actions(path)
            if actions:
                self.action_queue = [actions[0]]
                return

        self.action_queue = self.api.move_toward(*nearest.position)


@register_oracle("CooperativeTransport-v0")
class CooperativeTransportOracle(OracleAgent):
    """Push heavy boxes into holes with NPC cooperation.

    Strategy:
    1. Pick the closest (box, hole) pair.
    2. Compute the best push direction to move that box closer to the hole.
    3. Navigate to the push-from position (opposite side of box from push dir).
    4. Stay adjacent to box so the NPC pathfinds to an adjacent side.
    5. When NPC is adjacent to the same box, execute the push.
    6. Repeat until all boxes are delivered.
    """

    _DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def __init__(self, env):
        super().__init__(env)
        self._stuck_counter = 0
        self._last_state = None

    def reset(self, obs, info):
        self._stuck_counter = 0
        self._last_state = None
        super().reset(obs, info)

    def plan(self):
        grid = self.api.grid
        config = self.api.task_config
        ax, ay = self.api.agent_position

        box_positions = config.get("_box_positions", config.get("box_positions", []))
        hole_positions = config.get("_hole_positions", config.get("hole_positions", []))
        npc_pos = tuple(config.get("_npc_pos", config.get("npc_start", [-1, -1])))

        if not box_positions or not hole_positions:
            self.action_queue = [0]
            return

        # Pick the best (box, hole) pair: shortest Manhattan distance
        best_pair = None
        best_dist = float("inf")
        for bp in box_positions:
            bx, by = bp
            for hp in hole_positions:
                hx, hy = hp
                d = abs(bx - hx) + abs(by - hy)
                if d < best_dist:
                    best_dist = d
                    best_pair = (bp, hp)

        if best_pair is None:
            self.action_queue = [0]
            return

        bp, hp = best_pair
        bx, by = bp
        hx, hy = hp

        # Detect stuck state
        state_key = (bx, by, hx, hy, ax, ay)
        if self._last_state == state_key:
            self._stuck_counter += 1
        else:
            self._stuck_counter = 0
        self._last_state = state_key

        # NPC adjacent to this box?
        npc_adj = abs(npc_pos[0] - bx) + abs(npc_pos[1] - by) == 1

        # Score push directions by how much they reduce distance to hole
        push_options = []
        for ddx, ddy in self._DIRS:
            new_bx, new_by = bx + ddx, by + ddy
            new_dist = abs(new_bx - hx) + abs(new_by - hy)
            old_dist = abs(bx - hx) + abs(by - hy)
            improvement = old_dist - new_dist
            push_options.append((ddx, ddy, improvement))
        push_options.sort(key=lambda x: -x[2])

        # Try each push direction
        for pdx, pdy, imp in push_options:
            # Skip options that don't improve distance (unless stuck)
            if imp <= 0 and self._stuck_counter < 3:
                continue

            push_from = (bx - pdx, by - pdy)
            pfx, pfy = push_from

            # push_from must be walkable and in bounds
            if not (0 < pfx < grid.width - 1 and 0 < pfy < grid.height - 1):
                continue
            if not grid.is_walkable(push_from):
                continue

            # Destination for box must be valid
            land = (bx + pdx, by + pdy)
            lx, ly = land
            if not (0 < lx < grid.width - 1 and 0 < ly < grid.height - 1):
                continue
            land_t = int(grid.terrain[ly, lx])
            if land_t == int(CellType.WALL):
                continue
            # Box can land on HOLE (delivery), empty floor, or NPC position (NPC dodges)
            if land_t != int(CellType.HOLE):
                land_obj = int(grid.objects[ly, lx])
                if land_obj not in (int(ObjectType.NONE), int(ObjectType.NPC)):
                    continue

            # If agent is already at push_from and NPC is adjacent -> push!
            if (ax, ay) == push_from and npc_adj:
                step = self.api.step_action(pdx, pdy)
                if step is not None:
                    self.action_queue = [step]
                    return

            # If agent is at push_from but NPC not adjacent yet -> wait (noop)
            if (ax, ay) == push_from:
                self.action_queue = [0]
                return

            # Navigate to push_from position
            avoid = set()
            for bbp in box_positions:
                avoid.add((bbp[0], bbp[1]))
            path = self.api.bfs_path_positions((ax, ay), push_from, avoid=avoid)
            if path:
                actions = self.api.positions_to_actions(path)
                if actions:
                    self.action_queue = [actions[0]]
                    return

        # Fallback: move toward nearest box
        nearest_box = min(box_positions, key=lambda b: abs(ax - b[0]) + abs(ay - b[1]))
        self.action_queue = self.api.move_toward(nearest_box[0], nearest_box[1])


@register_oracle("ChaseEvade-v0")
class ChaseEvadeOracle(OracleAgent):
    """Evade coordinated enemy pack to survive for the required number of steps.

    Strategy: Read true enemy positions + behavior types from config/grid.
    Simulate enemies with type-aware behavior: chasers BFS-chase, ambushers
    target ahead, flankers pincer, trappers block escapes. Deeper lookahead
    for more enemies. Use freeze time to create distance.
    """

    _DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    _DIR_DELTAS = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}

    def plan(self):
        grid = self.api.grid
        ax, ay = self.api.agent_position
        config = self.api.task_config

        # Read TRUE enemy list from config (grid.objects loses overlaps)
        enemy_positions = list(config.get("_enemies", []))
        if not enemy_positions:
            self.action_queue = [0]
            return

        freeze_remaining = config.get("_freeze_remaining", 0)
        agent_dir = config.get("_agent_last_dir", 2)

        # Read behavior types from grid metadata
        enemy_behaviors = []
        for ex, ey in enemy_positions:
            enemy_behaviors.append(int(grid.metadata[ey, ex]))

        name_map = self.api.action_name_to_int
        deltas = [(0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)]
        move_names = ["move_up", "move_down", "move_left", "move_right", "noop"]

        def _walkable(x, y):
            if not (0 < x < grid.width - 1 and 0 < y < grid.height - 1):
                return False
            t = int(grid.terrain[y, x])
            return t in (0, 3, 4)  # EMPTY, WATER, ICE

        def _bfs_dist_map(tx, ty):
            """BFS distance map from target. Returns dict (x,y)->distance."""
            dist = {(tx, ty): 0}
            q = deque([(tx, ty)])
            while q:
                cx, cy = q.popleft()
                cd = dist[(cx, cy)]
                for ddx, ddy in self._DIRS:
                    nx, ny = cx + ddx, cy + ddy
                    if (nx, ny) not in dist and _walkable(nx, ny):
                        dist[(nx, ny)] = cd + 1
                        q.append((nx, ny))
            return dist

        def _greedy_step(ex, ey, tx, ty, occupied):
            """Greedy one-step move toward (tx,ty) using Manhattan distance."""
            best, best_d = (ex, ey), abs(ex - tx) + abs(ey - ty)
            for ddx, ddy in self._DIRS:
                nx, ny = ex + ddx, ey + ddy
                if _walkable(nx, ny) and (nx, ny) not in occupied:
                    d = abs(nx - tx) + abs(ny - ty)
                    if d < best_d:
                        best_d = d
                        best = (nx, ny)
            return best

        def _bfs_step(ex, ey, dist_map, occupied):
            """One-step move following BFS distance gradient."""
            cur_d = dist_map.get((ex, ey), 99999)
            best, best_d = (ex, ey), cur_d
            for ddx, ddy in self._DIRS:
                nx, ny = ex + ddx, ey + ddy
                if _walkable(nx, ny) and (nx, ny) not in occupied:
                    d = dist_map.get((nx, ny), 99999)
                    if d < best_d:
                        best_d = d
                        best = (nx, ny)
            return best

        def _simulate_enemies(enemies, behaviors, ax_sim, ay_sim, adir):
            """Type-aware enemy simulation: 1 BFS + greedy per non-chaser.

            Chasers: BFS toward agent (accurate).
            Ambushers: greedy toward 4-ahead target, chase if close.
            Flankers: greedy toward mirror of nearest ally, chase if close.
            Trappers: greedy toward agent-adj cell that minimizes escapes.
            """
            dist_map = _bfs_dist_map(ax_sim, ay_sim)
            positions = list(enemies)
            occupied = set(positions)
            new_positions = []

            for i, (ex, ey) in enumerate(positions):
                beh = behaviors[i] if i < len(behaviors) else 1
                occupied.discard((ex, ey))

                if beh == 2:  # Ambusher
                    d_agent = abs(ex - ax_sim) + abs(ey - ay_sim)
                    if d_agent <= 3:
                        npos = _bfs_step(ex, ey, dist_map, occupied)
                    else:
                        ddx, ddy = self._DIR_DELTAS.get(adir, (0, 1))
                        tx = max(1, min(grid.width - 2, ax_sim + ddx * 4))
                        ty = max(1, min(grid.height - 2, ay_sim + ddy * 4))
                        npos = _greedy_step(ex, ey, tx, ty, occupied)
                        if npos == (ex, ey):
                            npos = _bfs_step(ex, ey, dist_map, occupied)

                elif beh == 3:  # Flanker
                    # Find nearest other enemy to agent
                    best_ref = None
                    best_rd = 99999
                    for j, (ox, oy) in enumerate(positions):
                        if j == i:
                            continue
                        rd = abs(ox - ax_sim) + abs(oy - ay_sim)
                        if rd < best_rd:
                            best_rd = rd
                            best_ref = (ox, oy)

                    if best_ref is None:
                        npos = _bfs_step(ex, ey, dist_map, occupied)
                    else:
                        rx, ry = best_ref
                        tx = max(1, min(grid.width - 2, ax_sim + (ax_sim - rx)))
                        ty = max(1, min(grid.height - 2, ay_sim + (ay_sim - ry)))
                        d_t = abs(ex - tx) + abs(ey - ty)
                        d_a = abs(ex - ax_sim) + abs(ey - ay_sim)
                        if d_t <= 1 or d_a <= 2:
                            npos = _bfs_step(ex, ey, dist_map, occupied)
                        else:
                            npos = _greedy_step(ex, ey, tx, ty, occupied)
                            if npos == (ex, ey):
                                npos = _bfs_step(ex, ey, dist_map, occupied)

                elif beh == 4:  # Trapper
                    # Find agent-adjacent cell that minimizes escapes
                    enemy_set = set(positions) - {(ex, ey)}
                    best_tgt, best_rem, best_td = None, 99, 99
                    for dx, dy in self._DIRS:
                        cx, cy = ax_sim + dx, ay_sim + dy
                        if not _walkable(cx, cy):
                            continue
                        if (cx, cy) in enemy_set or (cx, cy) in occupied:
                            continue
                        rem = 0
                        for dx2, dy2 in self._DIRS:
                            nx, ny = ax_sim + dx2, ay_sim + dy2
                            if (nx, ny) == (cx, cy):
                                continue
                            if (nx, ny) in enemy_set or (nx, ny) in occupied:
                                continue
                            if _walkable(nx, ny):
                                rem += 1
                        td = abs(ex - cx) + abs(ey - cy)
                        if rem < best_rem or (rem == best_rem and td < best_td):
                            best_rem = rem
                            best_tgt = (cx, cy)
                            best_td = td

                    if best_tgt:
                        npos = _greedy_step(ex, ey, best_tgt[0], best_tgt[1], occupied)
                        if npos == (ex, ey):
                            npos = _bfs_step(ex, ey, dist_map, occupied)
                    else:
                        npos = _bfs_step(ex, ey, dist_map, occupied)

                else:  # Chaser (beh == 1) or unknown
                    npos = _bfs_step(ex, ey, dist_map, occupied)

                occupied.add(npos)
                new_positions.append(npos)

            return new_positions

        n_enemies = len(enemy_positions)
        depth = 5 if n_enemies <= 4 else 4

        # If enemies are frozen, use time to maximize distance from enemies
        if freeze_remaining > 0:
            switches = self.api.get_entities_of_type("switch")
            if switches:
                nearest_s = min(switches, key=lambda s: s.distance)
                if nearest_s.distance <= freeze_remaining:
                    sx, sy = nearest_s.position
                    safe = not any((sx, sy) == ep for ep in enemy_positions)
                    if safe:
                        self.action_queue = self.api.move_toward(sx, sy)
                        return

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
                escape_count = sum(
                    1 for ddx, ddy in self._DIRS if _walkable(nx + ddx, ny + ddy)
                )
                cx, cy = grid.width / 2, grid.height / 2
                center_bonus = -(abs(nx - cx) + abs(ny - cy))
                score = min_d * 200 + total_d * 10 + escape_count * 15 + center_bonus
                if score > best_score:
                    best_score = score
                    best_action = name_map[move_names[i]]
            self.action_queue = [best_action]
            return

        # Collect switch (freeze power-up) only if safe
        switches = self.api.get_entities_of_type("switch")
        if switches:
            nearest_s = min(switches, key=lambda s: s.distance)
            min_enemy_dist = min(
                abs(ax - ex) + abs(ay - ey) for ex, ey in enemy_positions
            )
            if nearest_s.distance <= 3 and min_enemy_dist > nearest_s.distance + 4:
                self.action_queue = self.api.move_toward(*nearest_s.position)
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

            if any((nx, ny) == ep for ep in enemy_positions):
                continue

            score = self._evaluate_deep(
                nx, ny,
                enemy_positions, enemy_behaviors, agent_dir,
                grid, _walkable, _simulate_enemies, depth,
            )

            if score > best_score:
                best_score = score
                best_action = name_map[move_names[i]]

        self.action_queue = [best_action]

    def _evaluate_deep(
        self, ax, ay,
        enemies, behaviors, agent_dir,
        grid, walkable_fn, sim_fn, depth,
    ):
        """Evaluate a position with minimax-style lookahead.

        At each level: simulate type-aware enemy movement, then pick
        the best agent move. Returns the leaf score of the best play.
        """
        if depth == 0 or not enemies:
            return self._score_position(ax, ay, enemies, grid, walkable_fn)

        next_enemies = sim_fn(enemies, behaviors, ax, ay, agent_dir)

        if any((ex, ey) == (ax, ay) for ex, ey in next_enemies):
            return -100000 * depth

        enemy_set = set(next_enemies)

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
                nx, ny,
                next_enemies, behaviors, agent_dir,
                grid, walkable_fn, sim_fn, depth - 1,
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

        # Count escape routes (walkable neighbors not blocked by enemies)
        enemy_set = set(enemies)
        escape_count = 0
        best_2step_min = 0
        for ddx, ddy in self._DIRS:
            fx, fy = ax + ddx, ay + ddy
            if walkable_fn(fx, fy) and (fx, fy) not in enemy_set:
                escape_count += 1
                md = min(abs(fx - ex) + abs(fy - ey) for ex, ey in enemies)
                best_2step_min = max(best_2step_min, md)

        # Strong center preference — penalize distance from center
        cx, cy = grid.width / 2, grid.height / 2
        center_dist = abs(ax - cx) + abs(ay - cy)

        # Wall proximity penalty: being near edges drastically reduces options
        wall_dist = min(ax - 1, grid.width - 2 - ax, ay - 1, grid.height - 2 - ay)
        wall_penalty = 0
        if wall_dist <= 1:
            wall_penalty = -1500
        elif wall_dist <= 2:
            wall_penalty = -600

        # Heavy penalty for dead ends / low escape routes
        trap_penalty = 0
        if escape_count <= 1:
            trap_penalty = -3000
        elif escape_count == 2:
            trap_penalty = -800

        # Encirclement penalty: enemies in 3-4 quadrants = being surrounded
        quadrants = set()
        for ex, ey in enemies:
            qx = 1 if ex >= ax else -1
            qy = 1 if ey >= ay else -1
            quadrants.add((qx, qy))
        encircle_penalty = -300 * max(0, len(quadrants) - 2)

        # Corridor penalty: if we're squeezed between enemies on same axis
        same_row = [ex for ex, ey in enemies if ey == ay]
        same_col = [ey for ex, ey in enemies if ex == ax]
        corridor_penalty = 0
        if len(same_row) >= 2:
            left = [ex for ex in same_row if ex < ax]
            right = [ex for ex in same_row if ex > ax]
            if left and right:
                corridor_penalty -= 500
        if len(same_col) >= 2:
            above = [ey for ey in same_col if ey < ay]
            below = [ey for ey in same_col if ey > ay]
            if above and below:
                corridor_penalty -= 500

        return (
            min_d * 300
            + total_d * 20
            + best_2step_min * 50
            + escape_count * 50
            - center_dist * 30
            + wall_penalty
            + trap_penalty
            + encircle_penalty
            + corridor_penalty
        )


@register_oracle("Herding-v0")
class HerdingOracle(OracleAgent):
    """Herd sheep into pen using wall-based flee-physics strategy.

    Key insight about flee physics (_DIRS = [(1,0),(-1,0),(0,1),(0,-1)]):
    - Sheep always flees to the FIRST direction that maximizes Manhattan
      distance from the agent. Due to direction ordering, RIGHT (+1,0) is
      strongly preferred when multiple directions tie.
    - To push a sheep in a specific direction, wall constraints are needed
      to block the preferred flee directions.

    Wall-push rules (agent positions that push sheep along a wall):
    - RIGHT wall -> push DOWN:  agent at (-1,0), (-1,-1), (-2,0) from sheep
    - RIGHT wall -> push UP:    agent at (-1,+1) from sheep
    - LEFT wall  -> push DOWN:  agent at (+1,0), (+1,-1), (+2,0) from sheep
    - LEFT wall  -> push UP:    agent at (+1,+1) from sheep
    - TOP wall   -> push RIGHT: agent to left of sheep (many positions)
    - TOP wall   -> push LEFT:  agent at (+1,0), (+1,+1), (+2,0) from sheep
    - BOTTOM wall-> push RIGHT: agent to left of sheep (many positions)
    - BOTTOM wall-> push LEFT:  agent at (+1,0), (+1,-1), (+2,0) from sheep

    Strategy:
    1. For each sheep, compute a multi-phase route to the pen using walls.
    2. Phase 1: push sheep to the nearest useful wall (horizontal push
       is easy - RIGHT is free, LEFT needs agent to right).
    3. Phase 2: push sheep along the wall toward the pen row/column.
    4. Phase 3: push sheep into the pen from the wall.
    """

    _FLEE_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    _MOVE_DELTAS = [(0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)]
    _MOVE_NAMES = ["move_up", "move_down", "move_left", "move_right", "noop"]

    def __init__(self, env):
        super().__init__(env)
        self._pen_cells: set[tuple[int, int]] = set()
        self._pen_cx = 0.0
        self._pen_cy = 0.0
        self._pen_min_x = 0
        self._pen_max_x = 0
        self._pen_min_y = 0
        self._pen_max_y = 0
        self._target_idx: int | None = None
        self._stuck_counter = 0
        self._no_progress_counter = 0
        self._best_pen_dist_seen = 999.0
        self._prev_sheep_pos: tuple[int, int] | None = None
        self._grid_w = 0
        self._grid_h = 0

    def reset(self, obs, info):
        config = self.api.task_config
        self._pen_cells = set(map(tuple, config.get("pen_cells", [])))
        if self._pen_cells:
            self._pen_cx = sum(p[0] for p in self._pen_cells) / len(
                self._pen_cells
            )
            self._pen_cy = sum(p[1] for p in self._pen_cells) / len(
                self._pen_cells
            )
            self._pen_min_x = min(p[0] for p in self._pen_cells)
            self._pen_max_x = max(p[0] for p in self._pen_cells)
            self._pen_min_y = min(p[1] for p in self._pen_cells)
            self._pen_max_y = max(p[1] for p in self._pen_cells)
        self._target_idx = None
        self._stuck_counter = 0
        self._no_progress_counter = 0
        self._best_pen_dist_seen = 999.0
        self._prev_sheep_pos = None
        grid = self.api.grid
        self._grid_w = grid.width
        self._grid_h = grid.height
        super().reset(obs, info)

    # -- Flee simulation ------------------------------------------------

    def _simulate_flee(self, sx, sy, ax, ay, grid, occupied=None):
        """Predict where sheep at (sx,sy) flees when agent is at (ax,ay).

        Mirrors HerdingTask's flee logic exactly.
        """
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
                if occupied and (nx, ny) in occupied:
                    continue
                best_d = d
                best = (nx, ny)
        return best

    def _walkable(self, x, y, grid):
        return (
            0 < x < grid.width - 1
            and 0 < y < grid.height - 1
            and int(grid.terrain[y, x]) == CellType.EMPTY
        )

    def _pen_dist(self, x, y):
        return abs(x - self._pen_cx) + abs(y - self._pen_cy)

    def _available_pen_cells(self, sheep_set):
        """Return pen cells not blocked by other sheep (captured or live)."""
        return self._pen_cells - sheep_set

    def _min_pen_dist(self, x, y, sheep_set=None):
        """Min Manhattan distance to an available pen cell.

        If sheep_set is provided, excludes pen cells blocked by other sheep
        (captured sheep physically occupy pen cells and block entry).
        """
        if sheep_set is not None:
            avail = self._available_pen_cells(sheep_set)
        else:
            avail = self._pen_cells
        if not avail:
            # All pen cells occupied — fall back to any pen cell
            avail = self._pen_cells
        if not avail:
            return 999
        return min(abs(x - px) + abs(y - py) for px, py in avail)

    # -- Target selection -----------------------------------------------

    def _pick_target(self, sheep, captured):
        """Pick which sheep to herd.

        Strategy: balance between pen proximity and agent proximity.
        Sheep already being herded (current target) gets preference
        to avoid switching costs.
        """
        out = [
            i
            for i, s in enumerate(sheep)
            if s not in self._pen_cells and s not in captured
        ]
        if not out:
            return None
        # Keep current target if still outside pen
        if self._target_idx is not None and self._target_idx in out:
            return self._target_idx

        sheep_set = set(sheep)
        ax, ay = self.api.agent_position

        # Score: weighted sum of pen distance and agent distance
        # Prefer sheep that are close to BOTH the agent AND the pen
        def score(i):
            sx, sy = sheep[i]
            pen_d = self._min_pen_dist(sx, sy, sheep_set)
            agent_d = abs(sx - ax) + abs(sy - ay)
            return pen_d * 1.5 + agent_d

        new_target = min(out, key=score)
        # Reset progress tracking when switching targets
        if new_target != self._target_idx:
            self._best_pen_dist_seen = 999.0
            self._no_progress_counter = 0
            self._stuck_counter = 0
        return new_target

    # -- Compute approach position for a given push ---------------------

    def _get_staging_positions(self, sheep_set):
        """Compute staging positions: cells adjacent to available pen cells
        from which a single push sends the sheep into the pen.

        For right-side pens, staging is one cell to the LEFT of pen entries.
        For left-side pens, staging is one cell to the RIGHT.
        The push direction (RIGHT or LEFT) is the easiest in flee physics.
        """
        avail_pen = self._available_pen_cells(sheep_set)
        if not avail_pen:
            avail_pen = self._pen_cells
        if not avail_pen:
            return []

        w = self._grid_w
        pen_on_right = self._pen_cx >= w / 2

        staging = []
        for px, py in avail_pen:
            if pen_on_right:
                stx = px - 1
            else:
                stx = px + 1
            # Staging must be outside the pen (otherwise sheep gets captured
            # there immediately, which is actually fine but not a "staging")
            if 0 < stx < w - 1 and (stx, py) not in self._pen_cells:
                staging.append(((stx, py), (px, py)))
        # Also add: if a pen cell is itself reachable by entering from the
        # easy-push side, it's effectively a staging position for itself
        # (the sheep just enters the pen directly)
        for px, py in avail_pen:
            staging.append(((px, py), (px, py)))
        return staging

    def _get_approach_pos(self, sx, sy, grid, sheep_set, captured, ax, ay):
        """Find the best position for agent to push sheep toward pen.

        Uses flee simulation: tries all positions at distance 1-2 from
        sheep and picks the one whose flee result best improves distance
        to AVAILABLE pen cells. Includes staging position awareness.
        """
        if (sx, sy) in self._pen_cells:
            return None

        # CRITICAL: include ALL other sheep (including captured) in occupied
        # set — the real game physics includes captured sheep in the
        # occupied check, blocking flee to those cells.
        occupied = set(sheep_set) - {(sx, sy)}
        avail_pen = self._available_pen_cells(sheep_set)
        if not avail_pen:
            avail_pen = self._pen_cells

        # Also compute staging positions for bonus scoring
        staging_set = {
            sp for sp, _ in self._get_staging_positions(sheep_set)
        }

        cur_min_pd = self._min_pen_dist(sx, sy, sheep_set)
        # Distance to nearest staging position
        cur_staging_d = (
            min(abs(sx - sp[0]) + abs(sy - sp[1]) for sp in staging_set)
            if staging_set
            else 999
        )

        candidates: list[tuple[float, tuple[int, int]]] = []

        for r in range(1, 3):
            for dx in range(-r, r + 1):
                dy_abs = r - abs(dx)
                for dy in ([-dy_abs, dy_abs] if dy_abs > 0 else [0]):
                    cax, cay = sx + dx, sy + dy
                    if not self._walkable(cax, cay, grid):
                        continue
                    if (cax, cay) in occupied:
                        continue
                    fx, fy = self._simulate_flee(
                        sx, sy, cax, cay, grid, occupied
                    )
                    if (fx, fy) == (sx, sy):
                        continue

                    # Distance to nearest AVAILABLE pen cell
                    new_min_pd = min(
                        abs(fx - px) + abs(fy - py) for px, py in avail_pen
                    )
                    improvement = cur_min_pd - new_min_pd

                    agent_dist = abs(cax - ax) + abs(cay - ay)
                    pen_bonus = (
                        50.0 if (fx, fy) in avail_pen else 0.0
                    )

                    # Bonus for reaching a staging position
                    staging_bonus = 0.0
                    if (fx, fy) in staging_set:
                        staging_bonus = 15.0

                    # Bonus for moving toward staging (even if not there yet)
                    if staging_set:
                        new_staging_d = min(
                            abs(fx - sp[0]) + abs(fy - sp[1])
                            for sp in staging_set
                        )
                        staging_improvement = cur_staging_d - new_staging_d
                        if staging_improvement > 0:
                            staging_bonus += staging_improvement * 5.0

                    # Wall bonus
                    wall_bonus = 0.0
                    if fx <= 1 or fx >= grid.width - 2:
                        wall_bonus += 2.0
                    if fy <= 1 or fy >= grid.height - 2:
                        wall_bonus += 2.0

                    total_score = (
                        improvement * 10.0
                        + pen_bonus
                        + staging_bonus
                        + wall_bonus
                        - agent_dist * 0.3
                    )
                    if total_score > 0:
                        candidates.append((total_score, (cax, cay)))

        if candidates:
            candidates.sort(key=lambda c: -c[0])
            return candidates[0][1]

        # Fallback: push sheep toward the nearest wall
        return self._get_wall_push_target(
            sx, sy, grid, occupied, ax, ay
        )

    def _get_wall_push_target(self, sx, sy, grid, occupied, ax, ay):
        """When no direct pen-improving push exists, push the sheep
        toward the nearest useful wall to set up a wall-constrained push.
        """
        w, h = grid.width, grid.height
        pen_right = self._pen_cx > w / 2
        pen_below = self._pen_cy > h / 2

        # Determine which wall to push toward first based on pen position
        # Priority: push to vertical wall matching pen side (always easy to
        # push RIGHT or LEFT), then use wall to push vertically.
        if pen_right:
            # Push sheep RIGHT toward right wall (x = w-2)
            # Agent should be to LEFT of sheep
            target_wall_x = w - 2
            if sx < target_wall_x:
                # Agent goes to (sx-1, sy) or (sx-2, sy) to push right
                for d in [1, 2]:
                    cax = sx - d
                    cay = sy
                    if self._walkable(cax, cay, grid) and (cax, cay) not in occupied:
                        fx, fy = self._simulate_flee(
                            sx, sy, cax, cay, grid, occupied
                        )
                        if fx > sx:  # moved right
                            return (cax, cay)
        else:
            # Push sheep LEFT toward left wall (x = 1)
            target_wall_x = 1
            if sx > target_wall_x:
                for d in [1, 2]:
                    cax = sx + d
                    cay = sy
                    if self._walkable(cax, cay, grid) and (cax, cay) not in occupied:
                        fx, fy = self._simulate_flee(
                            sx, sy, cax, cay, grid, occupied
                        )
                        if fx < sx:  # moved left
                            return (cax, cay)

        # Also try vertical wall push
        if pen_below:
            # Try to push DOWN
            for cax, cay in [(sx - 1, sy), (sx + 1, sy), (sx, sy - 1)]:
                if not self._walkable(cax, cay, grid):
                    continue
                if (cax, cay) in occupied:
                    continue
                fx, fy = self._simulate_flee(
                    sx, sy, cax, cay, grid, occupied
                )
                if fy > sy:  # moved down
                    return (cax, cay)
        else:
            # Try to push UP
            for cax, cay in [(sx - 1, sy + 1), (sx + 1, sy + 1)]:
                if not self._walkable(cax, cay, grid):
                    continue
                if (cax, cay) in occupied:
                    continue
                fx, fy = self._simulate_flee(
                    sx, sy, cax, cay, grid, occupied
                )
                if fy < sy:  # moved up
                    return (cax, cay)

        # Last resort: approach from opposite side of pen center
        dx_pen = self._pen_cx - sx
        dy_pen = self._pen_cy - sy
        if abs(dx_pen) >= abs(dy_pen):
            approach_x = sx - (2 if dx_pen > 0 else -2)
            approach_y = sy
        else:
            approach_x = sx
            approach_y = sy - (2 if dy_pen > 0 else -2)
        approach_x = max(1, min(w - 2, int(approach_x)))
        approach_y = max(1, min(h - 2, int(approach_y)))
        if self._walkable(approach_x, approach_y, grid):
            return (approach_x, approach_y)
        return None

    # -- Score a move by simulating flee of all nearby sheep -------------

    def _score_move(self, nax, nay, sheep, captured, grid):
        """Score agent position by simulating all nearby sheep flee.

        Heavily penalises pushing penned sheep out, rewards pushing into
        AVAILABLE pen cells, and scores by min-pen-distance improvement
        to available pen cells.
        """
        score = 0.0
        sheep_set = set(sheep)
        avail_pen = self._available_pen_cells(sheep_set)
        if not avail_pen:
            avail_pen = self._pen_cells

        for sx, sy in sheep:
            if (sx, sy) in captured:
                continue
            dist = abs(sx - nax) + abs(sy - nay)
            if dist > 2:
                continue
            occupied = sheep_set - {(sx, sy)}
            fx, fy = self._simulate_flee(sx, sy, nax, nay, grid, occupied)
            if (fx, fy) == (sx, sy):
                continue

            in_pen = (sx, sy) in self._pen_cells
            flee_in_avail = (fx, fy) in avail_pen

            if in_pen and (fx, fy) not in self._pen_cells:
                score -= 200  # very bad: pushing out of pen
            elif not in_pen and flee_in_avail:
                score += 80  # excellent: pushing into available pen cell
            elif not in_pen:
                old_mpd = self._min_pen_dist(sx, sy, sheep_set)
                new_mpd = min(
                    abs(fx - px) + abs(fy - py) for px, py in avail_pen
                )
                improvement = old_mpd - new_mpd
                score += improvement * 10
                # Bonus for pushing sheep onto a wall
                if fx <= 1 or fx >= grid.width - 2:
                    score += 2.0
                if fy <= 1 or fy >= grid.height - 2:
                    score += 2.0
        return score

    # -- Navigate to target avoiding sheep disruption -------------------

    def _navigate_to(self, ax, ay, target, sheep, captured, grid, tidx):
        """Navigate agent from (ax,ay) to target, avoiding disrupting
        other sheep. Uses progressively less strict avoidance.
        """
        # Protect penned sheep: avoid being within flee range (dist<=2)
        penned_adj: set[tuple[int, int]] = set()
        for i, s in enumerate(sheep):
            if s in self._pen_cells or s in captured:
                for ddx, ddy in self._FLEE_DIRS:
                    penned_adj.add((s[0] + ddx, s[1] + ddy))

        other_sheep = {
            s for i, s in enumerate(sheep) if i != tidx and s not in captured
        }

        for level in range(4):
            avoid = set(penned_adj)
            if level == 0:
                for s in other_sheep:
                    avoid.add(s)
                    for ddx, ddy in self._FLEE_DIRS:
                        avoid.add((s[0] + ddx, s[1] + ddy))
            elif level == 1:
                avoid |= other_sheep
            elif level == 2:
                pass  # just penned adjacency
            else:
                avoid = set()

            avoid.discard(target)
            avoid.discard((ax, ay))

            path = self.api.bfs_path_positions(
                (ax, ay),
                target,
                avoid=avoid if avoid else None,
            )
            if path and len(path) > 1:
                acts = self.api.positions_to_actions(path)
                if acts:
                    self.action_queue = [acts[0]]
                    return True

        self.action_queue = self.api.move_toward(target[0], target[1])
        return True

    # -- Main plan ------------------------------------------------------

    def plan(self):
        config = self.api.task_config
        grid = self.api.grid
        ax, ay = self.api.agent_position

        sheep_raw = config.get("_live_sheep", [])
        if not sheep_raw or not self._pen_cells:
            self.action_queue = [0]
            return

        sheep = [(int(s[0]), int(s[1])) for s in sheep_raw]
        captured = set(map(tuple, config.get("_captured_sheep", [])))
        out_count = sum(
            1
            for s in sheep
            if s not in self._pen_cells and s not in captured
        )
        if out_count == 0:
            self.action_queue = [0]
            return

        tidx = self._pick_target(sheep, captured)
        if tidx is None:
            self.action_queue = [0]
            return

        sx, sy = sheep[tidx]
        sheep_set = set(sheep)

        # Stuck detection: track both position changes AND progress
        cur_pd = self._min_pen_dist(sx, sy, sheep_set)
        if self._prev_sheep_pos is not None:
            if (sx, sy) == self._prev_sheep_pos:
                self._stuck_counter += 1
            else:
                self._stuck_counter = 0

        # Track progress toward pen (even if sheep position changes)
        if cur_pd < self._best_pen_dist_seen - 0.1:
            self._best_pen_dist_seen = cur_pd
            self._no_progress_counter = 0
        else:
            self._no_progress_counter += 1
        self._prev_sheep_pos = (sx, sy)

        # Force-switch target if no progress for too long
        if self._no_progress_counter > 60 or self._stuck_counter > 25:
            out_idx = [
                i
                for i, s in enumerate(sheep)
                if s not in self._pen_cells
                and s not in captured
                and i != tidx
            ]
            if out_idx:
                tidx = min(
                    out_idx, key=lambda i: self._pen_dist(*sheep[i])
                )
                self._stuck_counter = 0
                self._no_progress_counter = 0
                self._best_pen_dist_seen = 999.0
                self._prev_sheep_pos = None
                sx, sy = sheep[tidx]
                sheep_set = set(sheep)

        self._target_idx = tidx

        # When stuck for a while, try to reposition the sheep
        if self._stuck_counter > 12 or self._no_progress_counter > 25:
            self._do_reposition(ax, ay, sheep, captured, grid, sx, sy, tidx)
            return

        # Near the sheep: use greedy move evaluation
        dist_s = abs(ax - sx) + abs(ay - sy)
        if dist_s <= 3:
            self._do_greedy_move(ax, ay, sheep, captured, grid, sx, sy, tidx)
            return

        # Far from sheep: navigate to approach position
        approach = self._get_approach_pos(
            sx, sy, grid, sheep_set, captured, ax, ay
        )
        if approach is None:
            self.action_queue = self.api.move_toward(sx, sy)
            return

        self._navigate_to(ax, ay, approach, sheep, captured, grid, tidx)

    def _do_reposition(self, ax, ay, sheep, captured, grid, sx, sy, tidx):
        """When stuck, push the sheep to a new position and approach from
        a different angle.

        Two strategies tried in order:
        1. Find a push that moves sheep closer to an available pen cell
           (even if the approach requires going around the sheep first)
        2. Push sheep toward center of grid for more maneuverability
        """
        sheep_set = set(sheep)
        occupied = sheep_set - {(sx, sy)}
        avail_pen = self._available_pen_cells(sheep_set)
        if not avail_pen:
            avail_pen = self._pen_cells

        # Strategy 1: Find ANY push that improves pen distance (or at least
        # any push at all), considering positions we haven't tried yet.
        # We use a 2-step lookahead: push sheep, then check if from the
        # new position a further push toward pen is possible.
        best_target = None
        best_score = -9999.0
        cur_pd = self._min_pen_dist(sx, sy, sheep_set)

        for r in range(1, 3):
            for dx in range(-r, r + 1):
                dy_abs = r - abs(dx)
                for dy in ([-dy_abs, dy_abs] if dy_abs > 0 else [0]):
                    cax, cay = sx + dx, sy + dy
                    if not self._walkable(cax, cay, grid):
                        continue
                    if (cax, cay) in occupied:
                        continue
                    fx, fy = self._simulate_flee(
                        sx, sy, cax, cay, grid, occupied
                    )
                    if (fx, fy) == (sx, sy):
                        continue

                    new_pd = min(
                        abs(fx - px) + abs(fy - py) for px, py in avail_pen
                    )
                    pd_improvement = cur_pd - new_pd

                    # 2-step lookahead: after this push, is there a follow-up
                    # push that further improves pen distance?
                    followup_bonus = 0.0
                    new_occ = (occupied | {(fx, fy)}) - {(sx, sy)}
                    for r2 in range(1, 3):
                        for dx2 in range(-r2, r2 + 1):
                            dy2_abs = r2 - abs(dx2)
                            for dy2 in (
                                [-dy2_abs, dy2_abs]
                                if dy2_abs > 0
                                else [0]
                            ):
                                c2x, c2y = fx + dx2, fy + dy2
                                if not self._walkable(c2x, c2y, grid):
                                    continue
                                if (c2x, c2y) in new_occ:
                                    continue
                                f2x, f2y = self._simulate_flee(
                                    fx, fy, c2x, c2y, grid, new_occ
                                )
                                if (f2x, f2y) == (fx, fy):
                                    continue
                                f2_pd = min(
                                    abs(f2x - px) + abs(f2y - py)
                                    for px, py in avail_pen
                                )
                                if f2_pd < new_pd:
                                    bonus = (new_pd - f2_pd) * 5.0
                                    followup_bonus = max(followup_bonus, bonus)

                    agent_dist = abs(cax - ax) + abs(cay - ay)
                    score = (
                        pd_improvement * 10.0
                        + followup_bonus
                        - agent_dist * 0.3
                    )
                    # Accept ANY move (even negative improvement) if it has
                    # good followup potential
                    if score > best_score:
                        best_score = score
                        best_target = (cax, cay)

        if best_target is not None:
            avoid_sheep = {s for s in sheep if s not in captured}
            avoid_sheep.discard((sx, sy))
            avoid_sheep.discard(best_target)

            if (ax, ay) == best_target:
                self._do_greedy_move(
                    ax, ay, sheep, captured, grid, sx, sy, tidx
                )
                return

            path = self.api.bfs_path_positions(
                (ax, ay),
                best_target,
                avoid=avoid_sheep if avoid_sheep else None,
            )
            if path and len(path) > 1:
                acts = self.api.positions_to_actions(path)
                if acts:
                    self.action_queue = [acts[0]]
                    return

        # Fallback to greedy
        self._do_greedy_move(ax, ay, sheep, captured, grid, sx, sy, tidx)

    def _do_greedy_move(self, ax, ay, sheep, captured, grid, sx, sy, tidx):
        """Pick the best single move using flee simulation scoring.

        When stuck, tries to navigate to a flanking position.
        """
        name_map = self.api.action_name_to_int
        sheep_set = set(sheep)

        approach = self._get_approach_pos(
            sx, sy, grid, sheep_set, captured, ax, ay
        )

        best_act = 0
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
                # Don't step onto another sheep
                if any(
                    (nx, ny) == s
                    for s in sheep
                    if s != (sx, sy) and s not in captured
                ):
                    continue

            score = self._score_move(nx, ny, sheep, captured, grid)

            # Approach bonus
            if approach is not None:
                cur_d = abs(ax - approach[0]) + abs(ay - approach[1])
                new_d = abs(nx - approach[0]) + abs(ny - approach[1])
                score += (cur_d - new_d) * 3.0

            if is_noop:
                score -= 0.3

            act = name_map[self._MOVE_NAMES[i]]
            if score > best_score:
                best_score = score
                best_act = act

        # If stuck, try flanking to get to a better approach angle
        if best_score <= 0.5 and self._stuck_counter > 5:
            if approach is not None and (ax, ay) != approach:
                # Navigate to the approach position going around the sheep
                avoid_sheep = {
                    s for s in sheep if s not in captured
                }
                path = self.api.bfs_path_positions(
                    (ax, ay),
                    approach,
                    avoid=avoid_sheep,
                )
                if path and len(path) > 1:
                    acts = self.api.positions_to_actions(path)
                    if acts:
                        self.action_queue = [acts[0]]
                        return

            # Fallback flank: go to opposite side of sheep from pen
            dx_pen = self._pen_cx - sx
            dy_pen = self._pen_cy - sy
            if abs(dx_pen) >= abs(dy_pen):
                flank_x = sx - (2 if dx_pen > 0 else -2)
                flank_y = sy + (1 if self._stuck_counter % 2 == 0 else -1)
            else:
                flank_x = sx + (1 if self._stuck_counter % 2 == 0 else -1)
                flank_y = sy - (2 if dy_pen > 0 else -2)
            flank_x = max(1, min(grid.width - 2, int(flank_x)))
            flank_y = max(1, min(grid.height - 2, int(flank_y)))

            if self._walkable(flank_x, flank_y, grid):
                avoid_sheep = {s for s in sheep if s not in captured}
                path = self.api.bfs_path_positions(
                    (ax, ay),
                    (flank_x, flank_y),
                    avoid=avoid_sheep,
                )
                if path and len(path) > 1:
                    acts = self.api.positions_to_actions(path)
                    if acts:
                        self.action_queue = [acts[0]]
                        return

        self.action_queue = [best_act]


@register_oracle("EmergentStrategy-v0")
class EmergentStrategyOracle(OracleAgent):
    """Lure NPCs onto pressure plates to open barriers (plates lock permanently).

    Strategy for **follower** NPCs:
      1. Navigate to (plate_x, plate_y - 1) -- one cell north of the plate.
         The follower targets the cell the agent just left (stays 1 tile behind).
      2. Wait (noop) until the follower catches up to an adjacent cell.
      3. Walk south through the plate cell. The follower follows into the
         plate cell -> locks permanently, barrier stays open.
      4. Continue south through the now-open barrier.

    Strategy for **fearful** NPCs:
      Approach the NPC from the side opposite the plate so it flees
      toward / onto the plate. Once locked, barrier is permanently open.

    Strategy for **contrarian** NPCs:
      Move in the opposite direction of where the plate is relative to the
      NPC, so the contrarian walks toward the plate. Zone-gating ensures
      the NPC only reacts when the agent is in the same zone.

    Strategy for **mirror** NPCs:
      The mirror NPC tries to reach the mirrored position of the agent.
      Position the agent so that the mirror position is the plate, then wait.
    """

    def __init__(self, env):
        super().__init__(env)
        self._stall = 0
        self._min_zone = 0  # monotonically increasing zone tracker
        self._walking_south = False  # True while executing walk-through
        self._prev_pos = None

    def reset(self, obs, info):
        self._stall = 0
        self._min_zone = 0
        self._walking_south = False
        self._prev_pos = None
        super().reset(obs, info)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _npc_set(config):
        return {tuple(p) for p in config.get("_npc_positions", [])}

    @staticmethod
    def _locked_set(config):
        """Return set of indices of locked NPCs."""
        return {
            i for i, locked in enumerate(config.get("_npc_locked", []))
            if locked
        }

    def _zone_of(self, y, config):
        """Return the zone index a y-coordinate falls into.

        Being exactly ON a barrier row counts as the zone BELOW it
        (i.e., already past the barrier), which prevents the oracle
        from trying to reopen a barrier it has already crossed.
        """
        barrier_rows = config.get("barrier_rows", [])
        # On a barrier row -> already crossed it -> zone i+1
        for i, br in enumerate(barrier_rows):
            if y == br:
                return i + 1
        size = self.api.grid_size[1]
        zone_bounds = [1] + list(barrier_rows) + [size - 2]
        for i in range(len(zone_bounds) - 1):
            y_lo = zone_bounds[i] + (1 if i > 0 else 0)
            y_hi = zone_bounds[i + 1]
            if y_lo <= y <= y_hi:
                return i
        return 0

    def _step_toward(self, goal, avoid=None):
        """Return a single action int toward *goal*, or None."""
        ax, ay = self.api.agent_position
        path = self.api.bfs_path_positions((ax, ay), goal, avoid=avoid)
        if path and len(path) >= 2:
            acts = self.api.positions_to_actions(path[:2])
            if acts:
                return acts[0]
        acts = self.api.move_toward(*goal)
        return acts[0] if acts else None

    # ------------------------------------------------------------------
    # Main plan
    # ------------------------------------------------------------------

    def plan(self):  # noqa: C901
        ax, ay = self.api.agent_position
        config = self.api.task_config
        goal_e = self.api.get_nearest("goal")
        if not goal_e:
            return

        npc_positions = [tuple(p) for p in config.get("_npc_positions", [])]
        npc_types = config.get("npc_types", [])
        plates = config.get("plate_positions", [])
        plate_barrier_map = config.get(
            "plate_barrier_map", list(range(len(plates))),
        )
        barrier_open = config.get("_barrier_open", {})
        barrier_rows = config.get("barrier_rows", [])
        n_plates = config.get("n_plates", len(plates))
        npc_set = self._npc_set(config)
        locked = self._locked_set(config)
        goal_pos = goal_e.position

        # Stall detection
        pos_now = (ax, ay)
        if pos_now == self._prev_pos:
            self._stall += 1
        else:
            self._stall = 0
        self._prev_pos = pos_now

        # --- Walking-south mode: keep going south until past barrier ---
        if self._walking_south:
            # Check if we've passed the current target barrier row
            agent_zone = self._zone_of(ay, config)
            if agent_zone > self._min_zone:
                self._min_zone = agent_zone
                self._walking_south = False
                # Fall through to normal planning
            elif self._stall > 2:
                # Stuck on or near a plate — barrier hasn't opened yet.
                # Move laterally off the plate so the follower NPC can
                # step onto it. Try east first, then west.
                self._walking_south = False
                grid = self.api.grid
                for ddx in [1, -1]:
                    cx, cy = ax + ddx, ay
                    if (
                        grid.in_bounds((cx, cy))
                        and int(grid.terrain[cy, cx]) == int(CellType.EMPTY)
                        and (cx, cy) not in npc_set
                    ):
                        if ddx == 1:
                            act = self.api.action_name_to_int.get("move_right")
                        else:
                            act = self.api.action_name_to_int.get("move_left")
                        if act is not None:
                            self.action_queue = [act]
                            return
                # Can't move laterally either — noop and fall through
                self.action_queue = [0]
                return
            else:
                # Keep walking south
                down = self.api.action_name_to_int.get("move_down")
                if down is not None:
                    self.action_queue = [down]
                    return
                self._walking_south = False

        # --- Determine which plate/barrier to work on ---
        agent_zone = self._zone_of(ay, config)
        # Monotonically advance: once we've been in a higher zone, don't
        # regress (prevents oscillating back through barriers).
        if agent_zone > self._min_zone:
            self._min_zone = agent_zone
        agent_zone = self._min_zone

        # If a barrier ahead is already open (permanently), rush through
        for i in range(n_plates):
            bi = i
            if bi >= agent_zone and barrier_open.get(str(bi), False):
                # Barrier is open -- walk south through it
                down = self.api.action_name_to_int.get("move_down")
                if down is not None:
                    self._walking_south = True
                    self.action_queue = [down]
                    return
                break

        target_pi = None
        for pi in range(n_plates):
            bi = plate_barrier_map[pi]
            if bi >= agent_zone and not barrier_open.get(str(bi), False):
                target_pi = pi
                break

        # Past all barriers -> go to goal
        if target_pi is None:
            a = self._step_toward(goal_pos, avoid=npc_set - {goal_pos})
            if a is None:
                a = self._step_toward(goal_pos)
            if a is not None:
                self.action_queue = [a]
            return

        plate_pos = tuple(plates[target_pi])
        px, py = plate_pos
        bi = plate_barrier_map[target_pi]
        barrier_row = barrier_rows[bi] if bi < len(barrier_rows) else py + 1

        # --- Pick NPC to use (prefer follower in same zone, skip locked) ---
        best_idx = None
        best_score = float("inf")
        for i, (nx, ny) in enumerate(npc_positions):
            if i >= len(npc_types):
                continue
            if i in locked:
                continue  # skip locked NPCs
            nz = self._zone_of(ny, config)
            if nz != agent_zone:
                continue
            ntype = npc_types[i]
            d = abs(nx - px) + abs(ny - py)
            ts = {
                "follower": 0, "fearful": 1, "contrarian": 2, "mirror": 3,
            }.get(ntype, 4)
            s = ts * 10000 + d
            if s < best_score:
                best_score = s
                best_idx = i
        if best_idx is None:
            # Fallback: any unlocked NPC
            for i, (nx, ny) in enumerate(npc_positions):
                if i >= len(npc_types):
                    continue
                if i in locked:
                    continue
                d = abs(nx - px) + abs(ny - py)
                if d < best_score:
                    best_score = d
                    best_idx = i
        if best_idx is None:
            # All NPCs locked or none available -- try to go to goal anyway
            a = self._step_toward(goal_pos, avoid=npc_set - {goal_pos})
            if a is None:
                a = self._step_toward(goal_pos)
            if a is not None:
                self.action_queue = [a]
            else:
                self.action_queue = [0]
            return

        npc_pos = npc_positions[best_idx]
        npc_type = npc_types[best_idx]

        # NPC already on plate -> it will lock, rush through
        if npc_pos == plate_pos:
            a = self._step_toward(goal_pos, avoid=npc_set - {goal_pos})
            if a is None:
                a = self._step_toward(goal_pos)
            if a is not None:
                self.action_queue = [a]
            return

        # Dispatch per NPC type
        if npc_type == "follower":
            self._plan_follower(
                ax, ay, npc_pos, plate_pos, barrier_row, goal_pos, npc_set,
            )
        elif npc_type == "fearful":
            self._plan_fearful(
                ax, ay, npc_pos, plate_pos, barrier_row, goal_pos, npc_set,
            )
        elif npc_type == "contrarian":
            self._plan_contrarian(
                ax, ay, npc_pos, plate_pos, barrier_row, goal_pos, npc_set,
            )
        elif npc_type == "mirror":
            self._plan_mirror(
                ax, ay, npc_pos, plate_pos, barrier_row, goal_pos, npc_set,
                config,
            )
        else:
            self.action_queue = [0]

    # ------------------------------------------------------------------
    # Follower strategy
    # ------------------------------------------------------------------

    def _plan_follower(self, ax, ay, npc_pos, plate_pos, barrier_row,
                       goal_pos, npc_set):
        """Walk-through strategy for follower NPCs.

        The follower BFS-targets the cell the agent just left (stays 1
        tile behind). So when the agent walks south through the plate,
        the follower steps onto the plate cell and locks there.

        Primary approach (plate not on top row):
          1. Navigate to (px, py-1) -- one cell north of the plate.
          2. Wait until the follower is adjacent.
          3. Walk south through the plate and barrier.

        Fallback (plate on top row, i.e. py == 1):
          1. Navigate to (px+1, py) -- one cell east of the plate.
          2. Wait until the follower is adjacent.
          3. Walk west through the plate.
          4. Then walk south through the barrier.
        """
        px, py = plate_pos
        nx, ny = npc_pos
        grid = self.api.grid

        # Determine staging position and walk direction
        staging_north = (px, py - 1)
        use_south = True  # walk south through plate

        if (
            not grid.in_bounds(staging_north)
            or int(grid.terrain[staging_north[1], staging_north[0]])
            != int(CellType.EMPTY)
        ):
            # Can't stage north of plate (wall). Use east-side approach.
            staging_east = (px + 1, py)
            staging_west = (px - 1, py)
            if (
                grid.in_bounds(staging_east)
                and int(grid.terrain[staging_east[1], staging_east[0]])
                == int(CellType.EMPTY)
            ):
                staging = staging_east
                use_south = False
            elif (
                grid.in_bounds(staging_west)
                and int(grid.terrain[staging_west[1], staging_west[0]])
                == int(CellType.EMPTY)
            ):
                staging = staging_west
                use_south = False
            else:
                # No good staging spot, just try plate itself
                staging = plate_pos
                use_south = True
        else:
            staging = staging_north

        dist_npc_to_agent = abs(nx - ax) + abs(ny - ay)

        # ---- Phase 1: If NPC is too far, go fetch it ----
        if dist_npc_to_agent > 4:
            a = self._step_toward(npc_pos, avoid=npc_set - {npc_pos})
            if a is not None:
                self.action_queue = [a]
                return
            self.action_queue = [0]
            return

        # ---- Phase 2: NPC is in following range. Go to staging. ----
        if (ax, ay) != staging:
            a = self._step_toward(staging, avoid=npc_set - {staging})
            if a is not None:
                self.action_queue = [a]
                return
            a = self._step_toward(staging)
            if a is not None:
                self.action_queue = [a]
                return
            self.action_queue = [0]
            return

        # ---- Phase 3: At staging -- wait for NPC to be adjacent ----
        if dist_npc_to_agent > 1:
            self.action_queue = [0]
            return

        # ---- Phase 4: NPC is adjacent -- walk through plate ----
        if use_south:
            action = self.api.action_name_to_int.get("move_down")
        else:
            # Walking east-to-west through plate (staging was east)
            if staging[0] > px:
                action = self.api.action_name_to_int.get("move_left")
            else:
                action = self.api.action_name_to_int.get("move_right")

        if action is not None:
            self._walking_south = True
            self.action_queue = [action]
            return

        self.action_queue = [0]

    # ------------------------------------------------------------------
    # Fearful strategy
    # ------------------------------------------------------------------

    def _plan_fearful(self, ax, ay, npc_pos, plate_pos, barrier_row,
                      goal_pos, npc_set):
        """Scare the fearful NPC onto the plate.

        The fearful NPC flees from the agent when within distance 3,
        moving to maximize Manhattan distance. We position the agent
        on the opposite side of the NPC from the plate so the flee
        direction is toward the plate. Once on the plate, it locks.
        """
        px, py = plate_pos
        nx, ny = npc_pos
        grid = self.api.grid

        # Desired agent position: opposite side of NPC from plate.
        # Vector from NPC to plate
        d_n2p_x = px - nx
        d_n2p_y = py - ny

        # Approach from opposite side (behind the NPC relative to plate)
        candidates = []
        for dist in [1, 2]:
            sign_x = (-1 if d_n2p_x > 0 else (1 if d_n2p_x < 0 else 0))
            sign_y = (-1 if d_n2p_y > 0 else (1 if d_n2p_y < 0 else 0))
            cx, cy = nx + sign_x * dist, ny + sign_y * dist
            if (
                grid.in_bounds((cx, cy))
                and int(grid.terrain[cy, cx]) == int(CellType.EMPTY)
                and (cx, cy) not in npc_set
            ):
                candidates.append((cx, cy))

        # Also try all adjacencies that would push NPC toward plate
        for ddx, ddy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            cx, cy = nx + ddx, ny + ddy
            if (
                grid.in_bounds((cx, cy))
                and int(grid.terrain[cy, cx]) == int(CellType.EMPTY)
                and (cx, cy) not in npc_set
            ):
                # Agent at (cx,cy) => NPC flees in direction (nx-cx, ny-cy)
                flee_dx, flee_dy = nx - cx, ny - cy
                # Does flee roughly point toward plate?
                if (flee_dx * d_n2p_x + flee_dy * d_n2p_y) > 0:
                    candidates.append((cx, cy))

        # Pick closest reachable
        best = None
        best_len = float("inf")
        for c in candidates:
            p = self.api.bfs_path_positions(
                (ax, ay), c, avoid=npc_set - {c},
            )
            if p and len(p) < best_len:
                best_len = len(p)
                best = (c, p)

        if best:
            target, path = best
            if (ax, ay) == target:
                # In position -- just wait; NPC will flee
                self.action_queue = [0]
                return
            acts = self.api.positions_to_actions(path[:2])
            if acts:
                self.action_queue = [acts[0]]
                return

        # Fallback: move toward NPC
        a = self._step_toward(npc_pos, avoid=npc_set - {npc_pos})
        if a is not None:
            self.action_queue = [a]
        else:
            self.action_queue = [0]

    # ------------------------------------------------------------------
    # Contrarian strategy
    # ------------------------------------------------------------------

    def _plan_contrarian(self, ax, ay, npc_pos, plate_pos, barrier_row,
                         goal_pos, npc_set):
        """Guide contrarian NPC onto the plate by moving in the opposite direction.

        The contrarian moves opposite to the agent's last action when in the
        same zone. Zone-gating prevents drift while the agent is elsewhere.

        Anti-oscillation strategy: when the agent can't move on the primary
        axis, use the PERPENDICULAR axis instead (never reverse). This creates
        a zigzag that makes net progress on both axes without canceling work.
        """
        px, py = plate_pos
        nx, ny = npc_pos
        grid = self.api.grid

        # Vector from NPC to plate
        d_n2p_x = px - nx
        d_n2p_y = py - ny

        def _agent_can_move(action_name: str) -> bool:
            deltas = {
                "move_left": (-1, 0), "move_right": (1, 0),
                "move_up": (0, -1), "move_down": (0, 1),
            }
            ddx, ddy = deltas.get(action_name, (0, 0))
            tx, ty = ax + ddx, ay + ddy
            if not grid.in_bounds((tx, ty)):
                return False
            if int(grid.terrain[ty, tx]) != int(CellType.EMPTY):
                return False
            if (tx, ty) in npc_set:
                return False
            return True

        def _can_contrarian_move(direction: tuple[int, int]) -> bool:
            cx, cy = nx + direction[0], ny + direction[1]
            if not grid.in_bounds((cx, cy)):
                return False
            if int(grid.terrain[cy, cx]) != int(CellType.EMPTY):
                return False
            if (cx, cy) in npc_set or (cx, cy) == (ax, ay):
                return False
            return True

        # Build the set of helpful commands (push contrarian toward plate)
        # NEVER issue a command that pushes contrarian AWAY from plate.
        helpful: list[tuple[str, int]] = []  # (action_name, priority)

        if d_n2p_x > 0 and _can_contrarian_move((1, 0)):
            # NPC needs right → agent moves left
            helpful.append(("move_left", abs(d_n2p_x)))
        elif d_n2p_x < 0 and _can_contrarian_move((-1, 0)):
            # NPC needs left → agent moves right
            helpful.append(("move_right", abs(d_n2p_x)))

        if d_n2p_y > 0 and _can_contrarian_move((0, 1)):
            # NPC needs down → agent moves up
            helpful.append(("move_up", abs(d_n2p_y)))
        elif d_n2p_y < 0 and _can_contrarian_move((0, -1)):
            # NPC needs up → agent moves down
            helpful.append(("move_down", abs(d_n2p_y)))

        # Sort by distance remaining on that axis (largest first = more urgent)
        helpful.sort(key=lambda h: -h[1])

        # Try each helpful command; pick the first one the agent can execute
        for action_name, _ in helpful:
            if _agent_can_move(action_name):
                agent_action = self.api.action_name_to_int.get(action_name)
                if agent_action is not None:
                    self.action_queue = [agent_action]
                    return

        # Agent can't execute any helpful command. Reposition using a
        # PERPENDICULAR move (which is also helpful, or at least neutral).
        # Build set of safe repositioning moves that DON'T push contrarian
        # away from the plate on any axis.
        safe_actions = []
        for action_name in ["move_up", "move_down", "move_left", "move_right"]:
            if not _agent_can_move(action_name):
                continue
            # What direction would the contrarian move? Opposite of agent.
            opp = {
                "move_up": (0, 1), "move_down": (0, -1),
                "move_left": (1, 0), "move_right": (-1, 0),
            }
            cdx, cdy = opp[action_name]
            # Would this push contrarian further from plate on either axis?
            new_dx = d_n2p_x - cdx  # remaining x distance after push
            new_dy = d_n2p_y - cdy  # remaining y distance after push
            # Accept if we don't increase distance on the primary axis
            x_worse = abs(new_dx) > abs(d_n2p_x)
            y_worse = abs(new_dy) > abs(d_n2p_y)
            if not x_worse and not y_worse:
                safe_actions.append(action_name)
            elif not x_worse or not y_worse:
                # At least one axis improves or stays same
                safe_actions.append(action_name)

        for action_name in safe_actions:
            agent_action = self.api.action_name_to_int.get(action_name)
            if agent_action is not None:
                self.action_queue = [agent_action]
                return

        # Last resort: move toward NPC without canceling
        a = self._step_toward(npc_pos, avoid=npc_set - {npc_pos})
        if a is not None:
            self.action_queue = [a]
        else:
            self.action_queue = [0]

    # ------------------------------------------------------------------
    # Mirror strategy
    # ------------------------------------------------------------------

    def _plan_mirror(self, ax, ay, npc_pos, plate_pos, barrier_row,
                     goal_pos, npc_set, config):
        """Position agent so the mirror NPC's target is the plate.

        The mirror NPC BFS-targets mirror_pos = (W-1 - ax, H-1 - ay).
        We want mirror_pos == plate_pos, so the agent should be at:
          agent_x = (W-1) - plate_x
          agent_y = (H-1) - plate_y
        Then wait for the mirror NPC to walk onto the plate.
        """
        px, py = plate_pos
        gw = config.get("grid_width", self.api.grid_size[0])
        gh = config.get("grid_height", self.api.grid_size[1])
        grid = self.api.grid

        # Desired agent position to make mirror target = plate
        target_ax = (gw - 1) - px
        target_ay = (gh - 1) - py
        # Clamp to interior
        target_ax = max(1, min(gw - 2, target_ax))
        target_ay = max(1, min(gh - 2, target_ay))
        target_agent = (target_ax, target_ay)

        # Check if target agent position is walkable
        if (
            grid.in_bounds(target_agent)
            and int(grid.terrain[target_ay, target_ax]) == int(CellType.EMPTY)
            and target_agent not in npc_set
        ):
            if (ax, ay) == target_agent:
                # In position -- wait for mirror NPC to arrive
                self.action_queue = [0]
                return
            a = self._step_toward(target_agent, avoid=npc_set - {target_agent})
            if a is not None:
                self.action_queue = [a]
                return
            a = self._step_toward(target_agent)
            if a is not None:
                self.action_queue = [a]
                return

        # Fallback: try nearby positions that put mirror close to plate
        best = None
        best_d = float("inf")
        for ddx in range(-2, 3):
            for ddy in range(-2, 3):
                cx, cy = target_ax + ddx, target_ay + ddy
                if not grid.in_bounds((cx, cy)):
                    continue
                if int(grid.terrain[cy, cx]) != int(CellType.EMPTY):
                    continue
                if (cx, cy) in npc_set:
                    continue
                # Mirror pos for this agent pos
                mx = (gw - 1) - cx
                my = (gh - 1) - cy
                d = abs(mx - px) + abs(my - py)
                if d < best_d:
                    best_d = d
                    best = (cx, cy)

        if best:
            if (ax, ay) == best:
                self.action_queue = [0]
                return
            a = self._step_toward(best, avoid=npc_set - {best})
            if a is not None:
                self.action_queue = [a]
                return

        # Last resort: just wait
        self.action_queue = [0]
