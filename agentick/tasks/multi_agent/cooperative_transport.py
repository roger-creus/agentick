"""CooperativeTransport - Push heavy boxes into holes with NPC cooperation.

MECHANICS:
  - Multiple heavy BOX objects that CANNOT be pushed by the agent alone.
  - Destinations are HOLE terrain (CellType.HOLE) - push boxes INTO holes.
  - Push mechanic: Agent pushes box only when NPC is ALSO adjacent to the
    same box. Agent pushes toward the box; NPC must be on any adjacent side.
  - NPC behavior: wanders randomly. When agent is adjacent to a box, NPC
    pathfinds to the opposite side of that box.
  - Success = all boxes pushed into holes (box on HOLE -> both disappear).
"""

from collections import deque

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("CooperativeTransport-v0", tags=["multi_agent", "cooperation"])
class CooperativeTransportTask(TaskSpec):
    """Push heavy boxes into holes with NPC cooperation."""

    name = "CooperativeTransport-v0"
    description = "Push heavy boxes into holes with NPC cooperation"
    capability_tags = ["multi_agent", "cooperation"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=9,
            max_steps=100,
            params={"n_boxes": 1, "n_holes": 1, "n_obstacles": 0},
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=12,
            max_steps=200,
            params={"n_boxes": 2, "n_holes": 2, "n_obstacles": 3},
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=15,
            max_steps=350,
            params={"n_boxes": 3, "n_holes": 3, "n_obstacles": 6},
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=18,
            max_steps=500,
            params={"n_boxes": 4, "n_holes": 4, "n_obstacles": 9},
        ),
    }

    _DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        params = self.difficulty_config.params
        n_boxes = params.get("n_boxes", 1)
        n_holes = params.get("n_holes", 1)
        n_obs = params.get("n_obstacles", 0)

        for _attempt in range(50):
            grid = Grid(size, size)
            # Border walls
            grid.terrain[0, :] = CellType.WALL
            grid.terrain[-1, :] = CellType.WALL
            grid.terrain[:, 0] = CellType.WALL
            grid.terrain[:, -1] = CellType.WALL

            # Inner positions: boxes and holes need 2+ cells from wall so they
            # can be pushed from at least one side.
            inner = [
                (x, y)
                for x in range(2, size - 2)
                for y in range(2, size - 2)
            ]
            rng.shuffle(inner)
            # Outer ring (1 cell from wall) for NPC placement
            interior = [
                (x, y)
                for x in range(1, size - 1)
                for y in range(1, size - 1)
            ]

            # Place agent near center
            mid = size // 2
            agent_pos = (mid, mid)

            # Allocate positions for boxes and holes from inner area
            reserved = {agent_pos}
            box_hole_placed = []
            for pos in inner:
                if pos in reserved:
                    continue
                box_hole_placed.append(pos)
                reserved.add(pos)
                if len(box_hole_placed) >= n_boxes + n_holes:
                    break

            if len(box_hole_placed) < n_boxes + n_holes:
                continue

            # NPC from full interior
            npc_placed = None
            rng.shuffle(interior)
            for pos in interior:
                if pos not in reserved:
                    npc_placed = pos
                    reserved.add(pos)
                    break

            if npc_placed is None:
                continue

            placed = box_hole_placed + [npc_placed]

            box_positions = [list(placed[i]) for i in range(n_boxes)]
            hole_positions = [list(placed[n_boxes + i]) for i in range(n_holes)]
            npc_pos = list(placed[n_boxes + n_holes])

            # Place holes as HOLE terrain
            for hx, hy in hole_positions:
                grid.terrain[hy, hx] = CellType.HOLE

            # Place boxes as BOX objects
            for bx, by in box_positions:
                grid.objects[by, bx] = ObjectType.BOX

            # Place obstacles (walls), verify reachability after each
            all_key = set()
            all_key.add(agent_pos)
            all_key.add(tuple(npc_pos))
            for bx, by in box_positions:
                all_key.add((bx, by))
            # For reachability we need to check that agent can reach
            # all boxes and that boxes can (in principle) reach holes.
            # We use a custom flood fill that treats HOLE as impassable
            # for walking but we still need boxes adjacent to holes.

            obs_candidates = [
                (x, y)
                for x in range(1, size - 1)
                for y in range(1, size - 1)
                if (x, y) not in reserved
            ]
            rng.shuffle(obs_candidates)
            placed_obs = 0
            for ox, oy in obs_candidates:
                if placed_obs >= n_obs:
                    break
                grid.terrain[oy, ox] = CellType.WALL
                # Check agent can reach all boxes and NPC position
                reachable = self._flood_fill_walk(grid, agent_pos)
                ok = True
                for bx, by in box_positions:
                    if (bx, by) not in reachable:
                        ok = False
                        break
                if ok and tuple(npc_pos) not in reachable:
                    ok = False
                # Also check each hole has at least one walkable neighbor
                # (so a box could potentially be pushed into it)
                if ok:
                    for hx, hy in hole_positions:
                        has_neighbor = False
                        for dx, dy in self._DIRS:
                            nx, ny = hx + dx, hy + dy
                            if (nx, ny) in reachable:
                                has_neighbor = True
                                break
                        if not has_neighbor:
                            ok = False
                            break
                if not ok:
                    grid.terrain[oy, ox] = CellType.EMPTY
                else:
                    placed_obs += 1

            # Final reachability check
            reachable = self._flood_fill_walk(grid, agent_pos)
            ok = True
            for bx, by in box_positions:
                if (bx, by) not in reachable:
                    ok = False
                    break
            if ok and tuple(npc_pos) not in reachable:
                ok = False
            if ok:
                for hx, hy in hole_positions:
                    has_neighbor = False
                    for dx, dy in self._DIRS:
                        nx, ny = hx + dx, hy + dy
                        if (nx, ny) in reachable:
                            has_neighbor = True
                            break
                    if not has_neighbor:
                        ok = False
                        break
            if not ok:
                continue

            return grid, {
                "agent_start": agent_pos,
                "goal_positions": [],
                "npc_start": npc_pos,
                "box_positions": box_positions,
                "hole_positions": hole_positions,
                "n_boxes": n_boxes,
                "max_steps": self.get_max_steps(),
            }

        # Fallback: minimal valid layout
        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL
        mid = size // 2
        agent_pos = (mid, mid)
        box_positions = [[mid, mid - 2]]
        hole_positions = [[mid, 1]]
        npc_pos = [mid + 1, mid]
        grid.objects[mid - 2, mid] = ObjectType.BOX
        grid.terrain[1, mid] = CellType.HOLE
        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [],
            "npc_start": npc_pos,
            "box_positions": box_positions,
            "hole_positions": hole_positions,
            "n_boxes": 1,
            "max_steps": self.get_max_steps(),
        }

    @staticmethod
    def _flood_fill_walk(grid, start):
        """Flood fill that treats HOLE and WALL as impassable (walking only)."""
        if not grid.in_bounds(start):
            return set()
        sx, sy = start
        if grid.terrain[sy, sx] in (CellType.WALL, CellType.HOLE):
            return set()
        visited = {start}
        queue = deque([start])
        while queue:
            cx, cy = queue.popleft()
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = cx + dx, cy + dy
                if (nx, ny) not in visited and grid.in_bounds((nx, ny)):
                    if grid.terrain[ny, nx] not in (CellType.WALL, CellType.HOLE):
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        return visited

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def on_env_reset(self, agent, grid, config):
        config["_fell_in_hole"] = False
        self._config = config
        self._npc_pos = list(config["npc_start"])
        self._box_positions = [list(b) for b in config["box_positions"]]
        self._hole_positions = [list(h) for h in config["hole_positions"]]
        self._boxes_delivered = 0
        self._n_boxes = config["n_boxes"]

        # Place boxes on grid
        for bx, by in self._box_positions:
            grid.objects[by, bx] = ObjectType.BOX

        # Place holes on grid (terrain)
        for hx, hy in self._hole_positions:
            grid.terrain[hy, hx] = CellType.HOLE

        # Place NPC
        nx, ny = self._npc_pos
        if grid.terrain[ny, nx] not in (CellType.WALL, CellType.HOLE):
            grid.objects[ny, nx] = ObjectType.NPC
            grid.metadata[ny, nx] = 2  # facing down

        # Store in config for state access
        config["_box_positions"] = self._box_positions
        config["_hole_positions"] = self._hole_positions
        config["_npc_pos"] = self._npc_pos
        config["_boxes_delivered"] = 0

    def on_agent_moved(self, pos, agent, grid):
        """Detect if agent fell into a hole."""
        x, y = pos
        if grid.terrain[y, x] == CellType.HOLE:
            self._config["_fell_in_hole"] = True

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def can_agent_enter(self, pos, agent, grid) -> bool:
        x, y = pos
        obj = int(grid.objects[y, x])

        # Allow walking through NPC
        if obj == int(ObjectType.NPC):
            return True

        # Handle BOX pushing
        if obj == int(ObjectType.BOX):
            return self._try_push_box(x, y, agent, grid)

        # Normal walkability (HOLE terrain is rejected by is_walkable in TaskEnv)
        return True

    def _try_push_box(self, bx, by, agent, grid) -> bool:
        """Attempt to push box at (bx, by). Requires NPC adjacent to same box."""
        ax, ay = agent.position
        nx, ny = self._npc_pos

        # Check NPC is adjacent to this box
        if abs(nx - bx) + abs(ny - by) != 1:
            return False  # NPC not adjacent -> can't push

        # Push direction: from agent toward box
        dx, dy = bx - ax, by - ay
        nbx, nby = bx + dx, by + dy

        # Check destination in bounds (interior)
        if not (0 < nbx < grid.width - 1 and 0 < nby < grid.height - 1):
            return False

        dest_terrain = int(grid.terrain[nby, nbx])
        dest_obj = int(grid.objects[nby, nbx])

        # Destination is a HOLE -> deliver box!
        if dest_terrain == int(CellType.HOLE):
            # Remove box from current position
            grid.objects[by, bx] = ObjectType.NONE
            # Clear the hole terrain and any object on it
            grid.terrain[nby, nbx] = CellType.EMPTY
            grid.objects[nby, nbx] = ObjectType.NONE
            # Update tracking
            self._boxes_delivered += 1
            self._config["_boxes_delivered"] = self._boxes_delivered
            # Remove from position lists
            if [bx, by] in self._box_positions:
                self._box_positions.remove([bx, by])
            if [nbx, nby] in self._hole_positions:
                self._hole_positions.remove([nbx, nby])
            self._config["_box_positions"] = self._box_positions
            self._config["_hole_positions"] = self._hole_positions
            return True

        # Destination must be empty floor with no blocking object
        if dest_terrain == int(CellType.WALL):
            return False
        # Allow pushing into NPC position (NPC dodges) or empty
        if dest_obj not in (int(ObjectType.NONE), int(ObjectType.NPC)):
            return False
        # Also reject if another box is at destination
        if self._is_box_at(nbx, nby):
            return False

        # If NPC is at destination, displace it to a free adjacent cell
        if dest_obj == int(ObjectType.NPC):
            self._displace_npc(grid, nbx, nby, agent.position)

        # Move box
        grid.objects[by, bx] = ObjectType.NONE
        grid.objects[nby, nbx] = ObjectType.BOX
        # Update tracking
        for i, bp in enumerate(self._box_positions):
            if bp == [bx, by]:
                self._box_positions[i] = [nbx, nby]
                break
        self._config["_box_positions"] = self._box_positions
        return True

    def _displace_npc(self, grid, nx, ny, agent_pos):
        """Move NPC from (nx, ny) to any free adjacent cell."""
        grid.objects[ny, nx] = ObjectType.NONE
        grid.metadata[ny, nx] = 0
        for ddx, ddy in self._DIRS:
            cx, cy = nx + ddx, ny + ddy
            if (
                0 < cx < grid.width - 1
                and 0 < cy < grid.height - 1
                and grid.terrain[cy, cx] not in (CellType.WALL, CellType.HOLE)
                and grid.objects[cy, cx] == ObjectType.NONE
                and (cx, cy) != agent_pos
                and not self._is_box_at(cx, cy)
            ):
                self._npc_pos = [cx, cy]
                self._config["_npc_pos"] = self._npc_pos
                grid.objects[cy, cx] = ObjectType.NPC
                grid.metadata[cy, cx] = 2
                return
        # If no free cell, just place NPC at same spot (will be overwritten by box)
        self._npc_pos = [nx, ny]
        self._config["_npc_pos"] = self._npc_pos

    # ------------------------------------------------------------------
    # NPC step
    # ------------------------------------------------------------------

    def on_env_step(self, agent, grid, config, step_count):
        ax, ay = agent.position
        nx, ny = self._npc_pos
        old_nx, old_ny = nx, ny

        # Clear old NPC position
        if grid.objects[ny, nx] == ObjectType.NPC:
            grid.objects[ny, nx] = ObjectType.NONE
            grid.metadata[ny, nx] = 0

        # Determine NPC target: if agent is adjacent to any box, pathfind
        # to an adjacent side of that box (ideal cooperative position).
        target_pos = None
        agent_adj_box = None
        for bx, by in self._box_positions:
            if abs(ax - bx) + abs(ay - by) == 1:
                agent_adj_box = (bx, by)
                break

        if agent_adj_box is not None:
            bx, by = agent_adj_box
            # Collect all valid adjacent-to-box positions (not the agent's cell)
            adj_candidates = []
            # Prefer opposite side from agent
            opp_dx = bx - ax
            opp_dy = by - ay
            ideal_x = bx + opp_dx
            ideal_y = by + opp_dy
            ideal_x = max(1, min(grid.width - 2, ideal_x))
            ideal_y = max(1, min(grid.height - 2, ideal_y))
            if (
                grid.terrain[ideal_y, ideal_x] not in (CellType.WALL, CellType.HOLE)
                and grid.objects[ideal_y, ideal_x]
                in (ObjectType.NONE, ObjectType.NPC)
                and (ideal_x, ideal_y) != (ax, ay)
            ):
                adj_candidates.insert(0, (ideal_x, ideal_y))  # preferred

            for ddx, ddy in self._DIRS:
                cx, cy = bx + ddx, by + ddy
                if (cx, cy) in adj_candidates:
                    continue
                if (
                    0 < cx < grid.width - 1
                    and 0 < cy < grid.height - 1
                    and grid.terrain[cy, cx]
                    not in (CellType.WALL, CellType.HOLE)
                    and grid.objects[cy, cx]
                    in (ObjectType.NONE, ObjectType.NPC)
                    and (cx, cy) != (ax, ay)
                ):
                    adj_candidates.append((cx, cy))

            # BFS from NPC to any of these candidates
            for cand in adj_candidates:
                if (nx, ny) == cand:
                    target_pos = cand
                    break
                path = self._npc_bfs(grid, (nx, ny), cand, (ax, ay))
                if path and len(path) > 1:
                    target_pos = cand
                    # Take one step along BFS path
                    self._npc_pos = list(path[1])
                    break

            if target_pos is None and adj_candidates:
                # Greedy fallback
                target_pos = adj_candidates[0]

        if target_pos is not None and self._npc_pos == [nx, ny]:
            # If BFS didn't move us, do greedy step
            best, best_d = (nx, ny), abs(nx - target_pos[0]) + abs(ny - target_pos[1])
            for ddx, ddy in self._DIRS:
                cx, cy = nx + ddx, ny + ddy
                if (
                    0 < cx < grid.width - 1
                    and 0 < cy < grid.height - 1
                    and grid.terrain[cy, cx] not in (CellType.WALL, CellType.HOLE)
                    and (cx, cy) != (ax, ay)
                    and grid.objects[cy, cx] in (ObjectType.NONE,)
                    and not self._is_box_at(cx, cy)
                ):
                    d = abs(cx - target_pos[0]) + abs(cy - target_pos[1])
                    if d < best_d:
                        best_d, best = d, (cx, cy)
            self._npc_pos = list(best)
        elif target_pos is None:
            # Wander randomly
            candidates = []
            for ddx, ddy in self._DIRS:
                cx, cy = nx + ddx, ny + ddy
                if (
                    0 < cx < grid.width - 1
                    and 0 < cy < grid.height - 1
                    and grid.terrain[cy, cx] not in (CellType.WALL, CellType.HOLE)
                    and grid.objects[cy, cx] in (ObjectType.NONE,)
                    and (cx, cy) != (ax, ay)
                    and not self._is_box_at(cx, cy)
                ):
                    candidates.append((cx, cy))
            if candidates:
                rng = np.random.default_rng(step_count)
                idx = int(rng.integers(0, len(candidates)))
                self._npc_pos = list(candidates[idx])
            # else stay put

        config["_npc_pos"] = self._npc_pos

        # Place NPC with direction metadata
        nx2, ny2 = self._npc_pos
        if grid.terrain[ny2, nx2] not in (CellType.WALL, CellType.HOLE):
            grid.objects[ny2, nx2] = ObjectType.NPC
            ddx = nx2 - old_nx
            ddy = ny2 - old_ny
            if ddx > 0:
                grid.metadata[ny2, nx2] = 1  # right
            elif ddx < 0:
                grid.metadata[ny2, nx2] = 3  # left
            elif ddy < 0:
                grid.metadata[ny2, nx2] = 0  # up
            elif ddy > 0:
                grid.metadata[ny2, nx2] = 2  # down
            else:
                grid.metadata[ny2, nx2] = 2  # default down

    def _is_box_at(self, x, y):
        """Check if any tracked box is at (x, y)."""
        for bx, by in self._box_positions:
            if bx == x and by == y:
                return True
        return False

    def _npc_bfs(self, grid, start, goal, agent_pos):
        """BFS for NPC movement, avoiding walls, holes, boxes, and agent."""
        if start == goal:
            return [start]
        visited = {start}
        queue = deque([(start, [start])])
        while queue:
            (cx, cy), path = queue.popleft()
            for ddx, ddy in self._DIRS:
                nx, ny = cx + ddx, cy + ddy
                if (nx, ny) in visited:
                    continue
                if not (0 < nx < grid.width - 1 and 0 < ny < grid.height - 1):
                    continue
                if grid.terrain[ny, nx] in (CellType.WALL, CellType.HOLE):
                    continue
                if (nx, ny) == agent_pos:
                    continue
                if self._is_box_at(nx, ny) and (nx, ny) != goal:
                    continue
                visited.add((nx, ny))
                new_path = path + [(nx, ny)]
                if (nx, ny) == goal:
                    return new_path
                queue.append(((nx, ny), new_path))
        return None

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01  # step penalty

        config = new_state.get("config", {})
        if config.get("_fell_in_hole", False):
            return -0.5
        new_delivered = config.get("_boxes_delivered", 0)
        old_config = old_state.get("config", {})
        old_delivered = old_config.get("_boxes_delivered", 0)

        # Reward per box delivered
        deliveries = new_delivered - old_delivered
        if deliveries > 0:
            reward += 0.5 * deliveries

        # Approach shaping: agent distance to nearest undelivered box
        box_positions = config.get("_box_positions", [])
        if box_positions and "agent" in new_state:
            ax, ay = new_state["agent"].position
            min_dist = min(
                abs(ax - bx) + abs(ay - by) for bx, by in box_positions
            )
            # Compare with old distance
            old_boxes = old_config.get("_box_positions", box_positions)
            if old_boxes and "agent_position" in old_state:
                ox, oy = old_state["agent_position"]
                old_min = min(
                    abs(ox - bx) + abs(oy - by) for bx, by in old_boxes
                )
                reward += 0.02 * (old_min - min_dist)

        # Success bonus
        if self.check_success(new_state):
            reward += 1.0

        return reward

    # ------------------------------------------------------------------
    # Success
    # ------------------------------------------------------------------

    def check_done(self, state):
        config = state.get("config", {})
        if config.get("_fell_in_hole", False):
            return True
        return self.check_success(state)

    def check_success(self, state):
        config = state.get("config", {})
        if config.get("_fell_in_hole", False):
            return False
        n_boxes = config.get("n_boxes", getattr(self, "_n_boxes", 1))
        delivered = config.get("_boxes_delivered", 0)
        return delivered >= n_boxes

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0
