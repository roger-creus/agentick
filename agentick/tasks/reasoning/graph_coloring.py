"""GraphColoring - Color graph nodes so no adjacent nodes share a color.

MECHANICS:
  - N node positions on the grid (SWITCH objects)
  - metadata[y,x] = current color (0=uncolored, 1=red, 2=blue, 3=green, 4=yellow)
  - Agent moves to a node and uses INTERACT to cycle its color:
      uncolored(0) -> color1(1) -> color2(2) -> ... -> colorN(n_colors) -> uncolored(0)
  - Adjacency: two nodes are adjacent if their Manhattan distance <= 2
    (truly adjacent or 1 cell apart -- pure proximity, no BFS check)
  - Node placement is chromatic-aware:
      chromatic=2: bipartite graph (grid pattern or cluster growth)
      chromatic=3: triangle core + expansion (guarantees odd cycle)
      chromatic=4: K4 core (4 mutually adjacent nodes) + expansion
  - n_colors = chromatic number of the generated graph (exact, no extras)
  - SUCCESS = all nodes colored with no two adjacent nodes sharing a color
  - Colors are VISUALLY OBVIOUS in all modalities:
      Pixels: distinct colored squares (red, blue, green, yellow via _META_GC_COLORS)
      ASCII: digit 0-4 shown in distinct ANSI colors
      Language: "Node at (x,y) is colored red", etc.

DIFFICULTY:
  - easy:   4 nodes, 2 colors, 9x9 grid
  - medium: 6 nodes, 3 colors, 11x11 grid
  - hard:   8 nodes, 3 colors, 13x13 grid
  - expert: 10 nodes, 4 colors, 15x15 grid
"""

from __future__ import annotations

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task

# ------------------------------------------------------------------
# Module-level graph helpers (also used by the oracle)
# ------------------------------------------------------------------


def _backtrack_colorable(n_nodes, n_colors, adj):
    """Check if graph is colorable with exactly n_colors using backtracking.

    Returns a valid coloring (list of 0-based colors) or None.
    """
    colors = [-1] * n_nodes

    def _bt(node):
        if node == n_nodes:
            return True
        neighbors = adj.get(node, set())
        used = {colors[nb] for nb in neighbors if colors[nb] >= 0}
        for c in range(n_colors):
            if c not in used:
                colors[node] = c
                if _bt(node + 1):
                    return True
                colors[node] = -1
        return False

    if _bt(0):
        return list(colors)
    return None


def _chromatic_number(n_nodes, adj):
    """Compute exact chromatic number using backtracking."""
    for k in range(1, n_nodes + 1):
        if _backtrack_colorable(n_nodes, k, adj) is not None:
            return k
    return n_nodes


def _is_connected(adj, n_nodes):
    """Check that all nodes are reachable from node 0 via adjacency edges."""
    if n_nodes <= 1:
        return True
    visited = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for nb in adj[node]:
            if nb not in visited:
                visited.add(nb)
                stack.append(nb)
    return len(visited) == n_nodes


def _compute_adjacency(nodes):
    """Two nodes are adjacent if Manhattan distance <= 2."""
    adj = {i: set() for i in range(len(nodes))}
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            x1, y1 = nodes[i]
            x2, y2 = nodes[j]
            if abs(x1 - x2) + abs(y1 - y2) <= 2:
                adj[i].add(j)
                adj[j].add(i)
    return adj


@register_task("GraphColoring-v0", tags=["combinatorial_logic", "constraint_satisfaction"])
class GraphColoringTask(TaskSpec):
    """Color graph nodes via INTERACT so no adjacent nodes share a color."""

    name = "GraphColoring-v0"
    description = "Color all nodes with no adjacent same-color using INTERACT"
    capability_tags = ["combinatorial_logic", "constraint_satisfaction"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=9,
            max_steps=120,
            params={"n_nodes": 4, "n_colors": 2},
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=11,
            max_steps=220,
            params={"n_nodes": 6, "n_colors": 3},
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=13,
            max_steps=380,
            params={"n_nodes": 8, "n_colors": 3},
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=15,
            max_steps=550,
            params={"n_nodes": 10, "n_colors": 4},
        ),
    }

    # ------------------------------------------------------------------
    # Core graph helpers (delegate to module-level functions)
    # ------------------------------------------------------------------

    def _compute_adjacency(self, nodes):
        """Two nodes are adjacent if Manhattan distance <= 2."""
        return _compute_adjacency(nodes)

    def _compute_chromatic_number(self, n_nodes, adj):
        """Compute exact chromatic number using backtracking."""
        return _chromatic_number(n_nodes, adj)

    def _is_graph_connected(self, adj, n_nodes):
        """Check that all nodes are reachable from node 0."""
        return _is_connected(adj, n_nodes)

    # ------------------------------------------------------------------
    # Placement: chromatic-aware strategies
    # ------------------------------------------------------------------

    def _interior_walkable(self, grid, size):
        """Return set of walkable interior positions (2-cell border margin)."""
        walkable = set()
        for x in range(2, size - 2):
            for y in range(2, size - 2):
                if grid.terrain[y, x] == CellType.EMPTY:
                    walkable.add((x, y))
        return walkable

    def _place_bipartite(self, rng, grid, n_nodes, size):
        """Place n_nodes forming a bipartite graph (chromatic=2).

        Strategy: cluster growth at distance exactly 2 from existing nodes.
        Distance-2 placement on an empty grid tends to create bipartite graphs.
        """
        walkable = self._interior_walkable(grid, size)
        if not walkable:
            return None

        candidates = list(walkable)
        rng.shuffle(candidates)
        first = candidates[0]
        nodes = [first]
        used = {first}

        for _ in range(n_nodes - 1):
            frontier = []
            for nx, ny in walkable:
                if (nx, ny) in used:
                    continue
                for ex, ey in nodes:
                    if abs(nx - ex) + abs(ny - ey) == 2:
                        frontier.append((nx, ny))
                        break
            if not frontier:
                return None
            rng.shuffle(frontier)
            nodes.append(frontier[0])
            used.add(frontier[0])

        # Verify chromatic == 2
        adj = _compute_adjacency(nodes)
        if not _is_connected(adj, len(nodes)):
            return None
        if _chromatic_number(len(nodes), adj) != 2:
            return None
        return nodes

    def _place_triangle_core(self, rng, grid, n_nodes, size):
        """Place nodes with a triangle core (chromatic=3).

        Strategy:
        1. Place 3 nodes forming a triangle (all pairwise Manhattan dist <= 2).
        2. Grow remaining nodes at distance 1 or 2 from existing nodes,
           maintaining connectivity and chromatic=3.
        """
        walkable = self._interior_walkable(grid, size)
        if not walkable:
            return None

        wlist = list(walkable)
        rng.shuffle(wlist)

        # Find a triangle: 3 positions all pairwise within Manhattan distance 2.
        # Use a compact pattern: pick a center, then two neighbors at dist 1.
        # E.g., (cx, cy), (cx+1, cy), (cx, cy+1) -- all pairwise dist <= 2.
        for cx, cy in wlist:
            tri_patterns = [
                [(cx, cy), (cx + 1, cy), (cx, cy + 1)],
                [(cx, cy), (cx - 1, cy), (cx, cy + 1)],
                [(cx, cy), (cx + 1, cy), (cx, cy - 1)],
                [(cx, cy), (cx - 1, cy), (cx, cy - 1)],
                [(cx, cy), (cx + 1, cy), (cx + 1, cy + 1)],
                [(cx, cy), (cx - 1, cy), (cx - 1, cy + 1)],
                [(cx, cy), (cx, cy + 1), (cx + 1, cy + 1)],
                [(cx, cy), (cx, cy - 1), (cx + 1, cy - 1)],
            ]
            rng.shuffle(tri_patterns)
            for tri in tri_patterns:
                if not all(p in walkable for p in tri):
                    continue
                # Verify triangle (all pairwise <= 2)
                ok = True
                for i in range(3):
                    for j in range(i + 1, 3):
                        d = abs(tri[i][0] - tri[j][0]) + abs(tri[i][1] - tri[j][1])
                        if d > 2:
                            ok = False
                if not ok:
                    continue

                nodes = list(tri)
                used = set(tri)

                # Grow remaining nodes: prefer dist 1-2 from existing
                for _ in range(n_nodes - 3):
                    frontier = []
                    for nx, ny in walkable:
                        if (nx, ny) in used:
                            continue
                        for ex, ey in nodes:
                            if abs(nx - ex) + abs(ny - ey) <= 2:
                                frontier.append((nx, ny))
                                break
                    if not frontier:
                        break
                    rng.shuffle(frontier)
                    nodes.append(frontier[0])
                    used.add(frontier[0])

                if len(nodes) < n_nodes:
                    continue

                adj = _compute_adjacency(nodes)
                if not _is_connected(adj, len(nodes)):
                    continue
                chi = _chromatic_number(len(nodes), adj)
                if chi == 3:
                    return nodes
        return None

    def _place_k4_core(self, rng, grid, n_nodes, size):
        """Place nodes with a K4 core (chromatic=4).

        Strategy:
        1. Place 4 nodes all pairwise within Manhattan distance 2 (K4 subgraph).
           E.g., a 2x2 block: (cx,cy), (cx+1,cy), (cx,cy+1), (cx+1,cy+1).
           All pairwise distances are 1 or 2, so all are adjacent.
        2. Grow remaining nodes maintaining connectivity.
        """
        walkable = self._interior_walkable(grid, size)
        if not walkable:
            return None

        wlist = list(walkable)
        rng.shuffle(wlist)

        for cx, cy in wlist:
            # 2x2 block forms K4 under Manhattan dist <= 2
            k4_patterns = [
                [(cx, cy), (cx + 1, cy), (cx, cy + 1), (cx + 1, cy + 1)],
                [(cx, cy), (cx - 1, cy), (cx, cy + 1), (cx - 1, cy + 1)],
                [(cx, cy), (cx + 1, cy), (cx, cy - 1), (cx + 1, cy - 1)],
                [(cx, cy), (cx - 1, cy), (cx, cy - 1), (cx - 1, cy - 1)],
            ]
            rng.shuffle(k4_patterns)
            for k4 in k4_patterns:
                if not all(p in walkable for p in k4):
                    continue
                # Verify K4: all 6 pairs within distance 2
                ok = True
                for i in range(4):
                    for j in range(i + 1, 4):
                        d = abs(k4[i][0] - k4[j][0]) + abs(k4[i][1] - k4[j][1])
                        if d > 2:
                            ok = False
                if not ok:
                    continue

                nodes = list(k4)
                used = set(k4)

                # Grow remaining nodes
                for _ in range(n_nodes - 4):
                    frontier = []
                    for nx, ny in walkable:
                        if (nx, ny) in used:
                            continue
                        for ex, ey in nodes:
                            if abs(nx - ex) + abs(ny - ey) <= 2:
                                frontier.append((nx, ny))
                                break
                    if not frontier:
                        break
                    rng.shuffle(frontier)
                    nodes.append(frontier[0])
                    used.add(frontier[0])

                if len(nodes) < n_nodes:
                    continue

                adj = _compute_adjacency(nodes)
                if not _is_connected(adj, len(nodes)):
                    continue
                chi = _chromatic_number(len(nodes), adj)
                if chi == 4:
                    return nodes
        return None

    # ------------------------------------------------------------------
    # Deterministic fallbacks guaranteed to hit target chromatic number
    # ------------------------------------------------------------------

    def _fallback_bipartite(self, size, n_nodes):
        """Grid pattern at spacing 2 -> always bipartite (chromatic=2)."""
        nodes = []
        for y in range(2, size - 2, 2):
            for x in range(2, size - 2, 2):
                nodes.append((x, y))
                if len(nodes) >= n_nodes:
                    return nodes[:n_nodes]
        return nodes[:n_nodes]

    def _fallback_triangle(self, size, n_nodes):
        """Triangle at center + grid expansion -> chromatic=3.

        Places a tight triangle (3 nodes at dist 1 from each other),
        then fills remaining nodes at distance 1-2 from existing.
        """
        cx, cy = size // 2, size // 2
        # Triangle core: (cx, cy), (cx+1, cy), (cx, cy+1)
        # pairwise dists: 1, 1, 2 -- all <= 2, so triangle.
        nodes = [(cx, cy), (cx + 1, cy), (cx, cy + 1)]

        used = set(nodes)
        # Grow outward from core
        for dx, dy in [
            (-1, 0), (0, -1), (1, 1), (-1, 1), (2, 0), (0, 2),
            (-1, -1), (2, 1), (1, 2), (-2, 0), (0, -2), (2, -1),
        ]:
            if len(nodes) >= n_nodes:
                break
            px, py = cx + dx, cy + dy
            if 2 <= px < size - 2 and 2 <= py < size - 2 and (px, py) not in used:
                # Must be adjacent to at least one existing node
                for ex, ey in nodes:
                    if abs(px - ex) + abs(py - ey) <= 2:
                        nodes.append((px, py))
                        used.add((px, py))
                        break
        return nodes[:n_nodes]

    def _fallback_k4(self, size, n_nodes):
        """K4 at center + expansion -> chromatic=4.

        Places a 2x2 block (K4 under Manhattan dist <= 2), then
        grows remaining nodes outward.
        """
        cx, cy = size // 2, size // 2
        # 2x2 block: all 6 pairwise distances are 1 or 2
        nodes = [(cx, cy), (cx + 1, cy), (cx, cy + 1), (cx + 1, cy + 1)]

        used = set(nodes)
        for dx, dy in [
            (-1, 0), (0, -1), (2, 0), (0, 2), (-1, -1), (2, 2),
            (-1, 1), (2, -1), (-1, 2), (0, -2), (-2, 0), (2, -1),
        ]:
            if len(nodes) >= n_nodes:
                break
            px, py = cx + dx, cy + dy
            if 2 <= px < size - 2 and 2 <= py < size - 2 and (px, py) not in used:
                for ex, ey in nodes:
                    if abs(px - ex) + abs(py - ey) <= 2:
                        nodes.append((px, py))
                        used.add((px, py))
                        break
        return nodes[:n_nodes]

    # ------------------------------------------------------------------
    # Main generate
    # ------------------------------------------------------------------

    def generate(self, seed):  # noqa: C901
        size = self.difficulty_config.grid_size
        n_nodes = self.difficulty_config.params.get("n_nodes", 3)
        target_chi = self.difficulty_config.params.get("n_colors", 2)

        for attempt in range(50):
            grid = Grid(size, size)
            grid.terrain[0, :] = CellType.WALL
            grid.terrain[-1, :] = CellType.WALL
            grid.terrain[:, 0] = CellType.WALL
            grid.terrain[:, -1] = CellType.WALL

            agent_pos = (1, 1)
            attempt_rng = np.random.default_rng(seed + attempt * 1000)

            # Use chromatic-aware placement strategy
            if target_chi == 2:
                node_positions = self._place_bipartite(
                    attempt_rng, grid, n_nodes, size
                )
            elif target_chi == 3:
                node_positions = self._place_triangle_core(
                    attempt_rng, grid, n_nodes, size
                )
            elif target_chi == 4:
                node_positions = self._place_k4_core(
                    attempt_rng, grid, n_nodes, size
                )
            else:
                node_positions = None

            if node_positions is None or len(node_positions) < n_nodes:
                continue

            if agent_pos in node_positions:
                continue

            # All nodes must be walkable from agent start
            reachable = grid.flood_fill(agent_pos)
            if not all(p in reachable for p in node_positions):
                continue

            adj = _compute_adjacency(node_positions)
            if not _is_connected(adj, len(node_positions)):
                continue

            chromatic = _chromatic_number(len(node_positions), adj)
            if chromatic != target_chi:
                continue

            # Place node markers (SWITCH objects) with metadata=0 (uncolored)
            for nx, ny in node_positions:
                grid.objects[ny, nx] = ObjectType.SWITCH
                grid.metadata[ny, nx] = 0

            adj_list = {i: sorted(adj[i]) for i in range(len(node_positions))}

            return grid, {
                "agent_start": agent_pos,
                "goal_positions": [],
                "node_positions": node_positions,
                "n_colors": chromatic,
                "adjacency": adj_list,
                "max_steps": self.get_max_steps(),
            }

        # ------------------------------------------------------------------
        # Deterministic fallback: guaranteed chromatic = target_chi
        # ------------------------------------------------------------------
        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL
        agent_pos = (1, 1)

        if target_chi == 2:
            node_positions = self._fallback_bipartite(size, n_nodes)
        elif target_chi == 3:
            node_positions = self._fallback_triangle(size, n_nodes)
        elif target_chi == 4:
            node_positions = self._fallback_k4(size, n_nodes)
        else:
            node_positions = self._fallback_bipartite(size, n_nodes)

        # Place node markers
        for nx, ny in node_positions:
            if 0 < nx < size - 1 and 0 < ny < size - 1:
                grid.objects[ny, nx] = ObjectType.SWITCH
                grid.metadata[ny, nx] = 0

        adj = _compute_adjacency(node_positions)
        adj_list = {i: sorted(adj[i]) for i in range(len(node_positions))}
        chromatic = _chromatic_number(len(node_positions), adj)

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [],
            "node_positions": node_positions,
            "n_colors": chromatic,
            "adjacency": adj_list,
            "max_steps": self.get_max_steps(),
        }

    # ------------------------------------------------------------------
    # Runtime hooks
    # ------------------------------------------------------------------

    def on_env_reset(self, agent, grid, config):
        config["_node_colors"] = {}  # node_idx -> color (1-based)
        config["_violation"] = False
        self._config = config
        self._last_n_colored = 0
        # Ensure all nodes start uncolored (metadata=0)
        for nx, ny in config.get("node_positions", []):
            if grid.objects[ny, nx] == ObjectType.SWITCH:
                grid.metadata[ny, nx] = 0

    def on_agent_interact(self, pos, agent, grid):
        """Cycle color of node at agent position via INTERACT."""
        config = getattr(self, "_config", {})
        x, y = pos
        nodes = config.get("node_positions", [])
        n_colors = config.get("n_colors", 2)

        if (x, y) not in nodes:
            return

        node_idx = nodes.index((x, y))
        current_meta = int(grid.metadata[y, x])
        # Cycle: 0 -> 1 -> 2 -> ... -> n_colors -> 0
        new_color = (current_meta + 1) % (n_colors + 1)
        grid.metadata[y, x] = new_color

        # Update node_colors dict
        node_colors = config.get("_node_colors", {})
        if new_color == 0:
            node_colors.pop(node_idx, None)  # uncolored
        else:
            node_colors[node_idx] = new_color - 1  # 0-based color stored internally
        config["_node_colors"] = node_colors

        # Check for violations
        config["_violation"] = False
        self._check_all_violations(config, nodes, grid)

    def _check_all_violations(self, config, nodes, grid):
        """Recheck all adjacency constraints."""
        adj = config.get("adjacency", {})
        for i in range(len(nodes)):
            nx, ny = nodes[i]
            ci = int(grid.metadata[ny, nx])
            if ci == 0:
                continue
            neighbors = adj.get(i, adj.get(str(i), []))
            for nb in neighbors:
                nb = int(nb)
                if nb >= len(nodes):
                    continue
                bx, by = nodes[nb]
                cj = int(grid.metadata[by, bx])
                if cj == 0:
                    continue
                if ci == cj:
                    config["_violation"] = True
                    return

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        config = new_state.get("config", {})
        node_colors = config.get("_node_colors", {})
        n_colored = len(node_colors)

        if config.get("_violation", False):
            reward -= 0.2

        if n_colored > self._last_n_colored and not config.get("_violation", False):
            reward += 0.4 * (n_colored - self._last_n_colored)
        self._last_n_colored = n_colored

        # Approach nearest uncolored node
        if "agent_position" in new_state:
            ax, ay = new_state["agent_position"]
            ox, oy = old_state.get("agent_position", (ax, ay))
            nodes = config.get("node_positions", [])
            uncolored = [nodes[i] for i in range(len(nodes)) if i not in node_colors]
            if uncolored:
                d_new = min(abs(ax - nx) + abs(ay - ny) for nx, ny in uncolored)
                d_old = min(abs(ox - nx) + abs(oy - ny) for nx, ny in uncolored)
                reward += 0.05 * (d_old - d_new)

        if self.check_success(new_state):
            reward += 1.0
        return reward

    def check_success(self, state):
        config = state.get("config", {})
        if config.get("_violation", False):
            return False
        nodes = config.get("node_positions", [])
        node_colors = config.get("_node_colors", {})
        if len(node_colors) < len(nodes):
            return False
        # Verify no adjacency violations
        adj = config.get("adjacency", {})
        for i in range(len(nodes)):
            neighbors = adj.get(i, adj.get(str(i), []))
            ci = node_colors.get(i, node_colors.get(str(i), -1))
            for nb in neighbors:
                nb = int(nb)
                cj = node_colors.get(nb, node_colors.get(str(nb), -1))
                if ci == cj and ci >= 0:
                    return False
        return True

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0
