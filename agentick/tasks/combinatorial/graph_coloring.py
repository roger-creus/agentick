"""GraphColoring - Color graph nodes so no adjacent nodes share a color.

MECHANICS:
  - N node positions on the grid (SWITCH objects)
  - metadata[y,x] = current color (0=uncolored, 1=red, 2=blue, 3=green, 4=yellow)
  - Agent moves to a node and uses INTERACT to cycle its color:
      uncolored(0) → color1(1) → color2(2) → ... → colorN(n_colors) → uncolored(0)
  - Adjacency: two nodes are adjacent if their Manhattan distance ≤ 4 AND
    no wall blocks the direct shortest path between them
  - Cardinal-only adjacency: only horizontal/vertical proximity counts
  - SUCCESS = all nodes colored with no two adjacent nodes sharing a color
  - Colors are VISUALLY OBVIOUS in all modalities:
      Pixels: distinct colored squares (red, blue, green, yellow via _META_GC_COLORS)
      ASCII: digit 0-4 shown in distinct ANSI colors
      Language: "Node at (x,y) is colored red", etc.

DIFFICULTY:
  - easy:   3 nodes, 2 colors, no obstacles, small graph → easy to solve
  - medium: 5 nodes, 3 colors, 2 obstacles → requires planning
  - hard:   7 nodes, 3 colors, 4 obstacles → harder graph structure
  - expert: 9 nodes, 4 colors, 6 obstacles → complex constraint satisfaction
"""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


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
            params={"n_nodes": 3, "n_colors": 2, "n_obstacles": 0},
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=11,
            max_steps=220,
            params={"n_nodes": 5, "n_colors": 3, "n_obstacles": 2},
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=13,
            max_steps=380,
            params={"n_nodes": 7, "n_colors": 3, "n_obstacles": 4},
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=15,
            max_steps=550,
            params={"n_nodes": 9, "n_colors": 4, "n_obstacles": 6},
        ),
    }

    def _compute_adjacency(self, nodes, grid):
        """Two nodes adjacent if Manhattan distance ≤ 4 and no wall in between."""
        from collections import deque

        adj = {i: set() for i in range(len(nodes))}
        threshold = 4

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                x1, y1 = nodes[i]
                x2, y2 = nodes[j]
                d = abs(x1 - x2) + abs(y1 - y2)
                if d <= threshold:
                    # Check if path exists (no complete wall blocking)
                    start = nodes[i]
                    end = nodes[j]
                    visited = {start}
                    queue = deque([start])
                    found = False
                    while queue and not found:
                        cx, cy = queue.popleft()
                        if (cx, cy) == end:
                            found = True
                            break
                        cd = abs(cx - x1) + abs(cy - y1)
                        if cd >= threshold + 1:
                            continue
                        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                            nx, ny = cx + dx, cy + dy
                            if (
                                0 < nx < grid.width - 1
                                and 0 < ny < grid.height - 1
                                and (nx, ny) not in visited
                                and grid.terrain[ny, nx] != CellType.WALL
                            ):
                                visited.add((nx, ny))
                                queue.append((nx, ny))
                    if found:
                        adj[i].add(j)
                        adj[j].add(i)

        return adj

    def _has_valid_coloring(self, n_nodes, n_colors, adj):
        """Greedy check: can this graph be colored with n_colors?"""
        colors = [-1] * n_nodes
        for i in range(n_nodes):
            used = {colors[j] for j in adj[i] if colors[j] >= 0}
            for c in range(n_colors):
                if c not in used:
                    colors[i] = c
                    break
            if colors[i] < 0:
                return False
        return True

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        n_nodes = self.difficulty_config.params.get("n_nodes", 3)
        n_colors = self.difficulty_config.params.get("n_colors", 2)
        n_obstacles = self.difficulty_config.params.get("n_obstacles", 0)

        for attempt in range(30):
            grid = Grid(size, size)
            grid.terrain[0, :] = CellType.WALL
            grid.terrain[-1, :] = CellType.WALL
            grid.terrain[:, 0] = CellType.WALL
            grid.terrain[:, -1] = CellType.WALL

            agent_pos = (1, 1)

            # Place obstacles
            interior = [
                (x, y) for x in range(2, size - 2) for y in range(2, size - 2)
                if (x, y) != agent_pos
            ]
            rng.shuffle(interior)
            obs_placed = 0
            for ox, oy in interior:
                if obs_placed >= n_obstacles:
                    break
                grid.terrain[oy, ox] = CellType.WALL
                if len(grid.flood_fill(agent_pos)) > (size - 2) ** 2 // 2:
                    obs_placed += 1
                else:
                    grid.terrain[oy, ox] = CellType.EMPTY

            # Place nodes spread out
            free = [
                (x, y) for x in range(2, size - 2) for y in range(2, size - 2)
                if grid.terrain[y, x] == CellType.EMPTY and (x, y) != agent_pos
            ]
            rng.shuffle(free)

            node_positions = []
            min_dist = max(2, (size - 2) // (n_nodes + 1))
            for pos in free:
                if len(node_positions) >= n_nodes:
                    break
                if all(
                    abs(pos[0] - np2[0]) + abs(pos[1] - np2[1]) >= min_dist
                    for np2 in node_positions
                ):
                    node_positions.append(pos)

            if len(node_positions) < n_nodes:
                node_positions = free[:n_nodes]

            if len(node_positions) < n_nodes:
                continue

            reachable = grid.flood_fill(agent_pos)
            if not all(p in reachable for p in node_positions):
                continue

            adj = self._compute_adjacency(node_positions, grid)
            if not self._has_valid_coloring(len(node_positions), n_colors, adj):
                continue

            # Place node markers (SWITCH objects) with metadata=0 (uncolored)
            for nx, ny in node_positions:
                grid.objects[ny, nx] = ObjectType.SWITCH
                grid.metadata[ny, nx] = 0  # uncolored

            adj_list = {i: sorted(adj[i]) for i in range(len(node_positions))}

            return grid, {
                "agent_start": agent_pos,
                "goal_positions": [],
                "node_positions": node_positions,
                "n_colors": n_colors,
                "adjacency": adj_list,
                "max_steps": self.get_max_steps(),
            }

        # Fallback: simple linear graph
        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL
        agent_pos = (1, 1)
        nodes = [(3, 3), (5, 3), (7, 3)] if size > 8 else [(2, 2), (4, 2)]
        nodes = nodes[:n_nodes]
        for nx2, ny2 in nodes:
            if 0 < nx2 < size - 1 and 0 < ny2 < size - 1:
                grid.objects[ny2, nx2] = ObjectType.SWITCH
                grid.metadata[ny2, nx2] = 0
        adj_list = {}
        for i in range(len(nodes)):
            adj_list[i] = []
            if i > 0:
                adj_list[i].append(i - 1)
            if i < len(nodes) - 1:
                adj_list[i].append(i + 1)
        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [],
            "node_positions": nodes,
            "n_colors": n_colors,
            "adjacency": adj_list,
            "max_steps": self.get_max_steps(),
        }

    def on_env_reset(self, agent, grid, config):
        config["_node_colors"] = {}   # node_idx → color (1-based)
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
        # Cycle: 0 → 1 → 2 → ... → n_colors → 0
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
        adj = config.get("adjacency", {})
        neighbors = adj.get(node_idx, adj.get(str(node_idx), []))
        config["_violation"] = False  # re-check all violations
        self._check_all_violations(config, nodes, grid)

    def _check_all_violations(self, config, nodes, grid):
        """Recheck all adjacency constraints."""
        adj = config.get("adjacency", {})
        for i in range(len(nodes)):
            nx, ny = nodes[i]
            ci = int(grid.metadata[ny, nx])
            if ci == 0:
                continue  # uncolored
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
