"""GraphColoring - Assign colors to graph nodes such that no adjacent nodes share a color.

MECHANICS:
  - N node positions on the grid (SWITCH objects) with adjacency relationships
  - K color stations placed around the grid (KEY objects with metadata color IDs)
  - Agent picks up a color token from a station, then visits a node to assign that color
  - Adjacent nodes (connected by proximity, no wall blocking) must have different colors
  - Success = all nodes assigned valid colors (no two adjacent nodes share a color)
  - Tests constraint satisfaction and planning in spatial domains
"""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("GraphColoring-v0", tags=["combinatorial_logic", "constraint_satisfaction"])
class GraphColoringTask(TaskSpec):
    """Assign colors to graph nodes so no adjacent nodes share a color."""

    name = "GraphColoring-v0"
    description = "Color graph nodes with no adjacent same-color"
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
            max_steps=200,
            params={"n_nodes": 5, "n_colors": 3, "n_obstacles": 2},
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=13,
            max_steps=350,
            params={"n_nodes": 7, "n_colors": 3, "n_obstacles": 4},
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=15,
            max_steps=500,
            params={"n_nodes": 9, "n_colors": 4, "n_obstacles": 6},
        ),
    }

    def _compute_adjacency(self, nodes, grid):
        """Compute adjacency: two nodes are adjacent if BFS distance <= threshold."""
        from collections import deque

        adj = {i: set() for i in range(len(nodes))}
        threshold = max(5, grid.width // 2)

        for i in range(len(nodes)):
            # BFS from node i
            start = nodes[i]
            visited = {start: 0}
            queue = deque([start])
            while queue:
                cx, cy = queue.popleft()
                d = visited[(cx, cy)]
                if d >= threshold:
                    continue
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nx, ny = cx + dx, cy + dy
                    if (
                        0 < nx < grid.width - 1
                        and 0 < ny < grid.height - 1
                        and (nx, ny) not in visited
                        and grid.terrain[ny, nx] != CellType.WALL
                    ):
                        visited[(nx, ny)] = d + 1
                        queue.append((nx, ny))

            for j in range(i + 1, len(nodes)):
                if nodes[j] in visited:
                    adj[i].add(j)
                    adj[j].add(i)

        return adj

    def _has_valid_coloring(self, n_nodes, n_colors, adj):
        """Check if graph can be colored with n_colors (greedy check)."""
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

        for attempt in range(20):
            grid = Grid(size, size)
            grid.terrain[0, :] = CellType.WALL
            grid.terrain[-1, :] = CellType.WALL
            grid.terrain[:, 0] = CellType.WALL
            grid.terrain[:, -1] = CellType.WALL

            agent_pos = (1, 1)

            # Place obstacles
            interior = [
                (x, y)
                for x in range(2, size - 2)
                for y in range(2, size - 2)
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

            # Place nodes spread out on the grid
            free = [
                (x, y)
                for x in range(2, size - 2)
                for y in range(2, size - 2)
                if grid.terrain[y, x] == CellType.EMPTY and (x, y) != agent_pos
            ]
            rng.shuffle(free)

            # Spread nodes by picking well-separated positions
            node_positions = []
            min_dist = max(2, (size - 2) // (n_nodes + 1))
            for pos in free:
                if len(node_positions) >= n_nodes:
                    break
                if all(
                    abs(pos[0] - np[0]) + abs(pos[1] - np[1]) >= min_dist for np in node_positions
                ):
                    node_positions.append(pos)

            # Fallback: just take first n_nodes positions
            if len(node_positions) < n_nodes:
                node_positions = free[:n_nodes]

            if len(node_positions) < n_nodes:
                continue

            # Verify all nodes reachable
            reachable = grid.flood_fill(agent_pos)
            if not all(p in reachable for p in node_positions):
                continue

            # Compute adjacency and verify graph is colorable
            adj = self._compute_adjacency(node_positions, grid)
            if not self._has_valid_coloring(len(node_positions), n_colors, adj):
                continue

            # Place node markers (SWITCH objects)
            for nx, ny in node_positions:
                grid.objects[ny, nx] = ObjectType.SWITCH

            # Place color stations (KEY objects) — one per color, reachable
            color_stations = []
            station_candidates = [
                p for p in free if p not in set(node_positions) and p in reachable
            ]
            rng.shuffle(station_candidates)
            for c in range(n_colors):
                if c < len(station_candidates):
                    sx, sy = station_candidates[c]
                    grid.objects[sy, sx] = ObjectType.KEY
                    color_stations.append(station_candidates[c])

            if len(color_stations) < n_colors:
                continue

            # Serialize adjacency for config
            adj_list = {i: sorted(adj[i]) for i in range(len(node_positions))}

            return grid, {
                "agent_start": agent_pos,
                "goal_positions": [],
                "node_positions": node_positions,
                "color_stations": color_stations,
                "n_colors": n_colors,
                "adjacency": adj_list,
                "max_steps": self.get_max_steps(),
            }

        # Fallback: simple solvable instance
        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL
        agent_pos = (1, 1)
        nodes = [(3, 3), (size - 4, size - 4)]
        for nx, ny in nodes:
            grid.objects[ny, nx] = ObjectType.SWITCH
        grid.objects[2, 2] = ObjectType.KEY
        grid.objects[size - 3, size - 3] = ObjectType.KEY
        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [],
            "node_positions": nodes,
            "color_stations": [(2, 2), (size - 3, size - 3)],
            "n_colors": 2,
            "adjacency": {0: [1], 1: [0]},
            "max_steps": self.get_max_steps(),
        }

    def on_env_reset(self, agent, grid, config):
        config["_current_color"] = -1  # no color held
        config["_node_colors"] = {}  # node_idx -> color_id
        config["_violation"] = False
        self._config = config
        self._last_n_colored = 0

    def on_agent_moved(self, pos, agent, grid):
        config = getattr(self, "_config", {})
        x, y = pos
        nodes = config.get("node_positions", [])
        stations = config.get("color_stations", [])

        # Pick up color from station
        if (x, y) in stations:
            color_idx = stations.index((x, y))
            config["_current_color"] = color_idx
            # Don't remove station — can be revisited for different assignments

        # Assign color to node
        if (x, y) in nodes and config.get("_current_color", -1) >= 0:
            node_idx = nodes.index((x, y))
            color = config["_current_color"]
            node_colors = config.get("_node_colors", {})

            # Check adjacency constraint
            adj = config.get("adjacency", {})
            neighbors = adj.get(node_idx, adj.get(str(node_idx), []))
            for nb in neighbors:
                nb_color = node_colors.get(nb, node_colors.get(str(nb), -1))
                if nb_color == color:
                    config["_violation"] = True

            node_colors[node_idx] = color
            config["_node_colors"] = node_colors

            # Mark node as colored (remove SWITCH)
            grid.objects[y, x] = ObjectType.NONE

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        config = new_state.get("config", {})
        node_colors = config.get("_node_colors", {})
        n_colored = len(node_colors)

        if config.get("_violation", False):
            reward -= 0.3

        if n_colored > self._last_n_colored:
            if not config.get("_violation", False):
                reward += 0.3 * (n_colored - self._last_n_colored)
        self._last_n_colored = n_colored

        # Approach shaping toward nearest uncolored node (or color station if no color)
        if "grid" in new_state and "agent_position" in new_state:
            ax, ay = new_state["agent_position"]
            ox, oy = old_state.get("agent_position", (ax, ay))
            nodes = config.get("node_positions", [])
            uncolored = [n for i, n in enumerate(nodes) if i not in node_colors]

            if config.get("_current_color", -1) < 0:
                # Guide toward nearest color station
                stations = config.get("color_stations", [])
                if stations:
                    d_new = min(abs(ax - sx) + abs(ay - sy) for sx, sy in stations)
                    d_old = min(abs(ox - sx) + abs(oy - sy) for sx, sy in stations)
                    reward += 0.05 * (d_old - d_new)
            elif uncolored:
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
        # All nodes must be colored
        if len(node_colors) < len(nodes):
            return False
        # Verify no adjacency violations
        adj = config.get("adjacency", {})
        for i in range(len(nodes)):
            neighbors = adj.get(i, adj.get(str(i), []))
            for nb in neighbors:
                c_i = node_colors.get(i, node_colors.get(str(i), -1))
                c_nb = node_colors.get(nb, node_colors.get(str(nb), -1))
                if c_i == c_nb:
                    return False
        return True

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0
