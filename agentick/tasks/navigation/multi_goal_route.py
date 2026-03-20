"""ShortestPath task - Visit all goals with near-optimal efficiency.

SUCCESS METRIC:
  Success = all goals visited AND efficiency >= 90%
  Efficiency = optimal_path_length / steps_taken
  i.e. steps_taken <= optimal_path_length / 0.9  (~optimal * 1.11)
"""

from collections import deque
from itertools import permutations

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task

_EFFICIENCY_THRESHOLD = 0.9  # Success requires >= 90% path efficiency


def _bfs_distance(grid, start, end):
    """Compute shortest path distance between two points on the grid."""
    if start == end:
        return 0
    visited = {start}
    queue = deque([(start, 0)])
    while queue:
        (cx, cy), dist = queue.popleft()
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = cx + dx, cy + dy
            if (nx, ny) == end:
                return dist + 1
            if (nx, ny) in visited:
                continue
            if not (0 <= nx < grid.width and 0 <= ny < grid.height):
                continue
            if grid.terrain[ny, nx] == CellType.WALL:
                continue
            visited.add((nx, ny))
            queue.append(((nx, ny), dist + 1))
    return float("inf")


def _compute_optimal_tsp(grid, agent_pos, goal_positions):
    """Compute optimal TSP path length visiting all goals from agent_pos.

    For small N (<=6), uses exact brute-force over permutations.
    For larger N, uses nearest-neighbor greedy heuristic.
    """
    n = len(goal_positions)
    if n == 0:
        return 0

    all_points = [agent_pos] + list(goal_positions)
    # Precompute pairwise BFS distances
    dist_cache = {}
    for i, p1 in enumerate(all_points):
        for j, p2 in enumerate(all_points):
            if i < j:
                d = _bfs_distance(grid, p1, p2)
                dist_cache[(i, j)] = d
                dist_cache[(j, i)] = d
            elif i == j:
                dist_cache[(i, j)] = 0

    if n <= 6:
        # Exact: try all permutations
        best = float("inf")
        for perm in permutations(range(1, n + 1)):
            cost = dist_cache[(0, perm[0])]
            for k in range(len(perm) - 1):
                cost += dist_cache[(perm[k], perm[k + 1])]
            best = min(best, cost)
        return best
    else:
        # Greedy nearest neighbor
        visited_set = set()
        current = 0
        total = 0
        for _ in range(n):
            best_next, best_d = -1, float("inf")
            for j in range(1, n + 1):
                if j not in visited_set:
                    d = dist_cache[(current, j)]
                    if d < best_d:
                        best_d, best_next = d, j
            total += best_d
            visited_set.add(best_next)
            current = best_next
        return total


@register_task("ShortestPath-v0", tags=["planning", "optimization", "navigation"])
class ShortestPathTask(TaskSpec):
    """Visit all goals with near-optimal efficiency.

    Success = all goals visited AND efficiency (optimal / steps) >= 90%.
    """

    name = "ShortestPath-v0"
    description = "Visit all goals within a step budget of the optimal path"
    capability_tags = ["planning", "optimization", "navigation"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=7,
            max_steps=100,
            params={"n_goals": 2, "n_obstacles": 0, "n_decoys": 0},
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=10,
            max_steps=200,
            params={"n_goals": 3, "n_obstacles": 3, "n_decoys": 1},
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=13,
            max_steps=350,
            params={"n_goals": 4, "n_obstacles": 5, "n_decoys": 2},
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=15,
            max_steps=500,
            params={"n_goals": 5, "n_obstacles": 8, "n_decoys": 3},
        ),
    }

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        n_goals = self.difficulty_config.params.get("n_goals", 2)
        n_obstacles = self.difficulty_config.params.get("n_obstacles", 0)
        n_decoys = self.difficulty_config.params.get("n_decoys", 0)

        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        # Add random interior obstacles
        n_walls = max(n_obstacles, size // 2)
        for _ in range(n_walls):
            x, y = rng.integers(1, size - 1), rng.integers(1, size - 1)
            if grid.terrain[y, x] == CellType.EMPTY:
                grid.terrain[y, x] = CellType.WALL

        valid_positions = [
            (x, y)
            for y in range(1, size - 1)
            for x in range(1, size - 1)
            if grid.terrain[y, x] == CellType.EMPTY
        ]

        if len(valid_positions) < n_goals + 1:
            grid.terrain[1 : size - 1, 1 : size - 1] = CellType.EMPTY
            valid_positions = [
                (x, y) for x in range(1, size - 1) for y in range(1, size - 1)
            ]

        agent_idx = rng.choice(len(valid_positions))
        agent_pos = valid_positions[agent_idx]
        valid_positions.pop(agent_idx)

        reachable = grid.flood_fill(agent_pos)
        reachable_positions = [p for p in valid_positions if p in reachable]
        if len(reachable_positions) < n_goals:
            grid.terrain[1 : size - 1, 1 : size - 1] = CellType.EMPTY
            reachable_positions = [
                (x, y)
                for x in range(1, size - 1)
                for y in range(1, size - 1)
                if (x, y) != agent_pos
            ]
        rng.shuffle(reachable_positions)

        goal_positions = []
        for i in range(n_goals):
            if i >= len(reachable_positions):
                break
            goal_pos = reachable_positions[i]
            goal_positions.append(goal_pos)
            grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.GOAL

        # Compute optimal TSP path length for success check
        optimal_path = _compute_optimal_tsp(grid, agent_pos, goal_positions)

        # Place decoy targets
        decoy_positions = []
        remaining = [
            p for p in reachable_positions[n_goals:] if p not in set(goal_positions)
        ]
        for dp in remaining[:n_decoys]:
            dx, dy = dp
            grid.objects[dy, dx] = ObjectType.TARGET
            decoy_positions.append(dp)

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": goal_positions,
            "decoy_positions": decoy_positions,
            "goals_visited": [],
            "optimal_path_length": optimal_path,
            "max_steps": self.get_max_steps(),
        }

    def on_env_reset(self, agent, grid, config):
        config["goals_visited"] = []
        self._n_goals = len(config.get("goal_positions", []))
        self._visited_goals_set = set()
        self._last_visited_count = 0
        self._decoy_penalty = False
        self._steps_taken = 0
        self._config = config

    def on_agent_moved(self, new_pos, agent, grid):
        x, y = new_pos
        if grid.objects[y, x] == ObjectType.GOAL:
            if not hasattr(self, "_visited_goals_set"):
                self._visited_goals_set = set()
            if new_pos not in self._visited_goals_set:
                self._visited_goals_set.add(new_pos)
                grid.objects[y, x] = ObjectType.NONE
        elif grid.objects[y, x] == ObjectType.TARGET:
            grid.objects[y, x] = ObjectType.NONE
            self._decoy_penalty = True

    def on_env_step(self, agent, grid, config, step_count):
        self._steps_taken = step_count

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        if getattr(self, "_decoy_penalty", False):
            reward -= 0.2
            self._decoy_penalty = False
        visited = getattr(self, "_visited_goals_set", set())
        old_visited = getattr(self, "_last_visited_count", 0)
        new_visited = len(visited)
        if new_visited > old_visited:
            reward += 1.0
        self._last_visited_count = new_visited
        if self.check_success(new_state):
            reward += 1.0
        # Approach shaping
        if "agent" in new_state and "grid" in new_state:
            g = new_state["grid"]
            unvisited = [
                (x, y)
                for y in range(g.height)
                for x in range(g.width)
                if g.objects[y, x] == ObjectType.GOAL
            ]
            if unvisited:
                ax, ay = new_state["agent"].position
                ox, oy = old_state.get("agent_position", (ax, ay))
                d_new = min(abs(ax - gx) + abs(ay - gy) for gx, gy in unvisited)
                d_old = min(abs(ox - gx) + abs(oy - gy) for gx, gy in unvisited)
                reward += 0.05 * (d_old - d_new)
        return reward

    def check_done(self, state):
        """Episode ends when all goals are visited (regardless of budget)."""
        if "config" not in state:
            return False
        config = state["config"]
        n_goals = len(config.get("goal_positions", []))
        if n_goals == 0:
            return False
        visited = getattr(self, "_visited_goals_set", set())
        return len(visited) >= n_goals

    def check_success(self, state):
        """All goals visited with >= 90% efficiency (optimal / steps)."""
        if "config" not in state:
            return False
        config = state["config"]
        n_goals = len(config.get("goal_positions", []))
        if n_goals == 0:
            return False
        visited = getattr(self, "_visited_goals_set", set())
        if len(visited) < n_goals:
            return False
        # Check efficiency: optimal / steps >= 0.9
        optimal = config.get("optimal_path_length", 0)
        steps = max(getattr(self, "_steps_taken", 0), 1)
        if optimal <= 0:
            return True  # degenerate case: agent already at all goals
        return optimal / steps >= _EFFICIENCY_THRESHOLD

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0
