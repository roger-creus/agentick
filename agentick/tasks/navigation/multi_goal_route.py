"""MultiGoalRoute task - Visit multiple goals in optimal order (TSP-like)."""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("MultiGoalRoute-v0", tags=["planning", "optimization", "navigation"])
class MultiGoalRouteTask(TaskSpec):
    """
    Visit N goals in optimal order (traveling salesman problem variant).

    Agent must visit all goals. Reward is higher for shorter paths.
    """

    name = "MultiGoalRoute-v0"
    description = "Visit N goals in optimal order (TSP-like)"
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
        """Generate a multi-goal route task instance.

        Creates a walled grid with random interior obstacles and N goal
        positions placed at random valid locations. The agent is placed
        at a random valid position. The number of goals scales with
        difficulty. If insufficient valid positions exist, the interior
        is cleared as a fallback.

        Args:
            seed: Random seed for reproducible procedural generation.

        Returns:
            tuple: (grid, config) where grid is the initial Grid state
                with walls and multiple goals, and config contains
                agent_start, goal_positions, goals_visited list, and
                max_steps.
        """
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        n_goals = self.difficulty_config.params.get("n_goals", 2)
        n_obstacles = self.difficulty_config.params.get("n_obstacles", 0)
        n_decoys = self.difficulty_config.params.get("n_decoys", 0)

        grid = Grid(size, size)

        # Add border walls
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

        # Find valid positions
        valid_positions = []
        for y in range(1, size - 1):
            for x in range(1, size - 1):
                if grid.terrain[y, x] == CellType.EMPTY:
                    valid_positions.append((x, y))

        if len(valid_positions) < n_goals + 1:
            # Fallback: clear more space
            grid.terrain[1 : size - 1, 1 : size - 1] = CellType.EMPTY
            valid_positions = [(x, y) for x in range(1, size - 1) for y in range(1, size - 1)]

        # Place agent
        agent_idx = rng.choice(len(valid_positions))
        agent_pos = valid_positions[agent_idx]
        valid_positions.pop(agent_idx)

        # Only place goals in cells reachable from agent (connectivity check)
        reachable = grid.flood_fill(agent_pos)
        reachable_positions = [p for p in valid_positions if p in reachable]
        if len(reachable_positions) < n_goals:
            # Fallback: clear all interior obstacles to guarantee connectivity
            grid.terrain[1 : size - 1, 1 : size - 1] = CellType.EMPTY
            reachable_positions = [
                (x, y)
                for x in range(1, size - 1)
                for y in range(1, size - 1)
                if (x, y) != agent_pos
            ]
        rng.shuffle(reachable_positions)

        # Place goals spread out
        goal_positions = []
        for i in range(n_goals):
            if i >= len(reachable_positions):
                break
            goal_pos = reachable_positions[i]
            goal_positions.append(goal_pos)
            grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.GOAL

        # Place decoy targets (look like goals but don't count)
        decoy_positions = []
        remaining = [p for p in reachable_positions[n_goals:] if p not in set(goal_positions)]
        for dp in remaining[:n_decoys]:
            dx, dy = dp
            grid.objects[dy, dx] = ObjectType.TARGET
            decoy_positions.append(dp)

        config = {
            "agent_start": agent_pos,
            "goal_positions": goal_positions,
            "decoy_positions": decoy_positions,
            "goals_visited": [],
            "max_steps": self.get_max_steps(),
        }

        return grid, config

    def on_env_reset(self, agent, grid, config):
        """Reset visited-goal tracking on episode start."""
        config["goals_visited"] = []
        self._n_goals = len(config.get("goal_positions", []))
        self._visited_goals_set = set()
        self._last_visited_count = 0
        self._decoy_penalty = False
        self._config = config

    def on_agent_moved(self, new_pos, agent, grid):
        """Track goal visits and decoy penalties."""
        x, y = new_pos
        if grid.objects[y, x] == ObjectType.GOAL:
            if not hasattr(self, "_visited_goals_set"):
                self._visited_goals_set = set()
            if new_pos not in self._visited_goals_set:
                self._visited_goals_set.add(new_pos)
                grid.objects[y, x] = ObjectType.NONE  # consume the goal object
        elif grid.objects[y, x] == ObjectType.TARGET:
            # Decoy: consume it and flag penalty
            grid.objects[y, x] = ObjectType.NONE
            self._decoy_penalty = True

    def compute_dense_reward(self, old_state, action, new_state, info):
        """Reward for visiting goals, speed bonus, and decoy penalty."""
        reward = -0.01  # Step penalty encourages shorter paths
        # Decoy penalty
        if getattr(self, "_decoy_penalty", False):
            reward -= 0.2
            self._decoy_penalty = False
        # Goal visit reward
        visited = getattr(self, "_visited_goals_set", set())
        old_visited = getattr(self, "_last_visited_count", 0)
        new_visited = len(visited)
        if new_visited > old_visited:
            reward += 1.0
        self._last_visited_count = new_visited
        # Speed bonus on completion: reward faster routes
        if self.check_success(new_state):
            config = new_state.get("config", {})
            max_steps = config.get("max_steps", 100)
            steps_used = new_state.get("step_count", max_steps)
            speed_bonus = max(0.0, 1.0 - steps_used / max_steps)
            reward += speed_bonus
        # Approach shaping: guide toward nearest remaining (unvisited) goal
        if "agent" in new_state and "grid" in new_state:
            from agentick.core.types import ObjectType as OT

            g = new_state["grid"]
            unvisited = [
                (x, y)
                for y in range(g.height)
                for x in range(g.width)
                if g.objects[y, x] == OT.GOAL
            ]
            if unvisited:
                ax, ay = new_state["agent"].position
                ox, oy = old_state.get("agent_position", (ax, ay))
                d_new = min(abs(ax - gx) + abs(ay - gy) for gx, gy in unvisited)
                d_old = min(abs(ox - gx) + abs(oy - gy) for gx, gy in unvisited)
                reward += 0.05 * (d_old - d_new)
        return reward

    def check_success(self, state):
        """Check if all goals have been visited."""
        if "config" not in state:
            return False
        n_goals = len(state["config"].get("goal_positions", []))
        if n_goals == 0:
            return False
        # Use instance-level tracking (populated by on_agent_moved before this check)
        visited = getattr(self, "_visited_goals_set", set())
        return len(visited) >= n_goals

    def get_optimal_return(self, difficulty=None):
        """Optimal is visiting all goals with minimum steps."""
        diff = difficulty or self.difficulty
        n_goals = self.difficulty_configs[diff].params.get("n_goals", 2)
        return float(n_goals)  # 1.0 per goal in dense reward

    def get_random_baseline(self, difficulty=None):
        """Random agent unlikely to visit all goals."""
        return 0.0
