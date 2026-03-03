"""FogOfWarExploration task - Navigate under persistent fog with spatial memory."""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("FogOfWarExploration-v0", tags=["memory", "spatial_memory"])
class FogOfWarExplorationTask(TaskSpec):
    """Navigate a grid under persistent fog where only nearby tiles are visible.

    Fog never clears: at each step the agent can only see cells within
    Manhattan distance 2 (a diamond of 13 cells).  All other cells are
    hidden.  The agent must remember previously seen terrain, plan
    routes to unexplored areas, and locate the goal using spatial
    memory alone.  Decoy TARGETs look identical to the real GOAL until
    stepped on; guards patrol and end the episode on collision.

    Difficulty Levels:
        - easy: 7x7 grid, no decoys, no guards, 100 max steps.
        - medium: 10x10 grid, 2 decoys, no guards, 200 max steps.
        - hard: 13x13 grid, 3 decoys, 1 guard, 350 max steps.
        - expert: 15x15 grid, 4 decoys, 2 guards, 500 max steps.

    Capabilities Tested:
        - memory: The agent must remember previously seen cells to
          build an internal map and avoid revisiting dead ends.
        - spatial_memory: The agent must maintain a spatial model of
          explored vs unexplored regions under persistent fog.

    Example:
        >>> env = agentick.make("FogOfWarExploration-v0", difficulty="medium")
        >>> obs, info = env.reset(seed=42)
        >>> # Navigate the foggy grid using spatial memory to find the goal
    """

    name = "FogOfWarExploration-v0"
    description = "Navigate under persistent fog using spatial memory"
    capability_tags = ["memory", "spatial_memory"]

    difficulty_configs = {
        # n_decoys: fake TARGET lures (not real goal) | n_guards: patrol
        "easy": DifficultyConfig(
            name="easy",
            grid_size=7,
            max_steps=100,
            params={"n_decoys": 0, "n_guards": 0},
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=10,
            max_steps=200,
            params={"n_decoys": 2, "n_guards": 0},
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=13,
            max_steps=350,
            params={"n_decoys": 3, "n_guards": 1},
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=15,
            max_steps=500,
            params={"n_decoys": 4, "n_guards": 2},
        ),
    }

    def generate(self, seed):
        """Generate a fog-of-war exploration task instance.

        Creates a walled grid with randomly placed interior walls and a
        goal at a reachable position. The generator tries up to 10
        attempts to produce a solvable instance where the goal is
        reachable from the agent start via flood fill. Falls back to a
        simple open layout if all attempts fail.

        Args:
            seed: Random seed for reproducible procedural generation.

        Returns:
            tuple: (grid, metadata) where grid is the initial Grid state
                with walls and goal, and metadata contains agent_start,
                goal_positions, and max_steps.
        """
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        p = self.difficulty_config.params or {}
        n_decoys = p.get("n_decoys", 0)
        n_guards = p.get("n_guards", 0)

        # Randomize agent start corner
        corners = [(1, 1), (size - 2, 1), (1, size - 2), (size - 2, size - 2)]
        rng.shuffle(corners)

        # Try multiple times to generate a valid instance
        max_attempts = 10
        for attempt in range(max_attempts):
            grid = Grid(size, size)
            grid.terrain[0, :] = CellType.WALL
            grid.terrain[-1, :] = CellType.WALL
            grid.terrain[:, 0] = CellType.WALL
            grid.terrain[:, -1] = CellType.WALL

            # Add some walls
            for _ in range(size):
                x, y = rng.integers(1, size - 1), rng.integers(1, size - 1)
                grid.terrain[y, x] = CellType.WALL

            agent_pos = corners[0]

            # Find reachable positions from agent
            reachable = grid.flood_fill(agent_pos)
            if len(reachable) < n_decoys + n_guards + 2:
                continue

            reachable_list = list(reachable - {agent_pos})
            rng.shuffle(reachable_list)

            # For hard+ difficulties, enforce minimum goal distance from agent
            min_dist = size // 2 if self.difficulty in ("hard", "expert") else 0
            if min_dist > 0:
                far = [
                    p for p in reachable_list
                    if abs(p[0] - agent_pos[0]) + abs(p[1] - agent_pos[1]) >= min_dist
                ]
                goal_pos = far[0] if far else reachable_list[0]
            else:
                goal_pos = reachable_list[0]
            goal_x, goal_y = goal_pos
            grid.objects[goal_y, goal_x] = ObjectType.GOAL

            # Add decoy targets (look like goal in fog, but aren't)
            decoy_positions = reachable_list[1 : 1 + n_decoys]
            for dx, dy in decoy_positions:
                grid.objects[dy, dx] = ObjectType.TARGET

            # Place guards away from agent
            guard_candidates = reachable_list[1 + n_decoys :]
            guard_positions = guard_candidates[:n_guards]
            for gx, gy in guard_positions:
                grid.objects[gy, gx] = ObjectType.NPC

            return grid, {
                "agent_start": agent_pos,
                "goal_positions": [(goal_x, goal_y)],
                "_guard_positions": guard_positions,
                "_guard_dirs": [int(rng.integers(0, 4)) for _ in guard_positions],
                "_guard_seed": int(rng.integers(0, 2**31)),
                "max_steps": self.get_max_steps(),
            }

        # Fallback: simple solvable instance
        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL
        agent_pos = (1, 1)
        goal_pos = (size - 2, size - 2)
        grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.GOAL
        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [goal_pos],
            "_guard_positions": [],
            "_guard_dirs": [],
            "_guard_seed": 0,
            "max_steps": self.get_max_steps(),
        }

    _DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    def on_env_reset(self, agent, grid, config):
        config["_guard_collision"] = False
        config["_guard_rng"] = np.random.default_rng(config.get("_guard_seed", 0))
        self._config = config
        # Initialize fog: all cells start as fogged
        grid.metadata[:, :] = -1  # META_FOG
        # Reveal cells around agent start (vis=1 always)
        self._reveal_around(agent.position, grid)

    def _reveal_around(self, pos, grid):
        """Reveal all cells within Manhattan distance 2 of the agent."""
        ax, ay = pos
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if abs(dx) + abs(dy) <= 2:
                    nx, ny = ax + dx, ay + dy
                    if 0 <= nx < grid.width and 0 <= ny < grid.height:
                        grid.metadata[ny, nx] = 0

    def on_agent_moved(self, pos, agent, grid):
        """Re-fog entire grid, then reveal tiles within Manhattan distance 2."""
        grid.metadata[:, :] = -1
        self._reveal_around(pos, grid)

    def on_env_step(self, agent, grid, config, step_count):
        guards = config.get("_guard_positions", [])
        dirs = config.get("_guard_dirs", [])
        rng = config.get("_guard_rng")
        ax, ay = agent.position
        if not guards or rng is None:
            return
        for gx, gy in guards:
            if grid.objects[gy, gx] == ObjectType.NPC:
                grid.objects[gy, gx] = ObjectType.NONE
        new_g, new_d = [], []
        for i, (gx, gy) in enumerate(guards):
            d = dirs[i]
            dx, dy = self._DIRS[d]
            nx, ny = gx + dx, gy + dy
            if (
                0 < nx < grid.width - 1
                and 0 < ny < grid.height - 1
                and grid.terrain[ny, nx] == CellType.EMPTY
                and grid.objects[ny, nx] != ObjectType.GOAL
            ):
                new_g.append((nx, ny))
            else:
                d = int(rng.integers(0, 4))
                new_g.append((gx, gy))
            new_d.append(d)
            if (new_g[-1][0], new_g[-1][1]) == (ax, ay):
                config["_guard_collision"] = True
        config["_guard_positions"] = new_g
        config["_guard_dirs"] = new_d
        for gx, gy in new_g:
            if grid.terrain[gy, gx] == CellType.EMPTY:
                grid.objects[gy, gx] = ObjectType.NPC
        # Re-fog entire grid and reveal tiles within Manhattan distance 2
        grid.metadata[:, :] = -1
        self._reveal_around(agent.position, grid)

    def check_done(self, state):
        if state.get("config", {}).get("_guard_collision", False):
            return True
        return self.check_success(state)

    def compute_dense_reward(self, old_state, action, new_state, info):
        """Dense reward: step penalty + goal shaping + success bonus."""
        reward = -0.01
        if "agent" in new_state and "config" in new_state:
            config = new_state["config"]
            goal = config.get("goal_positions", [None])[0]
            ax, ay = new_state["agent"].position
            ox, oy = old_state.get("agent_position", (ax, ay))
            if goal and (ax != ox or ay != oy):  # only shape when agent actually moved
                old_d = abs(ox - goal[0]) + abs(oy - goal[1])
                new_d = abs(ax - goal[0]) + abs(ay - goal[1])
                reward += 0.05 * (old_d - new_d)  # mild shaping (fog limits knowledge)
        if self.check_success(new_state):
            reward += 1.0
        return reward

    def check_success(self, state):
        """Check if the task objective is complete.

        The task succeeds when the agent reaches the goal cell after
        exploring the fog-covered grid.

        Args:
            state: Current state dict containing 'grid' and 'agent' keys.

        Returns:
            True if the agent is on the goal cell, False otherwise.
        """
        if state.get("config", {}).get("_guard_collision", False):
            return False
        if "grid" not in state or "agent" not in state:
            return False
        x, y = state["agent"].position
        return bool(state["grid"].objects[y, x] == ObjectType.GOAL)

    def get_optimal_return(self, difficulty=None):
        """Get the optimal (maximum possible) return for this task.

        Args:
            difficulty: Difficulty level string, or None to use the
                current instance difficulty.

        Returns:
            Optimal return of 1.0 (sparse success reward).
        """
        return 1.0

    def get_random_baseline(self, difficulty=None):
        """Get expected return for a random agent baseline.

        A random agent explores inefficiently under fog-of-war
        conditions, yielding near-zero expected return.

        Args:
            difficulty: Difficulty level string, or None to use the
                current instance difficulty.

        Returns:
            Expected random agent return of 0.0.
        """
        return 0.0
