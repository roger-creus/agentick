"""MazeNavigation task - Solve procedurally generated mazes with key/door gates.

MECHANICS:
  - Procedurally generated perfect maze (recursive backtracker or binary tree)
  - At medium+: colored key/door pairs block the optimal path at choke points
  - Doors are MANDATORY — goal is unreachable without opening them
  - At hard+: hazard terrain in dead ends (avoidable but punishing)
  - No NPCs/guards — pure planning and navigation challenge
"""

import numpy as np

from agentick.core.entity import Entity
from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.generation.maze import MazeConfig, MazeGenerator
from agentick.generation.validation import find_optimal_path, verify_solvable
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("MazeNavigation-v0", tags=["planning", "spatial_reasoning", "navigation"])
class MazeNavigationTask(TaskSpec):
    """Navigate a procedurally generated maze; collect keys and open doors blocking the path."""

    name = "MazeNavigation-v0"
    description = "Solve procedurally generated mazes with key/door gates"
    capability_tags = ["planning", "spatial_reasoning", "navigation"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=7,
            max_steps=60,
            params={
                "loop_freq": 0.30,
                "n_hazards": 0,
                "n_doors": 0,
                "algorithm": "binary_tree",
            },
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=11,
            max_steps=120,
            params={
                "loop_freq": 0.0,
                "n_hazards": 0,
                "n_doors": 1,
                "algorithm": "recursive_backtracker",
            },
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=15,
            max_steps=250,
            params={
                "loop_freq": 0.0,
                "n_hazards": 4,
                "n_doors": 1,
                "algorithm": "recursive_backtracker",
            },
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=21,
            max_steps=500,
            params={
                "loop_freq": 0.0,
                "n_hazards": 6,
                "n_doors": 2,
                "algorithm": "recursive_backtracker",
            },
        ),
    }

    def _generate_maze(
        self,
        grid: Grid,
        rng: np.random.Generator,
        algorithm: str = "recursive_backtracker",
        loop_freq: float = 0.1,
    ):
        """Generate maze using generation engine."""
        maze_config = MazeConfig(
            algorithm=algorithm,
            loop_frequency=loop_freq,
            dead_end_density=0.7,  # Keep most dead ends
        )

        maze_gen = MazeGenerator(rng=rng)
        terrain = maze_gen.generate(grid.width, grid.height, maze_config)
        grid.terrain[:, :] = terrain

    def generate(self, seed):
        """Generate a maze navigation task instance.

        Creates a procedurally generated maze using recursive
        backtracking. The agent is placed near the top-left and the
        goal near the bottom-right to maximize path length. The
        generator ensures odd grid dimensions for maze compatibility,
        verifies solvability with up to 10 retries, and computes the
        optimal path. Falls back to a simple open grid if maze
        generation repeatedly fails.

        Args:
            seed: Random seed for reproducible procedural generation.

        Returns:
            tuple: (grid, config) where grid is the initial Grid state
                with maze walls and goal, and config contains
                agent_start, goal_positions, max_steps, and the
                optimal solution length and path.
        """
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        p = self.difficulty_config.params or {}
        algorithm = p.get("algorithm", "recursive_backtracker")
        loop_freq = p.get("loop_freq", 0.1)

        # Ensure odd size for maze generation
        if size % 2 == 0:
            size += 1

        grid = Grid(size, size)

        # Generate maze (pass algorithm from difficulty params)
        self._generate_maze(grid, rng, algorithm=algorithm, loop_freq=loop_freq)

        # Find valid positions (empty cells)
        valid_positions = []
        for y in range(size):
            for x in range(size):
                if grid.terrain[y, x] == CellType.EMPTY:
                    valid_positions.append((x, y))

        if len(valid_positions) < 2:
            raise ValueError("Generated maze has fewer than 2 empty cells")

        # Place agent at start (first valid position near top-left)
        agent_pos = None
        for y in range(1, size // 2):
            for x in range(1, size // 2):
                if grid.terrain[y, x] == CellType.EMPTY:
                    agent_pos = (x, y)
                    break
            if agent_pos:
                break

        if not agent_pos:
            agent_pos = valid_positions[0]

        # Place goal at opposite end (near bottom-right)
        goal_pos = None
        for y in range(size - 2, size // 2, -1):
            for x in range(size - 2, size // 2, -1):
                if grid.terrain[y, x] == CellType.EMPTY and (x, y) != agent_pos:
                    goal_pos = (x, y)
                    break
            if goal_pos:
                break

        if not goal_pos:
            max_dist = 0
            for pos in valid_positions:
                if pos != agent_pos:
                    dist = abs(pos[0] - agent_pos[0]) + abs(pos[1] - agent_pos[1])
                    if dist > max_dist:
                        max_dist = dist
                        goal_pos = pos
        grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.GOAL

        # Verify solvability and get optimal path
        max_retries = 10
        retry_count = 0
        while retry_count < max_retries:
            is_solvable = verify_solvable(grid, agent_pos, [goal_pos])
            if is_solvable:
                break
            self._generate_maze(grid, rng, algorithm=algorithm, loop_freq=loop_freq)
            retry_count += 1

        if retry_count >= max_retries:
            grid.terrain[:, :] = CellType.EMPTY
            grid.terrain[0, :] = CellType.WALL
            grid.terrain[-1, :] = CellType.WALL
            grid.terrain[:, 0] = CellType.WALL
            grid.terrain[:, -1] = CellType.WALL

        # Compute optimal solution length
        optimal_path, optimal_length = find_optimal_path(grid, agent_pos, [goal_pos])
        path_set = set(optimal_path) if optimal_path else set()
        used_cells = {agent_pos, goal_pos}

        # Place hazards at dead ends (hard/expert only)
        n_hazards = p.get("n_hazards", 0)
        if n_hazards > 0:
            dead_ends = []
            for pos in valid_positions:
                if pos == agent_pos or pos == goal_pos or pos in path_set:
                    continue
                px, py = pos
                neighbors = sum(
                    1
                    for ddx, ddy in self._DIRS
                    if 0 <= px + ddx < size
                    and 0 <= py + ddy < size
                    and grid.terrain[py + ddy, px + ddx] == CellType.EMPTY
                )
                if neighbors == 1:
                    dead_ends.append(pos)
            rng.shuffle(dead_ends)
            for hx, hy in dead_ends[:n_hazards]:
                grid.terrain[hy, hx] = CellType.HAZARD
                used_cells.add((hx, hy))

        # Place key/door pairs at medium+ difficulties.
        # Doors MUST be true chokepoints: removing them disconnects agent from goal.
        n_doors = p.get("n_doors", 0)
        door_positions = []
        key_positions = []
        if n_doors > 0 and optimal_path and len(optimal_path) > 4:
            for door_idx in range(n_doors):
                color = door_idx  # 0=gold, 1=red, 2=blue
                all_door_cells = set(door_positions)

                # Find TRUE chokepoints on the optimal path: cells whose removal
                # (treating all existing doors as walls too) disconnects agent
                # from goal. Prefer the middle portion of the path.
                chokepoints = []
                for pi in range(2, len(optimal_path) - 2):
                    p_cand = optimal_path[pi]
                    if p_cand in used_cells:
                        continue
                    test_blocked = all_door_cells | {p_cand}
                    reach = self._flood_reachable(grid, agent_pos, blocked=test_blocked)
                    if goal_pos not in reach:
                        chokepoints.append((pi, p_cand))

                if not chokepoints:
                    continue  # no chokepoint found for this door

                # Pick a chokepoint in the middle portion of the path
                mid_start = len(chokepoints) // 4
                mid_end = max(mid_start + 1, 3 * len(chokepoints) // 4 + 1)
                cp_idx = int(rng.integers(mid_start, min(mid_end, len(chokepoints))))
                path_idx, door_pos = chokepoints[cp_idx]

                ddx, ddy = door_pos
                grid.objects[ddy, ddx] = ObjectType.DOOR
                grid.metadata[ddy, ddx] = color
                door_positions.append(door_pos)
                used_cells.add(door_pos)

                # Find a key position: reachable from agent WITHOUT crossing
                # any door, preferring dead-end cells.
                all_door_cells = set(door_positions)
                reachable_from_agent = self._flood_reachable(
                    grid, agent_pos, blocked=all_door_cells,
                )
                key_cands = [
                    pos
                    for pos in valid_positions
                    if pos not in used_cells
                    and pos in reachable_from_agent
                    and grid.terrain[pos[1], pos[0]] == CellType.EMPTY
                    and grid.objects[pos[1], pos[0]] == ObjectType.NONE
                ]
                if key_cands:
                    dead_end_keys = [
                        kp
                        for kp in key_cands
                        if sum(
                            1
                            for dx2, dy2 in self._DIRS
                            if 0 <= kp[0] + dx2 < size
                            and 0 <= kp[1] + dy2 < size
                            and grid.terrain[kp[1] + dy2, kp[0] + dx2] == CellType.EMPTY
                        )
                        == 1
                    ]
                    if dead_end_keys:
                        kp = dead_end_keys[int(rng.integers(len(dead_end_keys)))]
                    else:
                        rng.shuffle(key_cands)
                        kp = key_cands[0]
                    kx, ky = kp
                    grid.objects[ky, kx] = ObjectType.KEY
                    grid.metadata[ky, kx] = color
                    key_positions.append(kp)
                    used_cells.add(kp)

        config = {
            "agent_start": agent_pos,
            "goal_positions": [goal_pos],
            "max_steps": self.get_max_steps(),
            "_optimal_solution_length": optimal_length,
            "_optimal_path": optimal_path,
            "door_positions": door_positions,
            "key_positions": key_positions,
        }

        return grid, config

    _DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    def _flood_reachable(self, grid, start, blocked=None):
        """BFS flood fill from *start*, treating *blocked* cells as impassable."""
        from collections import deque

        blocked = blocked or set()
        visited = {start}
        q = deque([start])
        while q:
            cx, cy = q.popleft()
            for dx, dy in self._DIRS:
                nx, ny = cx + dx, cy + dy
                if (
                    (nx, ny) not in visited
                    and 0 <= nx < grid.width
                    and 0 <= ny < grid.height
                    and grid.terrain[ny, nx] != CellType.WALL
                    and (nx, ny) not in blocked
                ):
                    visited.add((nx, ny))
                    q.append((nx, ny))
        return visited

    def on_env_reset(self, agent, grid, config):
        config["_hazard_hit"] = False
        agent.inventory.clear()
        self._config = config

    def can_agent_enter(self, pos, agent, grid) -> bool:
        x, y = pos
        if grid.objects[y, x] == ObjectType.DOOR:
            # Closed doors block movement; open doors (meta >= 10) are passable
            return int(grid.metadata[y, x]) >= 10
        return True

    def on_agent_interact(self, pos, agent, grid):
        """INTERACT on a closed door with matching key unlocks it."""
        if not grid.in_bounds(pos):
            return
        x, y = pos
        if grid.objects[y, x] != ObjectType.DOOR:
            return
        door_meta = int(grid.metadata[y, x])
        if door_meta >= 10:
            return  # already open
        door_color = door_meta
        matching = next(
            (
                e
                for e in agent.inventory
                if e.entity_type == "key"
                and e.properties.get("color") == door_color
            ),
            None,
        )
        if matching:
            agent.inventory.remove(matching)
            grid.metadata[y, x] = door_color + 10

    def on_agent_moved(self, pos, agent, grid):
        x, y = pos
        config = getattr(self, "_config", {})
        if grid.terrain[y, x] == CellType.HAZARD:
            config["_hazard_hit"] = True
        if grid.objects[y, x] == ObjectType.KEY:
            key_color = int(grid.metadata[y, x])
            key_ent = Entity(
                id=f"key_{x}_{y}",
                entity_type="key",
                position=(x, y),
                properties={"color": key_color},
            )
            agent.inventory.append(key_ent)
            grid.objects[y, x] = ObjectType.NONE
            grid.metadata[y, x] = 0

    def check_done(self, state):
        config = state.get("config", {})
        if config.get("_hazard_hit", False):
            return True
        return self.check_success(state)

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        config = new_state.get("config", {})
        if config.get("_hazard_hit", False):
            return -0.5
        if "config" in new_state:
            if "goal_positions" in config and config["goal_positions"]:
                goal_pos = config["goal_positions"][0]
                agent_pos = new_state["agent_position"]
                old_dist = abs(old_state["agent_position"][0] - goal_pos[0]) + abs(
                    old_state["agent_position"][1] - goal_pos[1]
                )
                new_dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
                reward += 0.05 * (old_dist - new_dist)
        if self.check_success(new_state):
            reward += 1.0
        return reward

    def check_success(self, state):
        config = state.get("config", {})
        if config.get("_hazard_hit", False):
            return False
        if "grid" not in state or "agent" not in state:
            return False
        x, y = state["agent"].position
        return bool(state["grid"].objects[y, x] == ObjectType.GOAL)

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0
