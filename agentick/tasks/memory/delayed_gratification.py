"""DelayedGratification - Resist nearby decoy rewards; reach the larger distant goal.

PROCEDURAL DIVERSITY (all per seed):
  - Agent, goal, and decoys placed in randomized positions
  - Decoy positions spread to tempt agent from multiple directions
  - Maze walls added at higher difficulties to make detours necessary
  - Decoy reward size varies slightly per difficulty

DIFFICULTY AXES:
  - More decoys (2/4/6/8 per difficulty)
  - Decoys closer to optimal path (harder to avoid)
  - Maze walls making the path longer (0/4/8/12 per difficulty)
  - Hazards at hard+ (0/0/3/6 per difficulty)
  - Shorter step budget relative to path length
"""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("DelayedGratification-v0", tags=["credit_assignment", "long_horizon"])
class DelayedGratificationTask(TaskSpec):
    """Resist nearby decoy rewards; find the larger distant goal."""

    name = "DelayedGratification-v0"
    description = "Skip decoy rewards, reach distant goal"
    capability_tags = ["credit_assignment", "long_horizon"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=7,
            max_steps=60,
            params={"n_decoys": 2, "n_walls": 0, "n_hazards": 0},
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=10,
            max_steps=100,
            params={"n_decoys": 4, "n_walls": 4, "n_hazards": 0},
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=13,
            max_steps=150,
            params={"n_decoys": 6, "n_walls": 8, "n_hazards": 3},
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=17,
            max_steps=160,
            params={"n_decoys": 12, "n_walls": 18, "n_hazards": 10},
        ),
    }

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        n_decoys = self.difficulty_config.params.get("n_decoys", 1)
        n_walls = self.difficulty_config.params.get("n_walls", 0)

        for attempt in range(20):
            grid = Grid(size, size)
            grid.terrain[0, :] = CellType.WALL
            grid.terrain[-1, :] = CellType.WALL
            grid.terrain[:, 0] = CellType.WALL
            grid.terrain[:, -1] = CellType.WALL

            # Randomize agent start and goal (opposite sides of grid)
            sides = [
                (1, int(rng.integers(1, size - 1))),
                (size - 2, int(rng.integers(1, size - 1))),
                (int(rng.integers(1, size - 1)), 1),
                (int(rng.integers(1, size - 1)), size - 2),
            ]
            rng.shuffle(sides)
            agent_pos = sides[0]
            goal_pos = sides[1]
            if abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1]) < size // 2:
                continue  # too close

            # Add random wall obstacles to make path non-trivial
            wall_cells = set()
            interior = [
                (x, y)
                for x in range(1, size - 1)
                for y in range(1, size - 1)
                if (x, y) != agent_pos and (x, y) != goal_pos
            ]
            rng.shuffle(interior)
            for wx, wy in interior[: n_walls * 3]:
                grid.terrain[wy, wx] = CellType.WALL
                # Verify path still exists
                reachable = grid.flood_fill(agent_pos)
                if goal_pos not in reachable:
                    grid.terrain[wy, wx] = CellType.EMPTY
                else:
                    wall_cells.add((wx, wy))
                    if len(wall_cells) >= n_walls:
                        break

            # Place decoys CLOSE to agent but NOT adjacent to spawn (min distance 3)
            # This ensures the agent has 2-3 clear cells to move before facing choices
            free = [
                (x, y)
                for x in range(1, size - 1)
                for y in range(1, size - 1)
                if grid.terrain[y, x] == CellType.EMPTY
                and (x, y) != agent_pos
                and (x, y) != goal_pos
            ]
            # Sort by closeness to agent (most tempting first)
            # but enforce minimum distance of 3 from spawn
            min_decoy_dist = 3
            free_far = [
                p
                for p in free
                if abs(p[0] - agent_pos[0]) + abs(p[1] - agent_pos[1]) >= min_decoy_dist
            ]
            free_far.sort(key=lambda p: abs(p[0] - agent_pos[0]) + abs(p[1] - agent_pos[1]))
            # Fallback to all free if not enough far positions
            if len(free_far) < n_decoys:
                free.sort(key=lambda p: abs(p[0] - agent_pos[0]) + abs(p[1] - agent_pos[1]))
                free_far = [
                    p for p in free if abs(p[0] - agent_pos[0]) + abs(p[1] - agent_pos[1]) >= 2
                ]
                if len(free_far) < n_decoys:
                    free_far = free
            decoy_positions = []
            used = {agent_pos, goal_pos}
            for p in free_far:
                if len(decoy_positions) >= n_decoys:
                    break
                if p not in used:
                    decoy_positions.append(p)
                    used.add(p)

            if len(decoy_positions) < n_decoys:
                continue

            # Verify goal reachable AND a path exists that avoids all decoys
            reachable = grid.flood_fill(agent_pos)
            if goal_pos not in reachable:
                continue

            # BFS check: path from agent to goal avoiding decoy cells
            decoy_set = set(decoy_positions)
            from collections import deque as _deque

            _q = _deque([(agent_pos,)])
            _visited = {agent_pos}
            _path_ok = False
            while _q:
                (_cur,) = _q.popleft()
                for _ddx, _ddy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                    _npos = (_cur[0] + _ddx, _cur[1] + _ddy)
                    if _npos in _visited:
                        continue
                    if _npos == goal_pos:
                        _path_ok = True
                        break
                    if not (0 < _npos[0] < size - 1 and 0 < _npos[1] < size - 1):
                        continue
                    if grid.terrain[_npos[1], _npos[0]] in (CellType.WALL, CellType.HOLE):
                        continue
                    if _npos in decoy_set:
                        continue
                    _visited.add(_npos)
                    _q.append((_npos,))
                if _path_ok:
                    break

            if not _path_ok:
                continue  # Decoys block all paths - retry

            grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.GOAL
            for dx, dy in decoy_positions:
                grid.objects[dy, dx] = ObjectType.KEY  # KEY = decoy visual

            # Place hazard terrain (must avoid on optimal path)
            n_hazards = self.difficulty_config.params.get("n_hazards", 0)
            hazard_candidates = [
                p for p in free if p not in used and grid.terrain[p[1], p[0]] == CellType.EMPTY
            ]
            rng.shuffle(hazard_candidates)
            hazard_positions = []
            for hp in hazard_candidates[: n_hazards * 2]:
                hx, hy = hp
                grid.terrain[hy, hx] = CellType.HAZARD
                # Check that a decoy-free AND hazard-free path still exists
                _q2 = _deque([(agent_pos,)])
                _vis2 = {agent_pos}
                _ok2 = False
                while _q2:
                    (_cur,) = _q2.popleft()
                    for _ddx, _ddy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                        _npos = (_cur[0] + _ddx, _cur[1] + _ddy)
                        if _npos in _vis2:
                            continue
                        if _npos == goal_pos:
                            _ok2 = True
                            break
                        if not (0 < _npos[0] < size - 1 and 0 < _npos[1] < size - 1):
                            continue
                        if grid.terrain[_npos[1], _npos[0]] in (
                            CellType.WALL,
                            CellType.HOLE,
                            CellType.HAZARD,
                        ):
                            continue
                        if _npos in decoy_set:
                            continue
                        _vis2.add(_npos)
                        _q2.append((_npos,))
                    if _ok2:
                        break
                if _ok2:
                    hazard_positions.append(hp)
                else:
                    grid.terrain[hy, hx] = CellType.EMPTY
                if len(hazard_positions) >= n_hazards:
                    break

            return grid, {
                "agent_start": agent_pos,
                "goal_positions": [goal_pos],
                "decoy_positions": decoy_positions,
                "n_decoys": n_decoys,
                "max_steps": self.get_max_steps(),
            }

        # Fallback — random corner start
        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL
        corners = [(1, 1), (size - 2, 1), (1, size - 2), (size - 2, size - 2)]
        rng.shuffle(corners)
        agent_pos = corners[0]
        goal_pos = corners[1]
        grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.GOAL
        grid.objects[1, 3] = ObjectType.KEY
        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [goal_pos],
            "decoy_positions": [(3, 1)],
            "n_decoys": 1,
            "max_steps": self.get_max_steps(),
        }

    def on_env_reset(self, agent, grid, config):
        self._decoy_taken = False
        self._config = config

    def on_agent_moved(self, pos, agent, grid):
        x, y = pos
        if grid.objects[y, x] == ObjectType.KEY:
            grid.objects[y, x] = ObjectType.NONE
            self._decoy_taken = True

    def compute_sparse_reward(self, old_state, action, new_state, info):
        if self._decoy_taken:
            return 0.05
        return 1.0 if self.check_success(new_state) else 0.0

    def compute_dense_reward(self, old_state, action, new_state, info):
        if self._decoy_taken:
            return -0.5
        if self.check_success(new_state):
            return 1.0
        reward = -0.01
        config = new_state.get("config", {})
        goal = config.get("goal_positions", [None])[0]
        if goal and "agent" in new_state:
            ax, ay = new_state["agent"].position
            ox, oy = old_state.get("agent_position", (ax, ay))
            reward += 0.05 * (
                (abs(ox - goal[0]) + abs(oy - goal[1])) - (abs(ax - goal[0]) + abs(ay - goal[1]))
            )
        return reward

    def check_success(self, state):
        if self._decoy_taken:
            return False
        if "grid" not in state or "agent" not in state:
            return False
        x, y = state["agent"].position
        return bool(state["grid"].objects[y, x] == ObjectType.GOAL)

    def check_done(self, state):
        if self._decoy_taken:
            return True
        return self.check_success(state)

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.05
