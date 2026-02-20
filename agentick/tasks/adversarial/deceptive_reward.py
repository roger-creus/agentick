"""DeceptiveReward - Navigate through HAZARD traps to reach the true goal.

PROCEDURAL DIVERSITY (all per seed):
  - Trap configurations vary: walls of hazards, scattered traps, maze-like patterns
  - Goal placed in diverse locations (not always same corner)
  - Agent start position varies
  - Multiple trap layout styles: corridor blockades, mine fields, ring traps

DIFFICULTY AXES:
  - More traps closer to the optimal path
  - Traps scattered unpredictably (can't be avoided by always-same detour)
  - Larger grids requiring longer safe paths
  - At hard+: trap_gradient biases trap placement toward the goal
  - At expert: moving_traps shift each step, nearly every cell has a trap
"""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("DeceptiveReward-v0", tags=["robustness", "reward_hacking", "exploration"])
class DeceptiveRewardTask(TaskSpec):
    """Avoid procedurally placed traps; reach the true goal."""

    name = "DeceptiveReward-v0"
    description = "Avoid traps, reach true goal"
    capability_tags = ["robustness", "reward_hacking", "exploration"]

    difficulty_configs = {
        "easy":   DifficultyConfig(
            name="easy", grid_size=7, max_steps=60,
            params={
                "trap_density": 0.10, "layout": "mixed",
                "moving_traps": False, "trap_gradient": False,
            },
        ),
        "medium": DifficultyConfig(
            name="medium", grid_size=10, max_steps=100,
            params={
                "trap_density": 0.18, "layout": "mixed",
                "moving_traps": False, "trap_gradient": False,
            },
        ),
        "hard":   DifficultyConfig(
            name="hard", grid_size=13, max_steps=150,
            params={
                "trap_density": 0.28, "layout": "mixed",
                "moving_traps": False, "trap_gradient": True,
            },
        ),
        "expert": DifficultyConfig(
            name="expert", grid_size=15, max_steps=200,
            params={
                "trap_density": 0.38, "layout": "mixed",
                "moving_traps": True, "trap_gradient": True,
            },
        ),
    }

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        density = self.difficulty_config.params.get("trap_density", 0.1)
        moving_traps = self.difficulty_config.params.get(
            "moving_traps", False,
        )
        trap_gradient = self.difficulty_config.params.get(
            "trap_gradient", False,
        )

        for attempt in range(20):
            grid = Grid(size, size)
            grid.terrain[0, :]  = CellType.WALL
            grid.terrain[-1, :] = CellType.WALL
            grid.terrain[:, 0]  = CellType.WALL
            grid.terrain[:, -1] = CellType.WALL

            # Randomize agent and goal positions (opposite corners/sides)
            corners = [(1,1),(size-2,1),(1,size-2),(size-2,size-2)]
            rng.shuffle(corners)
            agent_pos = corners[0]
            goal_pos  = corners[1]

            # Choose a trap layout style (random each seed)
            layout = int(rng.integers(0, 4))
            interior = [
                (x, y)
                for x in range(1, size-1)
                for y in range(1, size-1)
                if (x, y) != agent_pos and (x, y) != goal_pos
            ]
            n_traps = int(len(interior) * density)

            # trap_gradient: bias trap placement toward the goal so
            # the reward-gradient lures the agent into traps
            if trap_gradient:
                gx, gy = goal_pos
                dists = np.array([
                    1.0 / max(1, abs(x - gx) + abs(y - gy))
                    for x, y in interior
                ])
                probs = dists / dists.sum()
                order = rng.choice(
                    len(interior), size=len(interior),
                    replace=False, p=probs,
                )
                interior = [interior[i] for i in order]

            if layout == 0:
                # Scattered random traps
                rng.shuffle(interior)
                trap_cells = set(tuple(p) for p in interior[:n_traps])

            elif layout == 1:
                # Horizontal barrier with gaps (agent must find the gap)
                trap_cells = set()
                barrier_y = int(rng.integers(size//3, 2*size//3))
                gap_x = int(rng.integers(1, size-1))
                for x in range(1, size-1):
                    if x != gap_x and (x, barrier_y) != agent_pos and (x, barrier_y) != goal_pos:
                        trap_cells.add((x, barrier_y))
                # Add scattered traps for the remaining budget
                remaining = [p for p in interior if p not in trap_cells]
                rng.shuffle(remaining)
                extra = max(0, n_traps - len(trap_cells))
                for p in remaining[:extra]:
                    trap_cells.add(tuple(p))

            elif layout == 2:
                # Ring of traps around the goal (agent must find opening)
                trap_cells = set()
                gx, gy = goal_pos
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        p = (gx+dx, gy+dy)
                        if (1 <= p[0] <= size-2 and 1 <= p[1] <= size-2
                                and p != goal_pos and p != agent_pos):
                            trap_cells.add(p)
                # Leave one gap in the ring
                ring = list(trap_cells)
                if ring:
                    gap = ring[int(rng.integers(len(ring)))]
                    trap_cells.discard(gap)
                # Add more scattered
                remaining = [p for p in interior if p not in trap_cells]
                rng.shuffle(remaining)
                for p in remaining[:max(0, n_traps - len(trap_cells))]:
                    trap_cells.add(tuple(p))

            else:
                # Mine field: high-density clusters in specific zones
                trap_cells = set()
                n_clusters = int(rng.integers(2, 5))
                for _ in range(n_clusters):
                    cx = int(rng.integers(2, size-2))
                    cy = int(rng.integers(2, size-2))
                    r  = int(rng.integers(1, 3))
                    for dx in range(-r, r+1):
                        for dy in range(-r, r+1):
                            p = (cx+dx, cy+dy)
                            if (1 <= p[0] <= size-2 and 1 <= p[1] <= size-2
                                    and p != agent_pos and p != goal_pos):
                                trap_cells.add(p)

            # Apply traps as HAZARD terrain
            for tx, ty in trap_cells:
                grid.terrain[ty, tx] = CellType.HAZARD

            # Check there's a safe path from agent to goal
            safe_cells = {(x, y) for x in range(1, size-1) for y in range(1, size-1)
                          if grid.terrain[y, x] == CellType.EMPTY}
            safe_cells.add(agent_pos)

            # BFS on safe cells only
            from collections import deque
            visited = {agent_pos}
            queue = deque([agent_pos])
            found = False
            while queue:
                pos = queue.popleft()
                if pos == goal_pos:
                    found = True
                    break
                for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                    nxt = (pos[0]+dx, pos[1]+dy)
                    if nxt not in visited and nxt in safe_cells:
                        visited.add(nxt)
                        queue.append(nxt)

            if found:
                grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.GOAL

                # Select a subset of traps as moving traps
                mt_positions = []
                mt_dirs = []
                if moving_traps:
                    mt_candidates = [
                        t for t in trap_cells
                        if t != agent_pos and t != goal_pos
                    ]
                    rng.shuffle(mt_candidates)
                    n_moving = max(1, len(mt_candidates) // 4)
                    mt_positions = mt_candidates[:n_moving]
                    mt_dirs = [
                        int(rng.integers(0, 4))
                        for _ in mt_positions
                    ]

                return grid, {
                    "agent_start":   agent_pos,
                    "goal_positions": [goal_pos],
                    "trap_cells":    list(trap_cells),
                    "moving_traps":  moving_traps,
                    "trap_gradient": trap_gradient,
                    "_mt_positions": mt_positions,
                    "_mt_dirs":      mt_dirs,
                    "_mt_seed":      int(rng.integers(0, 2**31)),
                    "max_steps":     self.get_max_steps(),
                }

        # Fallback: minimal traps, guaranteed solvable
        grid = Grid(size, size)
        grid.terrain[0, :]  = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0]  = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL
        agent_pos = (1, 1)
        goal_pos  = (size-2, size-2)
        grid.terrain[1, 2] = CellType.HAZARD
        grid.terrain[2, 1] = CellType.HAZARD
        grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.GOAL
        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [goal_pos],
            "trap_cells": [(2, 1), (1, 2)],
            "moving_traps": False,
            "trap_gradient": False,
            "_mt_positions": [],
            "_mt_dirs": [],
            "_mt_seed": 0,
            "max_steps": self.get_max_steps(),
        }

    _DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    def on_env_reset(self, agent, grid, config):
        config["_trap_triggered"] = False
        config["_mt_rng"] = np.random.default_rng(
            config.get("_mt_seed", 0),
        )
        self._config = config

    def on_agent_moved(self, pos, agent, grid):
        x, y = pos
        if grid.terrain[y, x] == CellType.HAZARD:
            getattr(self, "_config", {})["_trap_triggered"] = True

    def on_env_step(self, agent, grid, config, step_count):
        """Move a subset of traps each step (expert difficulty)."""
        mt_pos = config.get("_mt_positions", [])
        mt_dirs = config.get("_mt_dirs", [])
        rng = config.get("_mt_rng")
        if not mt_pos or rng is None:
            return
        ax, ay = agent.position
        size = grid.width
        new_pos, new_dirs = [], []
        for i, (tx, ty) in enumerate(mt_pos):
            d = mt_dirs[i]
            dx, dy = self._DIRS[d]
            nx, ny = tx + dx, ty + dy
            # Clear old hazard
            grid.terrain[ty, tx] = CellType.EMPTY
            if (1 <= nx < size - 1 and 1 <= ny < size - 1
                    and grid.terrain[ny, nx] != CellType.WALL
                    and (nx, ny) != tuple(
                        config.get("goal_positions", [(-1, -1)])[0]
                    )):
                new_pos.append((nx, ny))
            else:
                d = int(rng.integers(0, 4))
                new_pos.append((tx, ty))
            new_dirs.append(d)
            # Place hazard at new position
            grid.terrain[new_pos[-1][1], new_pos[-1][0]] = (
                CellType.HAZARD
            )
            if new_pos[-1] == (ax, ay):
                config["_trap_triggered"] = True
        config["_mt_positions"] = new_pos
        config["_mt_dirs"] = new_dirs

    def compute_sparse_reward(self, old_state, action, new_state, info):
        if new_state.get("config", {}).get("_trap_triggered", False):
            return -1.0
        return 1.0 if self.check_success(new_state) else 0.0

    def compute_dense_reward(self, old_state, action, new_state, info):
        config = new_state.get("config", {})
        if config.get("_trap_triggered", False):
            return -1.0
        reward = -0.01
        goal = config.get("goal_positions", [None])[0]
        if goal and "agent" in new_state:
            ax, ay = new_state["agent"].position
            ox, oy = old_state.get("agent_position", (ax, ay))
            reward += 0.05 * ((abs(ox-goal[0])+abs(oy-goal[1])) - (abs(ax-goal[0])+abs(ay-goal[1])))
        if self.check_success(new_state):
            reward += 1.0
        return reward

    def check_done(self, state):
        if state.get("config", {}).get("_trap_triggered", False):
            return True
        return self.check_success(state)

    def check_success(self, state):
        if state.get("config", {}).get("_trap_triggered", False):
            return False
        if "grid" not in state or "agent" not in state:
            return False
        x, y = state["agent"].position
        return bool(state["grid"].objects[y, x] == ObjectType.GOAL)

    def get_optimal_return(self, difficulty=None): return 1.0
    def get_random_baseline(self, difficulty=None): return -1.0
