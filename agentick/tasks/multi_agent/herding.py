"""Herding - Herd NPC sheep into a pen (goal zone).

MECHANICS:
  - N sheep NPCs placed randomly on the grid
  - Sheep flee from agent (move away when agent is adjacent)
  - A pen zone (bottom-right corner, marked TARGET) is the goal
  - Success = all sheep inside the pen zone
  - Agent acts as a herder, using their movement to guide sheep
"""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("Herding-v0", tags=["multi_objective_control"])
class HerdingTask(TaskSpec):
    """Herd all sheep into the pen using movement-based pressure."""

    name = "Herding-v0"
    description = "Herd sheep into the goal pen"
    capability_tags = ["multi_objective_control"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=9,
            max_steps=120,
            params={
                "n_sheep": 2,
                "pen_size": 2,
                "n_obstacles": 0,
                "sheep_speed": 3,
                "pen_rand": False,
                "n_predators": 0,
                "leader_sheep": 0,
            },
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=12,
            max_steps=220,
            params={
                "n_sheep": 3,
                "pen_size": 3,
                "n_obstacles": 2,
                "sheep_speed": 2,
                "pen_rand": True,
                "n_predators": 0,
                "leader_sheep": 0,
            },
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=15,
            max_steps=350,
            params={
                "n_sheep": 4,
                "pen_size": 3,
                "n_obstacles": 4,
                "sheep_speed": 2,
                "pen_rand": True,
                "n_predators": 0,
                "leader_sheep": 0,
            },
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=18,
            max_steps=600,
            params={
                "n_sheep": 5,
                "pen_size": 3,
                "n_obstacles": 6,
                "sheep_speed": 1,
                "pen_rand": True,
                "n_predators": 0,
                "leader_sheep": 0,
            },
        ),
    }

    _DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        p = self.difficulty_config.params
        n_sheep = p.get("n_sheep", 2)
        pen_size = p.get("pen_size", 2)
        n_obs = p.get("n_obstacles", 0)
        pen_rand = p.get("pen_rand", False)

        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        if pen_rand:
            pen_corners = [
                (size - 1 - pen_size, size - 1 - pen_size),
                (1, size - 1 - pen_size),
                (size - 1 - pen_size, 1),
                (1, 1),
            ]
            pen_origin_x, pen_origin_y = pen_corners[int(rng.integers(0, 4))]
        else:
            pen_origin_x = size - 1 - pen_size
            pen_origin_y = size - 1 - pen_size

        possible_starts = [
            (1, 1),
            (1, size - 2),
            (size - 2, 1),
            (size - 2, size - 2),
        ]
        rng.shuffle(possible_starts)
        agent_pos = possible_starts[0]

        pen_cells = set()
        for py in range(
            pen_origin_y,
            min(pen_origin_y + pen_size, size - 1),
        ):
            for px in range(
                pen_origin_x,
                min(pen_origin_x + pen_size, size - 1),
            ):
                if 0 < px < size - 1 and 0 < py < size - 1 and (px, py) != agent_pos:
                    grid.objects[py, px] = ObjectType.TARGET
                    pen_cells.add((px, py))

        placed_obs = 0
        obs_candidates = [
            (x, y)
            for x in range(2, size - 2)
            for y in range(2, size - 2)
            if (x, y) not in pen_cells and (x, y) != agent_pos
        ]
        rng.shuffle(obs_candidates)
        for ox, oy in obs_candidates[: n_obs * 4]:
            if placed_obs >= n_obs:
                break
            grid.terrain[oy, ox] = CellType.WALL
            if len(grid.flood_fill(agent_pos)) < n_sheep + 2:
                grid.terrain[oy, ox] = CellType.EMPTY
            else:
                placed_obs += 1

        reachable = list(grid.flood_fill(agent_pos) - {agent_pos} - pen_cells)
        rng.shuffle(reachable)
        sheep_positions = list(reachable[:n_sheep])

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": list(pen_cells),
            "pen_cells": list(pen_cells),
            "sheep_positions": sheep_positions,
            "predator_positions": [],
            "leader_indices": [],
            "max_steps": self.get_max_steps(),
            "_rng_seed": int(rng.integers(0, 2**31)),
            "_sheep_speed": p.get("sheep_speed", 3),
        }

    # ── Dynamic sheep movement ────────────────────────────────────────────────

    def on_env_reset(self, agent, grid, config):
        self._config = config
        config["_live_sheep"] = list(config.get("sheep_positions", []))
        config["_captured_sheep"] = []  # sheep locked in pen (won't move)
        config["_predators"] = []
        config["_leader_indices"] = []
        config["_sheep_rng"] = np.random.default_rng(config.get("_rng_seed", 0))
        self._draw_sheep(grid, config["_live_sheep"], draw=True)

    def on_env_step(self, agent, grid, config, step_count):
        if "_live_sheep" not in config:
            return
        sheep = config["_live_sheep"]
        rng = config["_sheep_rng"]
        speed = config.get("_sheep_speed", 3)
        ax, ay = agent.position
        pen = set(map(tuple, config.get("pen_cells", [])))
        captured = set(map(tuple, config.get("_captured_sheep", [])))

        # Detect newly penned sheep and lock them — remove from grid to free tile
        newly_captured = []
        for s in sheep:
            if tuple(s) in pen and tuple(s) not in captured:
                captured.add(tuple(s))
                newly_captured.append(tuple(s))
        config["_captured_sheep"] = list(captured)
        # Remove newly captured sheep from grid so they don't block other sheep
        for sx, sy in newly_captured:
            if grid.objects[sy, sx] == ObjectType.SHEEP:
                grid.objects[sy, sx] = ObjectType.NONE
                grid.metadata[sy, sx] = 0
            # Restore TARGET marker for pen cell
            if (sx, sy) in pen:
                grid.objects[sy, sx] = ObjectType.TARGET

        if step_count % speed != 0:
            return

        self._draw_sheep(grid, sheep, draw=False)

        occupied = set(map(tuple, sheep))
        new_sheep = []
        for idx, (sx, sy) in enumerate(sheep):
            if tuple((sx, sy)) in captured:
                # Captured sheep stay put and don't occupy grid tiles
                new_sheep.append((sx, sy))
                continue

            dist_agent = abs(sx - ax) + abs(sy - ay)
            flee_from = None
            if dist_agent <= 2:
                flee_from = (ax, ay)

            if flee_from is not None:
                fx, fy = flee_from
                best = (sx, sy)
                best_d = abs(sx - fx) + abs(sy - fy)
                for dx, dy in self._DIRS:
                    nx, ny = sx + dx, sy + dy
                    d = abs(nx - fx) + abs(ny - fy)
                    if (
                        0 < nx < grid.width - 1
                        and 0 < ny < grid.height - 1
                        and grid.terrain[ny, nx] == CellType.EMPTY
                        and (nx, ny) not in occupied
                        and d > best_d
                    ):
                        best_d = d
                        best = (nx, ny)
                new_sheep.append(best)
            else:
                if rng.random() < 0.3:
                    moves = [(sx + dx, sy + dy) for dx, dy in self._DIRS]
                    valid = [
                        (x, y)
                        for x, y in moves
                        if (
                            0 < x < grid.width - 1
                            and 0 < y < grid.height - 1
                            and grid.terrain[y, x] == CellType.EMPTY
                            and (x, y) not in occupied
                        )
                    ]
                    if valid:
                        new_sheep.append(valid[int(rng.integers(len(valid)))])
                    else:
                        new_sheep.append((sx, sy))
                else:
                    new_sheep.append((sx, sy))

            occupied.discard((sx, sy))
            occupied.add(new_sheep[-1])

        # Store movement directions for directional sprites
        config["_sheep_dirs"] = []
        for idx2, (nx, ny) in enumerate(new_sheep):
            old_sx, old_sy = sheep[idx2] if idx2 < len(sheep) else (nx, ny)
            ddx, ddy = nx - old_sx, ny - old_sy
            if ddx > 0:
                config["_sheep_dirs"].append(1)  # right
            elif ddx < 0:
                config["_sheep_dirs"].append(3)  # left
            elif ddy < 0:
                config["_sheep_dirs"].append(0)  # up
            elif ddy > 0:
                config["_sheep_dirs"].append(2)  # down
            else:
                config["_sheep_dirs"].append(2)  # default down
        config["_live_sheep"] = new_sheep
        self._draw_sheep(grid, new_sheep, draw=True)

    def _draw_sheep(self, grid, sheep, draw: bool):
        config = getattr(self, "_config", {})
        pen = set(map(tuple, config.get("pen_cells", [])))
        captured = set(map(tuple, config.get("_captured_sheep", [])))
        sheep_dirs = config.get("_sheep_dirs", [])
        for idx, (sx, sy) in enumerate(sheep):
            if 0 <= sx < grid.width and 0 <= sy < grid.height:
                # Skip captured sheep — they've been removed from the grid
                if (sx, sy) in captured:
                    continue
                if draw:
                    grid.objects[sy, sx] = ObjectType.SHEEP
                    # Store direction in metadata for directional sprites
                    if idx < len(sheep_dirs):
                        grid.metadata[sy, sx] = sheep_dirs[idx]
                    else:
                        grid.metadata[sy, sx] = 2  # default down
                elif grid.objects[sy, sx] == ObjectType.SHEEP:
                    # Restore TARGET if this was a pen cell
                    if (sx, sy) in pen:
                        grid.objects[sy, sx] = ObjectType.TARGET
                    else:
                        grid.objects[sy, sx] = ObjectType.NONE
                    grid.metadata[sy, sx] = 0

    # ── Reward & success ─────────────────────────────────────────────────────

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        config = new_state.get("config", {})
        pen = set(map(tuple, config.get("pen_cells", [])))
        sheep = config.get("_live_sheep", [])
        if pen and sheep:
            # Continuous: reward sheep already in pen each step
            n_in_pen = sum(1 for s in sheep if tuple(s) in pen)
            reward += 0.05 * n_in_pen
            # Approach: agent toward nearest sheep not yet in pen
            out_sheep = [s for s in sheep if tuple(s) not in pen]
            if out_sheep and "agent_position" in new_state:
                ax, ay = new_state["agent_position"]
                ox, oy = old_state.get("agent_position", (ax, ay))
                ns_new = min(abs(ax - sx) + abs(ay - sy) for sx, sy in out_sheep)
                ns_old = min(abs(ox - sx) + abs(oy - sy) for sx, sy in out_sheep)
                reward += 0.02 * (ns_old - ns_new)
        if self.check_success(new_state):
            reward += 1.0
        return reward

    def check_success(self, state):
        """All sheep must be in the pen."""
        config = state.get("config", {})
        pen = set(map(tuple, config.get("pen_cells", [])))
        sheep = config.get("_live_sheep", config.get("sheep_positions", []))
        if not sheep or not pen:
            return False
        return all(tuple(s) in pen for s in sheep)

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0
