"""TimingChallenge - Time your crossing through a gap in a moving barrier.

MECHANICS:
  - A horizontal wall divides the grid; there is one GAP cell
  - A HAZARD blocker oscillates across the gap and neighboring cells
  - Agent must cross when the blocker is NOT in the gap
  - Stepping into the blocker's cell ends episode with penalty
  - Success = reach GOAL on the other side
"""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("TimingChallenge-v0", tags=["motor_control", "temporal_reasoning"])
class TimingChallengeTask(TaskSpec):
    """Time your crossing through the gap to avoid the moving blocker."""

    name = "TimingChallenge-v0"
    description = "Time your crossing through the moving blocker"
    capability_tags = ["motor_control", "temporal_reasoning"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=7,
            max_steps=60,
            params={
                "patrol_len": 3,
                "n_gaps": 1,
                "n_blockers": 1,
                "gap_rand": True,
                "blocker_speed_var": 0,
                "n_safe_spots": 0,
            },
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=9,
            max_steps=120,
            params={
                "patrol_len": 4,
                "n_gaps": 1,
                "n_blockers": 2,
                "gap_rand": True,
                "blocker_speed_var": 0,
                "n_safe_spots": 0,
            },
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=11,
            max_steps=200,
            params={
                "patrol_len": 5,
                "n_gaps": 2,
                "n_blockers": 2,
                "gap_rand": True,
                "blocker_speed_var": 1,
                "n_safe_spots": 1,
            },
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=13,
            max_steps=300,
            params={
                "patrol_len": 6,
                "n_gaps": 2,
                "n_blockers": 3,
                "gap_rand": True,
                "blocker_speed_var": 2,
                "n_safe_spots": 2,
            },
        ),
    }

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        p = self.difficulty_config.params
        patrol_len = p.get("patrol_len", 3)
        n_gaps = p.get("n_gaps", 1)
        n_blockers = p.get("n_blockers", 1)
        gap_rand = p.get("gap_rand", False)
        blocker_speed_var = p.get("blocker_speed_var", 0)
        n_safe_spots = p.get("n_safe_spots", 0)

        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        mid_row = size // 2
        if gap_rand:
            mid_row = int(rng.integers(size // 3, 2 * size // 3))
        mid_row = max(2, min(size - 3, mid_row))

        for x in range(1, size - 1):
            grid.terrain[mid_row, x] = CellType.WALL

        gap_cols = []
        if n_gaps == 1:
            gc = size // 2
            if gap_rand:
                gc = int(rng.integers(2, size - 2))
            gap_cols = [gc]
        else:
            gc1 = max(2, size // 3)
            gc2 = min(size - 3, 2 * size // 3)
            if gap_rand:
                gc1 = int(rng.integers(2, size // 2))
                gc2 = int(rng.integers(size // 2, size - 2))
            gap_cols = [gc1, gc2]

        for gc in gap_cols:
            grid.terrain[mid_row, gc] = CellType.EMPTY

        agent_pos = (gap_cols[0], max(1, mid_row - 2))
        goal_pos = (gap_cols[-1], min(size - 2, mid_row + 2))
        grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.GOAL

        # Place safe spots (alcoves near gaps on agent side)
        safe_positions = []
        if n_safe_spots > 0:
            safe_cands = []
            for gc in gap_cols:
                for dx in [-1, 1]:
                    sx = gc + dx
                    sy = mid_row - 1
                    if (
                        1 <= sx < size - 1
                        and 1 <= sy < size - 1
                        and grid.terrain[sy, sx] == CellType.WALL
                    ):
                        safe_cands.append((sx, sy))
            rng.shuffle(safe_cands)
            for sp in safe_cands[:n_safe_spots]:
                sx, sy = sp
                grid.terrain[sy, sx] = CellType.EMPTY
                safe_positions.append(sp)

        blocker_specs = []
        for b_idx, gc in enumerate(gap_cols):
            ps = max(1, gc - patrol_len // 2)
            pe = min(size - 2, ps + patrol_len - 1)
            for bi in range(n_blockers):
                start_x = int(rng.integers(ps, pe + 1))
                speed = 1
                if blocker_speed_var > 0:
                    speed = 1 + int(rng.integers(0, blocker_speed_var + 1))
                blocker_specs.append(
                    {
                        "x": start_x,
                        "row": mid_row,
                        "dir": 1,
                        "p0": ps,
                        "p1": pe,
                        "gap": gc,
                        "speed": speed,
                    }
                )

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [goal_pos],
            "mid_row": mid_row,
            "gap_cols": gap_cols,
            "gap_col": gap_cols[0],
            "patrol_start": (blocker_specs[0]["p0"] if blocker_specs else 1),
            "patrol_end": (blocker_specs[0]["p1"] if blocker_specs else 3),
            "_blocker_specs": blocker_specs,
            "safe_positions": safe_positions,
            "max_steps": self.get_max_steps(),
        }

    def on_env_reset(self, agent, grid, config):
        config["_collision"] = False
        specs = config.get("_blocker_specs", [])
        for i, s in enumerate(specs):
            config[f"_bx_{i}"] = s["x"]
            config[f"_bdir_{i}"] = s["dir"]
            if grid.terrain[s["row"], s["x"]] == CellType.EMPTY:
                grid.objects[s["row"], s["x"]] = ObjectType.ENEMY
                grid.metadata[s["row"], s["x"]] = 1 if s["dir"] > 0 else 3  # right or left

    def on_env_step(self, agent, grid, config, step_count):
        specs = config.get("_blocker_specs", [])
        ax, ay = agent.position
        for i, s in enumerate(specs):
            speed = s.get("speed", 1)
            if step_count % speed != 0:
                continue
            bx = config.get(f"_bx_{i}", s["x"])
            d = config.get(f"_bdir_{i}", 1)
            by = s["row"]
            p0, p1 = s["p0"], s["p1"]
            if grid.objects[by, bx] == ObjectType.ENEMY:
                grid.objects[by, bx] = ObjectType.NONE
                grid.metadata[by, bx] = 0
            new_x = bx + d
            if new_x > p1:
                d = -1
                new_x = bx - 1
            elif new_x < p0:
                d = 1
                new_x = bx + 1
            new_x = max(p0, min(p1, new_x))
            config[f"_bx_{i}"] = new_x
            config[f"_bdir_{i}"] = d
            grid.objects[by, new_x] = ObjectType.ENEMY
            grid.metadata[by, new_x] = 1 if d > 0 else 3  # right or left
            if ay == by and ax == new_x:
                config["_collision"] = True

    def compute_sparse_reward(self, old_state, action, new_state, info):
        if new_state.get("config", {}).get("_collision", False):
            return -0.5
        if self.check_success(new_state):
            return 1.0
        return 0.0

    def compute_dense_reward(self, old_state, action, new_state, info):
        if new_state.get("config", {}).get("_collision", False):
            return -0.5
        reward = -0.01
        # Shaping toward goal
        config = new_state.get("config", {})
        goal = config.get("goal_positions", [None])[0]
        if goal and "agent_position" in new_state:
            ax, ay = new_state["agent_position"]
            ox, oy = old_state.get("agent_position", (ax, ay))
            old_d = abs(ox - goal[0]) + abs(oy - goal[1])
            new_d = abs(ax - goal[0]) + abs(ay - goal[1])
            reward += 0.05 * (old_d - new_d)
        if self.check_success(new_state):
            reward += 1.0
        return reward

    def check_done(self, state):
        """Episode ends on collision OR reaching goal."""
        config = state.get("config", {})
        if config.get("_collision", False):
            return True
        return self.check_success(state)

    def check_success(self, state):
        """Success ONLY if agent reached goal (not collision)."""
        config = state.get("config", {})
        if config.get("_collision", False):
            return False  # collision = done but NOT success
        if "grid" not in state or "agent" not in state:
            return False
        x, y = state["agent"].position
        return bool(state["grid"].objects[y, x] == ObjectType.GOAL)

    def validate_instance(self, grid, config):
        agent_pos = tuple(config.get("agent_start", (1, 1)))
        goal_positions = config.get("goal_positions", [])
        reachable = grid.flood_fill(agent_pos)
        for gp in goal_positions:
            if tuple(gp) not in reachable:
                return False
        return True

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return -0.5
