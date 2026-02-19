"""TaskInterference - Multiple conflicting tasks with resource competition.

MECHANICS:
  - N sub-tasks (goal pairs: KEY item → matching GOAL delivery)
  - Picking up one KEY disables ability to pick up others (must deliver first)
  - Goals that move or become temporarily inaccessible when other goals are completed
  - Completing task A may block the direct path to task B (wall appears)
  - Limited step budget forces prioritization
  - Success = complete ALL sub-tasks despite interference
"""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("TaskInterference-v0", tags=["multi_task", "interference", "meta_learning"])
class TaskInterferenceTask(TaskSpec):
    """Complete multiple interfering sub-tasks: collect items and deliver to matching goals."""

    name = "TaskInterference-v0"
    description = "Complete interfering sub-tasks with resource competition"
    capability_tags = ["multi_task", "interference_resistance", "attention"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy", grid_size=9, max_steps=150,
            params={"n_tasks": 2, "interference": False},
        ),
        "medium": DifficultyConfig(
            name="medium", grid_size=11, max_steps=250,
            params={"n_tasks": 2, "interference": True},
        ),
        "hard": DifficultyConfig(
            name="hard", grid_size=13, max_steps=400,
            params={"n_tasks": 3, "interference": True},
        ),
        "expert": DifficultyConfig(
            name="expert", grid_size=15, max_steps=600,
            params={"n_tasks": 4, "interference": True},
        ),
    }

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        n_tasks = self.difficulty_config.params.get("n_tasks", 2)
        interference = self.difficulty_config.params.get("interference", False)

        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        agent_pos = (1, 1)

        free = [
            (x, y) for x in range(2, size - 2) for y in range(2, size - 2)
            if (x, y) != agent_pos
        ]
        rng.shuffle(free)

        # Place KEY items (collectibles) and GOAL delivery points
        # Each task: pick up KEY at position A, deliver to GOAL at position B
        key_positions = []
        goal_positions = []
        used = {agent_pos}

        for i in range(n_tasks):
            # Key position
            for pos in free:
                if pos not in used:
                    key_positions.append(pos)
                    used.add(pos)
                    break

            # Goal position (far from corresponding key for challenge)
            best_goal = None
            best_dist = 0
            for pos in free:
                if pos not in used:
                    d = abs(pos[0] - key_positions[-1][0]) + abs(pos[1] - key_positions[-1][1])
                    if d > best_dist:
                        best_dist = d
                        best_goal = pos
            if best_goal:
                goal_positions.append(best_goal)
                used.add(best_goal)

        # Place objects on grid
        for kx, ky in key_positions:
            grid.objects[ky, kx] = ObjectType.KEY
        for gx, gy in goal_positions:
            grid.objects[gy, gx] = ObjectType.GOAL

        # Interference walls: completing one task places a wall near another task's path
        interference_walls = []
        if interference and n_tasks >= 2:
            for i in range(min(n_tasks - 1, 3)):
                # Wall position between goal i and key i+1
                gi = goal_positions[i]
                ki = key_positions[(i + 1) % n_tasks]
                wx = (gi[0] + ki[0]) // 2
                wy = (gi[1] + ki[1]) // 2
                wx = max(1, min(size - 2, wx))
                wy = max(1, min(size - 2, wy))
                if (wx, wy) not in used and grid.terrain[wy, wx] == CellType.EMPTY:
                    interference_walls.append((wx, wy, i))

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": goal_positions,
            "key_positions": key_positions,
            "n_tasks": n_tasks,
            "interference": interference,
            "interference_walls": interference_walls,
            "max_steps": self.get_max_steps(),
        }

    def on_env_reset(self, agent, grid, config):
        config["_carrying"] = -1  # index of key being carried (-1 = none)
        config["_tasks_completed"] = set()
        config["_keys_collected"] = set()
        self._tasks_completed_last = 0
        self._config = config

    def on_agent_moved(self, pos, agent, grid):
        config = getattr(self, "_config", {})
        x, y = pos
        keys = config.get("key_positions", [])
        goals = config.get("goal_positions", [])
        carrying = config.get("_carrying", -1)
        completed = config.get("_tasks_completed", set())
        collected = config.get("_keys_collected", set())

        # Pick up key (only if not already carrying one)
        if carrying < 0:
            for i, (kx, ky) in enumerate(keys):
                if (x, y) == (kx, ky) and i not in collected:
                    if grid.objects[y, x] == ObjectType.KEY:
                        grid.objects[y, x] = ObjectType.NONE
                        config["_carrying"] = i
                        collected.add(i)
                        config["_keys_collected"] = collected
                        break

        # Deliver to matching goal
        elif carrying >= 0 and carrying < len(goals):
            gx, gy = goals[carrying]
            if (x, y) == (gx, gy):
                if grid.objects[y, x] == ObjectType.GOAL:
                    grid.objects[y, x] = ObjectType.NONE  # goal consumed
                    completed.add(carrying)
                    config["_tasks_completed"] = completed
                    config["_carrying"] = -1

                    # Interference: place wall if configured
                    if config.get("interference", False):
                        for wx, wy, trigger_task in config.get("interference_walls", []):
                            if trigger_task == carrying:
                                grid.terrain[wy, wx] = CellType.WALL

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        config = new_state.get("config", {})
        completed = config.get("_tasks_completed", set())
        n_done = len(completed)

        # Reward per task completed
        if n_done > self._tasks_completed_last:
            reward += 0.5 * (n_done - self._tasks_completed_last)
        self._tasks_completed_last = n_done

        # Approach shaping
        if "agent_position" in new_state and "grid" in new_state:
            ax, ay = new_state["agent_position"]
            ox, oy = old_state.get("agent_position", (ax, ay))
            carrying = config.get("_carrying", -1)
            keys = config.get("key_positions", [])
            goals = config.get("goal_positions", [])
            collected = config.get("_keys_collected", set())

            if carrying >= 0 and carrying < len(goals):
                # Guide toward delivery goal
                gx, gy = goals[carrying]
                d_new = abs(ax - gx) + abs(ay - gy)
                d_old = abs(ox - gx) + abs(oy - gy)
                reward += 0.05 * (d_old - d_new)
            else:
                # Guide toward nearest uncollected key
                uncollected = [
                    keys[i] for i in range(len(keys))
                    if i not in collected and i not in completed
                ]
                if uncollected:
                    d_new = min(abs(ax - kx) + abs(ay - ky) for kx, ky in uncollected)
                    d_old = min(abs(ox - kx) + abs(oy - ky) for kx, ky in uncollected)
                    reward += 0.05 * (d_old - d_new)

        if self.check_success(new_state):
            reward += 1.0
        return reward

    def check_success(self, state):
        config = state.get("config", {})
        if "grid" not in state or "agent" not in state:
            return False
        completed = config.get("_tasks_completed", set())
        n_tasks = config.get("n_tasks", 2)
        return len(completed) >= n_tasks

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0
