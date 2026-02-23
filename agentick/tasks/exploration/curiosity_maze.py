"""CuriosityMaze task - Explore a maze to discover and visit all landmarks."""

from __future__ import annotations

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("CuriosityMaze-v0", tags=["exploration", "memory", "navigation"])
class CuriosityMazeTask(TaskSpec):
    """Explore a maze to discover and visit all landmark objects.

    The agent is placed in a procedurally generated maze containing
    scattered LANDMARK objects (SWITCH, SCROLL, ORB, COIN). The agent
    must explore the maze and step on every landmark. A GOAL appears once
    all landmarks have been visited. No clues are given about landmark
    locations — pure exploration and coverage is required.

    Rewards exploration: +0.15 per landmark discovered, +1.0 for completing
    all. Step penalty -0.01 discourages loitering.

    Difficulty scales maze size, landmark count, and maze complexity.
    """

    name = "CuriosityMaze-v0"
    description = "Explore maze to discover and visit all hidden landmarks"
    capability_tags = ["exploration", "memory", "navigation"]

    _LANDMARK_TYPES = [
        ObjectType.SWITCH,
        ObjectType.SCROLL,
        ObjectType.ORB,
        ObjectType.COIN,
    ]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=9,
            max_steps=150,
            params={"n_landmarks": 3, "wall_density": 0.15},
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=13,
            max_steps=300,
            params={"n_landmarks": 5, "wall_density": 0.18},
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=17,
            max_steps=500,
            params={"n_landmarks": 7, "wall_density": 0.20},
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=21,
            max_steps=800,
            params={"n_landmarks": 10, "wall_density": 0.22},
        ),
    }

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        p = self.difficulty_config.params or {}
        n_landmarks = p.get("n_landmarks", 3)
        wall_density = p.get("wall_density", 0.15)

        for _attempt in range(20):
            grid = Grid(size, size)
            # Outer walls
            grid.terrain[0, :] = CellType.WALL
            grid.terrain[-1, :] = CellType.WALL
            grid.terrain[:, 0] = CellType.WALL
            grid.terrain[:, -1] = CellType.WALL

            # Generate maze-like walls via random scatter
            n_walls = int((size - 2) ** 2 * wall_density)
            for _ in range(n_walls):
                wx = int(rng.integers(1, size - 1))
                wy = int(rng.integers(1, size - 1))
                grid.terrain[wy, wx] = CellType.WALL

            # Agent at center
            cx, cy = size // 2, size // 2
            grid.terrain[cy, cx] = CellType.EMPTY
            agent_pos = (cx, cy)

            # Find reachable positions
            reachable = grid.flood_fill(agent_pos)
            reachable_list = list(reachable - {agent_pos})
            if len(reachable_list) < n_landmarks + 2:
                continue

            # Distribute landmarks far from agent (sort by distance, pick spread)
            reachable_list.sort(
                key=lambda p: abs(p[0] - cx) + abs(p[1] - cy), reverse=True
            )
            # Pick every Nth to spread landmarks
            step = max(1, len(reachable_list) // (n_landmarks + 1))
            landmark_positions = []
            for i in range(n_landmarks):
                idx = i * step
                if idx >= len(reachable_list):
                    idx = len(reachable_list) - 1
                lx, ly = reachable_list[idx]
                lx, ly = int(lx), int(ly)
                ltype = self._LANDMARK_TYPES[i % len(self._LANDMARK_TYPES)]
                grid.objects[ly, lx] = ltype
                grid.metadata[ly, lx] = i + 1  # landmark ID
                landmark_positions.append((lx, ly))

            return grid, {
                "agent_start": agent_pos,
                "goal_positions": landmark_positions,
                "_landmark_positions": landmark_positions,
                "_visited_landmarks": [],
                "max_steps": self.get_max_steps(),
            }

        # Fallback
        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL
        agent_pos = (1, 1)
        grid.objects[size - 2, size - 2] = ObjectType.SWITCH
        grid.metadata[size - 2, size - 2] = 1
        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [(size - 2, size - 2)],
            "_landmark_positions": [(size - 2, size - 2)],
            "_visited_landmarks": [],
            "max_steps": self.get_max_steps(),
        }

    def on_env_reset(self, agent, grid, config):
        config["_visited_landmarks"] = []
        self._config = config

    def on_agent_moved(self, pos, agent, grid):
        """Check if agent stepped on a landmark."""
        x, y = pos
        config = getattr(self, "_config", None)
        if config is None:
            return
        landmark_positions = config.get("_landmark_positions", [])
        visited = config.get("_visited_landmarks", [])
        if (x, y) in landmark_positions and (x, y) not in visited:
            visited.append((x, y))
            config["_visited_landmarks"] = visited
            # Remove the landmark object (mark as visited)
            grid.objects[y, x] = ObjectType.NONE
            grid.metadata[y, x] = 0
            # If all visited, place GOAL at agent position to end
            if len(visited) >= len(landmark_positions):
                # Place GOAL at last landmark for immediate success check
                grid.objects[y, x] = ObjectType.GOAL

    def on_env_step(self, agent, grid, config, step_count):
        self._config = config

    def check_success(self, state):
        if "config" not in state:
            return False
        config = state["config"]
        landmarks = config.get("_landmark_positions", [])
        visited = config.get("_visited_landmarks", [])
        return len(visited) >= len(landmarks) and len(landmarks) > 0

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        # Landmark discovery bonus
        old_visited = len(old_state.get("config", {}).get("_visited_landmarks", []))
        new_visited = len(new_state.get("config", {}).get("_visited_landmarks", []))
        if new_visited > old_visited:
            reward += 0.15

        # Distance shaping to nearest unvisited landmark
        if "agent" in new_state and "config" in new_state:
            config = new_state["config"]
            landmarks = config.get("_landmark_positions", [])
            visited = config.get("_visited_landmarks", [])
            remaining = [l for l in landmarks if l not in visited]
            if remaining:
                ax, ay = new_state["agent"].position
                ox, oy = old_state.get("agent_position", (ax, ay))
                old_min = min(abs(ox - l[0]) + abs(oy - l[1]) for l in remaining)
                new_min = min(abs(ax - l[0]) + abs(ay - l[1]) for l in remaining)
                reward += 0.02 * (old_min - new_min)

        if self.check_success(new_state):
            reward += 1.0
        return reward

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0
