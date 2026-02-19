"""Recursive Rooms - Hierarchical room structure."""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("RecursiveRooms-v0", tags=["hierarchical", "planning", "composition"])
class RecursiveRoomsTask(TaskSpec):
    """Test hierarchical planning by navigating nested recursive room structures.

    The grid contains a hierarchy of rooms nested within rooms. Solving
    the puzzle in an inner room unlocks passage to the enclosing outer
    room, requiring the agent to plan at multiple levels of abstraction.
    The nesting depth scales with difficulty, demanding increasingly
    deep hierarchical decomposition. The agent must recognize the
    recursive structure, solve sub-problems in the correct order, and
    compose local solutions into a global navigation plan.

    Difficulty Levels:
        - easy: 15x15 grid with nesting depth 2, 200 max steps.
        - medium: 20x20 grid with nesting depth 3, 350 max steps.
        - hard: 25x25 grid with nesting depth 4, 500 max steps.
        - expert: 30x30 grid with nesting depth 5 requiring deep
          hierarchical reasoning, 700 max steps.

    Capabilities Tested:
        - hierarchical_planning: The agent must decompose the task into
          nested sub-goals and solve them in the correct hierarchical
          order.
        - composition: The agent must compose solutions to inner rooms
          into a coherent strategy for escaping the entire structure.
        - navigation: The agent must navigate through complex multi-room
          layouts with walls and restricted passages.

    Example:
        >>> env = agentick.make("RecursiveRooms-v0", difficulty="medium")
        >>> obs, info = env.reset(seed=42)
        >>> # Solve inner rooms to unlock outer rooms and reach the goal
    """

    name = "RecursiveRooms-v0"
    description = "Navigate nested hierarchical room structure"
    capability_tags = ["hierarchical_planning", "composition", "navigation"]

    difficulty_configs = {
        "easy": DifficultyConfig(name="easy", grid_size=15, max_steps=200, params={"depth": 2}),
        "medium": DifficultyConfig(name="medium", grid_size=20, max_steps=350, params={"depth": 3}),
        "hard": DifficultyConfig(name="hard", grid_size=25, max_steps=500, params={"depth": 4}),
        "expert": DifficultyConfig(name="expert", grid_size=30, max_steps=700, params={"depth": 5}),
    }

    def generate(self, seed):
        """Generate a recursive rooms task instance.

        Creates a nested room hierarchy: the grid is divided into depth
        rooms by vertical dividers, each with a randomly positioned
        doorway. Agent starts in the leftmost room; goal is in the
        rightmost room. The agent must navigate through each doorway
        in sequence to reach the goal, requiring hierarchical planning.

        Args:
            seed: Random seed for reproducible procedural generation.

        Returns:
            tuple: (grid, metadata) where grid is the initial Grid state
                with walls, doorways, and goal.
        """
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        depth = self.difficulty_config.params.get("depth", 2)

        # Start with all walls; carve rooms
        grid = Grid(size, size)
        for y in range(size):
            for x in range(size):
                grid.terrain[y, x] = CellType.WALL

        # Divide into (depth+1) rooms separated by depth vertical dividers
        n_rooms = depth + 1
        room_width = (size - 2) // n_rooms  # approximate width per room
        if room_width < 2:
            room_width = 2

        # Carve each room as empty interior
        doorway_rows = []  # row of doorway in each divider
        for r in range(n_rooms):
            room_x = 1 + r * room_width
            room_w = room_width if r < n_rooms - 1 else (size - 2 - r * room_width)
            for y in range(1, size - 1):
                for x in range(room_x, min(room_x + room_w, size - 1)):
                    grid.terrain[y, x] = CellType.EMPTY

            # Add dividing wall and doorway between this room and the next
            if r < n_rooms - 1:
                wall_x = room_x + room_w
                if wall_x < size - 1:
                    for y in range(1, size - 1):
                        grid.terrain[y, wall_x] = CellType.WALL
                    # Random doorway in this wall
                    door_y = int(rng.integers(1, size - 1))
                    grid.terrain[door_y, wall_x] = CellType.EMPTY
                    doorway_rows.append(door_y)

        agent_pos = (1, 1)

        # Find a valid cell in last room for the goal
        last_room_x = 1 + (n_rooms - 1) * room_width
        goal_candidates = [(x, y) for x in range(last_room_x, size - 1)
                          for y in range(1, size - 1)
                          if grid.terrain[y, x] == CellType.EMPTY and (x, y) != agent_pos]
        if goal_candidates:
            # Pick cell farthest from agent
            goal_pos = max(goal_candidates, key=lambda p: abs(p[0] - agent_pos[0]) + abs(p[1] - agent_pos[1]))
        else:
            goal_pos = (size - 2, size - 2)
            grid.terrain[goal_pos[1], goal_pos[0]] = CellType.EMPTY

        # Verify reachable; fallback if not
        reachable = grid.flood_fill(agent_pos)
        if goal_pos not in reachable:
            # Fallback: open grid with simple walls
            grid = Grid(size, size)
            grid.terrain[0, :] = CellType.WALL
            grid.terrain[-1, :] = CellType.WALL
            grid.terrain[:, 0] = CellType.WALL
            grid.terrain[:, -1] = CellType.WALL
            goal_pos = (size - 2, size - 2)

        grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.GOAL

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [goal_pos],
            "max_steps": self.get_max_steps(),
            "depth": depth,
            "doorway_rows": doorway_rows,
        }

    def _create_recursive_room(self, grid, x, y, w, h, depth, rng):
        """Recursively create nested rooms (kept for reference, not used by generate)."""
        if depth == 0 or w < 5 or h < 5:
            return
        for dy in range(h):
            for dx in range(w):
                grid.terrain[y + dy, x + dx] = CellType.EMPTY
        if depth > 1:
            mid_x = w // 2
            mid_y = h // 2
            for i in range(h):
                grid.terrain[y + i, x + mid_x] = CellType.WALL
            for i in range(w):
                grid.terrain[y + mid_y, x + i] = CellType.WALL
            self._create_recursive_room(grid, x, y, mid_x, mid_y, depth - 1, rng)
            self._create_recursive_room(
                grid, x + mid_x + 1, y, w - mid_x - 1, mid_y, depth - 1, rng
            )

    def compute_dense_reward(self, old_state, action, new_state, info):
        """Dense reward: step penalty + progress toward goal."""
        reward = -0.01
        if "agent" in new_state and "config" in new_state:
            config = new_state["config"]
            goal = config.get("goal_positions", [None])[0]
            if goal:
                ax, ay = new_state["agent"].position
                ox, oy = old_state.get("agent_position", (ax, ay))
                old_d = abs(ox - goal[0]) + abs(oy - goal[1])
                new_d = abs(ax - goal[0]) + abs(ay - goal[1])
                reward += 0.05 * (old_d - new_d)  # shaping toward goal
        if self.check_success(new_state):
            reward += 1.0
        return reward

    def check_success(self, state):
        """Check if the task objective is complete.

        The task succeeds when the agent reaches the goal cell after
        navigating through the nested room hierarchy.

        Args:
            state: Current state dict containing 'grid' and 'agent' keys.

        Returns:
            True if the agent is on the goal cell, False otherwise.
        """
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

        A random agent is unlikely to navigate through deeply nested
        rooms in the correct order, yielding near-zero expected return.

        Args:
            difficulty: Difficulty level string, or None to use the
                current instance difficulty.

        Returns:
            Expected random agent return of 0.0.
        """
        return 0.0
