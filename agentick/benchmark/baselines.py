"""Baseline agents."""

import numpy as np


class RandomAgent:
    """Uniform random action selection."""

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, valid_actions):
        """Select random action from valid actions."""
        if valid_actions is None or len(valid_actions) == 0:
            return 0
        return self.rng.choice(valid_actions)

    def __call__(self, obs, info):
        """Select random action from valid actions."""
        valid_actions = info.get("valid_actions", [])
        if not valid_actions:
            return 0
        # Map action name to index
        return self.rng.integers(0, len(valid_actions))


class GreedyAgent:
    """Heuristic that moves toward nearest goal."""

    def __init__(self):
        self.rng = np.random.default_rng()

    def act(self, obs, valid_actions, state_dict=None):
        """Select action that moves toward goal."""
        # If state_dict provided, try to move toward goal
        if state_dict and "grid" in state_dict and "agent" in state_dict:
            agent_pos = state_dict["agent"]["position"]
            # Find goal in grid
            grid_objects = state_dict["grid"]["objects"]
            goal_pos = None
            for y, row in enumerate(grid_objects):
                for x, obj in enumerate(row):
                    if obj == 1:  # GOAL
                        goal_pos = (x, y)
                        break
                if goal_pos:
                    break

            if goal_pos:
                # Move toward goal
                dx = goal_pos[0] - agent_pos[0]
                dy = goal_pos[1] - agent_pos[1]

                # Try to move in direction of goal
                if abs(dy) > abs(dx):
                    if dy < 0 and 1 in valid_actions:
                        return 1  # MOVE_UP
                    elif dy > 0 and 2 in valid_actions:
                        return 2  # MOVE_DOWN
                else:
                    if dx < 0 and 3 in valid_actions:
                        return 3  # MOVE_LEFT
                    elif dx > 0 and 4 in valid_actions:
                        return 4  # MOVE_RIGHT

        # Fallback to random
        if valid_actions is not None and len(valid_actions) > 0:
            return self.rng.choice(valid_actions)
        return 0

    def __call__(self, obs, info):
        """Select action that moves toward goal."""
        # Simple greedy: just try to move right and down
        # (Assumes goal is typically in bottom-right)
        return self.rng.choice([2, 4])  # MOVE_DOWN or MOVE_RIGHT


class OracleAgent:
    """BFS/A* optimal solver supporting keys, doors, boxes, switches, and inventory."""

    def __init__(self, env=None):
        self.env = env
        self.plan = []

    def _compute_optimal_path(self):
        """Compute optimal path using sophisticated validation solver."""
        if not self.env:
            return None

        from agentick.core.types import ObjectType
        from agentick.generation.validation import find_optimal_path

        # Get current state
        agent_pos = self.env.agent.position
        grid = self.env.grid

        # Find goals
        goal_positions = []
        for y in range(grid.height):
            for x in range(grid.width):
                if grid.objects[y, x] == ObjectType.GOAL:
                    goal_positions.append((x, y))

        if not goal_positions:
            return None

        # Build config for complex mechanics
        config = {}

        # Find keys and doors
        keys = {}
        doors = {}
        for y in range(grid.height):
            for x in range(grid.width):
                if grid.objects[y, x] == ObjectType.KEY:
                    # Use object color if available
                    color = getattr(grid, "object_colors", np.zeros_like(grid.objects))[y, x]
                    keys[int(color)] = (x, y)
                elif grid.objects[y, x] == ObjectType.DOOR:
                    color = getattr(grid, "object_colors", np.zeros_like(grid.objects))[y, x]
                    doors[int(color)] = (x, y)

        if keys:
            config["keys"] = keys
        if doors:
            config["doors"] = doors

        # Find boxes
        boxes = []
        targets = []
        for y in range(grid.height):
            for x in range(grid.width):
                if grid.objects[y, x] == ObjectType.BOX:
                    boxes.append((x, y))
                elif grid.objects[y, x] == ObjectType.TARGET:
                    targets.append((x, y))

        if boxes:
            config["boxes"] = boxes
            config["targets"] = targets

        # Find switches
        switches = []
        for y in range(grid.height):
            for x in range(grid.width):
                if grid.objects[y, x] == ObjectType.SWITCH:
                    switches.append((x, y))

        if switches:
            config["switches"] = switches

        # Find collectible items (for inventory tasks)
        items = []
        for y in range(grid.height):
            for x in range(grid.width):
                if grid.objects[y, x] in (ObjectType.RESOURCE, ObjectType.TOOL):
                    items.append((x, y))

        if items:
            config["items"] = items

        # Compute optimal path
        path, _ = find_optimal_path(grid, agent_pos, goal_positions, config if config else None)
        return path

    def act(self, obs, valid_actions, state_dict=None):
        """Compute and follow optimal path."""
        if not self.plan and self.env:
            path = self._compute_optimal_path()
            if path and len(path) > 1:
                self.plan = path[1:]  # Exclude current position

        if self.plan and self.env:
            next_pos = self.plan.pop(0)
            agent_pos = self.env.agent.position

            # Determine direction
            dx = next_pos[0] - agent_pos[0]
            dy = next_pos[1] - agent_pos[1]

            if dy < 0:
                return 1  # MOVE_UP
            elif dy > 0:
                return 2  # MOVE_DOWN
            elif dx < 0:
                return 3  # MOVE_LEFT
            elif dx > 0:
                return 4  # MOVE_RIGHT

        return None  # Can't find path

    def __call__(self, obs, info):
        """Follow optimal path."""
        if not self.plan:
            path = self._compute_optimal_path()
            if path and len(path) > 1:
                self.plan = path[1:]  # Exclude current position

        if self.plan:
            next_pos = self.plan.pop(0)
            agent_pos = self.env.agent.position

            # Determine direction
            dx = next_pos[0] - agent_pos[0]
            dy = next_pos[1] - agent_pos[1]

            if dy < 0:
                return 1  # MOVE_UP
            elif dy > 0:
                return 2  # MOVE_DOWN
            elif dx < 0:
                return 3  # MOVE_LEFT
            elif dx > 0:
                return 4  # MOVE_RIGHT

        return 0  # NOOP
