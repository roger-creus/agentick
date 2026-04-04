"""BacktrackPuzzle - Reach the switch past the goal, then backtrack to claim it.

MECHANICS:
  - Agent must walk PAST the goal (locked) to reach the switch
  - Activating the switch unlocks the goal
  - Agent must BACKTRACK to the now-unlocked goal
  - Multiple layout variants: L-shaped, T-shaped, branching corridors
  - Dead ends and multiple paths at higher difficulties
  - Switch positions randomized (not always at corridor end)
"""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("BacktrackPuzzle-v0", tags=["memory", "planning"])
class BacktrackPuzzleTask(TaskSpec):
    """Activate a switch past the goal, then backtrack to claim it."""

    name = "BacktrackPuzzle-v0"
    description = "Overshoot goal to hit switch, then backtrack"
    capability_tags = ["memory", "planning"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=9,
            max_steps=80,
            params={"n_switches": 1, "n_dead_ends": 0},
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=11,
            max_steps=160,
            params={"n_switches": 2, "n_dead_ends": 1},
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=13,
            max_steps=280,
            params={"n_switches": 3, "n_dead_ends": 2},
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=15,
            max_steps=450,
            params={"n_switches": 4, "n_dead_ends": 3},
        ),
    }

    def _layout_L(self, grid, size, rng):
        """L-shaped corridor layout."""
        # Horizontal corridor at top third
        row1 = int(rng.integers(2, max(3, size // 3)))
        for x in range(1, size - 1):
            grid.terrain[row1, x] = CellType.EMPTY

        # Vertical corridor at right side connecting to horizontal at bottom
        col1 = size - 2 - int(rng.integers(0, 2))
        row2 = size - 2 - int(rng.integers(0, max(1, size // 4)))
        for y in range(row1, row2 + 1):
            grid.terrain[y, col1] = CellType.EMPTY

        # Bottom horizontal from col1 back left
        for x in range(1, col1 + 1):
            grid.terrain[row2, x] = CellType.EMPTY

        return (1, row1), row1, row2, col1

    def _layout_T(self, grid, size, rng):
        """T-shaped corridor layout."""
        # Main horizontal corridor
        row1 = size // 2
        for x in range(1, size - 1):
            grid.terrain[row1, x] = CellType.EMPTY

        # Vertical stem going down from middle
        col1 = size // 2
        for y in range(row1, size - 1):
            grid.terrain[y, col1] = CellType.EMPTY

        # Short branches at top of T
        branch_y = max(1, row1 - int(rng.integers(1, max(2, size // 4))))
        for y in range(branch_y, row1):
            grid.terrain[y, 1] = CellType.EMPTY
            grid.terrain[y, size - 2] = CellType.EMPTY

        return (col1, row1), row1, size - 2, col1

    def _layout_zigzag(self, grid, size, rng):
        """Zigzag corridor layout."""
        rows = []
        n_zigs = 2 + int(rng.integers(0, 2))
        row_spacing = max(2, (size - 2) // (n_zigs + 1))

        for i in range(n_zigs + 1):
            row = 1 + i * row_spacing
            row = min(row, size - 2)
            rows.append(row)
            for x in range(1, size - 1):
                grid.terrain[row, x] = CellType.EMPTY

        # Connect rows with vertical segments alternating left/right
        for i in range(len(rows) - 1):
            col = 1 if i % 2 == 0 else size - 2
            for y in range(rows[i], rows[i + 1] + 1):
                grid.terrain[y, col] = CellType.EMPTY

        return (size // 2, rows[0]), rows[0], rows[-1], size // 2

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        n_sw = self.difficulty_config.params.get("n_switches", 1)
        n_dead = self.difficulty_config.params.get("n_dead_ends", 0)

        for attempt in range(15):
            grid = Grid(size, size)
            grid.terrain[:, :] = CellType.WALL

            # Pick random layout
            layout_choice = int(rng.integers(0, 3))
            if layout_choice == 0:
                start, _, _, _ = self._layout_L(grid, size, rng)
            elif layout_choice == 1:
                start, _, _, _ = self._layout_T(grid, size, rng)
            else:
                start, _, _, _ = self._layout_zigzag(grid, size, rng)

            agent_pos = start

            # Ensure agent start is on empty terrain
            if grid.terrain[agent_pos[1], agent_pos[0]] != CellType.EMPTY:
                for y in range(1, size - 1):
                    for x in range(1, size - 1):
                        if grid.terrain[y, x] == CellType.EMPTY:
                            agent_pos = (x, y)
                            break
                    if grid.terrain[agent_pos[1], agent_pos[0]] == CellType.EMPTY:
                        break

            # Add dead-end branches
            empty_cells = [
                (x, y)
                for x in range(1, size - 1)
                for y in range(1, size - 1)
                if grid.terrain[y, x] == CellType.EMPTY
            ]
            dead_end_cells = []
            for _ in range(n_dead):
                if not empty_cells:
                    break
                # Pick a random corridor cell and extend a dead end
                rng.shuffle(empty_cells)
                for ex, ey in empty_cells:
                    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                    rng.shuffle(dirs)
                    for dx, dy in dirs:
                        nx, ny = ex + dx, ey + dy
                        n2x, n2y = ex + 2 * dx, ey + 2 * dy
                        if (
                            1 <= nx <= size - 2
                            and 1 <= ny <= size - 2
                            and grid.terrain[ny, nx] == CellType.WALL
                            and 1 <= n2x <= size - 2
                            and 1 <= n2y <= size - 2
                            and grid.terrain[n2y, n2x] == CellType.WALL
                        ):
                            grid.terrain[ny, nx] = CellType.EMPTY
                            grid.terrain[n2y, n2x] = CellType.EMPTY
                            dead_end_cells.append((n2x, n2y))
                            empty_cells.append((nx, ny))
                            empty_cells.append((n2x, n2y))
                            break
                    else:
                        continue
                    break

            # Find all reachable cells and compute distances from agent
            from collections import deque

            dist_from_agent = {agent_pos: 0}
            queue = deque([agent_pos])
            while queue:
                cx, cy = queue.popleft()
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nx, ny = cx + dx, cy + dy
                    if (
                        0 < nx < size - 1
                        and 0 < ny < size - 1
                        and (nx, ny) not in dist_from_agent
                        and grid.terrain[ny, nx] == CellType.EMPTY
                    ):
                        dist_from_agent[(nx, ny)] = dist_from_agent[(cx, cy)] + 1
                        queue.append((nx, ny))

            reachable = sorted(dist_from_agent.keys(), key=lambda p: dist_from_agent[p])
            if len(reachable) < n_sw + 3:
                continue

            # Goal: roughly 1/3 of the way along the longest path
            goal_idx = max(2, len(reachable) // 3)
            goal_pos = reachable[goal_idx]

            # Gate: one step before goal (closer to agent)
            gate_pos = None
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = goal_pos[0] + dx, goal_pos[1] + dy
                if (nx, ny) in dist_from_agent and dist_from_agent[(nx, ny)] < dist_from_agent[
                    goal_pos
                ]:
                    gate_pos = (nx, ny)
                    break
            if not gate_pos:
                gate_pos = reachable[max(1, goal_idx - 1)]
            grid.terrain[gate_pos[1], gate_pos[0]] = CellType.WALL

            # Switches: placed PAST the goal (farther from agent than goal)
            switch_candidates = [
                p
                for p in reachable
                if dist_from_agent.get(p, 0) > dist_from_agent.get(goal_pos, 0) + 1
                and p != agent_pos
                and p != goal_pos
                and p != gate_pos
            ]

            # Re-compute reachability after gate placement
            reachable_after = grid.flood_fill(agent_pos)
            switch_candidates = [p for p in switch_candidates if p in reachable_after]

            if len(switch_candidates) < n_sw:
                # Also consider dead-end cells
                switch_candidates.extend(
                    [
                        p
                        for p in dead_end_cells
                        if p in reachable_after
                        and p not in switch_candidates
                        and p != agent_pos
                        and p != goal_pos
                        and p != gate_pos
                    ]
                )

            if len(switch_candidates) < n_sw:
                continue

            rng.shuffle(switch_candidates)
            switch_positions = switch_candidates[:n_sw]

            # Place switch objects
            for sx, sy in switch_positions:
                grid.objects[sy, sx] = ObjectType.SWITCH

            # Verify: agent can reach all switches
            if not all(sw in reachable_after for sw in switch_positions):
                continue

            return grid, {
                "agent_start": agent_pos,
                "goal_positions": [goal_pos],
                "switch_positions": switch_positions,
                "switch_pos": switch_positions[0] if switch_positions else None,
                "gate_pos": gate_pos,
                "n_switches": n_sw,
                "max_steps": self.get_max_steps(),
            }

        # Fallback: guaranteed simple layout
        grid = Grid(size, size)
        grid.terrain[:, :] = CellType.WALL
        mid = size // 2
        byp = mid - 1
        for x in range(1, size - 1):
            grid.terrain[mid, x] = CellType.EMPTY
            grid.terrain[byp, x] = CellType.EMPTY
        gate_col = size // 3
        goal_col = gate_col + 1
        grid.terrain[mid, gate_col] = CellType.WALL
        agent_pos = (1, mid)
        goal_pos = (goal_col, mid)
        gate_pos = (gate_col, mid)
        switch_positions = [(size - 2, byp)]
        grid.objects[byp, size - 2] = ObjectType.SWITCH
        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [goal_pos],
            "switch_positions": switch_positions,
            "switch_pos": switch_positions[0],
            "gate_pos": gate_pos,
            "n_switches": 1,
            "max_steps": self.get_max_steps(),
        }

    def on_env_reset(self, agent, grid, config):
        config["_switches_activated"] = 0
        config["_all_activated"] = False
        self._switch_milestone_given = False
        self._config = config

    def can_agent_enter(self, pos, agent, grid) -> bool:
        """Allow walking through activated switches (metadata >= 100).

        Switches are solid by default, but once activated they become passable
        so the agent can continue through corridors to reach other switches.
        """
        x, y = pos
        if grid.is_object_blocking(pos):
            # Activated switches are passable
            if (
                grid.objects[y, x] == ObjectType.SWITCH
                and int(grid.metadata[y, x]) >= 100
            ):
                return True
            return False
        return True

    def on_agent_interact(self, pos, agent, grid):
        """INTERACT while facing a SWITCH activates it."""
        config = getattr(self, "_config", {})
        ax, ay = pos
        switches = config.get("switch_positions", [])

        for sw in list(switches):
            sx, sy = sw
            if (ax, ay) == (sx, sy) and grid.objects[sy, sx] == ObjectType.SWITCH:
                if grid.metadata[sy, sx] < 100:
                    # Keep switch object, mark as activated via metadata (renders as switch_on)
                    grid.metadata[sy, sx] = 100
                    config["_switches_activated"] = config.get("_switches_activated", 0) + 1
                    config["_switch_activated"] = True

        n_needed = config.get("n_switches", 1)
        if config.get("_switches_activated", 0) >= n_needed and not config.get(
            "_all_activated", False
        ):
            config["_all_activated"] = True
            gx, gy = config["gate_pos"]
            grid.terrain[gy, gx] = CellType.EMPTY
            gx2, gy2 = config["goal_positions"][0]
            grid.objects[gy2, gx2] = ObjectType.GOAL

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        config = new_state.get("config", {})
        activated = config.get("_switches_activated", 0)

        if activated > 0 and not self._switch_milestone_given:
            reward += 0.3 * activated
            self._switch_milestone_given = True

        if not config.get("_all_activated", False):
            switches = config.get("switch_positions", [])
            remaining = [
                sw
                for sw in switches
                if "grid" not in new_state
                or int(new_state["grid"].metadata[sw[1], sw[0]]) < 100
            ]
            if remaining and "agent" in new_state:
                ax, ay = new_state["agent"].position
                ox, oy = old_state.get("agent_position", (ax, ay))
                tgt = remaining[0]
                d_new = abs(ax - tgt[0]) + abs(ay - tgt[1])
                d_old = abs(ox - tgt[0]) + abs(oy - tgt[1])
                reward += 0.05 * (d_old - d_new)
        else:
            goal = config.get("goal_positions", [None])[0]
            if goal and "agent" in new_state:
                ax, ay = new_state["agent"].position
                ox, oy = old_state.get("agent_position", (ax, ay))
                d_new = abs(ax - goal[0]) + abs(ay - goal[1])
                d_old = abs(ox - goal[0]) + abs(oy - goal[1])
                reward += 0.05 * (d_old - d_new)

        if self.check_success(new_state):
            reward += 1.0
        return reward

    def check_success(self, state):
        config = state.get("config", {})
        if not config.get("_all_activated", False):
            return False
        if "grid" not in state or "agent" not in state:
            return False
        x, y = state["agent"].position
        return bool(state["grid"].objects[y, x] == ObjectType.GOAL)

    def validate_instance(self, grid, config):
        agent_pos = tuple(config.get("agent_start", (1, 1)))
        switch_positions = config.get("switch_positions", [])
        reachable = grid.flood_fill(agent_pos)
        # All switches must be reachable from agent (goal is behind the
        # gate and becomes reachable only after switches are activated)
        for sp in switch_positions:
            if tuple(sp) not in reachable:
                return False
        return bool(switch_positions)

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0
