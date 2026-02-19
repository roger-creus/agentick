"""CausalChain - Activate switches in order to unlock the goal room.

MECHANICS:
  - N switches placed on the grid, must be activated in ORDER (0 → 1 → ... → N-1)
  - Goal is in a WALLED ROOM accessible ONLY through a gate
  - Gate only opens when ALL switches have been activated in correct order
  - Wrong-order switch visits do nothing (switches only activate when it's their turn)
  - Tests causal reasoning and ordered planning

PROCEDURAL DIVERSITY (per seed):
  - Switch positions randomized
  - Goal room placed in a random corner
  - Gate location varies
  - Open arena with random interior obstacles
"""

import numpy as np
from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("CausalChain-v0", tags=["reasoning", "causal_reasoning"])
class CausalChainTask(TaskSpec):
    """Activate a chain of causes in order to unlock the goal room."""

    name = "CausalChain-v0"
    description = "Activate causal chain in order to unlock goal room"
    capability_tags = ["reasoning", "causal_reasoning"]

    difficulty_configs = {
        "easy":   DifficultyConfig(name="easy",   grid_size=9,  max_steps=100, params={"n_switches": 2}),
        "medium": DifficultyConfig(name="medium",  grid_size=11, max_steps=180, params={"n_switches": 3}),
        "hard":   DifficultyConfig(name="hard",    grid_size=13, max_steps=280, params={"n_switches": 4}),
        "expert": DifficultyConfig(name="expert",  grid_size=15, max_steps=420, params={"n_switches": 5}),
    }

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        n = self.difficulty_config.params.get("n_switches", 2)

        for attempt in range(20):
            grid = Grid(size, size)
            # Border walls only
            grid.terrain[0, :]  = CellType.WALL
            grid.terrain[-1, :] = CellType.WALL
            grid.terrain[:, 0]  = CellType.WALL
            grid.terrain[:, -1] = CellType.WALL

            # Choose a random corner for the goal room (2x2 room)
            corner = int(rng.integers(0, 4))
            if corner == 0:    # top-left
                room_cells = [(1,1),(2,1),(1,2),(2,2)]
                gate_cell  = (3, 1)
                gate_wall1 = (1, 3)
                room_walls = [(3,1),(3,2),(1,3),(2,3)]
            elif corner == 1:  # top-right
                room_cells = [(size-2,1),(size-3,1),(size-2,2),(size-3,2)]
                gate_cell  = (size-4, 1)
                room_walls = [(size-4,1),(size-4,2),(size-2,3),(size-3,3)]
            elif corner == 2:  # bottom-left
                room_cells = [(1,size-2),(2,size-2),(1,size-3),(2,size-3)]
                gate_cell  = (3, size-2)
                room_walls = [(3,size-2),(3,size-3),(1,size-4),(2,size-4)]
            else:              # bottom-right
                room_cells = [(size-2,size-2),(size-3,size-2),(size-2,size-3),(size-3,size-3)]
                gate_cell  = (size-4, size-2)
                room_walls = [(size-4,size-2),(size-4,size-3),(size-2,size-4),(size-3,size-4)]

            goal_cell = room_cells[0]

            # Wall off the goal room (except gate)
            for wx, wy in room_walls:
                if 0 < wx < size-1 and 0 < wy < size-1:
                    grid.terrain[wy, wx] = CellType.WALL
            # Gate starts as WALL (will open when all switches activated)
            gx, gy = gate_cell
            if 0 < gx < size-1 and 0 < gy < size-1:
                grid.terrain[gy, gx] = CellType.WALL

            # Agent starts in opposite area from goal room
            ax = size - 2 if goal_cell[0] < size // 2 else 1
            ay = size - 2 if goal_cell[1] < size // 2 else 1
            agent_pos = (ax, ay)

            # Place switches in the open area (not in goal room or agent start)
            forbidden = set(room_cells) | set(room_walls) | {gate_cell, agent_pos, goal_cell}
            free = [(x, y) for x in range(1, size-1) for y in range(1, size-1)
                    if (x, y) not in forbidden and grid.terrain[y, x] == CellType.EMPTY]
            rng.shuffle(free)

            if len(free) < n:
                continue  # retry

            # Spread switches across the grid for diversity
            switch_positions = []
            for _ in range(n):
                if not free:
                    break
                # Pick a position far from existing switches for spread
                if switch_positions:
                    best = max(free, key=lambda p: min(
                        abs(p[0]-s[0])+abs(p[1]-s[1]) for s in switch_positions))
                else:
                    best = free[rng.integers(len(free))]
                switch_positions.append(best)
                free = [p for p in free if p != best]

            if len(switch_positions) < n:
                continue

            # Add a few random interior obstacles for visual diversity
            n_obstacles = int(rng.integers(0, size // 3))
            obs_free = [p for p in free if p not in switch_positions]
            for _ in range(min(n_obstacles, len(obs_free))):
                ox, oy = obs_free[rng.integers(len(obs_free))]
                grid.terrain[oy, ox] = CellType.WALL
                obs_free = [p for p in obs_free if p != (ox, oy)]

            # Verify agent can reach all switches and is NOT in goal room
            reachable = grid.flood_fill(agent_pos)
            if all(sw in reachable for sw in switch_positions) and agent_pos not in room_cells:
                break
        else:
            # Fallback: simple layout
            grid = Grid(size, size)
            grid.terrain[0, :]  = CellType.WALL
            grid.terrain[-1, :] = CellType.WALL
            grid.terrain[:, 0]  = CellType.WALL
            grid.terrain[:, -1] = CellType.WALL
            agent_pos   = (1, 1)
            goal_cell   = (size-2, size-2)
            gate_cell   = (size-2, size-3)
            room_cells  = [goal_cell]
            room_walls  = [(size-3, size-2), (size-3, size-3)]
            for wx, wy in room_walls:
                grid.terrain[wy, wx] = CellType.WALL
            grid.terrain[gate_cell[1], gate_cell[0]] = CellType.WALL
            switch_positions = [(1 + i*2, 1) for i in range(n)]

        # Place switch markers
        for sx, sy in switch_positions:
            grid.objects[sy, sx] = ObjectType.TARGET  # unactivated = TARGET

        return grid, {
            "agent_start":      agent_pos,
            "goal_positions":   [goal_cell],
            "switch_positions": switch_positions,
            "gate_pos":         gate_cell,
            "room_cells":       room_cells,
            "n_switches":       n,
            "max_steps":        self.get_max_steps(),
        }

    # ── Hooks ────────────────────────────────────────────────────────────────

    def on_env_reset(self, agent, grid, config):
        config["_switch_progress"] = 0
        config["_gate_open"] = False
        self._last_progress = 0
        self._config = config
        # Ensure goal NOT placed (goal room starts locked)
        gx, gy = config["goal_positions"][0]
        grid.objects[gy, gx] = ObjectType.NONE
        # Redraw switches as TARGET
        for sx, sy in config["switch_positions"]:
            grid.objects[sy, sx] = ObjectType.TARGET

    def on_agent_moved(self, pos, agent, grid):
        """Activate next switch in sequence when stepped on."""
        config = getattr(self, "_config", {})
        progress = config.get("_switch_progress", 0)
        switches = config.get("switch_positions", [])
        ax, ay = pos

        if progress < len(switches):
            sx, sy = switches[progress]
            if (ax, ay) == (sx, sy) and grid.objects[sy, sx] == ObjectType.TARGET:
                grid.objects[sy, sx] = ObjectType.SWITCH  # activated = SWITCH visual
                config["_switch_progress"] = progress + 1

                # If all switches done, open gate and place goal
                if config["_switch_progress"] >= len(switches):
                    gx, gy = config["gate_pos"]
                    if 0 < gx < grid.width-1 and 0 < gy < grid.height-1:
                        grid.terrain[gy, gx] = CellType.EMPTY
                    config["_gate_open"] = True
                    goal_x, goal_y = config["goal_positions"][0]
                    grid.objects[goal_y, goal_x] = ObjectType.GOAL

    # ── Reward & success ─────────────────────────────────────────────────────

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        config = new_state.get("config", {})
        new_p = config.get("_switch_progress", 0)

        # Milestone per switch
        if new_p > self._last_progress:
            reward += 0.3 * (new_p - self._last_progress)
        self._last_progress = new_p

        # Approach shaping
        if "agent" in new_state:
            ax, ay = new_state["agent"].position
            ox, oy = old_state.get("agent_position", (ax, ay))
            switches = config.get("switch_positions", [])
            goal = config.get("goal_positions", [None])[0]

            if not config.get("_gate_open", False) and new_p < len(switches):
                tgt = switches[new_p]  # next switch
            elif config.get("_gate_open", False) and goal:
                tgt = goal
            else:
                tgt = None

            if tgt:
                d_new = abs(ax - tgt[0]) + abs(ay - tgt[1])
                d_old = abs(ox - tgt[0]) + abs(oy - tgt[1])
                reward += 0.05 * (d_old - d_new)

        if self.check_success(new_state):
            reward += 1.0
        return reward

    def check_success(self, state):
        """Success = agent at GOAL object (only possible after all switches activated)."""
        if "grid" not in state or "agent" not in state:
            return False
        x, y = state["agent"].position
        # Goal object only present after all switches activated — cannot cheat
        return bool(state["grid"].objects[y, x] == ObjectType.GOAL)

    def validate_instance(self, grid, config):
        return True  # gate is dynamic

    def get_optimal_return(self, difficulty=None): return 1.0
    def get_random_baseline(self, difficulty=None): return 0.0
