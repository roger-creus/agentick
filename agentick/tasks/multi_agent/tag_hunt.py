"""TagHunt - Tag all fleeing NPCs before time runs out.

MECHANICS:
  - N opponent NPCs (ENEMY objects) that evade the agent
  - NPCs flee from the agent with configurable probability, otherwise move randomly
  - SWITCH objects act as one-time freeze power-ups: stepping on one freezes ALL
    NPCs for 5 steps (switch is consumed on use)
  - Agent tags an NPC by stepping onto it (NPC is removed)
  - Success = tag all NPCs within max_steps
  - No safe zones, no bidirectional tagging, no cooldown
"""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task

_FREEZE_DURATION = 5


@register_task("TagHunt-v0", tags=["multi_agent", "competition"])
class TagHuntTask(TaskSpec):
    """Tag all fleeing NPCs using pursuit and freeze power-ups."""

    name = "TagHunt-v0"
    description = "Tag all evading NPCs before time runs out; use freeze switches strategically"
    capability_tags = ["multi_agent", "competition"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=7,
            max_steps=80,
            params={
                "n_targets": 1,
                "evade_prob": 0.50,
                "n_switches": 1,
                "n_obstacles": 0,
            },
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=10,
            max_steps=140,
            params={
                "n_targets": 2,
                "evade_prob": 0.65,
                "n_switches": 1,
                "n_obstacles": 3,
            },
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=13,
            max_steps=220,
            params={
                "n_targets": 3,
                "evade_prob": 0.80,
                "n_switches": 1,
                "n_obstacles": 5,
            },
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=15,
            max_steps=350,
            params={
                "n_targets": 4,
                "evade_prob": 0.90,
                "n_switches": 1,
                "n_obstacles": 7,
            },
        ),
    }

    _DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        p = self.difficulty_config.params
        n = p.get("n_targets", 1)
        n_switches = p.get("n_switches", 1)
        n_obs = p.get("n_obstacles", 0)

        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        agent_pos = (int(rng.integers(1, 3)), int(rng.integers(1, 3)))

        # Place obstacles (interior walls)
        interior = [
            (x, y)
            for x in range(1, size - 1)
            for y in range(1, size - 1)
            if abs(x - agent_pos[0]) + abs(y - agent_pos[1]) > 2
        ]
        rng.shuffle(interior)
        for i in range(min(n_obs, len(interior) // 4)):
            ox, oy = interior[i]
            grid.terrain[oy, ox] = CellType.WALL

        # Place freeze SWITCH objects
        walkable = [
            (x, y)
            for x in range(1, size - 1)
            for y in range(1, size - 1)
            if grid.terrain[y, x] == CellType.EMPTY and (x, y) != agent_pos
        ]
        rng.shuffle(walkable)
        switch_positions = []
        for pos in walkable:
            if len(switch_positions) >= n_switches:
                break
            sx, sy = pos
            grid.objects[sy, sx] = ObjectType.SWITCH
            switch_positions.append(pos)

        # Place NPCs far from agent
        npc_candidates = [
            (x, y)
            for x in range(1, size - 1)
            for y in range(1, size - 1)
            if grid.terrain[y, x] == CellType.EMPTY
            and grid.objects[y, x] == ObjectType.NONE
            and (x, y) != agent_pos
            and abs(x - agent_pos[0]) + abs(y - agent_pos[1]) > size // 2
        ]
        if len(npc_candidates) < n:
            npc_candidates = [
                (x, y)
                for x in range(1, size - 1)
                for y in range(1, size - 1)
                if grid.terrain[y, x] == CellType.EMPTY
                and grid.objects[y, x] == ObjectType.NONE
                and (x, y) != agent_pos
            ]
        rng.shuffle(npc_candidates)
        npc_positions = npc_candidates[:n]

        for nx, ny in npc_positions:
            grid.objects[ny, nx] = ObjectType.ENEMY
            grid.metadata[ny, nx] = 2  # default facing down

        # Solvability: verify all NPCs are reachable from agent
        reachable = grid.flood_fill(agent_pos)
        for pos in npc_positions:
            if pos not in reachable:
                # Clear interior walls to guarantee connectivity
                for x in range(1, size - 1):
                    for y in range(1, size - 1):
                        if grid.terrain[y, x] == CellType.WALL:
                            grid.terrain[y, x] = CellType.EMPTY
                break

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": list(npc_positions),
            "evade_prob": p.get("evade_prob", 0.5),
            "switch_positions": switch_positions,
            "_rng_seed": int(rng.integers(0, 2**31)),
            "max_steps": self.get_max_steps(),
        }

    def on_env_reset(self, agent, grid, config):
        config["_live_npcs"] = list(config.get("goal_positions", []))
        config["_npc_rng"] = np.random.default_rng(config.get("_rng_seed", 0))
        config["_freeze_remaining"] = 0
        config["_active_switches"] = list(config.get("switch_positions", []))
        self._last_n = len(config["_live_npcs"])
        self._config = config
        # Restore SWITCH objects
        for sx, sy in config.get("_active_switches", []):
            if grid.objects[sy, sx] == ObjectType.NONE:
                grid.objects[sy, sx] = ObjectType.SWITCH
        # Restore NPC objects
        for nx, ny in config["_live_npcs"]:
            grid.objects[ny, nx] = ObjectType.ENEMY
            grid.metadata[ny, nx] = 2  # default facing down

    def on_agent_moved(self, pos, agent, grid):
        config = getattr(self, "_config", {})
        ax, ay = pos

        # Tag NPC: agent steps onto an ENEMY
        if grid.objects[ay, ax] == ObjectType.ENEMY:
            grid.objects[ay, ax] = ObjectType.NONE
            grid.metadata[ay, ax] = 0
            config["_live_npcs"] = [
                (nx, ny) for nx, ny in config.get("_live_npcs", []) if (nx, ny) != (ax, ay)
            ]

    def on_agent_interact(self, pos, agent, grid):
        """INTERACT on a freeze SWITCH consumes it and freezes all NPCs."""
        if not grid.in_bounds(pos):
            return
        x, y = pos
        config = getattr(self, "_config", {})
        if grid.objects[y, x] == ObjectType.SWITCH:
            grid.objects[y, x] = ObjectType.NONE
            grid.metadata[y, x] = 0
            config["_freeze_remaining"] = _FREEZE_DURATION
            config["_active_switches"] = [
                (sx, sy)
                for sx, sy in config.get("_active_switches", [])
                if (sx, sy) != (x, y)
            ]

    def _is_npc_walkable(self, x, y, grid):
        """Check if an NPC can walk to (x, y)."""
        if not (0 < x < grid.width - 1 and 0 < y < grid.height - 1):
            return False
        if grid.terrain[y, x] != CellType.EMPTY:
            return False
        if grid.objects[y, x] == ObjectType.ENEMY:
            return False
        return True

    def on_env_step(self, agent, grid, config, step_count):
        npcs = config.get("_live_npcs", [])
        rng = config.get("_npc_rng")
        evade = config.get("evade_prob", 0.5)
        ax, ay = agent.position

        # Handle freeze timer
        freeze = config.get("_freeze_remaining", 0)
        if freeze > 0:
            config["_freeze_remaining"] = freeze - 1
            return  # NPCs are frozen, skip all movement

        # Erase old NPC positions
        for nx, ny in npcs:
            if grid.objects[ny, nx] == ObjectType.ENEMY:
                grid.objects[ny, nx] = ObjectType.NONE
                grid.metadata[ny, nx] = 0

        # Build occupied set: all NPCs only (NOT agent — NPCs must tag agent)
        occupied = set()
        for nx, ny in npcs:
            occupied.add((nx, ny))

        new_npcs = []
        for nx, ny in npcs:
            occupied.discard((nx, ny))  # remove self before choosing move
            dist = abs(nx - ax) + abs(ny - ay)
            roll = rng.random()

            if roll < evade:
                # Evade: move to maximize distance from agent
                best, best_d = (nx, ny), dist
                for dx, dy in self._DIRS:
                    cx, cy = nx + dx, ny + dy
                    if self._is_npc_walkable(cx, cy, grid) and (cx, cy) not in occupied:
                        d = abs(cx - ax) + abs(cy - ay)
                        if d > best_d:
                            best_d, best = d, (cx, cy)
                new_npcs.append(best)
            else:
                # Random move
                moves = [(nx + dx, ny + dy) for dx, dy in self._DIRS]
                valid = [
                    (x, y)
                    for x, y in moves
                    if self._is_npc_walkable(x, y, grid) and (x, y) not in occupied
                ]
                new_npcs.append(valid[int(rng.integers(len(valid)))] if valid else (nx, ny))
            occupied.add(new_npcs[-1])  # reserve new position

        # Place NPCs at new positions with direction metadata
        final = []
        for idx, (nx, ny) in enumerate(new_npcs):
            grid.objects[ny, nx] = ObjectType.ENEMY
            # Store movement direction in metadata for sprite orientation
            old_nx, old_ny = npcs[idx] if idx < len(npcs) else (nx, ny)
            ddx, ddy = nx - old_nx, ny - old_ny
            if ddx > 0:
                grid.metadata[ny, nx] = 1  # right
            elif ddx < 0:
                grid.metadata[ny, nx] = 3  # left
            elif ddy < 0:
                grid.metadata[ny, nx] = 0  # up
            elif ddy > 0:
                grid.metadata[ny, nx] = 2  # down
            else:
                grid.metadata[ny, nx] = 2  # default down
            final.append((nx, ny))
        config["_live_npcs"] = final

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        config = new_state.get("config", {})
        npcs = config.get("_live_npcs", [])
        new_n = len(npcs)

        # Reward for tagging NPCs
        if new_n < self._last_n:
            reward += 0.3 * (self._last_n - new_n)
        self._last_n = new_n

        # Approach shaping toward nearest live NPC
        if npcs and "agent" in new_state:
            ax, ay = new_state["agent"].position
            ox, oy = old_state.get("agent_position", (ax, ay))
            d_new = min(abs(ax - nx) + abs(ay - ny) for nx, ny in npcs)
            d_old = min(abs(ox - nx) + abs(oy - ny) for nx, ny in npcs)
            reward += 0.05 * (d_old - d_new)

        if self.check_success(new_state):
            reward += 1.0
        return reward

    def check_success(self, state):
        config = state.get("config", {})
        return len(config.get("_live_npcs", [1])) == 0

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0
