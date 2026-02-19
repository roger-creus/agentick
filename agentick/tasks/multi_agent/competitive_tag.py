"""CompetitiveTag - Tag all NPC opponents before time runs out.

MECHANICS:
  - N opponent NPCs start at random positions and evade the agent
  - Agent tags an NPC by stepping ON its cell (or NPC moves onto agent)
  - Success detection uses grid.objects[y,x]==ENEMY (robust, no position-tuple flip bug)
  - NPC running INTO agent also counts as a tag (symmetric collision)

DIFFICULTY AXES:
  - More targets + larger map + smarter evasion + more obstacles
"""

import numpy as np
from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("CompetitiveTag-v0", tags=["multi_agent", "competition"])
class CompetitiveTagTask(TaskSpec):
    """Tag all evading NPCs to win."""

    name = "CompetitiveTag-v0"
    description = "Tag all evading opponents"
    capability_tags = ["multi_agent", "competition"]

    difficulty_configs = {
        "easy":   DifficultyConfig(name="easy",   grid_size=7,  max_steps=100, params={"n_targets": 1, "evade_prob": 0.50, "n_obstacles": 0}),
        "medium": DifficultyConfig(name="medium",  grid_size=10, max_steps=200, params={"n_targets": 2, "evade_prob": 0.70, "n_obstacles": 2}),
        "hard":   DifficultyConfig(name="hard",    grid_size=13, max_steps=350, params={"n_targets": 3, "evade_prob": 0.85, "n_obstacles": 4}),
        "expert": DifficultyConfig(name="expert",  grid_size=15, max_steps=500, params={"n_targets": 4, "evade_prob": 1.00, "n_obstacles": 6}),
    }

    _DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        p = self.difficulty_config.params
        n  = p.get("n_targets", 1)
        ep = p.get("evade_prob", 0.5)
        no = p.get("n_obstacles", 0)

        grid = Grid(size, size)
        grid.terrain[0, :]  = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0]  = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        # Agent at random position
        agent_pos = (int(rng.integers(1, 3)), int(rng.integers(1, 3)))

        # Random obstacles (away from agent)
        interior = [(x, y) for x in range(1, size-1) for y in range(1, size-1)
                    if abs(x-agent_pos[0])+abs(y-agent_pos[1]) > 2]
        rng.shuffle(interior)
        for i in range(min(no, len(interior)//4)):
            ox, oy = interior[i]
            grid.terrain[oy, ox] = CellType.WALL

        # NPC targets spread on opposite side
        walkable = [(x, y) for x in range(1, size-1) for y in range(1, size-1)
                    if grid.terrain[y, x] == CellType.EMPTY and (x,y) != agent_pos
                    and abs(x-agent_pos[0])+abs(y-agent_pos[1]) > size//2]
        if len(walkable) < n:
            walkable = [(x, y) for x in range(1, size-1) for y in range(1, size-1)
                        if grid.terrain[y, x] == CellType.EMPTY and (x,y) != agent_pos]
        rng.shuffle(walkable)
        npc_positions = walkable[:n]

        for nx, ny in npc_positions:
            grid.objects[ny, nx] = ObjectType.ENEMY

        # Solvability check
        reachable = grid.flood_fill(agent_pos)
        for pos in npc_positions:
            if pos not in reachable:
                for x in range(1, size-1):
                    for y in range(1, size-1):
                        if grid.terrain[y, x] == CellType.WALL:
                            grid.terrain[y, x] = CellType.EMPTY
                break

        return grid, {
            "agent_start":    agent_pos,
            "goal_positions": list(npc_positions),
            "evade_prob":     ep,
            "_rng_seed":      int(rng.integers(0, 2**31)),
            "max_steps":      self.get_max_steps(),
        }

    def on_env_reset(self, agent, grid, config):
        config["_live_npcs"] = list(config.get("goal_positions", []))
        config["_npc_rng"]   = np.random.default_rng(config.get("_rng_seed", 0))
        self._last_n = len(config["_live_npcs"])
        self._config = config
        for nx, ny in config["_live_npcs"]:
            grid.objects[ny, nx] = ObjectType.ENEMY

    def on_agent_moved(self, pos, agent, grid):
        """Tag NPC if agent steps onto its cell — use grid object check (no X,Y flip bug)."""
        config = getattr(self, "_config", {})
        ax, ay = pos
        # Check grid directly: robust against any position-tuple ordering issues
        if grid.objects[ay, ax] == ObjectType.ENEMY:
            grid.objects[ay, ax] = ObjectType.NONE
            config["_live_npcs"] = [(nx, ny) for nx, ny in config.get("_live_npcs", [])
                                    if (nx, ny) != (ax, ay)]

    def on_env_step(self, agent, grid, config, step_count):
        """Move NPCs; also tag if NPC lands on agent cell."""
        npcs   = config.get("_live_npcs", [])
        rng    = config.get("_npc_rng")
        evade  = config.get("evade_prob", 0.5)
        ax, ay = agent.position

        # Erase
        for nx, ny in npcs:
            if grid.objects[ny, nx] == ObjectType.ENEMY:
                grid.objects[ny, nx] = ObjectType.NONE

        new_npcs = []
        for nx, ny in npcs:
            if rng.random() < evade:
                best, best_d = (nx, ny), abs(nx-ax)+abs(ny-ay)
                for dx, dy in self._DIRS:
                    cx, cy = nx+dx, ny+dy
                    if (0 < cx < grid.width-1 and 0 < cy < grid.height-1
                            and grid.terrain[cy, cx] == CellType.EMPTY
                            and grid.objects[cy, cx] != ObjectType.ENEMY):
                        d = abs(cx-ax)+abs(cy-ay)
                        if d > best_d:
                            best_d, best = d, (cx, cy)
                new_npcs.append(best)
            else:
                moves = [(nx+dx, ny+dy) for dx, dy in self._DIRS]
                valid = [(x, y) for x, y in moves
                         if (0 < x < grid.width-1 and 0 < y < grid.height-1
                             and grid.terrain[y, x] == CellType.EMPTY
                             and grid.objects[y, x] != ObjectType.ENEMY)]
                new_npcs.append(valid[int(rng.integers(len(valid)))] if valid else (nx, ny))

        # Draw; if NPC lands on agent → tag it too
        final = []
        for nx, ny in new_npcs:
            if (nx, ny) == (ax, ay):
                pass  # tagged by collision
            else:
                grid.objects[ny, nx] = ObjectType.ENEMY
                final.append((nx, ny))
        config["_live_npcs"] = final

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        config = new_state.get("config", {})
        npcs   = config.get("_live_npcs", [])
        new_n  = len(npcs)
        if new_n < self._last_n:
            reward += 0.5 * (self._last_n - new_n)
        self._last_n = new_n
        if npcs and "agent" in new_state:
            ax, ay = new_state["agent"].position
            ox, oy = old_state.get("agent_position", (ax, ay))
            d_new = min(abs(ax-nx)+abs(ay-ny) for nx, ny in npcs)
            d_old = min(abs(ox-nx)+abs(oy-ny) for nx, ny in npcs)
            reward += 0.05 * (d_old - d_new)
        if self.check_success(new_state):
            reward += 1.0
        return reward

    def check_success(self, state):
        config = state.get("config", {})
        return len(config.get("_live_npcs", [1])) == 0

    def get_optimal_return(self, difficulty=None): return 1.0
    def get_random_baseline(self, difficulty=None): return 0.0
