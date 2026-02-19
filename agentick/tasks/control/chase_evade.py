"""ChaseEvade - Agent must CATCH a moving target (NPC that evades).

MECHANICS:
  - NPC target(s) actively evade the agent
  - Agent must step onto the SAME CELL as an NPC to catch it (tag it)
  - Success detection: check grid.objects[y,x]==ENEMY (robust, not position-tuple comparison)
  - NPC also tags agent if it moves onto agent's cell

DIFFICULTY AXES (multi-dimensional):
  - easy:   1 target, small map, 60% evade, no obstacles
  - medium: 2 targets, medium map, 75% evade, 2 obstacles
  - hard:   3 targets, large map, 90% evade, 4 obstacles
  - expert: 4 targets, largest map, 100% evade, 6 obstacles
"""

import numpy as np
from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("ChaseEvade-v0", tags=["reactive_control", "prediction"])
class ChaseEvadeTask(TaskSpec):
    """Catch all evading NPC targets (tag by stepping on them)."""

    name = "ChaseEvade-v0"
    description = "Chase and catch all evading targets"
    capability_tags = ["reactive_control", "prediction"]

    difficulty_configs = {
        "easy":   DifficultyConfig(name="easy",   grid_size=7,  max_steps=100, params={"n_targets": 1, "evade_prob": 0.60, "n_obstacles": 0}),
        "medium": DifficultyConfig(name="medium",  grid_size=10, max_steps=200, params={"n_targets": 2, "evade_prob": 0.75, "n_obstacles": 2}),
        "hard":   DifficultyConfig(name="hard",    grid_size=13, max_steps=350, params={"n_targets": 3, "evade_prob": 0.90, "n_obstacles": 4}),
        "expert": DifficultyConfig(name="expert",  grid_size=15, max_steps=500, params={"n_targets": 4, "evade_prob": 1.00, "n_obstacles": 6}),
    }

    _DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # (dx, dy): up/down/left/right

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        p = self.difficulty_config.params
        n_targets    = p.get("n_targets", 1)
        evade_prob   = p.get("evade_prob", 0.6)
        n_obstacles  = p.get("n_obstacles", 0)

        grid = Grid(size, size)
        grid.terrain[0, :]  = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0]  = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        # Agent at top-left area (randomized a bit)
        agent_pos = (int(rng.integers(1, 3)), int(rng.integers(1, 3)))

        # Random interior obstacles (placed away from agent start)
        interior = [(x, y) for x in range(1, size-1) for y in range(1, size-1)
                    if abs(x-agent_pos[0])+abs(y-agent_pos[1]) > 2]
        rng.shuffle(interior)
        obstacle_cells = set()
        for i in range(min(n_obstacles, len(interior) // 4)):
            ox, oy = interior[i]
            grid.terrain[oy, ox] = CellType.WALL
            obstacle_cells.add((ox, oy))

        # Place targets spread across the far side of the grid
        walkable = [(x, y) for x in range(1, size-1) for y in range(1, size-1)
                    if grid.terrain[y, x] == CellType.EMPTY and (x, y) != agent_pos
                    and abs(x-agent_pos[0])+abs(y-agent_pos[1]) > size//2]
        if len(walkable) < n_targets:
            walkable = [(x, y) for x in range(1, size-1) for y in range(1, size-1)
                        if grid.terrain[y, x] == CellType.EMPTY and (x, y) != agent_pos]
        rng.shuffle(walkable)
        target_positions = walkable[:n_targets]

        for tx, ty in target_positions:
            grid.objects[ty, tx] = ObjectType.ENEMY

        # Verify agent can reach all targets
        reachable = grid.flood_fill(agent_pos)
        for t in target_positions:
            if t not in reachable:
                # fallback: no obstacles
                for x in range(1, size-1):
                    for y in range(1, size-1):
                        if grid.terrain[y, x] == CellType.WALL:
                            grid.terrain[y, x] = CellType.EMPTY
                break

        return grid, {
            "agent_start":    agent_pos,
            "goal_positions": list(target_positions),
            "evade_prob":     evade_prob,
            "_rng_seed":      int(rng.integers(0, 2**31)),
            "max_steps":      self.get_max_steps(),
        }

    # ── Hooks ─────────────────────────────────────────────────────────────────

    def on_env_reset(self, agent, grid, config):
        config["_live_targets"] = list(config.get("goal_positions", []))
        config["_evade_rng"]    = np.random.default_rng(config.get("_rng_seed", 0))
        # backward-compat alias (tests may use _live_target)
        config["_live_target"]  = config["_live_targets"][0] if config["_live_targets"] else None
        self._last_n = len(config["_live_targets"])
        self._config = config
        # Redraw all targets
        for tx, ty in config["_live_targets"]:
            grid.objects[ty, tx] = ObjectType.ENEMY

    def on_agent_moved(self, pos, agent, grid):
        """Tag any NPC the agent steps onto — fires BEFORE check_success."""
        config = getattr(self, "_config", {})
        ax, ay = pos
        # Use grid-object check (robust: doesn't depend on position-tuple format)
        if grid.objects[ay, ax] == ObjectType.ENEMY:
            grid.objects[ay, ax] = ObjectType.NONE
            targets = config.get("_live_targets", [])
            config["_live_targets"] = [(tx, ty) for tx, ty in targets if (tx, ty) != (ax, ay)]

    def on_env_step(self, agent, grid, config, step_count):
        """Move all live targets (greedy evasion). Also tag if NPC lands on agent."""
        targets  = config.get("_live_targets", [])
        rng      = config.get("_evade_rng")
        evade    = config.get("evade_prob", 0.7)
        ax, ay   = agent.position

        # Erase old positions
        for tx, ty in targets:
            if grid.objects[ty, tx] == ObjectType.ENEMY:
                grid.objects[ty, tx] = ObjectType.NONE

        new_targets = []
        for tx, ty in targets:
            if rng.random() < evade:
                # Greedy: move to maximize distance from agent
                best, best_d = (tx, ty), abs(tx-ax)+abs(ty-ay)
                for dx, dy in self._DIRS:
                    nx, ny = tx+dx, ty+dy
                    if (0 < nx < grid.width-1 and 0 < ny < grid.height-1
                            and grid.terrain[ny, nx] == CellType.EMPTY
                            and grid.objects[ny, nx] not in (ObjectType.ENEMY,)):
                        d = abs(nx-ax)+abs(ny-ay)
                        if d > best_d:
                            best_d, best = d, (nx, ny)
                new_targets.append(best)
            else:
                moves = [(tx+dx, ty+dy) for dx, dy in self._DIRS]
                valid = [(x, y) for x, y in moves
                         if (0 < x < grid.width-1 and 0 < y < grid.height-1
                             and grid.terrain[y, x] == CellType.EMPTY
                             and grid.objects[y, x] != ObjectType.ENEMY)]
                new_targets.append(valid[int(rng.integers(len(valid)))] if valid else (tx, ty))

        # Draw new positions; check if any landed on agent
        final_targets = []
        for tx, ty in new_targets:
            if (tx, ty) == (ax, ay):
                # NPC ran INTO agent — counts as a tag!
                pass  # don't add back to live targets
            else:
                grid.objects[ty, tx] = ObjectType.ENEMY
                final_targets.append((tx, ty))

        config["_live_targets"] = final_targets

    # ── Reward & success ─────────────────────────────────────────────────────

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        config = new_state.get("config", {})
        targets = config.get("_live_targets", [])
        new_n   = len(targets)

        # Per-tag reward
        if new_n < self._last_n:
            reward += 0.5 * (self._last_n - new_n)
        self._last_n = new_n

        # Approach toward nearest remaining target
        if targets and "agent" in new_state:
            ax, ay = new_state["agent"].position
            ox, oy = old_state.get("agent_position", (ax, ay))
            d_new = min(abs(ax-tx)+abs(ay-ty) for tx, ty in targets)
            d_old = min(abs(ox-tx)+abs(oy-ty) for tx, ty in targets)
            reward += 0.05 * (d_old - d_new)

        if self.check_success(new_state):
            reward += 1.0
        return reward

    def check_success(self, state):
        """All targets caught = success. Also succeeds if agent is on an ENEMY cell right now."""
        config = state.get("config", {})
        targets = config.get("_live_targets", None)
        if targets is not None and len(targets) == 0:
            return True
        # Grid-object check: agent co-located with an ENEMY
        if "agent" in state and "grid" in state:
            x, y = state["agent"].position
            if state["grid"].objects[y, x] == ObjectType.ENEMY:
                return True
        return False

    def get_optimal_return(self, difficulty=None): return 1.0
    def get_random_baseline(self, difficulty=None): return 0.0
