"""ChaseEvade - Agent must SURVIVE by evading pursuing enemies for N steps.

MECHANICS:
  - Enemy NPCs actively CHASE the agent (move toward it)
  - Agent must survive (not get caught) for a survival period
  - Enemy touches agent = episode ends in failure
  - Surviving all steps = success
  - Power-ups (RESOURCE) temporarily freeze enemies for K steps
  - Differentiated from CompetitiveTag: pure evasion/survival, no tagging back
"""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("ChaseEvade-v0", tags=["reactive_control", "prediction"])
class ChaseEvadeTask(TaskSpec):
    """Survive by evading all pursuing enemies for the required number of steps."""

    name = "ChaseEvade-v0"
    description = "Evade pursuing enemies and survive for N steps"
    capability_tags = ["reactive_control", "prediction"]

    difficulty_configs = {
        "easy":   DifficultyConfig(
            name="easy", grid_size=7, max_steps=40,
            params={
                "n_enemies": 1, "chase_prob": 0.60,
                "n_obstacles": 0, "enemy_speed": 1,
                "n_powerups": 1, "survival_steps": 30,
            },
        ),
        "medium": DifficultyConfig(
            name="medium", grid_size=10, max_steps=80,
            params={
                "n_enemies": 2, "chase_prob": 0.75,
                "n_obstacles": 3, "enemy_speed": 1,
                "n_powerups": 1, "survival_steps": 60,
            },
        ),
        "hard":   DifficultyConfig(
            name="hard", grid_size=13, max_steps=140,
            params={
                "n_enemies": 3, "chase_prob": 0.90,
                "n_obstacles": 5, "enemy_speed": 1,
                "n_powerups": 2, "survival_steps": 100,
            },
        ),
        "expert": DifficultyConfig(
            name="expert", grid_size=15, max_steps=220,
            params={
                "n_enemies": 4, "chase_prob": 1.00,
                "n_obstacles": 7, "enemy_speed": 2,
                "n_powerups": 2, "survival_steps": 160,
            },
        ),
    }

    _DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        p = self.difficulty_config.params
        n_enemies    = p.get("n_enemies", 1)
        chase_prob   = p.get("chase_prob", 0.6)
        n_obstacles  = p.get("n_obstacles", 0)
        enemy_speed  = p.get("enemy_speed", 1)
        n_powerups   = p.get("n_powerups", 0)
        survival     = p.get("survival_steps", 30)

        grid = Grid(size, size)
        grid.terrain[0, :]  = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0]  = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        # Agent starts center-ish
        agent_pos = (size // 2, size // 2)

        # Place obstacles
        interior = [
            (x, y) for x in range(1, size - 1)
            for y in range(1, size - 1)
            if abs(x - agent_pos[0]) + abs(y - agent_pos[1]) > 2
        ]
        rng.shuffle(interior)
        for i in range(min(n_obstacles, len(interior) // 4)):
            ox, oy = interior[i]
            grid.terrain[oy, ox] = CellType.WALL

        # Place enemies far from agent (corners and edges)
        walkable = [
            (x, y) for x in range(1, size - 1)
            for y in range(1, size - 1)
            if grid.terrain[y, x] == CellType.EMPTY
            and (x, y) != agent_pos
            and abs(x - agent_pos[0]) + abs(y - agent_pos[1]) > size // 2
        ]
        if len(walkable) < n_enemies:
            walkable = [
                (x, y) for x in range(1, size - 1)
                for y in range(1, size - 1)
                if grid.terrain[y, x] == CellType.EMPTY
                and (x, y) != agent_pos
            ]
        rng.shuffle(walkable)
        enemy_positions = walkable[:n_enemies]

        for ex, ey in enemy_positions:
            grid.objects[ey, ex] = ObjectType.ENEMY

        # Solvability: ensure connected
        reachable = grid.flood_fill(agent_pos)
        for ep in enemy_positions:
            if ep not in reachable:
                for x in range(1, size - 1):
                    for y in range(1, size - 1):
                        if grid.terrain[y, x] == CellType.WALL:
                            grid.terrain[y, x] = CellType.EMPTY
                break

        # Place power-ups (freeze enemies temporarily)
        used = {agent_pos} | set(enemy_positions)
        powerup_positions = []
        pw_candidates = [
            (x, y) for x in range(1, size - 1)
            for y in range(1, size - 1)
            if grid.terrain[y, x] == CellType.EMPTY and (x, y) not in used
        ]
        rng.shuffle(pw_candidates)
        for pp in pw_candidates[:n_powerups]:
            px, py = pp
            grid.objects[py, px] = ObjectType.POTION
            powerup_positions.append(pp)

        return grid, {
            "agent_start":       agent_pos,
            "goal_positions":    [],
            "chase_prob":        chase_prob,
            "enemy_speed":       enemy_speed,
            "survival_steps":    survival,
            "powerup_positions": powerup_positions,
            "_rng_seed":         int(rng.integers(0, 2**31)),
            "max_steps":         self.get_max_steps(),
        }

    # ── Hooks ─────────────────────────────────────────────────────────────────

    def on_env_reset(self, agent, grid, config):
        config["_enemies"] = [
            (x, y) for y in range(grid.height) for x in range(grid.width)
            if grid.objects[y, x] == ObjectType.ENEMY
        ]
        config["_evade_rng"] = np.random.default_rng(config.get("_rng_seed", 0))
        config["_caught"] = False
        config["_freeze_remaining"] = 0
        config["_steps_survived"] = 0
        self._config = config

    def on_agent_moved(self, pos, agent, grid):
        config = getattr(self, "_config", {})
        ax, ay = pos
        # Check if agent stepped on enemy
        if grid.objects[ay, ax] == ObjectType.ENEMY:
            config["_caught"] = True
        # Collect power-up (freeze enemies)
        if grid.objects[ay, ax] == ObjectType.POTION:
            grid.objects[ay, ax] = ObjectType.NONE
            config["_freeze_remaining"] = config.get("_freeze_remaining", 0) + 5

    def _move_enemy_once(self, ex, ey, ax, ay, grid, rng, chase_prob):
        """Move one enemy one step, chasing agent with probability chase_prob."""
        if rng.random() < chase_prob:
            # Chase: move toward agent
            best, best_d = (ex, ey), abs(ex - ax) + abs(ey - ay)
            for dx, dy in self._DIRS:
                nx, ny = ex + dx, ey + dy
                if (0 < nx < grid.width - 1
                        and 0 < ny < grid.height - 1
                        and grid.terrain[ny, nx] == CellType.EMPTY
                        and grid.objects[ny, nx] != ObjectType.ENEMY):
                    d = abs(nx - ax) + abs(ny - ay)
                    if d < best_d:
                        best_d, best = d, (nx, ny)
            return best
        # Random move
        moves = [(ex + dx, ey + dy) for dx, dy in self._DIRS]
        valid = [
            (x, y) for x, y in moves
            if (0 < x < grid.width - 1
                and 0 < y < grid.height - 1
                and grid.terrain[y, x] == CellType.EMPTY
                and grid.objects[y, x] != ObjectType.ENEMY)
        ]
        if valid:
            return valid[int(rng.integers(len(valid)))]
        return (ex, ey)

    def on_env_step(self, agent, grid, config, step_count):
        enemies = config.get("_enemies", [])
        rng = config.get("_evade_rng")
        chase = config.get("chase_prob", 0.7)
        speed = config.get("enemy_speed", 1)
        ax, ay = agent.position

        config["_steps_survived"] = step_count

        # Handle freeze
        freeze = config.get("_freeze_remaining", 0)
        if freeze > 0:
            config["_freeze_remaining"] = freeze - 1
            return  # enemies don't move when frozen

        # Erase old enemy positions
        for ex, ey in enemies:
            if grid.objects[ey, ex] == ObjectType.ENEMY:
                grid.objects[ey, ex] = ObjectType.NONE

        # Move enemies
        new_enemies = []
        for ex, ey in enemies:
            cx, cy = ex, ey
            for _ in range(speed):
                cx, cy = self._move_enemy_once(cx, cy, ax, ay, grid, rng, chase)
            new_enemies.append((cx, cy))

        # Place enemies and check for catching agent
        final = []
        for ex, ey in new_enemies:
            if (ex, ey) == (ax, ay):
                config["_caught"] = True
            else:
                grid.objects[ey, ex] = ObjectType.ENEMY
                final.append((ex, ey))

        config["_enemies"] = final

    # ── Reward & success ─────────────────────────────────────────────────────

    def compute_dense_reward(self, old_state, action, new_state, info):
        config = new_state.get("config", {})
        if config.get("_caught", False):
            return -1.0
        reward = 0.01  # small positive for surviving each step
        # Bonus for staying far from enemies
        if "agent" in new_state:
            ax, ay = new_state["agent"].position
            enemies = config.get("_enemies", [])
            if enemies:
                min_dist = min(abs(ax - ex) + abs(ay - ey) for ex, ey in enemies)
                reward += 0.01 * min(min_dist, 5)  # capped bonus for distance
        if self.check_success(new_state):
            reward += 1.0
        return reward

    def compute_sparse_reward(self, old_state, action, new_state, info):
        config = new_state.get("config", {})
        if config.get("_caught", False):
            return -1.0
        if self.check_success(new_state):
            return 1.0
        return 0.0

    def check_success(self, state):
        """Survived all required steps without being caught."""
        config = state.get("config", {})
        if config.get("_caught", False):
            return False
        survived = config.get("_steps_survived", 0)
        required = config.get("survival_steps", 30)
        return survived >= required

    def check_done(self, state):
        config = state.get("config", {})
        if config.get("_caught", False):
            return True
        return self.check_success(state)

    def get_optimal_return(self, difficulty=None): return 1.0
    def get_random_baseline(self, difficulty=None): return 0.0
