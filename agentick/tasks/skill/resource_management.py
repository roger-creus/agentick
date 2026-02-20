"""ResourceManagement - Economics with energy, health, and scarcity trade-offs.

MECHANICS:
  - Agent has ENERGY (depletes each step) and HEALTH
  - COIN objects restore energy; POTION objects restore health
  - HAZARD terrain drains health (1 per step on hazard)
  - WATER terrain costs double energy to traverse
  - Agent must collect enough coins (energy) and potions (health) to survive
    the journey through dangerous terrain to reach the GOAL
  - Running out of energy or health = episode ends in failure
  - Guards (NPC) patrol and deal damage on contact
  - Tests resource planning, trade-off reasoning, scarcity management
"""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task

_RM_DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]


@register_task("ResourceManagement-v0", tags=["planning", "resource_allocation"])
class ResourceManagementTask(TaskSpec):
    """Manage energy and health resources to survive journey to goal."""

    name = "ResourceManagement-v0"
    description = "Manage energy and health to survive journey to goal"
    capability_tags = ["planning", "resource_allocation"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy", grid_size=9, max_steps=100,
            params={
                "n_coins": 4, "n_potions": 2, "n_hazard_patches": 1,
                "n_water_patches": 1, "n_guards": 0,
                "start_energy": 40, "start_health": 3,
                "energy_per_coin": 15, "health_per_potion": 2,
            },
        ),
        "medium": DifficultyConfig(
            name="medium", grid_size=12, max_steps=180,
            params={
                "n_coins": 5, "n_potions": 3, "n_hazard_patches": 2,
                "n_water_patches": 2, "n_guards": 0,
                "start_energy": 50, "start_health": 4,
                "energy_per_coin": 12, "health_per_potion": 2,
            },
        ),
        "hard": DifficultyConfig(
            name="hard", grid_size=15, max_steps=300,
            params={
                "n_coins": 6, "n_potions": 3, "n_hazard_patches": 3,
                "n_water_patches": 3, "n_guards": 1,
                "start_energy": 60, "start_health": 5,
                "energy_per_coin": 10, "health_per_potion": 2,
            },
        ),
        "expert": DifficultyConfig(
            name="expert", grid_size=18, max_steps=500,
            params={
                "n_coins": 7, "n_potions": 4, "n_hazard_patches": 4,
                "n_water_patches": 4, "n_guards": 2,
                "start_energy": 70, "start_health": 5,
                "energy_per_coin": 10, "health_per_potion": 2,
            },
        ),
    }

    _DIRS = _RM_DIRS

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        p = self.difficulty_config.params

        grid = Grid(size, size)
        grid.terrain[0, :]  = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0]  = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        agent_pos = (1, 1)
        goal_pos = (size - 2, size - 2)
        grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.GOAL

        free = [
            (x, y) for x in range(1, size - 1) for y in range(1, size - 1)
            if (x, y) != agent_pos and (x, y) != goal_pos
        ]
        rng.shuffle(free)
        used = {agent_pos, goal_pos}

        # Place hazard patches (clusters of 2-3 hazard cells)
        n_haz = p.get("n_hazard_patches", 1)
        hazard_cells = []
        for _ in range(n_haz):
            candidates = [pp for pp in free if pp not in used]
            if not candidates:
                break
            center = candidates[int(rng.integers(len(candidates)))]
            cx, cy = center
            patch = [(cx, cy)]
            for dx, dy in _RM_DIRS:
                nx, ny = cx + dx, cy + dy
                if (1 <= nx < size - 1 and 1 <= ny < size - 1
                        and (nx, ny) not in used and (nx, ny) != goal_pos):
                    patch.append((nx, ny))
                    if len(patch) >= 3:
                        break
            for hx, hy in patch:
                grid.terrain[hy, hx] = CellType.HAZARD
                hazard_cells.append((hx, hy))
                used.add((hx, hy))
            # Verify still solvable
            reachable = grid.flood_fill(agent_pos)
            if goal_pos not in reachable:
                # Undo last patch
                for hx, hy in patch:
                    grid.terrain[hy, hx] = CellType.EMPTY
                    used.discard((hx, hy))
                    if (hx, hy) in hazard_cells:
                        hazard_cells.remove((hx, hy))

        # Place water patches
        n_water = p.get("n_water_patches", 1)
        water_cells = []
        for _ in range(n_water):
            candidates = [pp for pp in free if pp not in used]
            if not candidates:
                break
            center = candidates[int(rng.integers(len(candidates)))]
            cx, cy = center
            patch = [(cx, cy)]
            for dx, dy in _RM_DIRS:
                nx, ny = cx + dx, cy + dy
                if (1 <= nx < size - 1 and 1 <= ny < size - 1
                        and (nx, ny) not in used):
                    patch.append((nx, ny))
                    if len(patch) >= 3:
                        break
            for wx, wy in patch:
                grid.terrain[wy, wx] = CellType.WATER
                water_cells.append((wx, wy))
                used.add((wx, wy))

        # Place coins (energy)
        n_coins = p.get("n_coins", 4)
        coin_candidates = [pp for pp in free if pp not in used]
        rng.shuffle(coin_candidates)
        coin_positions = coin_candidates[:n_coins]
        for cx, cy in coin_positions:
            grid.objects[cy, cx] = ObjectType.COIN
            used.add((cx, cy))

        # Place potions (health)
        n_potions = p.get("n_potions", 2)
        potion_candidates = [pp for pp in free if pp not in used]
        rng.shuffle(potion_candidates)
        potion_positions = potion_candidates[:n_potions]
        for px, py in potion_positions:
            grid.objects[py, px] = ObjectType.POTION
            used.add((px, py))

        # Place guards
        n_guards = p.get("n_guards", 0)
        guard_candidates = [
            pp for pp in free if pp not in used
            and abs(pp[0] - agent_pos[0]) + abs(pp[1] - agent_pos[1]) > 3
        ]
        rng.shuffle(guard_candidates)
        guard_positions = guard_candidates[:n_guards]
        for gx, gy in guard_positions:
            grid.objects[gy, gx] = ObjectType.NPC
            used.add((gx, gy))

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [goal_pos],
            "start_energy": p.get("start_energy", 40),
            "start_health": p.get("start_health", 3),
            "energy_per_coin": p.get("energy_per_coin", 15),
            "health_per_potion": p.get("health_per_potion", 2),
            "_guard_positions": guard_positions,
            "_guard_dirs": [int(rng.integers(0, 4)) for _ in guard_positions],
            "_guard_seed": int(rng.integers(0, 2**31)),
            "max_steps": self.get_max_steps(),
        }

    def on_env_reset(self, agent, grid, config):
        config["_energy"] = config.get("start_energy", 40)
        config["_health"] = config.get("start_health", 3)
        config["_dead"] = False
        config["_guard_rng"] = np.random.default_rng(config.get("_guard_seed", 0))
        self._last_energy = config["_energy"]
        self._last_health = config["_health"]
        self._config = config

    def on_agent_moved(self, pos, agent, grid):
        config = getattr(self, "_config", {})
        x, y = pos

        # Collect coin (energy)
        if grid.objects[y, x] == ObjectType.COIN:
            grid.objects[y, x] = ObjectType.NONE
            config["_energy"] = config.get("_energy", 0) + config.get("energy_per_coin", 15)

        # Collect potion (health)
        if grid.objects[y, x] == ObjectType.POTION:
            grid.objects[y, x] = ObjectType.NONE
            config["_health"] = config.get("_health", 0) + config.get("health_per_potion", 2)

        # Guard collision
        if grid.objects[y, x] == ObjectType.NPC:
            config["_health"] = config.get("_health", 0) - 1

        # Hazard damage
        if grid.terrain[y, x] == CellType.HAZARD:
            config["_health"] = config.get("_health", 0) - 1

        # Water costs extra energy
        if grid.terrain[y, x] == CellType.WATER:
            config["_energy"] = config.get("_energy", 0) - 1  # extra cost

    def on_env_step(self, agent, grid, config, step_count):
        # Energy depletes each step
        config["_energy"] = config.get("_energy", 0) - 1

        # Check death conditions
        if config.get("_energy", 0) <= 0 or config.get("_health", 0) <= 0:
            config["_dead"] = True

        # Move guards
        guards = config.get("_guard_positions", [])
        dirs = config.get("_guard_dirs", [])
        rng = config.get("_guard_rng")
        ax, ay = agent.position
        if not guards or rng is None:
            return
        for gx, gy in guards:
            if grid.objects[gy, gx] == ObjectType.NPC:
                grid.objects[gy, gx] = ObjectType.NONE
        new_g, new_d = [], []
        for i, (gx, gy) in enumerate(guards):
            d = dirs[i]
            ddx, ddy = self._DIRS[d]
            nx, ny = gx + ddx, gy + ddy
            if (0 < nx < grid.width - 1 and 0 < ny < grid.height - 1
                    and grid.terrain[ny, nx] == CellType.EMPTY
                    and grid.objects[ny, nx] == ObjectType.NONE):
                new_g.append((nx, ny))
            else:
                d = int(rng.integers(0, 4))
                new_g.append((gx, gy))
            new_d.append(d)
            if new_g[-1] == (ax, ay):
                config["_health"] = config.get("_health", 0) - 1
        config["_guard_positions"] = new_g
        config["_guard_dirs"] = new_d
        for gx, gy in new_g:
            if grid.terrain[gy, gx] == CellType.EMPTY:
                grid.objects[gy, gx] = ObjectType.NPC

    def compute_dense_reward(self, old_state, action, new_state, info):
        config = new_state.get("config", {})
        if config.get("_dead", False):
            return -1.0
        reward = -0.01

        # Small reward for resource collection
        new_e = config.get("_energy", 0)
        new_h = config.get("_health", 0)
        if new_e > self._last_energy:
            reward += 0.1
        if new_h > self._last_health:
            reward += 0.15
        self._last_energy = new_e
        self._last_health = new_h

        # Approach goal
        if "agent" in new_state:
            goal = config.get("goal_positions", [None])[0]
            if goal:
                ax, ay = new_state["agent"].position
                ox, oy = old_state.get("agent_position", (ax, ay))
                old_d = abs(ox - goal[0]) + abs(oy - goal[1])
                new_d = abs(ax - goal[0]) + abs(ay - goal[1])
                reward += 0.05 * (old_d - new_d)

        if self.check_success(new_state):
            reward += 1.0
        return reward

    def compute_sparse_reward(self, old_state, action, new_state, info):
        config = new_state.get("config", {})
        if config.get("_dead", False):
            return -1.0
        if self.check_success(new_state):
            return 1.0
        return 0.0

    def check_success(self, state):
        config = state.get("config", {})
        if config.get("_dead", False):
            return False
        if "grid" not in state or "agent" not in state:
            return False
        x, y = state["agent"].position
        return bool(state["grid"].objects[y, x] == ObjectType.GOAL)

    def check_done(self, state):
        config = state.get("config", {})
        if config.get("_dead", False):
            return True
        return self.check_success(state)

    def validate_instance(self, grid, config):
        return True

    def get_optimal_return(self, difficulty=None): return 1.0
    def get_random_baseline(self, difficulty=None): return 0.0
