"""ResourceManagement - Keep energy stations alive for max_steps.

MECHANICS:
  - N energy stations (RESOURCE objects) scattered on the map
  - Each station has metadata = energy level (100 = full, 0 = dead)
  - Energy drains every few steps at different rates per station
  - Agent visits a station to RECHARGE it (steps on RESOURCE -> charges to 100)
  - If ANY station hits 0 -> episode FAIL (returns -1.0)
  - SUCCESS = survive all max_steps with NO station dying (no goal to reach)
  - Agent must JUGGLE: recharge critical stations in time

VISIBILITY (all modalities):
  - Pixels: station color changes as energy depletes
      green (>=80%) -> yellow-green -> yellow -> orange -> red/critical (<=20%)
      Via metadata value 0-100 on RESOURCE objects rendered by _energy_color()
  - ASCII: "r" character for RESOURCE; number in metadata shows energy level
  - Language: "Station 1 is at 80%, Station 2 is critical at 15%"

DIFFICULTY:
  - easy:   2 stations, slow drain (every 8 steps), small map, no obstacles
  - medium: 3 stations, moderate drain (every 5 steps), obstacles
  - hard:   4 stations, fast drain (every 3 steps), more obstacles, varied rates
  - expert: 5 stations, very fast drain (every 2 steps), varied rates, large map with obstacles
"""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task

_FULL_ENERGY = 100


@register_task("ResourceManagement-v0", tags=["planning", "resource_allocation"])
class ResourceManagementTask(TaskSpec):
    """Keep all energy stations alive for the entire episode duration.

    There is no goal to reach. Success means surviving max_steps without
    any station's energy dropping to zero. The agent recharges stations
    by stepping on them. Drain rates scale with difficulty.
    """

    name = "ResourceManagement-v0"
    description = "Keep energy stations alive by recharging them before they drain to zero"
    capability_tags = ["planning", "resource_allocation"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=9,
            max_steps=100,
            params={
                "n_stations": 2,
                "drain_interval": 8,
                "drain_amounts": [1, 1],
                "n_obstacles": 0,
            },
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=12,
            max_steps=180,
            params={
                "n_stations": 3,
                "drain_interval": 5,
                "drain_amounts": [1, 2, 1],
                "n_obstacles": 3,
            },
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=15,
            max_steps=300,
            params={
                "n_stations": 4,
                "drain_interval": 3,
                "drain_amounts": [1, 2, 2, 3],
                "n_obstacles": 5,
            },
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=18,
            max_steps=450,
            params={
                "n_stations": 5,
                "drain_interval": 2,
                "drain_amounts": [1, 2, 2, 3, 3],
                "n_obstacles": 7,
            },
        ),
    }

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        p = self.difficulty_config.params
        n_stations = p.get("n_stations", 2)
        n_obstacles = p.get("n_obstacles", 0)
        drain_amounts = p.get("drain_amounts", [1] * n_stations)

        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        corners = [(1, 1), (size - 2, 1), (1, size - 2), (size - 2, size - 2)]
        agent_pos = tuple(corners[int(rng.integers(0, len(corners)))])
        used = {agent_pos}

        # Place stations well spread out
        interior = [
            (x, y) for x in range(2, size - 2) for y in range(2, size - 2)
            if (x, y) not in used
        ]
        rng.shuffle(interior)

        station_positions = []
        min_dist = max(3, (size - 2) // (n_stations + 1))
        for pos in interior:
            if len(station_positions) >= n_stations:
                break
            if all(abs(pos[0] - sp[0]) + abs(pos[1] - sp[1]) >= min_dist
                   for sp in station_positions):
                station_positions.append(pos)
                used.add(pos)

        # Fallback if not enough separated positions
        if len(station_positions) < n_stations:
            extra = [p2 for p2 in interior if p2 not in used]
            station_positions.extend(extra[:n_stations - len(station_positions)])
        station_positions = station_positions[:n_stations]
        for sp in station_positions:
            used.add(sp)

        # Place station objects with full energy
        for sx, sy in station_positions:
            grid.objects[sy, sx] = ObjectType.RESOURCE
            grid.metadata[sy, sx] = _FULL_ENERGY

        # Interior obstacles — keep all stations reachable
        critical = [agent_pos] + station_positions
        wall_cands = [p2 for p2 in interior if p2 not in used]
        rng.shuffle(wall_cands)
        placed_walls = 0
        for p2 in wall_cands:
            if placed_walls >= n_obstacles:
                break
            wx, wy = p2
            grid.terrain[wy, wx] = CellType.WALL
            reachable = grid.flood_fill(agent_pos)
            if all(c in reachable for c in critical):
                placed_walls += 1
                used.add(p2)
            else:
                grid.terrain[wy, wx] = CellType.EMPTY

        # Pad drain_amounts to n_stations
        drain_amts = list(drain_amounts)
        while len(drain_amts) < n_stations:
            drain_amts.append(drain_amts[-1] if drain_amts else 1)
        drain_amts = drain_amts[:n_stations]

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [],
            "station_positions": station_positions,
            "drain_interval": p.get("drain_interval", 10),
            "drain_amounts": drain_amts,
            "n_stations": n_stations,
            "max_steps": self.get_max_steps(),
        }

    def on_env_reset(self, agent, grid, config):
        config["_dead"] = False
        config["_energy_levels"] = [_FULL_ENERGY] * config.get("n_stations", 2)
        self._config = config
        self._last_min_energy = _FULL_ENERGY
        self._sync_station_metadata(grid, config)

    def _sync_station_metadata(self, grid, config):
        levels = config.get("_energy_levels", [])
        for i, (sx, sy) in enumerate(config.get("station_positions", [])):
            if i < len(levels):
                level = max(0, min(_FULL_ENERGY, levels[i]))
                grid.metadata[sy, sx] = level
                if grid.objects[sy, sx] == ObjectType.NONE:
                    grid.objects[sy, sx] = ObjectType.RESOURCE

    def on_agent_moved(self, pos, agent, grid):
        """Recharge station when agent steps on it."""
        config = getattr(self, "_config", {})
        x, y = pos
        if grid.objects[y, x] == ObjectType.RESOURCE:
            stations = config.get("station_positions", [])
            levels = config.get("_energy_levels", [])
            if (x, y) in stations:
                idx = stations.index((x, y))
                if idx < len(levels):
                    levels[idx] = _FULL_ENERGY
                    config["_energy_levels"] = levels
                    grid.metadata[y, x] = _FULL_ENERGY

    def on_env_step(self, agent, grid, config, step_count):
        """Drain energy from all stations periodically; check failure."""
        drain_interval = config.get("drain_interval", 10)
        drain_amounts = config.get("drain_amounts", [1])
        stations = config.get("station_positions", [])
        levels = config.get("_energy_levels", [_FULL_ENERGY] * len(stations))

        if step_count > 0 and step_count % drain_interval == 0:
            for i in range(len(stations)):
                amt = drain_amounts[i] if i < len(drain_amounts) else 1
                levels[i] = max(0, levels[i] - amt)

        config["_energy_levels"] = levels
        self._sync_station_metadata(grid, config)

        if any(lv <= 0 for lv in levels):
            config["_dead"] = True

    def compute_dense_reward(self, old_state, action, new_state, info):
        config = new_state.get("config", {})
        if config.get("_dead", False):
            return -1.0

        reward = -0.01

        levels = config.get("_energy_levels", [])
        stations = config.get("station_positions", [])
        if stations and "agent_position" in new_state:
            ax, ay = new_state["agent_position"]
            ox, oy = old_state.get("agent_position", (ax, ay))
            # Guide toward most urgent (lowest energy) station
            min_e = min(levels) if levels else _FULL_ENERGY
            if min_e < 50 and levels:
                urgency = [(levels[i], stations[i]) for i in range(len(stations))]
                urgency.sort(key=lambda q: q[0])
                tx, ty = urgency[0][1]
                reward += 0.04 * (abs(ox - tx) + abs(oy - ty) - abs(ax - tx) - abs(ay - ty))

        # Reward for recharging a station
        old_levels = old_state.get("config", {}).get("_energy_levels", [])
        for i, lv in enumerate(levels):
            if i < len(old_levels) and lv > old_levels[i]:
                reward += 0.1

        # Penalty for energy dropping very low
        min_e = min(levels) if levels else _FULL_ENERGY
        if min_e < self._last_min_energy and min_e < 20:
            reward -= 0.05
        self._last_min_energy = min_e

        if self.check_success(new_state):
            reward += 1.0
        return reward

    def compute_sparse_reward(self, old_state, action, new_state, info):
        config = new_state.get("config", {})
        if config.get("_dead", False):
            return -1.0
        # Survival task: success reward is only given at truncation (handled by
        # TaskEnv._check_success).  Mid-episode sparse reward stays 0.
        return 0.0

    def check_done(self, state):
        config = state.get("config", {})
        if config.get("_dead", False):
            return True
        # No explicit done — episode ends only when truncated (max_steps) or station dies
        return False

    def check_success(self, state):
        """Success = all stations alive at episode end (checked at truncation)."""
        config = state.get("config", {})
        if config.get("_dead", False):
            return False
        # Success is evaluated at truncation (max_steps reached)
        # The env wrapper checks check_success at truncation
        # If we're still alive when this is called, we're successful
        return not config.get("_dead", False)

    def validate_instance(self, grid, config):
        agent_pos = tuple(config.get("agent_start", (1, 1)))
        reachable = grid.flood_fill(agent_pos)
        for sp in config.get("station_positions", []):
            if tuple(sp) not in reachable:
                return False
        return True

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0
