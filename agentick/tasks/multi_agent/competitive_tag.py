"""CompetitiveTag - Tag NPCs while avoiding being tagged back.

MECHANICS:
  - N opponent NPCs that both evade AND chase the agent
  - Safe zones (HAZARD terrain visually) where agent can't be tagged but also can't tag
  - Tag cooldown: after being tagged, agent is immune for K steps
  - NPCs alternate between fleeing and chasing based on distance
  - Score-based: +1 per NPC tagged, -0.5 if NPC tags agent
  - Success = tag all NPCs without net negative score
  - Differentiated from ChaseEvade: bidirectional tagging + safe zones
"""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("CompetitiveTag-v0", tags=["multi_agent", "competition"])
class CompetitiveTagTask(TaskSpec):
    """Tag all NPCs while avoiding being tagged back; use safe zones strategically."""

    name = "CompetitiveTag-v0"
    description = "Tag evading opponents while avoiding counter-tags"
    capability_tags = ["multi_agent", "competition"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=7,
            max_steps=100,
            params={
                "n_targets": 1,
                "evade_prob": 0.40,
                "chase_prob": 0.10,
                "n_safe_zones": 1,
                "tag_cooldown": 3,
                "n_obstacles": 0,
            },
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=10,
            max_steps=200,
            params={
                "n_targets": 2,
                "evade_prob": 0.50,
                "chase_prob": 0.20,
                "n_safe_zones": 2,
                "tag_cooldown": 2,
                "n_obstacles": 2,
            },
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=13,
            max_steps=350,
            params={
                "n_targets": 3,
                "evade_prob": 0.60,
                "chase_prob": 0.30,
                "n_safe_zones": 2,
                "tag_cooldown": 2,
                "n_obstacles": 4,
            },
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=15,
            max_steps=500,
            params={
                "n_targets": 4,
                "evade_prob": 0.55,
                "chase_prob": 0.40,
                "n_safe_zones": 3,
                "tag_cooldown": 1,
                "n_obstacles": 6,
            },
        ),
    }

    _DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        p = self.difficulty_config.params
        n = p.get("n_targets", 1)
        n_safe = p.get("n_safe_zones", 1)
        n_obs = p.get("n_obstacles", 0)

        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        agent_pos = (int(rng.integers(1, 3)), int(rng.integers(1, 3)))

        # Place obstacles
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

        # Place safe zones (ICE terrain — visually distinct, functional)
        walkable = [
            (x, y)
            for x in range(1, size - 1)
            for y in range(1, size - 1)
            if grid.terrain[y, x] == CellType.EMPTY and (x, y) != agent_pos
        ]
        rng.shuffle(walkable)
        safe_zone_positions = []
        for pos in walkable:
            if len(safe_zone_positions) >= n_safe:
                break
            sx, sy = pos
            grid.terrain[sy, sx] = CellType.ICE
            # Verify still connected
            reachable = grid.flood_fill(agent_pos)
            if len(reachable) > (size - 2) ** 2 // 3:
                safe_zone_positions.append(pos)
            else:
                grid.terrain[sy, sx] = CellType.EMPTY

        # Place NPCs on opposite side of grid from agent
        npc_candidates = [
            (x, y)
            for x in range(1, size - 1)
            for y in range(1, size - 1)
            if grid.terrain[y, x] == CellType.EMPTY
            and (x, y) != agent_pos
            and abs(x - agent_pos[0]) + abs(y - agent_pos[1]) > size // 2
        ]
        if len(npc_candidates) < n:
            npc_candidates = [
                (x, y)
                for x in range(1, size - 1)
                for y in range(1, size - 1)
                if grid.terrain[y, x] == CellType.EMPTY and (x, y) != agent_pos
            ]
        rng.shuffle(npc_candidates)
        npc_positions = npc_candidates[:n]

        for nx, ny in npc_positions:
            grid.objects[ny, nx] = ObjectType.ENEMY

        # Solvability check
        reachable = grid.flood_fill(agent_pos)
        for pos in npc_positions:
            if pos not in reachable:
                for x in range(1, size - 1):
                    for y in range(1, size - 1):
                        if grid.terrain[y, x] == CellType.WALL and (x, y) != (0, 0):
                            grid.terrain[y, x] = CellType.EMPTY
                break

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": list(npc_positions),
            "evade_prob": p.get("evade_prob", 0.5),
            "chase_prob": p.get("chase_prob", 0.1),
            "tag_cooldown": p.get("tag_cooldown", 2),
            "safe_zone_positions": safe_zone_positions,
            "_rng_seed": int(rng.integers(0, 2**31)),
            "max_steps": self.get_max_steps(),
        }

    def on_env_reset(self, agent, grid, config):
        config["_live_npcs"] = list(config.get("goal_positions", []))
        config["_npc_rng"] = np.random.default_rng(config.get("_rng_seed", 0))
        config["_agent_tagged_count"] = 0
        config["_cooldown_remaining"] = 0
        config["_score"] = 0.0
        self._last_n = len(config["_live_npcs"])
        self._last_score = 0.0
        self._config = config
        for nx, ny in config["_live_npcs"]:
            grid.objects[ny, nx] = ObjectType.ENEMY

    def _is_safe_zone(self, pos, grid):
        x, y = pos
        return grid.terrain[y, x] == CellType.ICE

    def on_agent_moved(self, pos, agent, grid):
        config = getattr(self, "_config", {})
        ax, ay = pos

        # Can't tag while in safe zone
        if self._is_safe_zone(pos, grid):
            return

        # Can't tag during cooldown
        if config.get("_cooldown_remaining", 0) > 0:
            return

        if grid.objects[ay, ax] == ObjectType.ENEMY:
            grid.objects[ay, ax] = ObjectType.NONE
            config["_live_npcs"] = [
                (nx, ny) for nx, ny in config.get("_live_npcs", []) if (nx, ny) != (ax, ay)
            ]
            config["_score"] = config.get("_score", 0) + 1.0

    def on_env_step(self, agent, grid, config, step_count):
        npcs = config.get("_live_npcs", [])
        rng = config.get("_npc_rng")
        evade = config.get("evade_prob", 0.5)
        chase = config.get("chase_prob", 0.1)
        ax, ay = agent.position

        # Decrease cooldown
        if config.get("_cooldown_remaining", 0) > 0:
            config["_cooldown_remaining"] -= 1

        # Erase old NPC positions
        for nx, ny in npcs:
            if grid.objects[ny, nx] == ObjectType.ENEMY:
                grid.objects[ny, nx] = ObjectType.NONE

        new_npcs = []
        for nx, ny in npcs:
            dist = abs(nx - ax) + abs(ny - ay)
            roll = rng.random()

            if roll < chase and dist > 2:
                # Chase mode: move toward agent
                best, best_d = (nx, ny), dist
                for dx, dy in self._DIRS:
                    cx, cy = nx + dx, ny + dy
                    if (
                        0 < cx < grid.width - 1
                        and 0 < cy < grid.height - 1
                        and grid.terrain[cy, cx] in (CellType.EMPTY, CellType.ICE)
                        and grid.objects[cy, cx] != ObjectType.ENEMY
                    ):
                        d = abs(cx - ax) + abs(cy - ay)
                        if d < best_d:
                            best_d, best = d, (cx, cy)
                new_npcs.append(best)
            elif roll < chase + evade:
                # Evade mode: move away from agent
                best, best_d = (nx, ny), dist
                for dx, dy in self._DIRS:
                    cx, cy = nx + dx, ny + dy
                    if (
                        0 < cx < grid.width - 1
                        and 0 < cy < grid.height - 1
                        and grid.terrain[cy, cx] in (CellType.EMPTY, CellType.ICE)
                        and grid.objects[cy, cx] != ObjectType.ENEMY
                    ):
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
                    if (
                        0 < x < grid.width - 1
                        and 0 < y < grid.height - 1
                        and grid.terrain[y, x] in (CellType.EMPTY, CellType.ICE)
                        and grid.objects[y, x] != ObjectType.ENEMY
                    )
                ]
                new_npcs.append(valid[int(rng.integers(len(valid)))] if valid else (nx, ny))

        # Check for NPC-tags-agent (bidirectional tagging)
        final = []
        for nx, ny in new_npcs:
            if (nx, ny) == (ax, ay):
                # NPC tags agent (unless agent in safe zone or has cooldown)
                if (
                    not self._is_safe_zone((ax, ay), grid)
                    and config.get("_cooldown_remaining", 0) <= 0
                ):
                    config["_score"] = config.get("_score", 0) - 0.5
                    config["_cooldown_remaining"] = config.get("tag_cooldown", 2)
                # NPC is still "tagged" by collision
                pass  # NPC removed
            else:
                grid.objects[ny, nx] = ObjectType.ENEMY
                final.append((nx, ny))
        config["_live_npcs"] = final

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        config = new_state.get("config", {})
        npcs = config.get("_live_npcs", [])
        new_n = len(npcs)
        score = config.get("_score", 0)

        # Reward for score changes
        if score != self._last_score:
            reward += score - self._last_score
        self._last_score = score

        if new_n < self._last_n:
            reward += 0.3 * (self._last_n - new_n)
        self._last_n = new_n

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
