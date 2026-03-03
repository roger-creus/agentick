"""ChaseEvade - Agent must SURVIVE by evading pursuing enemies for N steps.

MECHANICS:
  - 4 enemy behavior types inspired by Pac-Man ghosts, ALL use ObjectType.ENEMY
    (demon sprite). Behavior is distinguished by grid metadata:
    * metadata=1 Chaser: BFS shortest-path pursuit toward agent every step
    * metadata=2 Ambusher: Targets 4 tiles ahead of agent's facing direction
    * metadata=3 Flanker: Pincers from opposite side of agent vs nearest enemy
    * metadata=4 Trapper: Moves to block agent's best escape routes
  - Agent must survive (not get caught) for a survival period
  - Enemy touches agent = episode ends in failure
  - Surviving all steps = success
  - SWITCH objects freeze all enemies for 5 steps (one-time use per switch)
  - Differentiated from TagHunt: pure evasion/survival, no tagging back
"""

from collections import deque

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task

# Behavior metadata values (stored in grid.metadata)
_BEHAVIOR_META = {
    "chaser": 1,
    "ambusher": 2,
    "flanker": 3,
    "trapper": 4,
}

# Reverse lookup: metadata value -> behavior name
_META_BEHAVIOR = {v: k for k, v in _BEHAVIOR_META.items()}


@register_task("ChaseEvade-v0", tags=["reactive_control", "prediction"])
class ChaseEvadeTask(TaskSpec):
    """Survive by evading coordinated enemy pack for the required number of steps."""

    name = "ChaseEvade-v0"
    description = "Evade 4 enemy types (chaser, ambusher, flanker, trapper) and survive"
    capability_tags = ["reactive_control", "prediction"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=7,
            max_steps=40,
            params={
                "enemy_types": ["chaser"],
                "n_obstacles": 0,
                "n_powerups": 1,
                "survival_steps": 30,
            },
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=10,
            max_steps=80,
            params={
                "enemy_types": ["chaser", "ambusher"],
                "n_obstacles": 3,
                "n_powerups": 1,
                "survival_steps": 50,
            },
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=13,
            max_steps=140,
            params={
                "enemy_types": ["chaser", "ambusher", "flanker"],
                "n_obstacles": 5,
                "n_powerups": 1,
                "survival_steps": 80,
            },
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=15,
            max_steps=160,
            params={
                "enemy_types": ["chaser", "ambusher", "flanker", "trapper"],
                "n_obstacles": 8,
                "n_powerups": 2,
                "survival_steps": 100,
            },
        ),
    }

    _DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    # Direction deltas: 0=up, 1=right, 2=down, 3=left
    _DIR_DELTAS = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        p = self.difficulty_config.params
        enemy_types = p.get("enemy_types", ["chaser"])
        n_enemies = len(enemy_types)
        n_obstacles = p.get("n_obstacles", 0)
        n_powerups = p.get("n_powerups", 0)
        survival = p.get("survival_steps", 30)

        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        # Agent starts center-ish
        agent_pos = (size // 2, size // 2)

        # Place obstacles
        interior = [
            (x, y)
            for x in range(1, size - 1)
            for y in range(1, size - 1)
            if abs(x - agent_pos[0]) + abs(y - agent_pos[1]) > 2
        ]
        rng.shuffle(interior)
        for i in range(min(n_obstacles, len(interior) // 4)):
            ox, oy = interior[i]
            grid.terrain[oy, ox] = CellType.WALL

        # Place enemies far from agent (corners and edges)
        walkable = [
            (x, y)
            for x in range(1, size - 1)
            for y in range(1, size - 1)
            if grid.terrain[y, x] == CellType.EMPTY
            and (x, y) != agent_pos
            and abs(x - agent_pos[0]) + abs(y - agent_pos[1]) > size // 2
        ]
        if len(walkable) < n_enemies:
            walkable = [
                (x, y)
                for x in range(1, size - 1)
                for y in range(1, size - 1)
                if grid.terrain[y, x] == CellType.EMPTY and (x, y) != agent_pos
            ]
        rng.shuffle(walkable)
        enemy_positions = walkable[:n_enemies]

        for idx, (ex, ey) in enumerate(enemy_positions):
            etype = enemy_types[idx] if idx < len(enemy_types) else "chaser"
            grid.objects[ey, ex] = ObjectType.ENEMY
            grid.metadata[ey, ex] = _BEHAVIOR_META.get(etype, 1)

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
            (x, y)
            for x in range(1, size - 1)
            for y in range(1, size - 1)
            if grid.terrain[y, x] == CellType.EMPTY and (x, y) not in used
        ]
        rng.shuffle(pw_candidates)
        for pp in pw_candidates[:n_powerups]:
            px, py = pp
            grid.objects[py, px] = ObjectType.SWITCH
            powerup_positions.append(pp)

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [],
            "survival_steps": survival,
            "powerup_positions": powerup_positions,
            "_rng_seed": int(rng.integers(0, 2**31)),
            "max_steps": self.get_max_steps(),
            "_enemy_types": list(enemy_types),
        }

    # -- Hooks ----------------------------------------------------------------

    def on_env_reset(self, agent, grid, config):
        enemies = []
        for y in range(grid.height):
            for x in range(grid.width):
                if grid.objects[y, x] == ObjectType.ENEMY:
                    enemies.append((x, y))

        config["_enemies"] = enemies
        config["_evade_rng"] = np.random.default_rng(config.get("_rng_seed", 0))
        config["_caught"] = False
        config["_freeze_remaining"] = 0
        config["_steps_survived"] = 0
        config["_agent_last_dir"] = 2  # default facing down (south)
        self._config = config

    def on_agent_moved(self, pos, agent, grid):
        config = getattr(self, "_config", {})
        ax, ay = pos

        config["_agent_last_dir"] = int(agent.orientation)

        if grid.objects[ay, ax] == ObjectType.ENEMY:
            config["_caught"] = True

        if grid.objects[ay, ax] == ObjectType.SWITCH:
            grid.objects[ay, ax] = ObjectType.NONE
            config["_freeze_remaining"] = config.get("_freeze_remaining", 0) + 5

    # -- Enemy movement functions ---------------------------------------------

    @staticmethod
    def _walkable(x, y, grid):
        """Check if position is walkable (within interior walls)."""
        return (
            0 < x < grid.width - 1
            and 0 < y < grid.height - 1
            and grid.terrain[y, x] == CellType.EMPTY
        )

    def _bfs_next_step(self, ex, ey, tx, ty, grid, occupied):
        """BFS from (ex, ey) toward (tx, ty). Returns best adjacent cell.

        Uses reverse BFS from target to find true shortest path.
        If best step is occupied, tries alternative BFS directions before
        falling back to greedy.
        """
        if (ex, ey) == (tx, ty):
            return (ex, ey)

        # BFS from target back to enemy (reverse BFS gives distance map)
        visited = {(tx, ty): 0}
        queue = deque([(tx, ty)])
        while queue:
            cx, cy = queue.popleft()
            cd = visited[(cx, cy)]
            for dx, dy in self._DIRS:
                nx, ny = cx + dx, cy + dy
                if (nx, ny) not in visited and self._walkable(nx, ny, grid):
                    visited[(nx, ny)] = cd + 1
                    queue.append((nx, ny))

        if (ex, ey) not in visited:
            return self._greedy_toward(ex, ey, tx, ty, grid, occupied)

        # Pick the neighbor with the lowest distance to target
        cur_dist = visited[(ex, ey)]
        candidates = []
        for dx, dy in self._DIRS:
            nx, ny = ex + dx, ey + dy
            if (nx, ny) in visited and visited[(nx, ny)] < cur_dist:
                candidates.append((visited[(nx, ny)], nx, ny))

        candidates.sort()  # sort by distance (best first)
        for _, nx, ny in candidates:
            if (nx, ny) not in occupied:
                return (nx, ny)

        # All BFS-optimal steps occupied, try any walkable neighbor closer to target
        return self._greedy_toward(ex, ey, tx, ty, grid, occupied)

    def _greedy_toward(self, ex, ey, tx, ty, grid, occupied):
        """Greedy Manhattan move toward (tx, ty)."""
        best, best_d = (ex, ey), abs(ex - tx) + abs(ey - ty)
        for dx, dy in self._DIRS:
            nx, ny = ex + dx, ey + dy
            if (
                self._walkable(nx, ny, grid)
                and (nx, ny) not in occupied
            ):
                d = abs(nx - tx) + abs(ny - ty)
                if d < best_d:
                    best_d, best = d, (nx, ny)
        return best

    def _move_chaser(self, ex, ey, ax, ay, grid, occupied):
        """Chaser: BFS shortest-path pursuit toward agent every step."""
        return self._bfs_next_step(ex, ey, ax, ay, grid, occupied)

    def _move_ambusher(self, ex, ey, ax, ay, agent_dir, grid, occupied):
        """Ambusher: targets 4 tiles ahead of agent's facing direction.

        Cuts off escape routes by predicting where the agent is heading.
        Falls back to direct BFS pursuit when close or stuck.
        """
        ddx, ddy = self._DIR_DELTAS.get(agent_dir, (0, 1))
        tx = ax + ddx * 4
        ty = ay + ddy * 4
        tx = max(1, min(grid.width - 2, tx))
        ty = max(1, min(grid.height - 2, ty))

        # When close to agent or ambush target is useless, chase directly
        dist_to_agent = abs(ex - ax) + abs(ey - ay)
        if dist_to_agent <= 3 or (tx, ty) == (ex, ey):
            return self._bfs_next_step(ex, ey, ax, ay, grid, occupied)

        result = self._bfs_next_step(ex, ey, tx, ty, grid, occupied)
        if result == (ex, ey):
            return self._bfs_next_step(ex, ey, ax, ay, grid, occupied)
        return result

    def _move_flanker(self, idx, ex, ey, ax, ay, all_enemies, grid, occupied):
        """Flanker: pincers from opposite side of agent vs nearest other enemy.

        Creates coordinated pincer attacks by targeting the mirror position
        of the nearest ally relative to the agent. If the agent is between
        the flanker and another enemy, the agent is squeezed.
        """
        # Find the nearest OTHER enemy to the agent
        best_ref = None
        best_dist = 99999
        for i, (ox, oy) in enumerate(all_enemies):
            if i == idx:
                continue
            d = abs(ox - ax) + abs(oy - ay)
            if d < best_dist:
                best_dist = d
                best_ref = (ox, oy)

        if best_ref is None:
            # Only enemy left: just chase directly
            return self._bfs_next_step(ex, ey, ax, ay, grid, occupied)

        # Target = opposite side of agent from reference enemy
        # Vector: ref → agent → target (same direction, extends past agent)
        rx, ry = best_ref
        tx = ax + (ax - rx)
        ty = ay + (ay - ry)
        tx = max(1, min(grid.width - 2, tx))
        ty = max(1, min(grid.height - 2, ty))

        # If already at or near the pincer target, or close to agent, chase directly
        dist_to_target = abs(ex - tx) + abs(ey - ty)
        dist_to_agent = abs(ex - ax) + abs(ey - ay)
        if dist_to_target <= 1 or dist_to_agent <= 2:
            return self._bfs_next_step(ex, ey, ax, ay, grid, occupied)

        result = self._bfs_next_step(ex, ey, tx, ty, grid, occupied)
        if result == (ex, ey):
            return self._bfs_next_step(ex, ey, ax, ay, grid, occupied)
        return result

    def _move_trapper(self, ex, ey, ax, ay, all_enemies, grid, occupied):
        """Trapper: moves to the cell that blocks agent's best escape route.

        Evaluates all agent-adjacent cells and picks the one that, if
        the trapper occupied it, would leave the agent with the fewest
        escape options. BFS-paths toward that cell.
        """
        # Find all walkable cells adjacent to agent
        adj_cells = []
        for dx, dy in self._DIRS:
            cx, cy = ax + dx, ay + dy
            if self._walkable(cx, cy, grid):
                adj_cells.append((cx, cy))

        if not adj_cells:
            return self._bfs_next_step(ex, ey, ax, ay, grid, occupied)

        # Build set of all enemy positions except self
        enemy_set = set(all_enemies) - {(ex, ey)}

        # Score each adjacent cell: how many escapes remain if trapper were there?
        best_target = None
        best_remaining = 99
        best_dist = 99

        for cx, cy in adj_cells:
            if (cx, cy) in enemy_set:
                continue  # already blocked by another enemy
            # Count remaining escapes if trapper were at (cx, cy)
            remaining = 0
            for dx, dy in self._DIRS:
                nx, ny = ax + dx, ay + dy
                if (nx, ny) == (cx, cy):
                    continue  # blocked by trapper
                if (nx, ny) in enemy_set or (nx, ny) in occupied:
                    continue  # blocked by other enemy
                if not self._walkable(nx, ny, grid):
                    continue
                remaining += 1
            # Prefer cells that leave fewer escapes; break ties by distance
            d = abs(ex - cx) + abs(ey - cy)
            if remaining < best_remaining or (
                remaining == best_remaining and d < best_dist
            ):
                best_remaining = remaining
                best_target = (cx, cy)
                best_dist = d

        if best_target is None:
            return self._bfs_next_step(ex, ey, ax, ay, grid, occupied)

        return self._bfs_next_step(ex, ey, best_target[0], best_target[1], grid, occupied)

    def _dispatch_enemy_move(self, idx, ex, ey, ax, ay, config, grid, occupied,
                             all_enemies):
        """Dispatch enemy movement based on metadata behavior type.

        Reads grid.metadata[ey, ex] to determine behavior:
          1=chaser, 2=ambusher, 3=flanker, 4=trapper.
        Returns (new_x, new_y).
        """
        behavior = int(grid.metadata[ey, ex])
        etype = _META_BEHAVIOR.get(behavior, "chaser")

        if etype == "chaser":
            return self._move_chaser(ex, ey, ax, ay, grid, occupied)

        elif etype == "ambusher":
            agent_dir = config.get("_agent_last_dir", 2)
            return self._move_ambusher(ex, ey, ax, ay, agent_dir, grid, occupied)

        elif etype == "flanker":
            return self._move_flanker(
                idx, ex, ey, ax, ay, all_enemies, grid, occupied
            )

        elif etype == "trapper":
            return self._move_trapper(
                ex, ey, ax, ay, all_enemies, grid, occupied
            )

        # Fallback: chase
        return self._move_chaser(ex, ey, ax, ay, grid, occupied)

    def on_env_step(self, agent, grid, config, step_count):
        enemies = config.get("_enemies", [])
        ax, ay = agent.position

        config["_steps_survived"] = step_count

        # Handle freeze
        freeze = config.get("_freeze_remaining", 0)
        if freeze > 0:
            config["_freeze_remaining"] = freeze - 1
            return  # enemies don't move when frozen

        # Read behavior metadata before erasing old positions
        enemy_behaviors = []
        for ex, ey in enemies:
            enemy_behaviors.append(int(grid.metadata[ey, ex]))

        # Erase old enemy positions
        for ex, ey in enemies:
            if grid.objects[ey, ex] == ObjectType.ENEMY:
                grid.objects[ey, ex] = ObjectType.NONE
                grid.metadata[ey, ex] = 0

        # Build occupied set: all enemies only (NOT agent -- enemies must catch agent)
        occupied = set(enemies)

        # Track current positions (updated as each enemy moves) for coordination
        current_positions = list(enemies)

        # Move enemies one at a time, updating occupied set
        new_enemies = []
        for idx, (ex, ey) in enumerate(enemies):
            # Temporarily restore metadata so dispatch can read it
            behavior = enemy_behaviors[idx] if idx < len(enemy_behaviors) else 1
            grid.metadata[ey, ex] = behavior

            occupied.discard((ex, ey))  # remove self before choosing move
            nx, ny = self._dispatch_enemy_move(
                idx, ex, ey, ax, ay, config, grid, occupied, current_positions
            )
            new_enemies.append((nx, ny))
            occupied.add((nx, ny))  # reserve new position
            current_positions[idx] = (nx, ny)  # update for coordination

            # Clear temporary metadata
            grid.metadata[ey, ex] = 0

        # Place enemies and check for catching agent
        final = []
        for i, (ex, ey) in enumerate(new_enemies):
            behavior = enemy_behaviors[i] if i < len(enemy_behaviors) else 1

            if (ex, ey) == (ax, ay):
                config["_caught"] = True
            else:
                grid.objects[ey, ex] = ObjectType.ENEMY
                # Carry behavior metadata to the new position
                grid.metadata[ey, ex] = behavior
                final.append((ex, ey))

        config["_enemies"] = final

    # -- Reward & success -----------------------------------------------------

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

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0
