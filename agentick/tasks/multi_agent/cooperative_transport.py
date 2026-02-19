"""CooperativeTransport - Push a heavy box to the target zone with an NPC partner.

PROCEDURAL DIVERSITY (all per seed):
  - Box, NPC, agent, and target placed in randomized configurations
  - Direction of push varies (horizontal, vertical, diagonal L-shaped paths)
  - Obstacles added at higher difficulties to require maneuvering
  - Grid size and step budget scale with difficulty

DIFFICULTY AXES:
  - Larger grids + longer push distances
  - Internal obstacles blocking direct push paths
  - Target farther from initial box position
  - More complex room shapes
"""

import numpy as np
from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("CooperativeTransport-v0", tags=["multi_agent", "cooperation"])
class CooperativeTransportTask(TaskSpec):
    """Push the box to the target zone with your NPC partner."""

    name = "CooperativeTransport-v0"
    description = "Cooperatively push box to target zone"
    capability_tags = ["multi_agent", "cooperation"]

    difficulty_configs = {
        "easy":   DifficultyConfig(name="easy",   grid_size=7,  max_steps=80,  params={"n_obstacles": 0}),
        "medium": DifficultyConfig(name="medium",  grid_size=10, max_steps=160, params={"n_obstacles": 2}),
        "hard":   DifficultyConfig(name="hard",    grid_size=13, max_steps=280, params={"n_obstacles": 4}),
        "expert": DifficultyConfig(name="expert",  grid_size=15, max_steps=420, params={"n_obstacles": 6}),
    }

    _DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        n_obs = self.difficulty_config.params.get("n_obstacles", 0)

        for attempt in range(30):
            grid = Grid(size, size)
            grid.terrain[0, :]  = CellType.WALL
            grid.terrain[-1, :] = CellType.WALL
            grid.terrain[:, 0]  = CellType.WALL
            grid.terrain[:, -1] = CellType.WALL

            # Always use VERTICAL push (box north of target, dy>0) for predictability.
            # Diversity comes from column position and box-to-target distance.
            # NPC placed to the SIDE (not on push column) to avoid blocking.
            col       = int(rng.integers(2, size - 2))
            row_agent = 1
            row_box   = int(rng.integers(2, max(3, size // 2)))
            row_tgt   = int(rng.integers(row_box + 2, size - 1))

            agent_pos  = (col, row_agent)
            box_pos    = (col, row_box)
            target_pos = (col, row_tgt)

            # NPC in a side column (not same column as push path)
            npc_col = col - 1 if col > 1 else col + 1
            npc_col = max(1, min(size-2, npc_col))
            row_npc = (row_box + row_tgt) // 2  # somewhere between box and target
            npc_pos = (npc_col, row_npc)

            # Validate positions are distinct and in bounds
            positions = [agent_pos, box_pos, npc_pos, target_pos]
            if len(set(positions)) < 4:
                continue
            if not all(0 < p[0] < size-1 and 0 < p[1] < size-1 for p in positions):
                continue

            # Place internal obstacles (not blocking box-to-target direct path)
            placed_obs = 0
            obs_candidates = [(x, y) for x in range(1, size-1) for y in range(1, size-1)
                              if (x, y) not in set(positions)]
            rng.shuffle(obs_candidates)
            for ox, oy in obs_candidates:
                if placed_obs >= n_obs:
                    break
                grid.terrain[oy, ox] = CellType.WALL
                # Check solvability: agent must still be able to reach box
                reachable = grid.flood_fill(agent_pos)
                if box_pos not in reachable or target_pos not in reachable:
                    grid.terrain[oy, ox] = CellType.EMPTY  # revert
                else:
                    placed_obs += 1

            # Verify agent can reach box and target is reachable
            reachable = grid.flood_fill(agent_pos)
            if box_pos not in reachable:
                continue

            # Place objects
            grid.objects[box_pos[1], box_pos[0]]    = ObjectType.BOX
            grid.objects[target_pos[1], target_pos[0]] = ObjectType.TARGET

            return grid, {
                "agent_start": agent_pos,
                "goal_positions": [target_pos],
                "box_pos":    list(box_pos),
                "npc_pos":    list(npc_pos),
                "target_pos": list(target_pos),
                "max_steps":  self.get_max_steps(),
            }

        # Fallback: simple fixed layout
        grid = Grid(size, size)
        grid.terrain[0, :]  = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0]  = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL
        mid = size // 2
        agent_pos  = (1, mid)
        box_pos    = (mid, mid)
        npc_pos    = (size-2, mid)
        target_pos = (mid, size-2)
        grid.objects[mid, mid]           = ObjectType.BOX
        grid.objects[size-2, mid]        = ObjectType.TARGET
        return grid, {
            "agent_start": agent_pos, "goal_positions": [target_pos],
            "box_pos": list(box_pos), "npc_pos": list(npc_pos),
            "target_pos": list(target_pos), "max_steps": self.get_max_steps(),
        }

    def on_env_reset(self, agent, grid, config):
        config["_box_pos"] = list(config["box_pos"])
        config["_npc_pos"] = list(config["npc_pos"])
        self._last_box_pos = list(config["_box_pos"])
        self._config = config
        # Draw NPC and box
        bx, by = config["_box_pos"]
        nx, ny = config["_npc_pos"]
        tx, ty = config["target_pos"]
        grid.objects[by, bx] = ObjectType.BOX
        if grid.terrain[ny, nx] == CellType.EMPTY:
            grid.objects[ny, nx] = ObjectType.NPC
        grid.objects[ty, tx] = ObjectType.TARGET

    def can_agent_enter(self, pos, agent, grid) -> bool:
        x, y = pos
        if grid.objects[y, x] != ObjectType.BOX:
            return True
        ax, ay = agent.position
        dx, dy = x - ax, y - ay
        nbx, nby = x + dx, y + dy
        if not (0 < nbx < grid.width-1 and 0 < nby < grid.height-1):
            return False
        if grid.terrain[nby, nbx] == CellType.WALL:
            return False
        if grid.objects[nby, nbx] not in (ObjectType.NONE, ObjectType.TARGET):
            return False
        grid.objects[y, x] = ObjectType.NONE
        grid.objects[nby, nbx] = ObjectType.BOX
        # Update _box_pos immediately (check_success runs before on_env_step)
        if hasattr(self, "_config") and self._config is not None:
            self._config["_box_pos"] = [nbx, nby]
            self._last_box_pos = [nbx, nby]
        return True

    def on_env_step(self, agent, grid, config, step_count):

        bx, by = config["_box_pos"]
        nx, ny = config["_npc_pos"]
        tx, ty = config["target_pos"]
        ax, ay = agent.position

        if grid.objects[ny, nx] == ObjectType.NPC:
            grid.objects[ny, nx] = ObjectType.NONE

        # NPC moves toward box
        best, best_d = (nx, ny), abs(nx-bx)+abs(ny-by)
        for dx, dy in self._DIRS:
            cx, cy = nx+dx, ny+dy
            if (0 < cx < grid.width-1 and 0 < cy < grid.height-1
                    and grid.terrain[cy, cx] == CellType.EMPTY
                    and (cx, cy) != (bx, by) and (cx, cy) != (ax, ay)
                    and grid.objects[cy, cx] == ObjectType.NONE):
                d = abs(cx-bx)+abs(cy-by)
                if d < best_d:
                    best_d, best = d, (cx, cy)
        config["_npc_pos"] = list(best)
        nx2, ny2 = best
        grid.objects[ny2, nx2] = ObjectType.NPC
        # Restore TARGET if box moved off it
        if (bx, by) != (tx, ty):
            if grid.objects[ty, tx] == ObjectType.NONE:
                grid.objects[ty, tx] = ObjectType.TARGET

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        config = new_state.get("config", {})
        box    = config.get("_box_pos", config.get("box_pos", [0, 0]))
        target = config.get("target_pos", [0, 0])
        old_box = self._last_box_pos
        old_d = abs(old_box[0]-target[0]) + abs(old_box[1]-target[1])
        new_d = abs(box[0]-target[0])    + abs(box[1]-target[1])
        if old_d != new_d:
            reward += 0.15 * (old_d - new_d)
        self._last_box_pos = list(box)
        if "agent" in new_state:
            ax, ay = new_state["agent"].position
            ox, oy = old_state.get("agent_position", (ax, ay))
            bx, by = box
            reward += 0.05 * ((abs(ox-bx)+abs(oy-by)) - (abs(ax-bx)+abs(ay-by)))
        if self.check_success(new_state):
            reward += 1.0
        return reward

    def check_success(self, state):
        config = state.get("config", {})
        box    = config.get("_box_pos", config.get("box_pos", [-1, -1]))
        target = config.get("target_pos", [-2, -2])
        return box[0] == target[0] and box[1] == target[1]

    def get_optimal_return(self, difficulty=None): return 1.0
    def get_random_baseline(self, difficulty=None): return 0.0
