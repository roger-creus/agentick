"""CooperativeTransport - Push a heavy box to the target with NPC cooperation.

MECHANICS:
  - Heavy BOX requires cooperative pushing: agent pushes, NPC helps from other side
  - NPC actively pushes box toward target when adjacent and on correct side
  - Box moves when agent pushes (standard Sokoban push)
  - NPC also pushes box toward target independently when positioned correctly
  - Success = box reaches TARGET position
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
        "easy": DifficultyConfig(
            name="easy",
            grid_size=7,
            max_steps=100,
            params={"n_obstacles": 0},
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=10,
            max_steps=200,
            params={"n_obstacles": 2},
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=13,
            max_steps=350,
            params={"n_obstacles": 4},
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=15,
            max_steps=500,
            params={"n_obstacles": 6},
        ),
    }

    _DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        n_obs = self.difficulty_config.params.get("n_obstacles", 0)

        for attempt in range(30):
            grid = Grid(size, size)
            grid.terrain[0, :] = CellType.WALL
            grid.terrain[-1, :] = CellType.WALL
            grid.terrain[:, 0] = CellType.WALL
            grid.terrain[:, -1] = CellType.WALL

            # Place box near center, target farther away
            mid = size // 2
            box_pos = (mid, mid)

            # Target in a random direction from box
            direction = int(rng.integers(0, 4))
            dx, dy = self._DIRS[direction]
            dist = int(rng.integers(2, max(3, size // 2)))
            tx = max(1, min(size - 2, box_pos[0] + dx * dist))
            ty = max(1, min(size - 2, box_pos[1] + dy * dist))
            target_pos = (tx, ty)

            if target_pos == box_pos:
                continue

            # Agent behind box (opposite of push direction)
            # NPC starts beside the box (perpendicular axis) so it doesn't
            # block the push path between box and target.
            push_dx = 1 if tx > box_pos[0] else (-1 if tx < box_pos[0] else 0)
            push_dy = 1 if ty > box_pos[1] else (-1 if ty < box_pos[1] else 0)

            if push_dx != 0:
                agent_pos = (box_pos[0] - push_dx, box_pos[1])
                # NPC beside box on perpendicular axis (not blocking push path)
                npc_pos = (box_pos[0], box_pos[1] + (1 if rng.random() > 0.5 else -1))
            else:
                agent_pos = (box_pos[0], box_pos[1] - push_dy)
                # NPC beside box on perpendicular axis
                npc_pos = (box_pos[0] + (1 if rng.random() > 0.5 else -1), box_pos[1])

            # Clamp positions
            agent_pos = (
                max(1, min(size - 2, agent_pos[0])),
                max(1, min(size - 2, agent_pos[1])),
            )
            npc_pos = (
                max(1, min(size - 2, npc_pos[0])),
                max(1, min(size - 2, npc_pos[1])),
            )

            positions = [agent_pos, box_pos, npc_pos, target_pos]
            if len(set(positions)) < 4:
                continue
            if not all(0 < p[0] < size - 1 and 0 < p[1] < size - 1 for p in positions):
                continue

            # Place obstacles
            placed_obs = 0
            obs_candidates = [
                (x, y)
                for x in range(1, size - 1)
                for y in range(1, size - 1)
                if (x, y) not in set(positions)
            ]
            rng.shuffle(obs_candidates)
            for ox, oy in obs_candidates:
                if placed_obs >= n_obs:
                    break
                grid.terrain[oy, ox] = CellType.WALL
                reachable = grid.flood_fill(agent_pos)
                if box_pos not in reachable or target_pos not in reachable:
                    grid.terrain[oy, ox] = CellType.EMPTY
                else:
                    placed_obs += 1

            # Verify solvability
            reachable = grid.flood_fill(agent_pos)
            if box_pos not in reachable:
                continue

            grid.objects[box_pos[1], box_pos[0]] = ObjectType.BOX
            grid.objects[target_pos[1], target_pos[0]] = ObjectType.TARGET

            return grid, {
                "agent_start": agent_pos,
                "goal_positions": [target_pos],
                "box_pos": list(box_pos),
                "npc_pos": list(npc_pos),
                "target_pos": list(target_pos),
                "max_steps": self.get_max_steps(),
            }

        # Fallback
        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL
        mid = size // 2
        agent_pos = (1, mid)
        box_pos = (mid, mid)
        npc_pos = (size - 2, mid)
        target_pos = (mid, size - 2)
        grid.objects[mid, mid] = ObjectType.BOX
        grid.objects[size - 2, mid] = ObjectType.TARGET
        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [target_pos],
            "box_pos": list(box_pos),
            "npc_pos": list(npc_pos),
            "target_pos": list(target_pos),
            "max_steps": self.get_max_steps(),
        }

    def on_env_reset(self, agent, grid, config):
        config["_box_pos"] = list(config["box_pos"])
        config["_npc_pos"] = list(config["npc_pos"])
        self._last_box_pos = list(config["_box_pos"])
        self._config = config
        bx, by = config["_box_pos"]
        nx, ny = config["_npc_pos"]
        tx, ty = config["target_pos"]
        grid.objects[by, bx] = ObjectType.BOX
        if grid.terrain[ny, nx] == CellType.EMPTY:
            grid.objects[ny, nx] = ObjectType.NPC
            grid.metadata[ny, nx] = 2  # default facing down
        grid.objects[ty, tx] = ObjectType.TARGET

    def can_agent_enter(self, pos, agent, grid) -> bool:
        x, y = pos
        if grid.objects[y, x] != ObjectType.BOX:
            return True
        ax, ay = agent.position
        dx, dy = x - ax, y - ay
        nbx, nby = x + dx, y + dy
        if not (0 < nbx < grid.width - 1 and 0 < nby < grid.height - 1):
            return False
        if grid.terrain[nby, nbx] == CellType.WALL:
            return False
        if grid.objects[nby, nbx] not in (ObjectType.NONE, ObjectType.TARGET):
            return False
        grid.objects[y, x] = ObjectType.NONE
        grid.objects[nby, nbx] = ObjectType.BOX
        if hasattr(self, "_config") and self._config is not None:
            self._config["_box_pos"] = [nbx, nby]
            self._last_box_pos = [nbx, nby]
        return True

    def _npc_push_box(self, grid, config, agent_pos):
        """NPC pushes box toward target only when agent is also adjacent (cooperation)."""
        bx, by = config["_box_pos"]
        nx, ny = config["_npc_pos"]
        tx, ty = config["target_pos"]
        ax, ay = agent_pos

        # NPC must be adjacent to box
        if abs(nx - bx) + abs(ny - by) != 1:
            return False

        # Agent must also be adjacent to box for cooperative push
        if abs(ax - bx) + abs(ay - by) > 2:
            return False

        # Determine push direction (from NPC toward box)
        push_dx = bx - nx
        push_dy = by - ny

        # Only push if it moves box closer to target
        new_bx, new_by = bx + push_dx, by + push_dy
        old_dist = abs(bx - tx) + abs(by - ty)
        new_dist = abs(new_bx - tx) + abs(new_by - ty)
        if new_dist >= old_dist:
            return False

        # Check if push destination is valid
        if not (0 < new_bx < grid.width - 1 and 0 < new_by < grid.height - 1):
            return False
        if grid.terrain[new_by, new_bx] == CellType.WALL:
            return False
        if grid.objects[new_by, new_bx] not in (ObjectType.NONE, ObjectType.TARGET):
            return False

        # Execute push
        grid.objects[by, bx] = ObjectType.NONE
        grid.objects[new_by, new_bx] = ObjectType.BOX

        # NPC moves into box's old position
        if grid.objects[ny, nx] == ObjectType.NPC:
            grid.objects[ny, nx] = ObjectType.NONE
        config["_box_pos"] = [new_bx, new_by]
        config["_npc_pos"] = [bx, by]

        return True

    def on_env_step(self, agent, grid, config, step_count):
        bx, by = config["_box_pos"]
        nx, ny = config["_npc_pos"]
        old_nx, old_ny = nx, ny  # save for direction calculation
        tx, ty = config["target_pos"]
        ax, ay = agent.position

        # Clear old NPC position
        if grid.objects[ny, nx] == ObjectType.NPC:
            grid.objects[ny, nx] = ObjectType.NONE
            grid.metadata[ny, nx] = 0

        # Try NPC push first (every other step to avoid constant pushing)
        pushed = False
        if step_count % 2 == 0:
            pushed = self._npc_push_box(grid, config, (ax, ay))

        if not pushed:
            # NPC moves toward box (to get into push position)
            bx, by = config["_box_pos"]
            nx, ny = config["_npc_pos"]

            # Find the push position: opposite side of box from target
            ideal_x = bx - (1 if tx > bx else (-1 if tx < bx else 0))
            ideal_y = by - (1 if ty > by else (-1 if ty < by else 0))
            ideal_x = max(1, min(grid.width - 2, ideal_x))
            ideal_y = max(1, min(grid.height - 2, ideal_y))

            best, best_d = (nx, ny), abs(nx - ideal_x) + abs(ny - ideal_y)
            for dx, dy in self._DIRS:
                cx, cy = nx + dx, ny + dy
                if (
                    0 < cx < grid.width - 1
                    and 0 < cy < grid.height - 1
                    and grid.terrain[cy, cx] == CellType.EMPTY
                    and (cx, cy) != (bx, by)
                    and (cx, cy) != (ax, ay)
                    and grid.objects[cy, cx] in (ObjectType.NONE, ObjectType.TARGET)
                ):
                    d = abs(cx - ideal_x) + abs(cy - ideal_y)
                    if d < best_d:
                        best_d, best = d, (cx, cy)
            config["_npc_pos"] = list(best)

        # Redraw NPC with direction metadata
        nx2, ny2 = config["_npc_pos"]
        if grid.terrain[ny2, nx2] == CellType.EMPTY:
            grid.objects[ny2, nx2] = ObjectType.NPC
            ddx, ddy = nx2 - old_nx, ny2 - old_ny
            if ddx > 0:
                grid.metadata[ny2, nx2] = 1  # right
            elif ddx < 0:
                grid.metadata[ny2, nx2] = 3  # left
            elif ddy < 0:
                grid.metadata[ny2, nx2] = 0  # up
            elif ddy > 0:
                grid.metadata[ny2, nx2] = 2  # down
            else:
                grid.metadata[ny2, nx2] = 2  # default down

        # Restore TARGET if box moved off it
        bx, by = config["_box_pos"]
        if (bx, by) != (tx, ty):
            if grid.objects[ty, tx] == ObjectType.NONE:
                grid.objects[ty, tx] = ObjectType.TARGET

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        config = new_state.get("config", {})
        box = config.get("_box_pos", config.get("box_pos", [0, 0]))
        target = config.get("target_pos", [0, 0])
        old_box = self._last_box_pos
        old_d = abs(old_box[0] - target[0]) + abs(old_box[1] - target[1])
        new_d = abs(box[0] - target[0]) + abs(box[1] - target[1])
        if old_d != new_d:
            reward += 0.15 * (old_d - new_d)
        self._last_box_pos = list(box)
        if "agent" in new_state:
            ax, ay = new_state["agent"].position
            ox, oy = old_state.get("agent_position", (ax, ay))
            bx, by = box
            reward += 0.05 * ((abs(ox - bx) + abs(oy - by)) - (abs(ax - bx) + abs(ay - by)))
        if self.check_success(new_state):
            reward += 1.0
        return reward

    def check_success(self, state):
        config = state.get("config", {})
        box = config.get("_box_pos", config.get("box_pos", [-1, -1]))
        target = config.get("target_pos", [-2, -2])
        return box[0] == target[0] and box[1] == target[1]

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0
