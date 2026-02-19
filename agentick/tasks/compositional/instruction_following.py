"""InstructionFollowing - Navigate to the goal specified by an instruction.

MECHANICS:
  - N colored zones on the grid (TARGET cells in different quadrants)
  - An instruction (integer index) encoded in the observation specifies which zone
  - Agent must go to the correct zone matching the instruction
  - Wrong zone = penalty; correct zone = success
  - Tests grounded instruction following without language
"""

import numpy as np
from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task

_DIRS = [(0,-1),(0,1),(-1,0),(1,0)]


@register_task("InstructionFollowing-v0", tags=["language", "grounding", "instruction"])
class InstructionFollowingTask(TaskSpec):
    """Navigate to the zone specified by the encoded instruction."""

    name = "InstructionFollowing-v0"
    description = "Follow instruction to reach correct goal zone"
    capability_tags = ["language", "grounding", "instruction"]

    difficulty_configs = {
        # n_zones: target zones | n_distractors: fake zones | n_guards: patrolling NPCs
        "easy":   DifficultyConfig(name="easy",   grid_size=7,  max_steps=80,  params={"n_zones": 2, "n_distractors": 0, "n_guards": 0}),
        "medium": DifficultyConfig(name="medium",  grid_size=10, max_steps=150, params={"n_zones": 3, "n_distractors": 1, "n_guards": 0}),
        "hard":   DifficultyConfig(name="hard",    grid_size=13, max_steps=220, params={"n_zones": 4, "n_distractors": 2, "n_guards": 1}),
        "expert": DifficultyConfig(name="expert",  grid_size=15, max_steps=320, params={"n_zones": 5, "n_distractors": 3, "n_guards": 2}),
    }

    _DIRS = [(0,-1),(0,1),(-1,0),(1,0)]

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size          = self.difficulty_config.grid_size
        n             = self.difficulty_config.params.get("n_zones", 2)
        n_distractors = self.difficulty_config.params.get("n_distractors", 0)
        n_guards      = self.difficulty_config.params.get("n_guards", 0)

        grid = Grid(size, size)
        grid.terrain[0, :]  = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0]  = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        agent_pos = (1, 1)

        zone_positions = []
        lo = max(2, size // 3)
        hi = max(lo + 1, size - 1 - size // 3)
        quadrant_centers = [
            (lo,  lo), (hi,  lo), (lo,  hi), (hi,  hi), (size//2, size//2),
        ]
        for i in range(n):
            zx, zy = quadrant_centers[i % len(quadrant_centers)]
            zx = max(1, min(size-2, zx))
            zy = max(1, min(size-2, zy))
            if (zx, zy) == agent_pos:
                zx = min(size-2, zx + 1)
            zone_positions.append((zx, zy))

        instruction = int(rng.integers(0, n))
        true_goal = zone_positions[instruction]
        used = {agent_pos} | set(zone_positions)

        for i, (zx, zy) in enumerate(zone_positions):
            if i == instruction:
                grid.objects[zy, zx] = ObjectType.GOAL
            else:
                grid.objects[zy, zx] = ObjectType.TARGET

        # Distractors: extra TARGET cells not in zone_positions (visual noise)
        free = [(x, y) for x in range(1, size-1) for y in range(1, size-1)
                if (x, y) not in used]
        rng.shuffle(free)
        distractor_positions = []
        for p in free[:n_distractors]:
            dx2, dy2 = p
            grid.objects[dy2, dx2] = ObjectType.BOX  # different type from real zones
            distractor_positions.append(p)
            used.add(p)

        # Guards: NPC objects placed at distance from agent
        guard_candidates = [p for p in free[n_distractors:] if p not in used
                            and abs(p[0]-agent_pos[0])+abs(p[1]-agent_pos[1]) > 2
                            and p != true_goal]
        rng.shuffle(guard_candidates)
        guard_positions = guard_candidates[:n_guards]
        for gx, gy in guard_positions:
            grid.objects[gy, gx] = ObjectType.NPC
            used.add((gx, gy))

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [true_goal],
            "zone_positions": zone_positions,
            "instruction": instruction,
            "distractor_positions": distractor_positions,
            "_guard_positions": guard_positions,
            "_guard_dirs": [int(rng.integers(0, 4)) for _ in guard_positions],
            "_guard_seed": int(rng.integers(0, 2**31)),
            "max_steps": self.get_max_steps(),
        }

    def on_env_reset(self, agent, grid, config):
        """Reset wrong-zone flag, guard state; cache config for on_agent_moved."""
        config["_wrong_zone"] = False
        config["_guard_collision"] = False
        config["_guard_rng"] = np.random.default_rng(config.get("_guard_seed", 0))
        self._config = config

    def on_agent_moved(self, pos, agent, grid):
        """Detect wrong zone / guard collision immediately — fires BEFORE reward/done checks."""
        x, y = pos
        config = getattr(self, "_config", {})
        true_goal = config.get("goal_positions", [None])[0]
        if grid.objects[y, x] == ObjectType.TARGET and (x, y) != tuple(true_goal or ()):
            config["_wrong_zone"] = True
        if grid.objects[y, x] == ObjectType.NPC:
            config["_guard_collision"] = True

    def on_env_step(self, agent, grid, config, step_count):
        """Move guards and check NPC-onto-agent collision."""
        guards = config.get("_guard_positions", [])
        dirs   = config.get("_guard_dirs", [])
        rng    = config.get("_guard_rng")
        ax, ay = agent.position
        if not guards or rng is None:
            return
        for gx, gy in guards:
            if grid.objects[gy, gx] == ObjectType.NPC:
                grid.objects[gy, gx] = ObjectType.NONE
        new_g, new_d = [], []
        for i, (gx, gy) in enumerate(guards):
            d = dirs[i]; dx, dy = self._DIRS[d]; nx, ny = gx+dx, gy+dy
            if (0 < nx < grid.width-1 and 0 < ny < grid.height-1
                    and grid.terrain[ny, nx] == CellType.EMPTY
                    and grid.objects[ny, nx] not in (ObjectType.GOAL, ObjectType.TARGET)):
                new_g.append((nx, ny))
            else:
                d = int(rng.integers(0, 4)); new_g.append((gx, gy))
            new_d.append(d)
            if new_g[-1] == (ax, ay):
                config["_guard_collision"] = True
        config["_guard_positions"] = new_g
        config["_guard_dirs"] = new_d
        for gx, gy in new_g:
            if grid.terrain[gy, gx] == CellType.EMPTY:
                grid.objects[gy, gx] = ObjectType.NPC

    def compute_sparse_reward(self, old_state, action, new_state, info):
        if new_state.get("config", {}).get("_wrong_zone", False):
            return -0.5
        if self.check_success(new_state):
            return 1.0
        return 0.0

    def compute_dense_reward(self, old_state, action, new_state, info):
        config = new_state.get("config", {})
        if config.get("_wrong_zone", False):
            return -0.5
        reward = -0.01
        goal = config.get("goal_positions", [None])[0]
        if goal and "agent" in new_state:
            ax, ay = new_state["agent"].position
            ox, oy = old_state.get('agent_position', new_state['agent'].position)
            old_d = abs(ox-goal[0]) + abs(oy-goal[1])
            new_d = abs(ax-goal[0]) + abs(ay-goal[1])
            reward += 0.05 * (old_d - new_d)
        if self.check_success(new_state):
            reward += 1.0
        return reward

    def check_success(self, state):
        """True success: agent reached the CORRECT goal zone (no guard collision)."""
        config = state.get("config", {})
        if config.get("_guard_collision", False) or config.get("_wrong_zone", False):
            return False
        if "grid" not in state or "agent" not in state:
            return False
        x, y = state["agent"].position
        return bool(state["grid"].objects[y, x] == ObjectType.GOAL)

    def check_done(self, state):
        """Episode ends on correct zone, wrong zone, or guard collision."""
        if self.check_success(state):
            return True
        config = state.get("config", {})
        return config.get("_wrong_zone", False) or config.get("_guard_collision", False)

    def get_optimal_return(self, difficulty=None): return 1.0
    def get_random_baseline(self, difficulty=None): return 0.0
