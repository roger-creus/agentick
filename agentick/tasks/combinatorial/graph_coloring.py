"""GraphColoring - Visit all color zones in constraint-satisfying order.

MECHANICS:
  - Grid has N color zones (TARGET clusters) in distinct regions
  - Agent must visit ALL zones; but visiting two adjacent zones in wrong order
    gives a penalty (constraint violation)
  - Simplified: zones alternate (A,B,A,B...) — agent must visit all A's before B's
  - Correct full traversal → success
  - Tests constraint-aware planning in spatial domains
"""

import numpy as np
from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("GraphColoring-v0", tags=["combinatorial_logic", "constraint_satisfaction"])
class GraphColoringTask(TaskSpec):
    """Visit colored zones in constraint-satisfying order."""

    name = "GraphColoring-v0"
    description = "Visit all zones in constraint-satisfying order"
    capability_tags = ["combinatorial_logic", "constraint_satisfaction"]

    difficulty_configs = {
        # n_zones: graph regions to color | n_colors: palette size (more=harder chromatic number)
        # distractors: extra decoy zones that share borders (need revisiting)
        "easy":   DifficultyConfig(name="easy",   grid_size=7,  max_steps=100, params={"n_zones": 3,  "n_colors": 3, "distractors": 0}),
        "medium": DifficultyConfig(name="medium",  grid_size=10, max_steps=200, params={"n_zones": 5,  "n_colors": 4, "distractors": 1}),
        "hard":   DifficultyConfig(name="hard",    grid_size=13, max_steps=350, params={"n_zones": 7,  "n_colors": 4, "distractors": 2}),
        "expert": DifficultyConfig(name="expert",  grid_size=15, max_steps=500, params={"n_zones": 9,  "n_colors": 5, "distractors": 3}),
    }

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        n           = self.difficulty_config.params.get("n_zones", 3)
        n_colors    = self.difficulty_config.params.get("n_colors", 3)  # palette size (metadata)
        n_distractors = self.difficulty_config.params.get("distractors", 0)

        grid = Grid(size, size)
        grid.terrain[0, :]  = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0]  = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        agent_pos = (1, 1)

        free = [(x, y) for x in range(1, size - 1) for y in range(1, size - 1)
                if (x, y) != agent_pos]
        rng.shuffle(free)

        zone_centers = free[:n]
        color0 = [zone_centers[i] for i in range(n) if i % 2 == 0]
        color1 = [zone_centers[i] for i in range(n) if i % 2 == 1]
        used = set(zone_centers) | {agent_pos}

        for zx, zy in color0:
            grid.objects[zy, zx] = ObjectType.TARGET
        for zx, zy in color1:
            grid.objects[zy, zx] = ObjectType.SWITCH

        goal_pos = free[n] if len(free) > n else free[-1]
        grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.GOAL
        used.add(goal_pos)

        # Distractor zones: BOX objects that look like constraint nodes but are ignored
        distractor_positions = []
        for p in free[n + 1:]:
            if len(distractor_positions) >= n_distractors:
                break
            if p not in used:
                dx2, dy2 = p
                grid.objects[dy2, dx2] = ObjectType.BOX
                distractor_positions.append(p)
                used.add(p)

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [goal_pos],
            "color0_zones": color0,
            "color1_zones": color1,
            "n_zones": n,
            "n_colors": n_colors,
            "distractor_positions": distractor_positions,
            "max_steps": self.get_max_steps(),
        }

    # ── Hooks ────────────────────────────────────────────────────────────────

    def on_env_reset(self, agent, grid, config):
        """Init zone tracking; cache config for on_agent_moved."""
        config["_visited_color0"] = 0
        config["_visited_color1"] = 0
        config["_violation"] = False
        self._config = config
        self._last_n_visited = 0

    def on_agent_moved(self, pos, agent, grid):
        """Consume zone when stepped on — fires BEFORE reward/success."""
        config = getattr(self, "_config", {})
        c0 = config.get("color0_zones", [])
        c1 = config.get("color1_zones", [])
        ax, ay = pos

        if (ax, ay) in c0 and grid.objects[ay, ax] == ObjectType.TARGET:
            grid.objects[ay, ax] = ObjectType.NONE
            config["_visited_color0"] = config.get("_visited_color0", 0) + 1

        elif (ax, ay) in c1 and grid.objects[ay, ax] == ObjectType.SWITCH:
            grid.objects[ay, ax] = ObjectType.NONE
            config["_visited_color1"] = config.get("_visited_color1", 0) + 1
            # Constraint: must visit all color0 before any color1
            if config.get("_visited_color0", 0) < len(c0):
                config["_violation"] = True

    # ── Reward & success ─────────────────────────────────────────────────────

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        config = new_state.get("config", {})
        new_n = config.get("_visited_color0", 0) + config.get("_visited_color1", 0)
        if new_n > self._last_n_visited:
            if config.get("_violation", False):
                reward -= 0.2
            else:
                reward += 0.2
        self._last_n_visited = new_n
        # Approach shaping: toward nearest unvisited zone
        if "grid" in new_state and "agent_position" in new_state:
            from agentick.core.types import ObjectType
            g = new_state["grid"]
            zones = [(x,y) for y in range(g.height) for x in range(g.width)
                     if g.objects[y,x] in (ObjectType.TARGET, ObjectType.SWITCH)]
            if zones:
                ax, ay = new_state["agent_position"]
                ox, oy = old_state.get("agent_position", (ax, ay))
                nz_new = min(abs(ax-zx)+abs(ay-zy) for zx,zy in zones)
                nz_old = min(abs(ox-zx)+abs(oy-zy) for zx,zy in zones)
                reward += 0.05 * (nz_old - nz_new)  # stronger: outweighs step penalty
        if self.check_success(new_state):
            reward += 1.0
        return reward

    def check_success(self, state):
        """All zones visited IN CORRECT ORDER (color0 before color1) AND agent at goal."""
        config = state.get("config", {})
        if config.get("_violation", False):
            return False  # wrong color order = failure (ordering is the whole point)
        c0 = config.get("color0_zones", [])
        c1 = config.get("color1_zones", [])
        v0 = config.get("_visited_color0", 0)
        v1 = config.get("_visited_color1", 0)
        if v0 < len(c0) or v1 < len(c1):
            return False
        if "grid" not in state or "agent" not in state:
            return False
        x, y = state["agent"].position
        return bool(state["grid"].objects[y, x] == ObjectType.GOAL)

    def get_optimal_return(self, difficulty=None): return 1.0
    def get_random_baseline(self, difficulty=None): return 0.0
