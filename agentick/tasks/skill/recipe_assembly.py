"""RecipeAssembly - Collect all ingredients then reach the cooking station.

MECHANICS:
  - Multiple ingredient items (KEY objects) scattered on the grid
  - Agent must collect ALL ingredients (auto-pickup by stepping on them)
  - Then reach the GOAL (cooking station)
  - Wrong order = fine (just need all before goal)
  - Success = agent has collected all ingredients AND is at GOAL
"""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task


@register_task("RecipeAssembly-v0", tags=["compositional_logic", "planning"])
class RecipeAssemblyTask(TaskSpec):
    """Collect all ingredients then reach the cooking station."""

    name = "RecipeAssembly-v0"
    description = "Collect ingredients then reach cooking station"
    capability_tags = ["compositional_logic", "planning"]

    difficulty_configs = {
        "easy":   DifficultyConfig(name="easy",   grid_size=7,  max_steps=100, params={"n_ingredients": 2, "n_decoys": 0, "n_obstacles": 0}),
        "medium": DifficultyConfig(name="medium",  grid_size=10, max_steps=180, params={"n_ingredients": 3, "n_decoys": 2, "n_obstacles": 3}),
        "hard":   DifficultyConfig(name="hard",    grid_size=13, max_steps=300, params={"n_ingredients": 4, "n_decoys": 3, "n_obstacles": 5}),
        "expert": DifficultyConfig(name="expert",  grid_size=15, max_steps=480, params={"n_ingredients": 5, "n_decoys": 4, "n_obstacles": 7}),
    }

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size        = self.difficulty_config.grid_size
        n           = self.difficulty_config.params.get("n_ingredients", 2)
        n_decoys    = self.difficulty_config.params.get("n_decoys", 0)
        n_obstacles = self.difficulty_config.params.get("n_obstacles", 0)

        grid = Grid(size, size)
        grid.terrain[0, :]  = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0]  = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        agent_pos = (1, 1)
        goal_pos  = (size-2, size-2)
        grid.objects[goal_pos[1], goal_pos[0]] = ObjectType.GOAL

        free = [(x, y) for x in range(1, size-1) for y in range(1, size-1)
                if (x, y) != agent_pos and (x, y) != goal_pos]
        rng.shuffle(free)
        ingredient_positions = free[:n]
        used = {agent_pos, goal_pos} | set(ingredient_positions)

        for ix, iy in ingredient_positions:
            grid.objects[iy, ix] = ObjectType.KEY

        # Decoys: SWITCH objects (look different from KEY but add noise)
        decoy_positions = []
        for p in free[n:]:
            if len(decoy_positions) >= n_decoys:
                break
            if p not in used:
                dx2, dy2 = p
                grid.objects[dy2, dx2] = ObjectType.SWITCH
                decoy_positions.append(p)
                used.add(p)

        # Obstacle walls — flood-fill check
        wall_positions = []
        wall_candidates = [p for p in free if p not in used]
        critical = [agent_pos, goal_pos] + list(ingredient_positions)
        for p in wall_candidates:
            if len(wall_positions) >= n_obstacles:
                break
            wx, wy = p
            grid.terrain[wy, wx] = CellType.WALL
            reachable = grid.flood_fill(agent_pos)
            if all(q in reachable for q in critical):
                wall_positions.append(p)
                used.add(p)
            else:
                grid.terrain[wy, wx] = CellType.EMPTY

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [goal_pos],
            "ingredient_positions": ingredient_positions,
            "decoy_positions": decoy_positions,
            "n_ingredients": n,
            "max_steps": self.get_max_steps(),
        }

    # ── Auto-collect ingredients ──────────────────────────────────────────────
    # (Handled by TaskEnv._move_agent → on_agent_moved)

    def on_env_reset(self, agent, grid, config):
        agent.inventory.clear()  # prevent inventory leak between episodes
        self._last_n_ingredients = 0

    def on_agent_moved(self, pos, agent, grid):
        """Auto-pickup ingredient (KEY) when agent steps on it."""
        from agentick.core.entity import Entity
        x, y = pos
        if grid.objects[y, x] == ObjectType.KEY:
            grid.objects[y, x] = ObjectType.NONE
            agent.inventory.append(
                Entity(id=f"ingredient_{x}_{y}", entity_type="ingredient", position=pos)
            )

    # ── Reward & success ─────────────────────────────────────────────────────

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        if "agent" not in new_state or "config" not in new_state:
            return reward

        agent = new_state["agent"]
        config = new_state.get("config", {})
        n_needed = config.get("n_ingredients", 1)
        n_have = sum(1 for e in agent.inventory if e.entity_type == "ingredient")

        # Reward collecting each ingredient (use instance var to avoid mutable agent ref bug)
        if n_have > self._last_n_ingredients:
            reward += 0.3 * (n_have - self._last_n_ingredients)
        self._last_n_ingredients = n_have

        # When we have all ingredients: move toward goal
        if n_have >= n_needed:
            goal = config.get("goal_positions", [None])[0]
            if goal and "agent_position" in new_state:
                ax, ay = new_state["agent_position"]
                ox, oy = old_state.get("agent_position", (ax, ay))
                reward += 0.05 * (abs(ox-goal[0])+abs(oy-goal[1]) - abs(ax-goal[0])-abs(ay-goal[1]))
        else:
            # Move toward nearest uncollected ingredient
            from agentick.core.types import ObjectType as OT
            if "grid" in new_state and "agent_position" in new_state:
                grid = new_state["grid"]
                ings = [(x,y) for y in range(grid.height) for x in range(grid.width)
                        if grid.objects[y,x] == OT.KEY]
                if ings:
                    ax, ay = new_state["agent_position"]
                    ox, oy = old_state.get("agent_position", (ax, ay))
                    nd_new = min(abs(ax-ix)+abs(ay-iy) for ix,iy in ings)
                    nd_old = min(abs(ox-ix)+abs(oy-iy) for ix,iy in ings)
                    reward += 0.02 * (nd_old - nd_new)

        if self.check_success(new_state):
            reward += 1.0
        return reward

    def check_success(self, state):
        """Agent at goal AND has all ingredients."""
        if "grid" not in state or "agent" not in state or "config" not in state:
            return False
        x, y = state["agent"].position
        if state["grid"].objects[y, x] != ObjectType.GOAL:
            return False
        config = state.get("config", {})
        n_needed = config.get("n_ingredients", 1)
        n_have = sum(1 for e in state["agent"].inventory if e.entity_type == "ingredient")
        return n_have >= n_needed

    def get_optimal_return(self, difficulty=None): return 1.0
    def get_random_baseline(self, difficulty=None): return 0.0
