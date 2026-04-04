"""RecipeAssembly - Craft an item by collecting ingredients in the correct order.

MECHANICS:
  - A RECIPE is shown visually in a dedicated corner zone:
      sequence of TARGET objects with metadata = expected ingredient type
      (14=GEM/herb, 17=SCROLL/mushroom, 19=ORB/crystal, 18=COIN/reagent)
  - Ingredients (GEM, SCROLL, ORB, COIN) scattered on the map
  - A CHEF NPC (immovable, walkable) at the crafting station in the map center
  - Agent must collect ingredients in the EXACT recipe order:
      collect step-1 ingredient → walk to chef → collect step-2 → chef → ...
  - Wrong ingredient type: small penalty, ingredient stays on map
  - Completing each step lights up the recipe slot (TARGET → GOAL in recipe zone)
  - Final step: walk to chef with last ingredient → SUCCESS
  - Decoy ingredients: extra items of wrong types to confuse
  - Difficulty: recipe length, decoys, obstacles, multiple crafting steps

VISIBILITY:
  - Recipe zone: top-right corner, TARGET objects with typed metadata
    visible in ALL modalities (pixels, ASCII, language)
  - Current step indicator: currently-needed ingredient is highlighted
  - Crafting station: NPC object (chef), always visible
"""

import numpy as np

from agentick.core.grid import Grid
from agentick.core.types import CellType, ObjectType
from agentick.tasks.base import TaskSpec
from agentick.tasks.configs import DifficultyConfig
from agentick.tasks.registry import register_task

# Ingredient types used in recipes (ObjectType value → object placed on grid)
_INGREDIENT_TYPES = [
    ObjectType.GEM,     # 14 - herb (purple)
    ObjectType.SCROLL,  # 17 - mushroom (tan)
    ObjectType.ORB,     # 19 - crystal (pink)
    ObjectType.COIN,    # 18 - reagent (gold)
]
_INGREDIENT_NAMES = {
    ObjectType.GEM: "herb",
    ObjectType.SCROLL: "mushroom",
    ObjectType.ORB: "crystal",
    ObjectType.COIN: "reagent",
}


@register_task("RecipeAssembly-v0", tags=["compositional_logic", "planning"])
class RecipeAssemblyTask(TaskSpec):
    """Collect ingredients in recipe order, delivering each to crafting station."""

    name = "RecipeAssembly-v0"
    description = "Follow recipe: collect ingredients in correct order at crafting station"
    capability_tags = ["compositional_logic", "planning"]

    difficulty_configs = {
        "easy": DifficultyConfig(
            name="easy",
            grid_size=9,
            max_steps=120,
            params={"recipe_length": 2, "n_decoys": 0, "n_obstacles": 0},
        ),
        "medium": DifficultyConfig(
            name="medium",
            grid_size=11,
            max_steps=220,
            params={"recipe_length": 3, "n_decoys": 2, "n_obstacles": 3},
        ),
        "hard": DifficultyConfig(
            name="hard",
            grid_size=14,
            max_steps=380,
            params={"recipe_length": 4, "n_decoys": 3, "n_obstacles": 5},
        ),
        "expert": DifficultyConfig(
            name="expert",
            grid_size=16,
            max_steps=600,
            params={"recipe_length": 5, "n_decoys": 4, "n_obstacles": 7},
        ),
    }

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        size = self.difficulty_config.grid_size
        recipe_length = self.difficulty_config.params.get("recipe_length", 2)
        n_decoys = self.difficulty_config.params.get("n_decoys", 0)
        n_obstacles = self.difficulty_config.params.get("n_obstacles", 0)

        grid = Grid(size, size)
        grid.terrain[0, :] = CellType.WALL
        grid.terrain[-1, :] = CellType.WALL
        grid.terrain[:, 0] = CellType.WALL
        grid.terrain[:, -1] = CellType.WALL

        # Agent starts at a random corner
        corners = [(1, 1), (size - 2, 1), (1, size - 2), (size - 2, size - 2)]
        agent_pos = tuple(corners[int(rng.integers(0, len(corners)))])

        # Crafting station (chef NPC): center of map
        station_pos = (size // 2, size // 2)
        grid.objects[station_pos[1], station_pos[0]] = ObjectType.NPC
        grid.metadata[station_pos[1], station_pos[0]] = 2  # facing down

        used = {agent_pos, station_pos}

        # Generate recipe: sequence of ingredient types (may repeat)
        recipe_types = list(rng.choice(len(_INGREDIENT_TYPES), size=recipe_length, replace=True))
        recipe = [_INGREDIENT_TYPES[i] for i in recipe_types]

        # Recipe zone: top-right corner, one cell per recipe step
        # Placed as a row of TARGET objects with metadata = ingredient ObjectType int
        recipe_zone = []
        rx_start = max(1, size - recipe_length - 1)
        ry = 1
        for step_i, ing_type in enumerate(recipe):
            rx = rx_start + step_i
            if 0 < rx < size - 1:
                rpos = (rx, ry)
                # Place recipe slot: TARGET with metadata = ingredient type int
                grid.objects[ry, rx] = ObjectType.TARGET
                grid.metadata[ry, rx] = int(ing_type)
                recipe_zone.append(rpos)
                used.add(rpos)

        # Place ingredient objects scattered around (avoid recipe zone and station)
        free = [
            (x, y) for x in range(1, size - 1) for y in range(1, size - 1)
            if (x, y) not in used
        ]
        rng.shuffle(free)

        ingredient_positions = {}  # ingredient_type → list of positions
        for ing_type in recipe:
            for pos in free:
                if pos not in used:
                    ix, iy = pos
                    grid.objects[iy, ix] = ing_type
                    ingredient_positions.setdefault(ing_type, []).append(pos)
                    used.add(pos)
                    break

        # Decoy ingredients: wrong types scattered around
        decoy_positions = []
        decoy_types_available = [t for t in _INGREDIENT_TYPES if t not in recipe]
        if not decoy_types_available:
            decoy_types_available = _INGREDIENT_TYPES[:]
        for i in range(n_decoys):
            for pos in free:
                if pos not in used:
                    dx2, dy2 = pos
                    dt = decoy_types_available[i % len(decoy_types_available)]
                    grid.objects[dy2, dx2] = dt
                    decoy_positions.append(pos)
                    used.add(pos)
                    break

        # Place interior walls — flood-fill to keep everything reachable
        critical = [agent_pos, station_pos] + list(ingredient_positions.get(t, [None])[0]
                                                     for t in recipe
                                                     if ingredient_positions.get(t))
        critical = [c for c in critical if c is not None]
        wall_cands = [p for p in free if p not in used]
        rng.shuffle(wall_cands)
        placed_walls = 0
        for p in wall_cands:
            if placed_walls >= n_obstacles:
                break
            wx, wy = p
            grid.terrain[wy, wx] = CellType.WALL
            reachable = grid.flood_fill(agent_pos)
            if all(c in reachable for c in critical):
                placed_walls += 1
                used.add(p)
            else:
                grid.terrain[wy, wx] = CellType.EMPTY

        # Serialize ingredient positions
        ing_pos_serialized = {
            int(k): [list(v2) for v2 in vals]
            for k, vals in ingredient_positions.items()
        }

        return grid, {
            "agent_start": agent_pos,
            "goal_positions": [station_pos],
            "station_pos": station_pos,
            "recipe": [int(t) for t in recipe],
            "recipe_zone": recipe_zone,
            "ingredient_positions": ing_pos_serialized,
            "decoy_positions": decoy_positions,
            "recipe_length": recipe_length,
            "max_steps": self.get_max_steps(),
        }

    def can_agent_enter(self, pos, agent, grid):
        """Allow agent to walk onto chef NPC position."""
        x, y = pos
        if grid.objects[y, x] == ObjectType.NPC:
            config = getattr(self, "_config", {})
            station_pos = config.get("station_pos", (-1, -1))
            if (x, y) == tuple(station_pos):
                return True
        return True

    def on_env_reset(self, agent, grid, config):
        agent.inventory.clear()
        config["_step"] = 0           # current recipe step index (0-based)
        config["_steps_done"] = 0     # number of steps completed
        config["_at_station"] = False
        config["_last_penalty"] = False
        self._last_steps_done = 0
        self._config = config
        self._update_recipe_display(grid, config)

    def _update_recipe_display(self, grid, config):
        """Update recipe zone: completed steps → GOAL, pending → TARGET."""
        recipe_zone = config.get("recipe_zone", [])
        steps_done = config.get("_steps_done", 0)
        recipe = config.get("recipe", [])
        for i, (rx, ry) in enumerate(recipe_zone):
            if i < steps_done:
                grid.objects[ry, rx] = ObjectType.GOAL   # completed step
            else:
                grid.objects[ry, rx] = ObjectType.TARGET  # pending step
                if i < len(recipe):
                    grid.metadata[ry, rx] = recipe[i]

    def on_agent_moved(self, pos, agent, grid):
        """Handle ingredient pickup and crafting station interactions."""
        config = getattr(self, "_config", {})
        x, y = pos
        obj = grid.objects[y, x]
        step = config.get("_step", 0)
        recipe = config.get("recipe", [])
        station_pos = config.get("station_pos", (-1, -1))
        config["_last_penalty"] = False

        # At crafting station (chef NPC): check if holding correct next ingredient
        if (x, y) == tuple(station_pos):
            held = [e for e in agent.inventory if e.entity_type == "ingredient"]
            if held and step < len(recipe):
                needed = recipe[step]
                # Find ingredient of the needed type in inventory
                found = next((e for e in held if e.properties.get("ing_type") == int(needed)), None)
                if found:
                    agent.inventory.remove(found)
                    config["_steps_done"] = config.get("_steps_done", 0) + 1
                    config["_step"] = step + 1
                    self._update_recipe_display(grid, config)
            return

        # Wrong ingredient pickup: penalize
        if obj in _INGREDIENT_TYPES:
            # Is this ingredient the currently needed type?
            needed = recipe[step] if step < len(recipe) else None
            if needed is not None and obj == needed:
                # Correct ingredient - pick it up
                grid.objects[y, x] = ObjectType.NONE
                from agentick.core.entity import Entity
                e = Entity(
                    id=f"ing_{x}_{y}",
                    entity_type="ingredient",
                    position=pos,
                )
                e.properties["ing_type"] = int(obj)
                agent.inventory.append(e)
            else:
                # Wrong ingredient - small penalty (leave in place)
                config["_last_penalty"] = True

    def compute_dense_reward(self, old_state, action, new_state, info):
        reward = -0.01
        config = new_state.get("config", {})

        if config.get("_last_penalty", False):
            reward -= 0.1

        new_done = config.get("_steps_done", 0)
        if new_done > self._last_steps_done:
            reward += 0.5 * (new_done - self._last_steps_done)
        self._last_steps_done = new_done

        step = config.get("_step", 0)
        recipe = config.get("recipe", [])
        if "agent_position" in new_state and step < len(recipe):
            ax, ay = new_state["agent_position"]
            ox, oy = old_state.get("agent_position", (ax, ay))
            g = new_state.get("grid")
            if g is not None:
                needed_type = recipe[step]
                # If holding the ingredient, go to station; else go to ingredient
                held = any(
                    e.properties.get("ing_type") == int(needed_type)
                    for e in new_state.get("agent", new_state.get("agent", None)).inventory
                    if hasattr(new_state.get("agent", None), "inventory")
                ) if "agent" in new_state else False
                station_pos = config.get("station_pos", (-1, -1))
                if held:
                    tx, ty = station_pos
                else:
                    # Find nearest ingredient of needed type
                    ings = [
                        (xi, yi) for yi in range(g.height) for xi in range(g.width)
                        if g.objects[yi, xi] == needed_type
                    ]
                    if ings:
                        tx, ty = min(ings, key=lambda q: abs(q[0] - ax) + abs(q[1] - ay))
                    else:
                        tx, ty = station_pos
                reward += 0.03 * (abs(ox - tx) + abs(oy - ty) - abs(ax - tx) - abs(ay - ty))

        if self.check_success(new_state):
            reward += 1.0
        return reward

    def check_success(self, state):
        """All recipe steps completed."""
        config = state.get("config", {})
        recipe_length = config.get("recipe_length", 2)
        return config.get("_steps_done", 0) >= recipe_length

    def get_optimal_return(self, difficulty=None):
        return 1.0

    def get_random_baseline(self, difficulty=None):
        return 0.0
