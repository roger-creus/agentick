"""Oracle bots for skill tasks."""

from __future__ import annotations

from agentick.core.types import CellType
from agentick.oracles.base import OracleAgent
from agentick.oracles.registry import register_oracle


@register_oracle("ToolUse-v0")
class ToolUseOracle(OracleAgent):
    """Collect tools then cross barriers to reach goal.

    Tools auto-pickup on walk-over. Uses inventory to determine which barriers
    can be crossed. Collects tools in phases, crossing passable barriers to
    reach tools behind them.
    """

    # Tool name → entity type on grid; tool name → barrier terrain type
    _TOOL_ENTITY = {"torch": "gem", "bridge": "key", "hammer": "tool"}

    def plan(self):
        config = self.api.task_config
        ax, ay = self.api.agent_position
        agent = self.api.agent

        barrier_info = config.get("barrier_info", {})

        # Check which tools we currently have in inventory
        inv_types = set()
        for item in agent.inventory:
            inv_types.add(item.entity_type)

        # Build passable barrier set based on current inventory.
        # Use terrain_ok={0} so BFS only walks EMPTY cells + extra_passable.
        # This prevents routing through HAZARD/WATER without matching tools.
        passable = set()
        passable.add((ax, ay))  # Agent may be on barrier terrain after crossing
        for tool_name, binfo in barrier_info.items():
            if tool_name in inv_types:
                for cell in binfo.get("cells", []):
                    passable.add(tuple(cell))

        _EMPTY = {0}

        # Phase 1: Collect tools reachable via EMPTY terrain only
        # (no barrier crossing — collect all accessible tools first)
        phase1 = []
        for tool_name in barrier_info:
            if tool_name in inv_types:
                continue
            entity_type = self._TOOL_ENTITY.get(tool_name)
            if not entity_type:
                continue
            for item in self.api.get_entities_of_type(entity_type):
                path = self.api.bfs_path_positions(
                    (ax, ay),
                    item.position,
                    terrain_ok=_EMPTY,
                )
                if path:
                    phase1.append((item, path))
        if phase1:
            _, best_path = min(phase1, key=lambda x: len(x[1]))
            actions = self.api.positions_to_actions(best_path)
            if actions:
                self.action_queue = [actions[0]]
                return

        # Phase 2: Collect tools reachable via EMPTY + collected barriers
        phase2 = []
        for tool_name in barrier_info:
            if tool_name in inv_types:
                continue
            entity_type = self._TOOL_ENTITY.get(tool_name)
            if not entity_type:
                continue
            for item in self.api.get_entities_of_type(entity_type):
                path = self.api.bfs_path_positions(
                    (ax, ay),
                    item.position,
                    terrain_ok=_EMPTY,
                    extra_passable=passable,
                )
                if path:
                    phase2.append((item, path))
        if phase2:
            _, best_path = min(phase2, key=lambda x: len(x[1]))
            actions = self.api.positions_to_actions(best_path)
            if actions:
                self.action_queue = [actions[0]]
                return

        # All reachable tools collected - head toward goal
        goal = self.api.get_nearest("goal")
        if goal:
            path = self.api.bfs_path_positions(
                (ax, ay),
                goal.position,
                terrain_ok=_EMPTY,
                extra_passable=passable,
            )
            if path:
                actions = self.api.positions_to_actions(path)
                if actions:
                    self.action_queue = [actions[0]]
                    return
            self.action_queue = self.api.move_toward(*goal.position)


@register_oracle("EmergentStrategy-v0")
class EmergentStrategyOracle(OracleAgent):
    """Collect nearby keys for bonus, then rush to goal before barrier closes."""

    def plan(self):
        goal = self.api.get_nearest("goal")
        if not goal:
            return

        # Collect keys that are on the way (very close)
        keys = self.api.get_entities_of_type("key")
        close_keys = [k for k in keys if k.distance <= 3]
        if close_keys:
            nearest_key = min(close_keys, key=lambda k: k.distance)
            self.action_queue = self.api.move_to(*nearest_key.position)
            return

        # Head to goal
        self.action_queue = self.api.move_to(*goal.position)


@register_oracle("MultiRoomEscape-v0")
class MultiRoomEscapeOracle(OracleAgent):
    """Navigate rooms to reach goal, avoiding guards and hazards.

    Strategy: BFS to goal avoiding NPC cells + predicted positions + hazards.
    Step one at a time to react to guard movement.
    Fall back through progressively less cautious avoidance levels.
    """

    def plan(self):
        ax, ay = self.api.agent_position
        goal = self.api.get_nearest("goal")
        if not goal:
            return

        grid = self.api.grid
        config = self.api.task_config
        guards = config.get("_guard_positions", [])
        dirs = config.get("_guard_dirs", [])
        _DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        # Build guard avoidance with prediction
        avoid_wide = set()
        avoid_exact = set()
        for e in self.api.get_entities():
            if e.entity_type in ("npc", "enemy"):
                avoid_exact.add(e.position)
                avoid_wide.add(e.position)
                for dx, dy in _DIRS:
                    avoid_wide.add((e.position[0] + dx, e.position[1] + dy))

        # Predict next guard positions (wide avoidance only)
        for i, (gx, gy) in enumerate(guards):
            if i < len(dirs):
                d = dirs[i]
                ddx, ddy = _DIRS[d]
                nx, ny = gx + ddx, gy + ddy
                if (
                    0 < nx < grid.width - 1
                    and 0 < ny < grid.height - 1
                    and int(grid.terrain[ny, nx]) == int(CellType.EMPTY)
                ):
                    avoid_wide.add((nx, ny))
                    for dx, dy in _DIRS:
                        avoid_wide.add((nx + dx, ny + dy))

        # Hazard terrain
        hazard_cells = set()
        for y in range(grid.height):
            for x in range(grid.width):
                if int(grid.terrain[y, x]) == int(CellType.HAZARD):
                    hazard_cells.add((x, y))

        gp = goal.position

        # Level 1: wide guard avoidance + hazards
        avoid = (avoid_wide | hazard_cells) - {gp}
        path = self.api.bfs_path_positions((ax, ay), gp, avoid=avoid)
        if path:
            actions = self.api.positions_to_actions(path)
            if actions:
                self.action_queue = [actions[0]]
                return

        # Level 2: exact guard avoidance + hazards
        avoid = (avoid_exact | hazard_cells) - {gp}
        path = self.api.bfs_path_positions((ax, ay), gp, avoid=avoid)
        if path:
            actions = self.api.positions_to_actions(path)
            if actions:
                self.action_queue = [actions[0]]
                return

        # Level 3: hazards only
        avoid = hazard_cells - {gp}
        path = self.api.bfs_path_positions((ax, ay), gp, avoid=avoid)
        if path:
            actions = self.api.positions_to_actions(path)
            if actions:
                self.action_queue = [actions[0]]
                return

        # Level 4: no avoidance
        path = self.api.bfs_path_positions((ax, ay), gp)
        if path:
            actions = self.api.positions_to_actions(path)
            if actions:
                self.action_queue = [actions[0]]
                return

        self.action_queue = self.api.move_toward(*gp)


@register_oracle("RecipeAssembly-v0")
class RecipeAssemblyOracle(OracleAgent):
    """Follow recipe: collect ingredients in correct order, deliver to station.

    New mechanic (redesigned):
      - Recipe is a sequence of ingredient types (GEM, SCROLL, ORB, COIN)
      - Agent must collect the CURRENT step's ingredient, bring it to station
      - Then repeat for next step
      - Oracle reads recipe from task_config and navigates accordingly
    """

    _INGREDIENT_TYPE_NAMES = {
        14: "gem",    # GEM/herb
        17: "scroll", # SCROLL/mushroom
        19: "orb",    # ORB/crystal
        18: "coin",   # COIN/reagent
    }

    def plan(self):
        config = self.api.task_config
        recipe = config.get("recipe", [])
        step = config.get("_step", 0)
        ax, ay = self.api.agent_position

        if step >= len(recipe):
            # All steps done — should have already succeeded
            self.action_queue = [0]
            return

        # Currently needed ingredient type
        needed_type = recipe[step]
        entity_type_name = self._INGREDIENT_TYPE_NAMES.get(needed_type)

        # Check if we're already holding the needed ingredient
        agent = self.api.agent
        holding_needed = any(
            e.entity_type == "ingredient" and e.properties.get("ing_type") == needed_type
            for e in agent.inventory
        )

        if holding_needed:
            # Go to crafting station (GOAL object)
            goal = self.api.get_nearest("goal")
            if goal:
                self.action_queue = self.api.move_to(*goal.position)
            return

        # Find the needed ingredient on the map
        if entity_type_name:
            ings = self.api.get_entities_of_type(entity_type_name)
            if ings:
                nearest = min(ings, key=lambda e: e.distance)
                self.action_queue = self.api.move_to(*nearest.position)
                return

        # Fallback: go to goal
        goal = self.api.get_nearest("goal")
        if goal:
            self.action_queue = self.api.move_to(*goal.position)


@register_oracle("ResourceManagement-v0")
class ResourceManagementOracle(OracleAgent):
    """Keep all energy stations charged, then reach the goal.

    Redesigned task: RESOURCE objects drain over time (metadata = energy level).
    Oracle monitors energy levels and prioritizes visiting critical stations.
    After all stations are safe, heads toward goal.
    """

    def _get_station_levels(self) -> list[tuple[int, int, int]]:
        """Return list of (energy_level, x, y) for all RESOURCE objects."""
        from agentick.core.types import ObjectType as OT
        grid = self.api.grid
        stations = []
        for y in range(grid.height):
            for x in range(grid.width):
                if int(grid.objects[y, x]) == int(OT.RESOURCE):
                    level = int(grid.metadata[y, x])
                    stations.append((level, x, y))
        return stations

    def plan(self):
        config = self.api.task_config
        goal = self.api.get_nearest("goal")
        ax, ay = self.api.agent_position

        stations = self._get_station_levels()
        if not stations:
            if goal:
                self.action_queue = self.api.move_to(*goal.position)
            return

        # Find most critical station (lowest energy)
        stations.sort(key=lambda s: s[0])
        most_critical_level, cx, cy = stations[0]

        # If any station is critically low (<= 30%), go recharge it urgently
        if most_critical_level <= 30:
            self.action_queue = self.api.move_to(cx, cy)
            return

        # If any station is low (<= 60%), go recharge it unless goal is close
        if goal:
            goal_dist = abs(ax - goal.position[0]) + abs(ay - goal.position[1])
        else:
            goal_dist = 999

        if most_critical_level <= 60 and goal_dist > 3:
            self.action_queue = self.api.move_to(cx, cy)
            return

        # Otherwise: head toward goal
        if goal:
            self.action_queue = self.api.move_to(*goal.position)
