"""Oracle bots for skill tasks."""

from __future__ import annotations

from agentick.core.types import CellType
from agentick.oracles.base import OracleAgent
from agentick.oracles.registry import register_oracle


@register_oracle("ToolUse-v0")
class ToolUseOracle(OracleAgent):
    """Navigate the zigzag path, collecting tools and taking shortcuts.

    Strategy (shortcut-first):
      1. Build the set of barrier cells passable with the current inventory.
      2. Attempt BFS to the goal through those passable shortcuts.
      3. If no path exists (need a tool before the next barrier), find the
         nearest uncollected tool that is reachable via empty terrain + any
         already-passable barriers, and navigate to it.
      4. Fall back to move_toward goal.

    Tools auto-pickup on walk-over, so navigating through the tool's cell is
    sufficient to acquire it.
    """

    # Tool entity type name on grid → tool name used in config
    _TOOL_ENTITY = {"torch": "gem", "bridge": "key", "hammer": "tool"}

    def plan(self):
        config = self.api.task_config
        ax, ay = self.api.agent_position
        agent = self.api.agent

        barrier_info = config.get("barrier_info", {})

        # Inventory: collect the set of tool names currently held
        inv_types: set[str] = {item.entity_type for item in agent.inventory}

        # Build extra_passable: barrier cells that can be crossed with held tools.
        # Also include the agent's current position (they may be standing on a barrier
        # cell they just crossed).
        passable: set[tuple[int, int]] = {(ax, ay)}
        for tool_name, binfo in barrier_info.items():
            if tool_name in inv_types:
                for cell in binfo.get("cells", []):
                    passable.add(tuple(cell))

        empty_terrain = {0}

        # --- Step 1: Try to reach goal directly using shortcuts ---
        goal = self.api.get_nearest("goal")
        if goal:
            path = self.api.bfs_path_positions(
                (ax, ay),
                goal.position,
                terrain_ok=empty_terrain,
                extra_passable=passable,
            )
            if path:
                actions = self.api.positions_to_actions(path)
                if actions:
                    self.action_queue = [actions[0]]
                    return

        # --- Step 2: Cannot reach goal yet — collect nearest accessible tool ---
        # Try all three tool entity types; pick shortest reachable path.
        candidates: list[tuple[object, list]] = []
        for tool_name in barrier_info:
            if tool_name in inv_types:
                continue  # already have this tool
            entity_type = self._TOOL_ENTITY.get(tool_name)
            if not entity_type:
                continue
            for item in self.api.get_entities_of_type(entity_type):
                # Try reaching tool via empty terrain + already passable barriers
                path = self.api.bfs_path_positions(
                    (ax, ay),
                    item.position,
                    terrain_ok=empty_terrain,
                    extra_passable=passable,
                )
                if path:
                    candidates.append((item, path))
        if candidates:
            _, best_path = min(candidates, key=lambda x: len(x[1]))
            actions = self.api.positions_to_actions(best_path)
            if actions:
                self.action_queue = [actions[0]]
                return

        # --- Step 3: Fallback — move toward goal directly ---
        if goal:
            self.action_queue = self.api.move_toward(*goal.position)


@register_oracle("EmergentStrategy-v0")
class EmergentStrategyOracle(OracleAgent):
    """Scare sheep out of barrier gaps, collect keys, then cross to goal.

    Strategy:
      1. Check if any barrier gap is blocked by a SHEEP.
      2. If all gaps blocked, approach the nearest gap-blocking sheep to scare it.
         Position one cell above the gap so the sheep flees downward (away).
      3. Once a gap is clear, collect nearby keys for bonus reward.
      4. Head through the clear gap to the goal.
    """

    def plan(self):
        ax, ay = self.api.agent_position
        config = self.api.task_config
        goal = self.api.get_nearest("goal")
        if not goal:
            return

        barrier_row = config.get("barrier_row", 0)
        gap_xs = config.get("gap_xs", [])
        sheep_positions = set(map(tuple, config.get("_live_sheep", [])))

        # Identify which gaps are blocked by sheep
        blocked_gaps = [gx for gx in gap_xs if (gx, barrier_row) in sheep_positions]
        clear_gaps = [gx for gx in gap_xs if gx not in blocked_gaps]

        # Sheep also block cells above/below the gap — check those too
        # A gap is "usable" if we can walk into it (no sheep in the gap cell)
        # Also treat sheep as avoid-cells for BFS
        avoid = set(sheep_positions)

        if not clear_gaps:
            # All gaps blocked — approach the nearest sheep-in-gap to scare it.
            # Position ourselves one row above the gap; the sheep will flee south.
            best_target = None
            best_dist = float("inf")
            for gx in blocked_gaps:
                # Try to stand one cell above the gap (same column)
                approach_y = barrier_row - 1
                approach_pos = (gx, approach_y)
                d = abs(ax - gx) + abs(ay - approach_y)
                if d < best_dist:
                    best_dist = d
                    best_target = approach_pos
            if best_target:
                # Move toward the approach position, avoiding sheep
                path = self.api.bfs_path_positions(
                    (ax, ay), best_target, avoid=avoid - {best_target}
                )
                if path:
                    actions = self.api.positions_to_actions(path)
                    if actions:
                        self.action_queue = [actions[0]]
                        return
                # Fallback: just move toward it
                self.action_queue = self.api.move_toward(*best_target)
            return

        # At least one gap is clear — collect nearby keys first
        keys = self.api.get_entities_of_type("key")
        close_keys = [k for k in keys if k.distance <= 4]
        if close_keys:
            nearest_key = min(close_keys, key=lambda k: k.distance)
            path = self.api.bfs_path_positions(
                (ax, ay), nearest_key.position, avoid=avoid
            )
            if path:
                actions = self.api.positions_to_actions(path)
                if actions:
                    self.action_queue = [actions[0]]
                    return
            self.action_queue = self.api.move_to(*nearest_key.position)
            return

        # Head to goal through a clear gap, avoiding sheep
        path = self.api.bfs_path_positions(
            (ax, ay), goal.position, avoid=avoid
        )
        if path:
            actions = self.api.positions_to_actions(path)
            if actions:
                self.action_queue = [actions[0]]
                return

        # Fallback: direct move
        self.action_queue = self.api.move_toward(*goal.position)


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
    """Keep all energy stations alive for the full episode.

    No goal — success is surviving max_steps without any station dying.
    Oracle monitors energy levels and always visits the most critical station.
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
        stations = self._get_station_levels()
        if not stations:
            return

        # Always go to the most critical station (lowest energy)
        stations.sort(key=lambda s: s[0])
        most_critical_level, cx, cy = stations[0]
        self.action_queue = self.api.move_to(cx, cy)
