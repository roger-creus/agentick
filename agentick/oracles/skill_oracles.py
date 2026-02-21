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
    """Collect all ingredients (KEY objects), then reach goal."""

    def plan(self):
        config = self.api.task_config
        all_collected = config.get("_all_collected", False)

        if not all_collected:
            keys = self.api.get_entities_of_type("key")
            if keys:
                nearest = min(keys, key=lambda k: k.distance)
                self.action_queue = self.api.move_to(*nearest.position)
                return

        goal = self.api.get_nearest("goal")
        if goal:
            self.action_queue = self.api.move_to(*goal.position)


@register_oracle("ResourceManagement-v0")
class ResourceManagementOracle(OracleAgent):
    """Collect resources as needed, avoid hazards, and reach the goal.

    The task stores energy/health as integers in ``task_config["_energy"]``
    and ``task_config["_health"]``, NOT on the ``agent`` object (which is
    always 1.0).  The oracle reads from ``task_config`` and plans a
    hazard-aware path.
    """

    def _hazard_cells(self) -> set[tuple[int, int]]:
        """Return the set of all HAZARD terrain positions."""
        grid = self.api.grid
        cells: set[tuple[int, int]] = set()
        for y in range(grid.height):
            for x in range(grid.width):
                if int(grid.terrain[y, x]) == int(CellType.HAZARD):
                    cells.add((x, y))
        return cells

    def _safe_move_to(self, tx: int, ty: int) -> list[int]:
        """BFS path that avoids hazard cells when possible."""
        ax, ay = self.api.agent_position
        hazards = self._hazard_cells()
        # Try hazard-avoiding path first
        path = self.api.bfs_path_positions((ax, ay), (tx, ty), avoid=hazards)
        if path:
            return self.api.positions_to_actions(path)
        # Fall back to normal path (through hazards) if no safe route
        return self.api.move_to(tx, ty)

    def _count_hazards_on_path(self, tx: int, ty: int) -> int:
        """Count hazard cells on the shortest path to (tx, ty)."""
        ax, ay = self.api.agent_position
        path = self.api.bfs_path_positions((ax, ay), (tx, ty))
        if not path:
            return 0
        hazards = self._hazard_cells()
        return sum(1 for pos in path if pos in hazards)

    def plan(self):
        config = self.api.task_config
        goal = self.api.get_nearest("goal")
        ax, ay = self.api.agent_position

        # Read actual integer energy/health from task config
        energy = config.get("_energy", 0)
        health = config.get("_health", 0)

        # Estimate distance (and energy cost) to goal
        if goal:
            goal_dist = abs(ax - goal.position[0]) + abs(ay - goal.position[1])
        else:
            goal_dist = 0

        # Count hazard cells on the direct path to goal
        hazards_to_goal = self._count_hazards_on_path(*goal.position) if goal else 0

        coins = self.api.get_entities_of_type("coin")
        potions = self.api.get_entities_of_type("potion")

        # Health critical: need potions if health won't survive hazards
        # Each hazard cell costs 1 health, so we need health > hazards_to_goal.
        # Also refill if health is just 1 (one hazard step = death).
        if health <= hazards_to_goal or health <= 1:
            if potions:
                nearest = min(potions, key=lambda p: p.distance)
                self.action_queue = self._safe_move_to(*nearest.position)
                return

        # Energy critical: need coins if energy won't last to the goal
        # Energy drains 1 per step; water cells cost an extra 1.
        # Add a safety margin of 3 steps.
        if energy < goal_dist + 3:
            if coins:
                nearest = min(coins, key=lambda c: c.distance)
                self.action_queue = self._safe_move_to(*nearest.position)
                return

        # Moderate: pick up nearby resources opportunistically
        if energy < goal_dist + 15:
            if coins:
                nearest = min(coins, key=lambda c: c.distance)
                if nearest.distance <= 4:
                    self.action_queue = self._safe_move_to(*nearest.position)
                    return

        if health <= 3:
            if potions:
                nearest = min(potions, key=lambda p: p.distance)
                if nearest.distance <= 4:
                    self.action_queue = self._safe_move_to(*nearest.position)
                    return

        # Go to goal using a hazard-avoiding path
        if goal:
            self.action_queue = self._safe_move_to(*goal.position)
