"""Oracle bots for multi-agent tasks."""

from __future__ import annotations

from agentick.core.types import CellType, ObjectType
from agentick.oracles.base import OracleAgent
from agentick.oracles.registry import register_oracle


@register_oracle("CompetitiveTag-v0")
class CompetitiveTagOracle(OracleAgent):
    """Tag enemy NPCs by moving onto them.

    Strategy: chase nearest enemy, one step at a time (BFS).
    Avoid ICE safe zones when chasing (can't tag there).
    """

    def plan(self):
        enemies = self.api.get_entities_of_type("enemy")
        if not enemies:
            self.action_queue = [0]
            return

        ax, ay = self.api.agent_position
        grid = self.api.grid
        nearest = min(enemies, key=lambda e: e.distance)

        # Avoid safe zone positions (can't tag there)
        config = self.api.task_config
        safe_zones = set(map(tuple, config.get("safe_zone_positions", [])))

        # BFS to enemy, avoiding safe zones (except the enemy position itself)
        avoid = safe_zones - {nearest.position}
        path = self.api.bfs_path_positions(
            (ax, ay),
            nearest.position,
            avoid=avoid,
        )
        if path:
            actions = self.api.positions_to_actions(path)
            if actions:
                self.action_queue = [actions[0]]
                return

        # Fallback: ignore ICE avoidance
        path = self.api.bfs_path_positions(
            (ax, ay),
            nearest.position,
        )
        if path:
            actions = self.api.positions_to_actions(path)
            if actions:
                self.action_queue = [actions[0]]
                return

        self.action_queue = self.api.move_toward(*nearest.position)


@register_oracle("CooperativeTransport-v0")
class CooperativeTransportOracle(OracleAgent):
    """Push box toward target with NPC cooperation.

    Strategy: agent pushes box toward target. When the NPC blocks the direct
    push path, push the box sideways or even backwards to go around.
    Uses a simple planning approach: if the direct push is blocked, find
    any valid push direction and take it.
    """

    def __init__(self, env):
        super().__init__(env)
        self._stuck_counter = 0
        self._last_box_pos = None

    def reset(self, obs, info):
        self._stuck_counter = 0
        self._last_box_pos = None
        super().reset(obs, info)

    def _try_push(self, pdx, pdy, bx, by, grid, npc_pos, ax, ay):
        """Try to execute or navigate toward a push in direction (pdx, pdy).

        Returns an action int if possible, None otherwise.
        """
        push_from = (bx - pdx, by - pdy)
        pfx, pfy = push_from

        if not (0 < pfx < grid.width - 1 and 0 < pfy < grid.height - 1):
            return None
        if not grid.is_walkable((pfx, pfy)):
            return None

        land = (bx + pdx, by + pdy)
        lx, ly = land
        if not (0 < lx < grid.width - 1 and 0 < ly < grid.height - 1):
            return None
        if int(grid.terrain[ly, lx]) == int(CellType.WALL):
            return None
        land_obj = int(grid.objects[ly, lx])
        if land_obj not in (int(ObjectType.NONE), int(ObjectType.TARGET)):
            return None

        if push_from == npc_pos:
            return None

        if (ax, ay) == push_from:
            step = self.api.step_action(pdx, pdy)
            return step
        else:
            avoid = {(bx, by)}
            if npc_pos != (-1, -1):
                avoid.add(npc_pos)
            path = self.api.bfs_path_positions(
                (ax, ay),
                push_from,
                avoid=avoid,
            )
            if path:
                actions = self.api.positions_to_actions(path)
                if actions:
                    return actions[0]
        return None

    def plan(self):
        grid = self.api.grid
        config = self.api.task_config
        ax, ay = self.api.agent_position

        box_pos = config.get("_box_pos", config.get("box_pos"))
        target_pos = config.get("target_pos")

        if not box_pos or not target_pos:
            self.action_queue = [0]
            return

        bx, by = box_pos
        tx, ty = target_pos

        if bx == tx and by == ty:
            self.action_queue = [0]
            return

        npc_pos = tuple(config.get("_npc_pos", config.get("npc_pos", [-1, -1])))

        # Detect stuck state
        if self._last_box_pos == [bx, by]:
            self._stuck_counter += 1
        else:
            self._stuck_counter = 0
        self._last_box_pos = [bx, by]

        # Score each push direction
        push_options = []
        for ddx, ddy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            new_bx, new_by = bx + ddx, by + ddy
            new_dist = abs(new_bx - tx) + abs(new_by - ty)
            old_dist = abs(bx - tx) + abs(by - ty)
            improvement = old_dist - new_dist
            push_options.append((ddx, ddy, improvement))

        # Sort: best improvement first
        push_options.sort(key=lambda x: -x[2])

        # Phase 1: Try all pushes that improve distance
        for pdx, pdy, imp in push_options:
            if imp <= 0:
                break
            action = self._try_push(pdx, pdy, bx, by, grid, npc_pos, ax, ay)
            if action is not None:
                self.action_queue = [action]
                return

        # Phase 2: If stuck, try ANY valid push (including sideways/backwards)
        # This handles cases where the NPC blocks the direct path
        if self._stuck_counter >= 2:
            for pdx, pdy, _ in push_options:
                action = self._try_push(pdx, pdy, bx, by, grid, npc_pos, ax, ay)
                if action is not None:
                    self.action_queue = [action]
                    return

        # Phase 3: Move agent to let NPC reposition
        # The NPC wants to be on the opposite side of the box from the target.
        # If the agent is blocking that position, move away.
        push_dx = 1 if tx > bx else (-1 if tx < bx else 0)
        push_dy = 1 if ty > by else (-1 if ty < by else 0)
        npc_ideal_x = max(1, min(grid.width - 2, bx - push_dx))
        npc_ideal_y = max(1, min(grid.height - 2, by - push_dy))
        npc_ideal = (npc_ideal_x, npc_ideal_y)

        if (ax, ay) == npc_ideal or (
            abs(ax - npc_ideal[0]) + abs(ay - npc_ideal[1]) <= 1 and self._stuck_counter >= 1
        ):
            name_map = self.api.action_name_to_int
            deltas = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            move_names = ["move_up", "move_down", "move_left", "move_right"]
            best_action = None
            best_score = -999
            for i, (dx, dy) in enumerate(deltas):
                nx, ny = ax + dx, ay + dy
                if not (0 < nx < grid.width - 1 and 0 < ny < grid.height - 1):
                    continue
                if not grid.is_walkable((nx, ny)):
                    continue
                if (nx, ny) == (bx, by) or (nx, ny) == npc_pos:
                    continue
                dist_from_ideal = abs(nx - npc_ideal[0]) + abs(ny - npc_ideal[1])
                dist_to_box = abs(nx - bx) + abs(ny - by)
                score = dist_from_ideal * 10 - dist_to_box * 2
                if move_names[i] in name_map and score > best_score:
                    best_score = score
                    best_action = name_map[move_names[i]]
            if best_action is not None:
                self.action_queue = [best_action]
                return

        # Fallback: move toward box
        self.action_queue = self.api.move_toward(bx, by)
