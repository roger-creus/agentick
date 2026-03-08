"""Shared helpers for oracle bots."""

from __future__ import annotations

from agentick.core.types import ActionType, Direction

# ---------------------------------------------------------------------------
# Face-and-interact with a solid adjacent object
# ---------------------------------------------------------------------------

# Map from (dx, dy) delta to the move action that sets orientation toward it
DELTA_TO_MOVE = {
    (0, -1): int(ActionType.MOVE_UP),
    (0, 1): int(ActionType.MOVE_DOWN),
    (-1, 0): int(ActionType.MOVE_LEFT),
    (1, 0): int(ActionType.MOVE_RIGHT),
}

# Map from (dx, dy) delta to the Direction enum the agent will face
DELTA_TO_DIR = {
    (0, -1): Direction.NORTH,
    (0, 1): Direction.SOUTH,
    (-1, 0): Direction.WEST,
    (1, 0): Direction.EAST,
}

INTERACT = int(ActionType.INTERACT)


def interact_adjacent(
    agent_pos, agent_orientation, target_pos, grid, api,
    extra_passable=None, avoid=None,
):
    """Return action list to interact with a solid object at *target_pos*.

    The agent must stand adjacent, face the target, then INTERACT.

    Returns:
        A list with a single action integer (one step at a time), or an
        empty list if no path can be found.

    Args:
        extra_passable: Optional set of positions the BFS should treat as
            walkable even if they contain blocking objects (e.g. activated
            switches the agent can walk through).

    Logic:
        - If agent is ON target -> move to adjacent walkable cell first
        - If adjacent AND facing target -> [INTERACT]
        - If adjacent but NOT facing -> [MOVE toward target] (blocked, but
          updates orientation)
        - If not adjacent -> first step of BFS path to a walkable cell
          adjacent to target
    """
    ax, ay = agent_pos
    tx, ty = target_pos
    dx, dy = tx - ax, ty - ay

    # --- Agent is ON the target: step off first ---
    if dx == 0 and dy == 0:
        for ddx, ddy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            nx, ny = ax + ddx, ay + ddy
            if (
                grid.in_bounds((nx, ny))
                and grid.is_walkable((nx, ny))
                and not grid.is_object_blocking((nx, ny))
            ):
                return [DELTA_TO_MOVE[(ddx, ddy)]]
        return []

    # --- Already adjacent? ---
    if abs(dx) + abs(dy) == 1:
        needed_dir = DELTA_TO_DIR.get((dx, dy))
        if needed_dir is not None and agent_orientation == needed_dir:
            return [INTERACT]
        # Face the target: issue a move toward it (blocked by solid object,
        # but the engine still updates orientation)
        move = DELTA_TO_MOVE.get((dx, dy))
        if move is not None:
            return [move]
        return []

    # --- Not adjacent: BFS to a walkable cell next to the target ---
    ep = set(extra_passable) if extra_passable else set()

    adj_cells = []
    for ddx, ddy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
        nx, ny = tx + ddx, ty + ddy
        pos = (nx, ny)
        if not grid.in_bounds(pos) or not grid.is_walkable(pos):
            continue
        # Accept non-blocking cells and extra_passable cells (e.g. activated
        # switches the task allows walking through).
        if not grid.is_object_blocking(pos) or pos in ep:
            adj_cells.append(pos)

    if not adj_cells:
        return []

    # If the agent is currently on a blocking cell, mark it as extra_passable
    # so BFS can start from there.
    if grid.is_object_blocking(agent_pos):
        ep.add(agent_pos)
    passable = ep or None

    # Try each adjacent cell, pick shortest reachable path
    best_path = None
    for cell in adj_cells:
        path = api.bfs_path_positions(agent_pos, cell, extra_passable=passable, avoid=avoid)
        if path and (best_path is None or len(path) < len(best_path)):
            best_path = path

    if best_path:
        actions = api.positions_to_actions(best_path)
        if actions:
            return [actions[0]]

    # Fallback: move_toward the nearest adjacent cell
    nearest = min(adj_cells, key=lambda p: abs(p[0] - ax) + abs(p[1] - ay))
    fallback = api.move_toward(*nearest)
    return fallback if fallback else []
