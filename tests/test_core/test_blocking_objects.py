"""Tests for blocking-object handling in Grid, AgentickEnv, and action masking."""

from __future__ import annotations

from agentick.core.grid import Grid
from agentick.core.types import ObjectType

# ---------------------------------------------------------------------------
# Grid.bfs() and Grid.flood_fill() with check_objects
# ---------------------------------------------------------------------------


def _make_corridor_with_door(open_door: bool = False) -> Grid:
    """Create a 5x1 corridor with a closed (or open) door in the middle.

    Layout (y=0):  EMPTY  EMPTY  DOOR  EMPTY  EMPTY
    Agent would be at (0,0), goal at (4,0).
    """
    grid = Grid(height=1, width=5)
    grid.objects[0, 2] = ObjectType.DOOR
    if open_door:
        grid.metadata[0, 2] = 10  # open door convention
    return grid


def test_bfs_default_ignores_blocking_objects():
    """Default bfs() should path through closed doors (backward compat)."""
    grid = _make_corridor_with_door(open_door=False)
    path = grid.bfs((0, 0), (4, 0))
    assert path is not None
    assert (4, 0) in path


def test_bfs_check_objects_blocks_closed_door():
    """bfs(check_objects=True) should not path through a closed door."""
    grid = _make_corridor_with_door(open_door=False)
    path = grid.bfs((0, 0), (4, 0), check_objects=True)
    assert path is None


def test_bfs_check_objects_allows_open_door():
    """bfs(check_objects=True) should path through an open door (metadata>=10)."""
    grid = _make_corridor_with_door(open_door=True)
    path = grid.bfs((0, 0), (4, 0), check_objects=True)
    assert path is not None
    assert (4, 0) in path


def test_flood_fill_default_ignores_blocking_objects():
    """Default flood_fill should include cells behind closed doors."""
    grid = _make_corridor_with_door(open_door=False)
    reachable = grid.flood_fill((0, 0))
    assert (4, 0) in reachable


def test_flood_fill_check_objects_blocks_closed_door():
    """flood_fill(check_objects=True) should not reach behind closed door."""
    grid = _make_corridor_with_door(open_door=False)
    reachable = grid.flood_fill((0, 0), check_objects=True)
    assert (4, 0) not in reachable
    # Cells before the door should be reachable
    assert (0, 0) in reachable
    assert (1, 0) in reachable


def test_flood_fill_check_objects_allows_open_door():
    """flood_fill(check_objects=True) should reach behind open door."""
    grid = _make_corridor_with_door(open_door=True)
    reachable = grid.flood_fill((0, 0), check_objects=True)
    assert (4, 0) in reachable


def test_bfs_check_objects_blocks_lever():
    """bfs(check_objects=True) should not path through a lever."""
    grid = Grid(height=1, width=5)
    grid.objects[0, 2] = ObjectType.LEVER
    path = grid.bfs((0, 0), (4, 0), check_objects=True)
    assert path is None


def test_bfs_check_objects_blocks_switch():
    """bfs(check_objects=True) should not path through a switch."""
    grid = Grid(height=1, width=5)
    grid.objects[0, 2] = ObjectType.SWITCH
    path = grid.bfs((0, 0), (4, 0), check_objects=True)
    assert path is None


# ---------------------------------------------------------------------------
# AgentickEnv._move_agent() blocking
# ---------------------------------------------------------------------------


def test_env_blocks_movement_through_closed_door():
    """AgentickEnv should not let agent walk through a closed door."""
    from agentick.core.env import AgentickEnv
    from agentick.core.types import ActionType

    env = AgentickEnv(grid=Grid(height=1, width=5), render_mode="state_dict")
    env.grid.objects[0, 2] = ObjectType.DOOR
    env.agent.position = (1, 0)

    # Try to move right (into the door cell)
    env._move_agent(ActionType.MOVE_RIGHT)
    assert env.agent.position == (1, 0), "Agent should be blocked by closed door"


def test_env_allows_movement_through_open_door():
    """AgentickEnv should let agent walk through an open door."""
    from agentick.core.env import AgentickEnv
    from agentick.core.types import ActionType

    env = AgentickEnv(grid=Grid(height=1, width=5), render_mode="state_dict")
    env.grid.objects[0, 2] = ObjectType.DOOR
    env.grid.metadata[0, 2] = 10  # open

    env.agent.position = (1, 0)
    env._move_agent(ActionType.MOVE_RIGHT)
    assert env.agent.position == (2, 0), "Agent should pass through open door"


# ---------------------------------------------------------------------------
# Action mask with blocking objects
# ---------------------------------------------------------------------------


def test_action_mask_blocks_closed_door():
    """get_valid_actions() should mark movement into closed door as invalid."""
    from agentick.core.env import AgentickEnv
    from agentick.core.types import ActionType

    env = AgentickEnv(grid=Grid(height=3, width=3), render_mode="state_dict")
    # Place a closed door to the right of the agent
    env.agent.position = (0, 1)
    env.grid.objects[1, 1] = ObjectType.DOOR
    env._valid_actions_dirty = True

    mask = env.get_valid_actions()
    # MOVE_RIGHT = ActionType.MOVE_RIGHT.value = index in mask
    right_idx = ActionType.MOVE_RIGHT.value
    assert mask[right_idx] == 0, "Moving into closed door should be invalid"


def test_action_mask_allows_open_door():
    """get_valid_actions() should mark movement into open door as valid."""
    from agentick.core.env import AgentickEnv
    from agentick.core.types import ActionType

    env = AgentickEnv(grid=Grid(height=3, width=3), render_mode="state_dict")
    env.agent.position = (0, 1)
    env.grid.objects[1, 1] = ObjectType.DOOR
    env.grid.metadata[1, 1] = 10  # open
    env._valid_actions_dirty = True

    mask = env.get_valid_actions()
    right_idx = ActionType.MOVE_RIGHT.value
    assert mask[right_idx] == 1, "Moving into open door should be valid"
