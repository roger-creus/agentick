"""Tests for ActionSpace and action utilities."""

import numpy as np
import pytest

from agentick.core.actions import (
    ActionSpace,
    ActionType,
    action_to_direction,
    compute_action_mask,
    get_move_delta,
    is_movement_action,
)
from agentick.core.types import Direction


def test_action_space_basic():
    """Test basic action space."""
    space = ActionSpace()

    assert space.n_actions == 6
    assert space.contains(ActionType.NOOP)
    assert space.contains(ActionType.MOVE_UP)
    assert space.contains(ActionType.INTERACT)
    assert not space.contains(ActionType.ROTATE_LEFT)


def test_action_space_extended():
    """Test extended action space."""
    space = ActionSpace(extended=True)

    assert space.n_actions == 9
    assert space.contains(ActionType.ROTATE_LEFT)
    assert space.contains(ActionType.ROTATE_RIGHT)
    assert space.contains(ActionType.MOVE_FORWARD)


def test_action_space_custom():
    """Test custom action space."""
    actions = [ActionType.NOOP, ActionType.MOVE_UP, ActionType.MOVE_DOWN]
    space = ActionSpace(actions=actions)

    assert space.n_actions == 3
    assert space.contains(ActionType.MOVE_UP)
    assert not space.contains(ActionType.INTERACT)


def test_action_type_conversion():
    """Test conversion between action index and type."""
    space = ActionSpace()

    # Get action type from index
    action_type = space.get_action_type(0)
    assert action_type == ActionType.NOOP

    # Get index from action type
    idx = space.get_action_idx(ActionType.MOVE_UP)
    assert idx == 1
    assert space.get_action_type(idx) == ActionType.MOVE_UP


def test_action_names():
    """Test action name conversion."""
    space = ActionSpace()

    # Get name from index
    assert space.get_action_name(0) == "noop"
    assert space.get_action_name(1) == "move_up"

    # Parse name to index
    assert space.parse_action_name("move_up") == 1
    assert space.parse_action_name("MOVE_UP") == 1
    assert space.parse_action_name("Move Up") == 1

    # Invalid name
    with pytest.raises(ValueError):
        space.parse_action_name("invalid_action")


def test_get_all_action_names():
    """Test getting all action names."""
    space = ActionSpace()
    names = space.get_all_action_names()

    assert len(names) == 6
    assert "noop" in names
    assert "move_up" in names
    assert "interact" in names


def test_action_space_sample():
    """Test sampling from action space."""
    space = ActionSpace()

    # Sample without RNG
    action = space.sample()
    assert 0 <= action < space.n_actions

    # Sample with RNG
    rng = np.random.default_rng(42)
    action = space.sample(rng)
    assert 0 <= action < space.n_actions


def test_compute_action_mask_basic():
    """Test basic action mask computation."""
    space = ActionSpace()

    # 5x5 grid, all walkable
    grid = np.ones((5, 5), dtype=bool)
    agent_pos = (2, 2)

    mask = compute_action_mask(space, agent_pos, grid)

    # All movement actions should be valid from center
    assert mask[space.get_action_idx(ActionType.NOOP)]
    assert mask[space.get_action_idx(ActionType.MOVE_UP)]
    assert mask[space.get_action_idx(ActionType.MOVE_DOWN)]
    assert mask[space.get_action_idx(ActionType.MOVE_LEFT)]
    assert mask[space.get_action_idx(ActionType.MOVE_RIGHT)]


def test_compute_action_mask_walls():
    """Test action mask with walls."""
    space = ActionSpace()

    # 5x5 grid with walls
    grid = np.ones((5, 5), dtype=bool)
    grid[1, 2] = False  # Wall above
    grid[2, 3] = False  # Wall to the right

    agent_pos = (2, 2)
    mask = compute_action_mask(space, agent_pos, grid)

    # Movement blocked by walls
    assert not mask[space.get_action_idx(ActionType.MOVE_UP)]
    assert not mask[space.get_action_idx(ActionType.MOVE_RIGHT)]

    # Other directions still valid
    assert mask[space.get_action_idx(ActionType.MOVE_DOWN)]
    assert mask[space.get_action_idx(ActionType.MOVE_LEFT)]


def test_compute_action_mask_corner():
    """Test action mask at grid corner."""
    space = ActionSpace()

    grid = np.ones((5, 5), dtype=bool)
    agent_pos = (0, 0)  # Top-left corner

    mask = compute_action_mask(space, agent_pos, grid)

    # Can only move down and right from corner
    assert not mask[space.get_action_idx(ActionType.MOVE_UP)]
    assert not mask[space.get_action_idx(ActionType.MOVE_LEFT)]
    assert mask[space.get_action_idx(ActionType.MOVE_DOWN)]
    assert mask[space.get_action_idx(ActionType.MOVE_RIGHT)]


def test_compute_action_mask_interact():
    """Test action mask: INTERACT is always valid (no-op if nothing interactable)."""
    space = ActionSpace()

    grid = np.ones((5, 5), dtype=bool)
    agent_pos = (2, 2)

    # INTERACT is always valid regardless of can_interact flag
    mask = compute_action_mask(space, agent_pos, grid, can_interact=False)
    assert mask[space.get_action_idx(ActionType.INTERACT)]

    mask = compute_action_mask(space, agent_pos, grid, can_interact=True)
    assert mask[space.get_action_idx(ActionType.INTERACT)]


def test_get_move_delta():
    """Test getting movement deltas."""
    assert get_move_delta(ActionType.MOVE_UP) == (0, -1)
    assert get_move_delta(ActionType.MOVE_DOWN) == (0, 1)
    assert get_move_delta(ActionType.MOVE_LEFT) == (-1, 0)
    assert get_move_delta(ActionType.MOVE_RIGHT) == (1, 0)
    assert get_move_delta(ActionType.INTERACT) is None


def test_is_movement_action():
    """Test movement action detection."""
    assert is_movement_action(ActionType.MOVE_UP)
    assert is_movement_action(ActionType.MOVE_DOWN)
    assert is_movement_action(ActionType.MOVE_FORWARD)
    assert not is_movement_action(ActionType.INTERACT)
    assert not is_movement_action(ActionType.NOOP)


def test_action_to_direction():
    """Test action to direction conversion."""
    assert action_to_direction(ActionType.MOVE_UP) == Direction.NORTH
    assert action_to_direction(ActionType.MOVE_DOWN) == Direction.SOUTH
    assert action_to_direction(ActionType.MOVE_LEFT) == Direction.WEST
    assert action_to_direction(ActionType.MOVE_RIGHT) == Direction.EAST
    assert action_to_direction(ActionType.INTERACT) is None
