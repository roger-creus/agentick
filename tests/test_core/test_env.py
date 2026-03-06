"""Tests for base AgentickEnv."""

import numpy as np
import pytest

from agentick.core.env import AgentickEnv
from agentick.core.grid import Grid
from agentick.core.types import ActionType, CellType, ObjectType


@pytest.fixture
def simple_env():
    """Create a simple test environment."""
    grid = Grid(5, 5)
    grid.terrain[0, :] = CellType.WALL
    grid.terrain[:, 0] = CellType.WALL
    grid.terrain[4, :] = CellType.WALL
    grid.terrain[:, 4] = CellType.WALL
    grid.objects[3, 3] = ObjectType.GOAL

    env = AgentickEnv(grid=grid, render_mode="ascii", max_steps=50)
    env.agent.position = (2, 2)
    return env


def test_env_initialization(simple_env):
    """Test environment initialization."""
    assert simple_env.grid is not None
    assert simple_env.agent is not None
    assert simple_env.step_count == 0
    assert simple_env.max_steps == 50
    assert simple_env.render_mode == "ascii"


def test_env_reset(simple_env):
    """Test environment reset."""
    # Take some steps
    simple_env.step(1)
    simple_env.step(1)
    assert simple_env.step_count == 2

    # Reset
    obs, info = simple_env.reset(seed=42)
    assert simple_env.step_count == 0
    assert obs is not None
    assert isinstance(info, dict)


def test_env_step(simple_env):
    """Test environment step."""
    simple_env.reset()
    obs, reward, terminated, truncated, info = simple_env.step(1)  # MOVE_UP

    assert obs is not None
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert simple_env.step_count == 1


def test_env_move_actions(simple_env):
    """Test agent movement."""
    simple_env.reset()
    initial_pos = simple_env.agent.position

    # Move right
    simple_env.step(4)  # MOVE_RIGHT
    new_pos = simple_env.agent.position
    assert new_pos[0] == initial_pos[0] + 1
    assert new_pos[1] == initial_pos[1]


def test_env_wall_collision(simple_env):
    """Test that agent can't move through walls."""
    simple_env.reset()
    simple_env.agent.position = (1, 1)

    # Try to move into wall (up or left)
    simple_env.step(1)  # MOVE_UP (should hit wall)
    assert simple_env.agent.position == (1, 1)

    simple_env.step(3)  # MOVE_LEFT (should hit wall)
    assert simple_env.agent.position == (1, 1)


def test_env_noop(simple_env):
    """Test NOOP action."""
    simple_env.reset()
    initial_pos = simple_env.agent.position

    simple_env.step(0)  # NOOP
    assert simple_env.agent.position == initial_pos


def test_env_truncation(simple_env):
    """Test episode truncation at max_steps."""
    simple_env.reset()
    simple_env.max_steps = 5

    for i in range(5):
        obs, reward, terminated, truncated, info = simple_env.step(0)
        if i < 4:
            assert not truncated
        else:
            assert truncated


def test_env_done_raises_error(simple_env):
    """Test that stepping after done raises error."""
    simple_env.reset()
    simple_env.max_steps = 1

    simple_env.step(0)  # First step
    assert simple_env.done

    # Should raise error on next step
    with pytest.raises(RuntimeError):
        simple_env.step(0)


def test_env_get_valid_actions(simple_env):
    """Test getting valid action mask."""
    simple_env.reset()
    simple_env.agent.position = (2, 2)  # Center of 5x5 grid

    mask = simple_env.get_valid_actions()
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == bool
    assert len(mask) == simple_env.action_space_obj.n_actions

    # NOOP should always be valid
    noop_idx = simple_env.action_space_obj.get_action_idx(ActionType.NOOP)
    assert mask[noop_idx]


def test_env_get_text_observation(simple_env):
    """Test getting text observation."""
    simple_env.reset()
    text_obs = simple_env.get_text_observation()
    assert isinstance(text_obs, str)
    assert len(text_obs) > 0


def test_env_get_pixel_observation(simple_env):
    """Test getting pixel observation."""
    simple_env.reset()
    pixel_obs = simple_env.get_pixel_observation()
    assert isinstance(pixel_obs, np.ndarray)
    assert pixel_obs.dtype == np.uint8
    assert len(pixel_obs.shape) == 3
    assert pixel_obs.shape[2] == 3


def test_env_get_state_dict(simple_env):
    """Test getting state dictionary."""
    simple_env.reset()
    state_dict = simple_env.get_state_dict()
    assert isinstance(state_dict, dict)
    assert "grid" in state_dict
    assert "agent" in state_dict


def test_env_info_dict(simple_env):
    """Test info dict contents."""
    simple_env.reset()
    obs, reward, terminated, truncated, info = simple_env.step(0)

    assert "step_count" in info
    assert "max_steps" in info
    assert "valid_actions" in info
    assert "agent_position" in info
    assert info["step_count"] == 1
    assert info["max_steps"] == simple_env.max_steps


def test_env_render_ascii(simple_env):
    """Test ASCII rendering."""
    simple_env.reset()
    output = simple_env.render()
    assert isinstance(output, str)
    assert "#" in output  # Walls
    assert "A" in output  # Agent


def test_env_different_render_modes():
    """Test different render modes."""
    grid = Grid(5, 5)

    # ASCII mode
    env = AgentickEnv(grid=grid, render_mode="ascii")
    env.reset()
    obs = env.render()
    assert isinstance(obs, str)

    # Language mode
    env = AgentickEnv(grid=grid, render_mode="language")
    env.reset()
    obs = env.render()
    assert isinstance(obs, str)

    # Pixel mode
    env = AgentickEnv(grid=grid, render_mode="rgb_array")
    env.reset()
    obs = env.render()
    assert isinstance(obs, np.ndarray)

    # State dict mode
    env = AgentickEnv(grid=grid, render_mode="state_dict")
    env.reset()
    obs = env.render()
    assert isinstance(obs, dict)


def test_env_seed_reproducibility():
    """Test that same seed produces same results."""
    grid = Grid(5, 5)

    env1 = AgentickEnv(grid=grid, render_mode="ascii")
    env2 = AgentickEnv(grid=grid, render_mode="ascii")

    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)

    # Should produce same observation
    assert obs1 == obs2

    # Take same actions
    for action in [1, 4, 2, 3]:
        r1, *_ = env1.step(action)[1:2]
        r2, *_ = env2.step(action)[1:2]


def test_env_rotation_actions():
    """Test rotation actions."""
    from agentick.core.actions import ActionSpace

    # Create environment with extended action space
    env = AgentickEnv(render_mode="ascii")
    env.action_space_obj = ActionSpace(extended=True)
    env.action_space = env.action_space_obj.gym_space
    env.reset()

    # Initial orientation
    initial_dir = env.agent.orientation

    # Get action indices for rotation
    rotate_left_idx = env.action_space_obj.get_action_idx(ActionType.ROTATE_LEFT)
    rotate_right_idx = env.action_space_obj.get_action_idx(ActionType.ROTATE_RIGHT)

    # Rotate left
    env.step(rotate_left_idx)
    assert env.agent.orientation == initial_dir.rotate_left()

    # Rotate right
    env.step(rotate_right_idx)
    assert env.agent.orientation == initial_dir
