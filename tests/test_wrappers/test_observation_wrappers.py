"""Tests for observation wrappers."""

import numpy as np
import pytest

import agentick
from agentick.wrappers import (
    DictObservationWrapper,
    FlattenObservationWrapper,
    LanguageActionWrapper,
    PixelObservationWrapper,
    TextObservationWrapper,
)


@pytest.fixture
def base_env():
    """Create a base environment for testing."""
    return agentick.make("GoToGoal-v0", render_mode="ascii")


def test_text_observation_wrapper(base_env):
    """Test TextObservationWrapper."""
    wrapped = TextObservationWrapper(base_env)
    obs, info = wrapped.reset(seed=42)

    assert isinstance(obs, str)
    assert len(obs) > 0

    # Take a step
    obs, reward, terminated, truncated, info = wrapped.step(0)
    assert isinstance(obs, str)


def test_pixel_observation_wrapper(base_env):
    """Test PixelObservationWrapper."""
    wrapped = PixelObservationWrapper(base_env)
    obs, info = wrapped.reset(seed=42)

    assert isinstance(obs, np.ndarray)
    assert obs.dtype == np.uint8
    assert len(obs.shape) == 3  # (H, W, C)
    assert obs.shape[2] == 3  # RGB

    # Take a step
    obs, reward, terminated, truncated, info = wrapped.step(0)
    assert isinstance(obs, np.ndarray)


def test_dict_observation_wrapper(base_env):
    """Test DictObservationWrapper."""
    wrapped = DictObservationWrapper(base_env)
    obs, info = wrapped.reset(seed=42)

    assert isinstance(obs, dict)

    # Take a step
    obs, reward, terminated, truncated, info = wrapped.step(0)
    assert isinstance(obs, dict)


def test_flatten_observation_wrapper(base_env):
    """Test FlattenObservationWrapper."""
    wrapped = FlattenObservationWrapper(base_env)
    obs, info = wrapped.reset(seed=42)

    assert isinstance(obs, np.ndarray)
    assert obs.dtype == np.float32
    assert len(obs.shape) == 1  # Flattened

    # Check size matches grid
    grid = base_env.grid
    expected_size = grid.height * grid.width * 4
    assert obs.shape[0] == expected_size

    # Take a step
    obs, reward, terminated, truncated, info = wrapped.step(0)
    assert isinstance(obs, np.ndarray)
    assert obs.shape[0] == expected_size


def test_language_action_wrapper(base_env):
    """Test LanguageActionWrapper."""
    wrapped = LanguageActionWrapper(base_env)
    obs, info = wrapped.reset(seed=42)

    # Test with string action
    obs, reward, terminated, truncated, info = wrapped.step("move_up")
    assert not terminated or truncated

    # Test with integer action
    obs, reward, terminated, truncated, info = wrapped.step(0)
    assert not terminated or truncated


def test_wrapper_combinations(base_env):
    """Test combining multiple wrappers."""
    # Text wrapper
    wrapped = TextObservationWrapper(base_env)

    obs, info = wrapped.reset(seed=42)
    assert isinstance(obs, str)

    obs, reward, terminated, truncated, info = wrapped.step(1)
    assert isinstance(obs, str)

    # Pixel wrapper
    wrapped2 = PixelObservationWrapper(base_env)
    obs2, info2 = wrapped2.reset(seed=42)
    assert isinstance(obs2, np.ndarray)
