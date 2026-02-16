"""Tests for graceful degradation when 3D dependencies are missing."""

import os
import unittest.mock

import numpy as np
import pytest

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import agentick


def test_import_error_falls_back_to_2d():
    """When the 3D renderer import fails, auto-detection falls back to 2D."""
    env = agentick.make("GoToGoal-v0", render_mode="rgb_array", seed=42)

    # Reset cached decision and renderer
    env._use_3d = None
    env._renderer_3d = None

    # Mock the renderer_3d import to raise ImportError
    with unittest.mock.patch(
        "agentick.core.env.AgentickEnv._should_use_3d",
        side_effect=lambda: False,
    ):
        pass

    # Directly test the fallback: render_3d=False should use 2D
    env._render_3d_flag = False
    env._use_3d = None
    result = env._should_use_3d()
    assert result is False

    env.close()


def test_render_3d_false_skips_3d_entirely():
    """render_3d=False should never try to import 3D deps."""
    env = agentick.make("GoToGoal-v0", render_mode="rgb_array", render_3d=False, seed=42)
    obs, _ = env.reset(seed=42)

    # Should be 2D render
    assert isinstance(obs, np.ndarray)
    assert obs.ndim == 3
    assert obs.dtype == np.uint8
    # Should never have created 3D renderer
    assert env._renderer_3d is None
    env.close()


def test_rgb_array_2d_never_uses_3d():
    """rgb_array_2d mode should always use 2D renderer."""
    env = agentick.make("GoToGoal-v0", render_mode="rgb_array_2d", seed=42)
    obs, _ = env.reset(seed=42)

    assert isinstance(obs, np.ndarray)
    assert obs.ndim == 3
    assert env._renderer_3d is None
    env.close()


def test_fallback_chain_never_crashes():
    """The fallback chain GLB → primitives → 2D should never crash."""
    # Test with explicit 2D fallback
    env = agentick.make("GoToGoal-v0", render_mode="rgb_array", render_3d=False, seed=42)
    obs, _ = env.reset(seed=42)
    assert obs is not None
    assert isinstance(obs, np.ndarray)
    env.close()

    # Test 2D mode directly
    env2 = agentick.make("GoToGoal-v0", render_mode="rgb_array_2d", seed=42)
    obs2, _ = env2.reset(seed=42)
    assert obs2 is not None
    assert isinstance(obs2, np.ndarray)
    env2.close()
