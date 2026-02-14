"""Test Atari preprocessing wrappers."""

import numpy as np

import agentick
from agentick.wrappers import (
    FrameStack,
    GrayscaleObservation,
    ResizeObservation,
    make_atari_env,
)


def test_resize_observation():
    """Test ResizeObservation wrapper."""
    env = agentick.make("GoToGoal-v0", render_mode="rgb_array")
    env = ResizeObservation(env, size=(84, 84))

    obs, _ = env.reset(seed=42)

    # Check shape
    assert obs.shape == (84, 84, 3), f"Expected (84, 84, 3), got {obs.shape}"
    assert obs.dtype == np.uint8

    # Check step
    obs, _, _, _, _ = env.step(0)
    assert obs.shape == (84, 84, 3)

    env.close()


def test_grayscale_observation():
    """Test GrayscaleObservation wrapper."""
    env = agentick.make("GoToGoal-v0", render_mode="rgb_array")
    env = ResizeObservation(env, size=(84, 84))
    env = GrayscaleObservation(env)

    obs, _ = env.reset(seed=42)

    # Check shape
    assert obs.shape == (84, 84, 1), f"Expected (84, 84, 1), got {obs.shape}"
    assert obs.dtype == np.uint8

    # Check step
    obs, _, _, _, _ = env.step(0)
    assert obs.shape == (84, 84, 1)

    env.close()


def test_frame_stack():
    """Test FrameStack wrapper."""
    env = agentick.make("GoToGoal-v0", render_mode="rgb_array")
    env = ResizeObservation(env, size=(84, 84))
    env = GrayscaleObservation(env)
    env = FrameStack(env, n_frames=4)

    obs, _ = env.reset(seed=42)

    # Check shape
    assert obs.shape == (84, 84, 4), f"Expected (84, 84, 4), got {obs.shape}"
    assert obs.dtype == np.uint8

    # Check step
    obs, _, _, _, _ = env.step(0)
    assert obs.shape == (84, 84, 4)

    env.close()


def test_make_atari_env():
    """Test make_atari_env convenience function."""
    env = make_atari_env("GoToGoal-v0", seed=42)

    obs, _ = env.reset(seed=42)

    # Check final shape is (84, 84, 4)
    assert obs.shape == (84, 84, 4), f"Expected (84, 84, 4), got {obs.shape}"
    assert obs.dtype == np.uint8

    # Run a few steps
    for _ in range(10):
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert obs.shape == (84, 84, 4)
        if terminated or truncated:
            break

    env.close()


def test_make_atari_env_with_difficulty():
    """Test make_atari_env with difficulty parameter."""
    env = make_atari_env("GoToGoal-v0", seed=42, difficulty="hard")

    obs, _ = env.reset(seed=42)
    assert obs.shape == (84, 84, 4)

    env.close()
