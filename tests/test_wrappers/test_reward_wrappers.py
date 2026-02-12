"""Tests for reward wrappers."""

import pytest

import agentick
from agentick.wrappers import (
    CurriculumWrapper,
    DenseRewardWrapper,
    RewardScaleWrapper,
    SparseRewardWrapper,
)


@pytest.fixture
def base_env():
    """Create a base environment for testing."""
    return agentick.make("GoToGoal-v0", render_mode="ascii", reward_mode="dense")


def test_dense_reward_wrapper(base_env):
    """Test DenseRewardWrapper."""
    wrapped = DenseRewardWrapper(base_env)
    obs, info = wrapped.reset(seed=42)

    # Take steps and verify rewards are generated
    total_reward = 0
    for _ in range(10):
        obs, reward, terminated, truncated, info = wrapped.step(1)
        total_reward += reward
        if terminated or truncated:
            break

    # Dense rewards should provide feedback
    assert isinstance(total_reward, float)


def test_sparse_reward_wrapper(base_env):
    """Test SparseRewardWrapper."""
    wrapped = SparseRewardWrapper(base_env)
    obs, info = wrapped.reset(seed=42)

    # Take steps - sparse reward only on success
    for _ in range(5):
        obs, reward, terminated, truncated, info = wrapped.step(1)
        if not (terminated or truncated):
            assert reward <= 0  # No reward until success
        if terminated or truncated:
            break


def test_reward_scale_wrapper(base_env):
    """Test RewardScaleWrapper."""
    wrapped = RewardScaleWrapper(base_env, scale=0.1)
    obs, info = wrapped.reset(seed=42)

    obs, reward, terminated, truncated, info = wrapped.step(1)
    assert isinstance(reward, float)
    # Reward should be scaled
    assert abs(reward) < 10  # Reasonable bounds


def test_curriculum_wrapper():
    """Test CurriculumWrapper."""
    env = agentick.make("GoToGoal-v0", render_mode="ascii", difficulty="easy")
    wrapped = CurriculumWrapper(env, success_threshold=0.8, window_size=10)

    obs, info = wrapped.reset(seed=42)

    # Run some episodes
    for _ in range(5):
        done = False
        while not done:
            obs, reward, terminated, truncated, info = wrapped.step(1)
            done = terminated or truncated
        wrapped.reset()

    # Verify wrapper exists and works
    assert wrapped is not None


def test_wrapper_stacking():
    """Test stacking multiple reward wrappers."""
    env = agentick.make("GoToGoal-v0", render_mode="ascii")

    # Stack wrappers
    env = DenseRewardWrapper(env)
    env = RewardScaleWrapper(env, scale=0.5)

    obs, info = env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(1)

    assert isinstance(reward, float)
