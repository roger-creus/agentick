"""Tests for recording wrappers."""

import json

import pytest

import agentick
from agentick.wrappers import EpisodeRecorder, TrajectoryWrapper


@pytest.fixture
def base_env():
    """Create a base environment for testing."""
    return agentick.make("GoToGoal-v0", render_mode="ascii")


def test_episode_recorder(base_env, tmp_path):
    """Test EpisodeRecorder."""
    output_file = tmp_path / "episodes.json"
    wrapped = EpisodeRecorder(base_env, save_path=str(output_file))

    # Run an episode
    obs, info = wrapped.reset(seed=42)
    for _ in range(10):
        obs, reward, terminated, truncated, info = wrapped.step(1)
        if terminated or truncated:
            break

    # Reset to trigger save
    wrapped.reset()

    # Verify file was created
    assert output_file.exists()

    # Verify content
    with open(output_file) as f:
        data = json.load(f)
        assert len(data) > 0
        # Check first episode
        episode = data[0]
        assert "steps" in episode


def test_trajectory_wrapper(base_env):
    """Test TrajectoryWrapper."""
    wrapped = TrajectoryWrapper(base_env)

    # Run an episode
    obs, info = wrapped.reset(seed=42)
    for _ in range(10):
        obs, reward, terminated, truncated, info = wrapped.step(1)
        if terminated or truncated:
            break

    # Reset to finalize trajectory
    wrapped.reset()

    # Get trajectories
    trajectories = wrapped.get_trajectories()
    assert isinstance(trajectories, list)
    assert len(trajectories) > 0

    # Each trajectory should have observations, actions, rewards
    traj = trajectories[0]
    assert "observations" in traj
    assert "actions" in traj
    assert "rewards" in traj


def test_episode_recorder_without_save(base_env):
    """Test EpisodeRecorder without save path."""
    wrapped = EpisodeRecorder(base_env)

    # Run an episode
    obs, info = wrapped.reset(seed=42)
    for _ in range(5):
        obs, reward, terminated, truncated, info = wrapped.step(1)
        if terminated or truncated:
            break

    # Should work without errors
    assert len(wrapped.current_episode) > 0


def test_trajectory_wrapper_multiple_episodes(base_env):
    """Test TrajectoryWrapper with multiple episodes."""
    wrapped = TrajectoryWrapper(base_env)

    # Run 3 episodes
    for ep in range(3):
        obs, info = wrapped.reset(seed=42 + ep)
        for _ in range(5):
            obs, reward, terminated, truncated, info = wrapped.step(1)
            if terminated or truncated:
                break

    # Should have 2 complete trajectories (current one not finalized)
    trajectories = wrapped.get_trajectories()
    assert len(trajectories) >= 2
