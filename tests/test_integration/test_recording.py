"""Test recording wrappers."""

import json
import tempfile
from pathlib import Path

import pytest

import agentick
from agentick.wrappers.recording_wrappers import EpisodeRecorder


@pytest.mark.parametrize("task_name", agentick.list_tasks()[:3])
@pytest.mark.timeout(30)
def test_episode_recorder(task_name):
    """Test that episode recorder saves episodes correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "episode.json"
        env = agentick.make(task_name, difficulty="easy")
        env = EpisodeRecorder(env, save_path=str(save_path))

        try:
            # Run one episode
            obs, info = env.reset(seed=42)
            for _ in range(10):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break

            # Trigger save by resetting again
            env.reset(seed=43)

            # Check that episode was saved
            assert save_path.exists(), "Episode file not created"

            # Load and verify episode data
            with open(save_path) as f:
                episode_data = json.load(f)

            assert isinstance(episode_data, list)
            assert len(episode_data) > 0
            # First episode should have reset and steps
            first_episode = episode_data[0]
            assert len(first_episode) > 0
        finally:
            env.close()
