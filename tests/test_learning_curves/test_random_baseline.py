"""Test that random agent achieves approximately expected baseline."""

import numpy as np
import pytest

import agentick
from agentick.benchmark.baselines import RandomAgent


@pytest.mark.parametrize("task_name", agentick.list_tasks()[:10])  # Subset for speed
@pytest.mark.timeout(60)
def test_random_baseline_return(task_name):
    """Test that random agent achieves approximately expected return."""
    env = agentick.make(task_name, difficulty="easy", reward_mode="sparse")
    agent = RandomAgent()

    try:
        # Run 100 episodes
        returns = []
        for episode in range(100):
            obs, info = env.reset(seed=episode)
            episode_return = 0

            for step in range(env.spec.max_episode_steps or 100):
                valid_actions = env.get_valid_actions()
                action = agent.act(obs, valid_actions)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_return += reward

                if terminated or truncated:
                    break

            returns.append(episode_return)

        mean_return = np.mean(returns)
        _std_return = np.std(returns)

        # Just verify returns are reasonable (not NaN, not infinite)
        assert not np.isnan(mean_return)
        assert not np.isinf(mean_return)
        assert mean_return >= -100  # Shouldn't be too negative

        # Random agent should have low success rate
        # (success_rate should be in info, but we'll check returns are mostly low)
        _success_return = 1.0  # Sparse reward gives 1.0 on success
        success_rate = sum(1 for r in returns if r >= 0.9) / len(returns)
        assert success_rate < 0.5, f"Random agent too successful: {success_rate}"
    finally:
        env.close()
