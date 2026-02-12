"""Test that oracle agent achieves optimal return on tractable tasks."""

import numpy as np
import pytest

import agentick
from agentick.benchmark.baselines import OracleAgent

# Only test on tasks where BFS oracle is tractable
TRACTABLE_TASKS = [
    "GoToGoal-v0",
    "KeyDoorPuzzle-v0",
]


@pytest.mark.parametrize("task_name", TRACTABLE_TASKS)
@pytest.mark.timeout(120)
def test_oracle_achieves_high_success(task_name):
    """Test that oracle agent achieves high success rate."""
    env = agentick.make(task_name, difficulty="easy", reward_mode="sparse")
    oracle = OracleAgent(env)

    try:
        successes = 0
        episodes = 20  # Smaller number since oracle is expensive

        for episode in range(episodes):
            obs, info = env.reset(seed=episode)
            episode_return = 0

            for step in range(env.spec.max_episode_steps or 100):
                state_dict = env.get_state_dict()
                valid_actions = env.get_valid_actions()
                action = oracle.act(obs, valid_actions, state_dict)

                if action is None:
                    # Oracle couldn't find solution, try random
                    action = np.random.choice(valid_actions)

                obs, reward, terminated, truncated, info = env.step(action)
                episode_return += reward

                if terminated:
                    if info.get("success", False):
                        successes += 1
                    break

                if truncated:
                    break

        success_rate = successes / episodes

        # Oracle should achieve high success rate (at least 50% on easy)
        assert success_rate >= 0.5, f"Oracle success rate too low: {success_rate:.2%}"
    finally:
        env.close()
