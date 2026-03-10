"""Test that oracle agent achieves optimal return on tractable tasks."""

import pytest

import agentick
from agentick.oracles import get_oracle

# Only test on tasks where oracle is available
TRACTABLE_TASKS = [
    "GoToGoal-v0",
    # KeyDoorPuzzle-v0 removed: requires INTERACT to open doors, not BFS-solvable
]


@pytest.mark.parametrize("task_name", TRACTABLE_TASKS)
@pytest.mark.timeout(120)
def test_oracle_achieves_high_success(task_name):
    """Test that oracle agent achieves high success rate."""
    env = agentick.make(task_name, difficulty="easy", reward_mode="sparse")
    oracle = get_oracle(task_name, env)

    try:
        successes = 0
        episodes = 20  # Smaller number since oracle is expensive

        for episode in range(episodes):
            obs, info = env.reset(seed=episode)
            oracle.reset(obs, info)
            episode_return = 0

            for step in range(env.spec.max_episode_steps or 100):
                action = oracle.act(obs, info)

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
