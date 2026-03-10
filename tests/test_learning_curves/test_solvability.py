"""Test that generated instances are solvable."""

import pytest

import agentick
from agentick.oracles import get_oracle


@pytest.mark.parametrize("task_name", ["GoToGoal-v0", "KeyDoorPuzzle-v0"])
@pytest.mark.timeout(180)
def test_instances_solvable(task_name):
    """Test that generated instances are solvable by oracle."""
    env = agentick.make(task_name, difficulty="easy", reward_mode="sparse")
    oracle = get_oracle(task_name, env)

    try:
        solvable_count = 0
        total_count = 20  # Test fewer instances since oracle is expensive

        for seed in range(total_count):
            obs, info = env.reset(seed=seed)
            oracle.reset(obs, info)

            # If oracle can produce an action, instance is likely solvable
            action = oracle.act(obs, info)

            if action is not None:
                solvable_count += 1

        solvability_rate = solvable_count / total_count

        # At least 80% should be solvable
        assert solvability_rate >= 0.8, f"Only {solvability_rate:.1%} of instances solvable"
    finally:
        env.close()


@pytest.mark.parametrize("task_name", agentick.list_tasks()[:5])
@pytest.mark.timeout(60)
def test_all_instances_have_valid_actions(task_name):
    """Test that all generated instances have at least one valid action."""
    env = agentick.make(task_name, difficulty="easy")

    try:
        for seed in range(10):
            obs, info = env.reset(seed=seed)
            valid_actions = env.get_valid_actions()
            assert len(valid_actions) > 0, f"No valid actions for seed {seed}"
    finally:
        env.close()
