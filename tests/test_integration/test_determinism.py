"""Test that environments are deterministic with same seed."""

import pytest

import agentick


@pytest.mark.parametrize("task_name", agentick.list_tasks())
@pytest.mark.parametrize("seed", [0, 42, 123])
@pytest.mark.timeout(30)
def test_same_seed_identical_reset(task_name, seed):
    """Test that same seed produces identical initial state."""
    env = agentick.make(task_name, difficulty="easy", render_mode="state_dict")
    try:
        # Reset twice with same seed
        obs1, info1 = env.reset(seed=seed)
        obs2, info2 = env.reset(seed=seed)

        # States should be identical
        assert obs1["agent"]["position"] == obs2["agent"]["position"]
        assert obs1["grid"]["terrain"] == obs2["grid"]["terrain"]
        assert obs1["grid"]["objects"] == obs2["grid"]["objects"]
    finally:
        env.close()


@pytest.mark.parametrize("task_name", agentick.list_tasks()[:10])
@pytest.mark.timeout(30)
def test_same_seed_identical_episode(task_name):
    """Test that same seed and actions produce identical episode."""
    env1 = agentick.make(task_name, difficulty="easy")
    env2 = agentick.make(task_name, difficulty="easy")

    try:
        seed = 42
        obs1, info1 = env1.reset(seed=seed)
        obs2, info2 = env2.reset(seed=seed)

        # Take 10 identical actions
        for _ in range(10):
            action = env1.action_space.sample()

            obs1, reward1, term1, trunc1, info1 = env1.step(action)
            obs2, reward2, term2, trunc2, info2 = env2.step(action)

            # Results should be identical
            assert reward1 == reward2
            assert term1 == term2
            assert trunc1 == trunc2

            if term1 or trunc1:
                break
    finally:
        env1.close()
        env2.close()
