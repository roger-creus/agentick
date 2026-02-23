"""Test that all tasks support different reward modes."""

import pytest

import agentick

# Survival tasks may give non-zero sparse rewards due to station-death penalty
_SURVIVAL_TASKS = {"ResourceManagement-v0"}


@pytest.mark.parametrize("task_name", agentick.list_tasks())
@pytest.mark.parametrize("reward_mode", ["sparse", "dense"])
@pytest.mark.timeout(30)
def test_reward_mode_works(task_name, reward_mode):
    """Test that each task supports sparse and dense rewards."""
    env = agentick.make(task_name, difficulty="easy", reward_mode=reward_mode)
    try:
        obs, info = env.reset(seed=42)

        # Take 10 steps
        rewards = []
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)

            if terminated or truncated:
                break

        # Verify rewards are numeric
        assert all(isinstance(r, (int, float)) for r in rewards)

        # For sparse mode, most rewards should be 0 (unless we got lucky and succeeded)
        # Skip this check for survival tasks (their sparse reward may include -1.0
        # when a station dies, which can happen with random actions).
        if (
            reward_mode == "sparse"
            and task_name not in _SURVIVAL_TASKS
            and not any(info.get("success", False) for _ in range(len(rewards)))
        ):
            # Most rewards should be 0 or small negative
            non_zero_count = sum(1 for r in rewards if abs(r) > 0.001)
            # Allow some non-zero rewards but most should be zero
            assert non_zero_count <= len(rewards) * 0.5

        # For dense mode, we expect more varied rewards (shaping)
        # Just verify they exist and are reasonable
        assert all(r >= -100 for r in rewards), "Rewards too negative"
        assert all(r <= 100 for r in rewards), "Rewards too positive"
    finally:
        env.close()
