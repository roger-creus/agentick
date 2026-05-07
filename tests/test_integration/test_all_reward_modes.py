"""Test that all tasks support different reward modes."""

import pytest

import agentick

# Survival tasks may give non-zero sparse rewards due to station-death penalty
_SURVIVAL_TASKS = {"ResourceManagement-v0"}


@pytest.mark.parametrize("task_name", agentick.list_tasks())
@pytest.mark.timeout(30)
def test_reward_modes_work(task_name):
    """Each task must support sparse and dense rewards with bounded numeric values."""
    for reward_mode in ("sparse", "dense"):
        env = agentick.make(task_name, difficulty="easy", reward_mode=reward_mode)
        try:
            _obs, _info = env.reset(seed=42)

            rewards = []
            last_info = {}
            for _ in range(10):
                action = env.action_space.sample()
                _obs, reward, terminated, truncated, last_info = env.step(action)
                rewards.append(reward)

                if terminated or truncated:
                    break

            assert all(isinstance(r, (int, float)) for r in rewards)

            if (
                reward_mode == "sparse"
                and task_name not in _SURVIVAL_TASKS
                and not last_info.get("success", False)
            ):
                non_zero_count = sum(1 for r in rewards if abs(r) > 0.001)
                assert non_zero_count <= len(rewards) * 0.5

            assert all(r >= -100 for r in rewards), "Rewards too negative"
            assert all(r <= 100 for r in rewards), "Rewards too positive"
        finally:
            env.close()
