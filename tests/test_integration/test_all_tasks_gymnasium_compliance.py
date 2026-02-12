"""Test that all tasks pass Gymnasium's environment checker."""

import pytest
from gymnasium.utils.env_checker import check_env

import agentick


@pytest.mark.parametrize("task_name", agentick.list_tasks())
@pytest.mark.timeout(30)
def test_task_gymnasium_compliance(task_name):
    """Test that each task passes gymnasium environment checker."""
    env = agentick.make(task_name, difficulty="easy")
    try:
        check_env(env, skip_render_check=True)  # Skip render since we test that separately
    finally:
        env.close()


@pytest.mark.parametrize("task_name", agentick.list_tasks()[:5])  # Test on subset for speed
@pytest.mark.parametrize("difficulty", ["easy", "medium"])
@pytest.mark.timeout(30)
def test_task_difficulty_levels(task_name, difficulty):
    """Test that each task supports different difficulty levels."""
    env = agentick.make(task_name, difficulty=difficulty)
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()
