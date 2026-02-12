"""Tests for GoToGoal task."""

import pytest

import agentick


def test_go_to_goal_registered():
    """Test that GoToGoal is registered."""
    tasks = agentick.list_tasks()
    assert "GoToGoal-v0" in tasks


def test_go_to_goal_create():
    """Test creating GoToGoal environment."""
    env = agentick.make("GoToGoal-v0", difficulty="easy", seed=42)
    assert env is not None
    assert env.max_steps == 20


def test_go_to_goal_reset():
    """Test resetting GoToGoal environment."""
    env = agentick.make("GoToGoal-v0", difficulty="easy", seed=42)
    obs, info = env.reset()
    assert obs is not None
    assert "valid_actions" in info


def test_go_to_goal_step():
    """Test stepping in GoToGoal environment."""
    env = agentick.make("GoToGoal-v0", difficulty="easy", seed=42)
    env.reset()

    obs, reward, terminated, truncated, info = env.step(1)  # MOVE_UP
    assert obs is not None
    assert reward is not None


@pytest.mark.parametrize("seed", [0, 42, 123, 999, 2024])
def test_go_to_goal_generation(seed):
    """Test task generation with different seeds."""
    env = agentick.make("GoToGoal-v0", difficulty="medium", seed=seed)
    env.reset()

    # Check that grid and goal exist
    assert env.grid is not None
    assert env.grid.height == 10
    assert env.grid.width == 10

    # Check that there's a goal
    assert len(env.task_config["goal_positions"]) == 1


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard", "expert"])
def test_go_to_goal_difficulties(difficulty):
    """Test all difficulty levels."""
    env = agentick.make("GoToGoal-v0", difficulty=difficulty, seed=42)
    env.reset()

    size = env.task.difficulty_config.grid_size
    assert env.grid.height == size
    assert env.grid.width == size


def test_go_to_goal_success():
    """Test success condition."""
    env = agentick.make("GoToGoal-v0", difficulty="easy", seed=42)
    env.reset()

    # Move agent to goal
    goal_pos = env.task_config["goal_positions"][0]
    env.agent.position = goal_pos

    # Check success
    state = env._get_state_for_reward()
    state.update(
        {
            "grid": env.grid,
            "agent": env.agent,
            "config": env.task_config,
        }
    )

    assert env.task.check_success(state)


def test_go_to_goal_dense_reward():
    """Test dense reward shaping."""
    env = agentick.make("GoToGoal-v0", difficulty="easy", reward_mode="dense", seed=42)
    env.reset()

    # Take a step
    obs, reward, terminated, truncated, info = env.step(0)

    # Dense reward should include step penalty
    assert reward < 0.0  # Step penalty


def test_go_to_goal_sparse_reward():
    """Test sparse reward."""
    env = agentick.make("GoToGoal-v0", difficulty="easy", reward_mode="sparse", seed=42)
    env.reset()

    # Take a step (not reaching goal)
    obs, reward, terminated, truncated, info = env.step(0)

    # Sparse reward should be 0 until goal reached
    assert reward == 0.0


def test_go_to_goal_renders():
    """Test that all render modes work."""
    for mode in ["ascii", "language", "rgb_array", "state_dict"]:
        env = agentick.make("GoToGoal-v0", difficulty="easy", render_mode=mode, seed=42)
        env.reset()

        output = env.render()
        assert output is not None
