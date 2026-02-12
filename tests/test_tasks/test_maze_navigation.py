"""Tests for MazeNavigation task."""

import pytest

import agentick


def test_maze_navigation_registered():
    """Test that MazeNavigation is registered."""
    assert "MazeNavigation-v0" in agentick.list_tasks()


def test_maze_navigation_create():
    """Test creating MazeNavigation environment."""
    env = agentick.make("MazeNavigation-v0", difficulty="easy", seed=42)
    assert env is not None


def test_maze_navigation_reset():
    """Test resetting environment."""
    env = agentick.make("MazeNavigation-v0", difficulty="easy", seed=42)
    obs, info = env.reset()
    assert obs is not None


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard", "expert"])
def test_maze_navigation_difficulties(difficulty):
    """Test all difficulty levels."""
    env = agentick.make("MazeNavigation-v0", difficulty=difficulty, seed=42)
    env.reset()
    assert env.grid is not None


def test_maze_navigation_solvable():
    """Test that generated mazes are solvable."""
    env = agentick.make("MazeNavigation-v0", difficulty="easy", seed=42)
    env.reset()

    agent_pos = env.agent.position
    goal_pos = env.task_config["goal_positions"][0]

    # Check path exists
    path = env.grid.bfs(agent_pos, goal_pos)
    assert path is not None
    assert len(path) > 0
