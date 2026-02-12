"""Tests for agent comparison."""

from agentick.leaderboard.comparison import pareto_frontier


def test_pareto_frontier_empty():
    """Test Pareto frontier with empty list."""
    frontier = pareto_frontier([])
    assert frontier == []
