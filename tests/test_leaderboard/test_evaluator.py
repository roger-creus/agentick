"""Tests for leaderboard evaluator."""

from agentick.leaderboard.evaluator import LeaderboardEvaluator


def test_evaluator_initialization():
    """Test evaluator can be initialized."""
    evaluator = LeaderboardEvaluator(verbose=False)
    assert evaluator is not None


# More comprehensive tests would go here
# For now, basic initialization test passes
