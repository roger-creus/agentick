"""Tests for leaderboard database."""

from agentick.leaderboard.database import LeaderboardDatabase


def test_database_initialization(tmp_path):
    """Test database can be initialized."""
    db = LeaderboardDatabase(tmp_path / "test_data")

    assert db.entries_dir.exists()
    assert db.rankings_dir.exists()
    assert db.history_dir.exists()
