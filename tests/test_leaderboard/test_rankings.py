"""Tests for ranking computation."""

from agentick.leaderboard.rankings import compute_rankings, is_significantly_better


def test_compute_rankings_empty():
    """Test rankings with empty list."""
    rankings = compute_rankings([])
    assert rankings == []


def test_is_significantly_better():
    """Test significance testing."""
    # Non-overlapping CIs
    assert is_significantly_better(0.9, (0.85, 0.95), 0.5, (0.45, 0.55))

    # Overlapping CIs
    assert not is_significantly_better(0.8, (0.75, 0.85), 0.75, (0.70, 0.80))
