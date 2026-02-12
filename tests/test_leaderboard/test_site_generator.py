"""Tests for site generator."""

from agentick.leaderboard.site.generate import SiteGenerator


def test_site_generator_initialization(tmp_path):
    """Test site generator can be initialized."""
    generator = SiteGenerator(
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "site",
    )

    assert generator is not None
