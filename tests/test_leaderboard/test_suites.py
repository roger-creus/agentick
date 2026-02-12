"""Tests for benchmark suite definitions."""

import pytest

from agentick.leaderboard.seeds import verify_seeds
from agentick.leaderboard.suites import (
    BenchmarkSuite,
    generate_deterministic_seeds,
    get_suite,
    list_suites,
    verify_suite_integrity,
)


def test_suite_immutability():
    """Test that suite dataclass is frozen."""
    suite = get_suite("agentick-quick-v1")

    with pytest.raises(Exception):  # FrozenInstanceError
        suite.name = "modified"


def test_all_official_suites_exist():
    """Test that all 15+ official suites are defined."""
    suites = list_suites()
    assert len(suites) >= 15

    # Check key suites
    assert "agentick-full-v1" in suites
    assert "agentick-core-v1" in suites
    assert "agentick-quick-v1" in suites
    assert "agentick-navigation-v1" in suites


def test_suite_structure():
    """Test that suites have correct structure."""
    suite = get_suite("agentick-full-v1")

    assert isinstance(suite, BenchmarkSuite)
    assert suite.name == "agentick-full-v1"
    assert len(suite.tasks) == 38
    assert len(suite.eval_seeds) == 50
    assert suite.episodes_per_seed == 1
    assert suite.version == "1.0"


def test_seed_determinism():
    """Test that seeds are deterministically generated."""
    seeds1 = generate_deterministic_seeds("test-suite", 10)
    seeds2 = generate_deterministic_seeds("test-suite", 10)

    assert seeds1 == seeds2
    assert len(seeds1) == 10


def test_seed_uniqueness():
    """Test that different suite names generate different seeds."""
    seeds1 = generate_deterministic_seeds("suite-a", 10)
    seeds2 = generate_deterministic_seeds("suite-b", 10)

    assert seeds1 != seeds2


def test_seed_verification():
    """Test seed verification."""
    suite = get_suite("agentick-quick-v1")

    # Verify seeds match deterministic generation
    assert verify_seeds(suite.name, suite.eval_seeds)

    # Verify wrong seeds fail
    wrong_seeds = tuple(range(len(suite.eval_seeds)))
    assert not verify_seeds(suite.name, wrong_seeds)


def test_suite_hash_computation():
    """Test that suite hash can be computed."""
    suite = get_suite("agentick-full-v1")

    hash1 = suite.compute_hash()
    hash2 = suite.compute_hash()

    # Hash should be deterministic
    assert hash1 == hash2
    assert len(hash1) == 64  # SHA256 hex digest


def test_suite_integrity():
    """Test suite integrity verification."""
    suite = get_suite("agentick-quick-v1")

    assert verify_suite_integrity(suite)


def test_quick_suite_is_fast():
    """Test that quick suite is configured for speed."""
    suite = get_suite("agentick-quick-v1")

    assert suite.difficulty == "easy"
    assert len(suite.eval_seeds) == 10  # Small number of seeds
    assert suite.max_steps_override == 100  # Short episodes


def test_capability_suites():
    """Test that capability-specific suites exist and are correctly configured."""
    capability_suites = [
        "agentick-navigation-v1",
        "agentick-memory-v1",
        "agentick-reasoning-v1",
        "agentick-skill-v1",
        "agentick-control-v1",
        "agentick-combinatorial-v1",
    ]

    for suite_name in capability_suites:
        suite = get_suite(suite_name)
        assert len(suite.tasks) >= 3  # At least a few tasks
        assert len(suite.eval_seeds) == 30


def test_full_vs_core():
    """Test that full suite has more tasks than core."""
    full = get_suite("agentick-full-v1")
    core = get_suite("agentick-core-v1")

    assert len(full.tasks) > len(core.tasks)
    assert len(full.tasks) == 38
    assert len(core.tasks) == 27


def test_get_nonexistent_suite():
    """Test that getting nonexistent suite raises error."""
    with pytest.raises(ValueError):
        get_suite("nonexistent-suite")
