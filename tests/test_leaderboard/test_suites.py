"""Tests for benchmark suite definitions."""

import pytest

from agentick.leaderboard.seeds import generate_task_seeds, verify_seeds
from agentick.leaderboard.suites import (
    BenchmarkSuite,
    get_suite,
    list_suites,
    verify_suite_integrity,
)


def test_suite_immutability():
    """Test that suite dataclass is frozen."""
    suite = get_suite("agentick-full-v2")

    with pytest.raises(Exception):  # FrozenInstanceError
        suite.name = "modified"


def test_all_official_suites_exist():
    """Test that all 7 official suites are defined."""
    suites = list_suites()
    assert len(suites) == 7

    # Check key suites
    assert "agentick-full-v2" in suites
    assert "agentick-navigation-v2" in suites
    assert "agentick-planning-v2" in suites
    assert "agentick-reasoning-v2" in suites
    assert "agentick-memory-v2" in suites
    assert "agentick-generalization-v2" in suites
    assert "agentick-multiagent-v2" in suites


def test_suite_structure():
    """Test that suites have correct structure."""
    suite = get_suite("agentick-full-v2")

    assert isinstance(suite, BenchmarkSuite)
    assert suite.name == "agentick-full-v2"
    assert len(suite.tasks) == 37
    assert suite.n_eval_seeds == 25
    assert suite.n_train_seeds == 2000
    assert suite.episodes_per_seed == 1
    assert suite.version == "2.0"


def test_seed_determinism():
    """Test that seeds are deterministically generated."""
    seeds1 = generate_task_seeds("GoToGoal-v0", "medium", "eval", 10)
    seeds2 = generate_task_seeds("GoToGoal-v0", "medium", "eval", 10)

    assert seeds1 == seeds2
    assert len(seeds1) == 10


def test_seed_uniqueness():
    """Test that different task/difficulty combos generate different seeds."""
    seeds1 = generate_task_seeds("GoToGoal-v0", "easy", "eval", 10)
    seeds2 = generate_task_seeds("GoToGoal-v0", "medium", "eval", 10)

    assert seeds1 != seeds2


def test_seed_split_uniqueness():
    """Test that train and eval splits generate different seeds."""
    train = generate_task_seeds("GoToGoal-v0", "medium", "train", 10)
    eval_ = generate_task_seeds("GoToGoal-v0", "medium", "eval", 10)

    assert train != eval_


def test_seed_verification():
    """Test seed verification."""
    seeds = generate_task_seeds("GoToGoal-v0", "medium", "eval", 25)

    assert verify_seeds("GoToGoal-v0", "medium", "eval", seeds)

    wrong_seeds = tuple(range(25))
    assert not verify_seeds("GoToGoal-v0", "medium", "eval", wrong_seeds)


def test_suite_hash_computation():
    """Test that suite hash can be computed."""
    suite = get_suite("agentick-full-v2")

    hash1 = suite.compute_hash()
    hash2 = suite.compute_hash()

    assert hash1 == hash2
    assert len(hash1) == 64  # SHA256 hex digest


def test_suite_integrity():
    """Test suite integrity verification."""
    suite = get_suite("agentick-full-v2")
    assert verify_suite_integrity(suite)


def test_per_task_seeds():
    """Test that suite.get_eval_seeds() returns per-task seeds."""
    suite = get_suite("agentick-full-v2")

    seeds_a = suite.get_eval_seeds("GoToGoal-v0")
    seeds_b = suite.get_eval_seeds("MazeNavigation-v0")

    assert len(seeds_a) == 25
    assert len(seeds_b) == 25
    assert seeds_a != seeds_b


def test_capability_suites():
    """Test that capability-specific suites exist and are correctly configured."""
    capability_suites = [
        "agentick-navigation-v2",
        "agentick-planning-v2",
        "agentick-reasoning-v2",
        "agentick-memory-v2",
        "agentick-generalization-v2",
        "agentick-multiagent-v2",
    ]

    for suite_name in capability_suites:
        suite = get_suite(suite_name)
        assert len(suite.tasks) >= 3
        assert suite.n_eval_seeds == 25


def test_get_nonexistent_suite():
    """Test that getting nonexistent suite raises error."""
    with pytest.raises(ValueError):
        get_suite("nonexistent-suite")
