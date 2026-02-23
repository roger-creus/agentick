"""Tests for human baselines."""

import numpy as np
import pytest

from agentick.human.baselines import (
    HUMAN_BASELINES,
    compare_to_human,
    estimate_human_performance,
    get_all_baselines,
    get_baselines_by_difficulty,
    get_human_baseline,
    get_summary_statistics,
)
from agentick.tasks.registry import list_tasks


def test_all_tasks_have_baselines():
    """Test that all tasks have human baselines."""
    all_tasks = list_tasks()
    # Filter out test tasks (used in test_registry.py)
    all_tasks = [t for t in all_tasks if not t.startswith("Test")]
    baseline_tasks = set(HUMAN_BASELINES.keys())

    assert len(all_tasks) == len(baseline_tasks), (
        f"Task count mismatch: {len(all_tasks)} tasks, {len(baseline_tasks)} baselines"
    )

    for task in all_tasks:
        assert task in baseline_tasks, f"Missing baseline for task: {task}"


def test_baseline_structure():
    """Test that all baselines have required fields."""
    required_fields = {"success_rate", "avg_steps", "optimal_ratio", "learning_curve", "difficulty"}

    for task_name, baseline in HUMAN_BASELINES.items():
        assert all(field in baseline for field in required_fields), (
            f"Task {task_name} missing required fields"
        )

        # Check value ranges
        assert 0.0 <= baseline["success_rate"] <= 1.0, f"Invalid success_rate for {task_name}"
        assert baseline["avg_steps"] > 0, f"Invalid avg_steps for {task_name}"
        assert baseline["optimal_ratio"] >= 1.0, f"Invalid optimal_ratio for {task_name}"
        assert len(baseline["learning_curve"]) == 4, (
            f"Learning curve should have 4 points for {task_name}"
        )

        # Learning curve should be monotonically increasing
        curve = baseline["learning_curve"]
        for i in range(len(curve) - 1):
            assert curve[i] <= curve[i + 1], f"Learning curve not monotonic for {task_name}"

        # Last learning curve point should match success_rate
        assert abs(curve[-1] - baseline["success_rate"]) < 0.01, (
            f"Learning curve endpoint mismatch for {task_name}"
        )


def test_get_human_baseline():
    """Test getting a single baseline."""
    baseline = get_human_baseline("GoToGoal-v0")
    assert baseline["difficulty"] == "easy"
    assert baseline["success_rate"] > 0.9  # Easy task

    with pytest.raises(KeyError):
        get_human_baseline("NonexistentTask-v0")


def test_get_all_baselines():
    """Test getting all baselines."""
    all_baselines = get_all_baselines()
    assert len(all_baselines) == len(HUMAN_BASELINES)
    assert "GoToGoal-v0" in all_baselines

    # Verify it's a copy
    all_baselines["GoToGoal-v0"]["success_rate"] = 0.0
    assert HUMAN_BASELINES["GoToGoal-v0"]["success_rate"] != 0.0


def test_estimate_human_performance():
    """Test estimating performance over multiple attempts."""
    # First attempt
    perf1 = estimate_human_performance("GoToGoal-v0", num_attempts=1)
    assert perf1["attempt"] == 1
    assert 0.0 <= perf1["success_rate"] <= 1.0

    # Fourth attempt should be better
    perf4 = estimate_human_performance("GoToGoal-v0", num_attempts=4)
    assert perf4["attempt"] == 4
    assert perf4["success_rate"] >= perf1["success_rate"]
    assert perf4["avg_steps"] <= perf1["avg_steps"]
    assert perf4["optimal_ratio"] <= perf1["optimal_ratio"]


def test_compare_to_human():
    """Test comparing agent to human baseline."""
    comparison = compare_to_human("GoToGoal-v0", agent_success_rate=0.85)
    assert "success_rate_ratio" in comparison
    assert comparison["difficulty"] == "easy"

    # With steps
    comparison_with_steps = compare_to_human(
        "GoToGoal-v0", agent_success_rate=0.85, agent_avg_steps=15.0
    )
    assert "steps_ratio" in comparison_with_steps
    assert "agent_efficiency" in comparison_with_steps


def test_get_baselines_by_difficulty():
    """Test filtering baselines by difficulty."""
    easy_tasks = get_baselines_by_difficulty("easy")
    assert "GoToGoal-v0" in easy_tasks

    hard_tasks = get_baselines_by_difficulty("hard")
    assert len(hard_tasks) > 0
    assert all(b["difficulty"] == "hard" for b in hard_tasks.values())


def test_get_summary_statistics():
    """Test summary statistics."""
    stats = get_summary_statistics()

    assert stats["num_tasks"] == len(HUMAN_BASELINES)
    assert "success_rate" in stats
    assert "avg_steps" in stats
    assert "optimal_ratio" in stats
    assert "by_difficulty" in stats

    # Check ranges
    assert 0.0 < stats["success_rate"]["mean"] < 1.0
    assert stats["avg_steps"]["mean"] > 0
    assert stats["optimal_ratio"]["mean"] >= 1.0


def test_difficulty_distribution():
    """Test that difficulty levels are reasonable."""
    stats = get_summary_statistics()
    difficulty_counts = stats["by_difficulty"]

    # Should have tasks at different difficulty levels
    assert difficulty_counts["easy"] > 0
    assert difficulty_counts["hard"] > 0
    assert difficulty_counts["very-hard"] > 0

    # Total should match
    total = sum(difficulty_counts.values())
    assert total == len(HUMAN_BASELINES)


def test_learning_curves_realistic():
    """Test that learning curves are realistic."""
    for task_name, baseline in HUMAN_BASELINES.items():
        curve = baseline["learning_curve"]

        # Learning should show improvement
        improvement = curve[-1] - curve[0]
        assert improvement >= 0, f"No learning improvement for {task_name}"

        # Difficult tasks should show more improvement
        if baseline["difficulty"] in ["hard", "very-hard"]:
            assert improvement > 0.1, f"Too little learning for hard task {task_name}"


def test_consistency_with_difficulty():
    """Test that difficulty correlates with success rate."""
    easy = [b["success_rate"] for b in get_baselines_by_difficulty("easy").values()]
    hard = [b["success_rate"] for b in get_baselines_by_difficulty("hard").values()]
    very_hard = [b["success_rate"] for b in get_baselines_by_difficulty("very-hard").values()]

    # Easy tasks should generally have higher success rates
    if easy and hard:
        assert np.mean(easy) > np.mean(hard)

    if hard and very_hard:
        assert np.mean(hard) > np.mean(very_hard)


def test_optimal_ratio_consistency():
    """Test that optimal_ratio is consistent with difficulty."""
    for task_name, baseline in HUMAN_BASELINES.items():
        ratio = baseline["optimal_ratio"]

        # Easier tasks should have better (lower) optimal ratios
        if baseline["difficulty"] == "easy":
            assert ratio < 1.5, f"Optimal ratio too high for easy task {task_name}"
        elif baseline["difficulty"] == "very-hard":
            assert ratio > 1.3, f"Optimal ratio too low for very-hard task {task_name}"


def test_baseline_reproducibility():
    """Test that baselines are reproducible."""
    baseline1 = get_human_baseline("GoToGoal-v0")
    baseline2 = get_human_baseline("GoToGoal-v0")

    assert baseline1 == baseline2


def test_estimate_performance_edge_cases():
    """Test edge cases in performance estimation."""
    # Zero attempts (should use attempt 1)
    perf0 = estimate_human_performance("GoToGoal-v0", num_attempts=0)
    assert perf0["attempt"] == 0

    # Many attempts (should plateau)
    perf10 = estimate_human_performance("GoToGoal-v0", num_attempts=10)
    perf20 = estimate_human_performance("GoToGoal-v0", num_attempts=20)

    # Should plateau (not much difference after many attempts)
    assert abs(perf10["success_rate"] - perf20["success_rate"]) < 0.05


def test_all_tasks_covered():
    """Verify we have exactly the expected tasks."""
    expected_count = 40  # Update if tasks are added

    assert len(HUMAN_BASELINES) == expected_count, (
        f"Expected {expected_count} tasks, found {len(HUMAN_BASELINES)}"
    )
