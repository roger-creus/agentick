"""Tests for performance profiler."""

import pytest

from agentick.benchmark.profiler import (
    generate_profiling_report,
    identify_bottlenecks,
    profile_all_tasks,
    profile_render_modes,
    profile_task_step,
)


def test_profile_task_step():
    """Test profiling a single task."""
    result = profile_task_step("GoToGoal-v0", num_steps=100, warmup_steps=10)

    assert result.task_name == "GoToGoal-v0"
    assert result.num_steps == 100
    assert result.steps_per_sec > 0
    assert result.avg_step_time_us > 0
    assert result.p50_step_time_us > 0
    assert result.p95_step_time_us > 0
    assert result.p99_step_time_us > 0
    assert result.min_step_time_us <= result.p50_step_time_us
    assert result.p50_step_time_us <= result.p95_step_time_us
    assert result.p95_step_time_us <= result.p99_step_time_us
    assert result.render_mode == "state_dict"


def test_profile_multiple_render_modes():
    """Test profiling different render modes."""
    result_state = profile_task_step("GoToGoal-v0", num_steps=100, render_mode="state_dict")
    result_rgb = profile_task_step("GoToGoal-v0", num_steps=100, render_mode="rgb_array")

    assert result_state.render_mode == "state_dict"
    assert result_rgb.render_mode == "rgb_array"

    # state_dict should generally be faster than rgb_array
    # (but this is not guaranteed, so we just check both work)
    assert result_state.steps_per_sec > 0
    assert result_rgb.steps_per_sec > 0


def test_profile_all_tasks_small():
    """Test profiling all tasks with small step count."""
    # Use very small step count for testing
    results = profile_all_tasks(num_steps=50)

    # Filter out test tasks
    results = {k: v for k, v in results.items() if not k.startswith("Test")}

    # Should have results for all non-test tasks
    assert len(results) >= 38  # We have 38 tasks

    # All results should be valid
    for task_name, result in results.items():
        assert result.steps_per_sec > 0
        assert result.avg_step_time_us > 0


def test_generate_profiling_report():
    """Test generating a profiling report."""
    # Profile a few tasks
    results = {
        "GoToGoal-v0": profile_task_step("GoToGoal-v0", num_steps=100),
        "MazeNavigation-v0": profile_task_step("MazeNavigation-v0", num_steps=100),
    }

    report = generate_profiling_report(results)

    # Check report contains expected sections
    assert "AgentTick Performance Profile" in report
    assert "Summary Statistics" in report
    assert "Per-Task Performance" in report
    assert "GoToGoal-v0" in report
    assert "MazeNavigation-v0" in report
    assert "Steps/sec" in report


def test_profile_render_modes():
    """Test profiling different render modes."""
    results = profile_render_modes("GoToGoal-v0", num_steps=50)

    # Should have at least state_dict and rgb_array
    assert "state_dict" in results
    assert "rgb_array" in results
    # ansi may not be supported

    for mode, result in results.items():
        assert result.render_mode == mode
        assert result.steps_per_sec > 0


def test_identify_bottlenecks():
    """Test identifying performance bottlenecks."""
    # Create mock results
    results = {
        "FastTask": profile_task_step("GoToGoal-v0", num_steps=100),
    }

    # With very high threshold, should identify task as bottleneck
    bottlenecks = identify_bottlenecks(results, threshold_steps_per_sec=1e9)
    assert "FastTask" in bottlenecks

    # With very low threshold, should not identify as bottleneck
    bottlenecks = identify_bottlenecks(results, threshold_steps_per_sec=1.0)
    assert "FastTask" not in bottlenecks


def test_profiling_result_str():
    """Test string representation of profiling result."""
    result = profile_task_step("GoToGoal-v0", num_steps=100)
    result_str = str(result)

    assert "GoToGoal-v0" in result_str
    assert "Steps/sec" in result_str
    assert "Avg" in result_str
    assert "P50" in result_str
    assert "P95" in result_str
    assert "P99" in result_str


def test_profiling_reproducibility():
    """Test that profiling is reasonably reproducible."""
    result1 = profile_task_step("GoToGoal-v0", num_steps=1000, seed=42)
    result2 = profile_task_step("GoToGoal-v0", num_steps=1000, seed=42)

    # Results should be similar (within 20% due to timing variance)
    ratio = result1.steps_per_sec / result2.steps_per_sec
    assert 0.8 < ratio < 1.2


def test_profiling_different_tasks():
    """Test profiling different task types."""
    simple_task = profile_task_step("GoToGoal-v0", num_steps=100)
    complex_task = profile_task_step("SokobanPush-v0", num_steps=100)

    # Both should complete successfully
    assert simple_task.steps_per_sec > 0
    assert complex_task.steps_per_sec > 0


@pytest.mark.slow
def test_full_profiling_run():
    """Test full profiling of all tasks (slow)."""
    results = profile_all_tasks(num_steps=1000)

    # Should have results for most tasks
    assert len(results) >= 38

    # Generate report
    report = generate_profiling_report(results)
    assert len(report) > 0
    assert "Fastest Tasks" in report
    assert "Slowest Tasks" in report
