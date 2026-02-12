"""Tests for experiment runner."""

import json
from pathlib import Path

from agentick.experiments.config import ExperimentConfig
from agentick.experiments.runner import ExperimentRunner


def test_basic_run(tmp_path):
    """Test basic experiment run."""
    config = ExperimentConfig(
        name="test_run",
        agent={"type": "random"},
        tasks=["GoToGoal-v0"],
        n_episodes=2,
        n_seeds=1,
        output_dir=str(tmp_path / "results"),
    )

    runner = ExperimentRunner(config)
    results = runner.run()

    assert results is not None
    assert results.summary is not None
    assert results.per_task_results is not None
    assert "GoToGoal-v0" in results.per_task_results


def test_checkpoint_resume(tmp_path):
    """Test checkpoint and resume."""
    config = ExperimentConfig(
        name="test_checkpoint",
        agent={"type": "random"},
        tasks=["GoToGoal-v0", "MazeNavigation-v0"],
        n_episodes=2,
        n_seeds=1,
        output_dir=str(tmp_path / "results"),
    )

    runner = ExperimentRunner(config)

    # Run and save checkpoint
    output_dir = Path(config.output_dir)
    checkpoint_path = output_dir / ".checkpoint.json"

    # Simulate partial run by manually creating checkpoint
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_data = {
        "config": config.model_dump(),
        "completed_tasks": ["GoToGoal-v0"],
        "results": {
            "per_task": {
                "GoToGoal-v0": {
                    "success_rate": 0.5,
                    "mean_return": 5.0,
                }
            }
        },
    }
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint_data, f)

    # Resume
    results = runner.run(resume_from=str(checkpoint_path))

    assert results is not None
    # Should have both tasks
    assert "GoToGoal-v0" in results.per_task_results
    assert "MazeNavigation-v0" in results.per_task_results


def test_parallel_execution(tmp_path):
    """Test parallel execution."""
    config = ExperimentConfig(
        name="test_parallel",
        agent={"type": "random"},
        tasks=["GoToGoal-v0", "MazeNavigation-v0"],
        n_episodes=2,
        n_seeds=2,
        output_dir=str(tmp_path / "results"),
    )

    runner = ExperimentRunner(config)
    results = runner.run(n_parallel=2)

    assert results is not None
    assert "GoToGoal-v0" in results.per_task_results
    assert "MazeNavigation-v0" in results.per_task_results


def test_output_structure(tmp_path):
    """Test output directory structure."""
    config = ExperimentConfig(
        name="test_output",
        agent={"type": "random"},
        tasks=["GoToGoal-v0"],
        n_episodes=1,
        n_seeds=1,
        output_dir=str(tmp_path / "results"),
    )

    runner = ExperimentRunner(config)
    results = runner.run()

    output_dir = results.output_dir

    # Check structure
    assert (output_dir / "config.yaml").exists()
    assert (output_dir / "metadata.json").exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "per_task").exists()
    assert (output_dir / "per_task" / "GoToGoal-v0" / "metrics.json").exists()


def test_metadata_tracking(tmp_path):
    """Test metadata is captured."""
    config = ExperimentConfig(
        name="test_metadata",
        agent={"type": "random"},
        tasks=["GoToGoal-v0"],
        n_episodes=1,
        n_seeds=1,
        output_dir=str(tmp_path / "results"),
    )

    runner = ExperimentRunner(config)
    results = runner.run()

    metadata_path = results.output_dir / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)

    assert "timestamp" in metadata
    assert "agentick_version" in metadata
    assert "python_version" in metadata
    assert "platform" in metadata
