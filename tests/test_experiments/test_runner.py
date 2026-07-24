"""Tests for experiment runner."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

from agentick.experiments.config import ExperimentConfig
from agentick.experiments.runner import ExperimentResults, ExperimentRunner


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


def test_concurrent_difficulty_saves_preserve_both_payloads(tmp_path):
    """Concurrent jobs for one task must not lose a difficulty record."""
    output_dir = tmp_path / "results"
    config = ExperimentConfig(
        name="concurrent_save",
        agent={"type": "random"},
        tasks=["GoToGoal-v0"],
        n_episodes=1,
        n_seeds=1,
        output_dir=str(output_dir),
    )
    output_dir.mkdir(parents=True)
    config.to_yaml(output_dir / "config.yaml")

    def make_result(difficulty, seed, success):
        episode = {
            "seed": seed,
            "episode_idx": 0,
            "return": float(success),
            "length": 1,
            "success": success,
        }
        task_result = {
            "task_name": "GoToGoal-v0",
            "per_difficulty": {
                difficulty: {
                    "difficulty": difficulty,
                    "episodes": [episode],
                    "metrics": {"success_rate": float(success)},
                }
            },
            "aggregate_metrics": {
                "mean_return": float(success),
                "success_rate": float(success),
                "mean_length": 1.0,
            },
        }
        return ExperimentResults(
            config=config,
            output_dir=output_dir,
            metadata={"difficulty": difficulty},
            summary={"total_time_seconds": 1.0},
            per_task_results={"GoToGoal-v0": task_result},
        )

    barrier = Barrier(2)

    def save(result):
        barrier.wait()
        result.save()

    results = [
        make_result("easy", 101, True),
        make_result("hard", 202, False),
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(save, result) for result in results]
        for future in futures:
            future.result()

    metrics_path = output_dir / "per_task" / "GoToGoal-v0" / "metrics.json"
    with open(metrics_path) as f:
        metrics = json.load(f)

    assert set(metrics["per_difficulty"]) == {"easy", "hard"}
    assert metrics["per_difficulty"]["easy"]["episodes"][0]["seed"] == 101
    assert metrics["per_difficulty"]["hard"]["episodes"][0]["seed"] == 202
