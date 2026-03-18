"""End-to-end test of full pipeline."""

import pytest

from agentick.experiments.config import ExperimentConfig
from agentick.experiments.runner import ExperimentRunner


@pytest.mark.slow
def test_full_pipeline(tmp_path):
    """
    Test full pipeline: experiment -> analysis -> output.

    Should complete in under 5 minutes.
    """
    # Step 1: Run experiment
    config = ExperimentConfig(
        name="e2e_test",
        agent={"type": "random"},
        tasks=["GoToGoal-v0", "MazeNavigation-v0"],
        n_episodes=5,  # Keep small for speed
        n_seeds=2,
        output_dir=str(tmp_path / "results"),
    )

    runner = ExperimentRunner(config)
    results = runner.run()

    # Verify experiment ran
    assert results is not None
    assert hasattr(results, "summary")
    assert hasattr(results, "per_task_results")

    # Step 2: Verify output structure
    output_dir = results.output_dir
    assert (output_dir / "config.yaml").exists()
    assert (output_dir / "metadata.json").exists()
    assert (output_dir / "summary.json").exists()


@pytest.mark.slow
def test_quick_sanity_check(tmp_path):
    """Quick sanity check that runs in <2 minutes."""
    config = ExperimentConfig(
        name="sanity_check",
        agent={"type": "random"},
        tasks=["GoToGoal-v0"],
        n_episodes=3,
        n_seeds=1,
        output_dir=str(tmp_path / "results"),
    )

    runner = ExperimentRunner(config)
    results = runner.run()

    # Basic checks
    assert results.summary["total_episodes"] == 3
    assert "GoToGoal-v0" in results.per_task_results


@pytest.mark.slow
def test_multi_agent_comparison(tmp_path):
    """Test comparing multiple agents."""
    agents = [
        {"name": "random", "type": "random"},
        {"name": "greedy", "type": "greedy"},
    ]

    all_results = {}

    for agent_config in agents:
        config = ExperimentConfig(
            name=f"agent_{agent_config['name']}",
            agent={"type": agent_config["type"]},
            tasks=["GoToGoal-v0"],
            n_episodes=3,
            n_seeds=1,
            output_dir=str(tmp_path / f"results_{agent_config['name']}"),
        )

        runner = ExperimentRunner(config)
        all_results[agent_config["name"]] = runner.run()

    assert len(all_results) == 2
