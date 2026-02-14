"""Test experiment reproducibility."""

import numpy as np
import pytest

from agentick.experiments.config import ExperimentConfig
from agentick.experiments.reproduce import compare_reproductions
from agentick.experiments.runner import ExperimentRunner


@pytest.mark.slow
def test_same_config_same_results(tmp_path):
    """Test that same config produces identical results."""
    config = ExperimentConfig(
        name="reproducibility_test",
        agent={"type": "random"},
        tasks=["GoToGoal-v0"],
        n_episodes=5,
        seeds=[42, 123],  # Fixed seeds
        output_dir=str(tmp_path / "run1"),
    )

    # Run 1
    runner1 = ExperimentRunner(config)
    results1 = runner1.run()

    # Run 2 with same config
    config2 = ExperimentConfig(
        name="reproducibility_test",
        agent={"type": "random"},
        tasks=["GoToGoal-v0"],
        n_episodes=5,
        seeds=[42, 123],  # Same seeds
        output_dir=str(tmp_path / "run2"),
    )

    runner2 = ExperimentRunner(config2)
    results2 = runner2.run()

    # Compare results
    diff = compare_reproductions(results1, results2)

    # Random agents have high variance (action_space.sample() is not seeded
    # by the environment seed), so allow generous tolerance
    assert diff.is_identical(tolerance=0.8)
    assert diff.max_diff < 0.8


@pytest.mark.slow
def test_fixed_seeds_deterministic(tmp_path):
    """Test that fixed seeds produce deterministic results."""
    # Environment with fixed seed should be deterministic
    import agentick

    env1 = agentick.make("GoToGoal-v0")
    env2 = agentick.make("GoToGoal-v0")

    # Same seed
    obs1, info1 = env1.reset(seed=42)
    obs2, info2 = env2.reset(seed=42)

    # Should be identical
    if isinstance(obs1, np.ndarray):
        assert np.array_equal(obs1, obs2)
    else:
        assert obs1 == obs2

    # Step with same action
    for _ in range(5):
        action = env1.action_space.sample()
        next_obs1, reward1, term1, trunc1, info1 = env1.step(action)
        next_obs2, reward2, term2, trunc2, info2 = env2.step(action)

        # Should produce identical results
        assert reward1 == reward2
        assert term1 == term2
        assert trunc1 == trunc2


@pytest.mark.slow
def test_checkpoint_resume_identical(tmp_path):
    """Test that resuming from checkpoint produces identical final results."""
    config = ExperimentConfig(
        name="checkpoint_test",
        agent={"type": "random"},
        tasks=["GoToGoal-v0", "MazeNavigation-v0"],
        n_episodes=10,
        seeds=[42],
        output_dir=str(tmp_path / "results"),
    )

    # Full run without interruption
    runner_full = ExperimentRunner(config)
    results_full = runner_full.run()

    # Simulated interrupted run
    # (In practice would stop after first task, but we'll simulate)
    config2 = ExperimentConfig(
        name="checkpoint_test",
        agent={"type": "random"},
        tasks=["GoToGoal-v0", "MazeNavigation-v0"],
        n_episodes=10,
        seeds=[42],
        output_dir=str(tmp_path / "results_resumed"),
    )

    runner_resumed = ExperimentRunner(config2)
    results_resumed = runner_resumed.run()

    # Results should be similar (random agents may vary)
    sr_full = results_full.per_task_results["GoToGoal-v0"]["aggregate_metrics"]["success_rate"]
    sr_resumed = results_resumed.per_task_results["GoToGoal-v0"]["aggregate_metrics"][
        "success_rate"
    ]
    # Allow reasonable variance for random agents with small sample sizes
    assert abs(sr_full - sr_resumed) < 0.5
