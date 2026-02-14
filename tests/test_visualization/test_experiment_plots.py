"""Tests for experiment plotting utilities."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from agentick.visualization.experiment_plots import ExperimentPlotter


class TestExperimentPlotter:
    """Test ExperimentPlotter class."""

    @pytest.fixture
    def mock_experiment_dir(self):
        """Create mock experiment results directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir) / "test_experiment"
            exp_dir.mkdir()

            # Create results.json (what ExperimentPlotter expects)
            results = {
                "tasks": ["GoToGoal-v0", "MazeNavigation-v0"],
                "difficulties": ["easy"],
                "overall_score": 0.75,
                "task_scores": {
                    "GoToGoal-v0": {
                        "mean": 0.80,
                        "success_rate": 0.90,
                        "episode_scores": [0.7, 0.8, 0.9],
                        "episode_lengths": [15, 20, 25],
                    },
                    "MazeNavigation-v0": {
                        "mean": 0.70,
                        "success_rate": 0.70,
                        "episode_scores": [0.6, 0.7, 0.8],
                        "episode_lengths": [30, 35, 40],
                    },
                },
                "capability_scores": {
                    "navigation": {"mean": 0.75},
                    "memory": {"mean": 0.60},
                },
            }

            with open(exp_dir / "results.json", "w") as f:
                json.dump(results, f)

            yield exp_dir

    def test_plotter_init(self, mock_experiment_dir):
        """Test plotter initialization."""
        plotter = ExperimentPlotter(str(mock_experiment_dir))

        assert plotter.result_dir == mock_experiment_dir
        assert plotter.results is not None
        assert "tasks" in plotter.results

    def test_plotter_load_results(self, mock_experiment_dir):
        """Test loading experiment results."""
        plotter = ExperimentPlotter(str(mock_experiment_dir))

        assert plotter.results["overall_score"] == 0.75
        assert len(plotter.results["tasks"]) == 2

    def test_plotter_load_task_results(self, mock_experiment_dir):
        """Test accessing per-task results."""
        plotter = ExperimentPlotter(str(mock_experiment_dir))

        task_scores = plotter.results["task_scores"]
        assert "GoToGoal-v0" in task_scores
        assert "MazeNavigation-v0" in task_scores
        assert task_scores["GoToGoal-v0"]["mean"] == 0.80

    def test_plot_per_task_scores(self, mock_experiment_dir):
        """Test per-task scores plot generation."""
        plotter = ExperimentPlotter(str(mock_experiment_dir))
        # Should not raise
        plotter.plot_per_task_scores()

        plot_file = plotter.figures_dir / "per_task_scores.png"
        assert plot_file.exists()

    def test_plot_success_rate(self, mock_experiment_dir):
        """Test success rate plot generation."""
        plotter = ExperimentPlotter(str(mock_experiment_dir))
        plotter.plot_success_rate()

        plot_file = plotter.figures_dir / "success_rate.png"
        assert plot_file.exists()

    def test_plot_score_distribution(self, mock_experiment_dir):
        """Test score distribution plot generation."""
        plotter = ExperimentPlotter(str(mock_experiment_dir))
        plotter.plot_score_distribution()

        plot_file = plotter.figures_dir / "score_distribution.png"
        assert plot_file.exists()

    def test_plot_all(self, mock_experiment_dir):
        """Test plot_all generates all plots."""
        plotter = ExperimentPlotter(str(mock_experiment_dir))
        plotter.plot_all()

        # Check multiple plots were created
        plot_files = list(plotter.figures_dir.glob("*.png"))
        assert len(plot_files) >= 3

    def test_plotter_with_missing_results(self):
        """Test plotter handles missing results file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir) / "empty_experiment"
            exp_dir.mkdir()

            with pytest.raises(ValueError, match="Results file not found"):
                ExperimentPlotter(str(exp_dir))

    def test_plotter_with_empty_task_scores(self):
        """Test plotter handles empty task scores."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir) / "test_experiment"
            exp_dir.mkdir()

            results = {"tasks": [], "task_scores": {}}
            with open(exp_dir / "results.json", "w") as f:
                json.dump(results, f)

            plotter = ExperimentPlotter(str(exp_dir))
            # Should handle gracefully (no crash)
            plotter.plot_per_task_scores()
