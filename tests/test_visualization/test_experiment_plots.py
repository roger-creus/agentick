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

            # Create summary.json
            summary = {
                "mean_return": 0.75,
                "success_rate": 0.80,
                "mean_length": 25.5,
                "total_episodes": 100,
            }

            with open(exp_dir / "summary.json", "w") as f:
                json.dump(summary, f)

            # Create per-task directory structure
            per_task_dir = exp_dir / "per_task"
            per_task_dir.mkdir()

            # Create mock task results
            for task_name in ["GoToGoal-v0", "MazeNavigation-v0"]:
                task_dir = per_task_dir / task_name
                task_dir.mkdir()

                # Create metrics.json
                metrics = {
                    "task_name": task_name,
                    "aggregate_metrics": {
                        "mean_return": 0.75,
                        "success_rate": 0.80,
                        "mean_length": 25.0,
                    },
                }

                with open(task_dir / "metrics.json", "w") as f:
                    json.dump(metrics, f)

            yield exp_dir

    def test_plotter_init(self, mock_experiment_dir):
        """Test plotter initialization."""
        plotter = ExperimentPlotter(str(mock_experiment_dir))

        assert plotter.exp_dir == mock_experiment_dir
        assert plotter.summary is not None
        assert "mean_return" in plotter.summary

    def test_plotter_load_summary(self, mock_experiment_dir):
        """Test loading experiment summary."""
        plotter = ExperimentPlotter(str(mock_experiment_dir))

        assert plotter.summary["mean_return"] == 0.75
        assert plotter.summary["success_rate"] == 0.80

    def test_plotter_load_task_results(self, mock_experiment_dir):
        """Test loading per-task results."""
        plotter = ExperimentPlotter(str(mock_experiment_dir))

        task_results = plotter._load_task_results()

        assert len(task_results) == 2
        assert any(r["task"] == "GoToGoal-v0" for r in task_results)
        assert any(r["task"] == "MazeNavigation-v0" for r in task_results)

    def test_plot_per_task_scores(self, mock_experiment_dir):
        """Test per-task scores plot generation."""
        plotter = ExperimentPlotter(str(mock_experiment_dir))

        with tempfile.TemporaryDirectory() as output_dir:
            # Should not raise
            plotter.plot_per_task_scores(output_dir=output_dir)

            # Check file was created
            plot_file = Path(output_dir) / "per_task_scores.png"
            assert plot_file.exists()

    def test_plot_success_rate(self, mock_experiment_dir):
        """Test success rate plot generation."""
        plotter = ExperimentPlotter(str(mock_experiment_dir))

        with tempfile.TemporaryDirectory() as output_dir:
            plotter.plot_success_rate(output_dir=output_dir)

            plot_file = Path(output_dir) / "success_rate.png"
            assert plot_file.exists()

    def test_plot_score_distribution(self, mock_experiment_dir):
        """Test score distribution plot generation."""
        plotter = ExperimentPlotter(str(mock_experiment_dir))

        with tempfile.TemporaryDirectory() as output_dir:
            plotter.plot_score_distribution(output_dir=output_dir)

            plot_file = Path(output_dir) / "score_distribution.png"
            assert plot_file.exists()

    def test_plot_all(self, mock_experiment_dir):
        """Test plot_all generates all plots."""
        plotter = ExperimentPlotter(str(mock_experiment_dir))

        with tempfile.TemporaryDirectory() as output_dir:
            plotter.plot_all(output_dir=output_dir)

            # Check multiple plots were created
            output_path = Path(output_dir)
            plot_files = list(output_path.glob("*.png"))

            # Should have created multiple plots
            assert len(plot_files) >= 3

    def test_plotter_with_missing_summary(self):
        """Test plotter handles missing summary file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir) / "empty_experiment"
            exp_dir.mkdir()

            # Should raise or handle gracefully
            with pytest.raises(FileNotFoundError):
                ExperimentPlotter(str(exp_dir))

    def test_plotter_with_empty_task_dir(self, mock_experiment_dir):
        """Test plotter handles empty per_task directory."""
        plotter = ExperimentPlotter(str(mock_experiment_dir))

        # Remove task directories
        per_task_dir = mock_experiment_dir / "per_task"
        for task_dir in per_task_dir.iterdir():
            for f in task_dir.iterdir():
                f.unlink()
            task_dir.rmdir()

        # Should handle gracefully
        task_results = plotter._load_task_results()
        assert len(task_results) == 0
