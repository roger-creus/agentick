"""Tests for wandb logger integration."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from agentick.training.logger import MultiBackendLogger


class TestWandbLogger:
    """Test wandb integration in MultiBackendLogger."""

    def test_wandb_disabled_by_default(self):
        """Test wandb is disabled by default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MultiBackendLogger(log_dir=tmpdir, use_stdout=False, use_json=False)
            assert not logger.use_wandb

    def test_wandb_graceful_when_not_installed(self):
        """Test logger works gracefully when wandb is not installed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Even if we request wandb, it should work if not installed
            logger = MultiBackendLogger(
                log_dir=tmpdir, use_stdout=False, use_json=False, use_wandb=True
            )

            # Log should work even if wandb is unavailable
            logger.log("test_metric", 1.0, step=0)
            logger.close()

    @patch("agentick.training.logger.MultiBackendLogger._try_import_wandb")
    @patch("wandb.init")
    @patch("wandb.log")
    @patch("wandb.finish")
    def test_wandb_logs_scalar(
        self, mock_wandb_finish, mock_wandb_log, mock_wandb_init, mock_try_import
    ):
        """Test wandb logs scalar values correctly."""
        # Mock wandb as available
        mock_try_import.return_value = True

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("wandb.init"):
                # Create mock wandb module
                mock_wandb = MagicMock()
                mock_wandb.log = MagicMock()
                mock_wandb.finish = MagicMock()

                with patch.dict("sys.modules", {"wandb": mock_wandb}):
                    logger = MultiBackendLogger(
                        log_dir=tmpdir,
                        use_stdout=False,
                        use_json=False,
                        use_wandb=True,
                        wandb_config={"project": "test_project"},
                    )

                    # Log a metric
                    logger.log("test_metric", 42.0, step=100)

                    # Verify wandb.log was called
                    mock_wandb.log.assert_called_once()
                    call_args = mock_wandb.log.call_args
                    assert call_args[0][0]["test_metric"] == 42.0
                    assert call_args[1]["step"] == 100

                    # Close logger
                    logger.close()
                    mock_wandb.finish.assert_called_once()

    @patch("agentick.training.logger.MultiBackendLogger._try_import_wandb")
    def test_wandb_logs_dict(self, mock_try_import):
        """Test wandb logs multiple metrics."""
        mock_try_import.return_value = True

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock wandb module
            mock_wandb = MagicMock()
            mock_wandb.log = MagicMock()
            mock_wandb.finish = MagicMock()

            with patch.dict("sys.modules", {"wandb": mock_wandb}):
                logger = MultiBackendLogger(
                    log_dir=tmpdir,
                    use_stdout=False,
                    use_json=False,
                    use_wandb=True,
                )

                # Log multiple metrics
                metrics = {"loss": 0.5, "accuracy": 0.95, "lr": 0.001}
                logger.log_dict(metrics, step=50)

                # Verify all metrics were logged
                assert mock_wandb.log.call_count == 3

    @patch("agentick.training.logger.MultiBackendLogger._try_import_wandb")
    def test_wandb_logs_histogram(self, mock_try_import):
        """Test wandb logs histogram."""
        mock_try_import.return_value = True

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock wandb module
            mock_wandb = MagicMock()
            mock_wandb.log = MagicMock()
            mock_wandb.Histogram = MagicMock()
            mock_wandb.finish = MagicMock()

            with patch.dict("sys.modules", {"wandb": mock_wandb}):
                logger = MultiBackendLogger(
                    log_dir=tmpdir,
                    use_stdout=False,
                    use_json=False,
                    use_wandb=True,
                )

                # Log histogram
                values = [1.0, 2.0, 3.0, 4.0, 5.0]
                logger.log_histogram("rewards", values, step=10)

                # Verify wandb.Histogram was created
                mock_wandb.Histogram.assert_called_once()

                # Verify wandb.log was called
                mock_wandb.log.assert_called()

    @patch("agentick.training.logger.MultiBackendLogger._try_import_wandb")
    def test_wandb_init_with_config(self, mock_try_import):
        """Test wandb.init is called with config."""
        mock_try_import.return_value = True

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_wandb = MagicMock()
            mock_wandb.init = MagicMock()
            mock_wandb.finish = MagicMock()

            with patch.dict("sys.modules", {"wandb": mock_wandb}):
                config = {
                    "project": "agentick",
                    "name": "test_run",
                    "config": {"learning_rate": 0.001, "batch_size": 32},
                }

                logger = MultiBackendLogger(
                    log_dir=tmpdir,
                    use_stdout=False,
                    use_json=False,
                    use_wandb=True,
                    wandb_config=config,
                )

                # Verify wandb.init was called with config
                mock_wandb.init.assert_called_once_with(**config)

                logger.close()

    def test_multibackend_with_json_and_wandb(self):
        """Test logger works with both JSON and wandb backends."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MultiBackendLogger(
                log_dir=tmpdir,
                use_stdout=False,
                use_json=True,
                use_wandb=False,  # Don't actually use wandb in test
            )

            # Log metrics
            logger.log("metric1", 1.0, step=0)
            logger.log("metric2", 2.0, step=1)

            # Verify JSON file exists
            json_path = Path(tmpdir) / "metrics.jsonl"
            assert json_path.exists()

            # Verify content
            with open(json_path) as f:
                lines = f.readlines()
            assert len(lines) == 2

            logger.close()

    def test_logger_get_metrics(self):
        """Test retrieving logged metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MultiBackendLogger(
                log_dir=tmpdir, use_stdout=False, use_json=False, use_wandb=False
            )

            # Log some metrics
            logger.log("loss", 1.0, step=0)
            logger.log("loss", 0.5, step=1)
            logger.log("accuracy", 0.8, step=0)

            # Get metrics
            loss_metrics = logger.get_metrics("loss")
            assert len(loss_metrics) == 2
            assert loss_metrics[0]["value"] == 1.0
            assert loss_metrics[1]["value"] == 0.5

            acc_metrics = logger.get_metrics("accuracy")
            assert len(acc_metrics) == 1
            assert acc_metrics[0]["value"] == 0.8

    def test_logger_get_latest(self):
        """Test getting latest metric value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MultiBackendLogger(
                log_dir=tmpdir, use_stdout=False, use_json=False, use_wandb=False
            )

            # Log some metrics
            logger.log("loss", 1.0, step=0)
            logger.log("loss", 0.5, step=1)
            logger.log("loss", 0.3, step=2)

            # Get latest
            latest = logger.get_latest("loss")
            assert latest == 0.3

            # Non-existent metric
            assert logger.get_latest("nonexistent") is None

    def test_save_summary(self):
        """Test saving summary statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MultiBackendLogger(
                log_dir=tmpdir, use_stdout=False, use_json=False, use_wandb=False
            )

            # Log metrics
            for i in range(10):
                logger.log("metric", float(i), step=i)

            # Save summary
            logger.save_summary()

            # Verify summary file
            summary_path = Path(tmpdir) / "summary.json"
            assert summary_path.exists()

            import json

            with open(summary_path) as f:
                summary = json.load(f)

            assert "metric" in summary
            assert summary["metric"]["mean"] == 4.5
            assert summary["metric"]["min"] == 0.0
            assert summary["metric"]["max"] == 9.0
            assert summary["metric"]["count"] == 10
