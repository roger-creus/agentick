"""Multi-backend logging for training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class MultiBackendLogger:
    """
    Logger that supports multiple backends simultaneously.

    Backends: stdout, JSON file, wandb (if available), tensorboard (if available)
    """

    def __init__(
        self,
        log_dir: str | Path | None = None,
        use_stdout: bool = True,
        use_json: bool = True,
        use_wandb: bool = False,
        use_tensorboard: bool = False,
        wandb_config: dict[str, Any] | None = None,
        tensorboard_logdir: str | None = None,
    ):
        """
        Initialize multi-backend logger.

        Args:
            log_dir: Directory for file-based logs
            use_stdout: Print logs to stdout
            use_json: Save logs to JSON file
            use_wandb: Use Weights & Biases (requires wandb installed)
            use_tensorboard: Use TensorBoard (requires tensorboardX or torch.utils.tensorboard)
            wandb_config: Config dict for wandb.init()
            tensorboard_logdir: Directory for tensorboard logs
        """
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.use_stdout = use_stdout
        self.use_json = use_json
        self.use_wandb = use_wandb and self._try_import_wandb()
        self.use_tensorboard = use_tensorboard and self._try_import_tensorboard()

        self.logs: list[dict[str, Any]] = []

        # Initialize backends
        if self.use_json:
            self.json_path = self.log_dir / "metrics.jsonl"

        if self.use_wandb:
            import wandb

            wandb.init(**(wandb_config or {}))
            self.wandb = wandb

        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                from tensorboardX import SummaryWriter

            tb_dir = tensorboard_logdir or str(self.log_dir / "tensorboard")
            self.tb_writer = SummaryWriter(tb_dir)

    def log(self, key: str, value: float | int, step: int | None = None) -> None:
        """
        Log a scalar value.

        Args:
            key: Metric name
            value: Metric value
            step: Optional step/iteration number
        """
        log_entry = {"key": key, "value": value}
        if step is not None:
            log_entry["step"] = step

        self.logs.append(log_entry)

        # Stdout
        if self.use_stdout:
            step_str = f"[Step {step}] " if step is not None else ""
            print(f"{step_str}{key}: {value}")

        # JSON file
        if self.use_json:
            with open(self.json_path, "a") as f:
                json.dump(log_entry, f)
                f.write("\n")

        # Wandb
        if self.use_wandb:
            log_dict = {key: value}
            if step is not None:
                log_dict["step"] = step
            self.wandb.log(log_dict, step=step)

        # TensorBoard
        if self.use_tensorboard:
            self.tb_writer.add_scalar(key, value, step)

    def log_dict(self, metrics: dict[str, float | int], step: int | None = None) -> None:
        """
        Log multiple metrics at once.

        Args:
            metrics: Dict of metric name -> value
            step: Optional step/iteration number
        """
        for key, value in metrics.items():
            self.log(key, value, step)

    def log_histogram(
        self,
        key: str,
        values: list[float],
        step: int | None = None,
    ) -> None:
        """
        Log histogram of values.

        Only supported by TensorBoard and Wandb.

        Args:
            key: Metric name
            values: List of values
            step: Optional step number
        """
        if self.use_wandb:
            import numpy as np

            self.wandb.log({key: self.wandb.Histogram(np.array(values))}, step=step)

        if self.use_tensorboard:
            import numpy as np

            self.tb_writer.add_histogram(key, np.array(values), step)

    def save_summary(self, filename: str = "summary.json") -> None:
        """
        Save summary statistics of all logged metrics.

        Args:
            filename: Name of summary file
        """
        import numpy as np

        summary = {}

        # Group logs by key
        by_key: dict[str, list[float]] = {}
        for log in self.logs:
            key = log["key"]
            value = log["value"]
            if key not in by_key:
                by_key[key] = []
            by_key[key].append(value)

        # Compute statistics
        for key, values in by_key.items():
            summary[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "count": len(values),
            }

        # Save to file
        summary_path = self.log_dir / filename
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        if self.use_stdout:
            print(f"Summary saved to {summary_path}")

    def close(self) -> None:
        """Close all backend connections."""
        if self.use_wandb:
            self.wandb.finish()

        if self.use_tensorboard:
            self.tb_writer.close()

    def _try_import_wandb(self) -> bool:
        """Try importing wandb."""
        try:
            import wandb  # noqa: F401

            return True
        except ImportError:
            return False

    def _try_import_tensorboard(self) -> bool:
        """Try importing tensorboard."""
        try:
            from torch.utils.tensorboard import SummaryWriter  # noqa: F401

            return True
        except ImportError:
            try:
                from tensorboardX import SummaryWriter  # noqa: F401

                return True
            except ImportError:
                return False

    def get_metrics(self, key: str) -> list[dict[str, Any]]:
        """
        Get all logged values for a specific metric.

        Args:
            key: Metric name

        Returns:
            List of log entries for this metric
        """
        return [log for log in self.logs if log["key"] == key]

    def get_latest(self, key: str) -> float | None:
        """
        Get most recent value for a metric.

        Args:
            key: Metric name

        Returns:
            Latest value or None if not found
        """
        matching = self.get_metrics(key)
        return matching[-1]["value"] if matching else None


class StdoutLogger:
    """Simple logger that only prints to stdout."""

    def log(self, key: str, value: float | int, step: int | None = None) -> None:
        """Log to stdout."""
        step_str = f"[Step {step}] " if step is not None else ""
        print(f"{step_str}{key}: {value}")

    def log_dict(self, metrics: dict[str, float | int], step: int | None = None) -> None:
        """Log multiple metrics."""
        for key, value in metrics.items():
            self.log(key, value, step)

    def close(self) -> None:
        """No-op for stdout logger."""
        pass
