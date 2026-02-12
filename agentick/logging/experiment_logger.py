"""Experiment-level logging."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class ExperimentLogger:
    """Logger for experiment-level metadata."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.start_time = time.time()
        self.end_time: float | None = None
        self.task_timings: dict[str, float] = {}
        self.errors: list[dict[str, Any]] = []
        self.warnings: list[str] = []

    def log_task_start(self, task_name: str) -> None:
        """Log task start."""
        self.task_timings[task_name] = time.time()

    def log_task_end(self, task_name: str) -> None:
        """Log task end."""
        if task_name in self.task_timings:
            elapsed = time.time() - self.task_timings[task_name]
            self.task_timings[task_name] = elapsed

    def log_error(self, error: Exception, context: dict[str, Any] | None = None) -> None:
        """Log error."""
        self.errors.append(
            {
                "type": type(error).__name__,
                "message": str(error),
                "context": context or {},
                "timestamp": time.time(),
            }
        )

    def log_warning(self, message: str) -> None:
        """Log warning."""
        self.warnings.append(message)

    def finalize(self) -> None:
        """Finalize and save log."""
        self.end_time = time.time()

        log_data = {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_time": self.end_time - self.start_time,
            "task_timings": self.task_timings,
            "errors": self.errors,
            "warnings": self.warnings,
        }

        log_path = self.output_dir / "experiment.log.json"
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)
