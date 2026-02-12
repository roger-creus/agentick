"""Per-episode structured logging."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any


class EpisodeLogger:
    """Logger for episode trajectories."""

    def __init__(
        self,
        output_path: str | Path,
        verbosity: str = "standard",
        compress: bool = False,
        mode: str = "w",
    ):
        """
        Initialize episode logger.

        Args:
            output_path: Path to save log file
            verbosity: One of minimal, standard, full, debug
            compress: Whether to compress with gzip
            mode: File mode - 'w' for write (default), 'a' for append
        """
        self.output_path = Path(output_path)
        self.verbosity = verbosity
        self.compress = compress
        self.mode = mode
        self.steps: list[dict[str, Any]] = []

        # If appending, load existing data
        if mode == "a" and self.output_path.exists():
            try:
                self.steps = self.load(self.output_path)
            except Exception:
                # If loading fails, start fresh
                self.steps = []

    def log_step(
        self,
        step: int,
        action: dict[str, Any],
        observation: dict[str, Any],
        reward: dict[str, Any],
        info: dict[str, Any],
        agent_internals: dict[str, Any] | None = None,
    ) -> None:
        """
        Log a single step.

        Args:
            step: Step number
            action: Action dict
            observation: Observation dict
            reward: Reward dict
            info: Info dict
            agent_internals: Optional agent internal state
        """
        import time

        step_data = {
            "step": step,
            "timestamp_ms": int(time.time() * 1000),
            "action": action,
            "reward": reward,
            "info": info,
        }

        # Add observation based on verbosity
        if self.verbosity == "minimal":
            # Skip observation
            pass
        elif self.verbosity == "standard":
            step_data["observation"] = {"ascii": observation.get("ascii")}
        elif self.verbosity == "full":
            step_data["observation"] = observation
        elif self.verbosity == "debug":
            step_data["observation"] = observation
            step_data["agent_internals"] = agent_internals or {}

        self.steps.append(step_data)

    def save(self) -> None:
        """Save log to disk."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Auto-detect if compression is needed based on file extension
        use_compression = self.compress or self.output_path.suffix == ".gz"

        if use_compression:
            output_file = (
                self.output_path
                if self.output_path.suffix == ".gz"
                else self.output_path.with_suffix(".jsonl.gz")
            )
            with gzip.open(output_file, "wt") as f:
                for step in self.steps:
                    f.write(json.dumps(step) + "\n")
        else:
            with open(self.output_path, "w") as f:
                json.dump(self.steps, f, indent=2)

    def flush(self) -> None:
        """Flush buffered steps to disk without clearing buffer."""
        self.save()

    def close(self) -> None:
        """Save and close the logger."""
        self.save()
        self.steps.clear()

    @classmethod
    def load(cls, log_path: str | Path) -> list[dict[str, Any]]:
        """Load log from disk."""
        log_path = Path(log_path)

        if log_path.suffix == ".gz":
            with gzip.open(log_path, "rt") as f:
                return [json.loads(line) for line in f]
        else:
            with open(log_path) as f:
                return json.load(f)
