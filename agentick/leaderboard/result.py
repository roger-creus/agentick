"""Evaluation result models."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

class EpisodeResult(BaseModel):
    """Result for a single episode."""

    task_name: str
    difficulty: str
    seed: int
    episode_return: float
    steps: int
    success: bool
    trajectory: list[dict[str, Any]] | None = None  # Optional trajectory data


class EvaluationResult(BaseModel):
    """Complete evaluation result for an agent on a benchmark suite."""

    # === Identity ===
    submission: dict[str, Any] = Field(default_factory=dict)
    suite_name: str
    suite_version: str
    suite_hash: str

    # === Timing ===
    started_at: datetime
    completed_at: datetime
    wall_time_seconds: float

    # === Scores ===
    agentick_score: float
    agentick_score_ci: tuple[float, float]
    per_capability: dict[str, dict[str, Any]]  # Serialized CapabilityScore
    per_task: dict[str, dict[str, Any]]  # Serialized TaskScore

    # === Episodes ===
    episodes: list[EpisodeResult]

    # === Reproducibility ===
    reproducibility_verified: bool = False
    reproducibility_delta: float | None = None

    # === Metadata ===
    evaluator_version: str  # agentick version
    result_hash: str  # SHA256 of results
    hardware_info: dict[str, Any] = Field(default_factory=dict)

    # === Cost Tracking ===
    total_api_calls: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: float | None = None

    def to_json(self, path: str | Path) -> None:
        """Save result to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2, default=str)

    @classmethod
    def from_json(cls, path: str | Path) -> EvaluationResult:
        """Load result from JSON file."""
        with open(path) as f:
            data = json.load(f)

        return cls(**data)

    def get_summary(self) -> str:
        """Get human-readable summary of results."""
        lines = [
            f"=== Evaluation Results: {self.submission.agent_name} ===",
            f"Suite: {self.suite_name}",
            f"Agentick Score: {self.agentick_score:.3f} ({self.agentick_score_ci[0]:.3f}-{self.agentick_score_ci[1]:.3f})",
            f"Wall Time: {self.wall_time_seconds:.1f}s",
            "",
            "Per-Capability Scores:",
        ]

        for cap_name, cap_data in self.per_capability.items():
            score = cap_data.get("mean_normalized_score", 0.0)
            lines.append(f"  {cap_name}: {score:.3f}")

        if self.total_api_calls is not None:
            lines.append(f"\nAPI Calls: {self.total_api_calls}")
            lines.append(f"Tokens: {self.total_tokens}")
            lines.append(f"Estimated Cost: ${self.estimated_cost_usd:.2f}")

        return "\n".join(lines)
