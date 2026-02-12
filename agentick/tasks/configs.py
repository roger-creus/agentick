"""Pydantic configuration models for tasks."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class GridConfig(BaseModel):
    """Configuration for grid generation."""

    model_config = ConfigDict(extra="allow")

    height: int = Field(..., ge=3, le=100, description="Grid height")
    width: int = Field(..., ge=3, le=100, description="Grid width")
    wall_density: float = Field(0.0, ge=0.0, le=1.0, description="Fraction of cells that are walls")

    # Entity positions
    agent_start: tuple[int, int] | None = Field(None, description="Agent starting position")
    goal_positions: list[tuple[int, int]] = Field(
        default_factory=list, description="Goal positions"
    )

    # Task-specific configurations
    extra: dict[str, Any] = Field(default_factory=dict, description="Extra task-specific config")


class DifficultyConfig(BaseModel):
    """Configuration for a specific difficulty level."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(..., description="Difficulty name (easy, medium, hard, expert)")
    grid_size: int = Field(..., ge=3, description="Grid size (height=width=grid_size)")
    max_steps: int = Field(..., ge=10, description="Maximum steps per episode")

    # Task-specific parameters
    params: dict[str, Any] = Field(default_factory=dict, description="Task-specific parameters")


class TaskMetadata(BaseModel):
    """Metadata for a task."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(..., description="Task name")
    description: str = Field(..., description="Task description")
    capability_tags: list[str] = Field(..., description="Capability tags")
    difficulty_levels: list[str] = Field(..., description="Available difficulty levels")
    optimal_return: float = Field(..., description="Theoretical optimal return")
    random_baseline: float = Field(..., description="Expected random agent return")
