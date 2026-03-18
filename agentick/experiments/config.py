"""Experiment configuration with Pydantic models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class TrainingConfig(BaseModel):
    """Configuration for PPO/RL training runs."""

    total_timesteps: int = Field(default=500_000, description="Total training timesteps per task")
    n_envs: int = Field(default=8, description="Number of parallel training environments")
    eval_frequency: int = Field(default=50_000, description="Evaluate every N timesteps")
    n_eval_episodes: int = Field(default=10, description="Episodes per evaluation")
    checkpoint_frequency: int = Field(default=100_000, description="Save checkpoint every N steps")
    save_best_model: bool = Field(default=True, description="Keep best model by eval reward")
    device: str = Field(default="auto", description="Torch device (auto, cpu, cuda)")


class AgentConfig(BaseModel):
    """Configuration for agent."""

    type: str = Field(..., description="Agent type (random, oracle, ppo, etc.)")
    hyperparameters: dict[str, Any] = Field(
        default_factory=dict, description="Agent-specific hyperparameters"
    )

    class Config:
        extra = "allow"  # Allow additional fields


class ExperimentConfig(BaseModel):
    """Complete experiment specification."""

    name: str = Field(..., description="Experiment name")
    description: str = Field(default="", description="Experiment description")
    agent: AgentConfig = Field(..., description="Agent configuration")
    tasks: list[str] | str = Field(..., description="Task names or suite name")
    difficulties: list[str] = Field(default=["easy"], description="Difficulty levels to evaluate")
    n_episodes: int = Field(default=10, description="Episodes per task per difficulty", gt=0)
    n_seeds: int = Field(default=3, description="Number of seeds", gt=0)
    seeds: list[int] | None = Field(default=None, description="Explicit seeds or auto-generate")
    render_modes: list[str] = Field(default=["ascii"], description="Which observations to record")
    reward_mode: str = Field(default="sparse", description="Reward mode")
    record_trajectories: bool = Field(default=True, description="Record full episode trajectories")
    record_videos: bool = Field(default=False, description="Record episode videos")
    record_observations: bool = Field(
        default=False, description="Save observations in all modalities"
    )
    metrics: list[str] = Field(
        default_factory=lambda: ["mean_return", "success_rate", "mean_length"],
        description="Metrics to compute",
    )
    split: str = Field(default="eval", description="Seed split: 'train' or 'eval'")
    output_dir: str = Field(default="results", description="Output directory")
    tags: list[str] = Field(default_factory=list, description="Tags for filtering")
    training: TrainingConfig | None = Field(
        default=None, description="Training config (None = eval-only mode)"
    )
    base_config: str | None = Field(default=None, description="Base config to inherit from")

    class Config:
        extra = "allow"

    @field_validator("tasks")
    @classmethod
    def validate_tasks(cls, v):
        """Validate tasks field."""
        if isinstance(v, str):
            # Suite name
            valid_suites = [
                "full", "navigation", "planning", "reasoning",
                "memory", "generalization", "multi_agent",
            ]
            if v not in valid_suites and not v.endswith("-v0"):
                # Allow individual task names too
                pass
            return v
        elif isinstance(v, list):
            if not v:
                raise ValueError("tasks list cannot be empty")
            return v
        else:
            raise ValueError("tasks must be string or list of strings")

    @field_validator("difficulties")
    @classmethod
    def validate_difficulties(cls, v):
        """Validate difficulties."""
        valid_difficulties = ["easy", "medium", "hard", "expert"]
        for diff in v:
            if diff not in valid_difficulties:
                raise ValueError(f"Invalid difficulty: {diff}. Must be one of {valid_difficulties}")
        return v

    @field_validator("render_modes")
    @classmethod
    def validate_render_modes(cls, v):
        """Validate render modes."""
        valid_modes = [
            "ascii",
            "language",
            "language_structured",
            "rgb_array",
            "human",
            "state_dict",
        ]
        for mode in v:
            if mode not in valid_modes:
                raise ValueError(f"Invalid render mode: {mode}. Must be one of {valid_modes}")
        return v

    def to_yaml(self, path: str | Path) -> None:
        """Save config to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.model_dump()
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load config from YAML file."""
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)

        return cls(**data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        """Create config from dictionary."""
        return cls(**data)

    def inherit_from(self, base: ExperimentConfig) -> ExperimentConfig:
        """
        Create new config inheriting from base config.

        This config's values override base config's values.
        """
        base_dict = base.model_dump()
        override_dict = self.model_dump(exclude_unset=True)

        # Merge dictionaries
        merged = {**base_dict, **override_dict}

        # Special handling for nested dicts (agent.hyperparameters)
        if "agent" in override_dict and "agent" in base_dict:
            base_agent = base_dict["agent"]
            override_agent = override_dict["agent"]

            if "hyperparameters" in override_agent and "hyperparameters" in base_agent:
                merged_hyper = {
                    **base_agent.get("hyperparameters", {}),
                    **override_agent.get("hyperparameters", {}),
                }
                merged["agent"]["hyperparameters"] = merged_hyper

        return ExperimentConfig(**merged)

    def resolve_base(self, config_dir: str | Path) -> ExperimentConfig:
        """
        Resolve base config if specified.

        Args:
            config_dir: Directory containing config files

        Returns:
            Config with base config applied
        """
        if self.base_config is None:
            return self

        config_dir = Path(config_dir)
        base_path = config_dir / self.base_config

        if not base_path.exists():
            base_path = config_dir / f"{self.base_config}.yaml"

        if not base_path.exists():
            raise ValueError(f"Base config not found: {self.base_config}")

        base_config = ExperimentConfig.from_yaml(base_path)

        # Recursively resolve base's base if it has one
        if base_config.base_config is not None:
            base_config = base_config.resolve_base(config_dir)

        return self.inherit_from(base_config)

    def validate_config(self) -> list[str]:
        """
        Validate config and return list of errors.

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Check seeds
        if self.seeds is not None:
            if len(self.seeds) != self.n_seeds:
                errors.append(f"len(seeds)={len(self.seeds)} but n_seeds={self.n_seeds}")

        # Check metrics
        valid_metrics = [
            "mean_return",
            "success_rate",
            "mean_length",
            "std_return",
            "median_return",
            "min_return",
            "max_return",
            "action_efficiency",
            "exploration_efficiency",
            "mean_latency",
            "total_tokens",
            "total_api_calls",
            "total_cost_usd",
        ]
        for metric in self.metrics:
            if metric not in valid_metrics:
                errors.append(f"Unknown metric: {metric}")

        # Check agent type
        valid_agent_types = ["random", "oracle", "ppo", "dqn", "llm", "vlm", "custom"]
        if self.agent.type not in valid_agent_types:
            # Warning only, not error
            pass

        return errors

    def get_primary_render_mode(self) -> str | None:
        """Get the primary render mode, preferring agent observation modes."""
        if self.agent.type in ("llm", "vlm"):
            obs_modes = self.agent.hyperparameters.get("observation_modes", [])
            if obs_modes:
                return obs_modes[0]
        if self.render_modes:
            return self.render_modes[0]
        return None


def load_config(path: str | Path, config_dir: str | Path | None = None) -> ExperimentConfig:
    """
    Load experiment config from YAML, resolving base configs.

    Args:
        path: Path to config file
        config_dir: Directory containing base configs (defaults to parent of path)

    Returns:
        ExperimentConfig with base configs resolved
    """
    path = Path(path)

    if config_dir is None:
        config_dir = path.parent

    config = ExperimentConfig.from_yaml(path)
    config = config.resolve_base(config_dir)

    # Validate
    errors = config.validate_config()
    if errors:
        raise ValueError("Config validation errors:\n" + "\n".join(f"  - {e}" for e in errors))

    return config
