"""Agent submission specification and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class SubmissionSpec(BaseModel):
    """
    Complete specification for an agent submission.

    This defines everything needed to load, run, and evaluate an agent.
    """

    # === Identity ===
    agent_name: str = Field(..., description="Unique agent name (e.g., 'GPT-4o-TextAgent-v2')")
    author: str = Field(..., description="Author or organization name")
    description: str = Field(..., description="What the agent does and how it works")
    url: str | None = Field(None, description="Link to paper, repo, or blog post")
    tags: list[str] = Field(
        default_factory=list, description="Tags like ['llm', 'zero-shot', 'text']"
    )
    license: str = Field(
        "proprietary", description="License: 'proprietary', 'MIT', 'apache-2.0', etc."
    )
    open_weights: bool = Field(False, description="Are weights publicly available?")

    # === Agent Type ===
    agent_type: Literal[
        "api",  # Cloud API (OpenAI, Anthropic, custom)
        "huggingface",  # HuggingFace model ID
        "local_weights",  # Local .pt / safetensors path
        "code",  # Python file implementing AgentProtocol
        "docker",  # Docker image with HTTP agent server
        "git_repo",  # Git repo URL with setup instructions
    ] = Field(..., description="Type of agent submission")

    # === Observation Mode ===
    observation_mode: Literal[
        "ascii",  # ASCII text grid
        "language",  # Natural language description
        "language_structured",  # Structured language (JSON/dict)
        "rgb_array",  # Pixel observations
        "state_dict",  # Full state dictionary
    ] = Field(..., description="Observation format the agent expects")

    # === Connection Config (depends on agent_type) ===
    config: dict[str, Any] = Field(..., description="Agent-specific configuration")
    # For api:          {"provider": "openai", "model": "gpt-4o", "api_key_env": "OPENAI_API_KEY"}
    # For huggingface:  {"model_id": "meta-llama/Llama-3-8B", "device": "auto", "dtype": "float16"}
    # For local_weights: {"weights_path": "models/my_agent.pt", "model_class": "agents.MyAgent"}
    # For code:         {"script_path": "agents/my_agent.py", "class_name": "MyAgent"}
    # For docker:       {"image": "myagent:latest", "port": 8080, "endpoint": "/predict"}
    # For git_repo:     {"url": "https://github.com/...", "branch": "main", "setup_cmd": "pip install -e ."}

    # === Evaluation Config ===
    suites: list[str] = Field(..., description="Which benchmark suites to evaluate on")

    # === Metadata ===
    hardware: str | None = Field(
        None, description="Hardware used: '1x A100 80GB', 'CPU only', 'API'"
    )
    estimated_cost: str | None = Field(None, description="Estimated cost: '$50', 'free (local)'")
    training_data: str | None = Field(
        None, description="Training data: 'None (zero-shot)', 'Agentick oracle demos'"
    )
    training_compute: str | None = Field(
        None, description="Training compute: '8x A100 for 24h', 'N/A'"
    )

    @field_validator("agent_name")
    @classmethod
    def validate_agent_name(cls, v: str) -> str:
        """Validate agent name format."""
        if not v or len(v) < 3:
            raise ValueError("Agent name must be at least 3 characters")
        if not all(c.isalnum() or c in "-_." for c in v):
            raise ValueError("Agent name must be alphanumeric with -_. allowed")
        return v

    @field_validator("suites")
    @classmethod
    def validate_suites(cls, v: list[str]) -> list[str]:
        """Validate that suites exist."""
        from agentick.leaderboard.suites import OFFICIAL_SUITES

        for suite_name in v:
            if suite_name not in OFFICIAL_SUITES:
                available = ", ".join(OFFICIAL_SUITES.keys())
                raise ValueError(f"Unknown suite '{suite_name}'. Available: {available}")
        return v

    @field_validator("config")
    @classmethod
    def validate_config(cls, v: dict[str, Any], values) -> dict[str, Any]:
        """Validate config based on agent_type."""
        # Note: values.data contains already-validated fields
        agent_type = values.data.get("agent_type")

        if agent_type == "api":
            required = ["provider", "model"]
            for key in required:
                if key not in v:
                    raise ValueError(f"API agent config must include '{key}'")

        elif agent_type == "huggingface":
            if "model_id" not in v:
                raise ValueError("HuggingFace agent config must include 'model_id'")

        elif agent_type == "local_weights":
            if "weights_path" not in v:
                raise ValueError("Local weights agent config must include 'weights_path'")
            if "model_class" not in v:
                raise ValueError("Local weights agent config must include 'model_class'")

        elif agent_type == "code":
            if "script_path" not in v:
                raise ValueError("Code agent config must include 'script_path'")

        elif agent_type == "docker":
            required = ["image", "port", "endpoint"]
            for key in required:
                if key not in v:
                    raise ValueError(f"Docker agent config must include '{key}'")

        elif agent_type == "git_repo":
            if "url" not in v:
                raise ValueError("Git repo agent config must include 'url'")

        return v

    def to_yaml(self, path: str | Path) -> None:
        """Save submission spec to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> SubmissionSpec:
        """Load submission spec from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        return cls(**data)

    def validate_submission(self) -> list[str]:
        """
        Validate submission and return list of warnings (not errors).

        Returns:
            List of warning messages (empty if all good)
        """
        warnings = []

        # Check if API key env var is set (for API agents)
        if self.agent_type == "api":
            api_key_env = self.config.get("api_key_env")
            if api_key_env:
                import os

                if not os.getenv(api_key_env):
                    warnings.append(f"API key environment variable '{api_key_env}' not set")

        # Check if files exist (for local paths)
        if self.agent_type == "local_weights":
            weights_path = Path(self.config["weights_path"])
            if not weights_path.exists():
                warnings.append(f"Weights file not found: {weights_path}")

        elif self.agent_type == "code":
            script_path = Path(self.config["script_path"])
            if not script_path.exists():
                warnings.append(f"Script file not found: {script_path}")

        # Check hardware/cost metadata
        if self.agent_type in ["huggingface", "local_weights"] and not self.hardware:
            warnings.append("Consider specifying 'hardware' for local model submissions")

        if self.agent_type == "api" and not self.estimated_cost:
            warnings.append("Consider estimating 'estimated_cost' for API submissions")

        return warnings
