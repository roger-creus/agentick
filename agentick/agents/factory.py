"""Factory for creating agents from experiment configs."""

from __future__ import annotations

from typing import Any

from agentick.agents.backends import get_backend_class
from agentick.agents.base import BaseAgent
from agentick.agents.harness import HARNESS_REGISTRY
from agentick.experiments.config import AgentConfig


def create_agent(agent_config: AgentConfig) -> BaseAgent | None:
    """Create an agent from an AgentConfig.

    Returns None for agent types that don't use the harness system
    (e.g. "random", "ppo").

    Args:
        agent_config: Agent configuration from experiment YAML.

    Returns:
        BaseAgent instance, or None for non-LLM/VLM agents.
    """
    if agent_config.type not in ("llm", "vlm"):
        return None

    hp = agent_config.hyperparameters

    # --- Backend ---
    backend_name = hp.get("backend", "openai")
    backend_cls = get_backend_class(backend_name)

    # Collect backend-specific kwargs
    backend_kwargs: dict[str, Any] = {}
    if "model" in hp:
        backend_kwargs["model" if backend_name in ("openai", "anthropic") else "model_id"] = hp[
            "model"
        ]
    for key in (
        "api_key_env",
        "max_tokens",
        "temperature",
        "max_retries",
        "device",
        "dtype",
        "quantization",
        "max_new_tokens",
    ):
        if key in hp:
            backend_kwargs[key] = hp[key]

    backend = backend_cls(**backend_kwargs)

    # --- Harness ---
    harness_name = hp.get("harness", "markovian_zero_shot")
    if harness_name not in HARNESS_REGISTRY:
        raise ValueError(
            f"Unknown harness: {harness_name}. Available: {list(HARNESS_REGISTRY.keys())}"
        )
    harness_cls = HARNESS_REGISTRY[harness_name]
    harness_kwargs: dict[str, Any] = {}
    if "max_history" in hp:
        harness_kwargs["max_history"] = hp["max_history"]
    harness = harness_cls(**harness_kwargs)

    # --- Observation modes ---
    obs_modes = hp.get("observation_modes", ["language"])

    return BaseAgent(
        backend=backend,
        harness=harness,
        observation_modes=obs_modes,
        agent_name=hp.get("agent_name"),
    )
